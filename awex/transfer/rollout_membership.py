"""Append-only rollout membership at complete weight-publication boundaries.

The coordinator belongs to one driver. Participants acknowledge preparation and
publication through the application's RPC layer; it never transmits weights.
"""

from dataclasses import dataclass, replace
from threading import RLock
from typing import Iterable

Participant = tuple[str, str, int]


@dataclass(frozen=True)
class RolloutMembership:
    epoch: int
    training_world_size: int
    inference_tp_size: int
    engine_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "engine_ids", tuple(self.engine_ids))
        if self.epoch < 0 or self.training_world_size < 1 or self.inference_tp_size < 1:
            raise ValueError("Invalid membership dimensions or epoch")
        if not self.engine_ids or any(
            not isinstance(i, str) or not i for i in self.engine_ids
        ):
            raise ValueError("Nonempty string engine IDs are required")
        if len(set(self.engine_ids)) != len(self.engine_ids):
            raise ValueError("Engine IDs must be unique")
        if not 2 <= self.world_size <= 256:
            raise ValueError("Device v2 requires 2–256 transfer ranks")

    @property
    def inference_world_size(self) -> int:
        return len(self.engine_ids) * self.inference_tp_size

    @property
    def world_size(self) -> int:
        return self.training_world_size + self.inference_world_size

    @property
    def participants(self) -> tuple[Participant, ...]:
        return tuple(
            [
                ("rollout", engine, tp)
                for engine in self.engine_ids
                for tp in range(self.inference_tp_size)
            ]
            + [("training", "", rank) for rank in range(self.training_world_size)]
        )

    def rank(self, participant: Participant) -> int:
        try:
            return self.participants.index(participant)
        except ValueError as exc:
            raise ValueError(f"Unknown participant: {participant}") from exc

    def grow(self, engine_ids: Iterable[str]) -> "RolloutMembership":
        return replace(
            self, epoch=self.epoch + 1, engine_ids=self.engine_ids + tuple(engine_ids)
        )


class RolloutJoinCoordinator:
    """Keep new engines unavailable until all ranks publish their first snapshot.

    Join requests may arrive during an update. Applying them requires an idle
    publication boundary. A failed reconfiguration is terminal: applications must
    recover explicitly rather than serving a partially updated group.
    """

    def __init__(self, membership: RolloutMembership) -> None:
        self.membership = membership
        self.last_version = -1
        self._queued = []
        self._pending = None
        self._prepared = set()
        self._updated = set()
        self._version = None
        self._ready = False
        self._failure = None
        self._lock = RLock()

    def _check(self):
        if self._failure is not None:
            raise RuntimeError(f"Membership failed: {self._failure}")

    @property
    def serving_engine_ids(self) -> tuple[str, ...]:
        with self._lock:
            if (
                self._failure
                or not self._ready
                or self._pending
                or self._version is not None
            ):
                return ()
            return self.membership.engine_ids

    def request_join(self, engine_id: str, inference_tp_size: int) -> bool:
        with self._lock:
            self._check()
            if not isinstance(engine_id, str) or not engine_id:
                raise ValueError("Engine ID must be a nonempty string")
            if inference_tp_size != self.membership.inference_tp_size:
                raise ValueError(
                    "Dynamic joining currently requires the same inference TP"
                )
            pending_ids = self._pending.engine_ids if self._pending else ()
            if (
                engine_id in self.membership.engine_ids + pending_ids
                or engine_id in self._queued
            ):
                return False
            pending_count = (
                len(self._pending.engine_ids)
                if self._pending
                else len(self.membership.engine_ids)
            )
            if (
                self.membership.training_world_size
                + (pending_count + len(self._queued) + 1) * inference_tp_size
                > 256
            ):
                raise ValueError("Device v2 requires at most 256 transfer ranks")
            self._queued.append(engine_id)
            return True

    def prepare_join(self) -> RolloutMembership:
        with self._lock:
            self._check()
            if (
                self._version is not None
                or self._pending is not None
                or not self._ready
            ):
                raise RuntimeError("Joining requires a completed publication boundary")
            if not self._queued:
                raise ValueError("No engines requested to join")
            self._pending = self.membership.grow(self._queued)
            self._queued.clear()
            self._prepared.clear()
            return self._pending

    def acknowledge_prepared(self, participant: Participant, epoch: int) -> None:
        with self._lock:
            self._check()
            if self._pending is None or epoch != self._pending.epoch:
                raise ValueError("Stale or unknown membership epoch")
            if participant not in self._pending.participants:
                raise ValueError("Unknown preparation participant")
            self._prepared.add(participant)

    def commit_join(self) -> RolloutMembership:
        with self._lock:
            self._check()
            if self._pending is None or self._prepared != set(
                self._pending.participants
            ):
                raise RuntimeError("Every old and new rank must prepare before commit")
            self.membership = self._pending
            self._pending = None
            self._ready = False
            return self.membership

    def begin_publication(self, version: int) -> RolloutMembership:
        with self._lock:
            self._check()
            if self._pending is not None or self._version is not None:
                raise RuntimeError("A transition or publication is already in progress")
            if not isinstance(version, int) or version <= self.last_version:
                raise ValueError("Weight versions must increase monotonically")
            self._version = version
            self._updated.clear()
            return self.membership

    def acknowledge_updated(
        self, participant: Participant, epoch: int, version: int
    ) -> None:
        with self._lock:
            self._check()
            if epoch != self.membership.epoch or self._version != version:
                raise ValueError("Stale publication acknowledgement")
            if participant not in self.membership.participants:
                raise ValueError("Unknown publication participant")
            self._updated.add(participant)

    def finish_publication(self) -> int:
        with self._lock:
            self._check()
            if self._version is None or self._updated != set(
                self.membership.participants
            ):
                raise RuntimeError(
                    "All ranks must finish the full snapshot before serving"
                )
            self.last_version = self._version
            self._version = None
            self._ready = True
            return self.last_version

    def fail(self, reason: str) -> None:
        with self._lock:
            self._failure = reason or "unspecified failure"
