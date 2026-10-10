"""Pinned model recipes and placement checks for disaggregated benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from shardstream_benchmarks.parallel import resolve_megatron_parallelism


@dataclass(frozen=True)
class ModelRecipe:
    id: str
    model: str
    source: dict
    training: dict
    rollout: dict
    mtp: bool
    separate_nodes: bool
    max_total_gpus: int

    def validate(self, repository: Path) -> None:
        source = repository / self.source["snapshot"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != self.source["sha256"]:
            raise ValueError(f"{self.id}: upstream recipe snapshot changed")
        train = self.training
        parallelism = resolve_megatron_parallelism(
            tp_size=train["tp"],
            pp_size=train["pp"],
            cp_size=train["cp"],
            ep_size=train["ep"],
            expert_tp_size=train["etp"],
        )
        parallelism.validate_world_size(train["world_size"])
        rollout = self.rollout
        if rollout["engine"] != "vllm":
            raise ValueError(f"{self.id}: only vLLM rollout is supported")
        if (
            rollout["world_size"]
            != rollout["tp"] * rollout["dp"] * rollout["instances"]
        ):
            raise ValueError(
                f"{self.id}: rollout GPU count does not match TP/DP groups"
            )
        if rollout["attention_dp"] != rollout["dp"]:
            raise ValueError(f"{self.id}: vLLM attention DP must match engine DP")
        if rollout["ep"] not in (1, rollout["tp"] * rollout["dp"]):
            raise ValueError(f"{self.id}: vLLM EP must span the engine TP/DP group")
        if train["world_size"] + rollout["world_size"] > self.max_total_gpus:
            raise ValueError(f"{self.id}: training plus rollout exceeds GPU budget")

    def validate_placement(
        self, training_hosts: list[str], rollout_hosts: list[str]
    ) -> None:
        if not training_hosts or not rollout_hosts:
            raise ValueError("Both training and rollout hosts are required")
        if len(set(training_hosts)) != len(training_hosts):
            raise ValueError("Duplicate training host")
        if len(set(rollout_hosts)) != len(rollout_hosts):
            raise ValueError("Duplicate rollout host")
        if set(training_hosts) & set(rollout_hosts):
            raise ValueError("Training and rollout must use different physical hosts")
        for role, hosts, count in (
            ("training", training_hosts, self.training["world_size"]),
            ("rollout", rollout_hosts, self.rollout["world_size"]),
        ):
            if count % len(hosts) or count // len(hosts) > 8:
                raise ValueError(
                    f"{self.id}: {role} placement does not fit 8-GPU nodes"
                )


def load_recipes(repository: Path | None = None) -> dict[str, ModelRecipe]:
    repository = repository or Path(__file__).resolve().parents[2]
    raw = json.loads((repository / "benchmarks/recipes/models.json").read_text())
    recipes = {}
    for entry in raw["recipes"]:
        recipe = ModelRecipe(**entry)
        if recipe.id in recipes:
            raise ValueError(f"Duplicate recipe: {recipe.id}")
        recipe.validate(repository)
        recipes[recipe.id] = recipe
    return recipes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tuple(load_recipes()))
    parser.add_argument("--training-hosts", nargs="+", required=True)
    parser.add_argument("--rollout-hosts", nargs="+", required=True)
    args = parser.parse_args()
    recipe = load_recipes()[args.model]
    recipe.validate_placement(args.training_hosts, args.rollout_hosts)
    print(
        json.dumps(
            {
                "model": recipe.model,
                "training": recipe.training,
                "rollout": recipe.rollout,
                "training_hosts": args.training_hosts,
                "rollout_hosts": args.rollout_hosts,
                "total_gpus": recipe.training["world_size"]
                + recipe.rollout["world_size"],
                "source": recipe.source,
                "mtp": recipe.mtp,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
