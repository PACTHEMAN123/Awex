"""Register ShardStream inside every vLLM worker, including spawned DP workers.

An API-server monkey patch is inherited by forked TP workers, but not by fresh
DP processes. vLLM resolves this extension in each worker before constructing
its model worker. The methods remain on WorkerBase so existing integrations
and forked workers use the same implementation.
"""

from shardstream.integrations.vllm.plugin import _patch_shardstream_worker

_patch_shardstream_worker()


class ShardStreamWorkerExtension:
    """Import hook for vLLM's worker-extension loader."""

