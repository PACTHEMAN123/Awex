import pytest

from shardstream_benchmarks.recipe_runner import training_launch_arguments


@pytest.mark.parametrize("hosts,world_size", [(["train0"], 8), (["train0", "train1"], 16)])
def test_training_rank_zero_stays_on_metadata_host(hosts, world_size):
    payload = ["-m", "shardstream_benchmarks.model_exchange", "--recipe", "glm-4.7-flash"]
    for rank in range(len(hosts)):
        command = training_launch_arguments(hosts, world_size, rank, "run-123", payload)
        assert command[:2] == ["-m", "torch.distributed.run"]
        assert command[command.index("--rdzv-backend") + 1] == "static"
        assert command[command.index("--node-rank") + 1] == str(rank)
        assert command[command.index("--rdzv-endpoint") + 1] == "train0:18891"
        assert command[command.index("--rdzv-id") + 1] == "run-123"
        assert command[command.index("--nnodes") + 1] == str(len(hosts))
        assert command[command.index("--nproc-per-node") + 1] == "8"
        assert command[-len(payload):] == payload
