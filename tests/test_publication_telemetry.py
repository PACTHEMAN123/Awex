"""CPU-only regressions for a counter timeout and collective error agreement."""
import ast
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def load_telemetry():
    spec = importlib.util.spec_from_file_location(
        "telemetry_under_test", ROOT / "python/shardstream/profiling/telemetry.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_publication_telemetry(dist, requests):
    source = ROOT / "benchmarks/shardstream_benchmarks/model_exchange.py"
    tree = ast.parse(source.read_text())
    method = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "publication_telemetry")
    namespace = {
        "dist": dist, "requests": requests,
        "ThreadPoolExecutor": ThreadPoolExecutor,
        "torch": SimpleNamespace(cuda=SimpleNamespace(
            memory_allocated=lambda: 10, max_memory_allocated=lambda: 20)),
        "device_util": SimpleNamespace(current_device=lambda: 0),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
         namespace)
    return namespace["publication_telemetry"]


class TelemetryTests(unittest.TestCase):
    def test_timeout_retry_returns_only_new_real_counter_values(self):
        module = load_telemetry()
        timeout = subprocess.TimeoutExpired(["ethtool", "-S", "eth4"], 15)
        with patch.object(module.subprocess, "check_output", side_effect=[
            timeout, "tx_vport_rdma_unicast_bytes: 17\nrx_vport_rdma_unicast_bytes: 19"
        ]) as read:
            text, attempts = module._rdma_stats("eth4")
        self.assertEqual(attempts, 2)
        self.assertIn("17", text)
        self.assertEqual(read.call_count, 2)

    def test_persistent_timeout_and_command_failure_remain_errors(self):
        module = load_telemetry()
        timeout = subprocess.TimeoutExpired(["ethtool", "-S", "eth4"], 15)
        with patch.object(module.subprocess, "check_output", side_effect=timeout) as read:
            with self.assertRaises(subprocess.TimeoutExpired):
                module._rdma_stats("eth4")
            self.assertEqual(read.call_count, 2)
        failure = subprocess.CalledProcessError(1, ["ethtool", "-S", "eth4"])
        with patch.object(module.subprocess, "check_output", side_effect=failure) as read:
            with self.assertRaises(subprocess.CalledProcessError):
                module._rdma_stats("eth4")
            self.assertEqual(read.call_count, 1)

    def run_rank(self, driver, counters, request_error=None, remote_error=None):
        calls = []
        def gather(local, gathered, dst):
            calls.append("gather")
            if gathered is not None:
                gathered[:] = [local, {"rdma": {}}]
        def broadcast(status, src):
            calls.append("broadcast")
            if not driver:
                status[0] = remote_error
        dist = SimpleNamespace(gather_object=gather, broadcast_object_list=broadcast)
        response = Mock()
        response.json.return_value = {"rdma": {"mlx5_3:1": {"tx_bytes": 17}}}
        response.raise_for_status.side_effect = request_error
        requests = SimpleNamespace(get=Mock(return_value=response))
        method = load_publication_telemetry(dist, requests)
        subject = SimpleNamespace(local_rank=0, rank=0 if driver else 1,
                                  world_size=2, is_driver=driver,
                                  publication_endpoints=lambda: [(0, "node", 8080)])
        telemetry = SimpleNamespace(node_rdma_counters=counters)
        with patch.dict(sys.modules, {"shardstream.profiling.telemetry": telemetry}):
            try:
                result = method(subject)
            except RuntimeError as error:
                result = error
        return calls, result, requests

    def test_http_failure_is_broadcast_before_driver_raises(self):
        calls, result, _ = self.run_rank(True, lambda: {"rdma": {}},
                                         request_error=RuntimeError("HTTP500"))
        self.assertEqual(calls, ["gather", "broadcast"])
        self.assertIsInstance(result, RuntimeError)
        self.assertIn("HTTP500", str(result))

    def test_local_counter_failure_participates_in_both_collectives(self):
        def counters():
            raise subprocess.TimeoutExpired("ethtool", 15)
        calls, result, requests = self.run_rank(True, counters)
        self.assertEqual(calls, ["gather", "broadcast"])
        self.assertIsInstance(result, RuntimeError)
        self.assertIn("TimeoutExpired", str(result))
        requests.get.assert_not_called()

    def test_other_rank_observes_driver_error_before_continuing(self):
        calls, result, _ = self.run_rank(False, lambda: {"rdma": {}},
                                         remote_error="HTTP500")
        self.assertEqual(calls, ["gather", "broadcast"])
        self.assertIsInstance(result, RuntimeError)

    def test_success_preserves_real_training_and_rollout_counters(self):
        calls, result, _ = self.run_rank(True, lambda: {"rdma": {"mlx5_7:1": 23}})
        self.assertEqual(calls, ["gather", "broadcast"])
        self.assertEqual(result["training_ranks"][0]["rdma"], {"mlx5_7:1": 23})
        self.assertEqual(result["rollout_nodes"]["node"]["rdma"]["mlx5_3:1"]["tx_bytes"], 17)


if __name__ == "__main__":
    unittest.main()
