import subprocess
import sys


def test_importing_awex_does_not_import_training_stack():
    script = """
import sys
import awex

assert "megatron" not in sys.modules
assert "transformer_engine" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_vllm_registry_process_skips_full_awex_plugin():
    script = """
import sys
from types import SimpleNamespace

import awex.vllm_plugin_entrypoint as entrypoint

sys.modules["__main__"].__spec__ = SimpleNamespace(
    name="vllm.model_executor.models.registry"
)
entrypoint.register_awex_plugin()
assert "awex.vllm_plugin" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)
