"""Lightweight vLLM plugin entrypoint.

vLLM loads general plugins while inspecting model classes in a short-lived
subprocess. That subprocess does not need Awex worker patches, and importing
the full reader stack there can initialize optional Megatron/TE extensions.
"""

import sys


def _is_model_registry_process():
    main_spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    return getattr(main_spec, "name", None) == "vllm.model_executor.models.registry"


def register_awex_plugin():
    if _is_model_registry_process():
        return
    from awex.vllm_plugin import register_awex_plugin as register

    register()
