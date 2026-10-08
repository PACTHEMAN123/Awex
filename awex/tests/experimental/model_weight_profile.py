"""Validation/profiling tasks executed inside loaded vLLM model workers.

The reference is a full GPU clone of the actual HF-loaded model, used only by
this opt-in benchmark. Every update clears the real destination tensors and
checks every parameter exactly; a changing final norm identifies the version.
"""

import os
import time

import torch

from awex.tests.experimental.dynamic_rollout_profile import MockCompute


class ModelWeightProfile:
    def __init__(self, model):
        self.parameters = dict(model.named_parameters())
        if not self.parameters or any(
            p.dtype != torch.bfloat16 for p in self.parameters.values()
        ):
            raise ValueError("The model-weight benchmark requires all BF16 parameters")
        self.references = {
            name: parameter.detach().clone()
            for name, parameter in self.parameters.items()
        }
        self.pointers = {name: p.data_ptr() for name, p in self.parameters.items()}
        self.compute = None
        self.norm_name = "model.norm.weight"
        if self.norm_name not in self.parameters:
            raise ValueError("No Qwen3 final norm version marker")
        torch.cuda.synchronize()

    def command(self, operation, version):
        if any(
            p.data_ptr() != self.pointers[name] for name, p in self.parameters.items()
        ):
            raise AssertionError("A loaded model parameter was replaced")
        started_ns = time.time_ns()
        if operation == "clear":
            with torch.no_grad():
                for parameter in self.parameters.values():
                    parameter.zero_()
            torch.cuda.synchronize()
            result = {"cleared": True}
        elif operation == "verify":
            bad = []
            for name, parameter in self.parameters.items():
                expected = self.references[name]
                if name == self.norm_name:
                    expected = torch.full_like(expected, 1 + (version + 2) / 64)
                if not torch.equal(parameter, expected):
                    bad.append(name)
            if bad:
                raise AssertionError(
                    f"Real model BF16 mismatch at version {version}: {bad[:20]} (total={len(bad)})"
                )
            result = {
                "verified": True,
                "version": version,
                "parameter_count": len(self.parameters),
                "model_bytes": sum(
                    p.numel() * p.element_size() for p in self.parameters.values()
                ),
            }
        elif operation == "compute_start":
            if self.compute is not None:
                raise RuntimeError("Model profile compute already started")
            weight = next(p for p in self.parameters.values() if p.ndim == 2)
            self.compute = MockCompute(weight[:1024, :1024], batch=512, repeats=256)
            self.compute.start(300)
            result = {"compute_started": True}
        elif operation == "compute_stop":
            result = {"compute_events": self.compute.stop(300) if self.compute else []}
            self.compute = None
        else:
            raise ValueError("Unsupported model profile command")
        return {
            **result,
            "pid": os.getpid(),
            "pointers": self.pointers,
            "start_ns": started_ns,
            "end_ns": time.time_ns(),
        }


def model_profile_task(operation: str, version: int, **kwargs):
    scheduler = kwargs["model_context"]["scheduler"]
    profile = getattr(scheduler, "_awex_model_weight_profile", None)
    if profile is None:
        profile = ModelWeightProfile(kwargs["model"])
        scheduler._awex_model_weight_profile = profile
    result = profile.command(operation, version)
    reader = getattr(scheduler, "awes_weights_reader", None)
    if reader is not None and operation == "verify":
        result["transfer_metrics"] = getattr(reader, "last_transfer_metrics", {})
    return result
