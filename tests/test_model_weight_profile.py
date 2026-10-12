"""Detect stale weights, mixed router dtypes, and unintended storage replacement."""

import pytest
import torch

from shardstream.profiling.model import ModelWeightProfile, version_marker_factor


class TinyMixedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.norm = torch.nn.Linear(3, 1, bias=False, dtype=torch.bfloat16)
        self.model.norm.weight.data.fill_(0.75)
        self.router = torch.nn.Parameter(torch.tensor([0.003127, -1.23117]))


def test_full_validation_preserves_fp32_and_rejects_stale_update(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    model = TinyMixedModel()
    profile = ModelWeightProfile(model, "scale")
    profile.command("clear", 0)
    with pytest.raises(AssertionError, match="mismatch"):
        profile.command("verify", 0)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(profile.references[name])
        model.model.norm.weight.mul_(1 + 2 / 64)
    assert profile.command("verify", 0)["verified"]
    with pytest.raises(AssertionError, match="mismatch"):
        profile.command("verify", 1)
    with torch.no_grad():
        model.router.copy_(model.router.bfloat16().float())
    with pytest.raises(AssertionError, match="router"):
        profile.command("verify", 0)


def test_validation_rejects_replaced_parameters(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    model = TinyMixedModel()
    profile = ModelWeightProfile(model, "scale")
    model.router.data = model.router.detach().clone()
    with pytest.raises(AssertionError, match="replaced"):
        profile.command("clear", 0)


@pytest.mark.parametrize("updates", [3, 35])
def test_version_schedule_restores_checkpoint_and_rejects_previous_version(monkeypatch, updates):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    model = TinyMixedModel()
    profile = ModelWeightProfile(model, "scale", updates)
    factors = [version_marker_factor(version, updates) for version in range(-1, updates-1)]
    assert len(set(factors)) == updates
    assert factors[-1] == 1
    last_version = updates-2
    with torch.no_grad():
        model.model.norm.weight.copy_(profile.references[profile.norm_name] * factors[-2])
    with pytest.raises(AssertionError, match="mismatch"):
        profile.command("verify", last_version)
    profile.command("clear", last_version)
    with pytest.raises(AssertionError, match="mismatch"):
        profile.command("verify", last_version)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(profile.references[name])
    assert profile.command("verify", last_version)["verified"]
