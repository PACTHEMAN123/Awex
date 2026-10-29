"""Verify the independent checkpoint guard does not grow the CUDA allocator."""
import fcntl
import json
from pathlib import Path
import subprocess

import torch
from shardstream.integrations.publication.weightbridge_native import _model_probe_guard

with open('/mnt/fuse/verl-e2e/runs/five-model-baselines-20261012/matrix/active.lock', 'a+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                       '--format=csv,noheader'], text=True).strip()
    model = torch.nn.Linear(4096, 4096, bias=False, device='cuda', dtype=torch.bfloat16)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_reserved()
    with _model_probe_guard(model, 'cuda-smoke', 0):
        pass
    torch.cuda.synchronize()
    after = torch.cuda.memory_reserved()
    assert after == before, (before, after)
    rejected = False
    try:
        with _model_probe_guard(model, 'cuda-smoke-corrupt', 0):
            with torch.no_grad():
                model.weight.zero_()
    except RuntimeError as error:
        assert 'probe changed checkpoint parameters' in str(error)
        rejected = True
    assert rejected
    row = dict(passed=True, reserved_before=before, reserved_after=after,
               cuda_allocator_growth=after-before, corrupted_checkpoint_rejected=rejected)
    Path('/tmp/weightbridge-memory-1029/guard-smoke.json').write_text(json.dumps(row, indent=2) + '\n')
    print(json.dumps(row))
