# Removed legacy components

The retained integration path is local HF checkpoint loading through mbridge,
Megatron training metadata and conversion, vLLM inference metadata and conversion,
the ShardStream transport, veRL checkpoint integration, and real-model benchmarks.
Qwen3 dense and MoE share the fused QKV/expert converter; that converter is part
of the vLLM path and remains in `integrations/conversion/fused.py`.

Removed implementations:

- Colocated CUDA/CPU IPC publication, tensor IPC serialization and process-wide
  multiprocessing authentication mutation.
- NPU/HCCL device branches, MindSpeed patches, and NPU MoE transposition.
- Disk checkpoint publication and its load/ready polling barriers. Explicit
  tensor dumps used for correctness checks remain available.
- SGLang configuration/registry entry points and unused MLA conversion helpers.
- DCP conversion, remote model downloads and unused tokenizer/model-loader paths.

At cleanup revision `98f9504`, relative to pre-cleanup revision `66daf8f`,
library and benchmark Python code
shrinks from 17,298 to 14,507 lines and from 710 to 629 function/class definitions.
These totals exclude tests. Newly retained conversion tests check Qwen3 GQA splits,
MoE expert IDs, FP8 scale names/storage views and dense zero-copy source spans;
veRL locality tests check physical GPU rank mapping.

This cleanup changes no files under `include/`, `src/`, or `bindings/`, and no
CMake compiler settings. All 72 regression tests pass locally and in the H20
runtime. Kernel/protocol audit and full integration imports pass without loading
the old `awex` package.
The second cleanup removes another 584 lines and 24 definitions: an uncalled HF
loader, pairwise NCCL subgroup creation, an unused PP layout inference chain
(the actual model-derived PP map remains), Python FP8 conversion, unused transfer
chunk/rank-axis helpers, descriptor-count tooling and IPC serialization constants.
Library and benchmark Python code now has 13,931 lines and 605 definitions.
All 74 local regression tests pass. C++ runtime, CUDA transfer kernel and compiler
settings remain unchanged; the subsequent Torch binding correction restores
original steady-call GIL ownership and keeps explicit prepare releasing GIL.
The cleanup wheel was installed at `3dd7325` on all four H20 nodes;
its extension SHA-256 is unchanged and full integration imports pass. Four real BF16
GRPO configurations now qualify this cleanup; both PP1 FP8 cases also pass
10-step real GRPO. All 21 standalone cases pass full-checkpoint correctness, cache and generation. Original kernel-call GIL ownership is restored; explicit prepare still releases GIL. The isolated prototype is near baseline, but the final package repeats slower on FP8 naive, so that performance anomaly is retained for the superseded naive route. Final BF16 swizzle and FP8 GRPO core updates are near baseline, with all full-weight/cache/resource checks passed. PP4 is excluded by explicit user request.

At revision `3b69d74`, real Qwen3 BF16 Elastic correctness and adjacent original
transfer comparisons pass; preparation timing is recorded independently of
core weight updates. Qualification and measured experiment coverage are
recorded in `acceptance.md`. Experiments 8, 9, 11 and 12 have their required coverage. Experiment 10 has
complete real-model correctness coverage; its independent FP8 naive performance
anomaly is retained as historical diagnostic evidence despite final packaged-build correctness passing.

The user-defined node-grouped naive/off/swizzle rerun at `8039c11` completes all
21 real-model cases with independently archived full-weight/cache/generation
and actual-edge checks. Native bytes and transfer knobs remain unchanged.
This supersedes the previous naive route definition; old performance diagnostics
remain historical records. Current timings and retained off variation are in
`acceptance.md`. No additional control or Elastic queue is launched.
