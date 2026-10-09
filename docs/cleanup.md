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
All 74 local regression tests pass. Native files and compiler settings remain
unchanged; the installed H20 wheel must be refreshed before counting GPU runs
as acceptance of this second cleanup.

At revision `3b69d74`, real Qwen3 BF16 Elastic correctness and adjacent original
transfer comparisons pass; details and outstanding preparation timing questions
are recorded in `acceptance.md`. Experiments 8–11 remain incomplete.
