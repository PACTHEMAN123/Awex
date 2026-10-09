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
Real GPU correctness and timing acceptance remain pending; the cleanup does not
constitute evidence of unchanged measured transfer performance.
