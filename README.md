# ShardStream

CUDA communication for sharded model weight streams. The single Python backend
is `shardstream.Transport`. Plans and prepared tensor descriptors are reusable
across weight publications; membership epochs support preparing expanded rollout
groups before the next publication.

The initial extraction comes from the validated `device-v2-dynamic-rollout`
revision `6dabfbf7ede790c34011924136f132514fa4b60c`. Copyright and Apache-2.0
license notices remain intact.

## Layout

- `include/shardstream/`: CUDA protocol, work descriptors, lowering and topology.
- `src/`: CUDA kernels and launch entry points, independent of Python/PyTorch.
- `bindings/torch/`: optional tensor and stream binding.
- `python/shardstream/`: plans, metadata, membership, and transport orchestration.
- `tests/`: CPU plan/cache, routing, FP8 descriptor, and membership regression tests.
- `docs/`: migration provenance and experiment acceptance requirements.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DSHARDSTREAM_NCCL_ROOT=/path/to/nccl -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build -j2
cmake --install build --prefix /path/to/install
```

To install the Python extension with the current environment's PyTorch:

```sh
pip install --no-build-isolation . \
  -Ccmake.define.SHARDSTREAM_NCCL_ROOT=/path/to/nccl
```

This requires scikit-build-core, PyTorch, CUDA and NCCL development files.
Compilation is explicit; importing the transport never JIT-compiles an extension.

For CPU regression checks without a CUDA build:

```sh
PYTHONPATH=python python -m pytest -q
```

The refactor and GPU acceptance matrix are in progress. CPU checks do not prove
real weight-transfer correctness or performance equivalence; see
[acceptance requirements](docs/acceptance.md).
