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

An installed library can be consumed without Python or PyTorch:

```cmake
find_package(ShardStream 0.1 CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE ShardStream::shardstream)
```

Set `CMAKE_PREFIX_PATH` to the installation prefix. The public host API is in
`<shardstream/runtime.h>`; the caller owns tensor memory and CUDA streams.
A standalone C++ consumer of the installed `3dd7325` library passes link and
execution checks, with no PyTorch or Python dynamic dependencies.

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

The user-defined naive/off/swizzle routing matrix completes 21 real-model cases
with full-weight, cache, generation and actual path checks. See
[measurements and acceptance](docs/acceptance.md) for timings, retained variation
and original GRPO/Elastic evidence, and [routing](docs/ring-routing.md) for the
fixed-entry, node-local naive chain. CPU checks alone do not establish GPU
performance equivalence.
