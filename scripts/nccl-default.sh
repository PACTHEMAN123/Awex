#!/usr/bin/env bash
# Source this before launching Python or configuring a fresh native build.
# SHARDSTREAM_NCCL_ROOT may point to an isolated installation of the pinned wheel.
task_nccl_root="${SHARDSTREAM_NCCL_ROOT:-}"
if [ -z "$task_nccl_root" ]; then
  task_nccl_root="$(python - <<'PY'
from importlib.util import find_spec
spec = find_spec("nvidia.nccl")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("Install requirements/nccl-cu13.txt first or set SHARDSTREAM_NCCL_ROOT")
print(next(iter(spec.submodule_search_locations)))
PY
  )" || { return 1 2>/dev/null || exit 1; }
fi
python - "$task_nccl_root" <<'PY' || { return 1 2>/dev/null || exit 1; }
import ctypes
from pathlib import Path
import re
import sys
root = Path(sys.argv[1])
header = (root / "include/nccl.h").read_text()
version = tuple(int(re.search(r"#define\s+NCCL_" + name + r"\s+(\d+)", header)[1])
                for name in ("MAJOR", "MINOR", "PATCH"))
if version != (2, 31, 2) or not (root / "include/nccl_device.h").is_file():
    raise SystemExit(f"Expected NCCL 2.31.2 Device API headers, found {version}")
library = ctypes.CDLL(str(root / "lib/libnccl.so.2"))
runtime = ctypes.c_int()
if library.ncclGetVersion(ctypes.byref(runtime)) != 0 or runtime.value != 23102:
    raise SystemExit(f"Expected NCCL runtime 23102, found {runtime.value}")
PY
export SHARDSTREAM_NCCL_ROOT="$task_nccl_root"
export SHARDSTREAM_NCCL_LIB="$task_nccl_root/lib"
export LD_PRELOAD="$SHARDSTREAM_NCCL_LIB/libnccl.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
export LD_LIBRARY_PATH="$SHARDSTREAM_NCCL_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset task_nccl_root
