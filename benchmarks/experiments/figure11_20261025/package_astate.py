"""Freeze the official AState Python module and required non-runtime libraries."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

BASE = Path("/tmp/figure11-1025")
SOURCE = BASE / "astate"
TARGET = BASE / "astate-runtime"
RUNTIME = Path("/tmp/verl-nccl-1020-runtime")
REVISION = "6598daf2f8f493a1538d9cf4b5e2beb3acb80db9"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip() == REVISION
    patch = BASE / "source/benchmarks/experiments/figure11_20261025/patches/astate-explicit-rdma-devices.patch"
    actual = subprocess.check_output(["git", "-C", str(SOURCE), "diff", "--binary", "HEAD"])
    assert actual == patch.read_bytes(), "AState source must match the reviewed device configuration patch"
    native_ucx = "USE_ASTATE_RDMA_IMPL:BOOL=ON" in (SOURCE / "build/CMakeCache.txt").read_text()
    assert not TARGET.exists(), "do not replace an existing pinned artifact"
    TARGET.mkdir()
    shutil.copytree(SOURCE / "python/astate", TARGET / "python/astate",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    extension, = (TARGET / "python/astate/_core").glob("astate_cpp*.so")
    library_dir = TARGET / "lib"
    library_dir.mkdir()
    # Libraries already installed identically on every benchmark node are not
    # duplicated. Preserve the runtime's NCCL/CUDA choices.
    inherited = [str(RUNTIME / "lib"), "/usr/local/cuda/lib64",
                 "/tmp/verl-nccl-1020-nccl/nvidia/nccl/lib",
                 "/tmp/verl-nccl-1020-deps/nvidia/cuda_runtime/lib",
                 str(RUNTIME / "lib/python3.12/site-packages/torch/lib")]
    env = dict(os.environ, LD_LIBRARY_PATH=":".join(inherited))
    dependencies = subprocess.check_output(["ldd", str(extension)], env=env, text=True)
    assert "not found" not in dependencies, dependencies
    system_core = ("ld-linux", "libc.so", "libm.so", "libpthread.so", "librt.so", "libdl.so",
                   "libstdc++.so", "libgcc_s.so", "libibverbs.so", "libmlx5.so", "libnuma.so",
                   "libnl-", "libatomic.so")
    for name, filename in re.findall(r"\s+(\S+) => (/\S+) ", dependencies):
        if filename.startswith((str(RUNTIME), "/usr/local/cuda", "/tmp/verl-nccl-1020-")):
            continue
        if name.startswith(system_core):
            continue
        shutil.copy2(filename, library_dir / name, follow_symlinks=True)
    utrans = library_dir / "libutrans.so"
    # Some linkers record the versioned SONAME instead of the unversioned name.
    if not native_ucx and not utrans.exists():
        candidate, = library_dir.glob("libutrans.so*")
        utrans.symlink_to(candidate.name)
    manifest = dict(source_revision=REVISION,
                    transport="native-ucx-rdma" if native_ucx else "default-libutrans-rdma",
                    source_patch_sha256=digest(patch),
                    extension=str(extension), extension_sha256=digest(extension),
                    utrans_library=None if native_ucx else str(utrans),
                    utrans_sha256=None if native_ucx else digest(utrans),
                    ucx_static_sha256={name: digest(SOURCE / "install/lib" / name)
                                       for name in ["libucp.a", "libuct.a", "libucs.a", "libucm.a"]} if native_ucx else {},
                    ld_library_path=":".join([str(library_dir), *inherited]),
                    pythonpath=str(TARGET / "python"),
                    ldd=dependencies,
                    files={str(p.relative_to(TARGET)): digest(p)
                           for p in TARGET.rglob("*") if p.is_file()})
    (BASE / "astate-artifact.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env.update(LD_LIBRARY_PATH=manifest["ld_library_path"], PYTHONPATH=manifest["pythonpath"])
    imports = subprocess.check_output([str(RUNTIME / "bin/python"), "-c",
        "import torch,astate,astate._core.astate_cpp as native; print(torch.__version__,astate.__version__,native.__file__)"],
        env=env, text=True)
    print(imports, flush=True)
    archive = BASE / "astate-runtime.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        output.add(TARGET, arcname=TARGET.name)
        output.add(BASE / "astate-artifact.json", arcname="astate-artifact.json")
    print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size,
                          sha256=digest(archive), manifest=manifest)), flush=True)


if __name__ == "__main__":
    main()
