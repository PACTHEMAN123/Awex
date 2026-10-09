"""Compare native protocol tokens and PTX to the pinned pre-refactor source."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def normalize(text: str) -> list[str]:
    # Error text/comments and include paths can change; executable expressions cannot.
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)
    text = re.sub(r"^\s*#include[^\n]*", "", text, flags=re.M)
    text = text.replace("namespace awex", "namespace shardstream")
    text = text.replace("namespace nccl_device_v2", "namespace transport")
    text = text.replace("AWEX_NCCL_DEVICE_V2_HAS_GIN", "SHARDSTREAM_HAS_GIN")
    text = text.replace("launchDeviceV2", "launch").replace(
        "device_v2_kernel", "transport_kernel"
    )
    text = text.replace("v2Gin", "gin")
    text = re.sub(
        r"\bv2([A-Z][A-Za-z0-9_]*)", lambda m: m[1][0].lower() + m[1][1:], text
    )
    text = text.replace("V2", "")
    text = text.replace("topologyPowerOfTwoUp", "powerOfTwoUp")
    text = re.sub(r"\bthreadRoles\b", "roles", text)
    text = re.sub(r'"(?:\\.|[^"\\])*"', '"STRING"', text)
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|\S", text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    baseline = args.source / "awex/transfer/nccl_device_v2"
    manifest = json.loads((root / "docs/source-manifest.json").read_text())
    for record in manifest["native_files"]:
        original = args.source / record["source"]
        assert (
            hashlib.sha256(original.read_bytes()).hexdigest() == record["source_sha256"]
        ), f"Baseline source changed: {record['source']}"
    records = []
    for old in sorted(baseline.iterdir()):
        if (
            old.suffix not in {".h", ".cuh", ".cu"}
            or old.name == "nccl_device_v2_ext.cu"
        ):
            continue
        name = old.name.replace("device_v2_", "")
        new = root / ("src" if old.suffix == ".cu" else "include/shardstream") / name
        a, b = old.read_text(), new.read_text()
        assert normalize(a) == normalize(b), f"Protocol expressions changed: {old.name}"
        # PTX strings encode instructions, including literal .v2 vector modifiers.
        asm = r'asm\s+volatile\s*\(\s*("(?:\\.|[^"\\])*")'
        assert re.findall(asm, a) == re.findall(asm, b), f"PTX changed: {old.name}"
        records.append(
            {
                "file": str(new.relative_to(root)),
                "sha256": hashlib.sha256(new.read_bytes()).hexdigest(),
            }
        )
    print(
        json.dumps(
            {
                "passed": True,
                "scope": "kernel/protocol/lowering/topology expressions and inline PTX; excludes host runtime/binding refactor",
                "files": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
