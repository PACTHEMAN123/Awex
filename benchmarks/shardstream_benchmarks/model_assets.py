"""Fetch anonymous ModelScope snapshots and verify their advertised SHA256."""

import argparse
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_files(model):
    url = f"https://modelscope.cn/api/v1/models/{model}/repo/files?Revision=master&Recursive=true"
    with urllib.request.urlopen(url, timeout=30) as response:
        result = json.load(response)
    if not result.get("Success"):
        raise RuntimeError(f"Cannot list ModelScope snapshot: {model}")
    # Only root checkpoint/tokenizer/processor assets; skip alternate formats.
    return [
        file
        for file in result["Data"]["Files"]
        if file["Type"] == "blob"
        and "/" not in file["Path"]
        and file["Path"].endswith(
            (".safetensors", ".json", ".jinja", ".model", ".txt", ".tiktoken")
        )
    ]


def download_file(model, file, output, source_url=None):
    path = output / file["Path"]
    expected_size, expected_sha = file["Size"], file["Sha256"]
    if not expected_sha or len(expected_sha) != 64:
        raise ValueError(f"Missing snapshot checksum: {file['Path']}")
    if (
        path.exists()
        and path.stat().st_size == expected_size
        and sha256(path) == expected_sha
    ):
        return {
            "path": file["Path"],
            "bytes": expected_size,
            "sha256": expected_sha,
            "revision": file["Revision"],
        }
    temporary = path.with_name(path.name + ".partial")
    query = urllib.parse.urlencode(
        {"Revision": file["Revision"], "FilePath": file["Path"]}
    )
    url = f"https://modelscope.cn/api/v1/models/{model}/repo?{query}"
    if source_url:
        url = source_url.rstrip("/") + "/" + urllib.parse.quote(file["Path"])
    for attempt in range(3):
        try:
            offset = temporary.stat().st_size if temporary.exists() else 0
            if offset >= expected_size:
                if offset == expected_size and sha256(temporary) == expected_sha:
                    os.replace(temporary, path)
                    break
                temporary.unlink()
                offset = 0
            request = urllib.request.Request(url)
            if offset:
                request.add_header("Range", f"bytes={offset}-")
            with urllib.request.urlopen(request, timeout=60) as response:
                resumed = response.status == 206 and response.headers.get(
                    "Content-Range", ""
                ).startswith(f"bytes {offset}-")
                with temporary.open("ab" if resumed else "wb") as stream:
                    for block in iter(lambda: response.read(8 << 20), b""):
                        stream.write(block)
            if (
                temporary.stat().st_size != expected_size
                or sha256(temporary) != expected_sha
            ):
                temporary.unlink()
                raise RuntimeError(f"Checkpoint checksum mismatch: {file['Path']}")
            os.replace(temporary, path)
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)
    print(json.dumps({"verified": file["Path"], "bytes": expected_size}), flush=True)
    return {
        "path": file["Path"],
        "bytes": expected_size,
        "sha256": expected_sha,
        "revision": file["Revision"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument(
        "--source-url",
        help="Copy the same pinned, checksummed snapshot from another node",
    )
    args = parser.parse_args()
    if args.source_url:
        with urllib.request.urlopen(
            args.source_url.rstrip("/") + "/modelscope-source.json", timeout=30
        ) as response:
            saved_source = json.load(response)
        if saved_source["model"] != args.model:
            raise ValueError("Source URL belongs to a different model")
        files = saved_source["files"]
    else:
        files = snapshot_files(args.model)
    if not any(file["Path"].endswith(".safetensors") for file in files):
        raise RuntimeError("Snapshot has no safetensors checkpoint")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "modelscope-source.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved["model"] != args.model:
            raise ValueError("Output directory belongs to a different model")
        files = saved["files"]
    else:
        manifest.write_text(
            json.dumps({"model": args.model, "files": files}, indent=2) + "\n"
        )
    print(
        json.dumps(
            {
                "model": args.model,
                "files": len(files),
                "bytes": sum(file["Size"] for file in files),
            }
        ),
        flush=True,
    )
    if args.metadata_only:
        return
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        verified = list(
            pool.map(
                lambda file: download_file(
                    args.model, file, args.output, args.source_url
                ),
                files,
            )
        )
    (args.output / "verified-assets.json").write_text(
        json.dumps({"model": args.model, "files": verified}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
