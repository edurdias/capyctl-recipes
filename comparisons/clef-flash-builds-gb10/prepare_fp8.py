#!/usr/bin/env python3
"""Prepare the pinned upstream FP8 artifact without publisher runtime plugins."""
from concurrent.futures import ThreadPoolExecutor
import os
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
import urllib.request

REPO = "kurcontko/clef-flash-FP8-Dynamic"
REVISION = "7b43c2d799e7b5e9fef96b6137840088d75af0d6"
DEST = Path.home() / "clef-flash-fp8-7b43c2d7"
OUT = Path(__file__).resolve().parent


def download(path, url, size):
    progress = path.with_suffix(".download-ranges.json")
    state = json.loads(progress.read_text()) if progress.exists() else {"prefix": path.stat().st_size if path.exists() else 0, "completed": []}
    start = state["prefix"]
    if size < 1024 * 1024 * 1024:
        with urllib.request.urlopen(url, timeout=300) as response, path.open("wb") as output:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
        return
    # The CDN closed a monolithic transfer early; use bounded, verified ranges.
    ranges = [(i, offset, min(offset + 64 * 1024 * 1024, size) - 1)
              for i, offset in enumerate(range(start, size, 64 * 1024 * 1024)) if i not in state["completed"]]
    progress.write_text(json.dumps(state))
    lock = threading.Lock()
    fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
    def chunk(bounds):
        index, first, last = bounds
        for attempt in range(6):
            try:
                request = urllib.request.Request(url, headers={"Range": f"bytes={first}-{last}"})
                with urllib.request.urlopen(request, timeout=300) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {first}-{last}/{size}":
                        raise RuntimeError("server did not honor byte range")
                    offset = first
                    while data := response.read(8 * 1024 * 1024):
                        written = 0
                        while written < len(data):
                            written += os.pwrite(fd, data[written:], offset + written)
                        offset += len(data)
                    if offset != last + 1:
                        raise OSError(f"short byte range {first}-{last}: received through {offset}")
                break
            except OSError:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        with lock:
            state["completed"].append(index)
            progress.write_text(json.dumps(state))
        print(f"Completed bytes {first}-{last}", flush=True)
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(chunk, ranges))
    finally:
        os.close(fd)


def main():
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true") as response:
        meta = json.load(response)
    DEST.mkdir(exist_ok=True)
    files = []
    for entry in meta["siblings"]:
        src = entry["rfilename"]
        if src.startswith(("vllm_plugin/", "scripts/", ".")) or src.endswith(".py"):
            continue
        target = src.removeprefix("clef_head/")
        path = DEST / target
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.stat().st_size != entry["size"] or path.with_suffix(".download-ranges.json").exists():
            print(f"Downloading {src} ({entry['size']} bytes)", flush=True)
            download(path, f"https://huggingface.co/{REPO}/resolve/{REVISION}/{src}", entry["size"])
        sha = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
        if path.stat().st_size != entry["size"]:
            raise RuntimeError(f"size mismatch: {src}")
        expected = entry.get("lfs", {}).get("sha256")
        if expected and sha != expected:
            raise RuntimeError(f"SHA-256 mismatch: {src}")
        files.append({"source": src, "prepared": target, "bytes": entry["size"], "sha256": sha})
    manifest = {"repo": REPO, "revision": REVISION, "preparation": "clef_head/joint_head.* relocated to root; publisher plugin, scripts and Python omitted", "files": files, "bytes": sum(f["bytes"] for f in files)}
    (OUT / "fp8-preparation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
