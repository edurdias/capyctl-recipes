#!/usr/bin/env python3
"""Download immutable checkpoint snapshots with an existing engine environment."""
import argparse
import json
from pathlib import Path

PINS = {
    "nvidia": ("nvidia/Gemma-4-26B-A4B-NVFP4", "a19cfe00be84568a6867111c9a68c9c44fdcffe6"),
    "redhat": ("RedHatAI/gemma-4-26B-A4B-it-NVFP4", "5557756b8dce33ac72f2bd702b11729fdba3b839"),
    "redhat-fp8": ("RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic", "ed35d7abe5d940da41b4ff06eb482feb0be8cb44"),
    "unsloth": ("unsloth/gemma-4-26B-A4B-it-NVFP4", "20df0542b1a86ce19f495ac2eca2c7c12bce82f9"),
    "mtp": ("google/gemma-4-26B-A4B-it-assistant", "6e5aaaf4c42b98394530b8fda2e95cadd65c151c"),
    "dspark": ("makora-ai/gemma4-26b-a4b-dspark", "e6f739231e892fe4512c470136490d5441ffe62f"),
    "eagle3": ("RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3", "226b2692d48029aae52adc1d0021a9d58709d33c"),
    "dflash": ("z-lab/gemma-4-26B-A4B-it-DFlash", "77d4202772dfe50b2396ec7bac9cfffc7b9e7057"),
}


def model_path(key):
    repo, revision = PINS[key]
    manifest = Path.home() / "gemma4-engine-search/pins.json"
    if manifest.is_file():
        entry = json.loads(manifest.read_text()).get("models", {}).get(key)
        if entry and entry.get("revision") == revision:
            return Path(entry["path"])
    return Path.home() / "models" / (repo.replace("/", "--") + "@" + revision)


if __name__ == "__main__":
    from huggingface_hub import snapshot_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=PINS, default=list(PINS))
    parser.add_argument("--reuse", action="append", default=[], help="owner-confirmed pinned key=existing-path; do not download")
    parser.add_argument("--path", action="append", default=[], help="key=destination-path override")
    args = parser.parse_args()
    out = Path.home() / "gemma4-engine-search"
    out.mkdir(exist_ok=True)
    manifest_path = out / "pins.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    manifest = previous.get("models", {})
    failures = previous.get("failures", {})
    reused = dict(item.split("=", 1) for item in args.reuse)
    paths = dict(item.split("=", 1) for item in args.path)
    for key in args.only:
        repo, revision = PINS[key]
        dest = Path(paths.get(key, reused.get(key, model_path(key)))).expanduser()
        try:
            if key in reused:
                if not dest.is_dir() or not list(dest.glob("*.safetensors")):
                    raise ValueError("reused checkpoint directory lacks safetensors")
            else:
                snapshot_download(repo, revision=revision, local_dir=dest, token=False, max_workers=4)
            manifest[key] = {"repo": repo, "revision": revision, "path": str(dest),
                             "safetensors_bytes": sum(p.stat().st_size for p in dest.glob("*.safetensors"))}
            manifest[key]["provenance"] = "owner-confirmed existing pin" if key in reused else "snapshot_download pinned revision"
            failures.pop(key, None)
            print(json.dumps({key: manifest[key]}), flush=True)
        except Exception as exc:
            failures[key] = str(exc)
            print(json.dumps({"failed": key, "error": str(exc)}), flush=True)
        manifest_path.write_text(json.dumps({"models": manifest, "failures": failures}, indent=2) + "\n")
    raise SystemExit(bool(failures))
