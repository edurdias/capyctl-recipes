"""Convert microsoft/FrogNano-4B-2609 to the MLX affine 4-bit checkpoint this
recipe serves (capyctl/FrogNano-4B-2609-MLX-4bit).

usage: python convert.py SOURCE_DIR OUTPUT_DIR [--bits 4] [--group-size 64]

SOURCE_DIR is a local copy of microsoft/FrogNano-4B-2609 at
b90468c11a913c1916b4542b4b7a530ec42024a4. OUTPUT_DIR must not exist.
Run it with the Python of a venv that has mlx-lm 0.32.0 (mlx 0.32.3; the CPU
backend is enough). See README.md in this directory.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mlx.core as mx

EMBED = "model.language_model.embed_tokens.weight"
# <|im_end|> ends a chat turn; <|endoftext|> is the only eos config.json names.
EOS = [248046, 248044]
# mlx-lm with transformers 5.x rewrites the first two and drops the last two.
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]


def stage_untied(src: Path, stage: Path) -> None:
    """Link the source files into `stage` and add lm_head as a copy of embed_tokens."""
    for f in src.iterdir():
        if f.name not in ("config.json", "model.safetensors.index.json"):
            os.symlink(f.resolve(), stage / f.name)

    cfg = json.loads((src / "config.json").read_text())
    cfg["tie_word_embeddings"] = False
    cfg["text_config"]["tie_word_embeddings"] = False
    (stage / "config.json").write_text(json.dumps(cfg, indent=2))

    index = json.loads((src / "model.safetensors.index.json").read_text())
    shard = index["weight_map"][EMBED]
    embed = mx.load(str(src / shard))[EMBED]
    mx.save_safetensors(
        str(stage / "model-lm_head.safetensors"),
        {"lm_head.weight": embed},
        metadata={"format": "pt"},
    )
    index["weight_map"]["lm_head.weight"] = "model-lm_head.safetensors"
    (stage / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    print(f"staged untied lm_head {tuple(embed.shape)} {embed.dtype}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=64)
    a = ap.parse_args()

    src = a.source.resolve()
    if not (src / "config.json").is_file():
        sys.exit(f"{src} has no config.json")
    if a.output.exists():
        sys.exit(f"{a.output} exists; mlx_lm.convert writes a new directory")

    with tempfile.TemporaryDirectory(prefix="frognano-untied-") as tmp:
        stage = Path(tmp)
        stage_untied(src, stage)
        # mlx-lm's qwen3_5 loader drops the vision tower and the MTP layer.
        subprocess.run(
            [
                sys.executable, "-m", "mlx_lm", "convert",
                "--hf-path", str(stage),
                "--mlx-path", str(a.output),
                "-q", "--q-bits", str(a.bits), "--q-group-size", str(a.group_size),
            ],
            check=True,
        )

    for name in TOKENIZER_FILES:
        shutil.copyfile(src / name, a.output / name)
    (a.output / "generation_config.json").write_text(
        '{\n  "eos_token_id": ' + json.dumps(EOS) + "\n}\n"
    )
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
