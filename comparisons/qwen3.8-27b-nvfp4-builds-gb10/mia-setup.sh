#!/bin/bash
# Owner-approved (2026-11-03): MiaAI-Lab's exllamav3 fork on host A, for comparing their
# EXL3 Qwen3.8-27B checkpoint through CapyCTL as a custom engine. Home directory only.
# Mirrors the kit's own start.sh install (steps 1-5), with every version pinned.
set -euo pipefail
V=$HOME/exllamav3-mia-venv
KIT=$HOME/mia-kit
EXL3_COMMIT=63b32f001d7b2cfed3b3e3aaf25f534ba53cc7ed
KIT_COMMIT=1242187390f780dba907e30659f1639fd0b491c8
if [ ! -d "$KIT/.git" ]; then
  git clone -q https://github.com/MiaAI-Lab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw "$KIT"
fi
git -C "$KIT" checkout -q "$KIT_COMMIT"
python3.12 -m venv "$V"
"$V/bin/pip" install --quiet --upgrade pip==25.3 setuptools wheel typing_extensions packaging
"$V/bin/pip" install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
# MAX_JOBS=4 under nice when another run shares the machine (the run of 2026-11-03 built with 4).
export TORCH_CUDA_ARCH_LIST="12.0;12.1" CUDA_HOME=/usr/local/cuda MAX_JOBS=${MAX_JOBS:-8}
"$V/bin/pip" install --no-build-isolation "git+https://github.com/MiaAI-Lab/exllamav3@$EXL3_COMMIT"
"$V/bin/pip" install --quiet aiohttp huggingface_hub
"$V/bin/pip" freeze > "$HOME/q38-requant/exllamav3-venv-freeze.txt"
"$V/bin/python" -c 'import torch, exllamav3; print("torch", torch.__version__, "exllamav3", getattr(exllamav3, "__version__", "?"))'
# CapyCTL refuses a custom engine whose executable is a symlink, and the venv's python is one:
# the kit's server as one executable file in this environment.
cat >"$V/bin/mia-serve" <<EOF
#!$V/bin/python -u
# MiaAI-Lab kit server ($KIT/tools/serve_openai.py) as one executable in this
# environment, for a CapyCTL custom engine (the executable may not be a symlink).
import runpy, sys
sys.argv[0] = "$KIT/tools/serve_openai.py"
runpy.run_path(sys.argv[0], run_name="__main__")
EOF
chmod 755 "$V/bin/mia-serve"
"$V/bin/python" - <<'EOF'
import os
from huggingface_hub import snapshot_download
for repo, rev, dest in [
    ("Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw", "19441ac874c4018295da848e250f23511361cda4",
     "~/models/Mia-AiLab--Qwen3.8-27B-EXL3-3.5bpw@19441ac8"),
    ("Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw", "4f0436269bca761b071f05319e8e04a87cc633f9",
     "~/drafters/Mia-AiLab--Qwen3.8-27B-DFlash2-EXL3-5.0bpw@4f043626"),
]:
    print(snapshot_download(repo, revision=rev, local_dir=os.path.expanduser(dest)))
EOF
echo MIA-SETUP-DONE
