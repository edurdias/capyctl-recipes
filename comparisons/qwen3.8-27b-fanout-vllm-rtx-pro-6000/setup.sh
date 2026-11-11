#!/usr/bin/env bash
# Runs on the rented GPU machine. Installs only: uv, vLLM 0.32.0 (fresh venv), CapyCTL 0.1.12
# (release binary), the capyctl/Qwen3.8-27B-NVFP4 checkpoint and the DFlash2 drafter.
# Everything lives under $HOME; no system package, driver or CUDA change.
set -euo pipefail
umask 077
cd ~
exec > >(tee -a ~/fanout/setup.log) 2>&1
echo "== machine"
uname -m; . /etc/os-release; echo "$PRETTY_NAME"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
nvidia-smi | sed -n '3p'
free -g | sed -n '1,2p'; df -h ~ | tail -1; nproc

CUDA_MAJOR=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1\2/p')
if [ "${CUDA_MAJOR:-0}" -ge 130 ]; then TB=cu130; else TB=cu128; fi
echo "torch backend: $TB (driver CUDA ${CUDA_MAJOR})"

echo "== uv"
[ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh
UV=~/.local/bin/uv

echo "== downloads (background)"
mkdir -p ~/models ~/drafters
(
  $UV tool run --from 'huggingface_hub[cli,hf_xet]' hf download capyctl/Qwen3.8-27B-NVFP4 \
    --revision 0ce7e69f126b6b6f6fd2058373d9f81ec360adee --local-dir ~/models/qwen3.8-27b-capyctl-nvfp4 >~/fanout/dl-model.log 2>&1
  echo "model rc=$?" >>~/fanout/dl-model.log
) &
DL1=$!
(
  $UV tool run --from 'huggingface_hub[cli,hf_xet]' hf download z-lab/Qwen3.8-27B-DFlash2 \
    --revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 --local-dir ~/drafters/Qwen3.8-27B-DFlash2 >~/fanout/dl-drafter.log 2>&1
  echo "drafter rc=$?" >>~/fanout/dl-drafter.log
) &
DL2=$!

echo "== vLLM 0.32.0"
if [ ! -x ~/vllm-0.32.0-venv/bin/vllm ]; then
  # uv's own Python: Triton compiles a helper at start and needs Python.h, which the
  # image's system Python 3.12 lacks.
  $UV venv --python-preference only-managed -p 3.12 ~/vllm-0.32.0-venv
  $UV pip install --python ~/vllm-0.32.0-venv/bin/python vllm==0.32.0 --torch-backend=$TB
fi
# Confidential Computing guests only (the Verda RTX PRO 6000 CC image): mark the GPU
# ready, and keep vLLM off UVA input buffers, which the GPU cannot read there.
# Expects nouva/sitecustomize.py from this directory at ~/fanout/nouva/.
if nvidia-smi conf-compute -f 2>/dev/null | grep -q 'CC status: ON'; then
  nvidia-smi conf-compute -srs 1
  echo 'import sys; exec(open("'"$HOME"'/fanout/nouva/sitecustomize.py").read())' \
    >~/vllm-0.32.0-venv/lib/python3.12/site-packages/zz_capyctl_cc_nouva.pth
fi
~/vllm-0.32.0-venv/bin/python -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))'
$UV pip freeze --python ~/vllm-0.32.0-venv/bin/python | grep -iE '^(vllm|torch|triton|flashinfer|transformers)' 

echo "== CapyCTL 0.1.12"
cd ~/fanout
sha256sum -c --ignore-missing SHA256SUMS
tar -xzf capyctl-0.1.12-linux-x86_64.tar.gz
(cd capyctl-0.1.12-linux-x86_64 && sha256sum -c --quiet SHA256SUMS)
mkdir -p ~/.local/bin && install -m 0755 capyctl-0.1.12-linux-x86_64/bin/capyctl ~/.local/bin/capyctl
~/.local/bin/capyctl --version

wait $DL1 $DL2
cat ~/fanout/dl-model.log ~/fanout/dl-drafter.log | grep rc=
du -sh ~/models/qwen3.8-27b-capyctl-nvfp4 ~/drafters/Qwen3.8-27B-DFlash2
echo SETUP_OK
