#!/usr/bin/env bash
# Weights-only tensor reading, then greedy next-token agreement with BF16, on host A.
# One live session: run under tmux q38-tensors; marker ~/LIVE-SESSION-IN-USE.txt.
set -uo pipefail
# read_tensors.py (../../../vllm/qwen3.8-27b-capyctl-nvfp4-gb10/convert/), next_token_agree.py and
# agreement-prompts.json (../../qwen3.8-27b-sglang-dflash2-vs-mtp-eagle-gb10/accuracy/) sit in this
# directory on the host.
cd "$HOME/q38-requant/tensors" || exit 1
S=$HOME/models/qwen3.8-27b
FULL=$HOME/models/Qwen3.8-27B-NVFP4-Full
NV=$HOME/models/sources/huggingface/nvidia--Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80
GT=$HOME/q38-requant/ckpts/gittensor-model-hub--Qwen3.8-27B-NVFP4-RTX5090@5b7a687fc8211a5d631c8ca6a593dd37eb26ce33
say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a progress.log; }

MO=$HOME/modelopt-0.47.0-venv/bin/python
for b in "capyctl $FULL" "nvidia $NV" "gittensor $GT"; do
  # shellcheck disable=SC2086 # split "name path" into $1 $2 on purpose
  set -- $b
  say "read $1"
  $MO read_tensors.py --source "$S" --ckpt "$2" --name "$1" --out "read-$1.csv" --search > "read-$1.log" 2>&1 || say "read $1 failed"
done

export PATH=$HOME/vllm-0.32.0-venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
say "gen bf16 answers"
python next_token_agree.py gen --model "$S" --prompts agreement-prompts.json --out bf16-answers.json --max-tokens 512 \
  > gen-bf16.log 2>&1
[ -f bf16-answers.json ] || { say "gen failed"; exit 1; }
for b in "bf16 $S" "capyctl $FULL" "nvidia $NV" "gittensor $GT"; do
  # shellcheck disable=SC2086 # split "name path" into $1 $2 on purpose
  set -- $b
  say "score $1"
  # vLLM aborts on shutdown after writing its results; the file is what counts.
  python next_token_agree.py score --model "$2" --answers bf16-answers.json --name "$1" --out "score-$1.json" \
    > "score-$1.log" 2>&1
  if [ -f "score-$1.json" ]; then say "$(grep -h "^$1:" "score-$1.log")"; else say "score $1 failed"; fi
done
say "done"
