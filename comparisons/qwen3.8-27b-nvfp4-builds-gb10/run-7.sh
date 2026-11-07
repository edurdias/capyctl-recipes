#!/usr/bin/env bash
# Run 7: Shardio's Qwen3.8-27B recipe r13 (bld_qwen3.8-27b-sglang-xrargf) against ours, on one GB10.
#   a. RadixArk/Qwen3.8-27B-NVFP4 at 319f741c, the checkpoint r13 serves: its tensors read against
#      the BF16 source, and its greedy next-token agreement with BF16 (../tensors/, weights only and
#      BF16's own answers as the prefix).
#   b. r13's SGLang flags through CapyCTL, on RadixArk's checkpoint (sg-rx-r13) and on ours
#      (sg-full-r13): 1, 4 and 8 streams, 3 rounds, a 2k context sweep (the shape of runs 1 to 3).
# Waits for the checkpoint download (tmux q38-dl) to finish first.
set -uo pipefail
while tmux has-session -t =q38-dl 2>/dev/null; do sleep 30; done
RX=$HOME/q38-requant/ckpts/RadixArk--Qwen3.8-27B-NVFP4@319f741cce68d7914884900c138a1fbb70a42f30
T=$HOME/q38-requant/tensors
say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$T/progress.log"; }
grep -q '^exit=0' "$HOME/q38-requant/dl-radixark.log" || { say "radixark download failed"; exit 1; }
cd "$T" || exit 1
say "read radixark"
"$HOME/modelopt-0.47.0-venv/bin/python" read_tensors.py --source "$HOME/models/qwen3.8-27b" --ckpt "$RX" \
  --name radixark --out read-radixark.csv --search > read-radixark.log 2>&1 || say "read radixark failed"
say "score radixark"
( export PATH=$HOME/vllm-0.32.0-venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
  python next_token_agree.py score --model "$RX" --answers bf16-answers.json --name radixark \
    --out score-radixark.json > score-radixark.log 2>&1 )
if [ -f score-radixark.json ]; then say "$(grep -h '^radixark:' score-radixark.log)"; else say "score radixark failed"; fi
say "speed: r13 on radixark and on capyctl"
WORK=$HOME/q38-requant/run-7 CONC=1,4,8 ROUNDS=3 SWEEP=2k SWEEP_RUNS=2 ACCURACY="" \
  bash "$HOME/q38-requant/recipes/comparisons/qwen3.8-27b-nvfp4-builds-gb10/run-host.sh" sg-rx-r13 sg-full-r13
say "run 7 done"
