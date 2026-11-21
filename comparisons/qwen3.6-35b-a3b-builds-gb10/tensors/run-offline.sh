#!/usr/bin/env bash
set -euo pipefail
# Run on session-a after the pinned source download completes. No native engine.
work="$HOME/qwen36-work"
python="$HOME/modelopt-0.47.0-venv/bin/python"
cd "$work/tensors"
for variant in bf16 nvidia delta attention combined; do
  "$python" -u next_token_agree.py "$variant" --max-tokens 128 --batch-size 8 > "$work/$variant-agreement.log" 2>&1
done
"$python" summarize.py > "$work/family-ablation.log" 2>&1
