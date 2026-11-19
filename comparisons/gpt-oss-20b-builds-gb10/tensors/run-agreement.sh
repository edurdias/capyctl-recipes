#!/bin/sh
# Run on host A ONLY after the integration owner explicitly clears the GPU.
# The Python harness also refuses to start if a GPU compute process exists.
set -eu
export HF_HUB_OFFLINE=1
exec "$HOME/modelopt-0.47.0-venv/bin/python" "$HOME/gptoss-work/next_token_agree.py" \
  --source "$HOME/models/gpt-oss-20b" \
  --candidate "$HOME/models/gpt-oss-20b-capyctl-fp8attn" \
  --prompts "$HOME/gptoss-work/agreement-prompts.json" \
  --audit "$HOME/gptoss-work/checkpoint-audit.json" \
  --results "$HOME/gptoss-work/agreement" --batch-size 8
