#!/usr/bin/env bash
# Run 9: run 5 again on the checkpoint's second revision (block scales searched; both deployment.yaml files
# now pin 0ce7e69f): each recipe's file as it ships (rcp-vl, rcp-sg) with capyctl-bench's full sweeps,
# first start and warm start, then the SGLang recipe's settings (sg-full-df-tuned) for run 4's accuracy set.
# Run 8 measured the vLLM recipe's accuracy on this revision. Weights from the local copy, the same bytes
# as the Hugging Face revision.
set -euo pipefail
export WORK=${WORK:-$HOME/q38-requant/run-9}
export FULL=${FULL:-$HOME/q38-requant/cand-scale-search}
export ACCURACY="sg-full-df-tuned"
exec bash "$HOME/q38-requant/recipes/comparisons/qwen3.8-27b-nvfp4-builds-gb10/run-host.sh" rcp-vl rcp-sg sg-full-df-tuned
