#!/usr/bin/env bash
# Run 8: the checkpoint's second revision (block scales searched, ../../vllm/qwen3.8-27b-capyctl-nvfp4-gb10/
# convert/requant_scales.py) through the vLLM recipe's settings: run 3's variant vl-full-df-mbf16 (DFlash2, 7
# tokens, DeltaNet state cache bf16) at run 3's shape (1, 4 and 8 streams, 8 rounds, a 2k context sweep, 2
# runs), then run 4's accuracy set (agreement with BF16, GSM8K 250, 30 tool calls). Weights from the local
# copy, the same bytes as the Hugging Face revision.
set -euo pipefail
export WORK=${WORK:-$HOME/q38-requant/run-8}
export FULL=${FULL:-$HOME/q38-requant/cand-scale-search}
export CONC=1,4,8 ROUNDS=8 SWEEP=2k SWEEP_RUNS=2 ACCURACY="vl-full-df-mbf16"
exec bash "$HOME/q38-requant/recipes/comparisons/qwen3.8-27b-nvfp4-builds-gb10/run-host.sh" vl-full-df-mbf16
