#!/bin/bash
# Run 4: after the sweep (run 3) picked the recipe.
#   a. MiaAI-Lab's exllamav3 kit, speed only, with mia_speed.py (its server streams without a
#      token count); its accuracy is in run 2. 3 rounds at 1, 4 and 8 streams.
#   b. The two recipes as they ship, accuracy (agreement with BF16, GSM8K, tool calls): vLLM
#      0.32.0 with DFlash2 and the DeltaNet state cache in bf16, and SGLang 0.5.22 with DFlash2 and
#      MiaAI-Lab's flags. Speculative decoding verifies every token against the target, but the
#      bf16 state changes what the target computes, so its accuracy is measured, not assumed.
set -u
R=$HOME/q38-requant/recipes/comparisons/qwen3.8-27b-nvfp4-builds-gb10
WORK=$HOME/q38-requant/run-4a CONC=1,4,8 ROUNDS=3 ACCURACY="" bash "$R/run-host.sh" cu-mia cu-mia-df
WORK=$HOME/q38-requant/run-4b CONC=1 ROUNDS=1 SWEEP=2k SWEEP_RUNS=1 \
  ACCURACY="vl-full-df-mbf16 sg-full-df-tuned" bash "$R/run-host.sh" vl-full-df-mbf16 sg-full-df-tuned
