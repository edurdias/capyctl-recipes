#!/usr/bin/env bash
# Run 8b, straight after run 8: the first revision through the same variant and shape, speed only, as the
# same-day control (run 8's 8-stream median was 166 tok/s against run 3's 181 two days earlier).
set -euo pipefail
export WORK=${WORK:-$HOME/q38-requant/run-8b}
export FULL=${FULL:-$HOME/models/Qwen3.8-27B-NVFP4-Full}
export CONC=1,4,8 ROUNDS=8 SWEEP=2k SWEEP_RUNS=2 ACCURACY=""
exec bash "$HOME/q38-requant/recipes/comparisons/qwen3.8-27b-nvfp4-builds-gb10/run-host.sh" vl-full-df-mbf16
