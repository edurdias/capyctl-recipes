#!/usr/bin/env bash
# Run 6, after run 5: both recipes with the concurrency cap raised to 48, to find where total
# throughput saturates. Three rounds per stream count, no context sweep, no accuracy.
#   6:  rcp-vl-hi as shipped otherwise (16 GiB KV cache), 8 to 48 streams. rcp-sg-hi with an
#       85 GiB request was refused by CapyCTL's host policy before launch: SGLang keeps 49.7 GB of
#       DeltaNet and draft-token state for 48 running requests, so it needs 93.5 GiB.
#   6b: rcp-sg-hi with a 94 GiB request, 8 to 48 streams.
#   6c: rcp-vl-hi with a 32 GiB KV cache: in 6, vLLM ran at most 22 requests at once, the
#       16 GiB cache's limit (it holds vLLM's DeltaNet state blocks too), not the engine's.
set -euo pipefail
H=$HOME/q38-requant/recipes/comparisons/qwen3.8-27b-nvfp4-builds-gb10/run-host.sh
export ROUNDS=3 HI_CAP=48 ACCURACY=""
WORK=$HOME/q38-requant/run-6 CONC=8,12,16,24,32,48 bash "$H" rcp-vl-hi
WORK=$HOME/q38-requant/run-6b CONC=8,12,16,24,32,48 HI_SG_REQUEST=94GiB bash "$H" rcp-sg-hi
WORK=$HOME/q38-requant/run-6c CONC=16,24,32,40,48 HI_VL_KV=32GiB bash "$H" rcp-vl-hi
