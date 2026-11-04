#!/bin/bash
# Run 5, after run 4: each recipe's deployment.yaml as it ships (rcp-vl: vllm/, rcp-sg: sglang/),
# with capyctl-bench's full sweeps (context 0.5k to 32k, 3 runs; 1 to 8 streams, 5 rounds), the
# first start (kernel caches from the earlier runs on this machine) and a warm start.
while tmux has-session -t =q38-bench4 2>/dev/null; do sleep 60; done
export WORK=$HOME/q38-requant/run-5 ACCURACY=""
exec bash "$HOME/q38-requant/recipes/comparisons/qwen3.8-27b-nvfp4-builds-gb10/run-host.sh" rcp-vl rcp-sg
