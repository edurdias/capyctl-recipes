#!/bin/bash
# Run 3, chained after run 2: the search for the recipe. DFlash2 only, 8 rounds at 1, 4 and 8
# streams (8, 32 and 64 requests a variant) and the 2k context point twice.
#   1. Attribution: MiaAI-Lab's SGLang flags on nvidia's checkpoint (ours had them in run 2).
#   2. SGLang around the run 2 leader (ours, tuned): again (run-to-run spread), three
#      continuous decode steps, torch.compile, 6 and 12 draft tokens.
#   3. vLLM with its DeltaNet state cache in bf16, both checkpoints.
while tmux has-session -t =q38-bench2 2>/dev/null; do sleep 60; done
export WORK=$HOME/q38-requant/run-3 CONC=1,4,8 ROUNDS=8 SWEEP=2k SWEEP_RUNS=2
exec bash "$HOME/q38-requant/recipes/comparisons/qwen3.8-27b-nvfp4-builds-gb10/run-host.sh" \
  sg-nv-df-tuned sg-full-df-tuned sg-full-df-cds3 sg-full-df-tuned-compile \
  sg-full-df-tuned-k6 sg-full-df-tuned-k12 vl-nv-df-mbf16 vl-full-df-mbf16
