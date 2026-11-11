#!/usr/bin/env bash
# One hand launch of vLLM on 127.0.0.1:18800 with the given extra args/env, one greedy
# request, print the reply, stop. Usage: [ENV=..] bisect.sh <label> [vllm args...]
set -uo pipefail
label=$1; shift
M=~/models/qwen3.8-27b-capyctl-nvfp4
log=~/fanout/bisect-$label.log
~/vllm-0.32.0-venv/bin/vllm serve "$M" --served-model-name m --host 127.0.0.1 --port 18800 \
  --max-model-len 8192 --max-num-seqs 4 --language-model-only --gpu-memory-utilization 0.45 "$@" >"$log" 2>&1 &
pid=$!
for _ in $(seq 600); do curl -sf 127.0.0.1:18800/health >/dev/null && break; kill -0 $pid 2>/dev/null || break; sleep 1; done
reply=$(curl -s 127.0.0.1:18800/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"m","temperature":0,"max_tokens":40,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"What is the capital of France? Answer in one sentence."}]}')
echo "[$label] $(echo "$reply" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(repr(d["choices"][0]["message"]["content"]))' 2>/dev/null || tail -3 "$log")"
grep -E "Using .*NvFp4|NVFP4 GEMM" "$log" | sed 's/.*\] //' | sort -u
kill -TERM $pid; wait $pid 2>/dev/null
