#!/usr/bin/env bash
# On the rental: register vLLM, start standalone, deploy the recipe, run the experiment.
# Usage: serve.sh up | measure | down
set -uo pipefail
export PATH=~/.local/bin:$PATH
F=~/fanout
SD=$F/state
c() { capyctl --state-dir "$SD" "$@"; }
case ${1:-} in
up)
  mkdir -m 700 -p "$SD"
  if ! pgrep -f -- "--state-dir $SD start standalone" >/dev/null; then
    setsid nohup capyctl --state-dir "$SD" start standalone --listen 127.0.0.1:18743 \
      --management-listen 127.0.0.1:18744 --models-root ~/models --json >"$F/role.log" 2>&1 </dev/null &
    for _ in $(seq 120); do grep -q '"ready":true' "$F/role.log" 2>/dev/null && break; sleep 1; done
  fi
  grep -q '"ready":true' "$F/role.log" || { tail -20 "$F/role.log"; exit 1; }
  c engine list | grep -q vl032 ||
    c engine add ~/vllm-0.32.0-venv --name vl032 --approve-option=--speculative-config --approve-path ~/drafters
  sed "s|@HOME@|$HOME|" "$F/deployment.yaml" >"$F/deployment.rendered.yaml"
  c deploy model --file "$F/deployment.rendered.yaml"
  s=$SECONDS
  c start deployment q38 --wait
  rc=$?
  echo "start rc=$rc after $((SECONDS - s)) s"
  c status deployment q38
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
  exit $rc
  ;;
measure)
  out=$F/results/$(date -u +%Y%m%dT%H%M%S)
  mkdir -p "$out"
  common=(--model q38 --cred "$SD/identity/credentials" --endpoint 127.0.0.1:18743)
  python3 "$F/fanout.py" batch --streams 1 --rounds 1 --max-tokens 64 "${common[@]}" --out "$out/warmup" >/dev/null
  python3 "$F/fanout.py" fanout --rounds 3 --max-tokens 6144 "${common[@]}" --out "$out"
  python3 "$F/fanout.py" whole --rounds 3 --max-tokens 16384 "${common[@]}" --out "$out"
  python3 "$F/fanout.py" batch --streams 8 --rounds 3 --max-tokens 1024 "${common[@]}" --out "$out/batch8"
  python3 "$F/fanout.py" batch --streams 1 --rounds 3 --max-tokens 1024 "${common[@]}" --out "$out/batch1"
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv >"$out/nvidia-smi.csv"
  echo "RESULTS $out"
  ;;
down)
  c delete deployment q38 --stop
  pkill -TERM -f -- "--state-dir $SD start standalone"
  ;;
*) echo "usage: serve.sh up|measure|down" >&2; exit 2 ;;
esac
