#!/usr/bin/env bash
# Run on host A, using only the existing environment. Every Python GPU job
# enters convert/memory_guard.py's 8-GiB MemAvailable guard.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PY=$HOME/modelopt-0.47.0-venv/bin/python
mkdir -p "$HERE/results"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8
if [ "$#" -eq 0 ]; then
  set -- bf16 r1 r2 nvidia redhat unsloth r3 r3e
fi
for build in "$@"; do
  printf '%s start %s\n' "$(date -u +%FT%TZ)" "$build"
  /usr/bin/time -v "$PY" "$HERE/next_token_agree.py" "$build" \
    > "$HERE/results/$build.log" 2>&1
  printf '%s exited %s\n' "$(date -u +%FT%TZ)" "$build"
done
"$PY" "$HERE/summarize.py"
