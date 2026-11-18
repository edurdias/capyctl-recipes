#!/usr/bin/env bash
# Host A only; one guarded GPU process at a time. No conversion or engine.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PY=$HOME/modelopt-0.47.0-venv/bin/python
mkdir -p "$HERE/results"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
if [ "$#" -eq 0 ]; then
  set -- A B C D D-channel E G H I
fi
for variant in "$@"; do
  printf '%s start %s\n' "$(date -u +%FT%TZ)" "$variant"
  /usr/bin/time -v "$PY" "$HERE/ablate.py" "$variant" > "$HERE/results/$variant.log" 2>&1
  printf '%s exited %s\n' "$(date -u +%FT%TZ)" "$variant"
done
"$PY" "$HERE/summarize.py"
