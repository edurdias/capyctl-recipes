#!/usr/bin/env bash
set -euo pipefail
# session-a only, integration-owner-cleared load checks. Never install packages.
version="${1:?approved version required}"
checkpoint="${2:?checkpoint directory required}"
case "$version" in
  vllm-0.32.0|vllm-0.33.0) engine=vllm ;;
  sglang-0.5.22) engine=sglang ;;
  *) echo 'Only vLLM 0.32/0.33 or SGLang 0.5.22 are approved' >&2; exit 2 ;;
esac
residents="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)"
if [[ -n "$residents" ]]; then
  printf 'Refusing concurrent GPU work:\n%s\n' "$residents" >&2
  exit 1
fi
here="$(dirname "$(realpath "$0")")"
venv="$HOME/$version-venv"
# Calling a venv's Python alone does not put its ninja on PATH for FlashInfer JIT.
export PATH="$venv/bin:$PATH"
export PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}"
export QWEN36_LOAD_ENGINE="$engine"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export MAX_JOBS=2
exec "$venv/bin/python" -u "$here/native_load_only.py" "$engine" --checkpoint "$checkpoint"
