#!/usr/bin/env bash
# Measured SGLang argv and DSpark environment; no context override.
set -euo pipefail
source "$HOME/sglang-inkling-pr51926-venv/bin/inkling-env"
export PYTHONDONTWRITEBYTECODE=1
export MAX_JOBS=2
export SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL=1
export SGLANG_OPT_DSV41_DSPARK_MID_FOLD=1
export SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE=1
export SGLANG_OPT_DSV41_DSPARK_SKIP_PREFILLED_INPUTS=1
# Set this to your loopback bind value; no machine-specific address is published.
: "${INKLING_BIND:?Set INKLING_BIND to a loopback bind value}"
exec "$VIRTUAL_ENV/bin/sglang" serve \
  --model-path "$HOME/models/Inkling-2-Mini-capyctl-fp8attn" \
  --served-model-name inkling-2-mini \
  --tp 1 --ep 1 --attention-backend fa4 --dtype bfloat16 \
  --page-size 128 --mem-fraction-static 0.80 \
  --grammar-backend xgrammar --enable-multimodal \
  --reasoning-parser inkling --tool-call-parser inkling2 \
  --host "$INKLING_BIND" --port 18120 --enable-metrics \
  --speculative-algorithm DSPARK --speculative-num-draft-tokens 8
