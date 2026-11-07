#!/usr/bin/env bash
# Qwen3.8-27B: capyctl/Qwen3.8-27B-NVFP4 against nvidia/Qwen3.8-27B-NVFP4 and other published
# builds on one GB10, through CapyCTL, on SGLang 0.5.22 and vLLM 0.32.0 (with and without the
# DFlash2 drafter) and MiaAI-Lab's exllamav3 server as a custom engine, plus the accuracy
# checks against the BF16 source. Runs on the host itself, one deployment at a time; each is
# deleted before the next starts.
#
# usage: run-host.sh [variant ...]     (default: run 1, the variants in VARIANTS_ALL)
#
# Variants: sg-* SGLang, vl-* vLLM, cu-* custom engine. -nv nvidia, -full ours, -bf16 the source,
# -gt gittensor RTX5090, -qs QUASAR-QAT, -rh RedHatAI INT4, -un unsloth (paths from CKPTS/pins.json),
# cu-mia / cu-mia-df Mia's EXL3 3.5bpw without drafts / with their EXL3 DFlash2. -df adds the
# DFlash2 drafter; sg-full-df-tuned adds MiaAI-Lab's SGLang flags (bf16 DeltaNet state, two
# continuous decode steps). A published SGLang variant that fails to start is retried on vLLM.
#
# Environment (defaults in brackets):
#   WORK       run directory [~/q38-requant/run-<UTC time>]
#   CAPYCTL    CapyCTL binary [~/q38-requant/capyctl-0.1.10/capyctl-0.1.10-linux-aarch64/bin/capyctl]
#   RECIPES    this repository's checkout on the host [~/q38-requant/recipes]
#   FULL       the converted checkpoint [~/models/Qwen3.8-27B-NVFP4-Full]
#   NVIDIA     nvidia/Qwen3.8-27B-NVFP4 at 482ca0f3 [~/models/sources/huggingface/nvidia--...@482ca0f3...]
#   BF16       Qwen/Qwen3.8-27B at 1d4bf0f2 [~/models/qwen3.8-27b]
#   DRAFTER    z-lab/Qwen3.8-27B-DFlash2 at 50307d4c [~/drafters/Qwen3.8-27B-DFlash2]
#   SGLANG_VENV [~/sglang-0.5.22-venv]   VLLM_VENV [~/vllm-0.32.0-venv]
#   ROUNDS [3]  CONC [1,4,8]  SWEEP [2k,32k]  SWEEP_RUNS [2]  GSM_N [250]  GSM_MAX_TOKENS [2048]
#
# One live session at a time: refuses to start while any engine runs on the host.
set -euo pipefail
{
WORK=${WORK:-$HOME/q38-requant/run-$(date -u +%Y%m%dT%H%M%SZ)}
CAPYCTL=${CAPYCTL:-$HOME/q38-requant/capyctl-0.1.10/capyctl-0.1.10-linux-aarch64/bin/capyctl}
RECIPES=${RECIPES:-$HOME/q38-requant/recipes}
FULL=${FULL:-$HOME/models/Qwen3.8-27B-NVFP4-Full}
NVIDIA=${NVIDIA:-$HOME/models/sources/huggingface/nvidia--Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80}
BF16=${BF16:-$HOME/models/qwen3.8-27b}
DRAFTER=${DRAFTER:-$HOME/drafters/Qwen3.8-27B-DFlash2}
CKPTS=${CKPTS:-$HOME/q38-requant/ckpts}
MIA=${MIA:-$HOME/models/Mia-AiLab--Qwen3.8-27B-EXL3-3.5bpw@19441ac8}
MIA_VENV=${MIA_VENV:-$HOME/exllamav3-mia-venv}
SGLANG_VENV=${SGLANG_VENV:-$HOME/sglang-0.5.22-venv}
VLLM_VENV=${VLLM_VENV:-$HOME/vllm-0.32.0-venv}
ROUNDS=${ROUNDS:-3} CONC=${CONC:-1,4,8} SWEEP=${SWEEP:-2k,32k} SWEEP_RUNS=${SWEEP_RUNS:-2}
GSM_N=${GSM_N:-250} GSM_MAX_TOKENS=${GSM_MAX_TOKENS:-2048}
PORT=18543 MPORT=18544 ENGINE_PORTS=8300-8319
BENCH=$RECIPES/tools/capyctl-bench/capyctl_bench.py
HELPER=$RECIPES/comparisons/qwen3.8-27b-sglang-dflash2-vs-mtp-eagle-gb10/b18.py
ACC=$RECIPES/comparisons/qwen3.8-27b-sglang-dflash2-vs-mtp-eagle-gb10/accuracy
SD=$WORK/state RES=$WORK/results
export XDG_CONFIG_HOME=$WORK/config XDG_STATE_HOME=$WORK/xdg-state

VARIANTS_ALL="sg-nv sg-full sg-nv-df sg-full-df vl-nv vl-full vl-nv-df vl-full-df sg-bf16"
VARIANTS=${*:-$VARIANTS_ALL}
ACCURACY=${ACCURACY-"sg-nv sg-full sg-bf16 sg-gt sg-qs sg-rh sg-un vl-gt vl-qs vl-rh vl-un cu-mia-df"}

mkdir -p "$SD" "$RES" "$WORK/deployments" "$XDG_CONFIG_HOME" "$XDG_STATE_HOME"
say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$RES/progress.log"; }
c() { "$CAPYCTL" --state-dir "$SD" "$@"; }
memavail() { awk '/^MemAvailable:/ {printf "%.2f", $2 / 1048576}' /proc/meminfo; }

# The GPU's process list, not a command-line match: an SSH session whose command text names an
# engine would match a pgrep -f pattern (found 2026-11-03).
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then
  say "an engine is running on this host; refusing to start (one live session at a time)"
  exit 1
fi

engine_of() { case $1 in sg-*) echo sg522 ;; vl-*) echo vl032 ;; cu-mia-df) echo mia-exl3-dflash2 ;; cu-mia) echo mia-exl3-none ;; esac; }
pinned() { # repository -> its directory under CKPTS
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]["path"])' "$CKPTS/pins.json" "$1"
}
ckpt_of() {
  case $1 in
    *-full*) echo "$FULL" ;; *-nv*) echo "$NVIDIA" ;; *-bf16|*-bf16-rep) echo "$BF16" ;; cu-mia*) echo "$MIA" ;;
    *-gt) pinned gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 ;;
    *-qs) pinned QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 ;;
    *-rh) pinned RedHatAI/Qwen3.8-27B-INT4 ;;
    *-un) pinned unsloth/Qwen3.8-27B-NVFP4 ;;
    *-rx*) pinned RadixArk/Qwen3.8-27B-NVFP4 ;;
  esac
}

render() { # key -> deployment YAML
  local k=$1 name=q38-$1
  case $k in rcp-vl|rcp-sg|rcp-vl-hi|rcp-sg-hi)
    # A recipe's own deployment.yaml as it ships, with only the host's paths and profile
    # names: the checkpoint's local copy (the same bytes as the pinned Hugging Face revision,
    # checked by SHA-256) for `model: {hf: ...}`, the drafter's directory for /home/me/drafters.
    # The -hi variants raise only the concurrency cap to HI_CAP (and SGLang's memory request
    # to HI_SG_REQUEST: SGLang sizes its DeltaNet state pool, drafts included, by the cap; and
    # vLLM's KV cache to HI_VL_KV, which holds its DeltaNet state blocks) to find where
    # throughput saturates.
    local dir hi=()
    dir=$RECIPES/$(case $k in rcp-vl*) echo vllm ;; *) echo sglang ;; esac)/qwen3.8-27b-capyctl-nvfp4-gb10
    case $k in *-hi)
      hi=(-e "s|^  max_concurrent_requests: 8$|  max_concurrent_requests: ${HI_CAP:-48}|"
          -e "s|request: 65GiB|request: ${HI_SG_REQUEST:-85GiB}|"
          -e "s|memory: {kv_cache: 16GiB}|memory: {kv_cache: ${HI_VL_KV:-16GiB}}|") ;;
    esac
    sed -e "s|^name: .*|name: $name|" -e "s|^engine: vllm$|engine: vl032|" -e "s|^engine: sglang$|engine: sg522|" \
        -e "s|^model: {hf: capyctl/Qwen3.8-27B-NVFP4@[^}]*}|model: $FULL|" \
        -e "s|/home/me/drafters/Qwen3.8-27B-DFlash2|$DRAFTER|" "${hi[@]}" "$dir/deployment.yaml"
    return ;;
  esac
  printf 'schema_version: 1\nkind: deployment\nname: %s\nengine: %s\nmodel: %s\n' "$name" "$(engine_of "$k")" "$(ckpt_of "$k")"
  # The SGLang 0.5.22 environment has no memory saver (capability_missing:deep_park), and the
  # comparison measures serving, not parking: every variant restarts instead of parking.
  printf 'residency: restart_only\ntimeouts: {initialize: 1800s}\n'
  if [ "${k#cu-}" != "$k" ]; then
    # A custom engine takes resources, not engine_config.memory; the server sizes its own cache.
    printf 'devices: [{id: gpu0}]\nresources: {gpu: 40GiB, ram: 8GiB}\nengine_config:\n  context_length: 32768\n'
    return
  fi
  printf 'engine_config:\n  context_length: 32768\n  max_concurrent_requests: 8\n'
  case $k in
    sg-bf16|sg-bf16-rep)
      printf '  kv_cache_dtype: fp8_e4m3\n  memory: {request: 90GiB, kv_cache: 16GiB}\n' ;;
    sg-*-r13)
      # Shardio's r13 sets no KV cache dtype: the engine's default (bf16).
      printf '  quantization: modelopt\n  memory: {request: 65GiB, kv_cache: 16GiB}\n' ;;
    sg-nv*|sg-full*|sg-gt)
      printf '  quantization: modelopt\n  kv_cache_dtype: fp8_e4m3\n  memory: {request: 65GiB, kv_cache: 16GiB}\n' ;;
    sg-*)  # compressed-tensors builds: SGLang reads the format from the checkpoint
      printf '  kv_cache_dtype: fp8_e4m3\n  memory: {request: 65GiB, kv_cache: 16GiB}\n' ;;
    vl-*)
      printf '  kv_cache_dtype: fp8\n  language_model_only: true\n  memory: {kv_cache: 16GiB}\n' ;;
  esac
  # DFlash2 variants: <base>-df[-option...]. Options: tuned (MiaAI-Lab's SGLang flags: bf16
  # DeltaNet state, two continuous decode steps), cds3 (three continuous decode steps), compile
  # (torch.compile, SGLang), k6/k12 (draft tokens instead of 8 on SGLang, 7 on vLLM), mbf16 (vLLM's
  # DeltaNet state cache in bf16).
  case $k in
    *-r13)
      # Shardio's Qwen3.8-27B recipe r13 (bld_qwen3.8-27b-sglang-xrargf, SGLang 0.5.22): EAGLE drafts
      # from the checkpoint's MTP head, and decode rounds between prefill chunks. Its 262k context,
      # --max-mamba-cache-size 110 and LMCache are left out: this measures drafting and weights at
      # the context and concurrency of the other variants (CapyCTL sizes the state pool).
      printf '  accept_extra_args: true\n  extra_args:\n'
      printf '    - --speculative-algorithm\n    - EAGLE\n    - --speculative-num-steps\n    - "3"\n'
      printf '    - --speculative-eagle-topk\n    - "1"\n    - --speculative-num-draft-tokens\n    - "4"\n'
      printf '    - --enable-linear-replayssm-spec\n    - --prefill-decode-interval\n    - "1"\n'
      return ;;
  esac
  case $k in *-df|*-df-*) ;; *) return ;; esac
  local opts="-${k#*-df}-" draft=8 vdraft=7
  case $opts in *-k6-*) draft=6 vdraft=5 ;; *-k12-*) draft=12 vdraft=11 ;; esac
  printf '  accept_extra_args: true\n  extra_args:\n'
  case $k in
    sg-*)
      printf '    - --speculative-algorithm\n    - DFLASH\n    - --speculative-draft-model-path\n    - %s\n' "$DRAFTER"
      printf '    - --speculative-num-draft-tokens\n    - "%s"\n    - --speculative-draft-model-quantization\n    - unquant\n' "$draft"
      case $opts in *-tuned-*|*-cds3-*) printf '    - --mamba-ssm-dtype\n    - bfloat16\n' ;; esac
      case $opts in
        *-cds3-*) printf '    - --num-continuous-decode-steps\n    - "3"\n' ;;
        *-tuned-*) printf '    - --num-continuous-decode-steps\n    - "2"\n' ;;
      esac
      case $opts in *-compile-*) printf '    - --enable-torch-compile\n    - --torch-compile-max-bs\n    - "8"\n' ;; esac ;;
    vl-*)
      printf "    - --speculative-config\n    - '{\"method\":\"dflash\",\"model\":\"%s\",\"num_speculative_tokens\":%s}'\n" "$DRAFTER" "$vdraft"
      case $opts in *-mbf16-*) printf '    - --mamba-ssm-cache-dtype\n    - bfloat16\n' ;; esac ;;
  esac
}

setup() {
  say "setup: standalone CapyCTL $("$CAPYCTL" --version), work $WORK"
  c engine add "$SGLANG_VENV" --name sg522 \
    --approve-option=--speculative-draft-model-path --approve-path "$(dirname "$DRAFTER")" >>"$RES/cli.log" 2>&1
  c engine add "$VLLM_VENV" --name vl032 \
    --approve-option=--speculative-config --approve-path "$(dirname "$DRAFTER")" >>"$RES/cli.log" 2>&1
  case " $VARIANTS " in *" cu-"*)
    for d in none dflash2; do
      c engine add --custom "$MIA_VENV/bin/mia-serve" --name "mia-exl3-$d" \
        --manifest "$RECIPES/comparisons/qwen3.8-27b-nvfp4-builds-gb10/manifests/mia-exl3-$d.yaml" \
        --yes >>"$RES/cli.log" 2>&1
    done ;;
  esac
  setsid nohup "$CAPYCTL" --state-dir "$SD" start standalone --listen "127.0.0.1:$PORT" \
    --management-listen "127.0.0.1:$MPORT" --engine-ports "$ENGINE_PORTS" \
    --set host.resource_policy.memory.system.managed_limit=100GiB \
    --set host.resource_policy.memory.system.free_reserve=10GiB \
    --set host.resource_policy.memory.system.parked_limit=45GiB --json \
    >"$RES/role.log" 2>&1 </dev/null &
  echo $! >"$WORK/role.pid"
  for _ in $(seq 180); do grep -q '"ready":true' "$RES/role.log" 2>/dev/null && break; sleep 1; done
  grep -q '"ready":true' "$RES/role.log" || { say "standalone not ready in 180 s"; exit 1; }
  CRED=$(find "$SD" "$XDG_STATE_HOME" -name credentials -path '*identity*' 2>/dev/null | head -1)
  [ -n "$CRED" ] || { say "no credentials file under the state dir"; exit 1; }
  say "standalone ready; credentials $CRED"
  {
    echo "capyctl=$("$CAPYCTL" --version)"
    echo "sglang=$("$SGLANG_VENV/bin/python" -c 'import sglang; print(sglang.__version__)' 2>/dev/null)"
    echo "vllm=$("$VLLM_VENV/bin/python" -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
    "$SGLANG_VENV/bin/python" -c 'import torch; print("torch_sglang=" + torch.__version__)'
    "$VLLM_VENV/bin/python" -c 'import torch; print("torch_vllm=" + torch.__version__)'
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed 's/^/gpu=/'
    uname -r | sed 's/^/kernel=/'
    [ -x "$MIA_VENV/bin/python" ] &&
      "$MIA_VENV/bin/python" -c 'import importlib.metadata as m; print("exllamav3=" + m.version("exllamav3"))'
    for f in "$FULL" "$NVIDIA" "$BF16" "$DRAFTER" "$MIA" "$CKPTS"/*@*; do
      [ -d "$f" ] || continue
      # The index (or the one shard's size) names the checkpoint; its revision pins the bytes.
      printf 'weights %s index %s bytes %s\n' "$(basename "$f")" \
        "$(sha256sum "$f"/model.safetensors.index.json 2>/dev/null | cut -c1-16)" \
        "$(du -sbL --apparent-size "$f"/*.safetensors | awk '{s += $1} END {print s}')"
    done
  } >"$RES/versions.txt" 2>&1
}

# Delete with --stop until the engines are confirmed gone (pending answers repeat it).
teardown() {
  local id
  id=$(python3 -c 'import os,time; t=int(time.time()*1000); a="0123456789ABCDEFGHJKMNPQRSTVWXYZ"; r=int.from_bytes(os.urandom(10),"big"); s=""
for _ in range(10): s=a[t%32]+s; t//=32
u=""
for _ in range(16): u=a[r%32]+u; r//=32
print(s+u)')
  for _ in $(seq 30); do
    if c delete deployment "$1" --stop --request-id "$id" >>"$RES/cli.log" 2>&1; then
      grep -q pending <(tail -2 "$RES/cli.log") || return 0
    fi
    sleep 10
  done
  say "$1: delete did not complete; see cli.log"
}

variant() {
  local k=$1 n=q38-$1 before after s ready
  render "$k" >"$WORK/deployments/$n.yaml"
  say "== $k: deploy $n"
  if ! c deploy model --file "$WORK/deployments/$n.yaml" >>"$RES/cli.log" 2>&1; then
    say "$k: deploy refused; see cli.log"; tail -3 "$RES/cli.log" | tee -a "$RES/progress.log"; return 0
  fi
  before=$(memavail); s=$(date +%s)
  if ! c start deployment "$n" --wait >"$RES/start-$k.txt" 2>&1; then
    say "$k: start failed"; tail -5 "$RES/start-$k.txt" | tee -a "$RES/progress.log"
    c status deployment "$n" --json >"$RES/status-$k.json" 2>&1 || true
    teardown "$n"
    return 2
  fi
  ready=$(( $(date +%s) - s )); sleep 10; after=$(memavail)
  c status deployment "$n" --json >"$RES/status-$k.json" 2>&1 || true
  printf '%s\t%s\t%s\n' "$k" "$ready" "$(awk -v a="$before" -v b="$after" 'BEGIN {printf "%.1f", a - b}')" >>"$RES/startup.tsv"
  say "$k: ready in ${ready} s, MemAvailable down $(awk -v a="$before" -v b="$after" 'BEGIN {printf "%.1f", a - b}') GiB"
  case $k in rcp-*-hi)
    # High concurrency: stream counts only (CONC, ROUNDS), no context sweep, no warm start.
    python3 "$BENCH" run --endpoint "http://127.0.0.1:$PORT/v1" --api-key-file "$CRED" --model "$n" --label "$k" \
      --concurrency "$CONC" --rounds "$ROUNDS" \
      --memory-cmd "free -b | awk '/^Mem:/ {print \$3 / 2^30}'" \
      --meta capyctl="$("$CAPYCTL" --version)" --meta variant="$k" --meta gpu=GB10 \
      --out "$RES/bench-$k.json" >>"$RES/bench.log" 2>&1 || say "$k: bench exited non-zero; see bench.log"
    local hid
    hid=$(grep -oE '01[0-9A-HJKMNP-TV-Z]{24}' "$RES/status-$k.json" 2>/dev/null | head -1 || true)
    [ -n "$hid" ] && cat "$SD/logs/$hid"/*.log >"$RES/engine-$k.log" 2>/dev/null || true
    teardown "$n"
    say "$k: done"
    return 0 ;;
  esac
  if [ "${k#rcp-}" != "$k" ]; then
    # A recipe: the full capyctl-bench sweeps, then a warm start (`start` after `stop`).
    python3 "$BENCH" run --endpoint "http://127.0.0.1:$PORT/v1" --api-key-file "$CRED" --model "$n" --label "$k" \
      --concurrency 1-8 --rounds 5 --context-sweep 0.5k,1k,2k,4k,8k,16k,32k --runs 3 \
      --memory-cmd "free -b | awk '/^Mem:/ {print \$3 / 2^30}'" \
      --meta capyctl="$("$CAPYCTL" --version)" --meta variant="$k" --meta gpu=GB10 \
      --out "$RES/bench-$k.json" >>"$RES/bench.log" 2>&1 || say "$k: bench exited non-zero; see bench.log"
    c stop deployment "$n" >>"$RES/cli.log" 2>&1
    for _ in $(seq 60); do
      c status deployment "$n" --json 2>/dev/null | grep -q '"observed_state": *"stopped"' && break
      sleep 5
    done
    s=$(date +%s)
    if c start deployment "$n" --wait >"$RES/start-$k-warm.txt" 2>&1; then
      ready=$(( $(date +%s) - s ))
      printf '%s\t%s\t-\n' "$k-warm" "$ready" >>"$RES/startup.tsv"
      say "$k: warm start ready in ${ready} s"
    else
      say "$k: warm start failed; see start-$k-warm.txt"
    fi
    c status deployment "$n" --json >"$RES/status-$k-warm.json" 2>&1 || true
    local wid
    wid=$(grep -oE '01[0-9A-HJKMNP-TV-Z]{24}' "$RES/status-$k.json" 2>/dev/null | head -1 || true)
    [ -n "$wid" ] && cat "$SD/logs/$wid"/*.log >"$RES/engine-$k.log" 2>/dev/null || true
    teardown "$n"
    say "$k: done"
    return 0
  fi
  if [ "${k%-rep}" != "$k" ]; then
    # The BF16 source again, agreement only: how much greedy decoding moves between two
    # launches of the same weights (the noise floor of the agreement column).
    say "$k: agreement repeat (200 prompts)"
    python3 "$HELPER" agree --cred "$CRED" --endpoint "http://127.0.0.1:$PORT" --model "$n" \
      --prompts "$ACC/agreement-prompts.json" --max-tokens 256 --concurrency 8 --out "$RES/agree-$k.json" >>"$RES/cli.log" 2>&1
    teardown "$n"
    say "$k: done"
    return 0
  fi
  if [ "${k#cu-}" != "$k" ]; then
    # The exllamav3 server streams without a token count, which capyctl-bench needs; mia_speed.py
    # counts with the checkpoint's tokenizer, so it runs with that environment's Python.
    "$MIA_VENV/bin/python" "$RECIPES/comparisons/qwen3.8-27b-nvfp4-builds-gb10/mia_speed.py" \
      --endpoint "http://127.0.0.1:$PORT/v1" --cred "$CRED" --model "$n" \
      --prompts "$RECIPES/tools/capyctl-bench/prompts.json" --tokenizer "$MIA/tokenizer.json" \
      --concurrency "$CONC" --rounds "$ROUNDS" \
      --out "$RES/speed-$k.json" >>"$RES/bench.log" 2>&1 || say "$k: mia_speed exited non-zero; see bench.log"
  else
    python3 "$BENCH" run --endpoint "http://127.0.0.1:$PORT/v1" --api-key-file "$CRED" --model "$n" --label "$k" \
      --concurrency "$CONC" --rounds "$ROUNDS" --context-sweep "$SWEEP" --runs "$SWEEP_RUNS" \
      --memory-cmd "free -b | awk '/^Mem:/ {print \$3 / 2^30}'" \
      --meta capyctl="$("$CAPYCTL" --version)" --meta variant="$k" --meta gpu=GB10 \
      --out "$RES/bench-$k.json" >>"$RES/bench.log" 2>&1 || say "$k: bench exited non-zero; see bench.log"
  fi
  case " $ACCURACY " in *" $k "*)
    say "$k: accuracy (200-prompt agreement, GSM8K $GSM_N, 30 tool calls)"
    python3 "$HELPER" agree --cred "$CRED" --endpoint "http://127.0.0.1:$PORT" --model "$n" \
      --prompts "$ACC/agreement-prompts.json" --max-tokens 256 --concurrency 8 --out "$RES/agree-$k.json" >>"$RES/cli.log" 2>&1
    python3 "$HELPER" gsm8k --cred "$CRED" --endpoint "http://127.0.0.1:$PORT" --model "$n" \
      --data "$ACC/gsm8k-test-250.jsonl" --limit "$GSM_N" --max-tokens "$GSM_MAX_TOKENS" --concurrency 8 \
      --out "$RES/gsm8k-$k.json" >>"$RES/cli.log" 2>&1
    python3 "$HELPER" tools --cred "$CRED" --endpoint "http://127.0.0.1:$PORT" --model "$n" \
      --cases "$ACC/tool-cases.json" --max-tokens 2048 --concurrency 8 --out "$RES/tools-$k.json" >>"$RES/cli.log" 2>&1
    ;;
  esac
  # Launch logs live under the deployment's ID (<state>/logs/<deployment id>/<launch>.log).
  local id
  id=$(grep -oE '01[0-9A-HJKMNP-TV-Z]{24}' "$RES/status-$k.json" 2>/dev/null | head -1 || true)
  [ -n "$id" ] && cat "$SD/logs/$id"/*.log >"$RES/engine-$k.log" 2>/dev/null || true
  teardown "$n"
  say "$k: done"
}

# shellcheck disable=SC2329 # invoked by the EXIT trap
cleanup() {
  [ -f "$WORK/role.pid" ] && kill "$(cat "$WORK/role.pid")" 2>/dev/null || true
  say "role stopped"
}
trap cleanup EXIT
trap 'exit 130' HUP INT TERM

setup
for k in $VARIANTS; do
  if ! variant "$k"; then
    # A published build SGLang cannot start: the same checkpoint on vLLM.
    case $k in sg-gt|sg-qs|sg-rh|sg-un) variant "vl-${k#sg-}" || true ;; esac
  fi
done
say "all variants done"
exit 0
}
