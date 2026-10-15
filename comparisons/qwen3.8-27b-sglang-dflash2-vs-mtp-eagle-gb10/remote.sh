#!/usr/bin/env bash
# Host side of run.sh: the DFlash2 and MTP/EAGLE recipes of Qwen3.8-27B on SGLang
# 0.5.21, one GB10 host, through one CapyCTL standalone role with its own state
# and config directories, ports and kernel caches.
#
#   remote.sh run     every phase, in $WORK, from the settings in $WORK/b18.env
#                     (MODE=variants there: only the MTP/EAGLE residency and KV dtype variants)
#   remote.sh plan    print what `run` does, step by step, running nothing (any machine)
#   remote.sh down    delete what `run` created and stop its role (also run on exit)
#   remote.sh purge   remove $WORK (after run.sh copied the results back)
#
# An existing CapyCTL install on the host is not touched, and only the role this
# script started is ever signalled. Engine keys are never read; the inference
# key and admin token are read from the role's credentials file by b18.py and
# capyctl-bench only, never printed.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
cmd=${1:-}
WORK=${WORK:-$HERE}
# shellcheck disable=SC1091
. "$WORK/b18.env"
: "${PORT:?b18.env sets PORT}" "${MPORT:?}" "${ENGINE_PORTS:?}" "${ROUNDS:?}" "${PASSES:?}" "${RUNS:?}"
: "${CONC:?}" "${SWEEP:?}" "${STALL_TOKENS:?}" "${STALL_STREAMS:?}" "${STALL_TRIALS:?}" "${STALL_MAX_TOKENS:?}"
: "${B_SOURCE:?}" "${B_COMMIT:?}" "${MANAGED_LIMIT:?}" "${FREE_RESERVE:?}" "${PARKED_LIMIT:?}" "${PHASES:?}" "${CAPYCTL_COMMIT:?}"
MODE=${MODE:-full}
PLAN=0
[ "$cmd" = plan ] && PLAN=1
# A short state directory: SGLang's saver observation socket lives under it
# and a Unix socket path holds at most 107 bytes.
SD=$WORK/s
RES=$WORK/res
CRED=$SD/identity/credentials
CAPYCTL=${CAPYCTL_BIN:-capyctl}
export XDG_CONFIG_HOME=$WORK/config
export CAPYCTL_BENCH_DIR=$WORK/capyctl-bench
VENV=$HOME/sglang-0.5.21-venv
DRAFTERS=$HOME/drafters
DRAFTER=$DRAFTERS/Qwen3.8-27B-DFlash2
B_REPO=RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead
B_HUB=$HOME/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4-BF16-LMHead
B_SNAPSHOT=$B_HUB/snapshots/$B_COMMIT
B_WEIGHT_BYTES=23749332688
PIDFILE=$WORK/role.pid
EP=http://127.0.0.1:$PORT
HELPER=(python3 "$WORK/b18.py")
# Memory in use on the machine (unified memory), GiB, for capyctl-bench.
MEMUSED="awk '/^MemTotal:/ {t=\$2} /^MemAvailable:/ {a=\$2} END {print (t-a)/1048576}' /proc/meminfo"
# Engine processes of this run: argv names the SGLang venv.
ENGINE_RE="$VENV/bin/"
# Any engine on the host (the lab rule: one live session at a time).
ANY_ENGINE='sglang[.]launch_server|sglang_entr[y]|vllm_entr[y]|vllm[ ]serve|Engine[C]ore|sglang::schedule[r]|tensorfol[d]'

# Deployment keys: a = DFlash2, actx = DFlash2 with a 262144-token context,
# b = MTP/EAGLE, bn = MTP/EAGLE without --enable-mixed-chunk; in the variants
# mode v79 = MTP/EAGLE as committed (79 GiB, the refusal check), and with the
# request raised to 84 GiB: vdeep (deep), vrestart = residency restart_only,
# vautokv = restart_only without kv_cache_dtype stated.
VARIANT_KEYS="vdeep vrestart vautokv"
if [ "$MODE" = variants ]; then DEPLOY_KEYS="v79 $VARIANT_KEYS"; else DEPLOY_KEYS="a actx b bn"; fi
side_of() { case $1 in a|actx) echo a ;; *) echo b ;; esac; }
prof() { if [ "$(side_of "$1")" = a ]; then echo b18a; else echo b18b; fi; }
dep() {
  case $1 in
    a) echo qwen38-dflash2 ;; actx) echo qwen38-dflash2-ctx ;; b) echo qwen38-mtp-eagle ;;
    bn) echo qwen38-mtp-eagle-nomix ;; vdeep) echo qwen38-v-deep ;; vrestart) echo qwen38-v-restart ;;
    vautokv) echo qwen38-v-autokv ;; v79) echo qwen38-v-79gib ;;
  esac
}
var_name() { case $1 in vdeep) echo deep ;; vrestart) echo restart ;; vautokv) echo restart-autokv ;; v79) echo refusal79 ;; esac; }
yaml_of() {
  case $1 in
    a) echo deployment-dflash2.yaml ;; actx) echo deployment-dflash2-ctx.yaml ;;
    bn) echo deployment-mtp-eagle-nomix.yaml ;; *) echo deployment-mtp-eagle.yaml ;;
  esac
}
label() {
  case $1 in
    a) echo "DFlash2 (SGLang 0.5.21)" ;;
    v*) echo "MTP/EAGLE $(var_name "$1") (SGLang 0.5.21)" ;;
    *) echo "MTP/EAGLE (SGLang 0.5.21)" ;;
  esac
}
model_ref() {
  if [ "$1" = a ]; then echo nvidia/Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80
  else echo "$B_REPO@$B_COMMIT"; fi
}
drafter_ref() {
  if [ "$1" = a ]; then echo z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4
  else echo "none (the checkpoint's MTP layer)"; fi
}
has_phase() { case ",$PHASES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

# ---------------------------------------------------------------------------
# Effects. In plan mode every effect is printed instead (fd 3 is the terminal,
# so a step whose output is redirected still shows), waits are skipped and
# measured values read as placeholders.
# ---------------------------------------------------------------------------
exec 3>&1
mkdir -p "$RES"
show() { printf '    %s\n' "$*" | sed -e "s#$HOME#~#g" >&3; }
say() {
  if [ "$PLAN" = 1 ]; then printf '\n  %s\n' "$*" | sed -e "s#$HOME#~#g" >&3; return 0; fi
  printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$RES/progress.log"
}
run() { # run CMD...: execute it, or print it
  if [ "$PLAN" = 1 ]; then show "\$ $(printf '%q ' "$@")"; return 0; fi
  "$@"
}
c() { run "$CAPYCTL" --state-dir "$SD" "$@"; }
helper() { run "${HELPER[@]}" "$@"; }
rec() { # step result detail
  printf '%s\t%s\t%s\n' "$1" "$2" "$3" >>"$RES/lifecycle.tsv"
  [ "$PLAN" = 1 ] || say "[$2] $1: $3"
}
fig() { # side key value: one figure for the README
  if [ "$PLAN" = 1 ]; then show "record figure $1 $2"; return 0; fi
  printf '%s\t%s\t%s\n' "$1" "$2" "$3" >>"$RES/figures.tsv"
}
now() { date +%s.%N; }
elapsed() { if [ "$PLAN" = 1 ]; then echo 0; else printf '%.1f' "$(echo "$(now) - $1" | bc)"; fi; }
memavail() { # GiB
  if [ "$PLAN" = 1 ]; then echo 0; return; fi
  awk '/^MemAvailable:/ {printf "%.2f", $2/1048576}' /proc/meminfo
}
sub() { echo "$1 - $2" | bc; }
pause() { [ "$PLAN" = 1 ] || sleep "$1"; }
waitgone() {
  [ "$PLAN" = 1 ] && { show "wait until no process runs from ~/sglang-0.5.21-venv/bin/ (600 s)"; return 0; }
  local _
  for _ in $(seq 600); do pgrep -f -- "$ENGINE_RE" >/dev/null || return 0; sleep 1; done
  return 1
}

# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

preflight() {
  say "preflight: venv, drafter, MTP/EAGLE weights, no other engine, free ports"
  if [ "$PLAN" = 1 ]; then
    show "check ~/sglang-0.5.21-venv reports sglang 0.5.21$([ "$MODE" = variants ] || echo "; ~/drafters/Qwen3.8-27B-DFlash2/config.json exists")"
    [ "$B_SOURCE" = cache ] && show "check $B_SNAPSHOT holds 3 safetensors files, $B_WEIGHT_BYTES bytes, and its hub directory is not group/other-writable"
    show "check no engine process runs (pgrep '$ANY_ENGINE') and ports $PORT, $MPORT are free"
    return 0
  fi
  local got bad
  got=$("$VENV/bin/python" -c 'import sglang; print(sglang.__version__)' 2>/dev/null)
  if [ "$got" = 0.5.21 ]; then rec "venv" pass "sglang-0.5.21-venv reports $got"
  else rec "venv" fail "sglang-0.5.21-venv reports '${got}'"; return 1; fi
  if [ "$MODE" = variants ]; then :
  elif [ -f "$DRAFTER/config.json" ]; then rec "drafter" pass "drafters/Qwen3.8-27B-DFlash2 in the home directory"
  else rec "drafter" fail "drafters/Qwen3.8-27B-DFlash2 missing from the home directory"; return 1; fi
  if [ "$B_SOURCE" = cache ]; then
    got=$(find -L "$B_SNAPSHOT" -maxdepth 1 -name '*.safetensors' -printf '%s\n' 2>/dev/null | awk '{n++; s+=$1} END {print n+0, s+0}')
    if [ "$got" = "3 $B_WEIGHT_BYTES" ]; then rec "MTP/EAGLE weights" pass "HF cache snapshot $B_COMMIT complete"
    else rec "MTP/EAGLE weights" fail "HF cache snapshot: files and bytes '$got', expected '3 $B_WEIGHT_BYTES' (run.sh --b-source hf downloads them)"; return 1; fi
    bad=$(find "$(dirname "$B_HUB")" "$B_HUB" -maxdepth 0 -perm /022 2>/dev/null)
    if [ -n "$bad" ]; then rec "MTP/EAGLE weights" fail "the hub directory is group/other-writable; CapyCTL refuses it (invalid_root)"; return 1; fi
  fi
  if pgrep -af -- "$ANY_ENGINE" >/dev/null; then
    rec "no other engine" fail "an engine process is running on the host; one live session at a time"; return 1
  fi
  for p in "$PORT" "$MPORT"; do
    if ss -ltnH "sport = :$p" 2>/dev/null | grep -q .; then rec "ports" fail "port $p is in use"; return 1; fi
  done
  rec "preflight" pass "venv, drafter, weights, no engine, ports $PORT/$MPORT free"
}

record_versions() {
  say "record versions"
  if [ "$PLAN" = 1 ]; then show "write versions.txt: capyctl, sglang, torch, flashinfer, sglang-kernel, transformers, torch-memory-saver, driver, CUDA, kernel"; return 0; fi
  {
    echo "capyctl_version=$("$CAPYCTL" --version 2>/dev/null | head -1)"
    echo "capyctl_commit=$CAPYCTL_COMMIT"
    "$VENV/bin/python" - <<'PY' 2>/dev/null
import importlib.metadata as m, platform
for key, dist in (("sglang", "sglang"), ("torch", "torch"), ("flashinfer", "flashinfer-python"),
                  ("sglang_kernel", "sglang-kernel"), ("transformers", "transformers"),
                  ("torch_memory_saver", "torch_memory_saver"), ("triton", "triton")):
    try:
        print(f"{key}={m.version(dist)}")
    except m.PackageNotFoundError:
        print(f"{key}=not installed")
print(f"python={platform.python_version()}")
PY
    echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/^NVIDIA //')"
    echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
    echo "cuda=$(/usr/local/cuda/bin/nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
    echo "kernel=$(uname -r)"
    echo "mem_total_gib=$(awk '/^MemTotal:/ {printf "%.1f", $2/1048576}' /proc/meminfo)"
    echo "cpus=$(nproc)"
    echo "bench_python=$(python3 -c 'import platform; print(platform.python_version())')"
  } >"$RES/versions.txt"
}

# The profile's own kernel caches: SGLang, FlashInfer, Triton, TVM-FFI,
# Inductor, CUDA and DeepGEMM, all under one empty directory per recipe.
cache_env() { # dir -> CACHE_ARGS
  local d=$1
  CACHE_ARGS=(--env "SGLANG_CACHE_DIR=$d" --env "SGLANG_JIT_CACHE_DIR=$d/jit"
    --env "SGLANG_DG_CACHE_DIR=$d/deep_gemm" --env "FLASHINFER_WORKSPACE_BASE=$d"
    --env "TRITON_CACHE_DIR=$d/triton" --env "TORCHINDUCTOR_CACHE_DIR=$d/inductor"
    --env "CUDA_CACHE_PATH=$d/nv" --env "TVM_FFI_CACHE_DIR=$d/tvm-ffi")
}

render() { # key: the committed deployment file with this host's profile, drafter path, weights and variant
  local s=(-e "s#^name: .*#name: $(dep "$1")#" -e "s#^engine: .*#engine: $(prof "$1")#"
    -e "s#/home/me/drafters#$DRAFTERS#")
  if [ "$(side_of "$1")" = b ] && [ "$B_SOURCE" = cache ]; then s+=(-e "s#^model: .*#model: $B_SNAPSHOT#"); fi
  case $1 in vrestart|vautokv) s+=(-e '/^engine: /a residency: restart_only') ;; esac
  [ "$1" = vautokv ] && s+=(-e '/^  kv_cache_dtype: /d')
  # CapyCTL now refuses 79 GiB for this recipe and names 84 GiB; v79 keeps the
  # committed request to record that refusal.
  case $1 in vdeep|vrestart|vautokv) s+=(-e 's/request: 79GiB/request: 84GiB/') ;; esac
  sed "${s[@]}" "$WORK/cmp/$(yaml_of "$1")"
}
variant_note() {
  case $1 in
    v79) echo "; as committed, 79 GiB, expected to be refused at start" ;;
    vdeep) echo "; request 84GiB, residency deep (CapyCTL's default)" ;;
    vrestart) echo "; request 84GiB, residency: restart_only" ;;
    vautokv) echo "; request 84GiB, residency: restart_only, without kv_cache_dtype" ;;
  esac
}

setup() {
  say "setup: profiles with empty kernel caches, the standalone role, deployments $DEPLOY_KEYS"
  local side opts k n sides="a b"
  [ "$MODE" = variants ] && sides=b
  run mkdir -p "$WORK/jit-a" "$WORK/jit-b" "$SD" "$XDG_CONFIG_HOME" "$WORK/deployments"
  for side in $sides; do
    cache_env "$WORK/jit-$side"
    opts=()
    [ "$side" = a ] && opts=(--approve-option=--speculative-draft-model-path --approve-path "$DRAFTERS")
    if c engine add "$VENV" --name "$(prof "$side")" "${opts[@]}" "${CACHE_ARGS[@]}" >>"$RES/cli.log" 2>&1; then
      rec "engine add $(prof "$side")" pass "kernel caches in jit-$side"
    else rec "engine add $(prof "$side")" fail "see cli.log"; return 1; fi
  done
  if [ "$PLAN" = 1 ]; then
    show "\$ setsid nohup capyctl --state-dir $SD start standalone --listen 127.0.0.1:$PORT --management-listen 127.0.0.1:$MPORT --engine-ports $ENGINE_PORTS --set host.resource_policy.memory.system.managed_limit=$MANAGED_LIMIT --set host.resource_policy.memory.system.free_reserve=$FREE_RESERVE --set host.resource_policy.memory.system.parked_limit=$PARKED_LIMIT --json >res/role.log &"
    show "wait for '\"ready\":true' in role.log (180 s)"
  else
    setsid nohup "$CAPYCTL" --state-dir "$SD" start standalone --listen "127.0.0.1:$PORT" \
      --management-listen "127.0.0.1:$MPORT" --engine-ports "$ENGINE_PORTS" \
      --set "host.resource_policy.memory.system.managed_limit=$MANAGED_LIMIT" \
      --set "host.resource_policy.memory.system.free_reserve=$FREE_RESERVE" \
      --set "host.resource_policy.memory.system.parked_limit=$PARKED_LIMIT" --json \
      >"$RES/role.log" 2>&1 </dev/null &
    echo $! >"$PIDFILE"
    local _
    for _ in $(seq 180); do grep -q '"ready":true' "$RES/role.log" 2>/dev/null && break; sleep 1; done
    if ! grep -q '"ready":true' "$RES/role.log"; then rec "standalone ready" fail "no ready line in 180 s"; return 1; fi
  fi
  rec "standalone ready" pass "own state dir, 127.0.0.1:$PORT, management :$MPORT, managed $MANAGED_LIMIT, parked $PARKED_LIMIT"
  c list hosts >"$RES/list-hosts.txt" 2>&1
  for k in $DEPLOY_KEYS; do
    n=$(dep "$k")
    if [ "$PLAN" = 0 ]; then render "$k" >"$WORK/deployments/$n.yaml"
    elif [ "$(side_of "$k")" = a ]; then show "render $(yaml_of "$k") -> deployments/$n.yaml (engine $(prof "$k"), drafter under ~/drafters)"
    else show "render $(yaml_of "$k") -> deployments/$n.yaml (engine $(prof "$k"), model $([ "$B_SOURCE" = cache ] && echo "$B_SNAPSHOT" || echo "hf: $B_REPO@$B_COMMIT")$(variant_note "$k"))"; fi
    if [ "$k" = v79 ] && [ "$PLAN" = 0 ]; then
      # Refused already at deploy is also the expected outcome: record it there.
      "$CAPYCTL" --state-dir "$SD" deploy model --file "$WORK/deployments/$n.yaml" >"$RES/refusal-79-deploy.txt" 2>&1
      V79_DEPLOY_RC=$?
      if [ "$V79_DEPLOY_RC" != 0 ]; then
        rec "deploy $n" info "refused at deploy (exit $V79_DEPLOY_RC); see refusal-79-deploy.txt"
        continue
      fi
      echo "$n" >>"$WORK/deployments.txt"
      rec "deploy $n" pass "accepted; the refusal is expected at start"
    elif c deploy model --file "$WORK/deployments/$n.yaml" >>"$RES/cli.log" 2>&1; then
      echo "$n" >>"$WORK/deployments.txt"
      rec "deploy $n" pass "accepted"
    else
      rec "deploy $n" fail "refused; the reason is in cli.log"
      [ "$PLAN" = 1 ] || tail -5 "$RES/cli.log" | sed 's/^/    /' | tee -a "$RES/progress.log"
      return 1
    fi
  done
}

START_S="" LAUNCH_LOG="" READY_GIB="" BEFORE_GIB=""
startd() { # key label: start and wait; START_S, READY_GIB, LAUNCH_LOG
  local n s after
  n=$(dep "$1"); BEFORE_GIB=$(memavail); s=$(now)
  if c start deployment "$n" --wait >>"$RES/cli.log" 2>&1; then
    START_S=$(elapsed "$s")
    pause 10
    after=$(memavail)
    READY_GIB=$(sub "$BEFORE_GIB" "$after")
    if [ "$PLAN" = 1 ]; then LAUNCH_LOG='<newest state/logs/<deployment>/<launch>.log>'
    else LAUNCH_LOG=$(find "$SD/logs" -mindepth 2 -maxdepth 2 -name '*.log' -printf '%T@ %p\n' 2>/dev/null |
      sort -n | tail -1 | cut -d' ' -f2-); fi
    printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$START_S" "$READY_GIB" >>"$RES/startup.tsv"
    [ "$PLAN" = 1 ] || say "started $n ($2) in $START_S s; MemAvailable down $READY_GIB GiB"
    return 0
  fi
  say "start $n ($2) failed; see cli.log"
  c status deployment "$n" >>"$RES/cli.log" 2>&1
  return 1
}
stopd() { c stop deployment "$(dep "$1")" >>"$RES/cli.log" 2>&1; waitgone; pause 5; }

facts() { # label
  [ -n "$LAUNCH_LOG" ] || return 0
  helper facts "$LAUNCH_LOG" --out "$RES/facts-$1.json" >>"$RES/cli.log" 2>&1
}

bench() { # side out; rest: run options
  local side=$1 out=$2; shift 2
  run python3 "$CAPYCTL_BENCH_DIR/capyctl_bench.py" run --endpoint "$EP/v1" --api-key-file "$CRED" \
    --label "$(label "$side")" --memory-cmd "$MEMUSED" --meta gpu="${GPU:-GB10}" --meta engine=sglang \
    --meta engine_version=0.5.21 --meta capyctl="$CAPYCTL_COMMIT" --meta model="$(model_ref "$side")" \
    --meta drafter="$(drafter_ref "$side")" --out "$out" "$@" >>"$RES/bench.log" 2>&1
}

cold() { # side: the first start with empty kernel caches
  local k=$1 jobs
  say "cold start $(label "$k"): empty kernel caches, CapyCTL's MAX_JOBS"
  if startd "$k" "cold-$k"; then
    rec "cold start $k" pass "$START_S s"
    fig "$k" cold_s "$START_S"
    fig "$k" ready_gib "$READY_GIB"
    if [ "$PLAN" = 1 ]; then show "MAX_JOBS: the last 'engine build limits: MAX_JOBS=N (...)' line of role.log"; jobs=N
    else jobs=$(grep -o 'MAX_JOBS=[0-9]*[^"]*' "$RES/role.log" | tail -1); fi
    rec "MAX_JOBS $k" info "${jobs:-not logged}"
    fig "$k" max_jobs "$(echo "$jobs" | sed -n 's/^MAX_JOBS=\([0-9]*\).*/\1/p')"
    facts "cold-$k"
    c status deployment "$(dep "$k")" --json >"$RES/status-cold-$k.json" 2>>"$RES/cli.log"
    if [ "$PLAN" = 1 ]; then show "du -sh jit-$k"
    else fig "$k" jit_cache "$(du -sh "$WORK/jit-$k" 2>/dev/null | cut -f1)"; fi
    stopd "$k"
  else rec "cold start $k" fail "see cli.log"; fi
}

state_of() { # deployment -> observed_state
  [ "$PLAN" = 1 ] && { echo parked; return; }
  c status deployment "$1" --json >"$RES/state.json" 2>/dev/null
  "${HELPER[@]}" get "$RES/state.json" observed_state
}

life() { # side: warm start, engine figures, load read, checks, one park and wake, stop
  local k=$1 n r s base_gib parked_gib st
  n=$(dep "$k")
  say "lifecycle $(label "$k"): warm start, engine figures, checks, park and wake"
  startd "$k" "warm-life-$k" || { rec "warm start $k" fail "see cli.log"; return; }
  base_gib=$BEFORE_GIB
  rec "warm start $k" pass "$START_S s"
  fig "$k" warm_s "$START_S"
  fig "$k" ready_gib "$READY_GIB"
  facts "life-$k"
  c status deployment "$n" --json >"$RES/status-life-$k.json" 2>>"$RES/cli.log"
  c inspect deployment "$n" --json >"$RES/inspect-$k.json" 2>>"$RES/cli.log"
  if [ "$PLAN" = 1 ]; then show "Ready charge: resources.ready.allocations[0].bytes of inspect-$k.json"
  else fig "$k" ready_charge "$("${HELPER[@]}" get "$RES/inspect-$k.json" 'resources.ready.allocations[0].bytes')"; fi
  # The load read while a request runs: the engine's own running limit.
  if [ "$PLAN" = 1 ]; then
    helper reasoning --cred "$CRED" --endpoint "$EP" --model "$n" --out "$RES/reasoning-$k.json"
    show "(in the background) and 3 s later:"
    helper load --cred "$CRED" --mport "$MPORT" --name "$n" --out "$RES/load-$k.json"
  else
    "${HELPER[@]}" reasoning --cred "$CRED" --endpoint "$EP" --model "$n" --out "$RES/reasoning-$k.json" >>"$RES/cli.log" 2>&1 &
    r=$!
    sleep 3
    "${HELPER[@]}" load --cred "$CRED" --mport "$MPORT" --name "$n" --out "$RES/load-$k.json" >>"$RES/cli.log" 2>&1
    wait "$r"
    fig "$k" load_max_running "$("${HELPER[@]}" get "$RES/load-$k.json" 'instances[0].max_running.count') ($("${HELPER[@]}" get "$RES/load-$k.json" 'instances[0].max_running.source'))"
  fi
  helper toolcall --cred "$CRED" --endpoint "$EP" --model "$n" --out "$RES/toolcall-$k.json" >>"$RES/cli.log" 2>&1
  for check in reasoning toolcall; do
    if [ "$PLAN" = 1 ]; then rec "$check $k" plan ""; continue; fi
    r=$("${HELPER[@]}" get "$RES/$check-$k.json" summary.pass)
    rec "$check $k" "$([ "$r" = True ] && echo pass || echo fail)" "$("${HELPER[@]}" get "$RES/$check-$k.json" summary.detail)"
  done
  # One park and wake; the parked charge is CapyCTL's, the footprint MemAvailable's.
  s=$(now)
  if c park deployment "$n" >>"$RES/cli.log" 2>&1; then
    if [ "$PLAN" = 1 ]; then show "poll status until observed_state is parked (300 s)"
    else
      for _ in $(seq 300); do st=$(state_of "$n"); [ "$st" = parked ] && break; sleep 1; done
    fi
    fig "$k" park_s "$(elapsed "$s")"
    pause 10
    parked_gib=$(sub "$base_gib" "$(memavail)")
    fig "$k" parked_gib "$parked_gib"
    c status deployment "$n" --json >"$RES/status-parked-$k.json" 2>>"$RES/cli.log"
    if [ "$PLAN" = 1 ]; then show "parked charge: parked.bytes of status-parked-$k.json"
    else
      fig "$k" parked_charge "$("${HELPER[@]}" get "$RES/status-parked-$k.json" parked.bytes)"
      rec "park $k" "$([ "${st:-}" = parked ] && echo pass || echo fail)" "state ${st:-unknown}, MemAvailable $parked_gib GiB below idle"
    fi
    if [ "$PLAN" = 1 ]; then helper chat --cred "$CRED" --endpoint "$EP" --model "$n"
    else
      r=$("${HELPER[@]}" chat --cred "$CRED" --endpoint "$EP" --model "$n" 2>>"$RES/cli.log")
      fig "$k" wake_s "$(echo "$r" | cut -d' ' -f2)"
      rec "wake on request $k" "$([ "${r%% *}" = 200 ] && [ "${r##* }" = ok ] && echo pass || echo fail)" "HTTP ${r%% *}, answer ${r##* }, ${r#* } s"
    fi
    c status deployment "$n" --json >"$RES/status-woken-$k.json" 2>>"$RES/cli.log"
  else rec "park $k" fail "refused; see cli.log"; fi
  stopd "$k"
  if [ "$PLAN" = 0 ]; then
    if [ "$(state_of "$n")" = stopped ]; then rec "stop $k" pass "engine gone, stopped"
    else rec "stop $k" fail "state not stopped"; fi
  fi
}

conc() { # side pass
  local k=$1 p=$2
  say "short prompts $(label "$k"), pass $p: C $CONC, $ROUNDS rounds"
  startd "$k" "warm-conc-$k-pass$p" || { rec "conc $k pass $p" fail "start failed"; return; }
  fig "$k" warm_s "$START_S"
  if bench "$k" "$RES/conc-$k-pass$p.json" --model "$(dep "$k")" --concurrency "$CONC" --rounds "$ROUNDS" \
    --concurrency-max-tokens 512 --record-chunks; then rec "conc $k pass $p" pass "done"
  else rec "conc $k pass $p" fail "see bench.log"; fi
  facts "conc-$k-pass$p"
  stopd "$k"
}

stall() { # key seed-base: the long-prompt stall trials on the running deployment
  local k=$1 t
  for t in $(seq "$STALL_TRIALS"); do
    helper stall --cred "$CRED" --endpoint "$EP" --model "$(dep "$k")" --streams "$STALL_STREAMS" \
      --prompt-tokens "$STALL_TOKENS" --stream-max-tokens "$STALL_MAX_TOKENS" --seed "$2$t" \
      --out "$RES/stall-$(stall_name "$k")-t$t.json" >>"$RES/cli.log" 2>&1 ||
      rec "stall $k trial $t" fail "see cli.log"
  done
}
stall_name() { case $1 in actx) echo a ;; *) echo "$1" ;; esac; }

ctx() { # key: context sweep, then the stall trials, on one warm start
  local k=$1 s
  s=$(side_of "$k")
  say "context sweep $SWEEP and stall test $(label "$s") ($(dep "$k"))"
  startd "$k" "warm-ctx-$k" || { rec "ctx $k" fail "start failed"; return; }
  fig "$s" warm_s "$START_S"
  if has_phase ctx; then
    if bench "$s" "$RES/ctx-$s.json" --model "$(dep "$k")" --record-chunks --context-sweep "$SWEEP" \
      --runs "$RUNS" --max-tokens 128 --meta context=262144; then rec "ctx $s" pass "done"
    else rec "ctx $s" fail "see bench.log"; fi
  fi
  has_phase stall && stall "$k" "$([ "$s" = a ] && echo 1 || echo 2)"
  facts "ctx-$s"
  stopd "$k"
}

stall_nomix() {
  say "stall test MTP/EAGLE without --enable-mixed-chunk"
  startd bn warm-stall-bn || { rec "stall bn" fail "start failed"; return; }
  fig b warm_s "$START_S"
  facts "stall-bn"
  stall bn 3
  stopd bn
}

down() {
  local d pid
  if [ "$PLAN" = 1 ]; then
    say "down: delete the deployments, remove both profiles, stop the role (TERM to its own PID), keep the kernel caches' sizes"
    for d in $DEPLOY_KEYS; do c delete deployment "$(dep "$d")" --stop; done
    [ "$MODE" = variants ] || c engine remove b18a
    c engine remove b18b
    show "kill -TERM <role pid from role.pid, checked against its argv>; wait 60 s; record the drained line"
    return 0
  fi
  DEPLOYMENTS=""
  [ -f "$WORK/deployments.txt" ] && DEPLOYMENTS=$(tr '\n' ' ' <"$WORK/deployments.txt")
  if [ -f "$PIDFILE" ] && pid=$(cat "$PIDFILE") && [ -r "/proc/$pid/cmdline" ] &&
     tr '\0' ' ' <"/proc/$pid/cmdline" | grep -qF -- "$SD"; then
    for d in $DEPLOYMENTS; do c delete deployment "$d" --stop >>"$RES/cli.log" 2>&1; done
    waitgone
    for d in b18a b18b; do c engine remove "$d" >>"$RES/cli.log" 2>&1; done
    # A signal is the role's drained shutdown: admission closes, in-flight requests finish.
    kill -TERM "$pid"
    for _ in $(seq 60); do [ -d "/proc/$pid" ] || break; sleep 1; done
    grep -o '"drained":[a-z]*[^}]*' "$RES/role.log" | tail -1 >"$RES/drained.txt"
    say "role stopped: $(cat "$RES/drained.txt")"
    if pgrep -f -- "$ENGINE_RE" >/dev/null; then rec "cleanup" fail "an engine process from the venv is still running"
    else rec "cleanup" pass "deployments deleted, profiles removed, role stopped, no engine left"; fi
  fi
  rm -f "$PIDFILE"
}

finish() { local rc=$?; raw_stop >/dev/null; down; echo "$rc" >"$WORK/done"; }

phases() {
  local p
  preflight || exit 1
  record_versions
  setup || exit 1
  if has_phase cold; then cold a; cold b; fi
  if has_phase life; then life a; life b; fi
  if has_phase conc; then for p in $(seq "$PASSES"); do conc a "$p"; conc b "$p"; done; fi
  if has_phase ctx || has_phase stall; then ctx actx; ctx b; fi
  has_phase stall && stall_nomix
  say "phases done"
}

# One variant of the MTP/EAGLE recipe: what CapyCTL rendered (its public launch
# settings, the engine processes' environment names, SGLang's resolved server
# arguments) and the short-prompt figures, on one warm start.
variant() { # key
  local k=$1 v n
  v=$(var_name "$k") n=$(dep "$k")
  say "variant $v ($n): C $CONC, $ROUNDS rounds"
  startd "$k" "var-$v" || { rec "variant $v" fail "start failed"; return; }
  fig "$v" start_s "$START_S"
  fig "$v" ready_gib "$READY_GIB"
  helper public --name "$n" --out "$RES/public-var-$v.json" >>"$RES/cli.log" 2>&1
  c inspect deployment "$n" --json >"$RES/inspect-var-$v.json" 2>>"$RES/cli.log"
  # CapyCTL's charge: the effective configuration's resources (management API).
  if [ "$PLAN" = 1 ]; then
    helper effective --cred "$CRED" --mport "$MPORT" --id "<id from inspect-var-$v.json>" --out "$RES/effective-var-$v.json"
  else
    helper effective --cred "$CRED" --mport "$MPORT" --id "$("${HELPER[@]}" get "$RES/inspect-var-$v.json" id)" \
      --out "$RES/effective-var-$v.json" >>"$RES/cli.log" 2>&1
  fi
  if [ -n "$LAUNCH_LOG" ]; then
    helper serverargs "$LAUNCH_LOG" --out "$RES/serverargs-var-$v.json" >>"$RES/cli.log" 2>&1
  fi
  if bench "$k" "$RES/var-$v.json" --model "$n" --concurrency "$CONC" --rounds "$ROUNDS" \
    --concurrency-max-tokens 512 --record-chunks; then rec "variant $v" pass "done"
  else rec "variant $v" fail "see bench.log"; fi
  facts "var-$v"
  stopd "$k"
}

# The control: the same venv, checkpoint and SGLang options launched directly,
# without CapyCTL, before the role starts so nothing else holds memory. It
# fills the b profile's kernel caches, which the CapyCTL variants then reuse.
RAW_PGID=$WORK/raw.pgid
raw_stop() { # TERM the raw engine's own process group; KILL it after 120 s
  local pg _
  [ -f "$RAW_PGID" ] || return 0
  pg=$(cat "$RAW_PGID")
  # Only the group this script started: its leader's argv names the venv.
  if pgrep -g "$pg" -f -- "$ENGINE_RE" >/dev/null; then
    kill -TERM -- "-$pg" 2>/dev/null
    for _ in $(seq 120); do pgrep -g "$pg" >/dev/null || break; sleep 1; done
    if pgrep -g "$pg" >/dev/null; then kill -KILL -- "-$pg" 2>/dev/null; sleep 5; echo killed; fi
  fi
  rm -f "$RAW_PGID"
}

raw_variant() {
  local port="" p before after code s _ args st
  say "variant raw: SGLang 0.5.21 launched directly, no CapyCTL; C $CONC, $ROUNDS rounds"
  args=(-m sglang.launch_server --model-path "$B_SNAPSHOT" --served-model-name qwen3.8-27b
    --context-length 262144 --tool-call-parser qwen3_coder --reasoning-parser qwen3
    --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4 --enable-linear-replayssm-spec --max-mamba-cache-size 110
    --enable-mixed-chunk --enable-metrics --mem-fraction-static 0.65 --tp-size 1 --pp-size 1
    --host 127.0.0.1)
  cache_env "$WORK/jit-b"
  local envs=(MAX_JOBS=4 NVCC_THREADS=1 FLASHINFER_NVCC_THREADS=1 CUDA_HOME=/usr/local/cuda HF_HUB_OFFLINE=1
    "PATH=$VENV/bin:/usr/local/cuda/bin:/usr/bin:/bin")
  for p in "${CACHE_ARGS[@]}"; do [ "$p" = --env ] || envs+=("$p"); done
  if [ "$PLAN" = 1 ]; then
    show "pick the first free port in $ENGINE_PORTS (ss)"
    show "\$ mkdir -p $WORK/jit-b"
    show "\$ setsid env -u CAPYCTL_API_KEY ${envs[*]} $VENV/bin/python3 ${args[*]} --port <port> >res/raw-engine.log 2>&1 &   (record the process group in raw.pgid)"
    show "poll http://127.0.0.1:<port>/health every 5 s until 200 (1800 s; stop early if the process exits)"
    show "\$ python3 capyctl-bench/capyctl_bench.py run --endpoint http://127.0.0.1:<port>/v1 --model qwen3.8-27b (no key) --label 'MTP/EAGLE raw SGLang (SGLang 0.5.21)' --concurrency $CONC --rounds $ROUNDS --concurrency-max-tokens 512 --record-chunks --out res/var-raw.json"
    helper serverargs "$RES/raw-engine.log" --out "$RES/serverargs-var-raw.json"
    helper facts "$RES/raw-engine.log" --out "$RES/facts-var-raw.json"
    show "kill -TERM -- -<pgid>; wait 120 s (KILL after); check no process runs from ~/sglang-0.5.21-venv/bin/ and MemAvailable is back within 2 GiB of before"
    return 0
  fi
  for p in $(seq "${ENGINE_PORTS%-*}" "${ENGINE_PORTS#*-}"); do
    ss -ltnH "sport = :$p" 2>/dev/null | grep -q . || { port=$p; break; }
  done
  [ -n "$port" ] || { rec "variant raw" fail "no free port in $ENGINE_PORTS"; return; }
  [ -f "$B_SNAPSHOT/config.json" ] || { rec "variant raw" fail "no HF cache snapshot to launch from"; return; }
  mkdir -p "$WORK/jit-b"
  before=$(memavail); s=$(now)
  setsid env -u CAPYCTL_API_KEY "${envs[@]}" "$VENV/bin/python3" "${args[@]}" --port "$port" \
    >"$RES/raw-engine.log" 2>&1 </dev/null &
  echo $! >"$RAW_PGID"
  code=none
  for _ in $(seq 360); do
    sleep 5
    pgrep -g "$(cat "$RAW_PGID")" >/dev/null || { code=exited; break; }
    code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health")
    [ "$code" = 200 ] && break
  done
  if [ "$code" != 200 ]; then
    rec "variant raw" fail "no /health 200 (last: $code) in 1800 s; see raw-engine.log"
    raw_stop >/dev/null
    return
  fi
  fig raw start_s "$(elapsed "$s")"
  sleep 10
  fig raw ready_gib "$(sub "$before" "$(memavail)")"
  say "raw SGLang ready on port $port in $(elapsed "$s") s"
  helper serverargs "$RES/raw-engine.log" --out "$RES/serverargs-var-raw.json" >>"$RES/cli.log" 2>&1
  if env -u CAPYCTL_API_KEY python3 "$CAPYCTL_BENCH_DIR/capyctl_bench.py" run --endpoint "http://127.0.0.1:$port/v1" \
    --model qwen3.8-27b --label "MTP/EAGLE raw SGLang (SGLang 0.5.21)" --memory-cmd "$MEMUSED" \
    --meta gpu="${GPU:-GB10}" --meta engine=sglang --meta engine_version=0.5.21 --meta capyctl=none \
    --meta model="$(model_ref b)" --meta drafter="$(drafter_ref b)" --concurrency "$CONC" --rounds "$ROUNDS" \
    --concurrency-max-tokens 512 --record-chunks --out "$RES/var-raw.json" >>"$RES/bench.log" 2>&1
  then rec "variant raw" pass "done"
  else rec "variant raw" fail "see bench.log"; fi
  helper facts "$RES/raw-engine.log" --out "$RES/facts-var-raw.json" >>"$RES/cli.log" 2>&1
  st=$(raw_stop)
  pause 15
  after=$(memavail)
  if pgrep -f -- "$ENGINE_RE" >/dev/null; then rec "raw stop" fail "a process from the venv is still running"
  elif [ "$(echo "$before - $after > 2" | bc)" = 1 ]; then
    rec "raw stop" fail "MemAvailable $after GiB, $before before the launch"
  else rec "raw stop" pass "${st:+$st; }no SGLang process left, MemAvailable $after GiB ($before before)"; fi
}

# The committed MTP/EAGLE file asks for 79 GiB, which CapyCTL now refuses
# before anything starts, naming the request that holds the MTP draft's
# copies, its KV and the state. Recorded: the exit code and message of
# `start --wait`, and the status; then the deployment is deleted.
refusal_check() {
  local n=qwen38-v-79gib rc named line out=$RES/refusal-79.txt
  say "check: the committed 79 GiB MTP/EAGLE file is refused before start, naming 84 GiB"
  if [ "$PLAN" = 1 ]; then
    show "(if deploy already refused it: take exit code and message from refusal-79-deploy.txt, skip start)"
    show "\$ timeout 900 capyctl --state-dir $SD start deployment $n --wait >res/refusal-79.txt 2>&1   (expect a non-zero exit naming 84 GiB)"
    c status deployment "$n"
    show "record exit code and the request it names ('raise engine_config.memory.request to \"<N>GiB\"'): figures refusal79 rc, named, message"
    c delete deployment "$n" --stop
    return 0
  fi
  if [ -n "${V79_DEPLOY_RC:-}" ] && [ "$V79_DEPLOY_RC" != 0 ]; then
    out=$RES/refusal-79-deploy.txt rc=$V79_DEPLOY_RC
  else
    timeout 900 "$CAPYCTL" --state-dir "$SD" start deployment "$n" --wait >"$out" 2>&1
    rc=$?
    { echo "--- status"; c status deployment "$n"; } >>"$out" 2>&1
  fi
  line=$(grep -m1 -E 'memory\.request to' "$out" || grep -m1 -iE 'refus|insufficient|error' "$out")
  line=$(tr '\t' ' ' <<<"$line" | cut -c1-900)
  named=$(grep -oE 'memory\.request to "[0-9]+(GiB|MiB)"' <<<"$line" | grep -oE '[0-9]+(GiB|MiB)' | head -1)
  fig refusal79 rc "$rc"
  fig refusal79 named "${named:-none}"
  fig refusal79 message "${line:-none}"
  if [ "$rc" != 0 ] && [ "$named" = 84GiB ]; then rec "79 GiB refused" pass "exit $rc, names $named"
  elif [ "$rc" != 0 ]; then rec "79 GiB refused" fail "exit $rc, names ${named:-no request}, expected 84GiB"
  else rec "79 GiB refused" fail "start --wait succeeded (exit 0)"; fi
  c delete deployment "$n" --stop >>"$RES/cli.log" 2>&1
  waitgone
}

variant_phases() {
  local k names=raw
  preflight || exit 1
  record_versions
  raw_variant
  setup || exit 1
  refusal_check
  for k in $VARIANT_KEYS; do variant "$k"; names="$names,$(var_name "$k")"; done
  helper variants "$RES" "$RES/variants.md" --names "$names" >>"$RES/cli.log" 2>&1
  [ "$PLAN" = 1 ] || sed 's/^/    /' "$RES/variants.md" | tee -a "$RES/progress.log"
  say "variants done"
}

case $cmd in
  run)
    trap finish EXIT
    : >"$RES/lifecycle.tsv"
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/^NVIDIA //')
    if [ "$MODE" = variants ]; then
      say "B18 variants: MTP/EAGLE deep, restart_only, restart_only without kv_cache_dtype; C $CONC, $ROUNDS rounds"
      variant_phases
    else
      say "B18: DFlash2 vs MTP/EAGLE, Qwen3.8-27B, SGLang 0.5.21; phases $PHASES"
      phases
    fi ;;
  plan)
    if [ "$MODE" = variants ]; then
      printf 'Host phases (remote.sh run, detached on the host; variants %s):\n' "$DEPLOY_KEYS"
      variant_phases
    else
      printf 'Host phases (remote.sh run, detached on the host; phases %s):\n' "$PHASES"
      phases
    fi
    down
    printf '\n' ;;
  down) down ;;
  purge) cd / && rm -rf -- "$WORK" ;;
  *) sed -n '2,14p' "$0"; exit 2 ;;
esac
