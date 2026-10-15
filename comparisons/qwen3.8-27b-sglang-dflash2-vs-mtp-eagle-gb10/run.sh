#!/usr/bin/env bash
# Qwen3.8-27B on SGLang 0.5.21: the DFlash2 recipe against the MTP/EAGLE recipe on
# one GB10 host, through CapyCTL, with capyctl-bench, passes interleaved. Modelled
# on capyctl's scripts/live/engine-quick-check.sh, whose matrix helpers it uses.
#
# usage: run.sh [options]
#
#   --capyctl DIR      capyctl checkout to build CapyCTL from ($CAPYCTL_REPO, else
#                      ../capyctl beside this repository's main checkout)
#   --host a|b         lab host (a)
#   --port N           inference port of the run's own standalone role (18543); the
#                      management API takes N+1 and engines --engine-ports (8300-8319)
#   --rounds N         measured rounds per stream count and pass (5)
#   --passes N         interleaved passes per recipe (2)
#   --runs N           requests per context point (3)
#   --concurrency L    stream counts (1,4,8,16,22)
#   --sweep L          context points (2k,32k,128k)
#   --stall-tokens N   the cold long prompt of the stall test (80000)
#   --stall-streams N  streams decoding while it is prefilled (4)
#   --stall-trials N   trials per variant (2)
#   --b-source S       MTP/EAGLE weights: `cache`, the host's Hugging Face cache snapshot
#                      (default), or `hf`, downloaded by CapyCTL into its sources store
#   --phases L         cold,life,conc,ctx,stall (all)
#   --managed-limit S  standalone managed memory limit (100GiB)
#   --parked-limit S   standalone parked limit (45GiB)
#   --out DIR          local results (<capyctl main checkout>/target/live/0.1.3-backlog/B18/<run>)
#   --skip-build       use the CapyCTL binary already built on the host
#   --no-readme        leave this directory's README.md and bench/ as they are
#   --variants         only the MTP/EAGLE variants: first `raw`, the same venv and SGLang options
#                      launched directly (no CapyCTL, --mem-fraction-static 0.65); then one
#                      standalone with profile b18b: a check that the committed 79 GiB file is
#                      refused at start naming 84 GiB, and with `memory.request: 84GiB`: deep
#                      (CapyCTL's default), residency restart_only, and restart_only without
#                      kv_cache_dtype; C1,C8,C22, 3 rounds, one pass; writes variants.md
#                      (tok/s, accept length, rendered and resolved sizing, Ready footprint
#                      against CapyCTL's charge) under --out, never the README
#   --experiment       the catalog decision: raw SGLang (BF16-head checkpoint) first, then one
#                      standalone with profiles b18a/b18b sharing the raw start's kernel caches,
#                      all CapyCTL variants restart_only but one: BF16 head deep and restart_only,
#                      the catalog NVFP4-head checkpoint with the same EAGLE MTP flags, the catalog
#                      DFlash2 recipe with mixed chunks, 110 slots and 22 running; then the faster
#                      EAGLE MTP checkpoint at depths 2/1/3, 5/1/6, 4/4/8 (C1, C8). A start refused
#                      with a named memory request is redeployed once at it (skipped above the
#                      managed limit). Stall test on the NVFP4 head and the DFlash2 mix; accuracy
#                      (200-prompt greedy agreement, GSM8K subset, 30 tool calls) on both heads.
#                      C1,C8,C22, 3 rounds, one pass, 1 stall trial; writes experiment.md under --out
#   --gsm8k-n N        GSM8K questions per head (250);  --gsm8k-max-tokens N (2048)
#   --sweep-rounds N   measured rounds per depth in the sweep (2)
#   --dry-run          print every step, local and on the host, and contact no host
#
# Hosts come from the matrix harness's hosts.local.env ($CAPYCTL_MATRIX_HOSTS_ENV,
# else the capyctl main checkout's scripts/live/matrix/hosts.local.env); output
# never names a host. One live session at a time: the host phases refuse to start
# while any engine runs there.
set -euo pipefail

CMP=$(cd "$(dirname "$0")" && pwd)
RECIPES=$(git -C "$CMP" rev-parse --show-toplevel)
BENCH=$RECIPES/tools/capyctl-bench
usage() { sed -n '2,54p' "$0"; exit 2; }

CAPYCTL=${CAPYCTL_REPO:-} HOST_CODE=a PORT=18543 ENGINE_PORTS=8300-8319 ROUNDS="" PASSES="" RUNS=3
CONC="" SWEEP=2k,32k,128k STALL_TOKENS=80000 STALL_STREAMS=4 STALL_TRIALS=""
STALL_MAX_TOKENS=3072 B_SOURCE=cache PHASES=cold,life,conc,ctx,stall MANAGED_LIMIT=100GiB
PARKED_LIMIT=45GiB FREE_RESERVE=10GiB OUT="" SKIP_BUILD=0 README=1 DRY=0 MODE=full
GSM_N=250 GSM_MAX_TOKENS=2048 SWEEP_ROUNDS=2
B_COMMIT=009632fef96dd349150baa780c984e62e70e91fe
ARGS=("$@")
while [ $# -gt 0 ]; do
  case $1 in
    --capyctl) CAPYCTL=$2; shift ;;
    --host) HOST_CODE=$2; shift ;;
    --port) PORT=$2; shift ;;
    --rounds) ROUNDS=$2; shift ;;
    --passes) PASSES=$2; shift ;;
    --runs) RUNS=$2; shift ;;
    --concurrency) CONC=$2; shift ;;
    --sweep) SWEEP=$2; shift ;;
    --stall-tokens) STALL_TOKENS=$2; shift ;;
    --stall-streams) STALL_STREAMS=$2; shift ;;
    --stall-trials) STALL_TRIALS=$2; shift ;;
    --b-source) B_SOURCE=$2; shift ;;
    --phases) PHASES=$2; shift ;;
    --managed-limit) MANAGED_LIMIT=$2; shift ;;
    --parked-limit) PARKED_LIMIT=$2; shift ;;
    --out) OUT=$2; shift ;;
    --skip-build) SKIP_BUILD=1 ;;
    --no-readme) README=0 ;;
    --variants) MODE=variants ;;
    --experiment) MODE=experiment ;;
    --gsm8k-n) GSM_N=$2; shift ;;
    --gsm8k-max-tokens) GSM_MAX_TOKENS=$2; shift ;;
    --sweep-rounds) SWEEP_ROUNDS=$2; shift ;;
    --dry-run) DRY=1 ;;
    -h|--help) usage ;;
    *) echo "unknown option $1" >&2; usage ;;
  esac
  shift
done
case $B_SOURCE in cache|hf) ;; *) echo "--b-source is cache or hf" >&2; exit 2 ;; esac
# The variants and experiment modes: C1, C8, C22, 3 rounds, one pass, unless stated.
if [ "$MODE" != full ]; then CONC=${CONC:-1,8,22} ROUNDS=${ROUNDS:-3} PASSES=${PASSES:-1}; fi
[ "$MODE" = experiment ] && STALL_TRIALS=${STALL_TRIALS:-1}
CONC=${CONC:-1,4,8,16,22} ROUNDS=${ROUNDS:-5} PASSES=${PASSES:-2} STALL_TRIALS=${STALL_TRIALS:-2}
[ "$PORT" != 18443 ] || { echo "port 18443 is the quick check's; pick another" >&2; exit 2; }
MPORT=$((PORT + 1))

# The capyctl checkout to build from, and its main checkout (hosts file, output root).
if [ -z "$CAPYCTL" ]; then
  CAPYCTL=$(dirname "$(git -C "$RECIPES" rev-parse --path-format=absolute --git-common-dir)")/../capyctl
fi
CAPYCTL=$(cd "$CAPYCTL" && pwd)
CAPYCTL_MAIN=$(dirname "$(git -C "$CAPYCTL" rev-parse --path-format=absolute --git-common-dir)")
LIVE_DIR=$CAPYCTL/scripts/live
[ -f "$LIVE_DIR/matrix/lib.sh" ] || { echo "no scripts/live/matrix/lib.sh in $CAPYCTL (--capyctl)" >&2; exit 2; }
[ -f "$BENCH/capyctl_bench.py" ] || { echo "capyctl-bench not found in $BENCH" >&2; exit 2; }
export CAPYCTL_MATRIX_HOSTS_ENV=${CAPYCTL_MATRIX_HOSTS_ENV:-$CAPYCTL_MAIN/scripts/live/matrix/hosts.local.env}
if [ "$DRY" = 1 ] && [ ! -f "$CAPYCTL_MATRIX_HOSTS_ENV" ]; then
  echo "dry run: $CAPYCTL_MATRIX_HOSTS_ENV missing, using hosts.example.env" >&2
  export CAPYCTL_MATRIX_HOSTS_ENV=$LIVE_DIR/matrix/hosts.example.env
fi
export DRY_RUN=$DRY

# Hosts, sync and remote helpers of the matrix harness.
# shellcheck source=/dev/null
. "$LIVE_DIR/matrix/lib.sh"
HOST=$(resolve_host "$HOST_CODE")
HOST_ADDR=$(host_ip "$HOST")
# Every line this script and its children print goes through redact: no host
# name, address or home directory reaches the terminal or the output files.
redact() { sed -u -e "s#$REMOTE_HOME#~#g" -e "s#$HOME#~#g" -e "s#$HOST_ADDR#host-$HOST_CODE-addr#g" -e "s#$HOST#host-$HOST_CODE#g"; }
exec > >(redact) 2>&1

# The backlog's tree for 0.1.3 live work. The run's work directory is short,
# ~/c13b/<id>, because the role's state directory (<work>/s) holds SGLang's
# saver observation socket and a Unix socket path holds at most 107 bytes.
export CAPYCTL_REMOTE_TREE=${CAPYCTL_REMOTE_TREE:-$REMOTE_HOME/capyctl-0.1.3-live}
case $MODE in variants) RUN_ID=b18v ;; experiment) RUN_ID=b18x ;; *) RUN_ID=b18 ;; esac
RUN_ID=$RUN_ID-$(date +%m%d%H%M)
OUT=${OUT:-$CAPYCTL_MAIN/target/live/0.1.3-backlog/B18/$(date +%F-%H%M)-$(case $MODE in full) echo b18 ;; *) echo "$MODE" ;; esac)}
WORK=$REMOTE_HOME/c13b/$RUN_ID
step() { printf '\n== %s\n' "$*"; }
estimate() { # minutes, from the settings (GB10 figures of earlier runs)
  local levels=0 slow=0 c
  for c in ${CONC//,/ }; do levels=$((levels + 1)); [ "$c" -gt 8 ] && slow=$((slow + (c + 7) / 8 - 1)); done
  local build=$(( SKIP_BUILD ? 0 : 20 ))
  # Per CapyCTL start: a warm start (~5), one warm-up and the measured rounds
  # at ~30 s a round per stream count, a stop (~1). The raw SGLang start
  # builds the kernels (~10 with MAX_JOBS=4) and stops (~2).
  local per=$(( 6 + (ROUNDS + 1) * levels * 30 / 60 ))
  if [ "$MODE" = variants ]; then
    echo "about $((build + 7 + 2 + 4 * per + 5)) minutes: build $build, raw SGLang ~$((7 + per)) (kernel build included), 79 GiB refusal check ~2, 3 CapyCTL variants ~$per each, cleanup ~5"
    return
  fi
  if [ "$MODE" = experiment ]; then
    # Accuracy on one checkpoint: agreement ~4 (+1 for the BF16 repeat), GSM8K
    # ~0.054 min a question at C22 (about 900 tokens each), tool calls ~2.
    local gsm=$(( GSM_N * 54 / 1000 )) stall=$(( STALL_TRIALS * 4 )) sweep=$(( 3 * (6 + SWEEP_ROUNDS + 1) ))
    local acc=$(( 6 + gsm ))
    local total=$(( build + 7 + per + 2 + per + per + acc + 1 + per + 2 + stall + acc + per + 2 + stall + 5 + sweep + 5 ))
    echo "about $total minutes: build $build, raw SGLang ~$((7 + per)), setup ~2, BF16 deep ~$per, BF16 restart_only ~$((per + acc + 1)) (accuracy included), NVFP4 head ~$((per + 2 + stall + acc)) (stall, accuracy, one retry), DFlash2 mix ~$((per + 2 + stall + 5)) (stall, one retry, drafter kernels), depth sweep ~$sweep, cleanup ~5"
    return
  fi
  local conc=$(( PASSES * ( 2 * 5 + (ROUNDS + 1) * (levels * 2 * 25 + slow * 25) / 60 ) ))
  local ctx=$(( 2 * (5 + RUNS * 3) )) stall=$(( 5 + 3 * STALL_TRIALS * 4 ))
  echo "about $((build + 50 + 20 + conc + ctx + stall + 10)) minutes: build $build, cold starts ~50 (kernel builds), lifecycle ~20, short prompts ~$conc, context ~$ctx, stall ~$stall, cleanup ~10"
}
case $MODE in
  variants) step "B18 variants: MTP/EAGLE deep / restart_only / restart_only without kv_cache_dtype, host $HOST_CODE -> $OUT" ;;
  experiment) step "B18 experiment: raw SGLang, BF16 head deep/restart_only, NVFP4 head, DFlash2 mix, depth sweep, accuracy; host $HOST_CODE -> $OUT" ;;
  *) step "B18: DFlash2 vs MTP/EAGLE on SGLang 0.5.21, host $HOST_CODE -> $OUT" ;;
esac
echo "capyctl $(git -C "$CAPYCTL" rev-parse --short HEAD) from $CAPYCTL; origin/main is $(git -C "$CAPYCTL" rev-parse --short origin/main 2>/dev/null || echo unknown)"
echo "estimated duration: $(estimate)"
[ "$DRY" = 1 ] || mkdir -p "$OUT"

# 1. CapyCTL from the checkout, built on the host.
if [ "$SKIP_BUILD" = 1 ]; then
  rsh "$HOST" "test -x $CAPYCTL_REMOTE_TREE/target/release/capyctl" || die "no binary on host $HOST_CODE; drop --skip-build"
  COMMIT=$(git -C "$CAPYCTL" rev-parse HEAD)
else
  step "build CapyCTL $(git -C "$CAPYCTL" rev-parse --short HEAD) on host $HOST_CODE"
  "$LIVE_DIR/matrix/sync.sh" snapshot
  "$LIVE_DIR/matrix/sync.sh" push "$HOST"
  "$LIVE_DIR/matrix/sync.sh" build "$HOST"
  if [ "$DRY" = 1 ]; then COMMIT=$(git -C "$CAPYCTL" rev-parse HEAD); else COMMIT=$(cat "$SNAPSHOT/commit"); fi
fi

# 2. Helpers, deployment files, capyctl-bench and settings to the host's work directory.
step "prepare the host work directory"
ENVFILE=$(mktemp)
{
  printf '%s=%q\n' PORT "$PORT" MPORT "$MPORT" ENGINE_PORTS "$ENGINE_PORTS" ROUNDS "$ROUNDS" PASSES "$PASSES" \
    RUNS "$RUNS" CONC "$CONC" SWEEP "$SWEEP" STALL_TOKENS "$STALL_TOKENS" STALL_STREAMS "$STALL_STREAMS" \
    STALL_TRIALS "$STALL_TRIALS" STALL_MAX_TOKENS "$STALL_MAX_TOKENS" B_SOURCE "$B_SOURCE" B_COMMIT "$B_COMMIT" \
    MANAGED_LIMIT "$MANAGED_LIMIT" FREE_RESERVE "$FREE_RESERVE" PARKED_LIMIT "$PARKED_LIMIT" PHASES "$PHASES" CAPYCTL_COMMIT "$COMMIT" \
    MODE "$MODE" GSM_N "$GSM_N" GSM_MAX_TOKENS "$GSM_MAX_TOKENS" SWEEP_ROUNDS "$SWEEP_ROUNDS" \
    CAPYCTL_BIN "$CAPYCTL_REMOTE_TREE/target/release/capyctl"
} >"$ENVFILE"
rsh "$HOST" "rm -rf $WORK && mkdir -m 700 -p $WORK/cmp"
x rsync -rlpz --chmod=Dgo-w,Fgo-w "$CMP/remote.sh" "$CMP/b18.py" "$HOST:$WORK/"
x rsync -rlpz --chmod=Dgo-w,Fgo-w "$CMP"/deployment-*.yaml "$CMP/accuracy" "$HOST:$WORK/cmp/"
x rsync -rlpz --chmod=Dgo-w,Fgo-w --exclude __pycache__ --exclude tests "$BENCH/" "$HOST:$WORK/capyctl-bench/"
x rsync -z "$ENVFILE" "$HOST:$WORK/b18.env"

# 3. The phases run detached on the host (setsid), so a dropped connection does
# not stop them; this side follows the progress log until the done marker appears.
step "host phases on host $HOST_CODE (progress below)"
if [ "$DRY" = 1 ]; then
  log_cmd "$HOST" "cd $WORK && WORK=$WORK setsid nohup bash remote.sh run >run.log 2>&1 </dev/null &"
  PLANDIR=$(mktemp -d)
  cp "$CMP/remote.sh" "$CMP/b18.py" "$PLANDIR/"
  sed -e "s#^CAPYCTL_BIN=.*#CAPYCTL_BIN=capyctl#" "$ENVFILE" >"$PLANDIR/b18.env"
  WORK=$PLANDIR bash "$PLANDIR/remote.sh" plan | sed -e "s#$PLANDIR#$WORK#g"
  rm -rf "$PLANDIR"
  rc=0
else
  rsh "$HOST" "cd $WORK && WORK=$WORK setsid nohup bash remote.sh run >run.log 2>&1 </dev/null & echo started" &
  LAUNCH=$!
  seen=0
  while :; do
    sleep 20
    lines=$(rsh_out "$HOST" "" "tail -n +$((seen + 1)) $WORK/res/progress.log 2>/dev/null; true" 2>/dev/null) || continue
    if [ -n "$lines" ]; then printf '%s\n' "$lines"; seen=$((seen + $(printf '%s\n' "$lines" | wc -l))); fi
    rc=$(rsh_out "$HOST" "" "cat $WORK/done 2>/dev/null; true" 2>/dev/null) || continue
    [ -z "$rc" ] || break
  done
  wait "$LAUNCH" 2>/dev/null || true
fi
rm -f "$ENVFILE"
step "host phases ended with status $rc"

# 4. Results back, names and home directories redacted; the work directory removed.
step "copy the results back and remove the host's work directory"
x mkdir -p "$OUT/raw"
x rsync -rlz "$HOST:$WORK/res/" "$OUT/raw/"
x rsync -rlz "$HOST:$WORK/deployments" "$HOST:$WORK/run.log" "$OUT/raw/"
if [ "$DRY" = 1 ]; then
  log_cmd control-host "redact host names, addresses and the remote home in $OUT/raw"
else
  find "$OUT/raw" -type f \( -name '*.log' -o -name '*.txt' -o -name '*.tsv' -o -name '*.yaml' -o -name '*.json' \) \
    -exec sed -i -e "s#$REMOTE_HOME#~#g" -e "s#$HOST_ADDR#host-$HOST_CODE-addr#g" -e "s#$HOST#host-$HOST_CODE#g" {} +
fi
rsh "$HOST" "bash $WORK/remote.sh purge"

# 5. Pool the passes, report, settings, README (variants: the variant table).
step "$([ "$MODE" = variants ] && echo "variant table, report" || echo "pool passes, report, README")"
BENCH_COMMIT=$(git -C "$RECIPES" rev-parse HEAD 2>/dev/null || echo unknown)
SETTINGS=$(python3 -c 'import json, sys
v = sys.argv[1:]
print(json.dumps(dict(zip(v[0::2], v[1::2])), indent=1))' date "$(date +%F)" capyctl_commit "$COMMIT" \
  bench_commit "$BENCH_COMMIT" port "$PORT" managed_limit "$MANAGED_LIMIT" free_reserve "$FREE_RESERVE" parked_limit "$PARKED_LIMIT" \
  concurrency "$CONC" rounds "$ROUNDS" passes "$PASSES" sweep "$SWEEP" runs "$RUNS" \
  stall_tokens "$STALL_TOKENS" stall_streams "$STALL_STREAMS" stall_trials "$STALL_TRIALS" \
  stall_max_tokens "$STALL_MAX_TOKENS" b_source "$B_SOURCE" phases "$PHASES" mode "$MODE" \
  b_model_ref "RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead@$B_COMMIT" \
  command "comparisons/qwen3.8-27b-sglang-dflash2-vs-mtp-eagle-gb10/run.sh ${ARGS[*]}")
if [ "$DRY" = 1 ]; then
  log_cmd control-host "write $OUT/settings.json"
else
  printf '%s\n' "$SETTINGS" >"$OUT/settings.json"
fi
if [ "$MODE" = experiment ]; then
  # Rebuilt here from the redacted copies; the host printed the same tables.
  names="" files=()
  for n in raw bf16-deep bf16-restart nvfp4-restart dflash-mix-restart sweep-2-1-3 sweep-5-1-6 sweep-4-4-8; do
    if [ "$DRY" = 1 ] || [ -f "$OUT/raw/var-$n.json" ]; then
      names="$names${names:+,}$n"
      case $n in sweep-*) ;; *) files+=("$OUT/raw/var-$n.json") ;; esac
    fi
  done
  x python3 "$CMP/b18.py" experiment "$OUT/raw" "$OUT/experiment.md" --names "$names"
  x uv run --quiet --with matplotlib python3 "$BENCH/capyctl_bench.py" report "${files[@]}" \
    --out "$OUT/report" --summary --no-zip \
    --title "Qwen3.8-27B on SGLang 0.5.21: direct SGLang, BF16 and NVFP4 heads, DFlash2 mix" ||
    echo "report failed"
  step "done: $OUT (experiment.md, experiment.json, experiment.accuracy.json, raw/, report/, settings.json)"
  [ "$rc" = 0 ]
  exit
fi
if [ "$MODE" = variants ]; then
  # Rebuilt here from the redacted copies; the host printed the same table.
  x python3 "$CMP/b18.py" variants "$OUT/raw" "$OUT/variants.md"
  x uv run --quiet --with matplotlib python3 "$BENCH/capyctl_bench.py" report \
    "$OUT/raw/var-raw.json" "$OUT/raw/var-deep.json" "$OUT/raw/var-restart.json" "$OUT/raw/var-restart-autokv.json" \
    --out "$OUT/report" --summary --no-zip \
    --title "Qwen3.8-27B MTP/EAGLE on SGLang 0.5.21: direct SGLang and CapyCTL variants" ||
    echo "report failed"
  step "done: $OUT (variants.md, variants.json, raw/, report/, settings.json)"
  [ "$rc" = 0 ]
  exit
fi
x python3 "$CMP/b18.py" --bench "$BENCH" merge "$OUT/raw" "$OUT/results"
x uv run --quiet --with matplotlib python3 "$BENCH/capyctl_bench.py" report \
  "$OUT/results/dflash2.json" "$OUT/results/mtp-eagle.json" --out "$OUT/report" --summary \
  --title "Qwen3.8-27B on SGLang 0.5.21: DFlash2 vs MTP/EAGLE, one GB10" || echo "report failed"
if [ "$README" = 1 ]; then
  step "write $CMP/README.md, bench/ and results/"
  x mkdir -p "$CMP/bench/results" "$CMP/results"
  x rsync -a --exclude report.zip "$OUT/report/" "$CMP/bench/"
  x cp "$OUT/results/dflash2.json" "$OUT/results/mtp-eagle.json" "$CMP/bench/results/"
  x rsync -a --exclude 'state.json' --exclude 'conc-*' --exclude 'ctx-*' --include '*.json' --include '*.tsv' \
    --include 'versions.txt' --include 'drained.txt' --exclude '*' "$OUT/raw/" "$CMP/results/"
  x python3 "$CMP/b18.py" readme "$OUT" "$CMP/README.md"
fi
step "done: $OUT (raw/, results/, report/, settings.json)"
[ "$rc" = 0 ]
