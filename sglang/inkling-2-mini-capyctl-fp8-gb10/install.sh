#!/usr/bin/env bash
# Prepared only. Run after the machine's benchmark owner releases the machine.
# No sudo, service changes, shell-profile changes, or model launch.
set -euo pipefail
if [[ ${INKLING_INSTALL_LOW_PRIORITY:-0} != 1 ]]; then
  exec nice -n 19 ionice -c 3 env INKLING_INSTALL_LOW_PRIORITY=1 bash "$0" "$@"
fi
umask 022
VENV="$HOME/sglang-inkling-pr51926-venv"
SHA=92e894ef904bfdb5c0cf2f2a541cbd4eec023573
unset PYTHONPATH PYTHONHOME
[[ $(uname -m) == aarch64 ]] || { echo 'This plan requires Linux aarch64.' >&2; exit 1; }
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
[[ ! -e "$VENV" ]] || { echo "Refusing to mix with an existing environment: $VENV" >&2; exit 1; }
[[ -x /usr/local/cuda/bin/nvcc ]] || { echo 'The existing CUDA toolkit is required.' >&2; exit 1; }
CUDA_VERSION=$(/usr/local/cuda/bin/nvcc --version)
printf '%s\n' "$CUDA_VERSION"
[[ $CUDA_VERSION == *'release 13.0,'* ]] || { echo 'This plan is for the observed CUDA 13.0 toolkit.' >&2; exit 1; }
for tool in git curl gcc g++ cmake sha256sum; do
  command -v "$tool" >/dev/null || { echo "Missing existing prerequisite: $tool" >&2; exit 1; }
done
# Do not update the system toolkit or driver. Python's CUDA libraries live in the venv.
python3 -m venv "$VENV"
mkdir -p "$VENV/.cache/tmp"
cat > "$VENV/bin/inkling-env" <<'ENV'
# Source this file before every future build or launch, including a cold JIT.
export VIRTUAL_ENV="$HOME/sglang-inkling-pr51926-venv"
export CUDA_HOME=/usr/local/cuda
export PATH="$VIRTUAL_ENV/bin:$VIRTUAL_ENV/.cache/cargo/bin:$CUDA_HOME/bin:$PATH"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="$VIRTUAL_ENV/.cache/pycache"
export PIP_CONFIG_FILE=/dev/null
export PIP_REQUIRE_VIRTUALENV=true
export PIP_CACHE_DIR="$VIRTUAL_ENV/.cache/pip"
export XDG_CACHE_HOME="$VIRTUAL_ENV/.cache"
export HF_HOME="$VIRTUAL_ENV/.cache/huggingface"
export TORCH_HOME="$VIRTUAL_ENV/.cache/torch"
export TORCH_EXTENSIONS_DIR="$VIRTUAL_ENV/.cache/torch-extensions"
export TRITON_CACHE_DIR="$VIRTUAL_ENV/.cache/triton"
export FLASHINFER_WORKSPACE_BASE="$VIRTUAL_ENV/.cache/flashinfer"
export CUTE_DSL_CACHE_DIR="$VIRTUAL_ENV/.cache/cute-dsl"
export CUDA_CACHE_PATH="$VIRTUAL_ENV/.cache/cuda"
export TMPDIR="$VIRTUAL_ENV/.cache/tmp"
export CARGO_HOME="$VIRTUAL_ENV/.cache/cargo"
export RUSTUP_HOME="$VIRTUAL_ENV/.cache/rustup"
export RUSTUP_TOOLCHAIN=1.92.0
export CARGO_TARGET_DIR="$VIRTUAL_ENV/.cache/cargo-target"
export MAX_JOBS=2
export CARGO_BUILD_JOBS=2
export CMAKE_BUILD_PARALLEL_LEVEL=2
export NVCC_THREADS=1
export OMP_NUM_THREADS=2
export TORCH_CUDA_ARCH_LIST=12.1
ENV
# shellcheck source=/dev/null
source "$VENV/bin/inkling-env"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME" "$CARGO_TARGET_DIR"
python -m pip install --only-binary=:all: \
  'pip==26.3' 'setuptools==84.2.1' 'wheel==0.48.0' \
  'setuptools-rust==1.13.0' 'setuptools-scm==10.3.4' \
  'build==1.6.1' 'ninja==1.13.2' 'cmake==4.4.5'
SRC="$VENV/.cache/src/sglang-$SHA"
mkdir -p "$SRC"
git -C "$SRC" init -q
git -C "$SRC" fetch --depth 1 https://github.com/sgl-project/sglang.git "$SHA"
git -C "$SRC" checkout --detach FETCH_HEAD
[[ $(git -C "$SRC" rev-parse HEAD) == "$SHA" ]]
git -C "$SRC" submodule update --init --recursive --jobs 2
# The PR's own workspace requests Rust 1.92; keep rustup and cargo inside the venv.
RUSTUP_INIT="$VENV/.cache/rustup-init"
curl --fail --location --retry 3 \
  https://static.rust-lang.org/rustup/archive/1.29.0/aarch64-unknown-linux-gnu/rustup-init \
  --output "$RUSTUP_INIT"
curl --fail --location --retry 3 \
  https://static.rust-lang.org/rustup/archive/1.29.0/aarch64-unknown-linux-gnu/rustup-init.sha256 \
  --output "$RUSTUP_INIT.sha256"
python - "$RUSTUP_INIT" <<'PY'
import hashlib, pathlib, sys
p = pathlib.Path(sys.argv[1])
expected = pathlib.Path(str(p) + '.sha256').read_text().split()[0]
assert hashlib.sha256(p.read_bytes()).hexdigest() == expected
PY
chmod u+x "$RUSTUP_INIT"
"$RUSTUP_INIT" -y --no-modify-path --profile minimal --default-toolchain 1.92.0
# Generate the complete requirement set from the pinned source, preserving platform markers.
python - "$SRC/python/pyproject.toml" "$VENV/.cache/pr-requirements.txt" <<'PY'
import pathlib, sys, tomllib
project = tomllib.loads(pathlib.Path(sys.argv[1]).read_text())
pathlib.Path(sys.argv[2]).write_text('\n'.join(project['project']['dependencies']) + '\n')
PY
# cu130 matches the existing toolkit and the PR's Docker dependency index.
# Binary-only dependency installation fails rather than silently compiling a large dependency.
python -m pip install --only-binary=:all: \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --report "$VENV/.cache/dependency-install-report.json" \
  -r "$VENV/.cache/pr-requirements.txt"
# Keep all Rust extensions, including the image pipeline, instead of disabling build errors.
# CUDA custom kernels and FA4 remain lazy JIT; this command does not start a model.
python -m pip install --no-build-isolation --no-deps \
  --report "$VENV/.cache/source-install-report.json" "$SRC/python"
python -m pip freeze --all > "$VENV/.cache/installed-freeze.txt"
printf '%s\n' 'Installation finished; no model was launched.' \
  "Future shells: source $VENV/bin/inkling-env" \
  'Retain MAX_JOBS=2 for the first cold start; use at most 4 only while otherwise idle.'
