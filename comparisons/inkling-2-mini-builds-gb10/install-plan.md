# Inkling 2 Mini: isolated installation plan

Prepared 2026-11-20. **Not executed.** The machine is occupied by other benchmarks. This task only downloads model files and reads toolchain facts; it does not create the venv, install packages, compile kernels, or start a model.

## Pins and observed machine facts

- SGLang PR [51926](https://github.com/sgl-project/sglang/pull/51926), commit **92e894ef904bfdb5c0cf2f2a541cbd4eec023573**. Never install the moving PR head.
- Model **thinkingmachines/Inkling-2-Mini**, revision **6e7c8248f75265bdd4fbab5273304243b5d8c2cb**, including `mtp.safetensors`.
- New environment: `~/sglang-inkling-pr51926-venv`. No system-site packages, no existing environment reuse, no system driver/toolkit changes, no shell-profile edits.
- Observed: Linux **aarch64**, **Python 3.12.3**, NVIDIA driver **580.173.02**.
- `nvcc` is absent from the login PATH, but `/usr/local/cuda/bin/nvcc` and `/usr/local/cuda-13.0/bin/nvcc` both report **CUDA 13.0, V13.0.88**, build `cuda_13.0.r13.0/compiler.36424714_0`.
- `git`, `gcc`, `cmake`, `curl`, `ionice` exist. `cargo` and `rustc` were not found on the login PATH. The installer downloads an isolated Rust toolchain into its own cache. The future run must also have the existing `g++` and Python's `venv`/`ensurepip`; neither is silently installed system-wide.

## Required dependency set and wheel availability

The authority is the pinned [python/pyproject.toml](https://github.com/sgl-project/sglang/blob/92e894ef904bfdb5c0cf2f2a541cbd4eec023573/python/pyproject.toml). `install.sh` extracts **all** of its ordinary dependencies, including platform markers, rather than inventing a reduced Spark dependency set. No diffusion/test/all extras are needed by this cookbook recipe.

| Component | Exact requirement at this commit | Published Linux ARM/Python 3.12 artifact |
|---|---|---|
| Main package | SGLang source at the commit above | Build main wheel from pinned source; includes Rust extensions |
| Kernel distribution | **sglang-kernel==0.4.11** | `sglang_kernel-0.4.11-cp311-abi3-manylinux2014_aarch64.whl` |
| Kernel Python import | `sgl_kernel` | Supplied by **sglang-kernel**, not a second package; `sgl-kernel==0.4.11` is not published |
| PyTorch | **torch==2.14.1** | `torch-2.14.1-cp312-cp312-manylinux_2_28_aarch64.whl`; use cu130 extra index as the PR's Dockerfile does |
| Audio / image helpers | **torchaudio==2.11.0**, torchvision unpinned by PR | ARM wheels published; torchvision 0.29.2 has an ARM wheel and declares torch>=2.14.0 |
| Transformers | **5.19.0**, tokenizers **0.23.2** | Universal Transformers wheel; ARM ABI3 tokenizer wheel |
| FlashInfer | **flashinfer_python[cu13]==0.7.4** | Universal Python wheel, with CUDA kernels compiled lazily as needed |
| CuTe DSL | **nvidia-cutlass-dsl[cu13]==4.8.0** | Universal frontend; `nvidia_cutlass_dsl_libs_cu13-4.8.0-cp312-cp312-manylinux_2_28_aarch64.whl` |
| FA4 | Vendored under `sglang/kernels/ops/attention/flash_attn/cute` plus `fa4_sm120` | Included in the pinned source; do **not** substitute an independently installed upstream flash-attn wheel |
| TVM FFI | **apache-tvm-ffi==0.1.11** | ARM ABI3 wheel |
| Deep EP / GEMM / MLA | **sgl-deep-ep==0.1.3**, **sgl-deep-gemm==0.2.9**, **sgl-flash-mla==1.0.5** | ARM wheels published; EP's cp312 wheel requires torch==2.14.1 |
| Other kernel helpers | **humming-kernels[cu13]==0.1.18**, **quack-kernels==0.6.5**, **tilelang==0.1.12** | ARM or universal wheels published; TileLang needs glibc>=2.34 |
| Precision / render helpers | **compressed-tensors==0.18.0**, **tml-renderers==0.4.2** | Universal / ARM ABI3 wheels |
| Runtime helpers | **nvidia-mathdx==25.6.0**, cuda-python>=13.0, nvshmem4py-cu13, torch_memory_saver>=0.0.9.post1 | Universal / ARM wheels available; versions without exact pins remain resolver-selected |

These are **publication checks against PyPI JSON**, not imports, GPU compatibility tests, or an installed lockfile. The PyTorch wheel's metadata requires CUDA toolkit Python libraries **13.0.3**, cuDNN **9.24.0.43**, cuSPARSELt **0.8.1**, NCCL **2.30.7**, NVSHMEM **3.4.5**, and Triton **~3.8.0** on Linux. Their libraries install inside the new venv. The on-disk compiler remains the existing 13.0.88 toolkit; driver/runtime qualification remains unmeasured.

Other exact PR pins include blobfile 3.0.0, numba 0.65.1, openai 2.8.0, openai-harmony 0.0.4, outlines 0.1.11, sentencepiece 0.2.1, soundfile 0.13.1, timm 1.0.25, xgrammar 0.2.7, and ARM av 16.1.0. `sgl-fa3==0.1.0` is **x86-only and skipped**. `torchcodec==0.17.0` is **skipped on Linux ARM**. Do not manually force either package. The installer's generated `pr-requirements.txt` is the complete top-level list, including the PR's unpinned HTTP/tokenization/multimodal dependencies.

The full package set is **not transitively locked upstream**. `kernels` must remain >=0.14.1,<0.15, Pillow <13, NumPy must satisfy numba's <2.5, and python-dateutil <3; do not pin today's unconstrained latest versions over those bounds. The resolver's reports and final freeze are saved inside the venv cache. No claim of a fully resolved, reproducible transitive environment is made before that future resolver run.

Sources: [kernel metadata](https://pypi.org/pypi/sglang-kernel/0.4.11/json), [torch metadata](https://pypi.org/pypi/torch/2.14.1/json), [FlashInfer metadata](https://pypi.org/pypi/flashinfer-python/0.7.4/json), [CuTe CUDA 13 metadata](https://pypi.org/pypi/nvidia-cutlass-dsl-libs-cu13/4.8.0/json), [FA4 package metadata](https://github.com/sgl-project/sglang/blob/92e894ef904bfdb5c0cf2f2a541cbd4eec023573/python/sglang/kernels/ops/attention/flash_attn/cute/pyproject.toml), [Docker CUDA/Python/dependency choices](https://github.com/sgl-project/sglang/blob/92e894ef904bfdb5c0cf2f2a541cbd4eec023573/docker/Dockerfile#L194-L237).

### What builds, and why MAX_JOBS matters

1. The main source package discovers and builds its Rust extension modules. [rust/rust-toolchain.toml](https://github.com/sgl-project/sglang/blob/92e894ef904bfdb5c0cf2f2a541cbd4eec023573/rust/rust-toolchain.toml) requests **1.92**; the installer fixes **1.92.0**. Keep the image pipeline rather than disabling Rust to make installation appear successful.
2. The main CUDA kernel distribution has a published ARM wheel; no full kernel source build is planned. A missing/incompatible wheel is a stop condition, not permission to silently launch a large build.
3. Model-specific relative attention, loop operations, routed quantization, FlashInfer CUTLASS/CuTe expert kernels, Triton, and FA4 still perform **lazy JIT** during the first future start. Published wheels do not remove that cold-start work. Separate `flashinfer-cubin==0.7.4` and `flashinfer-jit-cache==0.7.4` releases were not found on PyPI; the PR Dockerfile defaults `INSTALL_FLASHINFER_JIT_CACHE=0`. Neither is required to install this recipe.
4. Set **MAX_JOBS=2**, **CARGO_BUILD_JOBS=2**, **CMAKE_BUILD_PARALLEL_LEVEL=2**, **NVCC_THREADS=1**, **OMP_NUM_THREADS=2**. Only consider MAX_JOBS=4 when otherwise idle. Preserve those limits for the first launch too. They limit parallel compilation, not the memory used by a single compiler invocation. A previous cold JIT OOM is accepted as ground truth; it was not reproduced.

## Deferred installation procedure

When the benchmark owner releases the machine, copy this script there and run:

```bash
bash install.sh
```

The script re-execs itself with `nice -n 19 ionice -c 3`, refuses to mix with an existing venv, creates the named venv, isolates pip/Rust/build/JIT caches **inside it**, pins and checks out the exact source commit, installs binary dependencies from the PR metadata, and builds only the main source package. It never launches a model. It does not modify the system toolkit, driver, service files, other venvs, or shell startup files. Models remain separate read-only inputs under `~/models`.

Build-tool pins are pip 26.3, setuptools 84.2.1, wheel 0.48.0, setuptools-rust 1.13.0, setuptools-scm 10.3.4, build 1.6.1, ninja 1.13.2, and cmake 4.4.5. Rustup 1.29.0 is downloaded with its SHA256 sidecar. All state is under the venv and its caches; the script uses the existing system compiler as a read-only prerequisite. Dependency installation is binary-only and intentionally stops if the resolver cannot satisfy the PR without source-building a dependency. No fallback installs into an old environment.

No installation or script checks were run in this preparation task. Main should perform shell syntax review, dependency resolution/import qualification, then one serialized cold start only after the machine is released. Treat the exact PR's dependencies as the baseline; do not copy dependencies from older released SGLang installations.

## Exact Spark cookbook commands

Source: [official cookbook](https://docs.sglang.io/cookbook/autoregressive/ThinkingMachines/Inkling-2-Mini), including the embedded command-panel cells, captured 2026-11-20. These are the cookbook flags, with its host/port placeholders expressed as shell variables. Set `BIND_ADDRESS` and `PORT` locally; no machine addresses are recorded here. These are **documentation only**, not commands run by `install.sh`.

Balanced:

```bash
MAX_JOBS=4 sglang serve --model-path thinkingmachines/Inkling-2-Mini --tp 1 --ep 1 --attention-backend fa4 --dtype bfloat16 --page-size 128 --mem-fraction-static 0.45 --grammar-backend xgrammar --enable-multimodal --reasoning-parser inkling --tool-call-parser inkling2 --host "$BIND_ADDRESS" --port "$PORT"
```

DSpark:

```bash
MAX_JOBS=4 SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL=1 SGLANG_OPT_DSV41_DSPARK_MID_FOLD=1 SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE=1 SGLANG_OPT_DSV41_DSPARK_SKIP_PREFILLED_INPUTS=1 sglang serve --model-path thinkingmachines/Inkling-2-Mini --tp 1 --ep 1 --attention-backend fa4 --dtype bfloat16 --page-size 128 --speculative-algorithm DSPARK --speculative-num-draft-tokens 8 --mem-fraction-static 0.45 --grammar-backend xgrammar --enable-multimodal --reasoning-parser inkling --tool-call-parser inkling2 --host "$BIND_ADDRESS" --port "$PORT"
```

For a **pinned local** run, first source `~/sglang-inkling-pr51926-venv/bin/inkling-env`, substitute `--model-path "$HOME/models/Inkling-2-Mini-6e7c8248f75265bdd4fbab5273304243b5d8c2cb"`, and change the command's `MAX_JOBS=4` to **2** for the first cold start. Those are deliberate deviations from the printed cookbook. The draft is bundled in that same directory; do not download another draft model or force `--quantization modelopt` based on the stub in config.json.

The cookbook does not cap Spark's context or max-running-requests in these two cells. Its memory fraction is an engine recipe, not yet a validated CapyCTL memory policy. No throughput, image correctness, parking, or end-to-end compatibility is claimed by this plan.
