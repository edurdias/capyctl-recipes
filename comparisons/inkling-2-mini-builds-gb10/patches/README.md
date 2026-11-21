# FP8 attention patch

Base: SGLang PR 51926, `92e894ef904bfdb5c0cf2f2a541cbd4eec023573`.
The diff is restricted to the approved isolated environment. Apply with
`patch -p1` at its pinned source root and `patch -p2` at its installed
`lib/python3.12/site-packages` root. Do not install it into another environment.

The explicit `fp8` precision group accepts only `wq_du`, `wk_dv`, `wv_dv`,
`wr_du`, `gate_attn`, and `wo_ud` weights in the 24 target attention layers.
Weights are E4M3FN matrices; each `.weight_scale` is FP32 `[output_channels, 1]`
and multiplies the corresponding quantized weight row. Activations use the
existing dynamic per-token FP8 method, returning BF16 to unchanged attention.
The existing compressed-tensors FP8 linear method owns allocation, loading,
post-load transposition, and kernel dispatch. The small sink projection is
split into a separate BF16 linear. The head, experts, routers, loop controllers,
sinks, convolutions, relative-logit basis and normalization tensors are unchanged.
Original precision metadata continues to select the original merged BF16 path.
Exact dtype, unknown-tensor, duplicate-tensor, and shard-completeness checks remain.

`../convert-fp8-attention.py` reads the pinned native checkpoint and writes the
selective checkpoint, precision metadata and a per-tensor SHA-256 manifest.
The old top-level quantization stub is removed: adding a bare `inkling2` stub
would shadow `hf_quant_config.json` and omit its required precision groups.
Untouched tensor payloads are copied directly from source bytes and independently
read back from the output; the bundled draft is copied and hashed unchanged.
The manifest's bytes/token are logical shape-derived traffic, not measured DRAM
traffic or a speed prediction. The converter uses round-to-nearest E4M3 with
per-row maximum absolute value scaling, not calibrated quality optimization.

`../run-fp8-attention.py` owns one loopback engine plus its 3 GiB memory watcher;
`../fp8-attention-check.py` records a 50-prompt, 64-token comparison and preliminary
c1/c8 plus exact-32,000-input-token timings. These short tests do not qualify the
checkpoint. Measured dispatch and results are recorded in the dedicated section
of `../NOTES.md` and retained run receipts.

`../test-fp8-attention.py` checks the precision contract against the real original
and exported metadata: native precisions stay identical, FP8 tensors/scales have
the declared dtypes, and the head, sinks and unsupported scale layouts are
rejected. Run it with the approved venv Python after applying the patch.

The exporter rebuilds `model.safetensors.index.json` from the written main-shard
header; it never copies the original index. The map includes every actual tensor,
including all 144 new FP8 scales, and no extra name. `metadata.total_size` is the
sum of actual tensor payload sizes; other original index metadata is preserved.
The conversion manifest records the index hash, byte size and exact header
coverage. `--rebuild-index-only` repairs this metadata on an existing export
without rewriting any tensor file.

The index regression uses a complete tiny 24-layer export, then a metadata-only
repair. It checks exact names/file mapping, total payload size, preserved metadata,
the manifest hash and unchanged tensor bytes. This does not widen the approved
three-file engine patch.
