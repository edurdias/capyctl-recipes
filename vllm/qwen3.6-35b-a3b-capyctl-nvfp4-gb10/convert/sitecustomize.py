"""Read-only per-layer execution diagnostics for cleared load-only checks.

Loaded only when QWEN36_LOAD_ENGINE is explicitly set. No installed package is
changed. Log after successful native execution, including packed-weight dtype.
"""
import functools
import inspect
import json
import os

engine = os.environ.get("QWEN36_LOAD_ENGINE")
if engine in ("vllm", "sglang"):
    if engine == "vllm":
        from vllm.model_executor.layers.quantization import modelopt as quant
        names = ["ModelOptLinearMethod"]
    else:
        from sglang.srt.layers.quantization import modelopt_quant as quant
        names = ["ModelOptNvFp4A16LinearMethod"]
    for cls in [c for _,c in inspect.getmembers(quant, inspect.isclass) if c.__module__ == quant.__name__]:
        if "get_quant_method" not in cls.__dict__:
            continue
        original = cls.get_quant_method
        @functools.wraps(original)
        def get_method(self, layer, prefix, *args, _original=original, **kwargs):
            result = _original(self, layer, prefix, *args, **kwargs)
            if result is not None:
                result._qwen36_prefix = prefix
            return result
        cls.get_quant_method = get_method
    for name in names:
        cls = getattr(quant, name)
        original = cls.apply
        @functools.wraps(original)
        def apply(self, layer, *args, _original=original, **kwargs):
            result = _original(self, layer, *args, **kwargs)
            if not getattr(self, "_qwen36_logged", False):
                self._qwen36_logged = True
                print("QWEN36-NATIVE-DENSE " + json.dumps({
                    "engine": engine, "prefix": getattr(self, "_qwen36_prefix", getattr(layer, "prefix", None)),
                    "method": type(self).__name__, "spec": str(getattr(self, "spec", "W4A16_NVFP4")),
                    "kernel": type(self.kernel).__name__ if hasattr(self, "kernel") else "apply_fp4_marlin_linear",
                    "weight_dtype": str(layer.weight.dtype), "weight_shape": list(layer.weight.shape),
                    "is_w4a16_nvfp4": getattr(layer, "is_w4a16_nvfp4", engine == "sglang"),
                    "successful_apply": True}), flush=True)
            return result
        cls.apply = apply
