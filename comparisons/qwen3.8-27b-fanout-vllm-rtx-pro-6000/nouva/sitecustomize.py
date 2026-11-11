# Test hook, outside the venv: treat UVA as unavailable (Confidential Computing guests
# cannot let the GPU read host memory mapped through UVA).
try:
    import vllm.utils.platform_utils as _p
    _p.is_uva_available = lambda: False
except Exception:
    pass
