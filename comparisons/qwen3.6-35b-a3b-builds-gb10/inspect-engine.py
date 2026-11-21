#!/usr/bin/env python3
"""Record non-secret package versions and installed Qwen kernel-source hashes."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys

result = {'packages': {}, 'sources': {}}
for name in ('sglang', 'vllm', 'torch', 'flashinfer-python', 'transformers', 'lmcache', 'datasets'):
    try:
        result['packages'][name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        pass
names = ['sglang/srt/models/qwen3_5_moe.py', 'sglang/srt/models/qwen3_next.py',
         'sglang/srt/models/dflash.py', 'sglang/srt/models/dspark.py',
         'sglang/srt/layers/attention/flashinfer_backend.py', 'sglang/srt/layers/quantization/modelopt_quant.py',
         'vllm/config/speculative.py', 'vllm/model_executor/models/qwen3_5.py',
         'vllm/model_executor/models/qwen3_5_mtp.py', 'vllm/model_executor/models/qwen3_dflash2.py',
         'vllm/v1/attention/backends/flashinfer.py']
for root in (Path(p) for p in sys.path if p.endswith('site-packages')):
    for name in names:
        path = root/name
        if path.is_file():
            result['sources'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
result['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], text=True).strip()
print(json.dumps(result, indent=2))
