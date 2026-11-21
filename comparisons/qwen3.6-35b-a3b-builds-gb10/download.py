#!/usr/bin/env python3
"""Download exact target/drafter revisions with size and LFS SHA-256 checks."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from huggingface_hub import snapshot_download

PINS = {
    'nvidia': ('nvidia/Qwen3.6-35B-A3B-NVFP4', '1355db6a052410cfd62085d94b58866fd0f2c3c5'),
    'dflash2': ('incoai/Qwen3.6-35B-A3B-DFlash2', '51ef7b6923ad6c14cb1bb41c37a9041446496ab6'),
    'dspark': ('RedHatAI/Qwen3.6-35B-A3B-speculator.dspark', '53814b238c3a6ce5f332066a6bedb9d179777ede'),
    'redhat': ('RedHatAI/Qwen3.6-35B-A3B-NVFP4', 'e94a930c8e9b8a7d3763e449289c261c9c7d3e5e'),
    'unsloth': ('unsloth/Qwen3.6-35B-A3B-NVFP4', '739af1e7aac320af1682ed1e0cce369af4c5265d'),
    'unsloth-fast': ('unsloth/Qwen3.6-35B-A3B-NVFP4-Fast', '1c3f884bc99aac2524f6d49bcbac8c88401afd66'),
    'qwen-fp8': ('Qwen/Qwen3.6-35B-A3B-FP8', 'c653bba46dfa06e9658dcc0b0dc87fe4685f14a2'),
    'intel-mixed': ('intel-ai/Qwen3.6-35B-A3B-MXFP4-Mixed-CT-AutoRound', 'fd91c11d638beb5f877f9a6de060678957b93d37'),
    'gguf-mxfp4': ('unsloth/Qwen3.6-35B-A3B-GGUF', 'a483e9e6cbd595906af30beda3187c2663a1118c'),
}
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--models', type=Path, default=Path.home()/'models')
p.add_argument('--out', type=Path, required=True)
p.add_argument('--only', nargs='+', choices=PINS, default=['nvidia', 'dflash2', 'dspark'])
a = p.parse_args()
result = {}
for name, (repo, revision) in PINS.items():
    if name not in a.only:
        continue
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true') as response:
        metadata = json.load(response)
    if metadata['sha'] != revision:
        raise ValueError('revision mismatch')
    if name == 'gguf-mxfp4':
        files = [f for f in metadata['siblings'] if f['rfilename'] == 'Qwen3.6-35B-A3B-MXFP4_MOE.gguf']
    else:
        files = [f for f in metadata['siblings'] if f['rfilename'].endswith(('.json', '.safetensors', '.jinja', '.txt', '.model'))]
    target = a.models/f'qwen36-{name}-{revision[:12]}'
    snapshot_download(repo, revision=revision, local_dir=target, allow_patterns=[f['rfilename'] for f in files])
    audit = []
    for f in files:
        path = target/f['rfilename']
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(block)
        sha = digest.hexdigest()
        expected = f.get('lfs', {}).get('sha256')
        if path.stat().st_size != f['size'] or (expected and expected != sha):
            raise ValueError(f'file mismatch: {name}/{f["rfilename"]}')
        audit.append({'file': f['rfilename'], 'size_bytes': path.stat().st_size, 'sha256': sha, 'hub_lfs_sha256': expected})
    result[name] = {'repo': repo, 'revision': revision, 'path': str(target), 'files': audit}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'downloaded': name, 'revision': revision, 'bytes': sum(f['size_bytes'] for f in audit)}), flush=True)
