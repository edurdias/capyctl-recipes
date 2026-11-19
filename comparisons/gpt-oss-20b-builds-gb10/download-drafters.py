#!/usr/bin/env python3
"""Download and hash the two pinned public drafters using an existing Hub client."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from huggingface_hub import snapshot_download

PINS = {
    'dflash': ('z-lab/gpt-oss-20b-DFlash', 'd53f6551543204c859e8bbaaddbd15d11b447af9'),
    'nebius-eagle3': ('nebius/EAGLE3-gpt-oss-20b', '9dd45bb1da8b1ddc8cabd52691d9ac170b41484f'),
}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--models', type=Path, default=Path.home()/'models')
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
result = {}
for name, (repo, revision) in PINS.items():
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true') as response:
        metadata = json.load(response)
    if metadata['sha'] != revision:
        raise ValueError('model metadata revision differs')
    files = [entry for entry in metadata['siblings'] if entry['rfilename'].endswith(('.json', '.safetensors'))]
    target = args.models/f'gpt-oss-20b-{name}'
    snapshot_download(repo, revision=revision, local_dir=target, allow_patterns=[entry['rfilename'] for entry in files])
    audit = []
    for entry in files:
        path = target/entry['rfilename']
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(block)
        sha = digest.hexdigest()
        expected = entry.get('lfs', {}).get('sha256')
        if path.stat().st_size != entry['size'] or (expected and sha != expected):
            raise ValueError(f'hash/size mismatch: {name}/{entry["rfilename"]}')
        audit.append({'file': entry['rfilename'], 'size_bytes': path.stat().st_size, 'sha256': sha, 'hub_lfs_sha256': expected})
    result[name] = {'repo': repo, 'revision': revision, 'path': str(target), 'files': audit}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({name: {'revision': record['revision'], 'files': len(record['files']), 'bytes': sum(f['size_bytes'] for f in record['files'])} for name, record in result.items()}))
