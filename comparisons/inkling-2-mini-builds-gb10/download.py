#!/usr/bin/env python3
"""Download and hash the pinned native checkpoint without installing packages."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

REPO = 'thinkingmachines/Inkling-2-Mini'
REVISION = 'ed40e7da625f594fc9c4990f5f66be24873c22d7'
target = Path.home()/'models/Inkling-2-Mini-ed40e7da'
target.mkdir(parents=True, exist_ok=True)
with urllib.request.urlopen(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true', timeout=120) as response:
    metadata = json.load(response)
assert metadata['sha'] == REVISION
(target/'download-manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
files = []
for item in metadata['siblings']:
    name = item['rfilename']
    path = target/name
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = item.get('lfs', {}).get('sha256')
    def digest(p):
        h = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024), b''):
                h.update(block)
        return h.hexdigest()
    if not path.exists() or path.stat().st_size != item['size'] or (expected and digest(path) != expected):
        partial = path.with_name(path.name+'.partial')
        with urllib.request.urlopen(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}', timeout=600) as response, partial.open('wb') as out:
            size, last = 0, time.monotonic()
            while block := response.read(8*1024*1024):
                out.write(block)
                size += len(block)
                if time.monotonic()-last > 30:
                    print(json.dumps({'file': name, 'bytes': size}), flush=True)
                    last = time.monotonic()
        assert partial.stat().st_size == item['size'], name
        if expected:
            assert digest(partial) == expected, name
        partial.replace(path)
    actual = digest(path)
    assert not expected or actual == expected, name
    files.append({'file': name, 'bytes': path.stat().st_size, 'sha256': actual, 'hub_lfs_sha256': expected})
    print(json.dumps(files[-1]), flush=True)
(target/'REVISION').write_text(REVISION+'\n')
receipt = {'repo': REPO, 'revision': REVISION, 'verified': True, 'files': files}
(target/'download-verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps({'complete': True, 'bytes': sum(f['bytes'] for f in files)}), flush=True)
