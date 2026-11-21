#!/usr/bin/env python3
"""Verify an existing target against the independently downloaded pinned record."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
reference = json.loads(a.manifest.read_text())['nvidia']
records = []
for expected in reference['files']:
    path = a.checkpoint/expected['file']
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    actual = {'file': expected['file'], 'size_bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
    if actual['size_bytes'] != expected['size_bytes'] or actual['sha256'] != expected['sha256']:
        raise ValueError('existing checkpoint differs: '+expected['file'])
    records.append(actual)
result = {'repo': reference['repo'], 'revision': reference['revision'], 'checkpoint': str(a.checkpoint),
          'all_file_hashes_match_independent_download': True, 'files': records}
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({'verified_files': len(records), 'bytes': sum(f['size_bytes'] for f in records)}), flush=True)
