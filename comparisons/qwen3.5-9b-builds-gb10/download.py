#!/usr/bin/env python3
"""Download pinned model inputs and verify Hub SHA256 / Git blob identities."""
import hashlib
import json
import urllib.request
from pathlib import Path
from huggingface_hub import snapshot_download

PINS = {
    'source': ('Qwen/Qwen3.5-9B', 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'),
    'baseline': ('cyankiwi/Qwen3.5-9B-AWQ-4bit', 'e4d2d39cede8c5d1716d824cd843e2889c648083'),
}

def main():
    results = {}
    for name, (repo, revision) in PINS.items():
        with urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true') as response:
            metadata = json.load(response)
        assert metadata['sha'] == revision
        files = [f for f in metadata['siblings'] if f['rfilename'].endswith(('.json', '.safetensors', '.jinja', '.txt', '.model'))]
        target = Path.home() / 'models' / f'qwen35-{name}-{revision[:12]}'
        snapshot_download(repo, revision=revision, local_dir=target, allow_patterns=[f['rfilename'] for f in files], max_workers=2)
        audit = []
        for entry in files:
            path = target / entry['rfilename']
            size = path.stat().st_size
            sha = hashlib.sha256()
            git = hashlib.sha1(f'blob {size}\0'.encode())
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    sha.update(block)
                    git.update(block)
            expected = entry.get('lfs', {}).get('sha256')
            assert size == entry['size'], path
            assert sha.hexdigest() == expected if expected else git.hexdigest() == entry['blobId'], path
            audit.append({'file': entry['rfilename'], 'size_bytes': size, 'sha256': sha.hexdigest(), 'hub_lfs_sha256': expected, 'git_blob': entry['blobId']})
        results[name] = {'repo': repo, 'revision': revision, 'directory': target.name, 'files': audit}
        (Path(__file__).parent / 'download-manifest.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps({'verified': name, 'bytes': sum(f['size_bytes'] for f in audit)}), flush=True)

if __name__ == '__main__':
    main()
