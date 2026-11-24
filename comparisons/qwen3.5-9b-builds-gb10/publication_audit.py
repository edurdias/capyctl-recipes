#!/usr/bin/env python3
"""Archive and sanitize once with --prepare, or refresh the publication inventory.

No inference or network access. Original evidence stays outside the repository.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys

HERE = Path(__file__).resolve().parent
ARCHIVE = Path('/var/tmp/capyctl-webui/qwen35/comparison-archive')
ORIGINAL = ARCHIVE / 'comparison-original'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def scrub(text):
    text = re.sub(r'/home/[^/\s\"\'\\]+', '/home/me', text)
    text = re.sub(r'\bhost [AB]\b', 'GPU owner', text)
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', 'localhost', text)
    return text


def numbers(value):
    if isinstance(value, dict):
        return {k: numbers(v) for k, v in value.items() if numbers(v) is not None}
    if isinstance(value, list):
        return [numbers(v) for v in value if numbers(v) is not None]
    if isinstance(value, (int, float, bool)):
        return value
    return None


def input_manifest():
    path = HERE.parents[1] / 'tools/capyctl-bench/capyctl_bench.py'
    spec = importlib.util.spec_from_file_location('input_bench', path)
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    entries = []
    for filename in ('g32-speed.json', 'incumbent-speed.json', 'c-int8head-speed.json'):
        data = json.loads((HERE / 'bench' / filename).read_text())
        for point in data['context_sweep']['points']:
            for index, request in enumerate(point['requests']):
                seed = point['target_prompt_tokens'] * 100 + index + 1
                content = helper.context_messages(point['sentences'], seed)[0]['content']
                entries.append({'evidence': 'bench/' + filename, 'target_prompt_tokens': point['target_prompt_tokens'],
                                'actual_prompt_tokens': request['prompt_tokens'], 'sentences': point['sentences'],
                                'seed': seed, 'generator': 'tools/capyctl-bench/capyctl_bench.py:context_messages',
                                'utf8_bytes': len(content.encode()), 'content_sha256': sha(content.encode())})
    return {'schema_version': 1, 'gb10': entries,
            'laptop': {'manifest': 'bench/laptop-input-manifest.json',
                       'generator': 'bench/laptop-run.py:speed', 'seed': None,
                       'algorithm': "Binary search the largest n whose exact pinned chat token count is at most target, then use ' data' * n + '\\nSummarize the text.'",
                       'targets': [8000, 27000, 29000, 30000],
                       'verification': 'Offline pinned tokenizers; accept hashes only where regenerated count equals successful native usage. Rejected and protocol-failed rows remain unavailable.'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        if ORIGINAL.exists():
            raise SystemExit('Original archive already exists; refusing to overwrite evidence.')
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        shutil.copytree(HERE, ORIGINAL)
        shutil.copy2(HERE / 'NOTES.md', ARCHIVE / 'NOTES.md')
        (HERE / 'NOTES.md').unlink()
        for path in HERE.rglob('*'):
            if path.is_file() and path.name != 'publication_audit.py':
                text = path.read_text()
                cleaned = scrub(text)
                if cleaned != text:
                    path.write_text(cleaned)
        (HERE / 'bench/input-manifest.json').write_text(json.dumps(input_manifest(), indent=2) + '\n')
    inventory = []
    unresolved = []
    for path in sorted(HERE.rglob('*')):
        if not path.is_file() or path.name == 'publication-audit.json':
            continue
        rel = path.relative_to(HERE)
        raw = path.read_bytes()
        text = raw.decode()
        original = ORIGINAL / rel
        record = {'path': str(rel), 'bytes': len(raw), 'sha256': sha(raw)}
        if original.exists():
            old = original.read_bytes()
            record['original_sha256'] = sha(old)
            record['changed'] = old != raw
            if path.suffix in ('.json', '.jsonl'):
                def load(s):
                    return [json.loads(line) for line in s.splitlines()] if path.suffix == '.jsonl' else json.loads(s)
                record['numeric_values_unchanged'] = numbers(load(old.decode())) == numbers(load(text))
                if not record['numeric_values_unchanged']:
                    unresolved.append(str(rel) + ': numeric JSON changed')
            else:
                record['matches_documented_scrub'] = raw == scrub(old.decode()).encode()
        if path.name != 'publication_audit.py' and scrub(text) != text:
            unresolved.append(str(rel) + ': unsanitized identity')
        if path.name == 'NOTES.md' or path.suffix == '.pyc' or '__pycache__' in path.parts:
            unresolved.append(str(rel) + ': private notes or cache')
        if re.search(r'\bhf_[A-Za-z0-9]{20,}\b|\bsk-[A-Za-z0-9]{20,}\b|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----', text):
            unresolved.append(str(rel) + ': credential-like text')
        inventory.append(record)
    audit = {'status': 'pass' if not unresolved else 'fail',
             'archive': str(ARCHIVE), 'notes_archived': (ARCHIVE / 'NOTES.md').exists(),
             'policy': ['User paths normalized to /home/me; runtime addresses and GPU-owner identities removed.',
                        'Numeric JSON values unchanged, including failed attempts, timing and scores.',
                        'Original input/checkpoint hashes retained; those hashes describe original bytes, not scrubbed text.',
                        'Generated model outputs and calibration conversations retained; credential-like strings checked.',
                        'Long inputs represented by procedural seed/hash references where originals are reconstructible; missing laptop provenance is explicit.'],
             'unresolved': unresolved, 'files': inventory}
    (HERE / 'publication-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({'status': audit['status'], 'files': len(inventory), 'unresolved': unresolved}))
    if unresolved:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
