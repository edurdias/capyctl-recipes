#!/usr/bin/env python3
"""Regenerate retained seed/hash inputs with the pinned CPU tokenizer; no engine test."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', required=True, type=Path)
    parser.add_argument('--originals', type=Path, help='optional archived input directory for whole-file byte comparisons')
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    cb = load('input_benchmark', ROOT/'tools/capyctl-bench/capyctl_bench.py')
    depth = load('input_depth', HERE.parent/'gemma-4-26b-a4b-nvfp4-builds-gb10/long-depth.py')
    verified = 0
    files = []
    manifests = [HERE/'bench/prepared-prompts.manifest.json', *sorted((HERE/'bench/runs').glob('*/context-prompts.manifest.json'))]
    for path in manifests:
        data = json.loads(path.read_text())
        for section in ('levels', 'long'):
            for rounds in data[section].values():
                for rows in rounds:
                    for row in rows:
                        messages, count = depth.prepare(tokenizer, cb, row['token_count'], int(row['seed']))
                        digest = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
                        assert digest == row['messages_sha256'], (path, row['id'])
                        assert count == row['token_count']
                        row['messages'] = messages
                        verified += 1
        # Restore original generator field order and JSON serialization.
        for section in ('levels', 'long'):
            for rounds in data[section].values():
                for rows in rounds:
                    for i, row in enumerate(rows):
                        rows[i] = {k: row[k] for k in ('id', 'messages', 'token_count', 'seed', 'messages_sha256')}
        encoded = (json.dumps(data, ensure_ascii=False)+'\n').encode()
        record = {'manifest': str(path.relative_to(HERE)), 'regenerated_sha256': hashlib.sha256(encoded).hexdigest()}
        if args.originals and path.parent.name != 'bench':
            original = args.originals/path.parent.name/'context-prompts.json'
            if original.exists():
                record['byte_identical'] = encoded == original.read_bytes()
                assert record['byte_identical'], original
        files.append(record)
    import sys
    sys.path.insert(0, str(HERE))
    probes = load('input_probes', HERE/'final-probes.py')
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp)/'fixtures.json'
        probes.prepare(SimpleNamespace(checkpoint=args.tokenizer, out=output))
        expected = json.loads((HERE/'bench/final-fixtures.manifest.json').read_text())
        actual = json.loads((output.with_name('fixtures.manifest.json')).read_text())
        assert actual == expected, 'final fixture manifest differs'
        verified += sum(len(actual[k]) for k in ('retrieval', 'copy'))
    print(json.dumps({'verified_inputs': verified, 'sha256_matches': verified, 'context_files': files, 'final_fixture_manifest_matches': True}, indent=2))


if __name__ == '__main__':
    main()
