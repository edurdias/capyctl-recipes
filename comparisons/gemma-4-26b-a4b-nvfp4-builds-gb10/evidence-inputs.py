#!/usr/bin/env python3
"""Regenerate archived benchmark inputs, checking every content SHA-256.

Verify all inputs: python3 evidence-inputs.py --verify
Hydrate a compact fixture/run: python3 evidence-inputs.py --source FILE --out FILE
Write one original input: python3 evidence-inputs.py --sha256 HASH --out FILE
No tokenizer, model weights, network, or GPU is required. Character offsets retain
exact token-sized boundaries chosen by the original tokenizer-based generators.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / 'bench/reference-data/input-manifest.json'


def benchmark():
    path = HERE.parents[1] / 'tools/capyctl-bench/capyctl_bench.py'
    spec = importlib.util.spec_from_file_location('capyctl_bench_evidence', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def regenerate(entry, helper):
    filler = helper.filler_text(entry['sentences'], entry['seed'])
    text = ''.join(filler[part['filler'][0]:part['filler'][1]]
                   if isinstance(part, dict) else part for part in entry['parts'])
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    if digest != entry['sha256']:
        raise ValueError('Regenerated input SHA-256 mismatch: ' + entry['sha256'])
    return text


def hydrate(value, entries, helper, cache):
    if isinstance(value, dict):
        if set(value) == {'input_sha256', 'input_reference'}:
            digest = value['input_sha256']
            if digest not in cache:
                cache[digest] = regenerate(entries[digest], helper)
            return cache[digest]
        return {key: hydrate(item, entries, helper, cache) for key, item in value.items()}
    if isinstance(value, list):
        return [hydrate(item, entries, helper, cache) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--verify', action='store_true')
    operation.add_argument('--source', type=Path)
    operation.add_argument('--sha256')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if not args.verify and not args.out:
        parser.error('--out is required when writing an input or hydrated JSON')
    manifest = json.loads(MANIFEST.read_text())
    entries = manifest['inputs']
    helper = benchmark()
    if hashlib.sha256(helper.filler_text(32, 12345).encode()).hexdigest() != manifest['filler_selftest_sha256']:
        raise ValueError('Canonical filler generator changed')
    if args.verify:
        for entry in entries.values():
            regenerate(entry, helper)
        print(json.dumps({'verified_inputs': len(entries), 'sha256_matches': len(entries)}))
    elif args.sha256:
        args.out.write_text(regenerate(entries[args.sha256], helper), encoding='utf-8')
    else:
        data = hydrate(json.loads(args.source.read_text()), entries, helper, {})
        relative = args.source.resolve().relative_to(HERE).as_posix()
        original = manifest['source_files'].get(relative)
        formatting = original['json_format'] if original else {'indent': 2, 'ensure_ascii': False}
        args.out.write_text(json.dumps(data, **formatting) + '\n')


if __name__ == '__main__':
    main()
