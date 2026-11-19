#!/usr/bin/env python3
"""Regenerate compact benchmark inputs and verify their original UTF-8 hashes.

Verify all inputs: python3 evidence-inputs.py --verify
Hydrate a compact run: python3 evidence-inputs.py --source FILE --out FILE
Write one input: python3 evidence-inputs.py --sha256 HASH --out FILE
No tokenizer, weights, network or GPU is required. Character boundaries preserve
exact token-sized documents from the original tokenizer-based preparation.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / 'bench/final-checks/input-manifest.json'


def benchmark():
    path = HERE.parents[1] / 'tools/capyctl-bench/capyctl_bench.py'
    spec = importlib.util.spec_from_file_location('capyctl_bench_evidence', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def describe(text, metadata, helper):
    """Keep the seed, filler slices and literal insertions, never a duplicate body."""
    seed = metadata['filler_seed']
    kind = metadata.get('kind', 'retrieval')
    sentences = 20000 if kind == 'decode' else 1000 if kind == 'copy' else 16000
    filler = helper.filler_text(sentences, seed)
    marker = 'BEGIN DOCUMENT\n'
    start = text.index(marker) + len(marker)
    end = text.index('\nEND DOCUMENT', start)
    parts, cursor = [text[:start]], 0
    for chunk in re.split(r'(\n(?:Retrieval record|Code): [^\n]+\n)', text[start:end]):
        if not chunk:
            continue
        if re.fullmatch(r'\n(?:Retrieval record|Code): [^\n]+\n', chunk):
            parts.append(chunk)
            continue
        size = 0
        while size < len(chunk) and cursor + size < len(filler) and chunk[size] == filler[cursor + size]:
            size += 1
        if size:
            parts.append({'filler': [cursor, cursor + size]})
            cursor += size
        if size < len(chunk):
            parts.append(chunk[size:])
    parts.append(text[end:])
    entry = {**metadata, 'sha256': hashlib.sha256(text.encode()).hexdigest(),
             'seed': seed, 'sentences': sentences, 'parts': parts,
             'utf8_bytes': len(text.encode())}
    if regenerate(entry, helper) != text:
        raise ValueError('Input descriptor does not reproduce original bytes')
    return entry


def regenerate(entry, helper):
    filler = helper.filler_text(entry['sentences'], entry['seed'])
    text = ''.join(filler[part['filler'][0]:part['filler'][1]]
                   if isinstance(part, dict) else part for part in entry['parts'])
    if hashlib.sha256(text.encode()).hexdigest() != entry['sha256']:
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
        parser.error('--out is required when writing regenerated data')
    manifest = json.loads(MANIFEST.read_text())
    entries, helper = manifest['inputs'], benchmark()
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
        original = manifest.get('source_files', {}).get(relative)
        formatting = original['json_format'] if original else {'indent': 2, 'ensure_ascii': False}
        args.out.write_text(json.dumps(data, **formatting) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
