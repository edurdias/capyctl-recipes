#!/usr/bin/env python3
"""Freeze six or twelve deterministic retrieval documents at requested lengths."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--existing', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lengths', default='111000,32000,64000,96000')
    parser.add_argument('--documents', type=int, choices=(6, 12), default=6)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('gate', Path(__file__).with_name('acceptance-gates.py'))
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    benchmark = gate.load_benchmark()
    existing = json.loads(args.existing.read_text())['documents']
    configurations = [(d['probe_id'], d['filler_seed'], tuple(i['target_depth_fraction'] for i in d['insertions']), d['expected_codes']) for d in existing]
    additions = [(111003, (.05, .35, .95)), (111004, (.20, .60, .90)), (111005, (.02, .48, .70)), (111006, (.25, .55, .98)), (111007, (.08, .45, .92)), (111008, (.12, .52, .88)), (111009, (.18, .38, .78)), (111010, (.04, .62, .96)), (111011, (.28, .58, .82)), (111012, (.22, .42, .72))]
    for index, (seed, depths) in enumerate(additions[:args.documents - 2], 3):
        rng = random.Random(seed)
        codes = ['GATE' + format(rng.getrandbits(64), '016X') for _ in range(3)]
        configurations.append((f'document-{index}', seed, depths, codes))
    corpus_name = 'retrieval-six' if args.documents == 6 else 'retrieval-twelve'
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {'method': f'{args.documents} fixed documents; original two preserved verbatim at 111000 tokens; seeded canonical pseudo-English filler and seeded codes for additions; shorter lengths regenerate identical seeds/codes/depth fractions.', 'source_sha256': hashlib.sha256(args.existing.read_bytes()).hexdigest(), 'lengths': {}}
    for length in map(int, args.lengths.split(',')):
        documents = []
        for index, (name, seed, depths, codes) in enumerate(configurations):
            if length == 111000 and index < len(existing):
                document = existing[index]
            else:
                gate.TARGET_TOKENS = length
                gate.DEPTHS = depths
                code_iterator = iter(codes)
                gate.fresh_code = lambda: next(code_iterator)
                document = gate.prepare_retrieval(tokenizer, benchmark, name, seed)
            document = {**document, 'target_prompt_tokens': length}
            documents.append(document)
        path = args.out / f'{corpus_name}-{length}.json'
        path.write_text(json.dumps({'documents': documents}, indent=2) + '\n')
        manifest['lengths'][str(length)] = {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'documents': [{key: value for key, value in document.items() if key != 'messages'} for document in documents]}
        print(json.dumps({'length': length, 'documents': len(documents), 'sha256': manifest['lengths'][str(length)]['sha256']}), flush=True)
    (args.out / f'{corpus_name}-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
