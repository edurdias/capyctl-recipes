#!/usr/bin/env python3
"""Export only public final-check evidence, replacing document text with seed/hash references."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--run', action='append', required=True, help='label=runtime directory')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    fixtures = json.loads(args.fixtures.read_text())
    spec = importlib.util.spec_from_file_location('evidence_inputs', Path(__file__).with_name('evidence-inputs.py'))
    evidence = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evidence)
    helper = evidence.benchmark()
    inputs = {}
    def register(content, metadata):
        digest = hashlib.sha256(content.encode()).hexdigest()
        inputs[digest] = evidence.describe(content, metadata, helper)
        return {'input_sha256': digest, 'input_reference': 'input-manifest.json#/inputs/' + digest}
    for fixture in fixtures['retrieval']:
        content = fixture['messages'][0]['content']
        metadata = {k: v for k, v in fixture.items() if k != 'messages'}
        register(content, {**metadata, 'variant': 'original'})
        document, _ = content.rsplit('\nEND DOCUMENT\n\n', 1)
        register(document + '\nEND DOCUMENT\n\nRequest: return exactly one line containing the three retrieval record codes in document order, separated by single spaces. Do not include any explanation.', {**metadata, 'variant': 'one-line'})
    for length, fixture in fixtures['decode'].items():
        register(fixture['messages'][0]['content'], {k: v for k, v in fixture.items() if k != 'messages'} | {'kind': 'decode', 'target_prompt_tokens': int(length)})
    for fixture in fixtures['copy']:
        register(fixture['messages'][0]['content'], {k: v for k, v in fixture.items() if k != 'messages'} | {'kind': 'copy', 'target_prompt_tokens': 2000})
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if k not in ('env',)}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, str):
            digest = hashlib.sha256(value.encode()).hexdigest()
            if digest in inputs:
                return {'input_sha256': digest, 'input_reference': '../input-manifest.json#/inputs/' + digest}
            return value
        return value
    args.out.mkdir(parents=True, exist_ok=True)
    for run in args.run:
        label, directory = run.split('=', 1)
        directory = Path(directory).expanduser()
        out = args.out / label
        out.mkdir(parents=True, exist_ok=True)
        for name in ('run.json', 'measure.json', 'acceptance.json', 'native-kv.json', 'memory.json'):
            source = directory / name
            if source.exists():
                (out / name).write_text(json.dumps(clean(json.loads(source.read_text())), indent=2) + '\n')
        source = directory / 'engine-evidence.txt'
        if source.exists():
            lines = [line for line in source.read_text().splitlines() if not re.search(r'api.key|authorization|credentials|secret', line, re.I)]
            (out / source.name).write_text('\n'.join(lines) + '\n')
    manifest = {
        'method': 'Exact canonical filler, original character boundaries, record insertions and tokenizer-selected padding; hydrate with evidence-inputs.py.',
        'filler_selftest_sha256': hashlib.sha256(helper.filler_text(32, 12345).encode()).hexdigest(),
        'inputs': inputs,
    }
    (args.out / 'input-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'exported_runs': len(args.run), 'input_hashes': len(inputs)}))


if __name__ == '__main__':
    main()
