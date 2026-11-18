#!/usr/bin/env python3
"""Compare fixed long documents and prompt formats without changing their records."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipes', type=Path, help='shared native-launch contract; helpers resolve beside this script')
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--api-key-file', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, default=Path.home()/'gemma4-engine-search/reference-data/retrieval-diagnostic.json')
    parser.add_argument('--lengths', help='comma-separated corpus lengths beside --fixtures; run in the listed order')
    parser.add_argument('--variants', default='original,one-line')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--reasoning', action='store_true')
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('gate_helpers', Path(__file__).with_name('acceptance-gates.py'))
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    cb = gate.load_benchmark()
    client = gate.Client(args.endpoint, args.model, cb.read_api_key(str(args.api_key_file) if args.api_key_file else None))
    fixtures = json.loads(args.fixtures.read_text())
    variants = args.variants.split(',')
    if args.lengths:
        fixtures = {'documents': [document for length in args.lengths.split(',') for document in json.loads((args.fixtures.parent / f'retrieval-six-{int(length)}.json').read_text())['documents']]}
    if not set(variants) <= {'original', 'one-line'} or args.repeats < 1:
        parser.error('invalid variants or repeats')
    result = {'method': 'Fixed documents, temperature 0, thinking off; primary all three expected codes present in any order and total codes found; secondary exact code sequence', 'fixtures_file': args.fixtures.name, 'requests': [], 'completed': False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.out.write_text(json.dumps(client.public(result), indent=2, ensure_ascii=False)+'\n')
    save()
    for fixture in fixtures['documents']:
        for variant in variants:
            messages = copy.deepcopy(fixture['messages'])
            if variant == 'one-line':
                document, _ = messages[0]['content'].rsplit('\nEND DOCUMENT\n\n', 1)
                messages[0]['content'] = document + '\nEND DOCUMENT\n\nRequest: return exactly one line containing the three retrieval record codes in document order, separated by single spaces. Do not include any explanation.'
            for repeat in range(args.repeats):
                exchange = client.request({'messages': messages, 'max_tokens': 256, 'chat_template_kwargs': {'enable_thinking': False}})
                answer = gate.content_text(gate.assistant_message(exchange))
                codes = re.findall(gate.CODE_PATTERN, answer)
                expected = fixture['expected_codes']
                ok = gate.request_ok(exchange)
                result['requests'].append({'document': fixture['probe_id'], 'target_prompt_tokens': fixture.get('target_prompt_tokens', fixture['local_prompt_tokens']), 'variant': variant, 'repeat': repeat+1, 'expected_codes': expected, 'extracted_codes': codes, 'codes_found': sum(code in codes for code in expected) if ok else 0, 'answer': answer, 'all_codes_present': ok and all(code in codes for code in expected), 'exact_ordered_codes': ok and codes == expected, 'exchange': exchange})
                save()
    if args.reasoning:
        result['reasoning_8192'] = gate.reasoning_probe(client)
    result['completed'] = True
    result['all_exact_ordered'] = all(r['exact_ordered_codes'] for r in result['requests'])
    result['all_three_present'] = all(r['all_codes_present'] for r in result['requests'])
    save()
    print(json.dumps({'completed': True, 'all_three_present': result['all_three_present'], 'all_exact_ordered': result['all_exact_ordered'], 'requests': len(result['requests'])}))
    # A measured model failure is evidence, not a transport-client crash.
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
