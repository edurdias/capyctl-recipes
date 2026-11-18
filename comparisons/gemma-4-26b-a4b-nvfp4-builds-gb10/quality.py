#!/usr/bin/env python3
"""Describe greedy numerical variation and compare native task accuracy.

Divergence positions use retokenized visible content/reasoning, not private
engine token IDs. Exact-equality evidence is retained, never relabeled a match.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def load(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(data.get('summary', data), indent=2))


def identity(run):
    return {k: run.get(k) for k in ('key', 'checkpoint', 'settings', 'env', 'drafter_path')}


def compatible(reference, candidate):
    return (reference['key'].split('-')[0] == candidate['key'].split('-')[0]
            and reference['key'].split('-')[2] == 'none'
            and all(reference[k] == candidate[k] for k in ('checkpoint', 'settings', 'env')))


def divergence(args):
    from transformers import AutoTokenizer
    baseline = load(args.baseline)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    reference = baseline['guard']['greedy']['1']
    groups = [('baseline-c1-vs-c8', baseline['guard']['greedy']['8'])]
    if args.repeat:
        repeat = load(args.repeat)
        if repeat['prompt_sha256'] != baseline['prompt_sha256']:
            raise ValueError('repeat prompt corpus differs')
        groups.append(('baseline-c1-vs-repeat-c1', repeat['guard']['greedy']['1']))
    for path in args.candidate:
        candidate = load(path)
        if candidate['prompt_sha256'] != baseline['prompt_sha256']:
            raise ValueError('candidate prompt corpus differs')
        groups.extend((f'{Path(path).parent.name}-c{c}', candidate['guard']['greedy'][c]) for c in ('1', '8'))
    result = {'method': 'first differing retokenized visible output token; equal answers right-censored at visible length',
              'tokenizer_checkpoint': str(args.checkpoint), 'prompt_sha256': baseline['prompt_sha256'], 'groups': []}
    for label, rows in groups:
        captures = []
        if len(rows) != len(reference):
            raise ValueError('capture count differs')
        for actual, expected in zip(rows, reference):
            if actual['id'] != expected['id']:
                raise ValueError('prompt order differs')
            # Reasoning precedes content in the visible output; encode each
            # separately so a joining separator cannot create an artificial token.
            actual_ids, expected_ids = [], []
            for field in ('reasoning', 'content'):
                actual_ids.extend(tokenizer.encode(actual.get(field) or '', add_special_tokens=False))
                expected_ids.extend(tokenizer.encode(expected.get(field) or '', add_special_tokens=False))
            first = next((i for i, (a, b) in enumerate(zip(actual_ids, expected_ids)) if a != b),
                         min(len(actual_ids), len(expected_ids)))
            same_tokens = actual_ids == expected_ids
            captures.append({'id': actual['id'], 'exact_capture_match': actual == expected,
                             'visible_token_match': same_tokens, 'first_divergence_token': None if same_tokens else first,
                             'first_divergence_or_censored_length': first,
                             'reference_visible_tokens': len(expected_ids), 'candidate_visible_tokens': len(actual_ids),
                             'reference_excerpt': tokenizer.decode(expected_ids[max(0, first-8):first+24]),
                             'candidate_excerpt': tokenizer.decode(actual_ids[max(0, first-8):first+24])})
        median = statistics.median(row['first_divergence_or_censored_length'] for row in captures)
        result['groups'].append({'label': label, 'mismatches': sum(not r['exact_capture_match'] for r in captures),
                                 'visible_token_mismatches': sum(not r['visible_token_match'] for r in captures),
                                 'median_first_divergence_or_censored_length': median,
                                 'early_divergence_review_required': median < 10, 'captures': captures})
    result['summary'] = [{k: row[k] for k in ('label', 'mismatches', 'median_first_divergence_or_censored_length',
                                             'early_divergence_review_required')} for row in result['groups']]
    save(args.out, result)


def wilson(k, n):
    z = 1.959963984540054
    p = k/n
    d = 1 + z*z/n
    center = (p + z*z/(2*n))/d
    radius = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [center-radius, center+radius]


def compare_scores(reference, candidate, field):
    n, m = reference['n'], candidate['n']
    a, b = reference[field], candidate[field]
    pa, pb = a/n, b/m
    la, ua = wilson(a, n)
    lb, ub = wilson(b, m)
    difference = pb-pa
    # Newcombe independent-proportions 95% interval from Wilson score bounds.
    lower = difference-math.sqrt((pb-lb)**2 + (ua-pa)**2)
    upper = difference+math.sqrt((ub-pb)**2 + (pa-la)**2)
    return {'reference': {'successes': a, 'n': n, 'wilson95': [la, ua]},
            'candidate': {'successes': b, 'n': m, 'wilson95': [lb, ub]},
            'difference': difference, 'difference95': [lower, upper],
            'no_detectable_degradation_95': upper >= 0}


def compare(args):
    ref, candidate = load(args.reference/'run.json'), load(args.candidate/'run.json')
    if ref.get('status') != 'measured' or candidate.get('status') != 'measured' or not compatible(ref, candidate):
        raise ValueError('native accuracy runs must be complete and have the same engine/checkpoint/settings/env')
    ref_method, method = load(args.reference/'accuracy/method.json'), load(args.candidate/'accuracy/method.json')
    if ref_method != method:
        raise ValueError('accuracy seeds/sizes/helper/data policy differs')
    guard = load(args.guard/'measure.json')['guard']
    guard_run = load(args.guard/'run.json')
    if identity(guard_run) != identity(candidate):
        raise ValueError('guard does not belong to the exact candidate config')
    results = {}
    errors = 0
    for name, field in (('gsm8k', 'correct'), ('tools', 'pass')):
        a, b = load(args.reference/f'accuracy/{name}.json'), load(args.candidate/f'accuracy/{name}.json')
        ids = 'index' if name == 'gsm8k' else 'id'
        if [r[ids] for r in a['records']] != [r[ids] for r in b['records']]:
            raise ValueError('accuracy cases differ')
        errors += sum(row.get('http_status') != 200 for data in (a, b) for row in data['records'])
        results[name] = compare_scores(a['summary'], b['summary'], field)
        if name == 'gsm8k':
            results[name]['truncated'] = {'reference': a['summary']['truncated'], 'candidate': b['summary']['truncated']}
    gsm = results['gsm8k']
    rate = gsm['candidate']['successes'] / gsm['candidate']['n']
    gsm['within_reference_wilson95'] = gsm['reference']['wilson95'][0] <= rate <= gsm['reference']['wilson95'][1]
    tools = results['tools']
    tools['equal_or_better_than_reference'] = tools['candidate']['successes'] >= tools['reference']['successes']
    approved = (args.output_review == 'pass' and guard['tools_passed'] and errors == 0
                and gsm['within_reference_wilson95'] and tools['equal_or_better_than_reference'])
    result = {'approved': approved, 'candidate': identity(candidate), 'reference': identity(ref),
              'reference_run': str(args.reference), 'candidate_accuracy_run': str(args.candidate),
              'guard_run': str(args.guard), 'output_review': args.output_review,
              'output_review_evidence': args.review_evidence, 'http_errors': errors,
              'three_tool_guard_passed': guard['tools_passed'],
              'method': 'native GSM8K-250 / 30 tools; GSM8K rate inside no-draft 95% Wilson interval and tools equal or better; difference intervals descriptive, not proof of equivalence',
              'accuracy_method_sha256': hashlib.sha256((args.candidate/'accuracy/method.json').read_bytes()).hexdigest(),
              **results}
    save(args.out, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('divergence')
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--repeat', type=Path)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--candidate', type=Path, action='append', default=[])
    p.add_argument('--out', type=Path, required=True)
    p = commands.add_parser('compare')
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--guard', type=Path, required=True)
    p.add_argument('--output-review', choices=('pass', 'fail'), required=True)
    p.add_argument('--review-evidence', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    (divergence if args.command == 'divergence' else compare)(args)


if __name__ == '__main__':
    main()
