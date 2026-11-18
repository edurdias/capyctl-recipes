#!/usr/bin/env python3
"""Compare fixed-document retrieval outcomes with exact paired sign tests.

The document, not the prompt variant or individual code, is the independent
unit. The primary endpoint sums all-three-present successes over both prompt
variants per document. Ordered exact and codes-found tests remain secondary.
"""
import argparse
import itertools
import json
import math
from pathlib import Path


def load(path):
    result = json.loads(path.read_text())
    if not result.get('completed'):
        raise ValueError('A paired run must be complete')
    records = {}
    for request in result['requests']:
        key = (request['document'], request['variant'])
        if key in records:
            raise ValueError('Expected one observation per document and variant')
        request['all_three_present'] = request['codes_found'] == len(request['expected_codes'])
        records[key] = request
    return records


def sign_test(differences):
    wins = sum(value > 0 for value in differences)
    losses = sum(value < 0 for value in differences)
    count = wins + losses
    tail = sum(math.comb(count, index) for index in range(min(wins, losses) + 1))
    return {'wins': wins, 'losses': losses, 'ties': len(differences) - count,
            'informative_documents': count,
            'two_sided_exact_p': min(1.0, 2 * tail / 2 ** count)}


def compare(reference, candidate):
    if reference.keys() != candidate.keys():
        raise ValueError('Paired runs must use exactly the same documents and variants')
    for key in reference:
        if (reference[key]['expected_codes'], reference[key]['target_prompt_tokens']) != (candidate[key]['expected_codes'], candidate[key]['target_prompt_tokens']):
            raise ValueError('Paired fixture identities differ')
    documents = sorted({key[0] for key in reference})
    variants = sorted({key[1] for key in reference})
    rows = []
    for document in documents:
        row = {'document': document, 'variants': {}}
        for variant in variants:
            left, right = reference[document, variant], candidate[document, variant]
            row['variants'][variant] = {
                'reference_all_three_present': left['all_three_present'],
                'candidate_all_three_present': right['all_three_present'],
                'reference_exact': left['exact_ordered_codes'],
                'candidate_exact': right['exact_ordered_codes'],
                'reference_codes_found': left['codes_found'],
                'candidate_codes_found': right['codes_found'],
                'reference_answer': left['answer'], 'candidate_answer': right['answer']}
        rows.append(row)
    tests = {}
    for metric, source in (('all_three_present', 'all_three_present'), ('exact', 'exact_ordered_codes'), ('codes_found', 'codes_found')):
        for scope, selected in [('both_variants', variants)] + [(variant, [variant]) for variant in variants]:
            differences = [sum(int(candidate[document, variant][source]) - int(reference[document, variant][source]) for variant in selected) for document in documents]
            tests[f'{metric}_{scope}'] = sign_test(differences)
    primary = tests['all_three_present_both_variants']
    return {'documents': len(documents), 'per_document': rows, 'paired_sign_tests': tests,
            'primary_candidate_worse_at_0_05': primary['two_sided_exact_p'] < .05 and primary['losses'] > primary['wins'],
            'primary_candidate_better_at_0_05': primary['two_sided_exact_p'] < .05 and primary['wins'] > primary['losses']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='LABEL=path/to/probes.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    runs = dict(item.split('=', 1) for item in args.run)
    if len(runs) != len(args.run):
        parser.error('Run labels must be unique')
    loaded = {name: load(Path(path)) for name, path in runs.items()}
    result = {'method': 'Exact two-sided paired sign test; independent unit is the document; ties excluded. Primary endpoint is all-three-present successes summed across the two prompt variants per document. Secondary ordered-exact, per-variant and codes-found tests are exploratory, without multiplicity adjustment.',
              'limits': 'Assumes independent document seeds and symmetric signs under the null. A nonsignificant result does not prove equivalence or exclude a smaller gap; a single run per fixture does not fully estimate engine numerical variation.',
              'runs': runs, 'comparisons': {}}
    for reference, candidate in itertools.combinations(loaded, 2):
        result['comparisons'][candidate + '_versus_' + reference] = compare(loaded[reference], loaded[candidate])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({name: comparison['paired_sign_tests']['all_three_present_both_variants'] for name, comparison in result['comparisons'].items()}))


if __name__ == '__main__':
    main()
