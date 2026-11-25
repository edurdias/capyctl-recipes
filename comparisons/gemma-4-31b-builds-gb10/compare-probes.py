#!/usr/bin/env python3
"""Compare every row's exact retrieval/copy observations with MSE on SGLang."""
import argparse
import json
from pathlib import Path
from measure import load

HERE = Path(__file__).resolve().parent
ROWS = ['mse-sg', 'mse-dflash2', 'google-vl', 'redhat-vl', 'nvidia-vl', 'published-vl030', 'mse-sg-bf16']


def evidence(directory):
    for parent in [directory / 'qualification', directory / 'probes', directory]:
        path = parent / 'gates.json'
        if path.exists():
            return json.loads(path.read_text()), str(path)
    return None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    quality = load('probe_wilson', HERE.parent / 'gpt-oss-20b-builds-gb10/quality.py')
    baseline, source = evidence(args.runs / 'mse-sg')
    if not baseline or not baseline.get('completed'):
        raise ValueError('completed MSE SGLang probe evidence is required')

    def summarize(records, field):
        return {'n': len(records), 'correct': sum(bool(row[field]) for row in records),
                'errors': sum(row['exchange'].get('http_status') != 200 or row['exchange'].get('error') is not None for row in records),
                'truncated': sum((row['exchange'].get('response') or {}).get('choices', [{}])[0].get('finish_reason') == 'length' for row in records)}

    def compare(first, second, identities_match):
        if not second['n']:
            return {'status': 'not measured', 'approved': False}
        result = quality.compare_scores(first, second, 'correct')
        result['mse_sglang'] = result.pop('reference')
        result['checks'] = {'same_cases': identities_match,
                            'zero_http_errors': first['errors'] == second['errors'] == 0,
                            'accuracy_at_or_above_mse_sglang_wilson95_lower': second['correct'] / second['n'] >= quality.wilson(first['correct'], first['n'])[0],
                            'truncation_at_or_below_mse_sglang_wilson95_upper': second['truncated'] / second['n'] <= quality.wilson(first['truncated'], first['n'])[1]}
        result['truncations'] = {'mse_sglang': first['truncated'], 'candidate': second['truncated']}
        result['approved'] = all(result['checks'].values())
        return result

    result = {'method': 'Document-level all-three-exact retrieval and strict copy, compared with MSE SGLang 95% Wilson lower accuracy / upper truncation bounds, zero HTTP errors and matched cases. Code counts are descriptive; codes within a document are not independent trials. Passing a relative finite gate is not absolute correctness or equivalence.', 'rows': {}}
    print('| Setup | 32k all-three / codes | 64k all-three / codes | Strict copy | Retrieval errors / caps | Copy errors / caps | Relative gates |')
    print('|---|---|---|---|---|---|---|')
    for name in ROWS:
        data, path = evidence(args.runs / name)
        if not data:
            print(f'| {name} | — | — | — | — | — | Not run |')
            continue
        row = {'completed': data.get('completed', False), 'source': path, 'retrieval': {}}
        retrieval_cells = []
        for length in (32000, 64000):
            first = [item for item in baseline['retrieval'] if item['target_prompt_tokens'] == length]
            second = [item for item in data['retrieval'] if item['target_prompt_tokens'] == length]
            a, b = summarize(first, 'all_three_present'), summarize(second, 'all_three_present')
            identities = lambda items: {(item['probe_id'], item['variant']) for item in items}
            score = compare(a, b, identities(first) == identities(second))
            codes = sum(item['codes_found'] for item in second)
            row['retrieval'][str(length)] = {'summary': b, 'codes_found': codes, 'codes_total': 3 * b['n'], 'comparison': score}
            retrieval_cells.append(f"{b['correct']}/{b['n']} · {codes}/{3*b['n']}" if b['n'] else 'Outside unchanged context')
        a, b = summarize(baseline['copy'], 'exact_copy'), summarize(data['copy'], 'exact_copy')
        row['copy'] = {'summary': b, 'comparison': compare(a, b,
            {item['filler_seed'] for item in baseline['copy']} == {item['filler_seed'] for item in data['copy']})}
        checks = [item['comparison']['approved'] for item in row['retrieval'].values()] + [row['copy']['comparison']['approved'], row['completed']]
        row['all_relative_gates_passed'] = all(checks)
        errors = sum(item['summary']['errors'] for item in row['retrieval'].values())
        caps = sum(item['summary']['truncated'] for item in row['retrieval'].values())
        print('| ' + ' | '.join([name, *retrieval_cells, f"{b['correct']}/{b['n']}", f'{errors} / {caps}',
                                 f"{b['errors']} / {b['truncated']}", 'pass' if row['all_relative_gates_passed'] else 'not all passed']) + ' |')
        result['rows'][name] = row
    args.out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
