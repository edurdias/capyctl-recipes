#!/usr/bin/env python3
"""Print the measured table and evaluate full-task Wilson gates without inventing missing runs."""
import argparse
import json
from pathlib import Path
from measure import load

HERE = Path(__file__).resolve().parent
ROWS = ['mse-sg', 'mse-dflash2', 'google-vl', 'redhat-vl', 'nvidia-vl', 'published-vl030']


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def qualification_root(directory):
    split = directory / 'qualification'
    return split if split.exists() else directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--leader', choices=ROWS)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--output-review', choices=['pass', 'fail'])
    args = parser.parse_args()
    quality = load('canonical_wilson', HERE.parent / 'gpt-oss-20b-builds-gb10/quality.py')
    result = {'rows': {}, 'leader': args.leader, 'full_gate': None}
    gsm_file = HERE.parents[1] / 'comparisons/qwen3.8-27b-sglang-dflash2-vs-mtp-eagle-gb10/accuracy/gsm8k-test-250.jsonl'
    screen_ids = {'gsm8k': {str(json.loads(line)['index']) for line in gsm_file.read_text().splitlines()[:50]},
                  'humaneval': {'HumanEval/' + str(index) for index in range(50)}}
    print('| Setup | c1 decode | c2 total | c4 total | c8 total | 8k / 32k / 64k decode | 50 / 30 / 50 screen | Peak drop GiB | Status |')
    print('|---|---:|---:|---:|---:|---|---|---:|---|')
    for name in ROWS:
        directory = args.runs / name
        run = read(directory / 'run.json')
        if not run:
            print(f'| {name} | — | — | — | — | — | — | — | Not run |')
            continue
        measure = read(directory / 'measure.json') or {'levels': []}
        full_root = qualification_root(directory)
        qualification_run = read(full_root / 'run.json') if full_root != directory else None
        short = {item['concurrency']: item for item in measure['levels'] if item['label'] == 'short'}
        long = {item['target_prompt_tokens']: item for item in measure['levels'] if item['label'] == 'long'}
        def speed(value):
            return f'{value:.3f}' if value is not None else '—'
        speeds = [speed(short.get(c, {}).get('decode_tps' if c == 1 else 'aggregate_tps')) for c in (1, 2, 4, 8)]
        longs = ' / '.join(speed(long.get(length, {}).get('decode_tps')) for length in (8000, 32000, 64000))
        task_counts = []
        for task, field in [('gsm8k', 'correct'), ('tools', 'pass'), ('humaneval', 'passed')]:
            data = read(directory / 'screen' / (task + '.json'))
            if data is None:
                data = read(full_root / 'full' / (task + '.json'))
                if data:
                    data = {**data, 'records': [record for record in data['records']
                            if task == 'tools' or record['id'] in screen_ids[task]]}
            task_counts.append(str(sum(bool(record[field]) for record in data['records'])) if data else '—')
        peaks = [item.get('peak_mem_available_drop_bytes') for item in (run, qualification_run) if item]
        peak = max((value for value in peaks if value is not None), default=None)
        peak_text = f'{peak / 1024 ** 3:.3f}' if peak is not None else '—'
        status = 'measured in split phases' if qualification_run and qualification_run.get('status') == 'measured' else run['status']
        print('| ' + ' | '.join([name, *speeds, longs, ' / '.join(task_counts), peak_text, status]) + ' |')
        result['rows'][name] = {'run': run, 'qualification_run': qualification_run, 'short': short, 'long': long, 'screen_successes': task_counts}
    if args.leader:
        baseline = qualification_root(args.runs / 'mse-sg') / 'full'
        candidate = qualification_root(args.runs / args.leader) / 'full'
        comparisons = {}
        checks = {'output_repetition_review': args.output_review == 'pass'}
        for task, field, expected in [('gsm8k', 'correct', 250), ('tools', 'pass', 30), ('humaneval', 'passed', 164)]:
            first, second = read(baseline / (task + '.json')), read(candidate / (task + '.json'))
            if first is None or second is None:
                checks[task + '_complete'] = False
                continue
            a, b = first['summary'], second['summary']
            comparisons[task] = quality.compare_scores(a, b, field)
            comparisons[task]['mse_sglang'] = comparisons[task].pop('reference')
            checks[task + '_complete'] = a['n'] == b['n'] == expected
            checks[task + '_http'] = a['errors'] == b['errors'] == 0
            checks[task + '_accuracy'] = b[field] / b['n'] >= quality.wilson(a[field], a['n'])[0]
            checks[task + '_truncation'] = b['truncated'] / b['n'] <= quality.wilson(a['truncated'], a['n'])[1]
            if task == 'tools':
                checks['all_tools'] = b[field] == expected
        for name in ('mse-sg', args.leader):
            gates = read(qualification_root(args.runs / name) / 'gates.json')
            checks[name + '_additional_probes_complete'] = bool(gates and gates.get('completed'))
            if gates and gates.get('completed'):
                checks[name + '_reasoning'] = gates['reasoning']['passed']
                checks[name + '_tool_round_trips'] = all(probe['passed'] for probe in gates['tools'])
                comparisons[name + '_retrieval'] = {
                    'observations': len(gates['retrieval']),
                    'all_three_present': sum(item['all_three_present'] for item in gates['retrieval']),
                    'codes_found': sum(item['codes_found'] for item in gates['retrieval'])}
                comparisons[name + '_copy'] = {'n': len(gates['copy']), 'strict': sum(item['exact_copy'] for item in gates['copy'])}
        result['full_gate'] = {'checks': checks, 'task_gate_passed': all(checks.values()), 'comparisons': comparisons,
                               'interpretation': 'Requested serving-setup comparison, not an isolated checkpoint/engine/drafter effect; finite Wilson gate is not equivalence.'}
    args.out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
