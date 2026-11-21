#!/usr/bin/env python3
"""One-sided baseline Wilson lower-bound gates; upper bounds descriptive only."""
import argparse
import math
import json
from pathlib import Path
def wilson(k, n):
    """95% Wilson interval; the gate checks only the relevant side."""
    z = 1.959963984540054
    p = k/n
    d = 1 + z*z/n
    center = (p + z*z/(2*n))/d
    radius = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [center-radius, center+radius]

def compare(baseline, candidate, output_review, review_evidence):
    baseline_run = json.loads((baseline/'run.json').read_text())
    run = json.loads((candidate/'run.json').read_text())
    if baseline_run['status'] != 'measured' or run['status'] != 'measured':
        raise ValueError('incomplete native measurement')
    checkpoint_comparison = baseline_run['checkpoint'] != run['checkpoint']
    baseline_method = json.loads((baseline/'accuracy/method.json').read_text())
    method = json.loads((candidate/'accuracy/method.json').read_text())
    screen = method.get('scope') == 'screen-50'
    if (baseline_method.get('scope') != 'full' or method.get('max_tokens') != 32768 or
            {k:v for k,v in baseline_method.items() if k != 'scope'} != {k:v for k,v in method.items() if k != 'scope'}):
        raise ValueError('requires matching 32768-token methods and a full baseline')
    results = {}
    for name, field, expected in [('gsm8k', 'correct', 250), ('tools', 'pass', 30), ('humaneval', 'passed', 164)]:
        ref = json.loads((baseline/f'accuracy/{name}.json').read_text())
        data = json.loads((candidate/f'accuracy/{name}.json').read_text())
        baseline_ids, candidate_ids = [r['id'] for r in ref['records']], [r['id'] for r in data['records']]
        if (not set(candidate_ids).issubset(baseline_ids) or (not screen and candidate_ids != baseline_ids)):
            raise ValueError('quality cases differ')
        row = {'baseline': {'successes': ref['summary'][field], 'n': ref['summary']['n'],
                            'wilson95': wilson(ref['summary'][field], ref['summary']['n'])},
               'candidate': {'successes': data['summary'][field], 'n': data['summary']['n'],
                             'wilson95': wilson(data['summary'][field], data['summary']['n'])}}
        candidate_expected = 50 if screen and name != 'tools' else expected
        row['complete'] = ref['summary']['n'] == expected and data['summary']['n'] == candidate_expected
        row['errors'] = {'baseline': ref['summary']['errors'], 'candidate': data['summary']['errors']}
        row['truncated'] = {'baseline': ref['summary']['truncated'], 'candidate': data['summary']['truncated']}
        row['not_below_baseline_wilson95_lower'] = data['summary'][field]/data['summary']['n'] >= row['baseline']['wilson95'][0]
        row['baseline_truncation_wilson95'] = wilson(ref['summary']['truncated'], ref['summary']['n'])
        row['candidate_truncation_rate'] = data['summary']['truncated']/data['summary']['n']
        row['truncation_not_above_baseline_wilson95_upper'] = row['candidate_truncation_rate'] <= row['baseline_truncation_wilson95'][1]
        row['passed'] = row['complete'] and not any(row['errors'].values()) and row['not_below_baseline_wilson95_lower'] and row['truncation_not_above_baseline_wilson95_upper']
        if name == 'tools':
            row['all_30_passed'] = data['summary'][field] == 30
            row['passed'] = row['passed'] and row['all_30_passed']
        results[name] = row
    passed = output_review == 'pass' and all(r['passed'] for r in results.values())
    return {'approved': passed and not screen, 'screen_passed': passed if screen else None,
            'scope': 'screen-50' if screen else 'full',
            'baseline_run': baseline.name, 'candidate_accuracy_run': candidate.name,
            'baseline_key': baseline_run['key'], 'candidate_key': run['key'],
            'comparison_type': 'checkpoint' if checkpoint_comparison else 'drafter',
            'output_review': output_review, 'output_review_evidence': review_evidence,
            'method': 'common 32768-token cap; length-limited responses count as wrong; candidate accuracy >= published-recipe baseline 95% Wilson lower bound; candidate truncation rate <= baseline 95% Wilson upper bound; tools 30/30; zero HTTP errors; explicit output review; screens never constitute full qualification',
            **results}

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output-review', choices=('pass', 'fail'), required=True)
    p.add_argument('--review-evidence', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = compare(a.baseline, a.candidate, a.output_review, a.review_evidence)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
