#!/usr/bin/env python3
"""Ordered agreement, drift and slices; agreement is not gold-label accuracy."""
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import decision_eval as harness


def wilson(same, n):
    z = 1.959963984540054
    p = same/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [center-half, center+half]


def summary(rows):
    same = sum(r['same_top'] for r in rows)
    low = [r for r in rows if r['margin'] <= .05]
    return {'questions': len(rows), 'same_top': same, 'agreement': same/len(rows), 'wilson95': wilson(same, len(rows)), 'mean_kl': sum(r['kl'] for r in rows)/len(rows), 'mean_tv': sum(r['tv'] for r in rows)/len(rows), 'gap_p50': harness.bench.percentile([r['gap'] for r in rows], 50), 'gap_p95': harness.bench.percentile([r['gap'] for r in rows], 95), 'gap_max': max(r['gap'] for r in rows), 'low_margin_questions': len(low), 'low_margin_flips': sum(not r['same_top'] for r in low), 'flips': [r for r in rows if not r['same_top']]}


def compare(reference, candidate):
    refs = [json.loads(s) for s in reference.read_text().splitlines()]
    candidates = [json.loads(s) for s in candidate.read_text().splitlines()]
    rows = []
    assert len(refs) == len(candidates)
    for ref, cand in zip(refs, candidates, strict=True):
        assert ref['id'] == cand['id'] and ref['payload_sha256'] == cand['payload_sha256'] and ref['question_order'] == cand['question_order']
        assert not ref['error'] and not cand['error']
        for q in ref['question_order']:
            p = harness.distribution(ref['answers'][q]); r = harness.distribution(cand['answers'][q])
            assert p.keys() == r.keys()
            for d in (p, r):
                assert d and all(math.isfinite(v) and 0 <= v <= 1 for v in d.values()) and abs(sum(d.values())-1) <= 1e-4
            values = sorted(p.values(), reverse=True)
            tokens = ref['input_tokens']
            length = '<=512' if tokens <= 512 else '513-2048' if tokens <= 2048 else '2049-8192' if tokens <= 8192 else '>8192'
            rows.append({'id': ref['id'], 'question': q, 'suite': ref['suite'], 'length': length, 'same_top': max(p, key=p.get) == max(r, key=r.get), 'reference_top': max(p, key=p.get), 'candidate_top': max(r, key=r.get), 'margin': values[0]-values[1], 'gap': max(abs(v-r[k]) for k,v in p.items()), 'tv': sum(abs(v-r[k]) for k,v in p.items())/2, 'kl': sum(v*math.log(v/max(r[k], 1e-9)) for k,v in p.items() if v > 0)})
    result = summary(rows)
    result.update(requests=len(refs), request_errors=0, by_suite={s:summary([r for r in rows if r['suite']==s]) for s in sorted({r['suite'] for r in rows})}, by_length={s:summary([r for r in rows if r['length']==s]) for s in sorted({r['length'] for r in rows})})
    return result


if __name__ == '__main__':
    print(json.dumps(compare(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
