#!/usr/bin/env python3
"""Summarize retained seeded probes without contacting an engine."""
import argparse
import json
from pathlib import Path
from quality import wilson


def summarize(directory):
    retrieval = json.loads((directory/'retrieval.json').read_text())
    copy = json.loads((directory/'copy.json').read_text())
    groups = []
    for length in sorted({q['prompt_tokens'] for q in retrieval['requests']}):
        rows = [q for q in retrieval['requests'] if q['prompt_tokens'] == length]
        groups.append({'prompt_tokens': length, 'observations': len(rows),
                       'all_three_present': sum(q['all_three_present'] for q in rows),
                       'codes_found': sum(q['codes_found'] for q in rows),
                       'codes_total': 3*len(rows),
                       'exact_ordered_codes': sum(q['exact_ordered_codes'] for q in rows),
                       'original_prompt_count_mismatches': sum(q['actual_prompt_tokens'] != length for q in rows if q['variant'] == 'original'),
                       'actual_prompt_tokens_by_variant': {v: sorted({q['actual_prompt_tokens'] for q in rows if q['variant'] == v}, key=lambda x: -1 if x is None else x) for v in ('original', 'one-line')},
                       'truncated': sum((q['exchange'].get('response') or {}).get('choices', [{}])[0].get('finish_reason') == 'length' for q in rows),
                       'passed': len(rows) == 12 and all(q['all_three_present'] for q in rows)})
    copies = copy['requests']
    c = {'documents': len(copies), 'http_errors': copy['http_errors'],
         'prompt_count_mismatches': sum(q['actual_prompt_tokens'] != 2000 for q in copies),
         'truncated': sum((q['exchange'].get('response') or {}).get('choices', [{}])[0].get('finish_reason') == 'length' for q in copies)}
    for metric in ('exact_copy', 'exact_code_present'):
        n, k = len(copies), sum(q[metric] for q in copies)
        c[metric] = {'successes': k, 'n': n, 'wilson95': wilson(k, n)}
    return {'completed': retrieval['completed'] and copy['completed'],
            'retrieval': {'levels': groups, 'http_errors': retrieval['http_errors'],
                          'all_document_gate': len(groups) == 5 and all(g['passed'] for g in groups)},
            'copy': c}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.write_text(json.dumps(summarize(a.directory), indent=2)+'\n')
