#!/usr/bin/env python3
"""Assemble retained measurements without launching inference."""
import argparse
import json
import math
import re
from pathlib import Path


def wilson(k, n):
    z = 1.959963984540054
    denominator = 1+z*z/n
    center = (k/n+z*z/(2*n))/denominator
    radius = z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/denominator
    return [max(0, center-radius), min(1, center+radius)]


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bench', type=Path, default=Path(__file__).resolve().parent/'bench')
    a = p.parse_args()
    matrix = load(a.bench/'final-matrix.json') or {'schema_version': 1, 'rows': {}}
    for row in ('openai-vl032-default-allocator', 'openai-vl032', 'capyctl-vl032', 'published-vl030'):
        base = a.bench/matrix['rows'].get(row, {}).get('run', row)
        performance_run = matrix['rows'].get(row, {}).get('performance_run')
        performance_base = a.bench/performance_run if performance_run else base
        run = load(base/'run.json')
        if not run:
            continue
        entry = {'status': run['status'], 'run': str(base.relative_to(a.bench)),
                 'ready_s': run.get('ready_s'), 'min_mem_available_bytes': run.get('min_mem_available_bytes'),
                 'peak_system_used_bytes': run.get('peak_system_used_bytes'),
                 'fp8_kv_confirmed': run.get('fp8_kv_confirmed'),
                 'fp8_attention_confirmed': run.get('fp8_attention_confirmed')}
        entry['engine_environment'] = run.get('engine_environment')
        entry['safetensors_load_strategy'] = run.get('settings', {}).get('engine_config', {}).get('vllm', {}).get('safetensors_load_strategy', 'lazy (engine default)')
        entry['gpu_pids_after'] = run.get('gpu_pids_after')
        if run.get('recipe_source_sha256'):
            entry['recipe_source_sha256'] = run['recipe_source_sha256']
            entry['recipe_substitution'] = run['recipe_substitution']
        if performance_run:
            entry['performance_run'] = performance_run
            entry['performance_evidence'] = load(performance_base/'run.json')
        native_path = base/'engine-evidence.txt'
        native = native_path.read_text() if native_path.exists() else ''
        entry['native_cache_policy'] = {}
        for field in ('kv_cache_dtype', 'enable_prefix_caching'):
            values = re.findall(r'\b'+field+r'=([A-Za-z0-9_]+)', native)
            entry['native_cache_policy'][field] = values[-1] if values else None
        prefix = entry['native_cache_policy']['enable_prefix_caching']
        entry['long_ttft_scope'] = ('cached repeated-document median, not uncached prefill' if prefix == 'True'
                                    else 'prefix caching disabled' if prefix == 'False' else 'cache policy unconfirmed')
        if run.get('error'):
            entry['error'] = run['error']
        speed = load(performance_base/'measure.json')
        entry['short'] = [{k: item[k] for k in ('streams', 'decode_tps', 'aggregate_tps', 'ttft_s')}
                          for item in speed.get('concurrency', [])] if speed else None
        decode = load(performance_base/'decode.json')
        entry['long_decode'] = [{k: item[k] for k in ('input_tokens', 'valid', 'decode_tps', 'aggregate_tps', 'ttft_s')}
                               for item in decode['requests']] if decode else None
        sweep = load(performance_base/'sweep.json')
        entry['sweep'] = [{k: item[k] for k in ('input_tokens', 'concurrency', 'valid', 'decode_tps', 'aggregate_tps', 'ttft_s')}
                          for item in sweep['requests']] if sweep else None
        quality = {}
        for name, field in (('gsm8k', 'correct'), ('tools', 'pass'), ('humaneval', 'passed')):
            data = load(base/'accuracy'/(name+'.json'))
            if data:
                quality[name] = {**data['summary'], 'success_field': field,
                                 'wilson95': wilson(data['summary'][field], data['summary']['n'])}
        entry['quality_screen' if row == 'published-vl030' else 'quality'] = quality or None
        if row in ('capyctl-vl032', 'openai-vl032'):
            entry['retrieval'] = None
            entry['copy'] = None
            retrieval = load(performance_base/'retrieval.json') or load(base/'retrieval.json')
            if retrieval:
                cells = {}
                for q in retrieval['requests']:
                    key = str(q['target_prompt_tokens'])+'-'+q['variant']
                    cell = cells.setdefault(key, {'n': 0, 'all_three_present': 0, 'ordered_exact': 0, 'codes_found': 0})
                    cell['n'] += 1
                    cell['all_three_present'] += q['all_three_present']
                    cell['ordered_exact'] += q['exact_ordered_codes']
                    cell['codes_found'] += q['codes_found']
                entry['retrieval'] = {'completed': retrieval['completed'], 'cells': cells}
            copying = load(performance_base/'copy.json') or load(base/'copy.json')
            if copying:
                entry['copy'] = {'completed': copying['completed'], 'summary': copying.get('summary')}
        matrix['rows'][row] = entry
    baseline = matrix['rows'].get('openai-vl032', {}).get('quality')
    candidate = matrix['rows'].get('capyctl-vl032', {}).get('quality')
    if baseline and candidate:
        gates = {}
        for name, n, field in (('gsm8k', 250, 'correct'), ('humaneval', 164, 'passed'), ('tools', 30, 'pass')):
            if name not in baseline or name not in candidate:
                continue
            b, c = baseline[name], candidate[name]
            lower = wilson(b[field], b['n'])[0]
            truncation_upper = wilson(b['truncated'], b['n'])[1]
            conditions = {'full_sample': b['n'] == c['n'] == n,
                          'accuracy': c[field] == n if name == 'tools' else c[field]/c['n'] >= lower,
                          'truncations': c['truncated']/c['n'] <= truncation_upper,
                          'zero_http_errors': c['errors'] == b['errors'] == 0}
            gates[name] = {'baseline_accuracy_lower95': lower, 'baseline_truncation_upper95': truncation_upper,
                           'conditions': conditions, 'passed': all(conditions.values())}
        matrix['gates'] = gates
        matrix['qualified'] = len(gates) == 3 and all(g['passed'] for g in gates.values())
    (a.bench/'final-matrix.json').write_text(json.dumps(matrix, indent=2)+'\n')


if __name__ == '__main__':
    main()
