#!/usr/bin/env python3
"""Summarize retained publication checks without rerunning model requests."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'bench/final-checks'


def read(label, filename='measure.json'):
    return json.loads((ROOT / label / filename).read_text())


def main():
    result = {'decode': {}, 'retrieval': {}, 'copy100': {}, 'shipped_recipe': {}}
    for name in ('baseline', 'dflash8'):
        requests = read('crossover-' + name)['requests'] + read('decode-' + name)['requests']
        result['decode'][name] = [{key: request[key] for key in ('input_tokens', 'valid', 'decode_tps', 'ttft_s')} for request in requests]
        retrieval = read('retrieval-' + name)
        rows = []
        for length in (32000, 64000, 96000, 111000):
            for variant in ('original', 'one-line'):
                requests = [q for q in retrieval['requests'] if q['target_prompt_tokens'] == length and q['variant'] == variant]
                rows.append({'input_tokens': length, 'variant': variant, 'n': len(requests), 'all_three_present': sum(q['all_three_present'] for q in requests), 'codes_found': sum(q['codes_found'] for q in requests), 'ordered_exact': sum(q['exact_ordered_codes'] for q in requests), 'http_errors': sum(q['exchange'].get('error') is not None for q in requests), 'finish_length': sum((q['exchange'].get('response') or {}).get('choices', [{}])[0].get('finish_reason') == 'length' for q in requests), 'api_prompt_tokens': [(q['exchange'].get('response') or {}).get('usage', {}).get('prompt_tokens') for q in requests]})
        result['retrieval'][name] = rows
        copy = read('copy-' + name)
        result['copy100'][name] = {**copy['summary'],
                                   'prompt_token_mismatches': sum(q['exchange'].get('response', {}).get('usage', {}).get('prompt_tokens') != 2000 for q in copy['requests']),
                                   'output_cap_hits': sum(q['exchange'].get('response', {}).get('choices', [{}])[0].get('finish_reason') == 'length' for q in copy['requests'])}
    baseline = {q['input_tokens']: q['decode_tps'] for q in result['decode']['baseline']}
    comparisons = [{'input_tokens': q['input_tokens'], 'dflash_over_baseline': q['decode_tps'] / baseline[q['input_tokens']]} for q in result['decode']['dflash8']]
    result['decode']['ratios'] = comparisons
    result['decode']['first_measured_length_without_gain'] = next((q['input_tokens'] for q in comparisons if q['dflash_over_baseline'] <= 1), None)
    shipped = read('shipped', 'run.json')
    result['shipped_recipe'] = {'ready_process_cold_s': shipped.get('ready_s'), 'ready_warm_s': shipped.get('ready_warm_s'), 'peak_gpu_compute_used_bytes': shipped.get('peak_gpu_compute_used_bytes'), 'peak_system_used_bytes': shipped.get('peak_system_used_bytes'), 'native_kv_confirmed': shipped.get('native_kv_confirmed'), 'lmcache': shipped.get('lmcache'), 'status': shipped.get('status'), 'gpu_pids_after': shipped.get('gpu_pids_after'), 'sweep': [{key: q[key] for key in ('input_tokens', 'concurrency', 'valid', 'decode_tps', 'aggregate_tps', 'ttft_s')} for q in read('shipped')['requests']], 'acceptance': read('shipped', 'acceptance.json'), 'retrieval111k': [row for row in result['retrieval']['dflash8'] if row['input_tokens'] == 111000]}
    result['shipped_recipe'].update({key: shipped.get(key) for key in
                                     ('capyctl_version', 'config_validated', 'recipe_source_sha256', 'recipe_substitutions')})
    result['tool_diagnostics'] = {name: read('tools-' + name)['counts'] for name in ('baseline', 'dflash8')}
    result['host_clean'] = json.loads((HERE / 'bench/final-cleanup.json').read_text())
    (ROOT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'shipped_recipe'}, indent=2))
    print(json.dumps({key: value for key, value in result['shipped_recipe'].items() if key not in ('acceptance', 'sweep')}, indent=2))


if __name__ == '__main__':
    main()
