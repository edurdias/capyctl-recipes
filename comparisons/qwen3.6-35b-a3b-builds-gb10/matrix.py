#!/usr/bin/env python3
"""Qwen3.6 published-recipe and no-draft/speculative deployments."""
import argparse
import json
from pathlib import Path

PINS = {'nvidia': '1355db6a052410cfd62085d94b58866fd0f2c3c5', 'dflash2': '51ef7b6923ad6c14cb1bb41c37a9041446496ab6', 'dspark': '53814b238c3a6ce5f332066a6bedb9d179777ede'}

def model_path(name):
    return Path.home()/'models'/f'qwen36-{name}-{PINS[name][:12]}'

def deployment(key, checkpoint=None, settings=(), drafter_path=None):
    engine, build, draft, *tail = key.split('-')
    depth = int(tail[0]) if tail else None
    if engine not in ('sg', 'vl') or build not in ('nvidia', 'delta', 'combined', 'capyctl', 'redhat', 'unsloth', 'unslothfast', 'qwenfp8', 'intelmixed') or draft not in ('none', 'mtp', 'dflash2', 'dspark'):
        raise ValueError('unknown recipe')
    if settings:
        raise ValueError('no undeclared tuning settings')
    config = {'context_length': 262144, 'kv_cache_dtype': 'auto', 'accept_extra_args': True}
    if engine == 'sg':
        config['memory'] = {'request': '100GiB', 'kv_cache': '40GiB'}
        config['sglang'] = {'tool_call_parser': 'qwen3_coder', 'reasoning_parser': 'qwen3',
                            'max_total_tokens': 2000000}
        extra = ['--moe-runner-backend', 'flashinfer_cutlass',
                 '--max-mamba-cache-size', '110', '--prefill-decode-interval', '1']
        if draft == 'mtp':
            extra += ['--speculative-algorithm', 'EAGLE', '--speculative-num-steps', str(depth),
                      '--speculative-eagle-topk', '1', '--speculative-num-draft-tokens', str(depth+1),
                      '--enable-linear-replayssm-spec']
        elif draft != 'none':
            extra += ['--speculative-algorithm', {'dflash2': 'DFLASH', 'dspark': 'DSPARK'}[draft],
                      '--speculative-draft-model-path', str(drafter_path or model_path(draft)),
                      '--speculative-draft-model-quantization', 'unquant',
                      '--speculative-draft-attention-backend', 'flashinfer',
                      '--speculative-num-draft-tokens', str(depth)]
    else:
        config.update({'kv_cache_dtype': 'fp8', 'language_model_only': True, 'max_concurrent_requests': 8,
                       'memory': {'kv_cache': '24GiB'},
                       'vllm': {'tool_call_parser': 'qwen3_coder', 'reasoning_parser': 'qwen3', 'max_num_batched_tokens': 8192}})
        extra = ['--attention-backend', 'FLASHINFER', '--moe-backend', 'marlin', '--enable-prefix-caching', '--async-scheduling']
        if draft != 'none':
            spec = {'method': 'dflash' if draft == 'dflash2' else draft,
                    'num_speculative_tokens': depth}
            if draft == 'mtp':
                spec['moe_backend'] = 'triton'
            else:
                spec['model'] = str(drafter_path or model_path(draft))
            extra += ['--speculative-config', json.dumps(spec)]
    config['extra_args'] = extra
    return {'schema_version': 1, 'kind': 'deployment', 'name': 'q36-'+key,
            'engine': 'sg522' if engine == 'sg' else 'vl032',
            'model': str(checkpoint or model_path('nvidia')), 'residency': 'restart_only', 'lmcache': 'off',
            'timeouts': {'initialize': '1800s'}, 'engine_config': config}

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('key')
    p.add_argument('--checkpoint')
    a = p.parse_args()
    print(json.dumps(deployment(a.key, a.checkpoint), indent=2))
