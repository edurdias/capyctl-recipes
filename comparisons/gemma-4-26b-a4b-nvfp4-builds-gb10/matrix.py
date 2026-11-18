#!/usr/bin/env python3
"""Render CapyCTL JSON-as-YAML deployments; no engine starts here.

Keys: sg|vl - nvidia|redhat|unsloth|r1|r2|r3|r3e|capyctl - none|mtp|dspark|eagle3|dflash - depth.
Examples: sg-nvidia-none, vl-redhat-mtp-2, sg-r2-dflash-8 --checkpoint PATH.
Use identical --setting options on a drafter and its no-draft reference.
"""
import argparse
import json
from pathlib import Path
from download import model_path


def deployment(key, checkpoint=None, settings=(), drafter_path=None):
    pieces = key.split('-')
    engine, build, drafter = pieces[:3]
    depth = int(pieces[3]) if len(pieces) == 4 else None
    if engine not in ('sg', 'vl') or build not in ('nvidia', 'redhat', 'unsloth', 'r1', 'r2', 'r3', 'r3e', 'capyctl'):
        raise ValueError('unknown engine/build')
    if drafter not in ('none', 'mtp', 'dspark', 'eagle3', 'dflash'):
        raise ValueError('unknown drafter')
    if drafter != 'none' and (depth is None or depth < 1):
        raise ValueError('drafter needs positive depth')
    if engine == 'vl' and drafter == 'dflash':
        raise ValueError('stock vLLM Gemma DFlash is not qualified; needs Gemma-specific patch')
    if build in ('r1', 'r2', 'r3', 'r3e', 'capyctl') and not checkpoint:
        raise ValueError('--checkpoint required for a converted build')
    config = {'context_length': 131072, 'kv_cache_dtype': 'fp8_e4m3',
              'memory': {'request': '80GiB'} if engine == 'sg' else {'kv_cache': '40GiB'}}
    common = ['--moe-runner-backend', 'flashinfer_cutlass'] if engine == 'sg' else []
    config['sglang' if engine == 'sg' else 'vllm'] = {
        'tool_call_parser': 'gemma4', 'reasoning_parser': 'gemma4',
        **({'chunked_prefill_size': 4096} if engine == 'sg' else {'max_num_batched_tokens': 4096})}
    for setting in settings:
        if setting.startswith('max-requests='):
            config['max_concurrent_requests'] = int(setting.split('=')[1])
        elif setting.startswith('attention='):
            common += ['--attention-backend', setting.split('=', 1)[1]]
        elif setting.startswith('moe='):
            backend = setting.split('=', 1)[1]
            if engine == 'sg':
                common[1] = backend
            else:
                common += ['--moe-backend', backend]
        elif setting.startswith('graph='):
            sizes = list(map(int, setting.split('=')[1].split(',')))
            common += ['--cuda-graph-bs' if engine == 'sg' else '--cudagraph-capture-sizes', *map(str, sizes)]
        else:
            raise ValueError('unknown setting ' + setting)
    if drafter != 'none':
        path = str(Path(drafter_path).expanduser() if drafter_path else model_path(drafter))
        if engine == 'vl':
            common += ['--speculative-config', json.dumps({'method': drafter, 'model': path, 'num_speculative_tokens': depth})]
        else:
            common += ['--speculative-algorithm', {'mtp': 'NEXTN', 'eagle3': 'EAGLE3', 'dspark': 'DSPARK', 'dflash': 'DFLASH'}[drafter],
                       '--speculative-draft-model-path', path, '--speculative-draft-model-quantization', 'unquant',
                       '--speculative-num-draft-tokens', str(depth + 1 if drafter in ('mtp', 'eagle3', 'dspark') else depth)]
            if drafter in ('mtp', 'eagle3'):
                common += ['--speculative-num-steps', str(depth), '--speculative-eagle-topk', '1']
    config.update({'accept_extra_args': True, 'extra_args': common})
    return {'schema_version': 1, 'kind': 'deployment', 'name': 'g4-' + key,
            'engine': 'sg522' if engine == 'sg' else 'vl032',
            'model': str(Path(checkpoint).expanduser()) if checkpoint else str(model_path(build)),
            'residency': 'restart_only', 'lmcache': 'off',
            'timeouts': {'initialize': '1800s'}, 'engine_config': config}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('key')
    parser.add_argument('--checkpoint')
    parser.add_argument('--setting', action='append', default=[])
    args = parser.parse_args()
    print(json.dumps(deployment(args.key, args.checkpoint, args.setting), indent=2))
