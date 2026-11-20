#!/usr/bin/env python3
"""Controlled GPT-OSS reference deployments and pinned drafter screens."""
import argparse
import json
from pathlib import Path

REVISION = '6cee5e81ee83917806bbde320786a8fb61efebee'
GGUF_REVISION = '41e8a94d15d62070b36cf4553a0250717ecf46b8'
ROWS = {
    'sg522-default': ('sg522', 'sglang', 'flashinfer_mxfp4'),
    'sg522-marlin': ('sg522', 'sglang', 'marlin'),
    'sg522-triton': ('sg522', 'sglang', 'triton_kernel'),
    'vl032-auto': ('vl032', 'vllm', 'auto'),
    'vl032-cutlass': ('vl032', 'vllm', 'flashinfer_cutlass'),
    'vl033-auto': ('vl033', 'vllm', 'auto'),
    'llama-b12242': ('llama12242', 'llamacpp', 'native'),
    'llama-b12242-eagle3': ('llama12242', 'llamacpp', 'eagle3'),
    'sg522-attn-flashinfer': ('sg522', 'sglang', 'flashinfer_mxfp4'),
    'sg522-attn-triton': ('sg522', 'sglang', 'flashinfer_mxfp4'),
}

DRAFTS = {}
for prefix, engine, backend in (('sg522', 'sglang', 'flashinfer_mxfp4'), ('vl032', 'vllm', 'auto')):
    for block in (8, 4, 6, 12, 16):
        key = f'{prefix}-dflash{block}'
        ROWS[key] = (prefix, engine, backend)
        DRAFTS[key] = {'kind': 'dflash', 'block_size': block,
                       'path': str(Path.home()/'models/gpt-oss-20b-dflash'),
                       'revision': 'd53f6551543204c859e8bbaaddbd15d11b447af9',
                       'baseline': prefix+('-default' if engine == 'sglang' else '-auto')}
    for steps in ((3, 5, 7) if engine == 'vllm' else (3, 5)):
        key = f'{prefix}-nebius{steps}'
        ROWS[key] = (prefix, engine, backend)
        DRAFTS[key] = {'kind': 'eagle3', 'steps': steps, 'topk': 1,
                       'path': str(Path.home()/'models/gpt-oss-20b-nebius-eagle3'),
                       'revision': '9dd45bb1da8b1ddc8cabd52691d9ac170b41484f',
                       'baseline': prefix+('-default' if engine == 'sglang' else '-auto')}


def deployment(key, checkpoint=None):
    profile, engine, backend = ROWS[key]
    model = str(Path(checkpoint).expanduser()) if checkpoint else str(Path.home()/'models'/('gpt-oss-20b-gguf' if engine == 'llamacpp' else 'gpt-oss-20b'))
    config = {'context_length': 131072, 'max_concurrent_requests': 16,
              'kv_cache_dtype': 'fp8_e4m3', 'accept_extra_args': True}
    if engine == 'sglang':
        config.update(memory={'request': '80GiB'}, sglang={'reasoning_parser': 'gpt-oss', 'tool_call_parser': 'gpt-oss'},
                      extra_args=['--cuda-graph-max-bs-decode', '16', '--swa-full-tokens-ratio', '0.1'])
        if backend != 'flashinfer_mxfp4':
            config['extra_args'] += ['--moe-runner-backend', backend]
        if key.startswith('sg522-attn-'):
            config['extra_args'] += ['--attention-backend', key.removeprefix('sg522-attn-')]
    elif engine == 'vllm':
        config.update(memory={'kv_cache': '40GiB'}, vllm={'reasoning_parser': 'openai_gptoss', 'tool_call_parser': 'openai'},
                      extra_args=['--max-cudagraph-capture-size', '16', '--no-enable-prefix-caching'])
        if backend != 'auto':
            config['extra_args'] += ['--moe-backend', backend]
    else:
        config.update(kv_cache_dtype='q8_0', memory={'request': '80GiB'},
                      llamacpp={'gguf_file': 'gpt-oss-20b-MXFP4.gguf'},
                      extra_args=['--flash-attn', 'on',
                                  '--jinja', '--reasoning-format', 'deepseek'])
        if backend == 'eagle3':
            config['extra_args'] += ['--spec-type', 'draft-eagle3', '--spec-draft-model',
                                     str(Path.home()/'models/gpt-oss-20b-gguf/eagle3-gpt-oss-20b-Q8_0.gguf'),
                                     '--spec-draft-n-max', '3']
    if key in DRAFTS:
        draft = DRAFTS[key]
        if engine == 'sglang':
            config['extra_args'] += ['--speculative-algorithm', draft['kind'].upper(),
                                     '--speculative-draft-model-path', draft['path'],
                                     '--speculative-draft-model-quantization', 'unquant']
            if draft['kind'] == 'dflash':
                config['extra_args'] += ['--speculative-dflash-block-size', str(draft['block_size'])]
            else:
                config['extra_args'] += ['--speculative-eagle-topk', '1',
                                         '--speculative-num-steps', str(draft['steps']),
                                         '--speculative-num-draft-tokens', str(draft['steps']+1)]
        else:
            spec = {'method': draft['kind'], 'model': draft['path'],
                    'num_speculative_tokens': draft['block_size']-1 if draft['kind'] == 'dflash' else draft['steps']}
            config['extra_args'] += ['--speculative-config', json.dumps(spec)]
    return {'schema_version': 1, 'kind': 'deployment', 'name': 'gptoss-'+key,
            'engine': profile, 'model': model, 'residency': 'restart_only', 'lmcache': 'off',
            'timeouts': {'initialize': '1800s'}, 'engine_config': config}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('key', choices=ROWS)
    p.add_argument('--checkpoint')
    a = p.parse_args()
    print(json.dumps(deployment(a.key, a.checkpoint), indent=2))
