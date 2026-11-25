#!/usr/bin/env python3
"""Canonical streaming c1/c2/c4/c8 and exact 8k/32k/64k decode; no prompt above 64k."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--context', type=int, default=98304)
    parser.add_argument('--endpoint', default='http://127.0.0.1:18631/v1')
    parser.add_argument('--model', default='gemma31')
    parser.add_argument('--api-key-file', type=Path)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    cb = load('canonical_bench', ROOT / 'tools/capyctl-bench/capyctl_bench.py')
    depth = load('canonical_depth', HERE.parent / 'gemma-4-26b-a4b-nvfp4-builds-gb10/long-depth.py')
    key = cb.read_api_key(str(args.api_key_file)) if args.api_key_file else None
    runner = cb.Runner(argparse.Namespace(model=args.model, temperature=0., extra_body=None, record_chunks=False),
                       cb.Endpoint(args.endpoint, key, timeout=2400), None)
    result = {'method': 'Three measured rounds after one short warmup per concurrency. Native usage tokens; 512 output-token cap. Default thinking. Decode excludes TTFT; aggregate includes prefill and queueing.',
              'client_timeout_s': 2400, 'levels': []}
    prompts = cb.load_prompts(ROOT / 'tools/capyctl-bench/prompts.json')

    def group(label, concurrency, inputs, target=None):
        rounds = [runner.round_(concurrency, rows, 0, 512) for rows in inputs]
        requests = [q for r in rounds for q in r['requests']]
        valid = all(not q.get('error') and not q.get('usage_missing') and q.get('decode_tps')
                    and (target is None or q.get('prompt_tokens') == target) for q in requests)
        aggregates = [sum(q.get('completion_tokens') or 0 for q in r['requests']) /
                      (max(q['sent_at'] + q['total_s'] for q in r['requests']) - min(q['sent_at'] for q in r['requests']))
                      for r in rounds]
        result['levels'].append({'label': label, 'concurrency': concurrency, 'target_prompt_tokens': target,
                                 'valid': valid, 'aggregate_tps': statistics.median(aggregates) if valid else None,
                                 'decode_tps': statistics.median(q['decode_tps'] for q in requests) if valid else None,
                                 'ttft_s': statistics.median(q['ttft_s'] for q in requests) if valid else None,
                                 'rounds': rounds})
        args.out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: v for k, v in result['levels'][-1].items() if k != 'rounds'}), flush=True)
        if not valid:
            raise RuntimeError('HTTP/usage/exact prompt count failure; retained evidence')

    offset = 0
    for concurrency in (1, 2, 4, 8):
        runner.round_(concurrency, prompts, offset, 512)
        offset += concurrency
        rounds = []
        for _ in range(3):
            rounds.append([prompts[(offset + i) % len(prompts)] for i in range(concurrency)])
            offset += concurrency
        group('short', concurrency, rounds)
    for target in (8000, 32000, 64000):
        if target + 512 > args.context:
            result['levels'].append({'label': 'long', 'target_prompt_tokens': target, 'status': 'outside unchanged recipe context'})
            args.out.write_text(json.dumps(result, indent=2) + '\n')
            continue
        rounds = []
        for index in range(3):
            seed = 313100 + target + index
            messages, count = depth.prepare(tokenizer, cb, target, seed)
            rounds.append([{'id': f'{target}-{seed}', 'messages': messages,
                            'messages_sha256': hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()}])
            if count != target or count > 64000:
                raise ValueError('unsafe or inexact input')
        group('long', 1, rounds, target)


if __name__ == '__main__':
    main()
