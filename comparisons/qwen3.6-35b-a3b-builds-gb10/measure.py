#!/usr/bin/env python3
"""Usage-token throughput, TTFT, native acceptance, exact-context decode."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import importlib.util
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--endpoint', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--api-key-file', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--context-prompts', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--phase', choices=('speed', 'long', 'published8k', 'shipped'), default='speed')
    p.add_argument('--native-metrics-endpoint', default='http://127.0.0.1:8300')
    a = p.parse_args()
    cb = load('capyctl_bench', ROOT/'tools/capyctl-bench/capyctl_bench.py')
    metrics = load('spec_metrics', HERE.parent/'gpt-oss-20b-builds-gb10/spec_metrics.py')
    key = cb.read_api_key(str(a.api_key_file) if a.api_key_file else None)
    runner = cb.Runner(argparse.Namespace(model=a.model, temperature=0.0, extra_body=None, record_chunks=False),
                       cb.Endpoint(a.endpoint, key, timeout=1800), None)
    contexts = json.loads(a.context_prompts.read_text())
    result = {'method': 'canonical capyctl-bench streaming timings; native usage tokens including thinking; default thinking',
              'phase': a.phase, 'temperature': 0, 'output_tokens': 256 if a.phase == 'published8k' else 512, 'levels': [],
              'aggregate_definition': 'sum(completion_tokens)/(latest completion - earliest send), median of three rounds',
              'decode_definition': '(completion_tokens-1)/(last text chunk-first text chunk)'}
    def save():
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=2)+'\n')
    def group(label, c, rounds, target=None):
        before = metrics.snapshot(a.native_metrics_endpoint)
        output_tokens = 256 if a.phase == 'published8k' else (512 if a.phase in ('long', 'shipped') or target is None else 128)
        measured = [runner.round_(c, rows, 0, output_tokens) for rows in rounds]
        acceptance = metrics.span(before, metrics.snapshot(a.native_metrics_endpoint))
        requests = [q for r in measured for q in r['requests']]
        valid = all(not q.get('error') and not q.get('usage_missing') and q.get('decode_tps')
                    and (target is None or q.get('prompt_tokens') == target) for q in requests)
        aggregates = []
        for r in measured:
            qs = r['requests']
            wall = max(q['sent_at']+q['total_s'] for q in qs)-min(q['sent_at'] for q in qs)
            aggregates.append(sum(q.get('completion_tokens') or 0 for q in qs)/wall)
        result['levels'].append({'label': label, 'streams': c, 'target_prompt_tokens': target, 'valid': valid,
              'aggregate_tps': statistics.median(aggregates) if valid else None,
              'decode_tps': statistics.median(q['decode_tps'] for q in requests) if valid else None,
              'ttft_s': statistics.median(q['ttft_s'] for q in requests) if valid else None,
              'acceptance': acceptance, 'rounds': measured,
              'input_references': [[{k:v for k,v in row.items() if k != 'messages'} for row in rows] for rows in rounds]})
        save()
        if not valid:
            raise RuntimeError('request failed or exact prompt length/usage missing; saved evidence')
    if a.phase in ('speed', 'shipped'):
        prompt_file = ROOT/'tools/capyctl-bench/prompts.json'
        prompts = cb.load_prompts(prompt_file)
        result['short_prompts_sha256'] = hashlib.sha256(prompt_file.read_bytes()).hexdigest()
        offset = 0
        for c in (1, 2, 4, 8):
            if a.phase == 'shipped' and c in (2, 4):
                offset += c * (a.rounds + 1)
                continue
            runner.round_(c, prompts, offset, 512)
            offset += c
            rounds = []
            for r in range(a.rounds):
                rounds.append([prompts[(offset+i) % len(prompts)] for i in range(c)])
                offset += c
            group('short', c, rounds)
        if a.phase == 'shipped':
            group('long', 1, contexts['long']['110000'][:1], 110000)
        else:
            for c in (1, 4):
                group('24k', c, contexts['levels'][str(c)], 24000)
    elif a.phase == 'published8k':
        group('published8k', 1, contexts['long']['8000'], 8000)
    else:
        for target, rounds in contexts['long'].items():
            group('long', 1, rounds, int(target))
    print(json.dumps({'phase': a.phase, 'levels': len(result['levels']), 'valid': True}), flush=True)

if __name__ == '__main__':
    main()
