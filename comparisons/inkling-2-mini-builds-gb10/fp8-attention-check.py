#!/usr/bin/env python3
"""Loopback-only short FP8 attention screen; not a qualification benchmark."""
import argparse
import concurrent.futures
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path
from transformers import AutoTokenizer

TOPICS = [
    'Explain TCP slow start with a numerical example.',
    'Write a Python LRU cache and explain its complexity.',
    'Prove that there are infinitely many prime numbers.',
    'Design a SQL schema for books, members, and library loans.',
    'Describe the lifecycle of a tomato plant in a temperate garden.',
    'Write a story about a lighthouse keeper finding a mysterious letter.',
    'Compare RAID 1, RAID 5, RAID 6, and RAID 10 for four disks.',
    'Explain why the printing press changed European society.',
    'Solve 3x + 17 = 65 and verify the result.',
    'Given a calendar lookup tool, explain how to schedule a meeting without conflicts.',
]
VARIANTS = ['Give a careful answer.', 'Use a concrete worked example.', 'State your assumptions explicitly.', 'Include the important edge cases.', 'Explain it to an engineering student.']

def request(payload, stream=False):
    req = urllib.request.Request('http://127.0.0.1:18130/generate', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=3600) as response:
        if not stream:
            return json.load(response)
        first = last = None
        final = None
        previous_count = 0
        for raw in response:
            if not raw.startswith(b'data: '):
                continue
            data = raw[6:].strip()
            if data == b'[DONE]':
                break
            final = json.loads(data)
            count = final['meta_info']['completion_tokens']
            if count > previous_count:
                now = time.monotonic()
                first = now if first is None else first
                last = now
                previous_count = count
    ended = time.monotonic()
    n = final['meta_info']['completion_tokens']
    return {'completion_tokens': n, 'prompt_tokens': final['meta_info']['prompt_tokens'], 'total_s': ended-started, 'ttft_s': first-started, 'decode_tps': (n-1)/(last-first) if n>1 and last>first else None}

def greedy(ids):
    response = request({'input_ids': ids, 'sampling_params': {'temperature': 0, 'max_new_tokens': 64, 'ignore_eos': True}, 'return_logprob': True, 'top_logprobs_num': 1, 'logprob_start_len': -1})
    tokens = [entry[1] for entry in response['meta_info']['output_token_logprobs']]
    if len(tokens) != 64:
        raise ValueError(f'Expected 64 greedy tokens, got {len(tokens)}')
    return {'tokens': tokens, 'text': response.get('text'), 'meta_info': response['meta_info']}

def forced(ids, continuation):
    # The native API includes a null for the first position of the scored range.
    # Start at the last prompt token, then score all 64 continuation positions.
    response = request({'input_ids': ids+continuation, 'sampling_params': {'temperature': 0, 'max_new_tokens': 0}, 'return_logprob': True, 'top_logprobs_num': 1, 'logprob_start_len': len(ids)-1})
    rows = response['meta_info']['input_top_logprobs']
    if len(rows) != 65 or rows[0] is not None:
        raise ValueError(f'Unexpected teacher-forcing null-prefix alignment: {rows[:2]}, rows={len(rows)}')
    tokens = [row[0][1] for row in rows[1:]]
    if len(tokens) != 64:
        raise ValueError(f'Expected 64 teacher-forced positions, got {len(tokens)}; meta keys={list(response["meta_info"])}')
    return {'top1': tokens, 'meta_info': response['meta_info']}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['baseline', 'candidate', 'speed'])
    parser.add_argument('--checkpoint', type=Path, default=Path.home()/'models/Inkling-2-Mini-ed40e7da')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = {'phase': args.phase, 'method': '50 fixed short prompts; greedy first 64 tokens including reasoning; teacher forcing on original greedy continuations; temperature zero; no EOS early stop', 'rows': []}
    def save():
        args.out.write_text(json.dumps(result, indent=2)+'\n')
    if args.phase in ('baseline', 'candidate'):
        baseline = json.loads(args.baseline.read_text()) if args.baseline else None
        prompts = [(topic+' '+variant) for variant in VARIANTS for topic in TOPICS]
        result['prompts_sha256'] = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
        for i, prompt in enumerate(prompts):
            ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True, add_generation_prompt=True, return_dict=False)
            generated = greedy(ids)
            continuation = baseline['rows'][i]['greedy']['tokens'] if baseline else generated['tokens']
            row = {'index': i, 'prompt': prompt, 'input_ids': ids, 'greedy': generated}
            result['rows'].append(row)
            save()
            teacher = forced(ids, continuation)
            row['teacher_forced'] = teacher
            if baseline:
                original = baseline['rows'][i]
                if original['input_ids'] != ids:
                    raise ValueError('Baseline prompt IDs changed')
                reference = original['greedy']['tokens']
                row['greedy_position_matches'] = sum(a==b for a,b in zip(reference, generated['tokens']))
                row['greedy_exact_sequence'] = reference == generated['tokens']
                row['teacher_top1_matches'] = sum(a==b for a,b in zip(original['teacher_forced']['top1'], teacher['top1']))
                row['greedy_common_prefix'] = next((j for j,(a,b) in enumerate(zip(reference, generated['tokens'])) if a!=b), 64)
            save()
            print(args.phase, i+1, flush=True)
        if baseline:
            result['summary'] = {'prompts': 50, 'positions': 3200, 'teacher_top1_matches': sum(r['teacher_top1_matches'] for r in result['rows']), 'teacher_top1_agreement': sum(r['teacher_top1_matches'] for r in result['rows'])/3200, 'greedy_position_matches': sum(r['greedy_position_matches'] for r in result['rows']), 'greedy_position_agreement': sum(r['greedy_position_matches'] for r in result['rows'])/3200, 'greedy_exact_sequences': sum(r['greedy_exact_sequence'] for r in result['rows']), 'mean_greedy_common_prefix': statistics.mean(r['greedy_common_prefix'] for r in result['rows'])}
    else:
        result['method'] = 'One unmeasured warmup per concurrency, one measured round, 256 generated tokens; aggregate tokens / request wall span; native streaming first-to-last-token decode'
        ids = [tokenizer.apply_chat_template([{'role': 'user', 'content': p}], tokenize=True, add_generation_prompt=True, return_dict=False) for p in TOPICS]
        def run_one(prompt_ids):
            return request({'input_ids': prompt_ids, 'sampling_params': {'temperature': 0, 'max_new_tokens': 256, 'ignore_eos': True}, 'stream': True}, stream=True)
        for concurrency in (1, 8):
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                list(pool.map(run_one, ids[:concurrency]))
                started = time.monotonic()
                rows = list(pool.map(run_one, ids[1:concurrency+1]))
                wall = time.monotonic()-started
            result['rows'].append({'label': 'short', 'concurrency': concurrency, 'aggregate_tps': sum(r['completion_tokens'] for r in rows)/wall, 'decode_tps_median': statistics.median(r['decode_tps'] for r in rows), 'requests': rows})
            save()
        seed = tokenizer.encode('A long technical document discusses reliable distributed systems, storage consistency, and network failures. ', add_special_tokens=False)
        long_ids = (seed * (32000//len(seed)+1))[:32000]
        row = run_one(long_ids)
        if row['prompt_tokens'] != 32000:
            raise ValueError('32k native prompt count mismatch')
        result['rows'].append({'label': '32k', 'concurrency': 1, **row, 'input_ids_sha256': hashlib.sha256(json.dumps(long_ids).encode()).hexdigest()})
    save()
    print(json.dumps(result.get('summary', {'completed': args.phase})), flush=True)

if __name__ == '__main__':
    main()
