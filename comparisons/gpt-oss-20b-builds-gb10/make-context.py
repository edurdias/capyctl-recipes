#!/usr/bin/env python3
"""Prepare canonical pseudo-English documents at exactly 24,000 chat tokens.

Run in an existing engine environment before launch. No model weights or GPU
context are loaded. Different early document seeds prevent prefix-cache reuse.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--rounds', type=int, default=3)
parser.add_argument('--reasoning-effort', default='low', choices=('low', 'medium', 'high'))
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('capyctl_bench', root/'tools/capyctl-bench/capyctl_bench.py')
cb = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cb
spec.loader.exec_module(cb)
tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)


def count(messages):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, reasoning_effort=args.reasoning_effort)
    return len(ids['input_ids'] if hasattr(ids, 'keys') else ids)


def prepare(seed):
    low, high = 0, 4096
    while count(cb.context_messages(high, seed)) < 24000:
        high *= 2
    while low + 1 < high:
        middle = (low + high)//2
        if count(cb.context_messages(middle, seed)) <= 24000:
            low = middle
        else:
            high = middle
    messages = cb.context_messages(low, seed)
    remaining = 24000-count(messages)
    # Insert before the final period: GPT-OSS merges the period/newline suffix.
    # Appending after that period splits the suffix and adds an extra token.
    document, request = messages[0]['content'].rsplit('\n\n', 1)
    messages[0]['content'] = document[:-1] + ' a'*remaining + document[-1] + '\n\n' + request
    actual = count(messages)
    if actual != 24000:
        raise ValueError(f'prompt {seed}: exact padding failed, got {actual} tokens')
    return {'id': f'24k-{seed}', 'messages': messages, 'token_count': actual}


result = {'target_prompt_tokens': 24000, 'tokenizer_checkpoint': str(args.checkpoint), 'reasoning_effort': args.reasoning_effort, 'levels': {}}
for c in (1, 4):
    result['levels'][str(c)] = [[prepare(24000000+c*10000+r*100+i) for i in range(c)] for r in range(args.rounds)]
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(result, ensure_ascii=False)+'\n')
print(json.dumps({'out': str(args.out), 'prompts': args.rounds*5, 'tokens_each': 24000}))
