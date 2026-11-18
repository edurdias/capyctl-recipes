#!/usr/bin/env python3
"""Prepare or measure fixed short-document, random-hex exact-copy fidelity."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re


def helper(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(args, gate):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    benchmark = gate.load_benchmark()
    documents = []
    for seed in range(200001, 200001 + args.count):
        rng = random.Random(seed)
        code = format(rng.getrandbits(64), '016X')
        depth = rng.uniform(.1, .9)
        filler = benchmark.filler_text(1000, seed)
        offsets = tokenizer(filler, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
        prefix = f'Document {seed}.\nBEGIN DOCUMENT\n'
        suffix = '\nEND DOCUMENT\n\nreturn the code exactly'

        def assemble(size, padding=''):
            position = offsets[int(size * depth)][0]
            end = offsets[size - 1][1]
            document = filler[:position] + f'\nCode: {code}\n' + filler[position:end] + padding
            return [{'role': 'user', 'content': prefix + document + suffix}]

        low, high = 1, min(len(offsets), args.tokens + 4096)
        if gate.token_count(tokenizer, assemble(high)) <= args.tokens:
            raise ValueError('Filler is too short')
        while low + 1 < high:
            middle = (low + high) // 2
            if gate.token_count(tokenizer, assemble(middle)) <= args.tokens:
                low = middle
            else:
                high = middle
        messages = assemble(low)
        count = gate.token_count(tokenizer, messages)
        padding, used = '', set()
        while count < args.tokens:
            for word in benchmark.WORDS:
                if word in used:
                    continue
                candidate = assemble(low, padding + ' ' + word)
                size = gate.token_count(tokenizer, candidate)
                if count < size <= args.tokens:
                    used.add(word)
                    padding += ' ' + word
                    messages, count = candidate, size
                    break
            else:
                raise ValueError('Exact short-context padding could not be constructed')
        documents.append({'document': f'copy-{seed}', 'filler_seed': seed,
                          'expected_code': code, 'target_depth_fraction': depth,
                          'local_prompt_tokens': count, 'messages': messages})
    result = {'method': 'Deterministic canonical pseudo-English filler, one seeded random uppercase 16-hex-digit code without a marker prefix, seeded depth between10% and90%, exact local chat-template sizing, fixed final prompt: return the code exactly.',
              'target_prompt_tokens': args.tokens, 'documents': documents}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'documents': len(documents), 'prompt_tokens': args.tokens,
                      'sha256': hashlib.sha256(args.out.read_bytes()).hexdigest()}))


def measure(args, gate):
    quality = helper('copy_quality', 'quality.py')
    benchmark = gate.load_benchmark()
    client = gate.Client(args.endpoint, args.model, benchmark.read_api_key(str(args.api_key_file) if args.api_key_file else None))
    fixtures = json.loads(args.fixtures.read_text())
    result = {'method': 'One serial observation per fixed document; temperature0, thinking off, output cap64. Primary exact copy: stripped full answer equals expected case-sensitive16-hex-digit string. Exact-code presence is separately reported to distinguish formatting from character-copy failures. Wilson95 binomial intervals use the existing quality.py helper.',
              'fixtures_sha256': hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
              'requests': [], 'completed': False}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.out.write_text(json.dumps(client.public(result), indent=2) + '\n')

    save()
    for document in fixtures['documents']:
        exchange = client.request({'messages': document['messages'], 'max_tokens': 64,
                                   'chat_template_kwargs': {'enable_thinking': False}})
        answer = gate.content_text(gate.assistant_message(exchange))
        ok = gate.request_ok(exchange)
        expected = document['expected_code']
        actual = ((exchange.get('response') or {}).get('usage') or {}).get('prompt_tokens')
        result['requests'].append({'document': document['document'], 'expected_code': expected,
                                   'answer': answer, 'exact_copy': ok and answer.strip() == expected,
                                   'exact_code_present': ok and bool(re.search(r'\b' + expected + r'\b', answer)),
                                   'prompt_tokens_match': actual == document['local_prompt_tokens'],
                                   'exchange': exchange})
        save()
    count = len(result['requests'])
    result['summary'] = {'documents': count, 'http_errors': sum(not gate.request_ok(request['exchange']) for request in result['requests']),
                         'prompt_token_mismatches': sum(not request['prompt_tokens_match'] for request in result['requests'])}
    for metric in ('exact_copy', 'exact_code_present'):
        successes = sum(request[metric] for request in result['requests'])
        result['summary'][metric] = {'successes': successes, 'n': count,
                                     'rate': successes / count, 'wilson95': quality.wilson(successes, count)}
    result['completed'] = True
    save()
    print(json.dumps(result['summary']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--recipes', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--endpoint')
    parser.add_argument('--model')
    parser.add_argument('--api-key-file', type=Path)
    parser.add_argument('--fixtures', type=Path, default=Path.home() / 'gemma4-engine-search/reference-data/copy-fidelity-2000.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--tokens', type=int, default=2000)
    args = parser.parse_args()
    if args.count < 1 or args.tokens < 100:
        parser.error('Positive document count and at least100 tokens are required')
    if args.prepare and not args.checkpoint:
        parser.error('--checkpoint is required for preparation')
    if not args.prepare and (not args.endpoint or not args.model):
        parser.error('--endpoint and --model are required for measurement')
    gate = helper('copy_gate', 'acceptance-gates.py')
    if args.prepare:
        prepare(args, gate)
    else:
        measure(args, gate)


if __name__ == '__main__':
    main()
