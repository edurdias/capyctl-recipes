#!/usr/bin/env python3
"""Gemma reasoning/tools, six-document 32k/64k retrieval, and 100-case copy probes."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import time
import urllib.error
import urllib.request
from measure import load

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--retrieval-copy-only', action='store_true')
    parser.add_argument('--context', type=int, default=98304)
    parser.add_argument('--endpoint', default='http://127.0.0.1:18631')
    parser.add_argument('--model', default='gemma31')
    parser.add_argument('--api-key-file', type=Path)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    gate = load('gemma_gates', HERE.parent / 'gemma-4-26b-a4b-nvfp4-builds-gb10/acceptance-gates.py')
    cb = gate.load_benchmark()
    key = cb.read_api_key(str(args.api_key_file)) if args.api_key_file else None

    class Client(gate.Client):
        def request(self, body):
            body = {'model': self.model, 'temperature': 0, 'stream': False, **body}
            record = {'request': body, 'response': None, 'http_status': None, 'error': None}
            started = time.monotonic()
            try:
                headers = {'Content-Type': 'application/json'}
                if self.key:
                    headers['Authorization'] = 'Bearer ' + self.key
                request = urllib.request.Request(self.url, json.dumps(body).encode(), headers)
                with urllib.request.urlopen(request, timeout=2400) as response:
                    record.update(http_status=response.status, response=json.load(response))
            except urllib.error.HTTPError as exc:
                record.update(http_status=exc.code, error=type(exc).__name__)
            except (OSError, ValueError) as exc:
                record['error'] = type(exc).__name__
            record['seconds'] = time.monotonic() - started
            return record

    client = Client(args.endpoint, args.model, key)
    result = {'method': 'Gemma default thinking except explicit reasoning-on probe; 2400-second timeout; retrieval output cap is min(2048, context-input); copy cap 2048. Six canonical document seeds at exact 32000 and 64000 input tokens where the unchanged recipe admits them, each with original and one-line request. 100 seeded 2000-token copy cases.',
              'context': args.context, 'unsupported_lengths': [],
              'completed': False, 'reasoning': None, 'tools': [], 'retrieval': [], 'copy': []}

    def persist():
        args.out.write_text(json.dumps(client.public(result), indent=2) + '\n')

    def count(messages):
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        return len(ids['input_ids'] if hasattr(ids, 'keys') else ids)

    def exact_document(target, seed, depths, codes, suffix):
        filler = cb.filler_text(20000 if target > 2000 else 1000, seed)
        offsets = tokenizer(filler, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
        def assemble(size, padding=''):
            cursor, parts = 0, []
            for depth, code in zip(depths, codes):
                position = offsets[int(size * depth)][0]
                parts.extend((filler[cursor:position], '\nRetrieval record: ' + code + '\n'))
                cursor = position
            parts.append(filler[cursor:offsets[size - 1][1]] + padding)
            return [{'role': 'user', 'content': f'Document {seed}.\nBEGIN DOCUMENT\n' + ''.join(parts) + '\nEND DOCUMENT\n\n' + suffix}]
        low, high = 1, min(len(offsets), target + 4096)
        while low + 1 < high:
            middle = (low + high) // 2
            if count(assemble(middle)) <= target:
                low = middle
            else:
                high = middle
        messages, padding, used = assemble(low), '', set()
        actual = count(messages)
        while actual < target:
            for word in cb.WORDS:
                if word in used:
                    continue
                candidate = assemble(low, padding + ' ' + word)
                size = count(candidate)
                if actual < size <= target:
                    used.add(word)
                    padding += ' ' + word
                    messages, actual = candidate, size
                    break
            else:
                raise ValueError('exact input padding failed')
        if actual != target or actual > 64000:
            raise ValueError('unsafe input length')
        return messages

    if not args.retrieval_copy_only:
        result['reasoning'] = gate.reasoning_probe(client)
        persist()
        for _ in range(3):
            result['tools'].append(gate.tool_probe(client))
            persist()
    seeds = json.loads((HERE / 'retrieval-seeds.json').read_text())
    for length in (32000, 64000):
        if length >= args.context:
            result['unsupported_lengths'].append(length)
            persist()
            continue
        for document in seeds:
            for variant in ('original', 'one-line'):
                suffix = ('Request: return only the three retrieval record codes in document order.' if variant == 'original'
                          else 'Request: return exactly one line containing the three retrieval record codes in document order, separated by single spaces. Do not include any explanation.')
                messages = exact_document(length, document['filler_seed'], document['depths'], document['expected_codes'], suffix)
                exchange = client.request({'messages': messages, 'max_tokens': min(2048, args.context - length)})
                answer = gate.content_text(gate.assistant_message(exchange))
                codes = re.findall(gate.CODE_PATTERN, answer)
                expected = document['expected_codes']
                ok = gate.request_ok(exchange)
                result['retrieval'].append({'probe_id': document['probe_id'], 'filler_seed': document['filler_seed'],
                    'variant': variant, 'target_prompt_tokens': length,
                    'messages_sha256': hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
                    'expected_codes': expected, 'extracted_codes': codes,
                    'codes_found': sum(code in codes for code in expected) if ok else 0,
                    'all_three_present': ok and all(code in codes for code in expected),
                    'exact_ordered_codes': ok and codes == expected, 'exchange': exchange})
                persist()
    for seed in range(200001, 200101):
        rng = random.Random(seed)
        code, depth = format(rng.getrandbits(64), '016X'), rng.uniform(.1, .9)
        messages = exact_document(2000, seed, (depth,), (code,), 'Return the code exactly.')
        exchange = client.request({'messages': messages, 'max_tokens': 2048})
        answer = gate.content_text(gate.assistant_message(exchange))
        result['copy'].append({'filler_seed': seed, 'expected_code': code,
                               'exact_copy': gate.request_ok(exchange) and answer.strip() == code,
                               'exact_code_present': gate.request_ok(exchange) and bool(re.search(r'\b' + code + r'\b', answer)),
                               'exchange': exchange})
        persist()
    result['completed'] = True
    persist()
    print(json.dumps({'completed': True, 'retrieval_observations': len(result['retrieval']),
                      'copy_observations': len(result['copy'])}), flush=True)


if __name__ == '__main__':
    main()
