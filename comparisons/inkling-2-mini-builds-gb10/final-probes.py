#!/usr/bin/env python3
"""Seeded retrieval and copy probes, using the existing Gemma/GPT-OSS corpus."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import re
from measure import load

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parent / 'gemma-4-26b-a4b-nvfp4-builds-gb10'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.checkpoint, local_files_only=True)
    cb = load('probe_bench', HERE.parents[1] / 'tools/capyctl-bench/capyctl_bench.py')
    def count(messages):
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        return len(ids['input_ids'] if hasattr(ids, 'keys') else ids)
    def document(target, seed, depths, codes, prefix, suffix):
        filler = cb.filler_text(40000 if target > 200000 else 20000, seed)
        offsets = tok(filler, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
        def assemble(size, padding=''):
            cursor, parts = 0, []
            for depth, code in zip(depths, codes):
                pos = offsets[int(size * depth)][0]
                parts.extend((filler[cursor:pos], f'\nRetrieval record: {code}\n' if len(codes) == 3 else f'\nCode: {code}\n'))
                cursor = pos
            tail = filler[cursor:offsets[size-1][1]]
            boundary = tail.rfind('.')
            tail = tail[:boundary] + padding + tail[boundary:] if boundary >= 0 else tail + padding
            return [{'role': 'user', 'content': prefix + ''.join(parts) + tail + suffix}]
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
                n = count(candidate)
                if actual < n <= target:
                    messages, actual = candidate, n
                    used.add(word)
                    padding += ' ' + word
                    break
            else:
                raise ValueError(f'Cannot pad {seed} to {target}; actual={actual}')
        return messages
    originals = json.loads((CORPUS/'bench/reference-data/retrieval-six-32000.json').read_text())['documents']
    fixtures = {'retrieval': [], 'copy': [], 'method': 'Default Inkling high (0.9) effort; exact chat-template token counts; existing seeded retrieval corpus.'}
    for length in (32000, 64000, 96000, 111000, 220000):
        for original in originals:
            seed = original['filler_seed']
            depths = [i['target_depth_fraction'] for i in original['insertions']]
            messages = document(length, seed, depths, original['expected_codes'],
                f'Document {seed}. Read this document and find its three retrieval records.\nReturn their codes in document order, exactly as written.\n\nBEGIN DOCUMENT\n',
                '\nEND DOCUMENT\n\nRequest: return only the three retrieval record codes in document order.')
            fixtures['retrieval'].append({'document': original['probe_id'], 'filler_seed': seed, 'expected_codes': original['expected_codes'], 'target_depth_fractions': depths, 'prompt_tokens': length, 'messages': messages})
    for seed in range(200001, 200101):
        rng = random.Random(seed)
        code, depth = format(rng.getrandbits(64), '016X'), rng.uniform(.1, .9)
        messages = document(2000, seed, [depth], [code], f'Document {seed}.\nBEGIN DOCUMENT\n', '\nEND DOCUMENT\n\nreturn the code exactly')
        fixtures['copy'].append({'document': f'copy-{seed}', 'filler_seed': seed, 'expected_code': code, 'target_depth_fraction': depth, 'prompt_tokens': 2000, 'messages': messages})
    save(a.out, fixtures)
    manifest = copy.deepcopy(fixtures)
    for category in ('retrieval', 'copy'):
        for row in manifest[category]:
            row['messages_sha256'] = hashlib.sha256(json.dumps(row.pop('messages'), ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    save(a.out.with_name('fixtures.manifest.json'), manifest)


def measure(a):
    gate = load('probe_gate', CORPUS/'acceptance-gates.py')
    cb = gate.load_benchmark()
    client = gate.Client(a.endpoint, a.model, cb.read_api_key(str(a.api_key_file) if a.api_key_file else None))
    fixtures = json.loads(a.fixtures.read_text())
    result = {'phase': a.phase, 'completed': False, 'thinking': 'default', 'temperature': 0, 'output_cap': 2048, 'requests': []}
    for row in fixtures[a.phase]:
        for variant in (('original', 'one-line') if a.phase == 'retrieval' else ('original',)):
            messages = copy.deepcopy(row['messages'])
            if variant == 'one-line':
                document, _ = messages[0]['content'].rsplit('\nEND DOCUMENT\n\n', 1)
                messages[0]['content'] = document + '\nEND DOCUMENT\n\nRequest: return exactly one line containing the three retrieval record codes in document order, separated by single spaces. Do not include any explanation.'
            exchange = client.request({'messages': messages, 'max_tokens': 2048})
            exchange['request'].pop('messages', None)
            answer = gate.content_text(gate.assistant_message(exchange))
            ok = gate.request_ok(exchange)
            usage = (exchange.get('response') or {}).get('usage') or {}
            choices = (exchange.get('response') or {}).get('choices') or [{}]
            truncated = choices[0].get('finish_reason') == 'length'
            ok = ok and not truncated
            item = {k: v for k, v in row.items() if k != 'messages'}
            item.update(variant=variant, answer=answer, exchange=exchange, actual_prompt_tokens=usage.get('prompt_tokens'))
            item['truncated'] = truncated
            item['messages_sha256'] = hashlib.sha256(json.dumps(messages, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
            if a.phase == 'retrieval':
                codes = re.findall(gate.CODE_PATTERN, answer)
                expected = row['expected_codes']
                item.update(extracted_codes=codes, codes_found=sum(code in codes for code in expected) if ok else 0, all_three_present=ok and all(code in codes for code in expected), exact_ordered_codes=ok and codes == expected)
            else:
                item.update(exact_copy=ok and answer.strip() == row['expected_code'], exact_code_present=ok and bool(re.search(r'\b'+row['expected_code']+r'\b', answer)))
            result['requests'].append(item)
            save(a.out, client.public(result))
    result['completed'] = True
    result['http_errors'] = sum(not gate.request_ok(q['exchange']) for q in result['requests'])
    save(a.out, client.public(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('prepare', 'retrieval', 'copy'))
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--fixtures', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--endpoint')
    p.add_argument('--model')
    p.add_argument('--api-key-file', type=Path)
    a = p.parse_args()
    prepare(a) if a.phase == 'prepare' else measure(a)
