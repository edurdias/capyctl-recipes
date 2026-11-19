#!/usr/bin/env python3
"""Final GPT-OSS serial probes; reuse the canonical Gemma inputs and timing client."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import random
import re
import statistics
import sys

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / 'gemma-4-26b-a4b-nvfp4-builds-gb10'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def prepare(args, gate, cb):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    # GPT-OSS uses reasoning_effort, not Gemma's enable_thinking switch.
    def count(tok, messages):
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, reasoning_effort='low')
        return len(ids['input_ids'] if hasattr(ids, 'keys') else ids)
    # Preserve GPT-OSS's period/newline suffix when adding exact-token padding.
    def exact_document(target, seed, depths, codes, prefix, suffix, sentences):
        filler = cb.filler_text(sentences, seed)
        offsets = tokenizer(filler, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
        def assemble(size, padding=''):
            end = offsets[size - 1][1]
            cursor, parts = 0, []
            for depth, code in zip(depths, codes):
                position = offsets[int(size * depth)][0]
                parts.extend((filler[cursor:position], f'\nRetrieval record: {code}\n' if len(codes) == 3 else f'\nCode: {code}\n'))
                cursor = position
            tail = filler[cursor:end]
            boundary = tail.rfind('.')
            tail = tail[:boundary] + padding + tail[boundary:] if boundary >= 0 else tail + padding
            parts.append(tail)
            return [{'role': 'user', 'content': prefix + ''.join(parts) + suffix}]
        low, high = 1, min(len(offsets), target + 4096)
        while low + 1 < high:
            middle = (low + high) // 2
            if count(tokenizer, assemble(middle)) <= target:
                low = middle
            else:
                high = middle
        messages, padding, used = assemble(low), '', set()
        actual = count(tokenizer, messages)
        while actual < target:
            for word in cb.WORDS:
                if word in used:
                    continue
                candidate = assemble(low, padding + ' ' + word)
                candidate_count = count(tokenizer, candidate)
                if actual < candidate_count <= target:
                    used.add(word)
                    padding += ' ' + word
                    messages, actual = candidate, candidate_count
                    break
            else:
                raise ValueError(f'Exact padding failed at {target}, seed {seed}, actual {actual}')
        return messages
    originals = json.loads((REFERENCE / 'bench/reference-data/retrieval-six-32000.json').read_text())['documents']
    fixtures = {'retrieval': [], 'decode': {}, 'copy': []}
    for length in (32000, 64000, 96000, 111000):
        for original in originals:
            depths = tuple(item['target_depth_fraction'] for item in original['insertions'])
            seed = original['filler_seed']
            messages = exact_document(length, seed, depths, original['expected_codes'],
                                      f'Document {seed}. Read this document and find its three retrieval records.\nReturn their codes in document order, exactly as written.\n\nBEGIN DOCUMENT\n',
                                      '\nEND DOCUMENT\n\nRequest: return only the three retrieval record codes in document order.', 16000)
            document = {key: original[key] for key in ('probe_id', 'filler_seed', 'expected_codes')}
            document.update(messages=messages, local_prompt_tokens=length,
                            target_depth_fractions=list(depths))
            fixtures['retrieval'].append({**document, 'target_prompt_tokens': length})
    for length in (500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 110000):
        messages = exact_document(length, 24010000, (), (),
                                  'Read the document.\nBEGIN DOCUMENT\n',
                                  '\nEND DOCUMENT\nWrite a detailed numbered explanation of how to organize a community library, including staffing, cataloguing, lending, funding and outreach.', 20000)
        fixtures['decode'][str(length)] = {'messages': messages, 'prompt_tokens': length, 'filler_seed': 24010000}
    for seed in range(200001, 200101):
        rng = random.Random(seed)
        code = format(rng.getrandbits(64), '016X')
        depth = rng.uniform(.1, .9)
        messages = exact_document(2000, seed, (depth,), (code,),
                                  f'Document {seed}.\nBEGIN DOCUMENT\n',
                                  '\nEND DOCUMENT\n\nreturn the code exactly', 1000)
        fixtures['copy'].append({'document': f'copy-{seed}', 'filler_seed': seed,
                                 'expected_code': code, 'target_depth_fraction': depth,
                                 'local_prompt_tokens': 2000, 'messages': messages})
    save(args.out / 'fixtures.json', fixtures)
    print(json.dumps({'prepared': True, 'retrieval_documents': len(fixtures['retrieval']), 'copy_documents': len(fixtures['copy'])}), flush=True)


def nonempty_tool_probe(client, gate):
    code, name = gate.fresh_code(), 'read_acceptance_code'
    tools = [{'type': 'function', 'function': {
        'name': name, 'description': 'Read the acceptance code.',
        'parameters': {'type': 'object', 'properties': {'request': {'type': 'string'}},
                       'required': ['request'], 'additionalProperties': False}}}]
    messages = [{'role': 'user', 'content': 'Call read_acceptance_code exactly once with request set to "acceptance". After receiving its result, quote the returned code exactly in your final answer.'}]
    first = client.request({'messages': messages, 'tools': tools,
                            'tool_choice': {'type': 'function', 'function': {'name': name}},
                            'max_tokens': 256})
    assistant = gate.assistant_message(first)
    calls = assistant.get('tool_calls')
    valid = False
    if isinstance(calls, list) and len(calls) == 1 and isinstance(calls[0], dict):
        call = calls[0]
        function = call.get('function')
        if isinstance(function, dict):
            try:
                arguments = json.loads(function.get('arguments', ''))
            except (ValueError, TypeError):
                arguments = None
            valid = (assistant.get('role') == 'assistant' and call.get('type') == 'function'
                     and isinstance(call.get('id'), str) and bool(call['id'])
                     and function.get('name') == name and arguments == {'request': 'acceptance'})
    second, tool_message = None, None
    if gate.request_ok(first) and valid:
        tool_message = {'role': 'tool', 'tool_call_id': calls[0]['id'], 'content': json.dumps({'code': code})}
        second = client.request({'messages': messages + [assistant, tool_message],
                                 'tools': tools, 'tool_choice': 'none', 'max_tokens': 256})
    final = gate.assistant_message(second) if second else {}
    extracted = re.findall(gate.CODE_PATTERN, gate.content_text(final))
    conditions = {'initial_http_succeeded': gate.request_ok(first),
                  'exactly_one_valid_requested_tool_call': valid,
                  'tool_result_http_succeeded': second is not None and gate.request_ok(second),
                  'answer_quotes_exact_returned_code': extracted == [code],
                  'no_further_tool_calls': second is not None and not final.get('tool_calls')}
    return {'expected_code': code, 'initial_exchange': first,
            'assistant_tool_calls_message': assistant, 'tool_result_message': tool_message,
            'answer_exchange': second, 'extracted_codes': extracted,
            'pass_conditions': conditions, 'passed': all(conditions.values())}


def measure(args, gate, cb):
    fixtures = json.loads(args.fixtures.read_text())
    key = cb.read_api_key(str(args.api_key_file))
    client = gate.Client(args.endpoint, args.model, key)
    result = {'phase': args.phase, 'reasoning_effort': 'low', 'temperature': 0, 'completed': False, 'requests': []}
    def persist():
        save(args.out, client.public(result))
    class EffortClient:
        def __init__(self, effort='low'):
            self.effort = effort
        def request(self, body):
            body = {k: v for k, v in body.items() if k != 'chat_template_kwargs'}
            return client.request({**body, 'reasoning_effort': self.effort})
    effort_client = EffortClient()
    if args.phase in ('decode', 'crossover', 'drafter-decode', 'sweep'):
        runner = cb.Runner(argparse.Namespace(model=args.model, temperature=0., extra_body=json.dumps({'reasoning_effort': 'low'}), record_chunks=False), cb.Endpoint(args.endpoint, key, timeout=1800), None)
        lengths = {'decode': (8000, 32000, 64000, 110000),
                   'crossover': (1000, 2000, 4000),
                   'drafter-decode': (4000, 8000, 32000),
                   'sweep': (500, 1000, 2000, 4000, 8000, 16000, 32000)}[args.phase]
        result['method'] = 'Three streaming rounds; 512 completion-token cap; decode excludes first-token latency; c1-c8 aggregate also includes prefill. Shared fixed exact-token prompt; radix cache remains at production default.'
        for length in lengths:
            for concurrency in ((1, 2, 4, 8) if args.phase == 'sweep' else (1,)):
                prompt = {'id': f'context-{length}', 'messages': fixtures['decode'][str(length)]['messages']}
                rounds = [runner.round_(concurrency, [prompt] * concurrency, 0, 512) for _ in range(3)]
                requests = [q for round_ in rounds for q in round_['requests']]
                valid = all(not q.get('error') and not q.get('usage_missing') and q.get('prompt_tokens') == length for q in requests)
                aggregates = [sum(q['completion_tokens'] for q in r['requests']) / (max(q['sent_at'] + q['total_s'] for q in r['requests']) - min(q['sent_at'] for q in r['requests'])) for r in rounds] if valid else []
                result['requests'].append({'input_tokens': length, 'concurrency': concurrency, 'valid': valid, 'decode_tps': statistics.median(q['decode_tps'] for q in requests) if valid else None, 'aggregate_tps': statistics.median(aggregates) if valid else None, 'ttft_s': statistics.median(q['ttft_s'] for q in requests) if valid else None, 'rounds': rounds})
                persist()
                if not valid:
                    raise RuntimeError('Timing request failed or token count mismatched; saved evidence')
    elif args.phase == 'retrieval':
        result['method'] = 'Six seeded documents per length, regenerated with GPT-OSS tokenizer at effort low; one serial temperature-zero observation per variant, 2048 output-token cap including reasoning. Primary: all three exact codes in any order and total codes found; ordered exact secondary.'
        for fixture in fixtures['retrieval']:
            for variant in ('original', 'one-line'):
                messages = copy.deepcopy(fixture['messages'])
                if variant == 'one-line':
                    document, _ = messages[0]['content'].rsplit('\nEND DOCUMENT\n\n', 1)
                    messages[0]['content'] = document + '\nEND DOCUMENT\n\nRequest: return exactly one line containing the three retrieval record codes in document order, separated by single spaces. Do not include any explanation.'
                exchange = effort_client.request({'messages': messages, 'max_tokens': 2048})
                answer = gate.content_text(gate.assistant_message(exchange))
                codes = re.findall(gate.CODE_PATTERN, answer)
                expected = fixture['expected_codes']
                ok = gate.request_ok(exchange)
                result['requests'].append({'document': fixture['probe_id'], 'filler_seed': fixture['filler_seed'], 'target_prompt_tokens': fixture['target_prompt_tokens'], 'variant': variant, 'expected_codes': expected, 'extracted_codes': codes, 'codes_found': sum(code in codes for code in expected) if ok else 0, 'all_three_present': ok and all(code in codes for code in expected), 'exact_ordered_codes': ok and codes == expected, 'answer': answer, 'exchange': exchange})
                persist()
    elif args.phase == 'copy':
        quality = load('copy_quality', HERE / 'quality.py')
        result['method'] = '100 seeded 2000-token documents, one uppercase 16-hex-digit code each; serial temperature0 effort low; 2048-token cap includes reasoning. Strict stripped answer equals code; Wilson95 interval.'
        for fixture in fixtures['copy']:
            exchange = effort_client.request({'messages': fixture['messages'], 'max_tokens': 2048})
            answer = gate.content_text(gate.assistant_message(exchange))
            expected = fixture['expected_code']
            ok = gate.request_ok(exchange)
            result['requests'].append({'document': fixture['document'], 'expected_code': expected, 'answer': answer, 'exact_copy': ok and answer.strip() == expected, 'exact_code_present': ok and bool(re.search(r'\b' + expected + r'\b', answer)), 'exchange': exchange})
            persist()
        result['summary'] = {'documents': len(result['requests']), 'http_errors': sum(not gate.request_ok(q['exchange']) for q in result['requests'])}
        for metric in ('exact_copy', 'exact_code_present'):
            successes = sum(q[metric] for q in result['requests'])
            result['summary'][metric] = {'successes': successes, 'n': 100, 'rate': successes / 100, 'wilson95': quality.wilson(successes, 100)}
    elif args.phase == 'acceptance':
        result['reasoning'] = {effort: gate.reasoning_probe(EffortClient(effort)) for effort in ('low', 'medium')}
        result['tools'] = gate.tool_probe(effort_client)
        result['required_string_schema'] = nonempty_tool_probe(effort_client, gate)
    elif args.phase == 'tool-diagnostic':
        result['method'] = 'Owner-requested diagnostic, retaining the original failed shipped gate unchanged. Three identical forced empty-schema gates and one required-string-schema variant; temperature0, effort low, 256 output cap per turn.'
        result['empty_schema'] = []
        for _ in range(3):
            result['empty_schema'].append(gate.tool_probe(effort_client))
            persist()
        result['required_string_schema'] = nonempty_tool_probe(effort_client, gate)
        result['counts'] = {'empty_schema': {'passed': sum(q['passed'] for q in result['empty_schema']), 'n': 3},
                            'required_string_schema': {'passed': int(result['required_string_schema']['passed']), 'n': 1}}
    result['completed'] = True
    persist()
    print(json.dumps({'phase': args.phase, 'completed': True, 'requests': len(result['requests'])}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('prepare', 'decode', 'crossover', 'drafter-decode', 'retrieval', 'copy', 'sweep', 'acceptance', 'tool-diagnostic'))
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--fixtures', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--endpoint')
    p.add_argument('--model')
    p.add_argument('--api-key-file', type=Path)
    args = p.parse_args()
    gate = load('final_gate', REFERENCE / 'acceptance-gates.py')
    cb = gate.load_benchmark()
    if args.phase == 'prepare':
        args.out.mkdir(parents=True, exist_ok=True)
        prepare(args, gate, cb)
    else:
        measure(args, gate, cb)


if __name__ == '__main__':
    main()
