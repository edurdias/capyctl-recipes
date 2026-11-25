#!/usr/bin/env python3
"""Canonical 50/30/50 screen or 250/30/164 gate with Gemma's default thinking."""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import sys
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = ROOT / 'comparisons/qwen3.8-27b-sglang-dflash2-vs-mtp-eagle-gb10/accuracy'
spec = importlib.util.spec_from_file_location('canonical_accuracy', HERE.parent / 'gpt-oss-120b-builds-gb10/accuracy.py')
canonical = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canonical)
helper = canonical.helper


def execute(item, content):
    if os.geteuid() == 0:
        raise RuntimeError('generated code must never run as root')
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', content, re.S)
    code = blocks[0] if blocks else content
    program = code + '\n' + item['test'] + '\ncheck(' + item['entry_point'] + ')\n'
    with tempfile.TemporaryDirectory(prefix='gemma31-he-') as directory:
        script = Path(directory) / 'candidate.py'
        script.write_text(program)
        command = ['bash', '-c', 'ulimit -v 1048576; ulimit -t 10; ulimit -f 1024; ulimit -u 64; exec "$1" -B -I "$2"',
                   'humaneval', '/usr/bin/python3', str(script)]
        try:
            run = subprocess.run(command, cwd=directory,
                                 env={'PATH': '/usr/bin:/bin', 'HOME': directory,
                                      'LANG': 'C.UTF-8', 'OPENBLAS_NUM_THREADS': '1'},
                                 capture_output=True, timeout=15)
            return {'passed': run.returncode == 0, 'returncode': run.returncode,
                    'test_stderr': run.stderr.decode(errors='replace')[-2000:]}
        except subprocess.TimeoutExpired:
            return {'passed': False, 'timeout': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--screen', action='store_true')
    parser.add_argument('--api-key-file', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--context', type=int)
    args = parser.parse_args()
    if (args.checkpoint is None) != (args.context is None):
        parser.error('--checkpoint and --context must be supplied together')
    tokenizer = None
    if os.geteuid() == 0:
        parser.error('generated code must not run as root')
    headers = {'Content-Type': 'application/json'}
    if args.api_key_file:
        headers['Authorization'] = 'Bearer ' + helper.credential(str(args.api_key_file), 'api_key')
    # Keep the pinned dataset in this run's writable cache, not a shared cache.
    os.environ['HF_HOME'] = str(args.out.parent / 'hf-cache')
    if args.checkpoint is not None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    from datasets import load_dataset
    human = list(load_dataset('openai/openai_humaneval', revision=canonical.HUMANEVAL_REVISION, split='test'))
    data = (ASSETS / 'gsm8k-test-250.jsonl').read_bytes()
    if hashlib.sha256(data).hexdigest() != canonical.GSM_HASH or len(human) != 164:
        raise ValueError('canonical dataset mismatch')
    gsm = [json.loads(line) for line in data.splitlines() if line.strip()]
    tools = json.loads((ASSETS / 'tool-cases.json').read_text())['cases']
    if args.screen:
        gsm, human = gsm[:50], human[:50]
    canonical.save(args.out / 'method.json', {
        'temperature': 0, 'thinking': 'checkpoint chat-template default; no reasoning_effort override',
        'max_tokens': 32768, 'client_timeout_s': 2400, 'concurrency': 8,
        'scope': 'screen-50' if args.screen else 'full',
        'truncation_policy': 'every length-limited response is wrong',
        'gsm8k_subset_sha256': canonical.GSM_HASH,
        'humaneval_revision': canonical.HUMANEVAL_REVISION,
        'tools_sha256': hashlib.sha256((ASSETS / 'tool-cases.json').read_bytes()).hexdigest(),
        'humaneval_execution': 'canonical unprivileged subprocess; 15-second wall, 10-second CPU, 1GiB address-space limits; minimal environment',
        'output_budget_policy': 'min(32768, context - exact chat-template prompt tokens)' if tokenizer else '32768',
        'context': args.context,
    })

    def request(item, kind):
        if kind == 'gsm8k':
            prompt = item['question'] + '\nSolve it step by step, then give the final answer as a number in \\boxed{}.'
        elif kind == 'tools':
            prompt = item['prompt']
        else:
            prompt = 'Implement the following Python function. Return the complete function with any required imports in one Python code block.\n\n' + item['prompt']
        body = {'model': args.model, 'temperature': 0, 'max_tokens': 32768,
                'messages': [{'role': 'user', 'content': prompt}]}
        if kind == 'tools':
            body.update(tools=item['tools'], tool_choice='auto')
        prompt_tokens = None
        if tokenizer is not None:
            ids = tokenizer.apply_chat_template(
                body['messages'], tools=body.get('tools'), tokenize=True, add_generation_prompt=True)
            prompt_tokens = len(ids['input_ids'] if hasattr(ids, 'keys') else ids)
            body['max_tokens'] = min(32768, args.context - prompt_tokens)
            if body['max_tokens'] <= 0:
                raise ValueError('task prompt does not fit the unchanged recipe context')
        started = time.monotonic()
        try:
            req = urllib.request.Request(args.endpoint.rstrip('/') + '/v1/chat/completions',
                                         json.dumps(body).encode(), headers)
            with urllib.request.urlopen(req, timeout=2400) as response:
                status, response_data = response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            status, response_data = exc.code, {}
        except (OSError, ValueError):
            status, response_data = 0, {}
        message = helper.message_parts(response_data)
        row = {'id': str(item.get('task_id', item.get('id', item.get('index')))),
               'http_status': status, 'seconds': time.monotonic() - started, **message}
        if tokenizer is not None:
            row.update(local_prompt_tokens=prompt_tokens,
                       native_prompt_tokens=response_data.get('usage', {}).get('prompt_tokens'),
                       max_tokens=body['max_tokens'])
        if kind == 'gsm8k':
            row['got'] = helper.final_number(message['content'])
            row['correct'] = helper.same_number(row['got'], item['answer'])
            row['answer'] = item['answer']
        elif kind == 'tools':
            calls = message['tool_calls']
            try:
                arguments = json.loads(calls[0]['arguments']) if calls else None
            except (ValueError, TypeError):
                arguments = None
            row['pass'] = bool(calls and calls[0]['name'] == item['expect']['name']
                               and isinstance(arguments, dict)
                               and all(key in arguments for key in item['expect']['required']))
        else:
            row.update(execute(item, message['content']) if status == 200 else {'passed': False})
        if row['finish_reason'] == 'length':
            row[{'gsm8k': 'correct', 'tools': 'pass', 'humaneval': 'passed'}[kind]] = False
        return row

    for name, items, field in [('gsm8k', gsm, 'correct'), ('tools', tools, 'pass'), ('humaneval', human, 'passed')]:
        records = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for future in concurrent.futures.as_completed([pool.submit(request, item, name) for item in items]):
                records.append(future.result())
                records.sort(key=lambda row: row['id'])
                summary = {'n': len(records), field: sum(bool(row[field]) for row in records),
                           'errors': sum(row['http_status'] != 200 for row in records),
                           'truncated': sum(row['finish_reason'] == 'length' for row in records)}
                canonical.save(args.out / (name + '.json'), {'records': records, 'summary': summary})
        print(json.dumps({'task': name, **summary}), flush=True)


if __name__ == '__main__':
    main()
