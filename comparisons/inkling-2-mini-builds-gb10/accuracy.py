#!/usr/bin/env python3
"""Pinned GSM8K-250/tools-30 plus HumanEval-164; greedy default thinking."""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

HUMANEVAL_REVISION = '7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544'
GSM_HASH = '687da67051177239f32b9e9fe602a753edd241824a627d59008c879b7e0bed80'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = ROOT/'comparisons/qwen3.8-27b-sglang-dflash2-vs-mtp-eagle-gb10/accuracy'
spec = importlib.util.spec_from_file_location('qwen_quality_helper', ASSETS.parent/'b18.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def execute(item, content):
    if os.geteuid() == 0:
        raise RuntimeError('generated code must never run as root')
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', content, re.S)
    code = blocks[0] if blocks else content
    # The prompt asks for a complete function, not an indented continuation.
    program = code+'\n'+item['test']+'\ncheck('+item['entry_point']+')\n'
    with tempfile.TemporaryDirectory(prefix='q36-he-') as directory:
        script = Path(directory)/'candidate.py'
        script.write_text(program)
        command = ['bash', '-c', 'ulimit -v 1048576; ulimit -t 10; ulimit -f 1024; ulimit -u 64; exec "$1" -I "$2"',
                   'humaneval', '/usr/bin/python3', str(script)]
        try:
            run = subprocess.run(command, cwd=directory, env={'PATH': '/usr/bin:/bin', 'HOME': directory,
                                 'LANG': 'C.UTF-8', 'OPENBLAS_NUM_THREADS': '1'},
                                 capture_output=True, timeout=15)
            return {'passed': run.returncode == 0, 'returncode': run.returncode,
                    'test_stderr': run.stderr.decode(errors='replace')[-2000:]}
        except subprocess.TimeoutExpired:
            return {'passed': False, 'timeout': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--endpoint', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--api-key-file', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-tokens', type=int, default=32768)
    p.add_argument('--screen', action='store_true', help='first 50 canonical GSM8K and HumanEval cases; all 30 tools; not full qualification')
    p.add_argument('--probe', action='store_true', help='20 canonical GSM8K problems at default high effort; report truncations')
    p.add_argument('--reasoning-effort', help='optional Inkling template effort; omitted uses default high (0.9)')
    p.add_argument('--request-timeout', type=float, default=2400, help='transport deadline in seconds; not a generation cap')
    p.add_argument('--humaneval-ids', nargs='+', help='run only these named HumanEval cases after transport failure')
    p.add_argument('--humaneval-only', action='store_true', help='Extend only HumanEval without requesting math or tools')
    p.add_argument('--resume-from', type=Path, help='Reuse unchanged screen observations when extending to full qualification')
    a = p.parse_args()
    if os.geteuid() == 0:
        p.error('run as an unprivileged account')
    from datasets import load_dataset
    human = list(load_dataset('openai/openai_humaneval', revision=HUMANEVAL_REVISION, split='test'))
    gsm_data = (ASSETS/'gsm8k-test-250.jsonl').read_bytes()
    if hashlib.sha256(gsm_data).hexdigest() != GSM_HASH or len(human) != 164:
        raise ValueError('canonical dataset mismatch')
    gsm = [json.loads(line) for line in gsm_data.splitlines() if line.strip()]
    tools = json.loads((ASSETS/'tool-cases.json').read_text())['cases']
    if a.screen:
        gsm, human = gsm[:50], human[:50]
    if a.probe:
        gsm, tools, human = gsm[:20], [], []
    if a.humaneval_ids:
        selected = set(a.humaneval_ids)
        human = [item for item in human if item['task_id'] in selected]
        if {item['task_id'] for item in human} != selected:
            raise ValueError('requested HumanEval ID is outside the selected canonical cases')
        gsm, tools = [], []
    if a.humaneval_only:
        gsm, tools = [], []
    key = None
    # CapyCTL credential files may be simple KEY=value text rather than JSON.
    if a.api_key_file:
        key = helper.credential(str(a.api_key_file), 'api_key')
    method = {'temperature': 0, 'thinking': 'model default high (0.9)' if a.reasoning_effort is None else a.reasoning_effort,
              'max_tokens': a.max_tokens, 'concurrency': 8,
              'scope': 'probe-20' if a.probe else ('screen-50' if a.screen else 'full'), 'truncation_policy': 'count every length-limited response as wrong',
              'gsm8k_revision': '740312add88f781978c0658806c59bc2815b9866', 'gsm8k_subset_sha256': GSM_HASH,
              'tools_sha256': hashlib.sha256((ASSETS/'tool-cases.json').read_bytes()).hexdigest(),
              'humaneval_dataset': 'openai/openai_humaneval', 'humaneval_revision': HUMANEVAL_REVISION,
              'humaneval_protocol': 'pass@1; one greedy complete-function response per task; first Python fence or whole content',
              'execution': {'uid_is_root': False, 'temporary_directory_per_test': True, 'wall_timeout_s': 15,
                            'ulimit_cpu_s': 10, 'ulimit_address_space_bytes': 1073741824,
                            'ulimit_file_blocks': 1024, 'ulimit_processes': 64, 'environment': 'minimal; no inherited credentials'}}
    if a.resume_from:
        previous_method = json.loads((a.resume_from/'method.json').read_text())
        if {k:v for k,v in previous_method.items() if k != 'scope'} != {k:v for k,v in method.items() if k != 'scope'}:
            raise ValueError('resumed quality method differs')
        previous_transport = json.loads((a.resume_from/'transport.json').read_text())
        if previous_transport['request_timeout_s'] != a.request_timeout:
            raise ValueError('resumed quality transport timeout differs')
    save(a.out/'method.json', method)
    save(a.out/'transport.json', {'request_timeout_s': a.request_timeout,
                                 'selected_humaneval_ids': a.humaneval_ids,
                                 'resumed_from': str(a.resume_from) if a.resume_from else None,
                                 'new_observation_launch_run': a.out.parent.name})
    def request(item, kind):
        if kind == 'gsm8k':
            prompt = item['question']+'\nSolve it step by step, then give the final answer as a number in \\boxed{}.'
        elif kind == 'tools':
            prompt = item['prompt']
        else:
            prompt = 'Implement the following Python function. Return the complete function with any required imports in one Python code block.\n\n'+item['prompt']
        body = {'model': a.model, 'temperature': 0, 'max_tokens': a.max_tokens,
                'messages': [{'role': 'user', 'content': prompt}]}
        if a.reasoning_effort is not None:
            body['chat_template_kwargs'] = {'reasoning_effort': a.reasoning_effort}
        if kind == 'tools':
            body.update(tools=item['tools'], tool_choice='auto')
        headers = {'Content-Type': 'application/json'}
        if key:
            headers['Authorization'] = 'Bearer '+key
        started = time.monotonic()
        try:
            req = urllib.request.Request(a.endpoint.rstrip('/')+'/v1/chat/completions', json.dumps(body).encode(), headers)
            with urllib.request.urlopen(req, timeout=a.request_timeout) as response:
                status, data = response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            status, data = exc.code, {}
        except (OSError, ValueError):
            status, data = 0, {}
        msg = helper.message_parts(data)
        row = {'id': str(item.get('task_id', item.get('id', item.get('index')))), 'http_status': status,
               'seconds': time.monotonic()-started, 'launch_run': a.out.parent.name, **msg}
        if kind == 'gsm8k':
            row['got'] = helper.final_number(msg['content'])
            row['correct'] = helper.same_number(row['got'], item['answer'])
            row['answer'] = item['answer']
        elif kind == 'tools':
            calls = msg['tool_calls']
            try:
                arguments = json.loads(calls[0]['arguments']) if calls else None
            except (ValueError, TypeError):
                arguments = None
            row['pass'] = bool(calls and calls[0]['name'] == item['expect']['name'] and isinstance(arguments, dict)
                               and all(k in arguments for k in item['expect']['required']))
        else:
            row.update(execute(item, msg['content']) if status == 200 else {'passed': False})
        if row['finish_reason'] == 'length':
            row[{'gsm8k': 'correct', 'tools': 'pass', 'humaneval': 'passed'}[kind]] = False
        return row
    for name, items, field in [('gsm8k', gsm, 'correct'), ('tools', tools, 'pass'), ('humaneval', human, 'passed')]:
        if not items:
            continue
        selected_ids = {str(item.get('task_id', item.get('id', item.get('index')))) for item in items}
        prior = a.resume_from/(name+'.json') if a.resume_from else None
        records = [{**r, 'launch_run': r.get('launch_run', a.resume_from.parent.name)}
                   for r in json.loads(prior.read_text())['records'] if r['id'] in selected_ids] if prior and prior.exists() else []
        reused_ids = {r['id'] for r in records}
        def persist():
            records.sort(key=lambda row: row['id'])
            summary = {'n': len(records), field: sum(bool(r[field]) for r in records),
                       'errors': sum(r['http_status'] != 200 for r in records),
                       'truncated': sum(r['finish_reason'] == 'length' for r in records)}
            save(a.out/(name+'.json'), {'records': records, 'summary': summary})
            return summary
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(request, item, name) for item in items
                       if str(item.get('task_id', item.get('id', item.get('index')))) not in reused_ids]
            for future in concurrent.futures.as_completed(futures):
                records.append(future.result())
                persist()
        summary = persist()
        save(a.out/(name+'-continuation.json'), {'reused_ids': sorted(reused_ids),
                                               'new_requests': len(futures)})
        print(json.dumps({'task': name, **summary}), flush=True)

if __name__ == '__main__':
    main()
