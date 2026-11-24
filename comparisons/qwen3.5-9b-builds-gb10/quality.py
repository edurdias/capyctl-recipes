#!/usr/bin/env python3
"""Reuse the pinned screen with a 32,768-token total context budget."""
import argparse
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import sys
import urllib.request

source = Path(__file__).resolve().parent.parent / 'qwen3.6-35b-a3b-builds-gb10/accuracy.py'
spec = importlib.util.spec_from_file_location('pinned_quality', source)
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)
original_urlopen = urllib.request.urlopen
original_save = quality.save


def save(path, value):
    if path.name == 'method.json':
        value['context_cap_tokens'] = 32768
        value['completion_budget'] = 'min(max_tokens, 32768 - exact /v1/tokenize chat prompt count)'
        value['client_timeout_seconds'] = 3600
        value['resume_policy'] = 'Reuse HTTP 200 records unchanged; retry only missing or failed requests.'
    original_save(path, value)


quality.save = save


def context_capped(request, *args, **kwargs):
    if isinstance(request, urllib.request.Request) and request.full_url.endswith('/v1/chat/completions'):
        body = json.loads(request.data)
        token_body = {key: value for key, value in body.items() if key != 'max_tokens'}
        token_request = urllib.request.Request(
            request.full_url.removesuffix('/chat/completions') + '/tokenize',
            json.dumps(token_body).encode(), dict(request.header_items()))
        with original_urlopen(token_request, timeout=30) as response:
            prompt_tokens = json.load(response)['count']
        body['max_tokens'] = min(body['max_tokens'], 32768 - prompt_tokens)
        if body['max_tokens'] <= 0:
            raise ValueError('Prompt exceeds total context budget')
        request.data = json.dumps(body).encode()
        kwargs['timeout'] = 3600
    return original_urlopen(request, *args, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--resume-from', type=Path)
    adapter_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if adapter_args.resume_from:
        retained = {}
        for name in ('gsm8k', 'tools', 'humaneval'):
            path = adapter_args.resume_from / (name + '.json')
            if path.exists():
                retained[name] = {
                    row['id']: row for row in json.loads(path.read_text())['records']
                    if row['http_status'] == 200
                }
        original_pool = concurrent.futures.ThreadPoolExecutor

        class ResumePool(original_pool):
            def submit(self, fn, *args, **kwargs):
                if len(args) != 2 or not isinstance(args[0], dict) or not isinstance(args[1], str):
                    return super().submit(fn, *args, **kwargs)
                item, kind = args
                key = str(item.get('task_id', item.get('id', item.get('index'))))
                row = retained.get(kind, {}).get(key)
                if row is not None:
                    future = concurrent.futures.Future()
                    future.set_result(row)
                    return future
                return super().submit(fn, *args, **kwargs)

        concurrent.futures.ThreadPoolExecutor = ResumePool
    urllib.request.urlopen = context_capped
    quality.main()
