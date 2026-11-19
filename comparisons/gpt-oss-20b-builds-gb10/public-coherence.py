#!/usr/bin/env python3
"""Four deterministic screening prompts; retain complete outputs before benchmarks."""
import argparse
import importlib.util
import json
from pathlib import Path
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--credentials', required=True)
p.add_argument('--model', required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
helper = Path(__file__).resolve().parents[2]/'tools/capyctl-bench/capyctl_bench.py'
spec = importlib.util.spec_from_file_location('bench', helper)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
key = bench.read_api_key(a.credentials)
rows = []
for prompt, expected in [('What is 7 + 8? Answer only with the integer.', '15'), ('What is the capital of France? Answer with just its name.', 'paris'), ('Translate the English word cat into Spanish. Answer with just the word.', 'gato'), ('Write the Python expression for the length of a list named items. Answer with just the expression.', 'len(items)')]:
    body = {'model': a.model, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0, 'reasoning_effort': 'low', 'max_tokens': 512}
    req = urllib.request.Request('http://127.0.0.1:18543/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json', 'Authorization': 'Bearer '+key})
    with urllib.request.urlopen(req, timeout=300) as response:
        answer = json.load(response)
    text = answer['choices'][0]['message'].get('content') or ''
    rows.append({'prompt': prompt, 'expected_substring': expected, 'response': answer, 'passed': expected in text.lower()})
report = {'method': 'four simple deterministic prompts; expected substring screen, complete response retained', 'records': rows, 'passed': all(r['passed'] for r in rows)}
a.out.write_text(json.dumps(report, indent=2)+'\n')
raise SystemExit(not report['passed'])
