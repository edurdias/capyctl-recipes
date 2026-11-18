#!/usr/bin/env python3
"""Run the existing Qwen comparison's pinned GSM8K and 30-tool-call method."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--endpoint', required=True, help='inference base without /v1')
parser.add_argument('--model', required=True)
parser.add_argument('--api-key-file', type=Path)
parser.add_argument('--out', type=Path, required=True, help='result directory')
parser.add_argument('--limit', type=int, default=250)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
reference = root/'comparisons/qwen3.8-27b-sglang-dflash2-vs-mtp-eagle-gb10'
helper = reference/'b18.py'
assets = reference/'accuracy'
data = assets/'gsm8k-test-250.jsonl'
expected = '687da67051177239f32b9e9fe602a753edd241824a627d59008c879b7e0bed80'
if hashlib.sha256(data.read_bytes()).hexdigest() != expected:
    raise SystemExit('canonical GSM8K subset hash mismatch')
spec = importlib.util.spec_from_file_location('qwen_accuracy_reference', helper)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
args.out.mkdir(parents=True, exist_ok=True)
settings = dict(endpoint=args.endpoint, model=args.model, cred=str(args.api_key_file) if args.api_key_file else None,
                concurrency=8, max_tokens=2048)
module.cmd_gsm8k(argparse.Namespace(**settings, data=str(data), limit=args.limit, out=str(args.out/'gsm8k.json')))
module.cmd_tools(argparse.Namespace(**settings, cases=str(assets/'tool-cases.json'), out=str(args.out/'tools.json')))
metadata = {'method': 'unchanged Qwen comparison b18.py GSM8K/tools', 'temperature': 0,
            'concurrency': 8, 'max_tokens': 2048, 'gsm8k_limit': args.limit,
            'gsm8k_dataset_revision': '740312add88f781978c0658806c59bc2815b9866',
            'gsm8k_subset_sha256': expected, 'helper_sha256': hashlib.sha256(helper.read_bytes()).hexdigest(),
            'tool_cases_sha256': hashlib.sha256((assets/'tool-cases.json').read_bytes()).hexdigest()}
(args.out/'method.json').write_text(json.dumps(metadata, indent=2)+'\n')
print(json.dumps({key: json.loads((args.out/(key+'.json')).read_text())['summary'] for key in ('gsm8k', 'tools')}))
