#!/usr/bin/env python3
"""CPU-only syntax/corpus/Harmony preparation check; no model is loaded."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer

root = Path(__file__).resolve().parent
source = Path.home() / 'models/gpt-oss-20b'
ast.parse((root / 'next_token_agree.py').read_text())
corpus = root / 'agreement-prompts.json'
rows = json.loads(corpus.read_text())['prompts']
tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
texts = [tokenizer.apply_chat_template(r['messages'], tools=r.get('tools'), tokenize=False, add_generation_prompt=True, reasoning_effort='low') for r in rows]
ids = [tokenizer(t, add_special_tokens=False)['input_ids'] for t in texts]
report = {'records': len(rows), 'categories': dict(Counter(r['category'] for r in rows)),
          'prompt_sha256': hashlib.sha256(corpus.read_bytes()).hexdigest(),
          'min_tokens': min(map(len, ids)), 'max_tokens': max(map(len, ids)),
          'harmony_low_effort': all('Reasoning: low' in t for t in texts), 'harness_ast_parse': True}
assert report['harmony_low_effort'] and report['records'] == 200
(root / 'agreement-preparation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
