#!/usr/bin/env python3
"""Rebuild laptop input hashes offline with the measured tokenizer, never inference.

Use the SGLang 0.5.22 Python environment (Transformers 5.19.0), python -B.
Only matches to retained native usage counts are marked verified.
"""
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
MODELS = [('incumbent', 'qwen35-baseline-e4d2d39cede8'),
          ('candidate-c', 'qwen35-capyctl-w4a16-g32-int8head')]


def main():
    entries = []
    for label, directory in MODELS:
        tokenizer = AutoTokenizer.from_pretrained(Path.home() / 'models' / directory, local_files_only=True)
        cache = {}
        for path in sorted((HERE / 'bench').glob(f'laptop-{label}*-speed.json')):
            if 'nextn' in path.name:
                continue
            data = json.loads(path.read_text())
            for point in data.get('contexts', []):
                target = point['target']
                if target not in cache:
                    def render(n):
                        content = ' data' * n + '\nSummarize the text.'
                        messages = [{'role': 'user', 'content': content}]
                        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, tools=None, return_dict=False)
                        return content, len(ids)
                    lo, hi = 1, target
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        if render(mid)[1] <= target:
                            lo = mid
                        else:
                            hi = mid - 1
                    content, count = render(lo)
                    cache[target] = {'repetitions': lo, 'regenerated_prompt_tokens': count,
                                     'utf8_bytes': len(content.encode()),
                                     'content_sha256': hashlib.sha256(content.encode()).hexdigest()}
                request = point['request']
                native = request.get('prompt_tokens')
                match = not request.get('error') and native == cache[target]['regenerated_prompt_tokens']
                entry = {'evidence': str(path.relative_to(HERE)), 'target': target,
                         'native_prompt_tokens': native, 'seed': None,
                         'generator': "' data' * repetitions + '\\nSummarize the text.'",
                         'tokenizer': directory, 'verified_against_native_usage': match}
                if match:
                    entry.update(cache[target])
                else:
                    entry['status'] = 'unavailable: no matching successful native usage count'
                entries.append(entry)
    out = HERE / 'bench/laptop-input-manifest.json'
    out.write_text(json.dumps({'transformers': '5.19.0', 'add_generation_prompt': True,
                              'tools': None, 'entries': entries}, indent=2) + '\n')
    print(json.dumps({'inputs': len(entries), 'verified': sum(e['verified_against_native_usage'] for e in entries)}))


if __name__ == '__main__':
    main()
