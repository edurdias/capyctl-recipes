#!/usr/bin/env python3
"""Freeze real chat, function-call and code conversations, disjoint from evaluation."""
import hashlib
import json
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import HfApi

SOURCES = [
    ('HuggingFaceH4/ultrachat_200k', 'train_sft', 'chat', 128),
    ('glaiveai/glaive-function-calling-v2', 'train', 'tool', 128),
    ('iamtarun/python_code_instructions_18k_alpaca', 'train', 'code', 128),
]

def main():
    rows, manifest = [], []
    for repo, split, category, count in SOURCES:
        revision = HfApi().dataset_info(repo).sha
        dataset = load_dataset(repo, split=split, revision=revision, streaming=True)
        selected = []
        for i, row in enumerate(dataset):
            if category == 'chat':
                messages = row['messages']
            elif category == 'tool':
                messages = [{'role': 'system', 'content': row['system']}, {'role': 'user', 'content': row['chat']}]
            else:
                prompt = row['instruction'] + ('\n' + row['input'] if row.get('input') else '')
                messages = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': row['output']}]
            rows.append({'category': category, 'source': repo, 'revision': revision, 'row': i, 'messages': messages})
            selected.append(i)
            if len(selected) == count:
                break
        assert len(selected) == count
        manifest.append({'repo': repo, 'revision': revision, 'split': split, 'rows': selected})
    output = Path(__file__).with_name('calibration.jsonl')
    output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    Path(__file__).with_name('calibration-manifest.json').write_text(json.dumps({'sources': manifest, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'samples': len(rows)}, indent=2) + '\n')

if __name__ == '__main__':
    main()
