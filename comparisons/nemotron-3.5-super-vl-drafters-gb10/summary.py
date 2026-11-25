#!/usr/bin/env python3
"""Render the retained measurements, not a new engine qualification."""
import base64
from html import escape
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    rows = {row: json.loads((HERE / 'results' / (row + '-safe-full') /
                            'performance.json').read_text()) for row in ('a', 'b')}
    gate = json.loads((HERE / 'results/b-safe-full/quality-gate.json').read_text())
    logo = base64.b64encode((ROOT / 'tools/capyctl-bench/assets/'
                            'capyctl-header-light.webp').read_bytes()).decode()
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="1000" '
           'viewBox="0 0 1400 1000" role="img" aria-labelledby="title desc">',
           '<title id="title">Nemotron 3.5 Super VL: native MTP, with a 64k retrieval limit</title>',
           '<desc id="desc">SGLang native MTP is faster and passes the task gate. '
           'Both rows have a seeded retrieval miss. MTP 96k and 111k retrieval was '
           'not completed after a hard freeze; its cause is unconfirmed.</desc>',
           '<rect width="1400" height="1000" fill="#fafaf9"/>',
           '<style>text{font-family:Arial,sans-serif;fill:#292524}'
           '.muted{fill:#57534e}.warn{fill:#9a3412}</style>',
           f'<image x="65" y="30" width="230" height="55" '
           f'href="data:image/webp;base64,{logo}"/>']

    def text(x, y, value, size=21, cls='', bold=False):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" class="{cls}" '
                   f'font-weight="{"bold" if bold else "normal"}">{escape(value)}</text>')

    text(65, 135, 'Nemotron 3.5 Super VL: native MTP is the elected recipe', 32, bold=True)
    text(65, 178, 'NVIDIA NVFP4 9838c915 · SGLang 0.5.22 · one GB10 · measured bounded pools', 21, 'muted')
    colors = {'a': '#89939e', 'b': '#ea580c'}
    for x, title, streams, metric, maximum in (
            (65, 'One stream · post-first-token decode', 1, 'decode_tps', 40),
            (755, 'Two streams · aggregate throughput', 2, 'aggregate_tps', 60)):
        text(x, 240, title, 23, bold=True)
        for i, row in enumerate(('a', 'b')):
            level = next(item for item in rows[row]['levels']
                         if item['label'] == 'short' and item['streams'] == streams)
            value = level[metric]
            y = 285 + i * 90
            text(x, y, 'a · no drafter' if row == 'a' else 'b · native MTP', 20)
            width = value / maximum * 450
            svg.append(f'<rect x="{x}" y="{y + 13}" width="{width:.2f}" '
                       f'height="30" rx="3" fill="{colors[row]}"/>')
            text(x + width + 12, y + 37, f'{value:.2f} tok/s', 20)
    text(65, 468, 'Median of three rounds after warmup · 512 output-token cap · default thinking', 19, 'muted')
    text(65, 525, 'Full task gate: PASS for MTP versus the same checkpoint and engine without drafts', 24, bold=True)
    text(65, 568, 'Task', 20, 'muted')
    text(470, 568, 'a · no drafter', 20, 'muted')
    text(790, 568, 'b · native MTP', 20, 'muted')
    for y, suite, label in ((610, 'gsm8k', 'GSM8K'), (651, 'tools', 'Tool calls'),
                            (692, 'humaneval', 'HumanEval')):
        evidence = gate['suites'][suite]
        text(65, y, label)
        text(470, y, f"{evidence['baseline_successes']}/{evidence['n']}")
        text(790, y, f"{evidence['candidate_successes']}/{evidence['n']}")
    text(65, 737, 'Retrieval 32k / 64k', 21)
    text(470, 737, '6/6 · 5/6')
    text(790, 737, '5/6 · 6/6')
    text(65, 778, 'Copy-100', 21)
    text(470, 778, '100/100')
    text(790, 778, '100/100')
    text(65, 827, 'Two paired GSM8K losses, one HumanEval gain; shared helper-function failures retained.', 19, 'muted')
    text(65, 866, 'Each row has one real seeded retrieval miss; the task gate does not erase it.', 19, 'muted')
    text(65, 917, 'MTP 96k/111k retrieval: NOT COMPLETED after a hard freeze; no recovered log.', 22, 'warn', True)
    text(65, 956, 'Separate vLLM automatic-KV OOM confirmed. Direct-engine evidence, not full-context or VL qualification.', 19, 'muted')
    svg.append('</svg>')
    (HERE / 'summary.svg').write_text('\n'.join(svg) + '\n')


if __name__ == '__main__':
    main()
