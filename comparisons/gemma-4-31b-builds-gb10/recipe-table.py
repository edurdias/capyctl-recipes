#!/usr/bin/env python3
"""Print retained timing tables and optionally render the comparison chart."""
import argparse
import base64
import importlib.util
import json
from pathlib import Path
import sys
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
LABELS = {'mse-sg': 'MSE · SGLang', 'mse-dflash2': 'MSE · DFlash2',
          'google-vl': 'Google QAT · vLLM 0.32', 'redhat-vl': 'RedHatAI NVFP4',
          'nvidia-vl': 'NVIDIA NVFP4', 'published-vl030': 'Unchanged Google recipe'}


def rows():
    summary = json.loads((HERE / 'bench/summary.json').read_text())
    return [(LABELS[key], row['short']['1']['decode_tps'],
             row['short']['8']['aggregate_tps'],
             row['long'].get('64000', {}).get('decode_tps'))
            for key, row in summary['rows'].items()]


def chart(data):
    spec = importlib.util.spec_from_file_location('chart_bench', HERE.parents[1] / 'tools/capyctl-bench/capyctl_bench.py')
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    logo = base64.b64encode(helper.THEMES['light']['logo'].read_bytes()).decode()
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="910" viewBox="0 0 1400 910">',
             '<rect width="1400" height="910" fill="#fafaf9"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#292524}</style>',
             f'<image x="55" y="25" width="240" height="55" href="data:image/webp;base64,{logo}"/>',
             '<text x="55" y="130" font-size="34" font-weight="bold">Gemma 4 31B: I qualify no replacement</text>',
             '<text x="55" y="170" font-size="22">GB10 · measured engine/checkpoint setups, not an isolated drafter test</text>',
             '<text x="400" y="225" font-size="22">Short c1 decode tokens/s</text>',
             '<text x="900" y="225" font-size="22">64k decode tokens/s</text>']
    for i, (label, short, _, long) in enumerate(data):
        y = 260 + i * 70
        color = '#0f766e' if i == 1 else '#64748b'
        parts.append(f'<text x="55" y="{y+23}" font-size="20">{escape(label)}</text>')
        for x, value, maximum in [(400, short, 20), (900, long, 12)]:
            if value is None:
                parts.append(f'<text x="{x}" y="{y+23}" font-size="19">Unsupported at unchanged 32k context</text>')
            else:
                width = value / maximum * 290
                parts.extend([f'<rect x="{x}" y="{y}" width="{width:.2f}" height="30" fill="{color}"/>',
                              f'<text x="{x+width+10:.2f}" y="{y+23}" font-size="20">{value:.3f}</text>'])
    parts.extend(['<text x="55" y="720" font-size="24" font-weight="bold">DFlash2: +65.1% short c1; −32.9% at 64k versus MSE SGLang</text>',
                  '<text x="55" y="765" font-size="21">MSE FP8/0.25: retrieval 7/12 at 32k, 3/12 at 64k; strict copy 85/100.</text>',
                  '<text x="55" y="802" font-size="21">RedHatAI: 11/12, 11/12 retrieval; copy 97/100. Screen-level only; full gates still needed.</text>',
                  '<text x="55" y="839" font-size="21">MSE DFlash2 passes full task checks but fails the broader reasoning/output gate.</text>',
                  '<text x="55" y="880" font-size="17">Three measured speed rounds; 512-token cap. Retrieval: 12 document variants per length; copy: 100 cases.</text>', '</svg>'])
    (HERE / 'summary.svg').write_text('\n'.join(parts) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chart', action='store_true')
    args = parser.parse_args()
    data = rows()
    print('| Setup | Short c1 decode | Short c8 aggregate | 64k decode |')
    print('|---|---:|---:|---:|')
    for label, short, c8, long in data:
        value = 'Unsupported' if long is None else f'{long:.3f}'
        print(f'| {label} | {short:.3f} | {c8:.3f} | {value} |')
    if args.chart:
        chart(data)


if __name__ == '__main__':
    main()
