#!/usr/bin/env python3
"""Print measured laptop tables; optionally render the publication summary SVG."""
import argparse
import base64
import importlib.util
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent


def rows():
    summary = json.loads((HERE / 'bench/laptop-summary.json').read_text())
    result = []
    for key, label in [('incumbent', 'cyankiwi AWQ @ e4d2d39c'), ('candidate-c', 'CapyCTL C (unpublished)')]:
        row = summary[key]
        speed = [statistics.median(r['aggregate_tps'] for r in level['rounds'])
                 for level in row['speed']['levels']]
        q = row['quality']
        result.append(dict(label=label, speed=speed, memory=row['memory'],
                           math=q['gsm8k']['correct'], tools=q['tools']['pass'],
                           code=q['humaneval']['passed'], trunc=q['humaneval']['truncated']))
    return result


def chart(data):
    path = HERE.parents[1] / 'tools/capyctl-bench/capyctl_bench.py'
    spec = importlib.util.spec_from_file_location('chart_bench', path)
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    logo = base64.b64encode(helper.THEMES['light']['logo'].read_bytes()).decode()
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="860" viewBox="0 0 1400 860">',
             '<rect width="1400" height="860" fill="#fafaf9"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#292524}.muted{fill:#57534e}</style>',
             f'<image x="60" y="25" width="240" height="55" href="data:image/webp;base64,{logo}"/>',
             '<text x="60" y="130" font-size="34" font-weight="bold">Qwen3.5-9B: faster, but the coding gate fails</text>',
             '<text x="60" y="175" font-size="21">SGLang 0.5.22 · 16 GB RTX 4090 Laptop GPU · target-only · no new checkpoint</text>',
             '<text x="60" y="235" font-size="24" font-weight="bold">Median short-prompt aggregate tokens/s</text>']
    colors = ['#89939e', '#0f766e']
    for j, level in enumerate([1, 2, 4, 8]):
        y = 280 + j * 77
        parts.append(f'<text x="60" y="{y+22}" font-size="20">{level} stream(s)</text>')
        for i, row in enumerate(data):
            value = row['speed'][j]
            yy = y + i * 27
            width = value / 700 * 730
            parts += [f'<rect x="220" y="{yy}" width="{width:.2f}" height="23" fill="{colors[i]}"/>',
                      f'<text x="{230+width:.2f}" y="{yy+18}" font-size="17">{value:.1f}</text>']
    for i, row in enumerate(data):
        y = 625 + i * 65
        m = row['memory']
        parts += [f'<rect x="60" y="{y-18}" width="18" height="18" fill="{colors[i]}"/>',
                  f'<text x="90" y="{y}" font-size="21">{row["label"]}: GSM8K {row["math"]}/250 · HumanEval {row["code"]}/164 · code truncations {row["trunc"]}/164</text>',
                  f'<text x="90" y="{y+28}" font-size="19">Sampled peak {m["peak_mib"]:,} MiB · headroom {m["margin_mib"]:,} MiB</text>']
    parts += ['<text x="60" y="780" font-size="21" fill="#9f1239">C: 6.10% code truncation exceeds AWQ’s 5.24% Wilson upper gate. I keep AWQ.</text>',
              '<text x="60" y="825" font-size="17" class="muted">Measured 2026-11-22–23. Three speed rounds; repeated prompts may hit prefix cache. Memory sampled every 0.5 s.</text>',
              '</svg>']
    (HERE / 'summary.svg').write_text('\n'.join(parts) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chart', action='store_true')
    args = parser.parse_args()
    data = rows()
    print('| Build | c1 / c2 / c4 / c8 aggregate tok/s | GSM8K | Tools | HumanEval | Code truncations | Peak / headroom MiB |')
    print('|---|---|---|---|---|---|---|')
    for row in data:
        m = row['memory']
        print(f'| {row["label"]} | ' + ' / '.join(f'{v:.2f}' for v in row['speed']) +
              f' | {row["math"]}/250 | {row["tools"]}/30 | {row["code"]}/164 | {row["trunc"]}/164 | {m["peak_mib"]} / {m["margin_mib"]} |')
    if args.chart:
        chart(data)


if __name__ == '__main__':
    main()
