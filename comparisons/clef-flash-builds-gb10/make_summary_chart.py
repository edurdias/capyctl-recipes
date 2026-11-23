#!/usr/bin/env python3
"""Render measured agreement as a branded SVG without optional dependencies."""
import base64
import html
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'tools/capyctl-bench'))
import capyctl_bench as bench


def main():
    logo = base64.b64encode(bench.THEMES['light']['logo'].read_bytes()).decode()
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="820" viewBox="0 0 1400 820">',
             '<rect width="1400" height="820" fill="#fafaf9"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#292524}.muted{fill:#57534e}</style>',
             f'<image x="65" y="35" width="240" height="55" href="data:image/webp;base64,{logo}"/>',
             '<text x="65" y="140" font-size="34" font-weight="bold">Clef-Flash: only the GB10 mixed build is published</text>',
             '<text x="65" y="185" font-size="22" class="muted">Stock SGLang 0.5.22 · GB10 · 5,294 held-out questions</text>']
    specs = [('gb10-fp8', 'Published FP8', '#89939e'),
             ('clef-flash-capyctl-fp8-r1', 'FP8 r1 (not published)', '#89939e'),
             ('gb10-nvfp4', 'Published NVFP4', '#0369a1'),
             ('clef-flash-capyctl-mixed-r1', 'CapyCTL mixed r1', '#0f766e')]
    def x(value):
        return 365 + (value - 96.8) / 2.6 * 800
    for tick in [97, 97.5, 98, 98.5, 99]:
        xx = x(tick)
        parts += [f'<line x1="{xx}" y1="240" x2="{xx}" y2="600" stroke="#d6d3d1"/>',
                  f'<text x="{xx}" y="630" text-anchor="middle" font-size="18">{tick}%</text>']
    for i, (tag, name, color) in enumerate(specs):
        v = json.loads((HERE / f'results/{tag}-vs-bf16-heldout.json').read_text())
        v = v.get('overall', v)
        agreement = v['agreement'] * 100
        lo, hi = [q * 100 for q in v['wilson95']]
        y = 275 + i * 85
        parts += [f'<text x="335" y="{y+8}" text-anchor="end" font-size="23">{html.escape(name)}</text>',
                  f'<rect x="365" y="{y-22}" width="{x(agreement)-365}" height="44" fill="{color}"/>',
                  f'<line x1="{x(lo)}" y1="{y}" x2="{x(hi)}" y2="{y}" stroke="#292524" stroke-width="3"/>',
                  f'<line x1="{x(lo)}" y1="{y-9}" x2="{x(lo)}" y2="{y+9}" stroke="#292524" stroke-width="3"/>',
                  f'<line x1="{x(hi)}" y1="{y-9}" x2="{x(hi)}" y2="{y+9}" stroke="#292524" stroke-width="3"/>',
                  f'<text x="{x(hi)+15}" y="{y+7}" font-size="21">{agreement:.2f}%</text>']
    parts += [f'<line x1="{x(99)}" y1="220" x2="{x(99)}" y2="600" stroke="#9f1239" stroke-width="3" stroke-dasharray="8 6"/>',
              '<text x="65" y="690" font-size="20">Descriptive 95% Wilson ranges; questions within a request may correlate.</text>',
              '<text x="65" y="730" font-size="20">FP8 r1 laptop: 98.357% vs published FP8 98.451%; only 23.1875 MiB memory headroom. Not published.</text>',
              '<text x="65" y="770" font-size="20">I publish mixed for improved agreement over published NVFP4, despite missing the 99% target.</text>',
              '</svg>']
    (HERE / 'summary.svg').write_text('\n'.join(parts) + '\n')


if __name__ == '__main__':
    main()
