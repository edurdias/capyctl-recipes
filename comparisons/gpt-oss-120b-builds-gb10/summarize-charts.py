#!/usr/bin/env python3
"""Render retained GPT-OSS-120B measurements, never inference."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'tools/capyctl-bench'))
import capyctl_bench as cb

SERIES = [('capyctl-vl032', 'CapyCTL · vLLM 0.32', 0),
          ('openai-vl032', 'OpenAI · vLLM 0.32', 2),
          ('published-vl030', 'Published OpenAI · vLLM 0.30', 1)]


def render(plt, matrix, path, theme):
    from PIL import Image
    palette = cb.mpl_theme(plt, theme)
    fig = plt.figure(figsize=(12, 7), dpi=160, facecolor=palette['bg'])
    logo = Image.open(palette['logo'])
    ax = fig.add_axes((.04, .88, .12, .065))
    ax.imshow(logo)
    ax.axis('off')
    fig.text(.04, .82, 'GPT-OSS-120B: MXFP4 experts + FP8 attention', fontsize=21,
             weight='bold', color=palette['text'])
    fig.text(.04, .775, 'One GB10 · no drafter · median of three rounds · 512-token cap · normal EOS',
             fontsize=10, color=palette['muted'])
    left = fig.add_axes((.08, .27, .38, .41))
    right = fig.add_axes((.57, .27, .38, .41))
    for ax in (left, right):
        cb.mpl_panel_style(ax, palette, 'wide')
        ax.set_ylabel('tokens/s')
        ax.grid(axis='y', alpha=.3)
    for key, label, index in SERIES:
        row = matrix['rows'][key]
        color = palette['palette'][index]
        left.plot([1, 2, 4, 8], [v['aggregate_tps'] for v in row['short']], marker='o', label=label, color=color)
        right.plot([v['input_tokens']/1000 for v in row['long_decode']],
                   [v['decode_tps'] for v in row['long_decode']], marker='o', color=color)
    left.set_title('Total throughput (includes prefill)', fontsize=12)
    left.set_xlabel('Concurrent streams')
    left.set_xticks([1, 2, 4, 8])
    right.set_title('One-stream decode (excludes TTFT)', fontsize=12)
    right.set_xlabel('Exact input tokens, thousands')
    right.set_xticks([8, 32, 64, 110])
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .15), ncol=3,
               frameon=False, labelcolor=palette['text'], fontsize=9)
    fig.text(.04, .12, 'Matched 0.32: GSM8K 239/250 · tools 30/30 · HumanEval 157/164, both checkpoints.', fontsize=10, color=palette['text'])
    fig.text(.04, .085, 'Retrieval: CapyCTL 47/48 vs OpenAI 48/48; one 111k one-line miss. Both strict copy 100/100.', fontsize=10, color=palette['text'])
    fig.text(.04, .045, 'Retained evidence: bench/final-matrix.json · vLLM 0.30 uses prefix caching and auto KV; quality is screen-only.', fontsize=8, color=palette['muted'])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=palette['bg'])
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, default=HERE/'bench/final-matrix.json')
    parser.add_argument('--out', type=Path, default=HERE/'bench/summary')
    args = parser.parse_args()
    plt = cb.load_matplotlib('GPT-OSS-120B summary')
    if plt is None:
        raise SystemExit('Matplotlib and Pillow are required.')
    matrix = json.loads(args.matrix.read_text())
    for theme in ('light', 'dark'):
        render(plt, matrix, args.out/('summary-wide'+('-dark' if theme == 'dark' else '')+'.png'), theme)


if __name__ == '__main__':
    main()
