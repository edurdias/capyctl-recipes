#!/usr/bin/env python3
"""Render retained Inkling GB10 evidence as light/dark cards; no inference."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'tools/capyctl-bench'))
import capyctl_bench as cb

spec = importlib.util.spec_from_file_location('recipe_table', HERE / 'recipe-table.py')
table = importlib.util.module_from_spec(spec)
spec.loader.exec_module(table)
SERIES = [
    ('sg-fp8-dspark-host-a', 'CapyCTL FP8 attention · SGLang DSpark', '#ed782f'),
    ('tv-std', 'Community recipe · vLLM BF16 KV', '#89939e'),
    ('tv-fp8', 'Community recipe · vLLM FP8 KV', '#4c91d5'),
    ('sg-dspark-host-a', 'Original weights · SGLang DSpark', '#59aa8a'),
    ('sg-fp8-plain-host-a', 'CapyCTL FP8 attention · no drafter', '#aa7ec7'),
    ('sg-plain-warm', 'Original weights · no drafter', '#c4a04e'),
]


def render(plt, matrix, path, dark=False):
    from PIL import Image
    palette = cb.mpl_theme(plt, 'dark' if dark else 'light')
    bg, fg, muted = ('#141b24', '#f2f4f8', '#aab6c5') if dark else ('#ffffff', '#172333', '#59687a')
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor(bg)
    width, height = fig.get_size_inches() * fig.dpi
    logo = Image.open(palette['logo'])
    logo_h = 34
    logo_ax = fig.add_axes((44 / width, 1 - 68 / height,
                            logo_h * logo.width / logo.height / width, logo_h / height))
    logo_ax.imshow(logo)
    logo_ax.axis('off')
    fig.subplots_adjust(left=.075, right=.97, top=.66, bottom=.25, wspace=.24)
    fig.text(.055, .87, "CapyCTL's Inkling 2 Mini recipe", fontsize=27, weight='bold', color=fg)
    fig.text(.055, .81, 'A balanced-workload gain, not a win at every point', fontsize=21, color=fg)
    fig.text(.055, .755, 'GB10 · SGLang PR #51926 + loader patch · FP8 attention · DSpark gamma 7 / block 8', fontsize=13, color=muted)
    data = {row['run']: row for row in table.rows(matrix)}
    for index, (key, name, color) in enumerate(SERIES):
        row = data[key]
        short, long = table.points(row), table.points(row, 'long', 'long')
        style = {'marker': 'o', 'color': color, 'lw': 3.2 if index == 0 else 1.8,
                 'zorder': 10 if index == 0 else 2}
        axes[0].plot(range(4), [short[c]['aggregate_tps'] for c in (1, 2, 4, 8)], label=name, **style)
        axes[1].plot(range(4), [long[c]['decode_tps'] for c in (8000, 32000, 64000, 110000)], **style)
    for ax, title, ticks, xlabel in zip(
            axes, ('Short prompts · aggregate tokens/s', 'Long prompts · one-stream decode tokens/s'),
            (('1', '2', '4', '8'), ('8k', '32k', '64k', '110k')),
            ('Concurrent streams', 'Exact input tokens')):
        ax.set_facecolor(bg)
        ax.set_title(title, color=fg, fontsize=15, pad=18, loc='left')
        ax.set_xticks(range(4), ticks)
        ax.set_xlabel(xlabel, color=muted, labelpad=12)
        ax.tick_params(colors=muted)
        ax.grid(axis='y', alpha=.16, color=muted)
        ax.set_ylim(bottom=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .105), ncol=3,
               frameon=False, fontsize=10, labelcolor=fg)
    fig.text(.055, .061, 'Median of 3 rounds · default high effort · elected setup loses 1-stream throughput and 32k decode', fontsize=11, color=muted)
    fig.text(.055, .026, 'Full quality: 240/250 math · 28/30 tools · 149/164 code · retrieval 56/60 (strict gate fails for both recipes)', fontsize=11, color=muted)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=bg)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, default=HERE / 'bench/final-matrix.json')
    parser.add_argument('--out', type=Path, default=HERE / 'bench/summary')
    args = parser.parse_args()
    plt = cb.load_matplotlib('Inkling summary PNGs')
    if plt is None:
        raise SystemExit('Matplotlib and Pillow are required.')
    matrix = json.loads(args.matrix.read_text())
    for dark in (False, True):
        path = args.out / ('summary-wide-dark.png' if dark else 'summary-wide.png')
        render(plt, matrix, path, dark)
        print('wrote', path)


if __name__ == '__main__':
    main()
