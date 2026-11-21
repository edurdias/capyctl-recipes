#!/usr/bin/env python3
"""Render retained Qwen3.6 GB10 results and light/dark post cards; no inference."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'tools/capyctl-bench'))
import capyctl_bench as cb

spec = importlib.util.spec_from_file_location('recipe_table', HERE / 'recipe-table.py')
table = importlib.util.module_from_spec(spec)
spec.loader.exec_module(table)
SERIES = [('cd', 'CapyCTL · vLLM DFlash2', '#ed782f'),
          ('n1', 'NVIDIA published · vLLM MTP4', '#89939e'),
          ('pv1', 'Published vLLM-site · MTP3', '#4c91d5'),
          ('ps', 'Published SGLang cookbook', '#59aa8a'),
          ('pgm', 'Published GGUF · MTP', '#aa7ec7'),
          ('pg', 'Published GGUF · no drafter', '#cfaa58')]


def render(plt, matrix, path, dark=False):
    from PIL import Image
    palette = cb.mpl_theme(plt, 'dark' if dark else 'light')
    bg, fg, muted = ('#141b24', '#f2f4f8', '#aab6c5') if dark else ('#ffffff', '#172333', '#59687a')
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor(bg)
    width, height = fig.get_size_inches() * fig.dpi
    logo = Image.open(palette['logo'])
    logo_h = 34
    ax_logo = fig.add_axes((44 / width, 1 - 68 / height,
                            logo_h * logo.width / logo.height / width, logo_h / height))
    ax_logo.imshow(logo)
    ax_logo.axis('off')
    fig.subplots_adjust(left=.075, right=.97, top=.66, bottom=.24, wspace=.24)
    fig.text(.055, .87, "CapyCTL's Qwen3.6-35B-A3B recipe", fontsize=27, weight='bold', color=fg)
    fig.text(.055, .81, 'Faster than the published GB10 recipes tested here', fontsize=21, color=fg)
    fig.text(.055, .755, 'One GB10 · vLLM 0.32 · DeltaNet NVFP4 · DFlash2 block 8', fontsize=14, color=muted)
    data = {r['run']: r for r in table.rows(matrix)}
    for key, name, color in SERIES:
        r = data[key]
        short, long = table.points(r), table.points(r, 'long', 'long')
        axes[0].plot(range(4), [short[c]['aggregate_tps'] for c in (1, 2, 4, 8)], marker='o', color=color, lw=3 if key == 'cd' else 1.8, label=name)
        axes[1].plot(range(4), [long[c]['decode_tps'] for c in (8000, 32000, 64000, 110000)], marker='o', color=color, lw=3 if key == 'cd' else 1.8)
    for ax, title, ticks, xlabel in zip(axes, ('Short prompts · aggregate tokens/s', 'Long prompts · one-stream decode tokens/s'),
                                      (('1', '2', '4', '8'), ('8k', '32k', '64k', '110k')), ('Concurrent streams', 'Exact input tokens')):
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
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .105), ncol=3, frameon=False, fontsize=11, labelcolor=fg)
    fig.text(.055, .061, 'Median of 3 rounds · same prompts on one GB10 · decode excludes time to first token', fontsize=11, color=muted)
    fig.text(.055, .026, 'CapyCTL: GSM8K 234/250 · tools 30/30 · HumanEval 159/164 · retrieval 60/60 · published rows include failed screens', fontsize=11, color=muted)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=bg)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--matrix', type=Path, default=HERE / 'bench/final-matrix.json')
    ap.add_argument('--out', type=Path, default=HERE / 'bench/summary')
    ap.add_argument('--x-card', type=Path, default=Path('/var/tmp/capyctl-webui/qwen36/x-card.png'))
    ap.add_argument('--x-card-dark', type=Path, default=Path('/var/tmp/capyctl-webui/qwen36/x-card-dark.png'))
    args = ap.parse_args()
    plt = cb.load_matplotlib('Qwen3.6 summary PNGs')
    if plt is None:
        raise SystemExit('Matplotlib required; use uv run --with matplotlib --with pillow.')
    matrix = json.loads(args.matrix.read_text())
    for dark, external in ((False, args.x_card), (True, args.x_card_dark)):
        out = args.out / ('summary-wide-dark.png' if dark else 'summary-wide.png')
        render(plt, matrix, out, dark)
        external.parent.mkdir(parents=True, exist_ok=True)
        external.write_bytes(out.read_bytes())
        print('wrote', out, 'and', external)


if __name__ == '__main__':
    main()
