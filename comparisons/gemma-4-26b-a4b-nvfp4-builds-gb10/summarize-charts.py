#!/usr/bin/env python3
"""Plot the measured Gemma recipes using capyctl-bench's summary theme.

python3 summarize-charts.py
uv run --with matplotlib --with pillow python3 summarize-charts.py

Reads bench/final-matrix.json and tensors/results/summary.json; never runs a benchmark.
PNG support is optional, as in capyctl-bench. --x-card writes an additional post card.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools" / "capyctl-bench"))
import capyctl_bench as cb

# Exact measured configurations, not the recommendations list (which may lag the rows).
RECIPES = (
    ("r3", "vLLM 0.32.0", "MTP 3", "CapyCTL r3 · vLLM MTP3"),
    ("r3", "SGLang 0.5.22", "DFlash 8", "CapyCTL r3 · SGLang DFlash8"),
    ("r3", "SGLang 0.5.22", "MTP 3", "CapyCTL r3 · SGLang MTP3"),
    ("NVIDIA", "SGLang 0.5.22", "none", "NVIDIA · SGLang, no drafts"),
)


def select(rows, build, engine, drafter):
    matches = [r for r in rows if (r["build"], r["engine"], r.get("drafter")) ==
               (build, engine, drafter) and r.get("speed_run")]
    if len(matches) != 1:
        raise ValueError(f"Expected one speed row for {(build, engine, drafter)}")
    return matches[0]


def rate(row, streams):
    point = row.get("concurrency", {}).get(str(streams), {})
    return point.get("decode_tps" if streams == 1 else "aggregate_tps")


def fraction(row, key):
    result = row.get(key)
    if not result:
        return "not measured"
    return f"{result['correct' if key == 'gsm8k' else 'pass']}/{result['n']}"


def base_figure(plt, image, theme, kind, title, subtitle):
    palette = cb.mpl_theme(plt, theme)
    width, height = cb.SUMMARY_SIZES[kind]
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=200)
    fig.patch.set_facecolor(palette["bg"])
    logo = image.open(palette["logo"])
    logo_height = 34
    axes = fig.add_axes((44 / width, 1 - 68 / height,
                         logo_height * logo.width / logo.height / width,
                         logo_height / height))
    axes.imshow(logo)
    axes.axis("off")
    fig.text(44 / width, 1 - 104 / height, title, va="top", fontsize=23,
             fontweight="bold", color=palette["text"])
    fig.text(44 / width, 1 - 148 / height, subtitle, va="top", fontsize=12,
             color=palette["muted"])
    return fig, palette


def bars(ax, values, labels, colors, palette, kind, title, unit):
    cb.mpl_panel_style(ax, palette, kind)
    count = len(values)
    present = [v for v in values if v is not None]
    top = max(present, default=1)
    for index, (value, color) in enumerate(zip(values, colors)):
        y = count - index - 1
        if value is None:
            ax.text(0, y, "not measured", va="center", color=palette["muted"])
            continue
        ax.barh(y, value, color=color, height=0.58)
        ax.text(value + top * 0.025, y, f"{value:.2f}", va="center", fontsize=12,
                fontweight="bold", color=palette["text"])
    ax.set_yticks(range(count), list(reversed(labels)), fontsize=11)
    ax.set_ylim(-0.6, count - 0.4)
    ax.set_xlim(0, top * 1.28)
    ax.set_xticks([])
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=25)
    ax.annotate(unit, xy=(0, 1), xycoords="axes fraction", xytext=(0, 4),
                textcoords="offset points", fontsize=10, color=palette["muted"])
    for spine in ax.spines.values():
        spine.set_visible(False)


def summary(plt, image, matrix, path, theme, kind):
    fig, palette = base_figure(
        plt, image, theme, kind, "Gemma 4 26B-A4B: measured recipes",
        "One GB10 · CapyCTL 0.1.12 · temperature 0 · search medians of 3 rounds · 2026-11-17")
    rows = [select(matrix["rows"], *recipe[:3]) for recipe in RECIPES]
    labels = [recipe[3] for recipe in RECIPES]
    colors = [palette["palette"][i] for i in range(len(rows))]
    panels = (
        ([rate(r, 1) for r in rows], "One-stream decode", "tokens/s, higher is better"),
        ([rate(r, 8) for r in rows], "Eight-stream total", "tokens/s, higher is better"),
        ([r.get("ttft24k", {}).get("1") for r in rows], "24k-input first token", "seconds, lower is better"),
    )
    if kind == "wide":
        positions = [(0.285 + index * 0.235, 0.35, 0.215, 0.32) for index in range(3)]
    else:
        positions = [(0.39, 0.67 - index * 0.21, 0.56, 0.16) for index in range(3)]
    for index, ((values, title, unit), position) in enumerate(zip(panels, positions)):
        ax = fig.add_axes(position)
        bars(ax, values, labels if kind == "tall" or index == 0 else [""] * len(rows),
             colors, palette, kind, title, unit)
    width, height = cb.SUMMARY_SIZES[kind]
    y = 0.27 if kind == "wide" else 0.18
    fig.text(44 / width, y, "Native task quality: GSM8K / tool calls", fontsize=13,
             color=palette["text"], fontweight="bold")
    for index, (row, label) in enumerate(zip(rows, labels)):
        fig.text(44 / width, y - 0.032 * (index + 1),
                 f"{label}: {fraction(row, 'gsm8k')} / {fraction(row, 'tools')}",
                 fontsize=10.5, color=colors[index])
    foot = ["NVIDIA reference: SGLang 0.5.22, 131k-context production configuration. Engines/drafters differ.",
            "512-token cap, normal EOS. TTFT: exactly 24,000 input tokens. One math answer is 0.4 percentage points.",
            "Source: bench/final-matrix.json · github.com/edurdias/capyctl-recipes"]
    for index, line in enumerate(foot):
        fig.text(44 / width, (50 - index * 15) / height, line, fontsize=9.5,
                 color=palette["muted"])
    fig.savefig(path, facecolor=palette["bg"])
    plt.close(fig)


def post_card(plt, image, matrix, agreement, path):
    fig, palette = base_figure(
        plt, image, "light", "wide", "Gemma 4: same NVIDIA recipe, faster checkpoint",
        "One GB10 · NVIDIA's coding recipe · same pinned workload and vLLM 0.32.0 · Google MTP4")
    colors = [palette["palette"][0], palette["palette"][1]]
    official = {row["build"]: row for row in matrix["official_workload_runs"]}
    axes = fig.add_axes((0.23, 0.41, 0.65, 0.27))
    bars(axes, [official[key]["decode_tokens_per_second_mean"] for key in ("r3", "NVIDIA")],
         ["CapyCTL r3", "NVIDIA"], colors, palette, "wide",
         "Coding workload, mean decode", "tokens/s, higher is better")
    metrics = (
        ("BF16 next-token agreement, weights only", "r3", "nvidia"),
        ("GSM8K, 250 questions", None, None),
    )
    for index, (label, ours_key, nvidia_key) in enumerate(metrics):
        x = 0.04 + index * 0.49
        fig.text(x, 0.30, label, fontsize=13, color=palette["muted"])
        if ours_key:
            values = [f"{100 * agreement[key]['summary']['agreement']:.3f}%" for key in (ours_key, nvidia_key)]
        else:
            values = ["244/250", "244/250"]
        for offset, (value, name, color) in enumerate(zip(values, ("CapyCTL r3", "NVIDIA"), colors)):
            fig.text(x + offset * 0.22, 0.235, value, fontsize=27, fontweight="bold", color=color)
            fig.text(x + offset * 0.22, 0.197, name, fontsize=11, color=palette["muted"])
    foot = ["10 measured coding requests after 3 warmups · mean decode · 8192-token output budget · 2026-11-17",
            "Same reconstructed workload/flags; not an exact AIPerf reproduction. No coding-correctness score collected.",
            "Agreement: 62,658 BF16 positions, weights only. Native vLLM no-draft quality: both 244/250 GSM8K, 30/30 tools.",
            "github.com/edurdias/capyctl-recipes · sources: final-matrix.json and tensors/results/summary.json"]
    for index, line in enumerate(foot):
        fig.text(0.04, 0.12 - index * 0.024, line, fontsize=9.5, color=palette["muted"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=palette["bg"])
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=HERE / "bench/final-matrix.json")
    parser.add_argument("--out", type=Path, default=HERE / "bench/summary")
    parser.add_argument("--x-card", type=Path)
    args = parser.parse_args()
    plt = cb.load_matplotlib("Gemma summary PNGs")
    if plt is None:
        return
    try:
        from PIL import Image
    except ImportError:
        print("Pillow is not installed: PNGs skipped. Use uv run --with matplotlib --with pillow.", file=sys.stderr)
        return
    matrix = json.loads(args.matrix.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    for theme in ("light", "dark"):
        for kind in ("wide", "tall"):
            suffix = "-dark" if theme == "dark" else ""
            path = args.out / f"summary-{kind}{suffix}.png"
            summary(plt, Image, matrix, path, theme, kind)
            print("wrote", path)
    if args.x_card:
        agreement = json.loads((HERE / "tensors/results/summary.json").read_text())
        post_card(plt, Image, matrix, agreement, args.x_card)
        print("wrote", args.x_card)


if __name__ == "__main__":
    main()
