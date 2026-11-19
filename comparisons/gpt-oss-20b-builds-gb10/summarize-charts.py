#!/usr/bin/env python3
"""Render measured GPT-OSS summaries and light/dark post cards; never run inference.

python3 summarize-charts.py --x-card /var/tmp/capyctl-webui/gptoss/x-card.png \
  --x-card-dark /var/tmp/capyctl-webui/gptoss/x-card-dark.png
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools/capyctl-bench"))
import capyctl_bench as cb

spec = importlib.util.spec_from_file_location("recipe_table", HERE / "recipe-table.py")
recipe_table = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe_table)
WIDTH, HEIGHT = cb.SUMMARY_SIZES["wide"]
# Four measured no-drafter setups; hero orange, production grey.
SERIES = (
    ("capyctl", "CapyCTL · vLLM 0.32", 0),
    ("vl032-auto", "OpenAI · vLLM 0.32", 2),
    ("sg522-default", "Today's production · OpenAI, SGLang 0.5.22", 1),
    ("llama-b12242", "ggml-org GGUF · llama.cpp", 5),
)


def label(fig, x, y, text, **kw):
    return fig.text(x / WIDTH, 1 - y / HEIGHT, text, **kw)


def color_for(palette, index):
    return palette["palette"][index]


def card(plt, Image, matrix, path, theme):
    palette = cb.mpl_theme(plt, theme)
    text, muted = palette["text"], palette["muted"]
    fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=200)
    fig.patch.set_facecolor(palette["bg"])
    logo = Image.open(palette["logo"])
    logo_h = 34
    ax_logo = fig.add_axes((44 / WIDTH, 1 - 68 / HEIGHT,
                            logo_h * logo.width / logo.height / WIDTH, logo_h / HEIGHT))
    ax_logo.imshow(logo)
    ax_logo.axis("off")
    label(fig, 44, 94, "CapyCTL's gpt-oss-20b recipe:\nfaster than production at every load and context",
          fontsize=23, fontweight="bold", color=text, va="top", linespacing=1.12)
    label(fig, 44, 178, "One GB10 · no drafters · tested at 1/2/4/8 streams and 8k–110k input · FP8 KV · LMCache off",
          fontsize=11.3, color=muted, va="top")
    measured = {r["key"]: r for r in recipe_table.rows(matrix)}

    # Aggregate at every concurrency, matching recipe-table.py rather than mixing c1 decode.
    label(fig, 44, 229, "Total throughput, tokens/s", fontsize=12.5, fontweight="bold", color=text)
    label(fig, 44, 249, "Same prompts · includes prefill · median of 3 rounds", fontsize=10, color=muted)
    top, bottom, left, right = 271, 602, 290, 690
    ax = fig.add_axes((left / WIDTH, 1 - bottom / HEIGHT,
                       (right - left) / WIDTH, (bottom - top) / HEIGHT))
    cb.mpl_panel_style(ax, palette, "wide")
    count, gap = len(SERIES), 2.0
    for group, n in enumerate(recipe_table.LEVELS):
        for i, (key, _, color_index) in enumerate(SERIES):
            value = measured[key]["concurrency"][str(n)]["aggregate_tps"]
            y = group * (count + gap) + i
            ax.barh(y, value, height=0.8, color=color_for(palette, color_index))
            ax.text(value + 2.5, y, f"{value:.1f}", va="center", fontsize=10.5,
                    fontweight="bold" if key == "capyctl" else "normal", color=text)
        hero = measured["capyctl"]["concurrency"][str(n)]["aggregate_tps"]
        production = measured["sg522-default"]["concurrency"][str(n)]["aggregate_tps"]
        same_engine = measured["vl032-auto"]["concurrency"][str(n)]["aggregate_tps"]
        gains = ((f"{n} stream{'s' if n > 1 else ''}", 11.5, "bold"),
                 (f"{100 * (hero / production - 1):+.0f}% vs production", 10, "normal"),
                 (f"{100 * (hero / same_engine - 1):+.0f}% vs OpenAI · vLLM", 10, "normal"))
        centre = group * (count + gap) + (count - 1) / 2
        for offset, (line, size, weight) in enumerate(gains):
            ax.annotate(line, xy=(0, centre + (offset - 1) * 0.95),
                        xycoords=("axes fraction", "data"), xytext=(-12, 0),
                        textcoords="offset points", ha="right", va="center",
                        fontsize=size, fontweight=weight, color=text)
    ax.set_xlim(0, 225)
    ax.set_ylim(4 * (count + gap) - gap - 0.4, -0.7)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Same-fixture post-first-token decode, with setup labels carried by the line legend.
    label(fig, 735, 229, "Long-input decode, one stream", fontsize=12.5, fontweight="bold", color=text)
    label(fig, 735, 249, "Exact input tokens · excludes TTFT · 3 rounds", fontsize=10, color=muted)
    long_data = json.loads((HERE / "bench/fp8-attention/long-context-comparison.json").read_text())["decode"]
    la = fig.add_axes((770 / WIDTH, 1 - 382 / HEIGHT, 372 / WIDTH, 111 / HEIGHT))
    cb.mpl_panel_style(la, palette, "wide")
    for field, name, index in (("candidate_tps", "CapyCTL", 0),
                                ("vllm_baseline_tps", "OpenAI · vLLM", 2),
                                ("sglang_production_tps", "Today's production", 1)):
        values = [p[field] for p in long_data]
        color = color_for(palette, index)
        la.plot(range(4), values, color=color, marker="o", linewidth=2, markersize=3, label=name)
        la.annotate(f"{values[-1]:.1f}", (3, values[-1]), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=9, color=color, fontweight="bold" if index == 0 else "normal")
    la.set_xticks(range(4), ["8k", "32k", "64k", "110k"])
    la.set_ylim(28, 61)
    la.set_xlim(-0.15, 3.6)
    la.set_yticks([30, 40, 50, 60])
    la.tick_params(labelsize=9.5)
    la.grid(axis="y", alpha=0.45)
    la.set_ylabel("tokens/s", fontsize=9, labelpad=3)
    la.legend(loc="lower left", bbox_to_anchor=(-0.08, 1.02), frameon=False,
              fontsize=8, ncol=3, handlelength=1.5, columnspacing=0.9, borderaxespad=0)

    # Gemma-style compact table: full setup label, then values. The color key serves both panels.
    label(fig, 735, 426, "Time to first token · lower is better", fontsize=12, fontweight="bold", color=text)
    short_rights, long_rights = (804, 852, 900, 948), (1052, 1154)
    label(fig, 735, 447, "Short prompt, ms", fontsize=9.5, color=muted)
    label(fig, 1008, 447, "24k prompt, s", fontsize=9.5, color=muted)
    for n, x in zip((*recipe_table.LEVELS, 1, 4), (*short_rights, *long_rights)):
        label(fig, x, 463, str(n), fontsize=9, ha="right", color=muted)
    for i, (key, name, index) in enumerate(SERIES):
        y = 484 + i * 34
        label(fig, 735, y, "■", fontsize=10, color=color_for(palette, index))
        label(fig, 750, y, name, fontsize=9.5, color=text,
              fontweight="bold" if key == "capyctl" else "normal")
        for n, x in zip(recipe_table.LEVELS, short_rights):
            value = measured[key]["concurrency"][str(n)]["ttft_s"] * 1000
            label(fig, x, y + 15, f"{value:.0f}", fontsize=10, ha="right", color=text)
        for n, x in zip((1, 4), long_rights):
            value = measured[key]["ttft24k"][str(n)]
            label(fig, x, y + 15, f"{value:.2f}", fontsize=10, ha="right", color=text)
    label(fig, 735, 617, "llama.cpp leads 1-stream decode; one math repetition/cap failure.",
          fontsize=8.4, color=muted)
    label(fig, 44, 639, "GSM8K 237/250 vs production 232/250 · tools 30/30 · retrieval 48/48 · next-token agreement 98.15%; OpenAI self-consistency 98.39% (different targets)",
          fontsize=8.6, color=muted)
    label(fig, 44, 657, "github.com/edurdias/capyctl-recipes · comparisons/gpt-oss-20b-builds-gb10 · bench/final-matrix.json · 2026-11-19",
          fontsize=8.8, color=muted)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=palette["bg"])
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", type=Path, default=HERE / "bench/final-matrix.json")
    ap.add_argument("--out", type=Path, default=HERE / "bench/summary")
    ap.add_argument("--x-card", type=Path)
    ap.add_argument("--x-card-dark", type=Path)
    args = ap.parse_args()
    plt = cb.load_matplotlib("GPT-OSS summary PNGs")
    if plt is None:
        raise SystemExit("Matplotlib is required; use uv run --with matplotlib --with pillow.")
    from PIL import Image
    matrix = json.loads(args.matrix.read_text())
    for theme, external in (("light", args.x_card), ("dark", args.x_card_dark)):
        suffix = "-dark" if theme == "dark" else ""
        destination = args.out / f"summary-wide{suffix}.png"
        card(plt, Image, matrix, destination, theme)
        print("wrote", destination)
        if external:
            # Same rendered card and bytes, rather than a separate chart with drifting labels.
            external.parent.mkdir(parents=True, exist_ok=True)
            external.write_bytes(destination.read_bytes())
            print("wrote", external)


if __name__ == "__main__":
    main()
