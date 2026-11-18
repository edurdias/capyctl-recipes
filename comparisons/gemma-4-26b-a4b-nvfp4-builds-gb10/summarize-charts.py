#!/usr/bin/env python3
"""Plot the measured Gemma recipes using capyctl-bench's summary theme.

python3 summarize-charts.py
uv run --with matplotlib --with pillow python3 summarize-charts.py

Reads bench/final-matrix.json and tensors/results/summary.json; never runs a benchmark.
PNG support is optional, as in capyctl-bench. --x-card writes an additional post card.
"""

import argparse
import importlib.util
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




def load_sibling(name, filename):
    """Import a sibling script whose file name is not a valid module name."""
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recipe_table = load_sibling("recipe_table", "recipe-table.py")

# The recipe card's series: (build, engine, drafter, label, short name, palette index). The hero first
# in the brand orange (palette index 0), the production setup last in grey; the rest calm and distinct.
CARD_SERIES = (
    ("r3", "vLLM 0.32.0", "MTP 3", "CapyCTL · vLLM + MTP 3", "CapyCTL", 0),
    ("RedHat", "vLLM 0.32.0", "MTP 3", "RedHat NVFP4 · vLLM + MTP 3", "RedHat", 2),
    ("Unsloth", "vLLM 0.32.0", "MTP 3", "Unsloth NVFP4 · vLLM + MTP 3", "Unsloth", 3),
    ("NVIDIA", "vLLM 0.32.0", "MTP 3", "NVIDIA NVFP4 · vLLM + MTP 3", "NVIDIA", 4),
    ("NVIDIA", "SGLang 0.5.22", "none", "Today's production · NVIDIA, SGLang, no drafter", "today's production", 1),
)
HERO, PRODUCTION = 0, len(CARD_SERIES) - 1  # indexes into CARD_SERIES
AGREEMENT_KEYS = {"r3": "r3", "RedHat": "redhat", "Unsloth": "unsloth", "NVIDIA": "nvidia"}
CARD_WIDTH, CARD_HEIGHT = cb.SUMMARY_SIZES["wide"]  # logical px; rendered at 2x


def card_series(matrix):
    """[(spec, matrix row, {streams: (tok/s, TTFT s)}, {streams: 24k-prompt TTFT s})] for CARD_SERIES,
    with recipe-table.py's exact row selection and math."""
    out = []
    for spec in CARD_SERIES:
        row = select(matrix["rows"], *spec[:3])
        data = recipe_table.measure(row)
        if not data or not data.get("concurrency"):
            raise ValueError(f"No measure.json with concurrency rounds for {spec[:3]}")
        out.append((spec, row) + recipe_table.levels(data))
    return out


def percent(value, base):
    return 100 * (value / base - 1)


def signed(pct):
    """+93% / −6.1%: one decimal under 10 so a near tie reads as one, with a real minus sign."""
    return (f"{pct:+.1f}%" if abs(pct) < 10 else f"{pct:+.0f}%").replace("-", "−")


def text_width(fig, text, size, weight="normal"):
    """Width in logical px (the figure is rendered at 2x), measured with the figure's font."""
    t = fig.text(0, 0, text, fontsize=size, fontweight=weight)
    width = t.get_window_extent(renderer=fig.canvas.get_renderer()).width
    t.remove()
    return width / 2


def wrap(fig, text, size, max_px, weight="normal"):
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and text_width(fig, candidate, size, weight) > max_px:
            lines.append(current)
            candidate = word
        current = candidate
    return lines + [current]


def card_text(fig, x, y, text, max_px=None, **kw):
    """fig.text at logical px from the top-left; raises instead of overflowing max_px."""
    if max_px is not None and text_width(fig, text, kw.get("fontsize", 11), kw.get("fontweight", "normal")) > max_px:
        raise ValueError(f"Text wider than its {max_px:.0f}px slot: {text!r}")
    return fig.text(x / CARD_WIDTH, 1 - y / CARD_HEIGHT, text, **kw)


def throughput_panel(fig, ax, palette, series, left, right):
    """Horizontal bars per stream level, the hero first in each group and its value bold. The stream
    label and the hero's lead over the next-best recipe and over production sit left of the axes,
    between `left` and the axes; title, note and legend go above, from `left` to `right`."""
    cb.mpl_panel_style(ax, palette, "wide")
    text, muted, colors = palette["text"], palette["muted"], palette["palette"]
    levels, count = recipe_table.LEVELS, len(series)
    pitch = count + 1.8  # slots per group: the bars and a gap
    size, pad = 10, 6  # label font; logical px between a bar and its value
    box = ax.get_position()
    axes_px, column_px = box.width * CARD_WIDTH, box.x0 * CARD_WIDTH - left - 10
    production = series[PRODUCTION][2]

    bars = []
    for g, n in enumerate(levels):
        for i, (spec, _, short, _) in enumerate(series):
            weight = "bold" if i == HERO else "normal"
            label = f"{short[n][0]:.1f}"
            bars.append((g * pitch + i, short[n][0], colors[spec[5]], label, weight,
                         pad + text_width(fig, label, size, weight)))
    top = max(b[1] for b in bars)
    xmax = max([top / 0.7] + [value / (1 - used / axes_px) for _, value, _, _, _, used in bars])
    unit = xmax / axes_px  # data units per logical px
    for y, value, color, label, weight, _ in bars:
        ax.barh(y, value, color=color, height=0.8)
        ax.text(value + pad * unit, y, label, va="center", fontsize=size, fontweight=weight, color=text)

    for g, n in enumerate(levels):
        ranked = sorted(((short[n][0], spec[4]) for spec, _, short, _ in series), reverse=True)
        (best, best_name), (runner, runner_name) = ranked[:2]
        if best_name != series[HERO][0][4]:
            raise ValueError(f"The hero is not the fastest at {n} streams: {ranked[0]}")
        lead = percent(best, runner)
        lines = ((f"{n} stream{'s' if n > 1 else ''}", 11.5, "bold", text),
                 (f"{signed(lead)} vs {runner_name}, {'a tie' if lead < 2 else 'next best'}", size, "normal", text),
                 (f"{signed(percent(best, production[n][0]))} vs production", size, "normal", text))
        centre = g * pitch + (count - 1) / 2
        for k, (line, font, weight, color) in enumerate(lines):
            if text_width(fig, line, font, weight) > column_px:
                raise ValueError(f"Lead text wider than its {column_px:.0f}px column: {line!r}")
            ax.annotate(line, xy=(0, centre + (k - 1) * 1.1), xycoords=("axes fraction", "data"), xytext=(-10, 0),
                        textcoords="offset points", ha="right", va="center", fontsize=font, fontweight=weight,
                        color=color)
    ax.set_xlim(0, xmax)
    ax.set_ylim(len(levels) * pitch - 1.8 - 0.5, -0.6)  # first group at the top
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    y = (1 - box.y1) * CARD_HEIGHT  # the axes' top edge, logical px from the top
    card_text(fig, left, y - 44, "Total throughput, tokens/s", fontsize=12.5, fontweight="bold", color=text)
    fig.text(right / CARD_WIDTH, 1 - (y - 44) / CARD_HEIGHT, "Higher is better · median of 3 rounds",
             ha="right", fontsize=9.5, color=muted)
    x = left
    for spec, *_ in series:
        name = spec[4][0].upper() + spec[4][1:]
        swatch(fig, x, y - 24.5, 9, colors[spec[5]])
        card_text(fig, x + 14, y - 20, name, fontsize=10, color=text, va="center")
        x += 14 + text_width(fig, name, 10) + 20
    if x - 20 > right:
        raise ValueError("Legend wider than the throughput panel")


def ttft_table(fig, palette, series, x, y, width):
    """Time to first token as a compact table: a swatch and name per recipe (the card's legend) over
    its ms at 1, 2, 4, 8 streams and its seconds with a 24,000-token prompt at 1 and 4 streams.
    Returns the y below the table."""
    text, muted, colors = palette["text"], palette["muted"], palette["palette"]
    columns = [(n, "short") for n in recipe_table.LEVELS] + [(n, "long") for n in (1, 4)]
    step = 60
    rights = [x + width - (len(columns) - 1 - i) * step for i in range(len(columns))]
    values = {(n, kind): [short[n][1] * 1000 if kind == "short" else long[n]
                          for _, _, short, long in series] for n, kind in columns}
    lowest = {key: min(range(len(series)), key=lambda i: vals[i]) for key, vals in values.items()}

    card_text(fig, x, y, "Time to first token", fontsize=12.5, fontweight="bold", color=text)
    y += 24
    mid = (rights[0] - step + rights[3]) / 2
    fig.text(mid / CARD_WIDTH, 1 - y / CARD_HEIGHT, "~50-token prompt, ms", ha="center", fontsize=9.5, color=muted)
    mid = (rights[4] - step + rights[5]) / 2
    fig.text(mid / CARD_WIDTH, 1 - y / CARD_HEIGHT, "24,000-token prompt, s", ha="center", fontsize=9.5, color=muted)
    y += 16
    card_text(fig, x, y, "streams", fontsize=9.5, color=muted)
    for (n, _), right in zip(columns, rights):
        fig.text(right / CARD_WIDTH, 1 - y / CARD_HEIGHT, str(n), ha="right", fontsize=9.5, color=muted)
    y += 14
    rule(fig, x, x + width, y, palette["rule"])
    y += 8
    for i, (spec, _, _, _) in enumerate(series):
        color = colors[spec[5]]
        weight = "bold" if i == HERO else "normal"
        lines = wrap(fig, spec[3], 10, width - 18, weight)
        swatch(fig, x, y + 9, 9, color)
        for line in lines:
            y += 14
            card_text(fig, x + 18, y, line, fontsize=10, fontweight=weight, color=color if i == HERO else text,
                      va="center")
        y += 18
        for key, right in zip(columns, rights):
            value = values[key][i]
            fig.text(right / CARD_WIDTH, 1 - y / CARD_HEIGHT, f"{value:.0f}" if key[1] == "short" else f"{value:.2f}",
                     ha="right", va="center", fontsize=11, color=text,
                     fontweight="bold" if lowest[key] == i else "normal")
        y += 16
    return y


def rule(fig, x0, x1, y, color):
    from matplotlib.lines import Line2D
    fig.add_artist(Line2D([x0 / CARD_WIDTH, x1 / CARD_WIDTH], [1 - y / CARD_HEIGHT] * 2,
                          transform=fig.transFigure, color=color, linewidth=1))


def swatch(fig, x, y, side, color):
    """A side×side square whose top-left is at logical px (x, y)."""
    from matplotlib.patches import Rectangle
    fig.add_artist(Rectangle((x / CARD_WIDTH, 1 - (y + side) / CARD_HEIGHT), side / CARD_WIDTH,
                             side / CARD_HEIGHT, transform=fig.transFigure, facecolor=color, edgecolor="none"))


def recipe_card(plt, image, matrix, agreement, theme, path):
    """2400x1350 card: the hero recipe's tokens/s against the other recipes and today's production
    setup at 1, 2, 4 and 8 streams, with its lead per level; time to first token as a table."""
    series = card_series(matrix)
    fig, palette = base_figure(
        plt, image, theme, "wide", "CapyCTL's Gemma 4 recipe: fastest at every concurrency",
        "One GB10 · Gemma 4 26B A4B · same prompts · temperature 0")
    text, muted = palette["text"], palette["muted"]

    top, bottom = 240, 612  # logical px from the top
    ax = fig.add_axes((290 / CARD_WIDTH, 1 - bottom / CARD_HEIGHT, 400 / CARD_WIDTH, (bottom - top) / CARD_HEIGHT))
    throughput_panel(fig, ax, palette, series, 44, 690)

    hero, production = series[HERO], series[PRODUCTION]
    costs = [percent(hero[2][n][1], production[2][n][1]) for n in recipe_table.LEVELS]
    costs_24k = [percent(hero[3][n], production[3][n]) for n in (1, 4)]
    y = ttft_table(fig, palette, series, 724, 192, 432)
    note = (f"Lower is better; bold is the lowest in its column. Today's production answers first soonest: "
            f"CapyCTL's first token comes {min(costs):.0f}–{max(costs):.0f}% later at 1–8 streams and "
            f"{min(costs_24k):.0f}–{max(costs_24k):.0f}% later with a 24,000-token prompt.")
    for line in wrap(fig, note, 9.5, 432):
        y += 14
        card_text(fig, 724, y, line, fontsize=9.5, color=muted)
    if y > bottom:
        raise ValueError(f"The first-token table runs into the footer: {y:.0f}px")

    def scores(row):
        gsm8k, tools = row.get("gsm8k"), row.get("tools")
        return f"{gsm8k['correct']}/{gsm8k['n']}, {tools['pass']}/{tools['n']}" if gsm8k and tools else None

    seen, agree = set(), []
    for spec, *_ in series:
        if spec[0] not in seen:
            seen.add(spec[0])
            agree.append(f"{spec[4]} {100 * agreement[AGREEMENT_KEYS[spec[0]]]['summary']['agreement']:.2f}%")
    scored = [(spec[4].removeprefix("today's "), scores(row)) for spec, row, _, _ in series if scores(row)]
    foot = ["Same next token as BF16: " + ", ".join(agree) + " · GSM8K/tools: " + "; ".join(f"{name} {score}" for name, score in scored),
            "github.com/edurdias/capyctl-recipes · comparisons/gemma-4-26b-a4b-nvfp4-builds-gb10 · "
            "bench/final-matrix.json · 2026-11-17"]
    for index, line in enumerate(foot):
        card_text(fig, 44, CARD_HEIGHT - 40 + index * 16, line, CARD_WIDTH - 88, fontsize=9.5, color=muted, va="baseline")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=palette["bg"])
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=HERE / "bench/final-matrix.json")
    parser.add_argument("--out", type=Path, default=HERE / "bench/summary")
    parser.add_argument("--x-card", type=Path, help="write the light post card here")
    parser.add_argument("--x-card-dark", type=Path, help="write the dark post card here")
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
    if args.x_card or args.x_card_dark:
        agreement = json.loads((HERE / "tensors/results/summary.json").read_text())
        for theme, path in (("light", args.x_card), ("dark", args.x_card_dark)):
            if path:
                recipe_card(plt, Image, matrix, agreement, theme, path)
                print("wrote", path)


if __name__ == "__main__":
    main()
