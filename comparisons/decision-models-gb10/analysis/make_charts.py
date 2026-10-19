#!/usr/bin/env python3
"""Every chart of ../analysis.md, from the results files of this comparison.

    uv run -q --with matplotlib --with pillow python3 \
        comparisons/decision-models-gb10/analysis/make_charts.py

Inputs (paths relative to comparisons/decision-models-gb10/):

- bench/results/*.json: the eight deployments, capyctl-bench 0.3.0 results files.
- bench/summary.md: the series table's `engine_peak_gib`, CapyCTL's measured peak of each engine
  process (`startup.measured`), passed to `capyctl-bench report` as series meta.
- bench/overhead/results/*.json: CapyCTL's per-request cost before #152 (llama-server directly and
  through CapyCTL, same launch).
- READY_S below: ready times from the six recipes' READMEs and this run's ready_s meta.

Optional inputs, drawn when present (rerun this script after adding them):

- bench/reference/basal-ultra.json: basal-omni-1.0-ultra served directly by basal-serve (not through
  CapyCTL), a capyctl-bench results file of the same schema. Whatever sections and accuracy sets it
  has are drawn. Its memory is read from meta `engine_peak_gib` (GiB) when set.
- bench/overhead2/results/{julia1-oh,clefflash-oh}-{capyctl,direct}.json: the same overhead test on
  CapyCTL main 07ee2ea6 (after #152).

Writes analysis/*.png, light theme, 2000 px wide.
"""
from __future__ import annotations

import json
import math
import re
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMP = HERE.parent
REPO = COMP.parent.parent
sys.path.insert(0, str(REPO / "tools" / "capyctl-bench"))
import capyctl_bench as cb  # noqa: E402  theme, logo, tick and label-placement helpers

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

T = cb.mpl_theme(plt, "light")
DPI = 250
WIDTH_IN = 8.0  # 2000 px
URL = "github.com/edurdias/capyctl-recipes"

DEPLOYMENTS = [  # results file, model, engine
    ("clef-sgl", "Clef", "SGLang"),
    ("clef-lc", "Clef", "llama.cpp"),
    ("clefflash-sgl", "Clef-Flash", "SGLang"),
    ("clefflash-lc", "Clef-Flash", "llama.cpp"),
    ("kev4b-lc", "Kev-4B", "llama.cpp"),
    ("lev-lc", "lev", "llama.cpp"),
    ("julia1-lc", "Julia-1", "llama.cpp"),
    ("laya-lc", "Laya", "llama.cpp"),
]
BASAL_FILE = COMP / "bench" / "reference" / "basal-ultra.json"
BASAL = ("basal-ultra", "basal-omni-1.0-ultra", "basal-serve")
OVERHEAD = {"before": COMP / "bench" / "overhead" / "results", "after": COMP / "bench" / "overhead2" / "results"}

# Ready times, seconds (cold: first deploy, including the checkpoint digest; warm: after a stop).
# From the recipes' READMEs (llamacpp/clef-gb10, llamacpp/clef-flash-gb10, sglang/clef-gb10,
# sglang/clef-flash-gb10, "Numbers" tables).
READY_S = {
    "clef-lc": (78.4, 6.1), "clefflash-lc": (21.3, 3.6),
    "clef-sgl": (423.6, 385.8), "clefflash-sgl": (179.8, 127.4),
}
# Chat models to set the decision models beside, from their recipes in this repository
# (peak memory measured by CapyCTL on the cold start).
CHAT_MODELS = [
    ("Qwen3.6 35B-A3B NVFP4 on SGLang", 34.6),  # sglang/qwen3.6-35b-a3b-nvfp4-32gib-gb10
    ("gpt-oss-120b on vLLM", 93.9),             # vllm/gpt-oss-120b-gb10
]
MANAGED_LIMIT_GIB = 108  # this run's `capyctl start standalone` managed limit
MIN_BIN = 10  # reliability diagrams draw confidence bins with at least this many answers

MODEL_COLOR = {
    "Clef": "#ea580c", "Clef-Flash": "#0369a1", "Kev-4B": "#6d28d9", "lev": "#a16207",
    "Julia-1": "#be185d", "Laya": "#4d7c0f", "basal-omni-1.0-ultra": "#0f766e",
}
ENGINE_MARKER = {"llama.cpp": "o", "SGLang": "s", "basal-serve": "D"}
ENGINE_LINE = {"llama.cpp": "-", "SGLang": "--", "basal-serve": ":"}
ENGINE_COLOR = {"llama.cpp": "#57534e", "SGLang": "#ea580c"}

SETS = [("typed_decisions", "Typed Decisions"), ("api_bank", "API-Bank"), ("gpqa_diamond", "GPQA Diamond"),
        ("home_appliances", "Home appliances"), ("ag_news", "AG News"), ("boolq", "BoolQ")]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def engine_peaks() -> dict[str, float]:
    """{results label: engine_peak_gib} from bench/summary.md's series table."""
    text = (COMP / "bench" / "summary.md").read_text()
    return {m.group(1): float(m.group(2))
            for m in re.finditer(r"^\| [^|]+ \| `([\w-]+)` \|.*engine_peak_gib=([\d.]+)", text, re.M)}


class Run:
    def __init__(self, key: str, model: str, engine: str, data: dict, peak: float | None):
        self.key, self.model, self.engine, self.data = key, model, engine, data
        self.dec = data.get("decisions") or {}
        self.peak = peak
        self.direct = engine == "basal-serve"

    @property
    def label(self) -> str:
        return self.model if self.direct else f"{self.model} · {self.engine}"

    @property
    def color(self) -> str:
        return MODEL_COLOR[self.model]

    @property
    def marker(self) -> str:
        return ENGINE_MARKER[self.engine]

    @property
    def line(self) -> str:
        return ENGINE_LINE[self.engine]

    def point(self, label: str) -> dict | None:
        for p in (self.dec.get("latency") or {}).get("points") or []:
            if p["label"] == label and p["summary"].get("latency_p50_s") is not None:
                return p["summary"]
        return None

    def p50_ms(self, label: str) -> float | None:
        s = self.point(label)
        return None if s is None else s["latency_p50_s"] * 1000

    def errors(self, label: str) -> int:
        s = self.point(label)
        return 0 if s is None else s.get("errors") or 0

    def accuracy_set(self, name: str) -> dict | None:
        for s in (self.dec.get("accuracy") or {}).get("sets") or []:
            if s["name"] == name:
                return s
        return None

    def score(self, name: str) -> float | None:
        """Share right, 0..1: accuracy (classic, typed) or the kit's score (Decision Index components)."""
        s = self.accuracy_set(name)
        if s is None:
            return None
        sm = s["summary"]
        return sm.get("score") if s["kind"] == "di" else sm.get("accuracy")

    def mean_six(self) -> float | None:
        vals = [self.score(n) for n, _ in SETS]
        return None if any(v is None for v in vals) else sum(vals) / len(vals)


def load_runs() -> tuple[list[Run], Run | None]:
    peaks = engine_peaks()
    runs = [Run(k, m, e, json.loads((COMP / "bench" / "results" / f"{k}.json").read_text()), peaks.get(k))
            for k, m, e in DEPLOYMENTS]
    basal = None
    if BASAL_FILE.exists():
        data = json.loads(BASAL_FILE.read_text())
        peak = (data.get("meta") or {}).get("engine_peak_gib")
        basal = Run(*BASAL, data, float(peak) if peak not in (None, "") else None)
    return runs, basal


def load_overhead(which: str) -> dict | None:
    """{model: {path: {point label: p50 ms}}} for the before/after overhead test, or None if absent."""
    d = OVERHEAD[which]
    out: dict = {}
    for model, stem in (("Julia-1", "julia1-oh"), ("Clef-Flash", "clefflash-oh")):
        for path in ("capyctl", "direct"):
            f = d / f"{stem}-{path}.json"
            if not f.exists():
                return None
            dec = json.loads(f.read_text())["decisions"]
            out.setdefault(model, {})[path] = {p["label"]: p["summary"]["latency_p50_s"] * 1000
                                               for p in dec["latency"]["points"]}
    return out


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def fmt_ms(v: float) -> str:
    if v >= 10000:
        return f"{v / 1000:.1f} s"
    if v >= 1000:
        return f"{v / 1000:.2f} s"
    return f"{v:.1f} ms" if v < 10 else f"{v:.0f} ms"


def fmt_s(v: float) -> str:
    return f"{v:.1f} s" if v < 10 else f"{v:.0f} s"


def ms_tick(v: float, _=None) -> str:
    if v >= 1000:
        return f"{v / 1000:g} s"
    return f"{v:g} ms"


def tok_tick(v: float, _=None) -> str:
    return f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"


class Frame:
    """A figure with the CapyCTL logo, a title, a subtitle and a footer; `box` is what is left."""

    def __init__(self, height_in: float, title: str, subtitle: str, foot: str):
        self.fig = plt.figure(figsize=(WIDTH_IN, height_in), dpi=DPI)
        self.fig.patch.set_facecolor(T["bg"])
        self.h = height_in
        pad = 0.3
        y = pad  # inches from the top
        logo = T["logo"]
        if logo.exists():
            from PIL import Image
            img = Image.open(logo)
            lh = 0.28
            lw = lh * img.width / img.height
            ax = self.fig.add_axes((pad / WIDTH_IN, 1 - (y + lh) / height_in, lw / WIDTH_IN, lh / height_in))
            ax.imshow(img)
            ax.axis("off")
            y += lh + 0.14
        for line in self.wrap(title, 19, "bold", WIDTH_IN - 2 * pad):
            self.fig.text(pad / WIDTH_IN, 1 - y / height_in, line, fontsize=19, fontweight="bold",
                          color=T["text"], va="top")
            y += 19 / 72 * 1.3
        y += 0.04
        for line in self.wrap(subtitle, 12.5, "normal", WIDTH_IN - 2 * pad):
            self.fig.text(pad / WIDTH_IN, 1 - y / height_in, line, fontsize=12.5, color=T["muted"], va="top")
            y += 12.5 / 72 * 1.35
        foot_lines = self.wrap(foot, 9.5, "normal", WIDTH_IN - 2 * pad)
        fy = 0.16 + 9.5 / 72 * 1.35 * (len(foot_lines) + 1)
        for k, line in enumerate(foot_lines):
            self.fig.text(pad / WIDTH_IN, (fy - 0.06 - k * 9.5 / 72 * 1.35) / height_in, line, fontsize=9.5,
                          color=T["muted"], va="top")
        self.fig.text(pad / WIDTH_IN, (fy - 0.06 - len(foot_lines) * 9.5 / 72 * 1.35) / height_in, URL,
                      fontsize=9.5, color=T["accent_text"], va="top", fontweight="bold")
        self.top = 1 - (y + 0.25) / height_in
        self.bottom = (fy + 0.1) / height_in
        self.left = (pad + 0.05) / WIDTH_IN
        self.right = 1 - pad / WIDTH_IN

    def below(self, inches: float) -> float:
        """The figure fraction `inches` under the top of the free box (room for panel titles)."""
        return self.top - inches / self.h

    def wrap(self, text: str, size: float, weight: str, width_in: float) -> list[str]:
        r = self.fig.canvas.get_renderer()
        out, cur = [], ""
        for word in text.split():
            cand = f"{cur} {word}".strip()
            t = self.fig.text(0, 0, cand, fontsize=size, fontweight=weight)
            w = t.get_window_extent(renderer=r).width / DPI
            t.remove()
            if cur and w > width_in:
                out.append(cur)
                cur = word
            else:
                cur = cand
        return out + ([cur] if cur else [])

    def save(self, name: str) -> Path:
        path = HERE / name
        self.fig.savefig(path, facecolor=T["bg"])
        plt.close(self.fig)
        print(f"wrote {path.relative_to(REPO)}")
        return path


def style(ax, grid: str = "y") -> None:
    ax.set_facecolor(T["bg"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(T["rule"])
    ax.tick_params(labelsize=11.5, length=0, pad=5, colors=T["muted"])
    if grid:
        ax.grid(True, axis=grid, color=T["rule"], linewidth=0.8)
    ax.set_axisbelow(True)


def panel_title(ax, title: str, sub: str = "", x: float = 0) -> None:
    """Bold title and a muted line under it, left-aligned at x (axes fraction; negative reaches over tick labels)."""
    ax.set_title(title, loc="left", x=x, fontsize=14, fontweight="bold", color=T["text"], pad=24 if sub else 8)
    if sub:
        ax.text(x, 1.02, sub, transform=ax.transAxes, fontsize=10.5, color=T["muted"], va="bottom")


def spread(ys: list[float], gap: float, lo: float, hi: float) -> list[float]:
    """Positions near ys (display units), at least `gap` apart, kept inside [lo, hi]."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    pos = [ys[i] for i in order]
    for _ in range(200):
        moved = False
        for k in range(1, len(pos)):
            d = pos[k] - pos[k - 1]
            if d < gap:
                shift = (gap - d) / 2
                pos[k - 1] -= shift
                pos[k] += shift
                moved = True
        pos = [min(max(p, lo), hi) for p in pos]
        if not moved:
            break
    out = [0.0] * len(ys)
    for k, i in enumerate(order):
        out[i] = pos[k]
    return out


def end_labels(fig, ax, items: list[tuple[float, float, str, str]], size: float = 11) -> None:
    """Labels right of the axes for line ends (x, y data, text, color), spread so none overlap."""
    fig.canvas.draw()
    to_disp = ax.transData.transform
    bbox = ax.get_window_extent()
    anchors = [to_disp((x, y)) for x, y, _, _ in items]
    gap = size / 72 * DPI * 1.3
    ys = spread([a[1] for a in anchors], gap, bbox.y0 - gap * 2, bbox.y1 + gap)
    inv = fig.transFigure.inverted()
    x_text = bbox.x1 + 0.12 * DPI
    for (x, y, text, color), a, ty in zip(items, anchors, ys):
        fx, fy = inv.transform((x_text, ty))
        fig.text(fx, fy, text, fontsize=size, color=color, va="center", ha="left", fontweight="bold")
        (ax0, ay0), (ax1, ay1) = inv.transform(a), inv.transform((x_text - 0.03 * DPI, ty))
        fig.add_artist(Line2D([ax0 + 0.006, ax1], [ay0, ay1], transform=fig.transFigure, color=color,
                              linewidth=0.7, alpha=0.6))


def place_point_labels(fig, ax, points: list[tuple[Run, float, float]], size: float = 11) -> None:
    """Each point labelled beside its own marker, by capyctl-bench's place_labels."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts, sizes = [], []
    for run, x, y in points:
        t = ax.text(x, y, run.label, fontsize=size, color=T["text"], ha="left", va="bottom", zorder=5)
        bb = t.get_window_extent(renderer=r)
        texts.append(t)
        sizes.append((bb.width, bb.height))
    anchors = [tuple(ax.transData.transform((x, y))) for _, x, y in points]
    half = math.sqrt(110) / 2 * DPI / 72
    placed = cb.place_labels(anchors, sizes, tuple(ax.bbox.extents), marker=half)
    inv = ax.transData.inverted()
    for t, a, (b, end) in zip(texts, anchors, placed):
        t.set_position(inv.transform((b[0], b[1])))
        if end is not None:
            (px, py), (qx, qy) = inv.transform(a), inv.transform(end)
            ax.plot([px, qx], [py, qy], color=T["muted"], linewidth=0.8, zorder=3)


def bar_label(ax, x: float, v: float, text: str, size: float = 10.5, color: str | None = None) -> None:
    ax.annotate(text, (x, v), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                fontsize=size, fontweight="bold", color=color or T["text"])


SOURCE = ("One NVIDIA GB10, 128 GB unified memory. CapyCTL main 075023d7 (Julia-1 and Laya: the build merged as "
          "8824b622), llama.cpp v0.7.0, SGLang 0.5.22, capyctl-bench 0.3.0.")
BASAL_NOTE = "basal-omni-1.0-ultra: served directly by basal-serve --mode eager, not through CapyCTL."


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def chart_quality_speed(runs: list[Run], basal: Run | None) -> None:
    allr = runs + ([basal] if basal else [])
    fr = Frame(10.4, "Quality against speed",
               "Accuracy against the time to answer one question about a 2k-token text (p50, log scale). "
               "Up and to the left is better.",
               SOURCE + (" " + BASAL_NOTE if basal else "")
               + " Mean of six: the unweighted mean of Typed Decisions, API-Bank, GPQA Diamond, Home appliances, "
               "AG News and BoolQ.")
    gs = fr.fig.add_gridspec(2, 1, left=fr.left + 0.06, right=fr.right, top=fr.below(0.55), bottom=fr.bottom + 0.05,
                             hspace=0.42)
    panels = [
        ("Typed Decisions, accuracy %", "400 agent-workflow cases, 2,000 typed questions",
         lambda r: r.score("typed_decisions")),
        ("Mean of six tests, %", "a specialist that is weak elsewhere drops here",
         lambda r: r.mean_six()),
    ]
    xs = [r.p50_ms("1q 2k") for r in allr if r.p50_ms("1q 2k")]
    x0, x1 = min(xs) / 1.8, max(xs) * 2.2
    for k, (title, sub, f) in enumerate(panels):
        ax = fr.fig.add_subplot(gs[k])
        style(ax, grid="both")
        pts = [(r, r.p50_ms("1q 2k"), f(r) * 100) for r in allr if r.p50_ms("1q 2k") and f(r) is not None]
        for r, x, y in pts:
            ax.scatter([x], [y], s=110, color=r.color, marker=r.marker, edgecolors=T["bg"], linewidths=1, zorder=4)
        ys = [p[2] for p in pts]
        ax.set_xscale("log")
        ax.set_xlim(x0, x1)
        ax.set_ylim(math.floor((min(ys) - 6) / 10) * 10, min(100, math.ceil((max(ys) + 5) / 10) * 10))
        ax.xaxis.set_major_locator(FixedLocator(cb.log_ticks_125(x0, x1)))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(ms_tick))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("1 question, 2k-token text, p50 (log)", fontsize=11.5, color=T["muted"])
        panel_title(ax, title, sub)
        place_point_labels(fr.fig, ax, pts, size=11)
        if basal and f(basal) is None and k == 1:
            ax.text(0.99, 0.03, "basal-omni-1.0-ultra: not run on all six", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=10, color=T["muted"])
    fr.save("quality-speed.png")


def chart_latency_input(runs: list[Run], basal: Run | None) -> None:
    allr = runs + ([basal] if basal else [])
    fr = Frame(7.6, "Time grows with the length of the text",
               "One question per request, p50 latency against the input tokens the engine counted "
               "(text, question and template), log-log. Every text is new.",
               SOURCE + " Julia-1 and Laya: 8,192-token context; at the 8k point 8 of 100 inputs ran over it "
                        "and were refused, the p50 is of the rest (hollow marker)."
               + (" " + BASAL_NOTE if basal else ""))
    ax = fr.fig.add_axes((fr.left + 0.07, fr.bottom + 0.06, 0.53, fr.top - fr.bottom - 0.06))
    style(ax, grid="both")
    items = []
    allx, ally = [], []
    for r in allr:
        pts = []
        for lab in ("1q 200", "1q 2k", "1q 8k"):
            s = r.point(lab)
            if s:
                pts.append((s["input_tokens"]["median"], s["latency_p50_s"] * 1000, r.errors(lab)))
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=r.color, linestyle=r.line, linewidth=2.2, zorder=3)
        for x, y, err in pts:
            ax.scatter([x], [y], s=46, marker=r.marker, zorder=4,
                       color=T["bg"] if err else r.color, edgecolors=r.color, linewidths=1.6)
        allx += [p[0] for p in pts]
        ally += [p[1] for p in pts]
        items.append((pts[-1][0], pts[-1][1], f"{r.label}  {fmt_ms(pts[-1][1])}", r.color))
    ax.set_xscale("log")
    ax.set_yscale("log")
    xlo, xhi = 180, max(allx) * 1.25
    ylo, yhi = min(ally) / 1.5, max(ally) * 1.6
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.xaxis.set_major_locator(FixedLocator(cb.log_ticks_125(xlo, xhi)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(tok_tick))
    ax.yaxis.set_major_locator(FixedLocator([t for t in (10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000,
                                                         50000, 100000, 200000) if ylo <= t <= yhi]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(ms_tick))
    ax.set_xlabel("input tokens (log)", fontsize=11.5, color=T["muted"])
    ax.set_ylabel("latency p50 (log)", fontsize=11.5, color=T["muted"])
    end_labels(fr.fig, ax, items, size=10.5)
    fr.save("latency-input.png")


def chart_amortization(runs: list[Run], basal: Run | None) -> None:
    allr = runs + ([basal] if basal else [])
    fr = Frame(7.4, "Several questions, one text",
               "One request with 1, 3 or 5 questions about the same 2k-token text: its time as a multiple of "
               "the 1-question request (p50). Dotted: the same questions sent one per request.",
               SOURCE + (" " + BASAL_NOTE if basal else ""))
    ax = fr.fig.add_axes((fr.left + 0.06, fr.bottom + 0.06, 0.58, fr.top - fr.bottom - 0.06))
    style(ax, grid="y")
    ax.plot([1, 5], [1, 5], color=T["muted"], linestyle=":", linewidth=1.8, zorder=2)
    items = [(5, 5, "one per request  5×", T["muted"])]
    top = 5.0
    for r in allr:
        base = r.p50_ms("1q 2k")
        if not base:
            continue
        pts = [(n, r.p50_ms(f"{n}q 2k") / base) for n in (1, 3, 5) if r.p50_ms(f"{n}q 2k")]
        top = max(top, max(p[1] for p in pts))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=r.color, linestyle=r.line, linewidth=2.2,
                marker=r.marker, markersize=6.5, zorder=3)
        n_last, v_last = pts[-1]
        items.append((n_last, v_last, f"{r.label}  {v_last:.2f}×", r.color))
    ax.set_xlim(0.6, 5.2)
    ax.set_ylim(0, math.ceil(top + 0.4))
    ax.set_xticks([1, 3, 5])
    ax.set_xticklabels(["1 question", "3 questions", "5 questions"], fontsize=11.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    end_labels(fr.fig, ax, items, size=10.5)
    fr.save("amortization.png")


def chart_reliability(runs: list[Run], basal: Run | None) -> None:
    fr = Frame(11.6, "Can you trust the probabilities?",
               "Typed Decisions, 2,000 answers per deployment. Across: how sure the model was of its top "
               "answer; up: how often that answer was right; 10 confidence bins, those with at least "
               f"{MIN_BIN} answers (marker area: answers). On the dotted line a 70% answer is right 70% of the "
               "time; below it the model is overconfident.",
               SOURCE + " Clef and Clef-Flash: llama.cpp solid, SGLang dashed. ECE: expected calibration error, "
                        "the answer-weighted mean gap between confidence and accuracy." + (" " + BASAL_NOTE if basal else ""))
    gs = fr.fig.add_gridspec(3, 3, left=fr.left + 0.05, right=fr.right, top=fr.below(0.3), bottom=fr.bottom + 0.03,
                             hspace=0.5, wspace=0.28)
    by_model: dict[str, list[Run]] = {}
    for r in runs:
        by_model.setdefault(r.model, []).append(r)
    models = ["Clef", "Clef-Flash", "Kev-4B", "lev", "Julia-1", "Laya"]
    slots = [(i // 3, i % 3) for i in range(7)]
    group = [by_model[m] for m in models] + [[basal] if basal else []]
    names = models + ["basal-omni-1.0-ultra"]
    for (row, col), name, rs in zip(slots, names, group):
        ax = fr.fig.add_subplot(gs[row, col])
        style(ax, grid="both")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.plot([0, 1], [0, 1], color=T["muted"], linestyle=":", linewidth=1.2, zorder=2)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:g}%"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:g}%"))
        ax.tick_params(labelsize=10)
        ax.set_title(name, loc="left", fontsize=12.5, fontweight="bold", color=T["text"], pad=6)
        if not rs or rs[0].accuracy_set("typed_decisions") is None:
            ax.text(0.5, 0.5, "pending:\nbench/reference/\nbasal-ultra.json", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color=T["muted"],
                    bbox={"facecolor": T["bg"], "edgecolor": T["rule"], "boxstyle": "round,pad=0.4"})
            continue
        notes = []
        for r in sorted(rs, key=lambda r: r.engine != "llama.cpp"):
            sm = r.accuracy_set("typed_decisions")["summary"]
            bins = [b for b in sm["ece_bins"] if b["n"] >= MIN_BIN]
            cx = [b["confidence"] for b in bins]
            cy = [b["accuracy"] for b in bins]
            ax.plot(cx, cy, color=r.color, linestyle=r.line, linewidth=1.8, zorder=3)
            ax.scatter(cx, cy, s=[min(160, 8 + b["n"] / 5) for b in bins], color=r.color, marker=r.marker,
                       edgecolors=T["bg"], linewidths=0.6, zorder=4, alpha=0.9, clip_on=False)
            if r.engine == "llama.cpp" or r.direct:
                ax.fill_between(cx, cy, cx, color=r.color, alpha=0.12, zorder=1, linewidth=0)
            notes.append(f"ECE {sm['ece']:.3f}" + (f" ({r.engine})" if len(rs) > 1 else ""))
        ax.text(0.04, 0.96, "\n".join(notes), transform=ax.transAxes, ha="left", va="top", fontsize=9.5,
                color=T["text"], linespacing=1.3)
    # ECE of every deployment, sorted; names after the values so nothing runs into the panel on the left
    ax = fr.fig.add_subplot(gs[2, 1:])
    style(ax, grid="")
    rs = [r for r in runs + ([basal] if basal else []) if r.accuracy_set("typed_decisions")]
    rs.sort(key=lambda r: -r.accuracy_set("typed_decisions")["summary"]["ece"])
    vals = [r.accuracy_set("typed_decisions")["summary"]["ece"] for r in rs]
    ax.barh(range(len(rs)), vals, color=[r.color for r in rs], height=0.7, zorder=2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    for i, (r, v) in enumerate(zip(rs, vals)):
        ax.annotate(f"{v:.3f}  {r.label}", (v, i), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=9.5, color=T["text"])
    ax.set_xlim(0, max(vals) * 2.1)
    ax.xaxis.set_major_locator(FixedLocator([0, 0.1, 0.2]))
    ax.tick_params(labelsize=10)
    ax.set_title("ECE, lower is better", loc="left", fontsize=12.5, fontweight="bold", color=T["text"], pad=6)
    fr.save("reliability-typed.png")


def chart_engines(runs: list[Run]) -> None:
    rd = {r.key: r for r in runs}
    fr = Frame(14.6, "The same model on llama.cpp and SGLang",
               "Clef (27B) and Clef-Flash: llama.cpp runs the Q8_0 GGUF, SGLang the BF16 checkpoint. "
               "SGLang reads long texts about twice as fast; llama.cpp starts in seconds and holds a "
               "fraction of the memory; the answers are the same within sampling noise.",
               SOURCE + " Peak memory: CapyCTL's measured peak of the engine process; SGLang sizes its cache "
                        "from the memory it is given, so its peak follows the reservation. Ready times from the "
                        "recipes.")
    H = fr.h

    def rect(x: float, y: float, w: float, h: float) -> tuple:
        """Axes at x inches from the left, y inches under the free box's top, w x h inches."""
        return (x / WIDTH_IN, fr.top - (y + h) / H, w / WIDTH_IN, h / H)

    engines = ["llama.cpp", "SGLang"]
    colors = [ENGINE_COLOR[e] for e in engines]

    def hbars(ax, rows: list[str], values: list[list[float]], fmt) -> None:
        n = len(values[0])
        bh = 0.8 / n
        top = max(v for vs in values for v in vs)
        for i, vs in enumerate(values):
            for k, v in enumerate(vs):
                y = i + (k - (n - 1) / 2) * bh
                ax.barh(y, v, height=bh * 0.9, color=colors[k], zorder=2)
                ax.annotate(fmt(v), (v, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=10,
                            fontweight="bold", color=T["text"])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, fontsize=10.5, color=T["text"], linespacing=1.15)
        ax.invert_yaxis()
        ax.set_xlim(0, top * 1.3)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)

    eh = [matplotlib.patches.Patch(color=c, label=e) for e, c in zip(engines, colors)]
    fr.fig.legend(handles=eh, loc="upper left", bbox_to_anchor=(fr.left - 0.01, fr.top + 0.01), ncol=2, fontsize=11.5,
                  frameon=False)
    # latency: one panel, a row per model and text size, the speed-up in the row label
    y = 0.95
    ax = fr.fig.add_axes(rect(2.25, y, 5.4, 3.5))
    style(ax, grid="")
    rows, vals = [], []
    for model, key in (("Clef-Flash", "clefflash"), ("Clef", "clef")):
        for s, name in (("200", "200 tokens"), ("2k", "2k tokens"), ("8k", "8k tokens")):
            lc, sg = rd[f"{key}-lc"].p50_ms(f"1q {s}"), rd[f"{key}-sgl"].p50_ms(f"1q {s}")
            rows.append(f"{model}, {name}\nSGLang {lc / sg:.1f}× faster")
            vals.append([lc, sg])
    hbars(ax, rows, vals, fmt_ms)
    panel_title(ax, "Latency", "1 question, p50, lower is better", x=-2.0 / 5.4)
    # memory and ready time side by side
    y += 3.5 + 0.95
    ax = fr.fig.add_axes(rect(1.25, y, 2.3, 1.7))
    style(ax, grid="")
    hbars(ax, ["Clef-Flash", "Clef"], [[rd["clefflash-lc"].peak, rd["clefflash-sgl"].peak],
                                       [rd["clef-lc"].peak, rd["clef-sgl"].peak]], lambda v: f"{v:.1f} GiB")
    panel_title(ax, "Peak memory", "GiB, lower is better", x=-0.95 / 2.3)
    ax = fr.fig.add_axes(rect(5.25, y, 2.4, 1.7))
    style(ax, grid="")
    rows, vals = [], []
    for model, key in (("Clef-Flash", "clefflash"), ("Clef", "clef")):
        lc, sg = READY_S[f"{key}-lc"], READY_S[f"{key}-sgl"]
        rows += [f"{model}, warm", f"{model}, cold"]
        vals += [[lc[1], sg[1]], [lc[0], sg[0]]]
    hbars(ax, rows, vals, fmt_s)
    panel_title(ax, "Ready time", "seconds; cold is the first deploy", x=-1.35 / 2.4)
    # quality: SGLang minus llama.cpp, per test, against sampling noise
    y += 1.7 + 0.95
    ax = fr.fig.add_axes(rect(2.25, y, 3.9, 2.7))
    style(ax, grid="x")
    n_rows = {}
    for name, _ in SETS:
        s = rd["clef-lc"].accuracy_set(name)["summary"]
        n_rows[name] = s.get("decisions") or s.get("items") or s.get("requests")
    lim = 13
    for i, (name, title) in enumerate(SETS):
        ps = [rd[k].score(name) for k in ("clef-lc", "clef-sgl", "clefflash-lc", "clefflash-sgl")]
        p = sum(ps) / len(ps)
        noise = 1.96 * math.sqrt(2 * p * (1 - p) / n_rows[name]) * 100
        ax.add_patch(matplotlib.patches.Rectangle((-noise, i - 0.38), 2 * noise, 0.76, color=T["rule"], zorder=1,
                                                  linewidth=0))
        for off, model, key, col in ((-0.14, "Clef", "clef", 0), (0.14, "Clef-Flash", "clefflash", 1)):
            d = (rd[f"{key}-sgl"].score(name) - rd[f"{key}-lc"].score(name)) * 100
            ax.scatter([d], [i + off], s=70, color=MODEL_COLOR[model], zorder=3, edgecolors=T["bg"])
            ax.text(lim * (1.2 + 0.42 * col), i, f"{d:+.1f}", ha="center", va="center", fontsize=10.5,
                    fontweight="bold", color=MODEL_COLOR[model], clip_on=False)
    for col, model in enumerate(("Clef", "Clef-Flash")):
        ax.text(lim * (1.2 + 0.42 * col), -0.75, model, ha="center", va="bottom", fontsize=10.5, fontweight="bold",
                color=MODEL_COLOR[model], clip_on=False)
    ax.axvline(0, color=T["muted"], linewidth=1)
    ax.set_yticks(range(len(SETS)))
    ax.set_yticklabels([f"{t} ({n_rows[n]:,})" for n, t in SETS], fontsize=10.5, color=T["text"])
    ax.set_ylim(len(SETS) - 0.5, -0.5)
    ax.set_xlim(-lim, lim)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+g}" if v else "0"))
    ax.set_xlabel("SGLang minus llama.cpp, points", fontsize=11, color=T["muted"])
    panel_title(ax, "Accuracy, SGLang minus llama.cpp",
                "grey: 95% sampling noise of the difference (rows or answers in brackets)", x=-2.0 / 3.9)
    fr.save("engines.png")


def chart_memory(runs: list[Run], basal: Run | None) -> None:
    allr = [r for r in runs + ([basal] if basal else []) if r.peak is not None]
    allr.sort(key=lambda r: r.peak)
    fr = Frame(8.4, "What fits beside a chat model",
               f"Peak memory of each decision deployment (GiB). The vertical lines are what is left of a "
               f"{MANAGED_LIMIT_GIB} GiB managed limit (this run's, on a 128 GB GB10) once a chat model is running.",
               SOURCE + " Peak: CapyCTL's measured peak of the engine process. Chat model peaks from their recipes "
                        "in this repository. SGLang's peak follows the memory it is given, so it is not a minimum."
               + (" " + BASAL_NOTE if basal else ""))
    ax = fr.fig.add_axes((fr.left + 0.25, fr.bottom + 0.06, fr.right - fr.left - 0.27, fr.top - fr.bottom - 0.1))
    style(ax, grid="")
    ys = range(len(allr))
    ax.barh(ys, [r.peak for r in allr], color=[r.color for r in allr], height=0.66, zorder=2)
    for i, r in enumerate(allr):
        ax.annotate(f"{r.peak:.2f} GiB" if r.peak < 1 else f"{r.peak:.1f} GiB", (r.peak, i), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=10.5, fontweight="bold", color=T["text"],
                    bbox={"facecolor": T["bg"], "edgecolor": "none", "pad": 0.5}, zorder=5)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r.label for r in allr], fontsize=11, color=T["text"])
    ax.set_xlim(0, MANAGED_LIMIT_GIB + 4)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("GiB", fontsize=11, color=T["muted"])
    n = len(allr)
    # (x, label, line style, label row above the bars, label side): the smaller chat model's label sits
    # just above the bars, left of its line; the other two a row higher, so no label meets a line
    (small, small_gib), (big, big_gib) = sorted(CHAT_MODELS, key=lambda c: c[1])
    lines = [(MANAGED_LIMIT_GIB, f"{MANAGED_LIMIT_GIB} GiB\nmanaged limit", "-", 1, "right"),
             (MANAGED_LIMIT_GIB - small_gib, f"{MANAGED_LIMIT_GIB - small_gib:.1f} GiB left beside\n"
              f"{small}\n({small_gib} GiB)", "--", 0, "right"),
             (MANAGED_LIMIT_GIB - big_gib, f"{MANAGED_LIMIT_GIB - big_gib:.1f} GiB left beside\n"
              f"{big}\n({big_gib} GiB)", "--", 1, "left")]
    row_y = [n - 0.3, n + 1.9]
    for x, text, ls, row, side in lines:
        ax.vlines(x, -0.6, row_y[row] + 1.0 + 0.55 * (text.count("\n") - 1), color=T["text"], linestyle=ls,
                  linewidth=1.3, zorder=1.5)
        ax.text(x - 1.2 if side == "right" else x + 1.2, row_y[row], text, ha=side, va="bottom", fontsize=9.5,
                color=T["text"], linespacing=1.15)
    ax.set_ylim(-0.6, n + 3.6)
    fr.save("memory.png")


POINTS = [("1q 200", "1 question\n200 tokens"), ("1q 2k", "1 question\n2k tokens"),
          ("3q 200", "3 questions\n200 tokens"), ("3q 2k", "3 questions\n2k tokens")]


def chart_overhead() -> None:
    before, after = load_overhead("before"), load_overhead("after")
    fr = Frame(10.2, "What CapyCTL adds to each request",
               "p50 latency through CapyCTL minus the same llama-server launched directly, one request at a "
               "time, 100 requests per point. Before: the build used for this run. After: main 07ee2ea6, with #152.",
               "One NVIDIA GB10. llama.cpp v0.7.0, capyctl-bench 0.3.0. Under each point: its p50 without CapyCTL "
               "in the first test." + ("" if after else " After #152: pending (bench/overhead2/results)."))
    gs = fr.fig.add_gridspec(2, 1, left=fr.left + 0.02, right=fr.right, top=fr.below(0.95), bottom=fr.bottom + 0.1,
                             hspace=0.75)
    colors = {"before": "#a8a29e", "after": T["accent"]}
    top, low = 0, 0
    data = {"before": before, "after": after}
    for d in data.values():
        if d:
            added = [d[m]["capyctl"][p] - d[m]["direct"][p] for m in d for p, _ in POINTS]
            top, low = max(top, max(added)), min(low, min(added))
    for row, model in enumerate(("Julia-1", "Clef-Flash")):
        ax = fr.fig.add_subplot(gs[row, 0])
        style(ax, grid="")
        for gi, (p, _) in enumerate(POINTS):
            for k, which in enumerate(("before", "after")):
                x = gi + (k - 0.5) * 0.38
                d = data[which]
                if d is None:
                    ax.text(x, top * 0.04, "pending", rotation=90, ha="center", va="bottom", fontsize=9.5,
                            color=T["muted"])
                    continue
                v = d[model]["capyctl"][p] - d[model]["direct"][p]
                ax.bar(x, v, width=0.34, color=colors[which], zorder=2)
                bar_label(ax, x, max(v, 0), f"{v:+.1f}", size=10.5)
        ax.set_xticks(range(len(POINTS)))
        ax.set_xticklabels([f"{lab}\ndirect {fmt_ms(before[model]['direct'][p])}" for p, lab in POINTS],
                           fontsize=10.5, color=T["text"])
        ax.set_ylim(low * 1.3, top * 1.25)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        panel_title(ax, model, "ms added, lower is better")
    eh = [matplotlib.patches.Patch(color=colors["before"], label="before #152"),
          matplotlib.patches.Patch(color=colors["after"], label="after #152 (main 07ee2ea6)")]
    fr.fig.legend(handles=eh, loc="upper left", bbox_to_anchor=(fr.left - 0.01, fr.top + 0.01), ncol=2, fontsize=10.5,
                  frameon=False)
    fr.save("overhead.png")


def main() -> int:
    runs, basal = load_runs()
    print("basal-omni-1.0-ultra:", "drawn" if basal else f"pending ({BASAL_FILE.relative_to(COMP)} not found)")
    print("overhead after #152:", "drawn" if load_overhead("after") else "pending (bench/overhead2/results not found)")
    chart_quality_speed(runs, basal)
    chart_latency_input(runs, basal)
    chart_amortization(runs, basal)
    chart_reliability(runs, basal)
    chart_engines(runs)
    chart_memory(runs, basal)
    chart_overhead()
    return 0


if __name__ == "__main__":
    sys.exit(main())
