"""The summary cards of this comparison, in capyctl-bench's summary style.

uv run --with matplotlib --with pillow python3 card.py

Writes summary/summary-wide.png and summary/hero-wide.png (and their -dark variants)
next to this file, with the theme, logo and size of tools/capyctl-bench.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools" / "capyctl-bench"))
import capyctl_bench as cb  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

OUT = HERE / "summary"
W, H = cb.SUMMARY_SIZES["wide"]
PAD = 44

# Medians of 3 rounds (results/*/fanout.jsonl, whole.jsonl).
THROUGHPUT = [  # label, tok/s, kind
    ("8 lanes, 16 GiB KV cache", 1566, "ours"),
    ("8 lanes, 32 GB-sized", 849, "ours"),
    ("one stream", 310, "ours2"),
    ("Nerv: 5090, NInfer (reported)", 1600, "reported"),
    ("Ash: 5090, TensorFold (reported)", 665, "reported"),
]
PAGE = [  # memory, fan-out s, one stream s (run 2: lanes end on their own)
    ("16 GiB KV cache", 19.9, 31.3),
    ("sized for 32 GB", 31.3, 25.7),
]


def card(theme: str, path: Path) -> None:
    T = cb.mpl_theme(plt, theme)
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=200)
    fig.patch.set_facecolor(T["bg"])
    fx = lambda x: x / W  # noqa: E731
    fy = lambda y: 1 - y / H  # noqa: E731

    y = 34
    img = Image.open(T["logo"])
    lh = 34
    ax = fig.add_axes((fx(PAD), fy(y + lh), fx(lh * img.width / img.height), lh / H))
    ax.imshow(img)
    ax.axis("off")
    y += lh + 22
    fig.text(fx(PAD), fy(y), "Qwen3.8-27B: one coding task, 8 parallel lanes",
             fontsize=25, fontweight="bold", color=T["text"], va="top")
    y += 25 * 1.45 + 4
    fig.text(fx(PAD), fy(y), "RTX PRO 6000 Blackwell · vLLM 0.32.0 + DFlash2 · CapyCTL NVFP4 · "
             "measured through CapyCTL · 2026-11-11", fontsize=13.5, color=T["muted"], va="top")

    orange, stone = T["palette"][0], T["palette"][1]
    big = 15

    # Left: combined throughput.
    a1 = fig.add_axes((fx(PAD + 265), fy(560), fx(420), 330 / H))
    cb.mpl_panel_style(a1, T, "wide")
    labels = [r[0] for r in THROUGHPUT][::-1]
    vals = [r[1] for r in THROUGHPUT][::-1]
    kinds = [r[2] for r in THROUGHPUT][::-1]
    for i, (v, k) in enumerate(zip(vals, kinds)):
        if k == "reported":
            a1.barh(i, v, height=0.62, color="none", edgecolor=T["muted"], hatch="///", linewidth=1)
        else:
            a1.barh(i, v, height=0.62, color=orange if k == "ours" else stone)
        a1.text(v + 25, i, f"{v:,}", va="center", fontsize=12.5, fontweight="bold",
                color=T["muted"] if k == "reported" else T["text"])
    a1.set_yticks(range(len(labels)), labels, fontsize=11.5)
    for t, k in zip(a1.get_yticklabels(), kinds):
        t.set_color(T["accent_text"] if k == "ours" else T["muted"] if k == "reported" else T["text"])
    a1.set_xlim(0, 1900)
    a1.set_xticks([])
    a1.grid(False)
    for s in a1.spines.values():
        s.set_visible(False)
    a1.set_facecolor(T["bg"])
    fig.text(fx(PAD), fy(185), "Combined throughput", fontsize=big, fontweight="bold", color=T["text"], va="top")
    fig.text(fx(PAD), fy(208), "tok/s, end to end, higher is better; lanes capped at 2,048 tokens",
             fontsize=11, color=T["muted"], va="top")

    # Right: time to a finished page.
    a2 = fig.add_axes((fx(800), fy(560), fx(355), 330 / H))
    xs = range(len(PAGE))
    w = 0.36
    for i, (name, fan, one) in enumerate(PAGE):
        a2.bar(i - w / 2, fan, w, color=orange)
        a2.bar(i + w / 2, one, w, color=stone)
        a2.text(i - w / 2, fan + 0.8, f"{fan:.1f}", ha="center", fontsize=12, fontweight="bold", color=T["text"])
        a2.text(i + w / 2, one + 0.8, f"{one:.1f}", ha="center", fontsize=12, fontweight="bold", color=T["text"])
    a2.set_xticks(list(xs), [p[0] for p in PAGE], fontsize=11.5, color=T["text"])
    a2.set_ylim(0, 38)
    a2.set_yticks([])
    a2.tick_params(axis="x", length=0)
    for s in ("top", "right", "left"):
        a2.spines[s].set_visible(False)
    a2.spines["bottom"].set_color(T["rule"])
    a2.set_facecolor(T["bg"])
    fig.text(fx(800), fy(185), "Time to a finished page", fontsize=big, fontweight="bold", color=T["text"], va="top")
    fig.text(fx(800), fy(208), "seconds, lower is better; lanes end on their own", fontsize=11,
             color=T["muted"], va="top")
    from matplotlib.patches import Rectangle
    for k, (lab, col) in enumerate((("8-lane fan-out", orange), ("one stream", stone))):
        x0 = 800 + k * 150
        fig.patches.append(Rectangle((fx(x0), fy(244)), fx(14), 14 / H, color=col,
                                     transform=fig.transFigure, figure=fig))
        fig.text(fx(x0 + 22), fy(231), lab, fontsize=11.5, color=T["text"], va="top")

    foot = ("One prompt split into 8 sections, sent as 8 concurrent requests and merged; temperature 0, "
            "thinking off, medians of 3 rounds. RTX PRO 6000: the RTX 5090's GB202 chip and 1,792 GB/s, "
            "~10% more SMs. Hatched bars are other people's posts, not measured here.")
    import textwrap
    for k, ln in enumerate(textwrap.wrap(foot, 100)):
        fig.text(fx(PAD), fy(H - 62 + k * 15.75), ln, fontsize=10.5, color=T["muted"], va="top")
    fig.text(1 - fx(PAD), fy(H - 62), "github.com/edurdias/capyctl-recipes", fontsize=10.5,
             color=T["accent_text"], va="top", ha="right", fontweight="bold")
    fig.savefig(path, facecolor=T["bg"])
    plt.close(fig)

def hero(theme: str, path: Path) -> None:
    """One number up front, the way the posts it answers lead with theirs."""
    T = cb.mpl_theme(plt, theme)
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=200)
    fig.patch.set_facecolor(T["bg"])
    fx = lambda x: x / W  # noqa: E731
    fy = lambda y: 1 - y / H  # noqa: E731

    img = Image.open(T["logo"])
    lh = 34
    ax = fig.add_axes((fx(PAD), fy(34 + lh), fx(lh * img.width / img.height), lh / H))
    ax.imshow(img)
    ax.axis("off")

    fig.text(fx(PAD), fy(118), "Qwen3.8-27B, one coding task split into 8 parallel lanes",
             fontsize=19, fontweight="bold", color=T["text"], va="top")
    fig.text(fx(PAD), fy(150), "on vLLM 0.32.0 + DFlash2, through CapyCTL", fontsize=19,
             color=T["muted"], va="top")

    # The number.
    fig.text(fx(PAD - 6), fy(190), "1,566", fontsize=120, fontweight="bold", color=T["accent"], va="top")
    fig.text(fx(PAD + 4), fy(372), "tok/s combined", fontsize=30, fontweight="bold", color=T["text"], va="top")
    fig.text(fx(PAD + 4), fy(418), "one RTX PRO 6000 Blackwell · 16 GiB KV cache · median of 3 rounds",
             fontsize=14, color=T["muted"], va="top")

    # Supporting figures, right column.
    x0 = 760
    stats = [
        ("~238", "tok/s per lane"),
        ("310", "tok/s, one stream"),
        ("849", "tok/s, memory sized for a 32 GB card"),
    ]
    for k, (num, lab) in enumerate(stats):
        yy = 214 + k * 92
        fig.text(fx(x0), fy(yy), num, fontsize=38, fontweight="bold", color=T["text"], va="top")
        fig.text(fx(x0), fy(yy + 54), lab, fontsize=13, color=T["muted"], va="top")
    from matplotlib.lines import Line2D
    fig.lines.append(Line2D([fx(x0 - 34)] * 2, [fy(200), fy(480)], color=T["rule"], linewidth=1.5,
                            transform=fig.transFigure, figure=fig))

    foot = ("Qwen3.8-27B CapyCTL NVFP4, DFlash2 drafts, temperature 0, thinking off; one prompt split "
            "into 8 sections, sent as 8 concurrent requests, lanes capped at 2,048 tokens. RTX PRO 6000: "
            "the RTX 5090's GB202 chip and 1,792 GB/s, ~10% more SMs.")
    import textwrap
    for k, ln in enumerate(textwrap.wrap(foot, 100)):
        fig.text(fx(PAD), fy(H - 62 + k * 15.75), ln, fontsize=10.5, color=T["muted"], va="top")
    fig.text(1 - fx(PAD), fy(H - 62), "github.com/edurdias/capyctl-recipes", fontsize=10.5,
             color=T["accent_text"], va="top", ha="right", fontweight="bold")
    fig.savefig(path, facecolor=T["bg"])
    plt.close(fig)



OUT.mkdir(parents=True, exist_ok=True)
card("light", OUT / "summary-wide.png")
card("dark", OUT / "summary-wide-dark.png")
hero("light", OUT / "hero-wide.png")
hero("dark", OUT / "hero-wide-dark.png")
print("wrote", OUT)
