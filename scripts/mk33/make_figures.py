"""Figures for the Mk33 memory study, built from the raw result files in the repository.

    pip install matplotlib
    python assets/make_assets.py --fetch-fonts     # once, for the brand typefaces (optional)
    python scripts/mk33/make_figures.py

Reads docs/mk33/results/ and writes PNGs to docs/mk33/figures/ in the project's visual
style (the dark "summit" theme of the logo and banner). Every results folder with a
``rows.jsonl`` from ``memory_quality.py`` is picked up, so adding a model is a matter of
adding its folder. With three or more models an extra cross-model chart is written.
Pass a results directory and an output directory as two arguments to plot other runs.
"""

import json
import math
import random
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/mk33/results"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "docs/mk33/figures"
KERNELS = next(iter(sorted(RES.glob("*/kv_kernels.json"))), None)
OUT.mkdir(parents=True, exist_ok=True)

# Summit theme (assets/make_assets.py). Series colors are the reference dark categorical
# palette, validated on this background: adjacent CVD dE >= 8.4, normal-vision >= 19.3,
# all >= 3:1 contrast. Text methods are also dashed (secondary encoding).
BG, PANEL, INK, INK2, MUTED = "#0c0e12", "#0c0e12", "#f4f2ed", "#a3abb8", "#6d7584"
GRID, EDGE, ACCENT = "#1d222b", "#2b3240", "#f0a43a"
METHODS = [
    ("kv:h2o", "H2O (KV)", "#3987e5", "-"),
    ("kv:keynorm", "Key-norm (KV)", "#d95926", "-"),
    ("kv:random", "Random (KV)", "#199e70", "-"),
    ("kv:streaming", "Streaming (KV)", "#c98500", "-"),
    ("text:recent", "Truncation (text)", "#d55181", "--"),
    ("text:retrieve", "RAG / BM25 (text)", "#008300", "--"),
]
COLOR = {m: c for m, _, c, _ in METHODS}
LABEL = {m: lab for m, lab, _, _ in METHODS}
BUDGETS = [1024, 2048, 4096]
BUDGET_COLORS = {1024: "#7a4e12", 2048: "#c47f22", 4096: "#f5c26b"}  # amber ordinal, validated
PYTORCH, TRITON = "#3987e5", "#c98500"  # validated pair

FONT_DIR = ROOT / "assets" / ".fonts"
for f in FONT_DIR.glob("*.ttf"):
    font_manager.fontManager.addfont(f)
_names = {font_manager.FontProperties(fname=f).get_name() for f in FONT_DIR.glob("*.ttf")}
# the npm builds name every Space Grotesk weight "Space Grotesk Light"; match on the prefix
SANS = next((n for n in sorted(_names) if n.startswith("Space Grotesk")), "DejaVu Sans")
MONO = next((n for n in sorted(_names) if n.startswith("JetBrains Mono")), "DejaVu Sans Mono")

plt.rcParams.update({
    "font.family": SANS, "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": EDGE, "axes.facecolor": PANEL, "figure.facecolor": BG,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 11.5, "axes.titlecolor": INK,
    "legend.frameon": False, "legend.labelcolor": INK2, "savefig.dpi": 200, "savefig.bbox": "tight",
    "savefig.facecolor": BG, "xtick.major.size": 0, "ytick.major.size": 0,
})


def mono_ticks(ax):
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontfamily(MONO)
        t.set_fontsize(8.5)


def title(fig, text, sub):
    fig.text(0.0, 1.075, "MK33  ·  MEMORY STUDY", fontsize=8.5, color=ACCENT, fontfamily=MONO,
             ha="left", va="bottom")
    fig.text(0.0, 1.02, text, fontsize=14, fontweight="bold", color=INK, ha="left", va="bottom")
    fig.text(0.0, 0.985, sub, fontsize=9.5, color=INK2, ha="left", va="bottom")


def boot(xs, n=5000, seed=0):
    rng = random.Random(seed)
    m = sorted(statistics.mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n) - 1]


def short_name(model: str) -> str:
    return model.split("/")[-1].replace("-Instruct", "").replace("-instruct", "")


def load(folder: Path):
    rows = [json.loads(line) for line in (folder / "rows.jsonl").read_text().splitlines()]
    val, secs = {}, {}
    for r in rows:
        val.setdefault((r["method"], r["budget"], r["window"]), []).append(r["nll"])
        secs.setdefault(r["method"], []).append(r["seconds"])
    val = {k: statistics.mean(v) for k, v in val.items()}
    windows = sorted({w for (m, _, w) in val if m == "full"})
    full = {w: val[("full", 0, w)] for w in windows}
    delta = {(m, b): [val[(m, b, w)] - full[w] for w in windows]
             for (m, *_) in METHODS for b in BUDGETS if all((m, b, w) in val for w in windows)}
    model = json.loads((folder / "summary.json").read_text())["model"]
    return short_name(model), delta, {m: statistics.mean(s) for m, s in secs.items()}, len(windows)


DATA = [load(f) for f in sorted(p.parent for p in RES.glob("*/rows.jsonl"))]
ORDER = ["Qwen2.5-3B", "Llama-3.2-3B"]  # the original two first, then the rest by name
DATA.sort(key=lambda d: (ORDER.index(d[0]) if d[0] in ORDER else len(ORDER), d[0]))
N = min(d[3] for d in DATA)
NM = len(DATA)


def grid(n, w=5, h=4.2, sharey=True):
    cols = min(n, 3 if n > 4 else 2)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(w * cols, h * rows), sharey=sharey, squeeze=False)
    flat = axes.ravel().tolist()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n], rows, cols


# 1. Loss increase vs budget --------------------------------------------------------
ymax = max(statistics.mean(v) for _, d, _, _ in DATA for v in d.values()) * 1.08
fig, axes, rows, cols = grid(NM)
for i, (ax, (name, delta, _, _)) in enumerate(zip(axes, DATA)):
    for m, label, color, ls in METHODS:
        if (m, BUDGETS[0]) not in delta:
            continue
        ys = [statistics.mean(delta[(m, b)]) for b in BUDGETS]
        ax.plot(BUDGETS, ys, ls, color=color, lw=2.2, marker="o", ms=6, mec=BG, mew=1.6, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(BUDGETS, ["1K", "2K", "4K"])
    ax.minorticks_off()
    ax.set_title(name, loc="left")
    ax.set_ylim(0, ymax)
    if i // cols == rows - 1:
        ax.set_xlabel("Context tokens kept (of 8,192)")
    if i % cols == 0:
        ax.set_ylabel("Loss increase over full cache (nats)")
    mono_ticks(ax)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=6 if cols > 1 else 2, fontsize=9, handlelength=2.4,
           bbox_to_anchor=(0.5, -0.1 / rows))
title(fig, "Cost of forgetting, by method and memory budget",
      f"Mean over {N} windows; lower is better. Solid = KV-cache eviction, dashed = text-level (truncation, RAG).")
fig.savefig(OUT / "loss_vs_budget.png")
plt.close(fig)

# 2. Paired difference vs random, with 95% CI -----------------------------------------
others = [mm for mm in METHODS if mm[0] != "kv:random"]
fig, axes, rows, cols = grid(NM, h=4.4)
for i, (ax, (name, delta, _, _)) in enumerate(zip(axes, DATA)):
    for k, (m, label, _, _) in enumerate(others):
        for j, b in enumerate(BUDGETS):
            if (m, b) not in delta:
                continue
            d = [a - r for a, r in zip(delta[(m, b)], delta[("kv:random", b)])]
            lo, hi = boot(d)
            y = k + (j - 1) * 0.22
            ax.plot([lo, hi], [y, y], color=BUDGET_COLORS[b], lw=2.2, solid_capstyle="round")
            ax.plot(statistics.mean(d), y, "o", color=BUDGET_COLORS[b], ms=6.5, mec=BG, mew=1.6,
                    label=f"{b // 1024}K tokens kept" if k == 0 else None)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(others)), [lab for _, lab, _, _ in others])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title(name, loc="left")
    ax.set_xlabel("Difference from random eviction (nats)")
    ax.text(0.02, -0.17, "← better than random", transform=ax.transAxes, fontsize=8.5, color=MUTED,
            fontfamily=[MONO, "DejaVu Sans Mono"])
    ax.text(0.98, -0.17, "worse than random →", transform=ax.transAxes, fontsize=8.5, color=MUTED,
            fontfamily=[MONO, "DejaVu Sans Mono"], ha="right")
    for t in ax.get_xticklabels():
        t.set_fontfamily(MONO)
        t.set_fontsize(8.5)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.12 / rows))
title(fig, "Which methods beat random eviction?",
      f"Paired difference per window, mean and bootstrap 95% interval over {N} windows. "
      "An interval clear of zero is a real difference.")
fig.savefig(OUT / "vs_random.png")
plt.close(fig)

# 3. Quality vs compute at 2K -----------------------------------------------------------
y2 = max(statistics.mean(d[(m, 2048)]) for _, d, _, _ in DATA for m, *_ in METHODS if (m, 2048) in d) * 1.12
xmax = max(s for _, _, sec, _ in DATA for s in sec.values()) * 1.45
OFFSETS = {"text:retrieve": ((0, 11), "center"), "text:recent": ((0, -12), "center")}
fig, axes, rows, cols = grid(NM)
for i, (ax, (name, delta, secs, _)) in enumerate(zip(axes, DATA)):
    pts = []
    for m, label, color, _ in METHODS:
        if (m, 2048) not in delta:
            continue
        x, y = secs[m], statistics.mean(delta[(m, 2048)])
        ax.plot(x, y, "o", color=color, ms=9, mec=BG, mew=2)
        pts.append((m, label, x, y))
    # KV labels to the right of their points; within a cluster, stack them with a minimum gap
    gap = 0.055 * y2
    kv = sorted([p for p in pts if p[0].startswith("kv")], key=lambda p: p[3])
    ly = {}
    for j, (m, label, x, y) in enumerate(kv):
        near = [ly[q[0]] for q in kv[:j] if abs(q[2] - x) < 0.15 * xmax]
        ly[m] = max([y] + [v + gap for v in near])
    shift = {}  # center each stacked cluster on its points
    for m, label, x, y in kv:
        shift[m] = ly[m] - y
    for m, label, x, y in pts:
        if m in OFFSETS:
            off, ha = OFFSETS[m]
            ax.annotate(label, (x, y), xytext=off, textcoords="offset points", fontsize=8.5, color=INK2,
                        va="center", ha=ha)
        else:
            ax.annotate(label, (x, y), xytext=(x + 0.035 * xmax, ly[m]), textcoords="data", fontsize=8.5,
                        color=INK2, va="center", ha="left",
                        arrowprops=dict(arrowstyle="-", color=EDGE, lw=0.8, shrinkA=0, shrinkB=5)
                        if abs(shift[m]) > 0.2 * gap else None)
    ax.set_title(name, loc="left")
    if i // cols == rows - 1:
        ax.set_xlabel("Seconds per measurement (T4)")
    if i % cols == 0:
        ax.set_ylabel("Loss increase at 2K tokens kept (nats)")
    ax.set_xlim(-0.12 * xmax, xmax)  # room for the centered text-method labels left of the points
    ax.set_xticks([t for t in ax.get_xticks() if 0 <= t <= xmax])
    ax.set_ylim(0, y2)
    mono_ticks(ax)
title(fig, "Quality vs compute at equal memory",
      "Text methods read only the kept tokens; KV methods read all 8,192 first. H2O uses eager attention.")
fig.savefig(OUT / "quality_vs_compute.png")
plt.close(fig)

# 4. Kernel bandwidth -------------------------------------------------------------------
if KERNELS is not None:
    k = json.loads(KERNELS.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, op, label in zip(axes, ["gather", "key_norm"], ["Fused K/V gather", "Key L2 norms"]):
        rows_ = [r for r in k["rows"] if r["op"] == op]
        xs = range(len(rows_))
        w = 0.36
        for off, field, name, color in ((-w / 2 - 0.01, "torch_gbs", "PyTorch", PYTORCH),
                                        (w / 2 + 0.01, "triton_gbs", "Triton (Mk33)", TRITON)):
            vals = [r[field] for r in rows_]
            bars = ax.bar([x + off for x in xs], vals, width=w, color=color, label=name)
            for bb, v in zip(bars, vals):
                ax.text(bb.get_x() + bb.get_width() / 2, v + 4, f"{v:.0f}", ha="center", fontsize=8,
                        color=INK2, fontfamily=MONO)
        ax.axhline(320, color=ACCENT, lw=1, ls=(0, (4, 3)), alpha=0.8)
        ax.text(len(rows_) - 0.5, 324, "T4 peak 320 GB/s", ha="right", va="bottom", fontsize=8.5,
                color=ACCENT, fontfamily=MONO)
        ax.set_xticks(list(xs), [f"{r['tokens'] // 1024}K" for r in rows_])
        ax.set_xlabel("Cached tokens per layer")
        ax.set_title(label, loc="left")
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 350)
        mono_ticks(ax)
    axes[0].set_ylabel("Achieved memory bandwidth (GB/s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.12))
    title(fig, f"KV eviction kernels: PyTorch vs Triton on a {k.get('gpu', 'GPU')}",
          "float16, 8 KV heads, head dim 128, keeping 25% of rows. Higher is better.")
    fig.savefig(OUT / "kernel_bandwidth.png")
    plt.close(fig)

# 5. Across models (only with 3 or more) --------------------------------------------------
if NM >= 3:
    show = ["kv:h2o", "kv:random", "text:recent", "text:retrieve"]
    budget = 2048
    fig, ax = plt.subplots(figsize=(10, 0.55 * NM + 1.6))
    names = [d[0] for d in DATA]
    for row, (name, delta, _, _) in enumerate(DATA):
        vals = [(m, statistics.mean(delta[(m, budget)])) for m in show if (m, budget) in delta]
        lo_v, hi_v = min(v for _, v in vals), max(v for _, v in vals)
        ax.plot([lo_v, hi_v], [row, row], color=EDGE, lw=2, zorder=1)
        for m, v in vals:
            ax.plot(v, row, "o", color=COLOR[m], ms=9, mec=BG, mew=1.8, zorder=2,
                    label=LABEL[m] if row == 0 else None)
    ax.set_yticks(range(NM), names)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(left=0)
    ax.set_xlabel(f"Loss increase over full cache at {budget // 1024}K tokens kept (nats)")
    mono_ticks(ax)
    for t in ax.get_yticklabels():
        t.set_fontfamily(SANS)
        t.set_fontsize(10)
        t.set_color(INK)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.06 - 0.25 / NM))
    title(fig, "How each model handles forgetting",
          f"Same corpus, windows and budget for every model; mean over {N} windows. Lower is better.")
    fig.savefig(OUT / "models.png")
    plt.close(fig)

print("wrote", sorted(p.name for p in OUT.glob("*.png")))
