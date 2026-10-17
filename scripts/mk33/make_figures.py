"""Figures for the Mk33 memory study, built from the raw result files in the repository.

    pip install matplotlib
    python scripts/mk33/make_figures.py

Reads docs/mk33/results/ and writes four PNGs to docs/mk33/figures/. Pass a different
results directory and output directory as two arguments to plot new runs.
"""

import json
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
MEMQ = {"qwen": "t4-memq-qwen2.5-3b", "llama": "t4-memq-llama3.2-3b"}
KERNELS = RES / "t4-qwen2.5-3b" / "kv_kernels.json"
OUT.mkdir(parents=True, exist_ok=True)

# Reference palette (validated: adjacent CVD dE >= 9.1, normal-vision >= 19.6).
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
METHODS = [  # fixed order = fixed color; text methods also dashed (secondary encoding)
    ("kv:h2o", "H2O (KV)", "#2a78d6", "-"),
    ("kv:keynorm", "Key-norm (KV)", "#eb6834", "-"),
    ("kv:random", "Random (KV)", "#1baf7a", "-"),
    ("kv:streaming", "Streaming (KV)", "#eda100", "-"),
    ("text:recent", "Truncation (text)", "#e87ba4", "--"),
    ("text:retrieve", "RAG / BM25 (text)", "#008300", "--"),
]
BUDGETS = [1024, 2048, 4096]
BUDGET_COLORS = {1024: "#86b6ef", 2048: "#3987e5", 4096: "#184f95"}  # ordinal blue
MODELS = [("qwen", "Qwen2.5-3B"), ("llama", "Llama-3.2-3B")]

_inter = [f for f in font_manager.findSystemFonts() if "Inter" in Path(f).name]
for f in _inter:
    font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": "Inter" if _inter else "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": GRID, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "semibold", "axes.titlesize": 11, "axes.titlecolor": INK,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def boot(xs, n=5000, seed=0):
    rng = random.Random(seed)
    m = sorted(statistics.mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n) - 1]


def load(model):
    rows = [json.loads(line) for line in (RES / MEMQ[model] / "rows.jsonl").read_text().splitlines()]
    val, secs = {}, {}
    for r in rows:
        val.setdefault((r["method"], r["budget"], r["window"]), []).append(r["nll"])
        secs.setdefault(r["method"], []).append(r["seconds"])
    val = {k: statistics.mean(v) for k, v in val.items()}
    windows = sorted({w for (_, _, w) in val})
    full = {w: val[("full", 0, w)] for w in windows}
    delta = {(m, b): [val[(m, b, w)] - full[w] for w in windows] for (m, *_ ) in METHODS for b in BUDGETS}
    return delta, {m: statistics.mean(s) for m, s in secs.items()}, len(windows)


DATA = {m: load(m) for m, _ in MODELS}
N = DATA["qwen"][2]


def title(fig, text, sub):
    fig.text(0.0, 1.02, text, fontsize=13, fontweight="semibold", color=INK, ha="left", va="bottom")
    fig.text(0.0, 0.985, sub, fontsize=9.5, color=INK2, ha="left", va="bottom")


# 1. Loss increase vs budget --------------------------------------------------------
YMAX = max(statistics.mean(v) for key, _ in MODELS for v in DATA[key][0].values()) * 1.08
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, (key, name) in zip(axes, MODELS):
    delta = DATA[key][0]
    for m, label, color, ls in METHODS:
        ys = [statistics.mean(delta[(m, b)]) for b in BUDGETS]
        ax.plot(BUDGETS, ys, ls, color=color, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(BUDGETS, ["1K", "2K", "4K"])
    ax.minorticks_off()
    ax.set_title(name, loc="left")
    ax.set_xlabel("Context tokens kept (of 8,192)")
    ax.set_ylim(0, YMAX)
axes[0].set_ylabel("Loss increase over full cache (nats)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9, handlelength=2.2,
           bbox_to_anchor=(0.5, -0.1))
title(fig, "Cost of forgetting, by method and memory budget",
      f"Mean over {N} windows; lower is better. Solid = KV-cache eviction, dashed = text-level (truncation, RAG).")
fig.savefig(OUT / "loss_vs_budget.png")
plt.close(fig)

# 2. Paired difference vs random, with 95% CI -----------------------------------------
others = [mm for mm in METHODS if mm[0] != "kv:random"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
for ax, (key, name) in zip(axes, MODELS):
    delta = DATA[key][0]
    for i, (m, label, _, _) in enumerate(others):
        for j, b in enumerate(BUDGETS):
            d = [a - r for a, r in zip(delta[(m, b)], delta[("kv:random", b)])]
            lo, hi = boot(d)
            y = i + (j - 1) * 0.22
            ax.plot([lo, hi], [y, y], color=BUDGET_COLORS[b], lw=2, solid_capstyle="round")
            ax.plot(statistics.mean(d), y, "o", color=BUDGET_COLORS[b], ms=6.5, mec=SURFACE, mew=1.5,
                    label=f"{b // 1024}K tokens kept" if i == 0 else None)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(others)), [lab for _, lab, _, _ in others])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title(name, loc="left")
    ax.set_xlabel("Difference from random eviction (nats)")
    ax.text(0.02, -0.17, "← better than random", transform=ax.transAxes, fontsize=8.5, color=MUTED)
    ax.text(0.98, -0.17, "worse than random →", transform=ax.transAxes, fontsize=8.5, color=MUTED, ha="right")
axes[1].legend(loc="lower right", fontsize=9)
title(fig, "Which methods beat random eviction?",
      f"Paired difference per window, mean and bootstrap 95% interval over {N} windows. "
      "An interval clear of zero is a real difference.")
fig.savefig(OUT / "vs_random.png")
plt.close(fig)

# 3. Quality vs compute at 2K -----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
KV_LABEL = ((9, 0), "left")  # KV points: label to the right
OFFSETS = {"text:retrieve": ((0, 11), "center"), "text:recent": ((0, -12), "center")}  # above / below
Y2 = max(statistics.mean(DATA[k][0][(m, 2048)]) for k, _ in MODELS for m, *_ in METHODS) * 1.12
for ax, (key, name) in zip(axes, MODELS):
    delta, secs, _ = DATA[key]
    for m, label, color, _ in METHODS:
        x, y = secs[m], statistics.mean(delta[(m, 2048)])
        ax.plot(x, y, "o", color=color, ms=9, mec=SURFACE, mew=2)
        off, ha = OFFSETS.get(m, KV_LABEL)
        if key == "llama" and m in ("kv:streaming", "kv:random"):  # 0.014 apart: spread the labels
            off = (9, 6) if m == "kv:streaming" else (9, -6)
        ax.annotate(label, (x, y), xytext=off, textcoords="offset points", fontsize=8.5,
                    color=INK2, va="center", ha=ha)
    ax.set_title(name, loc="left")
    ax.set_xlabel("Seconds per measurement (T4)")
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, Y2)
axes[0].set_ylabel("Loss increase at 2K tokens kept (nats)")
title(fig, "Quality vs compute at equal memory",
      "Text methods read only the kept tokens; KV methods read all 8,192 first. H2O uses eager attention.")
fig.savefig(OUT / "quality_vs_compute.png")
plt.close(fig)

# 4. Kernel bandwidth -------------------------------------------------------------------
k = json.loads(KERNELS.read_text())
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, op, label in zip(axes, ["gather", "key_norm"], ["Fused K/V gather", "Key L2 norms"]):
    rows = [r for r in k["rows"] if r["op"] == op]
    xs = range(len(rows))
    w = 0.36
    for off, field, name, color in ((-w / 2 - 0.01, "torch_gbs", "PyTorch", "#2a78d6"),
                                    (w / 2 + 0.01, "triton_gbs", "Triton (Mk33)", "#eb6834")):
        vals = [r[field] for r in rows]
        bars = ax.bar([x + off for x in xs], vals, width=w, color=color, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 4, f"{v:.0f}", ha="center", fontsize=8, color=INK2)
    ax.axhline(320, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(len(rows) - 0.5, 324, "T4 peak 320 GB/s", ha="right", va="bottom", fontsize=8.5, color=MUTED)
    ax.set_xticks(list(xs), [f"{r['tokens'] // 1024}K" for r in rows])
    ax.set_xlabel("Cached tokens per layer")
    ax.set_title(label, loc="left")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 350)
axes[0].set_ylabel("Achieved memory bandwidth (GB/s)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.12))
title(fig, "KV eviction kernels: PyTorch vs Triton on a Tesla T4",
      "float16, 8 KV heads, head dim 128, keeping 25% of rows. Higher is better.")
fig.savefig(OUT / "kernel_bandwidth.png")
plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
