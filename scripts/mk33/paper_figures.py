"""Figures for the paper (light theme, vector PDF) from saved results.

    python scripts/mk33/paper_figures.py            # writes paper/figures/*.pdf

Memory study: docs/mk33/results/t4-memq-*/ (summary.json and rows.jsonl).
Long runs: docs/mk33/results/long-runs/ (result.json per run; summary.json from
scripts/mk33/long_run_stats.py, which must run first).
The website and README figures come from scripts/mk33/make_figures.py instead.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs/mk33/results"
OUT = ROOT / "paper/figures"

MODELS = [("t4-memq-qwen2.5-3b", "Qwen2.5-3B-Instruct"), ("t4-memq-llama3.2-3b", "Llama-3.2-3B-Instruct")]
# Okabe-Ito palette (color-blind safe). Solid = KV-cache eviction, dashed = text-level.
METHODS = [
    ("kv:h2o", "H2O (KV)", "#0072B2", "-", "o"),
    ("kv:keynorm", "Key-norm (KV)", "#D55E00", "-", "s"),
    ("kv:random", "Random (KV)", "#009E73", "-", "^"),
    ("kv:streaming", "Streaming (KV)", "#E69F00", "-", "v"),
    ("text:recent", "Truncation (text)", "#CC79A7", "--", "D"),
    ("text:retrieve", "BM25 retrieval (text)", "#56B4E9", "--", "P"),
]


def style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "pdf.fonttype": 42,
    })
    return plt


def loss_vs_budget(plt):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    for ax, (folder, name) in zip(axes, MODELS):
        s = json.loads((RESULTS / folder / "summary.json").read_text(encoding="utf-8"))
        for i, (key, label, color, ls, marker) in enumerate(METHODS):
            cells = sorted((c for c in s["cells"] if c["method"] == key), key=lambda c: c["budget"])
            xs = [c["budget"] for c in cells]
            # Small horizontal offsets keep the intervals readable.
            off = [x * (1 + 0.025 * (i - 2.5)) for x in xs]
            ys = [c["vs_full"] for c in cells]
            err = [[c["vs_full"] - c["vs_full_ci"][0] for c in cells], [c["vs_full_ci"][1] - c["vs_full"] for c in cells]]
            ax.errorbar(off, ys, yerr=err, color=color, ls=ls, marker=marker, ms=3.5, lw=1.2, elinewidth=0.7,
                        capsize=1.5, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1024, 2048, 4096], ["1K", "2K", "4K"])
        ax.minorticks_off()
        ax.set_title(name)
        ax.set_xlabel("context tokens kept (of 8,192)")
    axes[0].set_ylabel("loss increase (nats)")
    axes[0].set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(OUT / "loss_vs_budget.pdf")
    plt.close(fig)


def quality_vs_compute(plt, budget=2048):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    for ax, (folder, name) in zip(axes, MODELS):
        s = json.loads((RESULTS / folder / "summary.json").read_text(encoding="utf-8"))
        rows = [json.loads(l) for l in (RESULTS / folder / "rows.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip()]
        for key, label, color, _, marker in METHODS:
            secs = [r["seconds"] for r in rows if r["method"] == key and r["budget"] == budget]
            cell = next(c for c in s["cells"] if c["method"] == key and c["budget"] == budget)
            ax.scatter(statistics.mean(secs), cell["vs_full"], color=color, marker=marker, s=26, zorder=3,
                       label=label)
        ax.set_title(name)
        ax.set_xlabel("seconds per measurement (T4)")
        ax.set_xlim(0, 11)
    axes[0].set_ylabel(f"loss increase, {budget // 1024}K kept (nats)")
    axes[0].set_ylim(0, 0.42)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(OUT / "quality_vs_compute.pdf")
    plt.close(fig)


def long_runs(plt):
    base = RESULTS / "long-runs"
    summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))
    first = {(r["run"], r["mode"]): r["first_compaction_round"] for r in summary["rows"]}
    runs = sorted({r["run"] for r in summary["rows"]})
    fig, axes = plt.subplots(1, len(runs), figsize=(6.6, 1.9), sharey=True)
    styles = {"truncate": ("#CC79A7", "--", "truncate"), "rlc": ("#0072B2", "-", "rlc")}
    for ax, n in zip(axes, runs):
        for mode, (color, ls, label) in styles.items():
            res = json.loads((base / f"run{n}" / mode / "result.json").read_text(encoding="utf-8"))
            # The saved curve is padded to the round budget; draw only the rounds played.
            curve = [max(v, 0.0) for v in (res.get("curve") or [])[: res["rounds"]]]
            ax.step(range(1, len(curve) + 1), curve, where="post", color=color, ls=ls, lw=1.3, label=label)
            if curve and res["stop"] != "round_budget":  # run ended early (time budget)
                ax.plot(len(curve), curve[-1], marker="x", color=color, ms=4, mew=1.1)
            fc = first.get((n, mode))
            if fc:
                ax.axvline(fc, color=color, ls=":", lw=0.8)
        ax.set_title(f"run {n}" + (" (confounded)" if n == 4 else ""), fontsize=8.5)
        ax.set_xlim(0, 41)
        ax.set_xticks([1, 20, 40])
        ax.set_xlabel("round")
    axes[0].set_ylabel("best normalized score")
    axes[0].set_ylim(-0.05, 1.7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "long_runs.pdf")
    fig.savefig(base / "long_runs.png", dpi=200)  # for the README
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    plt = style()
    if "--long-only" not in sys.argv:
        loss_vs_budget(plt)
        quality_vs_compute(plt)
    if (RESULTS / "long-runs" / "summary.json").exists():
        long_runs(plt)
    print(f"wrote figures to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
