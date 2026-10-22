"""Plot best-so-far score by round for each memory mode in a bench results folder.

    python scripts/mk33/plot_curves.py bench/long            # writes bench/long/curves.png

Each run is drawn faintly and the mean per mode boldly. A dotted line marks the round at
which context compaction first happened in each run, the point where memory starts to
matter. Needs matplotlib (pip install matplotlib).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def first_compaction_round(traj_path: Path) -> int | None:
    if not traj_path.exists():
        return None
    t = json.loads(traj_path.read_text(encoding="utf-8"))
    comps = t.get("metadata", {}).get("compactions") or []
    if not comps:
        return None
    turn = comps[0]["turn"]
    rounds = [x["round"] for x in t["turns"] if x["index"] <= turn]
    return max(rounds) if rounds else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="bench results folder (the --out of bench-run)")
    ap.add_argument("--save", help="image path (default: <out>/curves.png)")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(a.out)
    by_mode: dict[str, list[dict]] = {}
    for res in sorted(out.glob("runs/*/*/*/result.json")):
        row = json.loads(res.read_text(encoding="utf-8"))
        if row.get("error") or not row.get("curve"):
            continue
        row["_compact"] = first_compaction_round(res.parent / "trajectory.json")
        by_mode.setdefault(row["mode"], []).append(row)
    if not by_mode:
        print(f"no finished runs under {out}", file=sys.stderr)
        return 1

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"rlc": "#3b5bdb", "truncate": "#868e96", "retrieve": "#2f9e44", "rlc+retrieve": "#e8590c"}
    for mode, rows in sorted(by_mode.items()):
        c = colors.get(mode)
        n = max(len(r["curve"]) for r in rows)
        for r in rows:
            ax.plot(range(1, len(r["curve"]) + 1), r["curve"], color=c, alpha=0.25, lw=1)
            if r["_compact"]:
                ax.axvline(r["_compact"], color=c, ls=":", lw=1, alpha=0.6)
        mean = [statistics.mean(r["curve"][i] for r in rows if len(r["curve"]) > i) for i in range(n)]
        line, = ax.plot(range(1, n + 1), mean, color=c, lw=2.5, label=f"{mode} (n={len(rows)})")
        rr = [r["retry_rate"] for r in rows if r.get("retry_rate") is not None]
        if rr:
            line.set_label(f"{mode} (n={len(rows)}, re-try rate {statistics.mean(rr):.2f})")
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.text(1, 1.01, "reference solution", fontsize=8, va="bottom")
    ax.set_xlabel("round")
    ax.set_ylabel("best normalized score so far")
    ax.set_title("Long run: does the agent keep improving after its context fills up?")
    ax.legend(title="dotted: first compaction", fontsize=9, title_fontsize=8)
    fig.tight_layout()
    path = Path(a.save) if a.save else out / "curves.png"
    fig.savefig(path, dpi=200)
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
