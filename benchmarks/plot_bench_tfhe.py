#!/usr/bin/env python3
"""Plot FHE computation time vs. graph size from bench_tfhe's CSV log.

Usage:
    python3 plot_bench_tfhe.py [data/bench_tfhe_results.csv] [output.png]
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "graph": row["graph"],
                "N": int(row["N"]),
                "T": int(row["T"]),
                "fhe_ms": float(row["fhe_ms"]),
                "ok": row["matches_plaintext"] == "YES",
            })
    return rows


def plot(rows, out_path):
    rows = sorted(rows, key=lambda r: r["N"])
    xs = [r["N"] for r in rows]
    ys_s = [r["fhe_ms"] / 1000.0 for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))

    line_color = "#3b6fa0"
    fail_color = "#c0392b"

    ax.plot(xs, ys_s, color=line_color, linewidth=2, zorder=2)
    ok_mask = [r["ok"] for r in rows]
    ax.scatter(
        [x for x, ok in zip(xs, ok_mask) if ok],
        [y for y, ok in zip(ys_s, ok_mask) if ok],
        s=70, color=line_color, edgecolors="white", linewidths=1.5, zorder=3,
    )
    if not all(ok_mask):
        ax.scatter(
            [x for x, ok in zip(xs, ok_mask) if not ok],
            [y for y, ok in zip(ys_s, ok_mask) if not ok],
            s=90, color=fail_color, marker="x", linewidths=2.5, zorder=4,
            label="mismatch vs. plaintext",
        )

    # Direct labels: seconds, next to each point.
    for x, y in zip(xs, ys_s):
        label = f"{y:.1f}s"
        ax.annotate(
            label, (x, y), xytext=(8, 8), textcoords="offset points",
            fontsize=9, color="#333333",
        )

    ax.set_xticks(xs)
    ax.set_xlabel("Graph size N (nodes)")
    ax.set_ylabel("FHE computation time (s)")
    ax.set_title("CPU TFHE bounded transitive closure — time vs. N")

    ax.grid(True, which="major", axis="both", linewidth=0.5, color="#dddddd", zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    if not all(ok_mask):
        ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/bench_tfhe_results.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else str(
        Path(csv_path).with_name("bench_tfhe_plot.png")
    )

    rows = load_results(csv_path)
    if not rows:
        print(f"No rows found in {csv_path}")
        sys.exit(1)

    plot(rows, out_path)


if __name__ == "__main__":
    main()
