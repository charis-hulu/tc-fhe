#!/usr/bin/env python3
"""Visualize graphs stored as text adjacency matrices (see btc::save_graph).

File format:
    line 1: N
    lines 2..N+1: N space-separated 0/1 values (row-major adjacency matrix)

Usage:
    python3 visualize_graph.py data/graph_N8.txt [output.png]
    python3 visualize_graph.py data/*.txt   # batch mode, writes <name>.png next to each input
"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_graph(path):
    with open(path) as f:
        n = int(f.readline())
        adj = np.zeros((n, n), dtype=int)
        for i in range(n):
            row = f.readline().split()
            for j in range(n):
                adj[i, j] = int(row[j])
    return adj


def draw_graph(adj, title, out_path):
    n = adj.shape[0]
    fig, ax = plt.subplots(figsize=(6, 6))

    # Circular layout.
    angles = [2 * math.pi * i / n - math.pi / 2 for i in range(n)]
    pos = {i: (math.cos(a), math.sin(a)) for i, a in enumerate(angles)}

    # Edges (directed, with arrows). Self-loops drawn as small loops.
    for i in range(n):
        for j in range(n):
            if not adj[i, j]:
                continue
            if i == j:
                x, y = pos[i]
                loop_r = 0.09
                cx, cy = x * 1.15, y * 1.15
                circle = plt.Circle((cx, cy), loop_r, fill=False,
                                     color="#3b6fa0", linewidth=1.5, zorder=1)
                ax.add_patch(circle)
                continue

            x0, y0 = pos[i]
            x1, y1 = pos[j]
            dx, dy = x1 - x0, y1 - y0
            dist = math.hypot(dx, dy)
            if dist == 0:
                continue
            shrink = 0.13 / dist
            sx0, sy0 = x0 + dx * shrink, y0 + dy * shrink
            sx1, sy1 = x1 - dx * shrink, y1 - dy * shrink
            ax.annotate(
                "", xy=(sx1, sy1), xytext=(sx0, sy0),
                arrowprops=dict(arrowstyle="-|>", color="#3b6fa0",
                                 lw=1.5, shrinkA=0, shrinkB=0,
                                 mutation_scale=14, alpha=0.85),
                zorder=1,
            )

    # Nodes on top of edges.
    xs = [pos[i][0] for i in range(n)]
    ys = [pos[i][1] for i in range(n)]
    ax.scatter(xs, ys, s=650, color="#f2a93b", edgecolors="#8a5a1a",
               linewidths=1.5, zorder=2)
    for i in range(n):
        ax.annotate(str(i), pos[i], ha="center", va="center",
                     fontsize=12, fontweight="bold", color="#3a2a10", zorder=3)

    ax.set_title(title)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    inputs = sys.argv[1:]
    # If exactly one input and a second positional arg looks like an output path.
    explicit_out = None
    if len(inputs) == 2 and not inputs[1].endswith(".txt"):
        explicit_out = inputs[1]
        inputs = inputs[:1]

    for in_path in inputs:
        in_path = Path(in_path)
        adj = load_graph(in_path)
        out_path = explicit_out if explicit_out else str(in_path.with_suffix(".png"))
        draw_graph(adj, f"{in_path.stem} (N={adj.shape[0]})", out_path)


if __name__ == "__main__":
    main()
