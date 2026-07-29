#!/usr/bin/env python3
"""Generate a random directed graph adjacency matrix for an arbitrary N.

Writes data/graph_N{N}.txt in the text format expected by btc::load_graph
(include/btc/graph_io.hpp): first line N, followed by N rows of N
space-separated 0/1 values, row-major, no self-loops.

Usage:
    python3 scripts/gen_graph.py N [--density 0.2] [--seed 2024] [--out-dir data]
"""
import argparse
import random
from pathlib import Path


def gen_graph(n: int, density: float, seed: int):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        row = [0] * n
        for j in range(n):
            if i != j and rng.random() < density:
                row[j] = 1
        rows.append(row)
    return rows


def write_graph(rows, path: Path):
    n = len(rows)
    with open(path, "w") as f:
        f.write(f"{n}\n")
        for row in rows:
            f.write(" ".join(str(v) for v in row) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("n", type=int, help="number of graph nodes N")
    ap.add_argument("--density", type=float, default=0.2, help="edge probability (default 0.2)")
    ap.add_argument("--seed", type=int, default=2024, help="RNG seed (default 2024, matches gen_graphs.cpp)")
    ap.add_argument("--out-dir", default="data", help="output directory (default data)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"graph_N{args.n}.txt"

    rows = gen_graph(args.n, args.density, args.seed)
    write_graph(rows, path)
    print(f"Wrote {path} (N={args.n}, density={args.density}, seed={args.seed})")


if __name__ == "__main__":
    main()
