"""
Build SRBench ground-truth taxonomy CSV (133 Feynman + Strogatz).

No model required — parses PMLB metadata.yaml for each problem.

Run:
    python3 build_srbench_taxonomy.py
    python3 build_srbench_taxonomy.py --download   # fetch missing datasets first

Output:
    results/srbench/taxonomy.csv
    results/srbench/taxonomy_summary.txt
"""

from __future__ import annotations

import argparse
import csv
import os

from results_paths import SRBENCH, ensure_dirs
from srbench_data import (
    download_all,
    load_metadata,
    problem_list,
    benchmark_name,
    extract_formula,
    categorize_formula,
)

DESTROI_OPS = {"add", "mul", "inv", "sqrt", "log", "sin"}


def destroi_overlap(ops_str: str) -> str:
    if not ops_str:
        return ""
    present = set(ops_str.split(","))
    mapped = set()
    for op in present:
        if op in DESTROI_OPS:
            mapped.add(op)
        elif op in ("sub", "div", "pow", "exp", "cos", "tan"):
            pass  # not in DeSTrOI 6-op vocab
    return ",".join(sorted(mapped))


def destroi_compatible(n_features: int, ops_str: str) -> bool:
    """Heuristic: k<=10 and uses only DeSTrOI-detectable operator families."""
    if n_features > 10:
        return False
    overlap = destroi_overlap(ops_str)
    present = set(ops_str.split(",")) if ops_str else set()
    alien = present - {"add", "mul", "sub", "div", "pow", "inv", "sqrt", "log", "sin", "cos", "tan", "exp", "abs"}
    return len(alien) == 0


def build(download: bool = False) -> list[dict]:
    problems = problem_list()
    if download:
        print(f"Downloading {len(problems)} datasets…")
        download_all(problems)

    rows = []
    for i, problem in enumerate(problems):
        meta = load_metadata(problem)
        formula = extract_formula(meta)
        features = meta.get("features") or []
        n_features = len(features)
        feature_names = ",".join(f.get("name", "") for f in features)
        bench = benchmark_name(problem)
        tags = categorize_formula(formula, n_features, bench)
        rows.append({
            "id": i,
            "problem": problem,
            "benchmark": bench,
            "gt_formula": formula,
            "feature_names": feature_names,
            "destroi_compatible": int(destroi_compatible(n_features, tags["ops"])),
            "destroi_ops_overlap": destroi_overlap(tags["ops"]),
            **tags,
        })
        print(f"  [{i+1:3d}/{len(problems)}] {problem}", flush=True)
    return rows


def summarize(rows: list[dict]) -> str:
    n = len(rows)
    fey = sum(r["benchmark"] == "feynman" for r in rows)
    stro = sum(r["benchmark"] == "strogatz" for r in rows)
    compat = sum(r["destroi_compatible"] for r in rows)
    lines = [
        f"SRBench ground-truth taxonomy — {n} problems",
        f"  Feynman:  {fey}",
        f"  Strogatz: {stro}",
        f"  DeSTrOI-compatible (heuristic): {compat}",
        "",
        "By complexity:",
    ]
    for c in ("simple", "medium", "complex"):
        sub = [r for r in rows if r["complexity"] == c]
        lines.append(f"  {c:<8s} n={len(sub)}")
    lines.append("")
    lines.append("By n_features:")
    from collections import Counter
    for k, v in sorted(Counter(r["n_features"] for r in rows).items()):
        lines.append(f"  D={k}  n={v}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true")
    args = p.parse_args()

    ensure_dirs()
    rows = build(download=args.download)

    csv_path = os.path.join(SRBENCH, "taxonomy.csv")
    sum_path = os.path.join(SRBENCH, "taxonomy_summary.txt")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows)
    with open(sum_path, "w") as f:
        f.write(summary)

    print(f"\nSaved → {csv_path}")
    print(f"Saved → {sum_path}\n")
    print(summary)


if __name__ == "__main__":
    main()
