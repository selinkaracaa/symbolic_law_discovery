"""
Benchmark NeurIPS 2022 transformer on SRBench ground-truth problems
(Feynman ~119 + Strogatz 14 = 133).

Matches SRBench protocol: 75/25 train/test split, R² on held-out test points.
Ground-truth formulas and structural tags come from PMLB metadata.yaml.

Run:
    python3 benchmark_srbench_gt.py --download          # fetch all data first
    python3 benchmark_srbench_gt.py --n 5               # quick smoke test
    python3 benchmark_srbench_gt.py                     # full 133 problems
    python3 benchmark_srbench_gt.py --benchmark feynman # Feynman only

Output:
    results/srbench/gt_benchmark.csv
    results/srbench/gt_benchmark_summary.txt
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "symbolicregression"))

import numpy as np
import torch
import symbolicregression
from sklearn.model_selection import train_test_split

from benchmark_transformer import MODEL_PATH, make_estimator
from results_paths import SRBENCH, ensure_dirs
from srbench_data import download_all, problem_list, read_problem

CSV_PATH = os.path.join(SRBENCH, "gt_benchmark.csv")
SUM_PATH = os.path.join(SRBENCH, "gt_benchmark_summary.txt")

R2_GOOD = 0.99
R2_OK = 0.95


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 10:
        return float("nan")
    yt, yp = y_true[m], y_pred[m]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def fit_and_score(est, x_fit, y_fit, x_test, y_test):
    est.fit(x_fit, y_fit)
    info = est.retrieve_tree(with_infos=True)
    tree = info.get("relabed_predicted_tree")
    pred_str = tree.infix()[:200] if tree is not None else ""
    if tree is None:
        return float("nan"), pred_str
    top = est.top_k_features[0]
    x_eval = x_test[:, top]
    y_test_2d = y_test.reshape(-1, 1) if y_test.ndim == 1 else y_test
    r2 = est.evaluate_tree(tree, x_eval, y_test_2d, metric="r2")
    return float(r2), pred_str


def summarize(rows: list[dict]) -> str:
    r2 = np.array([r["r2_test"] for r in rows], dtype=float)
    valid = np.isfinite(r2)
    r2v = r2[valid]

    lines = [
        "SRBench ground-truth benchmark — NeurIPS 2022 Transformer",
        f"Problems run: {len(rows)}",
        f"Valid R²:     {valid.sum()}",
        f"Mean R²:      {r2v.mean():.4f}",
        f"Median R²:    {np.median(r2v):.4f}",
        f"R² ≥ {R2_GOOD}:   {(r2v >= R2_GOOD).sum()} ({100*(r2v >= R2_GOOD).mean():.1f}%)  [SRBench/Kamienny metric]",
        f"R² ≥ {R2_OK}:    {(r2v >= R2_OK).sum()} ({100*(r2v >= R2_OK).mean():.1f}%)",
        f"R² < 0:       {(r2v < 0).sum()} ({100*(r2v < 0).mean():.1f}%)",
        "",
        "Published reference (Kamienny et al. 2022, NeurIPS):",
        "  ~4th on SRBench overall; competitive R²≥0.99 on Feynman;",
        "  weaker on Strogatz (time-ordered ODE trajectories).",
        "",
    ]

    def section(title, key):
        lines.append(title)
        groups = {}
        for r in rows:
            g = r.get(key, "?")
            groups.setdefault(g, []).append(r["r2_test"])
        for g in sorted(groups, key=lambda x: str(x)):
            vals = np.array(groups[g], dtype=float)
            v = vals[np.isfinite(vals)]
            if len(v) == 0:
                continue
            lines.append(
                f"  {str(g):<16s} n={len(v):3d}  "
                f"mean={v.mean():.3f}  median={np.median(v):.3f}  "
                f"≥{R2_GOOD}: {(v >= R2_GOOD).sum():3d} ({100*(v >= R2_GOOD).mean():.0f}%)"
            )
        lines.append("")

    section("By sub-benchmark:", "benchmark")
    section("By # features:", "n_features")
    section("By operator count:", "complexity")
    section("By nestedness:", "nest_category")
    section("Has trig:", "has_trig")
    section("Has exp/log:", "has_exp_log")

    lines.append("Worst 10:")
    for r in sorted(rows, key=lambda x: x["r2_test"] if np.isfinite(x["r2_test"]) else 999)[:10]:
        lines.append(
            f"  R²={r['r2_test']:7.4f}  {r['benchmark']:<8s}  "
            f"D={r['n_features']} ops={r['n_operators']:2d}  {r['problem']}"
        )

    lines.append("")
    lines.append("Best 10:")
    for r in sorted(rows, key=lambda x: -x["r2_test"] if np.isfinite(x["r2_test"]) else -999)[:10]:
        lines.append(
            f"  R²={r['r2_test']:7.4f}  {r['benchmark']:<8s}  "
            f"D={r['n_features']} ops={r['n_operators']:2d}  {r['problem']}"
        )

    return "\n".join(lines)


def save_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def run(
    problems,
    seed,
    n_trees,
    max_input_points,
    max_number_bags,
    max_train_rows,
    start=0,
    csv_path=CSV_PATH,
):
    print("\nLoading NeurIPS Transformer…", flush=True)
    torch_model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    rows = []
    if start > 0 and os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["r2_test"] = float(r["r2_test"])
            for k in ("n_features", "n_operators", "nestedness", "has_trig",
                      "has_exp_log", "has_sqrt", "has_div", "has_pow"):
                if k in r:
                    r[k] = int(r[k])

    t0 = time.time()
    for i, problem in enumerate(problems[start:], start=start):
        X, y, _, info = read_problem(problem)
        x_fit, x_test, y_fit, y_test = train_test_split(
            X, y, test_size=0.25, shuffle=True, random_state=seed
        )
        if max_train_rows is not None and len(x_fit) > max_train_rows:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(x_fit), size=max_train_rows, replace=False)
            x_fit, y_fit = x_fit[idx], y_fit[idx]
        y_fit = y_fit.reshape(-1, 1) if y.ndim == 1 else y_fit

        try:
            est = make_estimator(
                torch_model,
                n_trees=n_trees,
                max_input_points=max_input_points,
                max_number_bags=max_number_bags,
            )
            t_fit = time.time()
            r2, pred = fit_and_score(est, x_fit, y_fit, x_test, y_test)
            fit_sec = time.time() - t_fit
            err = ""
        except Exception as e:
            r2 = float("nan")
            pred = ""
            fit_sec = float("nan")
            err = str(e)[:120]

        row = {
            "id": i,
            "problem": problem,
            "gt_formula": info["formula"][:200],
            "predicted": pred,
            "r2_test": r2,
            "fit_sec": fit_sec,
            "error": err,
            **{k: info[k] for k in (
                "benchmark", "n_features", "n_operators", "nestedness",
                "complexity", "nest_category", "ops",
                "has_trig", "has_exp_log", "has_sqrt", "has_div", "has_pow",
            )},
        }
        rows.append(row)
        save_csv(csv_path, rows)

        elapsed = time.time() - t0
        done = i - start + 1
        eta = elapsed / done * (len(problems) - i - 1)
        flag = "✓" if r2 >= R2_GOOD else "~" if r2 >= R2_OK else "✗"
        print(
            f"  [{i+1:3d}/{len(problems)}] {flag} R²={r2:7.4f}  "
            f"{info['benchmark']:<8s} D={info['n_features']}  "
            f"{problem}  (~{eta:.0f}s left)",
            flush=True,
        )

    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true", help="Download all datasets then exit")
    p.add_argument("--benchmark", choices=("all", "feynman", "strogatz"), default="all")
    p.add_argument("--n", type=int, default=None, help="Limit number of problems")
    p.add_argument("--start", type=int, default=0, help="Resume from problem index")
    p.add_argument("--seed", type=int, default=29910, help="train_test_split random_state")
    p.add_argument("--n-trees", type=int, default=10, help="Beam candidates (Kamienny 2022 uses 10)")
    p.add_argument("--max-input-points", type=int, default=200)
    p.add_argument("--max-number-bags", type=int, default=10)
    p.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Subsample training rows after split (None = full 75%%; use 2000 for faster CPU runs)",
    )
    args = p.parse_args()

    ensure_dirs()
    problems = problem_list()
    if args.benchmark != "all":
        problems = [pr for pr in problems if pr.startswith(args.benchmark)]
    if args.n is not None:
        problems = problems[: args.n]

    if args.download:
        print(f"Downloading {len(problems)} SRBench ground-truth datasets…")
        download_all(problems)
        print("Done.")
        if args.n is None and not args.start:
            return

    print(f"\nSRBench ground-truth: {len(problems)} problems "
          f"({sum(p.startswith('feynman') for p in problems)} Feynman, "
          f"{sum(p.startswith('strogatz') for p in problems)} Strogatz)")
    print(f"Seed={args.seed}  n_trees={args.n_trees}  max_input_points={args.max_input_points}\n")

    rows = run(
        problems,
        seed=args.seed,
        n_trees=args.n_trees,
        max_input_points=args.max_input_points,
        max_number_bags=args.max_number_bags,
        max_train_rows=args.max_train_rows,
        start=args.start,
    )

    summary = summarize(rows)
    with open(SUM_PATH, "w") as f:
        f.write(summary)

    print(f"\nSaved → {CSV_PATH}")
    print(f"Saved → {SUM_PATH}\n")
    print(summary)


if __name__ == "__main__":
    main()
