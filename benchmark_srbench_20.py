"""
SRBench 20-problem benchmark: Kamienny E2E vs DeSTrOI+Kamienny.

Per problem runs both methods (same train/test split):
  1. Transformer alone
  2. DeSTrOI operator mask → Transformer

Run:
    python3 benchmark_srbench_20.py --download
    python3 benchmark_srbench_20.py --n 1              # smoke test
    python3 benchmark_srbench_20.py                    # full 20 (~3–4 h CPU)

Output:
    results/srbench/srbench_20_benchmark.csv
    results/srbench/srbench_20_benchmark_summary.txt
"""

from __future__ import annotations

import argparse
import csv
import json
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
from sklearn.model_selection import train_test_split

from benchmark_three_way import (
    DESTROI_OPS,
    OP_TOKEN,
    THRESHOLD,
    DeSTrOIPredictor,
    destroi_accuracy,
)
from benchmark_transformer import MODEL_PATH, make_estimator
from results_paths import SRBENCH, ensure_dirs
from srbench_data import download_all, read_problem

SUBSET_PATH = os.path.join(ROOT, "datasets", "srbench", "benchmark_subset_20.json")
CSV_PATH = os.path.join(SRBENCH, "srbench_20_benchmark.csv")
SUM_PATH = os.path.join(SRBENCH, "srbench_20_benchmark_summary.txt")

R2_GOOD = 0.99
R2_OK = 0.95
N_DESTROI_POINTS = 500


def load_subset(path: str) -> list[str]:
    with open(path) as f:
        return json.load(f)


def gt_destroi_ops(ops_str: str) -> set[str]:
    if not ops_str:
        return set()
    return {o for o in ops_str.split(",") if o in DESTROI_OPS}


def pick_destroi_slice(X: np.ndarray, y: np.ndarray, seed: int, n_max: int = N_DESTROI_POINTS):
    rng = np.random.default_rng(seed)
    n = min(len(X), n_max)
    idx = rng.choice(len(X), size=n, replace=False) if len(X) > n else np.arange(len(X))
    Xs = X[idx]
    ys = y[idx].astype(float)
    if X.shape[1] == 1:
        return Xs[:, 0], Xs[:, 0], ys
    if X.shape[1] == 2:
        return Xs[:, 0], Xs[:, 1], ys
    cors = []
    for j in range(X.shape[1]):
        c = np.corrcoef(Xs[:, j], ys)[0, 1]
        cors.append(abs(c) if np.isfinite(c) else 0.0)
    top2 = np.argsort(cors)[-2:]
    return Xs[:, top2[0]], Xs[:, top2[1]], ys


def fit_and_score(est, x_fit, y_fit, x_test, y_test, forbidden=None):
    if forbidden:
        est.fit(x_fit, y_fit, forbidden_token_ids=forbidden)
    else:
        est.fit(x_fit, y_fit)
    info = est.retrieve_tree(with_infos=True)
    tree = info.get("relabed_predicted_tree")
    pred_str = tree.infix()[:300] if tree is not None else ""
    if tree is None:
        return float("nan"), pred_str
    top = est.top_k_features[0]
    x_eval = x_test[:, top]
    y_test_2d = y_test.reshape(-1, 1) if y_test.ndim == 1 else y_test
    r2 = est.evaluate_tree(tree, x_eval, y_test_2d, metric="r2")
    return float(r2), pred_str


def summarize(rows: list[dict]) -> str:
    by_method = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)
    n_probs = len({r["problem"] for r in rows})

    lines = [
        "SRBench 20-problem benchmark — Kamienny E2E vs DeSTrOI+Kamienny",
        f"Problems: {n_probs}",
        "",
    ]

    for method, mrows in [("e2e", by_method.get("e2e", [])), ("destroi_e2e", by_method.get("destroi_e2e", []))]:
        if not mrows:
            continue
        r2 = np.array([r["r2_test"] for r in mrows], dtype=float)
        r2v = r2[np.isfinite(r2)]
        label = "Transformer alone" if method == "e2e" else "DeSTrOI + Transformer"
        lines += [
            f"--- {label} ---",
            f"  Mean R²:   {r2v.mean():.4f}" if len(r2v) else "  Mean R²:   n/a",
            f"  Median R²: {np.median(r2v):.4f}" if len(r2v) else "  Median R²: n/a",
            f"  R² ≥ 0.99: {(r2v >= R2_GOOD).sum()} / {len(r2v)}",
            f"  R² ≥ 0.95: {(r2v >= R2_OK).sum()} / {len(r2v)}",
            f"  Mean fit time: {np.nanmean([float(r['fit_sec']) for r in mrows if r.get('fit_sec')]):.1f}s",
            "",
        ]

    if by_method.get("e2e") and by_method.get("destroi_e2e"):
        e2e = {r["problem"]: float(r["r2_test"]) for r in by_method["e2e"]}
        comb = {r["problem"]: float(r["r2_test"]) for r in by_method["destroi_e2e"]}
        deltas = [comb[p] - e2e[p] for p in e2e if p in comb and np.isfinite(e2e[p]) and np.isfinite(comb[p])]
        if deltas:
            d = np.array(deltas)
            lines += [
                "--- Head-to-head (DeSTrOI+E2E minus E2E) ---",
                f"  Mean ΔR²: {d.mean():+.4f}",
                f"  Better (Δ>0.005): {(d > 0.005).sum()}",
                f"  Worse  (Δ<-0.005): {(d < -0.005).sum()}",
                f"  Similar: {(np.abs(d) <= 0.005).sum()}",
                "",
            ]

    acc = [float(r["destroi_acc"]) for r in by_method.get("destroi_e2e", []) if r.get("destroi_acc")]
    if acc:
        lines += [f"Mean DeSTrOI operator accuracy: {np.mean(acc):.1%}", ""]

    lines.append("Per problem:")
    seen = []
    for r in rows:
        if r["problem"] in seen:
            continue
        seen.append(r["problem"])
    for p in [r["problem"] for r in rows if r["method"] == "e2e"]:
        e = next(r for r in rows if r["problem"] == p and r["method"] == "e2e")
        c = next(r for r in rows if r["problem"] == p and r["method"] == "destroi_e2e")
        et, ct = float(e["r2_test"]), float(c["r2_test"])
        delta = ct - et if np.isfinite(et) and np.isfinite(ct) else float("nan")
        lines.append(
            f"  {p:<28s}  E2E={et:7.4f}  Comb={ct:7.4f}  "
            f"Δ={delta:+.4f}  acc={float(c['destroi_acc']):.0%}  blocked={c.get('blocked', '')}"
        )

    lines.extend(["", "--- By SRBench group (E2E) ---"])
    for bench in ("feynman", "strogatz"):
        mrows = [r for r in by_method.get("e2e", []) if r.get("benchmark") == bench]
        if not mrows:
            continue
        r2v = np.array([float(r["r2_test"]) for r in mrows], dtype=float)
        r2v = r2v[np.isfinite(r2v)]
        rate = 100 * (r2v >= R2_GOOD).mean() if len(r2v) else 0
        lines.append(
            f"  {bench:<10s}  n={len(r2v):2d}  "
            f"R²≥0.99: {(r2v >= R2_GOOD).sum()}/{len(r2v)} ({rate:.0f}%)  "
            f"mean R²={r2v.mean():.3f}"
        )
    lines.extend([
        "",
        "NOTE: Published Kamienny: Feynman 84.8% and Strogatz 35.7% are success RATES",
        "  over all 119 + 14 problems — not comparable to this 20-problem pilot.",
    ])
    return "\n".join(lines)


def save_csv(path: str, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys(), quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        w.writerows(rows)


def run_problem(problem, i, n_total, torch_model, destroi, seed, n_trees,
                max_input_points, max_number_bags, max_train_rows):
    X, y, _, info = read_problem(problem)
    print(
        f"\n  Problem {i+1}/{n_total}: {problem}  "
        f"({info['benchmark']}, D={info['n_features']}, {info['complexity']})",
        flush=True,
    )
    x_fit, x_test, y_fit, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=True, random_state=seed
    )
    if max_train_rows is not None and len(x_fit) > max_train_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x_fit), size=max_train_rows, replace=False)
        x_fit, y_fit = x_fit[idx], y_fit[idx]
    y_fit = y_fit.reshape(-1, 1) if y.ndim == 1 else y_fit

    x1, x2, y_dest = pick_destroi_slice(x_fit, y_fit.ravel(), seed)
    y_clip = np.clip(y_dest, *np.percentile(y_dest, [1, 99]))
    scores, _, absent = destroi.predict(x1, x2, y_clip, threshold=THRESHOLD)
    acc = destroi_accuracy(scores, gt_destroi_ops(info["ops"]))
    blocked = sorted(set(absent) & DESTROI_OPS)
    forbidden = [OP_TOKEN[op] for op in blocked if op in OP_TOKEN]

    base = {
        "problem": problem,
        "benchmark": info["benchmark"],
        "n_features": info["n_features"],
        "gt_formula": info["formula"][:200],
        "gt_ops": info["ops"],
        "destroi_acc": acc,
        "blocked": ",".join(blocked),
        "destroi_scores": ";".join(f"{k}:{scores[k]:.3f}" for k in sorted(scores)),
    }

    out_rows = []
    for method, forb, label in (("e2e", None, "E2E"), ("destroi_e2e", forbidden, "DeSTrOI+E2E")):
        print(f"    → [{i+1}/{n_total}] {problem} — {label}…", flush=True)
        try:
            est = make_estimator(torch_model, n_trees=n_trees,
                                 max_input_points=max_input_points,
                                 max_number_bags=max_number_bags)
            t0 = time.time()
            r2, pred = fit_and_score(est, x_fit, y_fit, x_test, y_test, forbidden=forb)
            fit_sec = time.time() - t0
            err = ""
        except Exception as e:
            r2, pred, fit_sec, err = float("nan"), "", float("nan"), str(e)[:200]

        out_rows.append({**base, "id": i, "method": method, "r2_test": r2,
                         "predicted": pred, "fit_sec": round(fit_sec, 2) if np.isfinite(fit_sec) else fit_sec,
                         "error": err})
        flag = "✓" if r2 >= R2_GOOD else "~" if r2 >= R2_OK else "✗"
        print(f"  [{i+1:2d}/{n_total}] {label:<12s} {flag} R²={r2:7.4f}  {problem}", flush=True)
    return out_rows


def run(problems, seed, n_trees, max_input_points, max_number_bags, max_train_rows,
        start=0, csv_path=CSV_PATH):
    print("\nLoading NeurIPS Transformer…", flush=True)
    torch_model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    print("Loading DeSTrOI weights…", flush=True)
    destroi = DeSTrOIPredictor()

    rows = []
    if start > 0 and os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["r2_test"] = float(r["r2_test"])
            r["n_features"] = int(r["n_features"])
            r["destroi_acc"] = float(r["destroi_acc"])
            if r.get("fit_sec"):
                r["fit_sec"] = float(r["fit_sec"])

    t0 = time.time()
    for i, problem in enumerate(problems[start:], start=start):
        rows.extend(run_problem(
            problem, i, len(problems), torch_model, destroi, seed, n_trees,
            max_input_points, max_number_bags, max_train_rows))
        save_csv(csv_path, rows)
        elapsed = time.time() - t0
        done = i - start + 1
        eta = elapsed / done * (len(problems) - i - 1)
        print(f"    (~{eta/60:.1f} min left)\n", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description="SRBench 20: E2E vs DeSTrOI+E2E")
    p.add_argument("--download", action="store_true")
    p.add_argument("--subset", default=SUBSET_PATH)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--seed", type=int, default=29910)
    p.add_argument("--n-trees", type=int, default=10)
    p.add_argument("--max-input-points", type=int, default=200)
    p.add_argument("--max-number-bags", type=int, default=10)
    p.add_argument("--max-train-rows", type=int, default=2000)
    args = p.parse_args()

    ensure_dirs()
    problems = load_subset(args.subset)
    if args.n is not None:
        problems = problems[: args.n]
    if args.download:
        print(f"Downloading {len(problems)} datasets…")
        download_all(problems)

    print(f"\nSRBench benchmark: {len(problems)} problems (E2E + DeSTrOI+E2E each)")
    print(f"seed={args.seed}  n_trees={args.n_trees}  max_train_rows={args.max_train_rows}\n")

    rows = run(problems, args.seed, args.n_trees, args.max_input_points,
               args.max_number_bags, args.max_train_rows, start=args.start)
    summary = summarize(rows)
    with open(SUM_PATH, "w") as f:
        f.write(summary)
    print(f"\nSaved → {CSV_PATH}\nSaved → {SUM_PATH}\n\n{summary}")


if __name__ == "__main__":
    main()
