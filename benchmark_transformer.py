"""
Benchmark the NeurIPS transformer on N random formulas (DeSTrOI 6-op vocabulary).

Generates random symbolic trees the same way DeSTrOI trains, fits the transformer
on 200 points, reports R² on 500 held-out points, and saves a CSV summary.

Run:
    python3 benchmark_transformer.py              # 100 formulas (default)
    python3 benchmark_transformer.py --n 20     # quick smoke test
    python3 benchmark_transformer.py --n 100 --seed 0

Output:
    results/transformer/synthetic_destroi_vocab/transformer_benchmark.csv
    results/transformer/synthetic_destroi_vocab/transformer_benchmark_summary.txt
"""

import sys, os, csv, time, argparse, warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
DESTORI = os.path.join(ROOT, "Symbolic-Prediction-master")
sys.path.insert(0, os.path.join(ROOT, "symbolicregression"))
sys.path.insert(0, DESTORI)

import numpy as np
import sympy as sp
import torch
import symbolicregression

from common.symbolic_tree import (
    OPERATOR_LIST,
    generate_random_tree,
    get_value,
    get_classification_labels,
    get_symbolic_form,
    get_node_from_symbolic_form,
)
from common.preset import (
    TERMINAL_PROB, CONSTANT_PROB, CONSTANT_RANGE,
    MIN_DEPTH, MAX_DEPTH, VAR_RANGE_LOW, VAR_RANGE_HIGH, Y_THRESH, NOISE,
)
from results_paths import TRANSFORMER_SYNTH, LOGS, ensure_dirs

MODEL_PATH = os.path.join(ROOT, "symbolicregression", "model.pt")
REPLACE_SYMPY = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
UNARIES = {"inv", "sqrt", "log", "sin"}
N_FIT, N_TEST = 200, 500


def make_estimator(torch_model, n_trees=50, max_input_points=200, max_number_bags=10):
    return symbolicregression.model.SymbolicTransformerRegressor(
        model=torch_model,
        max_input_points=max_input_points,
        max_number_bags=max_number_bags,
        n_trees_to_refine=n_trees,
        rescale=True,
    )


def sample_formula_and_data(rng, nsample_fit=N_FIT, nsample_test=N_TEST):
    """Random tree + (x,y) points; reject until values are in range."""
    while True:
        root = generate_random_tree(
            2, TERMINAL_PROB, CONSTANT_PROB, CONSTANT_RANGE, [MIN_DEPTH, MAX_DEPTH]
        )
        formula = get_symbolic_form(root)
        labels = get_classification_labels(root)
        ops = {OPERATOR_LIST[i] for i, v in enumerate(labels) if v > 0}

        pts_fit, pts_test = [], []
        for target, buf in ((nsample_fit, pts_fit), (nsample_test, pts_test)):
            while len(buf) < target:
                x = rng.uniform(VAR_RANGE_LOW, VAR_RANGE_HIGH, (target * 4, 2))
                for xi in x:
                    try:
                        y = get_value(root, xi)
                        if np.isfinite(y) and abs(y) < Y_THRESH:
                            buf.append((xi[0], xi[1], y + rng.normal(0, NOISE)))
                            if len(buf) >= target:
                                break
                    except Exception:
                        pass
            if len(buf) < target:
                break
        if len(pts_fit) >= nsample_fit and len(pts_test) >= nsample_test:
            fit = np.array(pts_fit[:nsample_fit])
            test = np.array(pts_test[:nsample_test])
            return root, formula, ops, fit[:, :2], fit[:, 2], test[:, :2], test[:, 2]


def formula_tags(formula, ops):
    """Simple structural tags for grouping failures."""
    root = get_node_from_symbolic_form(formula)
    top = OPERATOR_LIST[root.operator] if not root.is_terminal else "leaf"

    def child_ops(node):
        if node.is_terminal:
            return []
        op = OPERATOR_LIST[node.operator]
        kids = []
        for c in node.children:
            if c.is_terminal:
                kids.append("leaf")
            else:
                kids.append(OPERATOR_LIST[c.operator])
        return [op] + kids

    kids = child_ops(root)
    mul_unary = (
        top == "mul"
        and len(root.children) == 2
        and all(
            not c.is_terminal and OPERATOR_LIST[c.operator] in UNARIES
            for c in root.children
        )
    )
    return {
        "top_op": top,
        "n_ops": len(ops),
        "has_inv": int("inv" in ops),
        "has_log": int("log" in ops),
        "has_sin": int("sin" in ops),
        "has_sqrt": int("sqrt" in ops),
        "mul_unary_pair": int(mul_unary),
        "ops": ",".join(sorted(ops)),
    }


def r2_from_infix(raw, x_test, y_test):
    x0s, x1s = sp.symbols("x_0 x_1")
    s = raw
    for op, rep in REPLACE_SYMPY.items():
        s = s.replace(op, rep)
    try:
        expr = sp.parse_expr(s, local_dict={"x_0": x0s, "x_1": x1s})
        f = sp.lambdify([x0s, x1s], expr, modules="numpy")
        yp = np.array(f(x_test[:, 0], x_test[:, 1]), dtype=float).flatten()
        m = np.isfinite(yp) & np.isfinite(y_test)
        if m.sum() < 10:
            return float("nan")
        ss_r = np.sum((y_test[m] - yp[m]) ** 2)
        ss_t = np.sum((y_test[m] - y_test[m].mean()) ** 2)
        return float(1 - ss_r / ss_t) if ss_t > 0 else float("nan")
    except Exception:
        return float("nan")


def run_benchmark(n, seed, n_trees):
    print(f"\nLoading transformer from {MODEL_PATH} …", flush=True)
    torch_model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    rng = np.random.default_rng(seed)

    rows = []
    t0 = time.time()
    for i in range(n):
        root, formula, ops, x_fit, y_fit, x_test, y_test = sample_formula_and_data(rng)
        tags = formula_tags(formula, ops)

        est = make_estimator(torch_model, n_trees=n_trees)
        try:
            est.fit(x_fit, y_fit)
            pred = est.retrieve_tree(with_infos=True)["relabed_predicted_tree"].infix()
            r2 = r2_from_infix(pred, x_test, y_test)
        except Exception as e:
            pred = f"ERROR: {e}"
            r2 = float("nan")

        row = {
            "id": i,
            "formula": formula,
            "predicted": pred[:120],
            "r2": r2,
            **tags,
        }
        rows.append(row)

        status = "OK" if r2 >= 0.95 else "weak" if r2 >= 0.5 else "fail"
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (n - i - 1)
        print(
            f"  [{i+1:3d}/{n}] R²={r2:7.4f}  {status:4s}  ops={tags['ops']:<20s}  "
            f"top={tags['top_op']:<4s}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s left)",
            flush=True,
        )

    return rows


def summarize(rows):
    r2 = np.array([r["r2"] for r in rows], dtype=float)
    valid = np.isfinite(r2)
    r2v = r2[valid]

    lines = []
    lines.append(f"Total formulas: {len(rows)}")
    lines.append(f"Valid R²:       {valid.sum()}")
    lines.append(f"Mean R²:        {r2v.mean():.4f}")
    lines.append(f"Median R²:      {np.median(r2v):.4f}")
    lines.append(f"R² ≥ 0.95:      {(r2v >= 0.95).sum()} ({100*(r2v >= 0.95).mean():.0f}%)")
    lines.append(f"R² ≥ 0.50:      {(r2v >= 0.5).sum()} ({100*(r2v >= 0.5).mean():.0f}%)")
    lines.append(f"R² < 0:         {(r2v < 0).sum()} ({100*(r2v < 0).mean():.0f}%)")
    lines.append("")

    def bucket(name, mask):
        sub = r2[mask & valid]
        if len(sub) == 0:
            return
        lines.append(
            f"  {name:<22s}  n={len(sub):3d}  "
            f"mean={sub.mean():.3f}  median={np.median(sub):.3f}  "
            f"fail%={100*(sub < 0.5).mean():.0f}"
        )

    lines.append("By feature:")
    bucket("has inv", np.array([r["has_inv"] for r in rows], bool))
    bucket("has log", np.array([r["has_log"] for r in rows], bool))
    bucket("mul+2 unaries", np.array([r["mul_unary_pair"] for r in rows], bool))
    bucket("no inv", np.array([not r["has_inv"] for r in rows], bool))

    lines.append("")
    lines.append("Worst 10:")
    worst = sorted(rows, key=lambda r: r["r2"] if np.isfinite(r["r2"]) else 999)[:10]
    for r in worst:
        lines.append(f"  R²={r['r2']:7.4f}  {r['formula'][:70]}")

    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100, help="Number of random formulas")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-trees", type=int, default=50, help="Beam candidates to refine")
    args = p.parse_args()

    ensure_dirs()
    rows = run_benchmark(args.n, args.seed, args.n_trees)

    tag = f"n{args.n}_seed{args.seed}"
    csv_path = os.path.join(TRANSFORMER_SYNTH, f"transformer_benchmark_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    summary = summarize(rows)
    sum_path = os.path.join(TRANSFORMER_SYNTH, f"transformer_benchmark_summary_{tag}.txt")
    default_csv = os.path.join(TRANSFORMER_SYNTH, "transformer_benchmark.csv")
    default_sum = os.path.join(TRANSFORMER_SYNTH, "transformer_benchmark_summary.txt")
    with open(sum_path, "w") as f:
        f.write(summary)
    with open(default_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    with open(default_sum, "w") as f:
        f.write(summary)

    print(f"\nSaved → {csv_path}")
    print(f"Saved → {sum_path}")
    print(f"Also  → {default_csv}\n")
    print(summary)


if __name__ == "__main__":
    main()
