"""
In-domain benchmark: NeurIPS transformer on formulas from its own generator.

Uses symbolicregression/envs/generators.py (18 ops, same procedure as training),
not DeSTrOI's 6-op trees.

Run:
    python3 benchmark_transformer_indomain.py
    python3 benchmark_transformer_indomain.py --num-formulas 100 --seed 0
    python3 benchmark_transformer_indomain.py --num-formulas 100 --seed 0 --start 42

Output:
    results/transformer/indomain/transformer_indomain_benchmark_n{N}_seed{S}.csv
    results/transformer/indomain/transformer_indomain_benchmark_summary_n{N}_seed{S}.txt
"""

import sys, os, re, csv, time, argparse, warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
SR = os.path.join(ROOT, "symbolicregression")
sys.path.insert(0, SR)

import numpy as np
import torch
import symbolicregression
from parsers import get_parser
from symbolicregression.envs import ENVS
from symbolicregression.envs.generators import operators_real
from results_paths import TRANSFORMER_INDOMAIN, LOGS, ensure_dirs

MODEL_PATH = os.path.join(SR, "model.pt")
N_FIT = 200
INPUT_DIM = 2
PRED_SIGMA = "1.0"


def make_env(seed):
    argv = sys.argv
    sys.argv = [argv[0]]
    try:
        params = get_parser().parse_args([])
    finally:
        sys.argv = argv
    env = ENVS["functions"](params)
    env.rng = np.random.RandomState(seed)
    return env


def make_estimator(torch_model, n_trees=50):
    return symbolicregression.model.SymbolicTransformerRegressor(
        model=torch_model,
        max_input_points=N_FIT,
        n_trees_to_refine=n_trees,
        rescale=True,
    )


def ops_in_tree(tree):
    tokens = set(tree.prefix().split(","))
    return {t for t in tokens if t in operators_real}


def r2_tree(env, tree, x, y):
    try:
        fn = env.simplifier.tree_to_numexpr_fn(tree)
        yp = fn(x)[:, 0]
        m = np.isfinite(yp) & np.isfinite(y)
        if m.sum() < 10:
            return float("nan")
        ss_r = np.sum((y[m] - yp[m]) ** 2)
        ss_t = np.sum((y[m] - y[m].mean()) ** 2)
        return float(1 - ss_r / ss_t) if ss_t > 0 else float("nan")
    except Exception:
        return float("nan")


def sample_instance(env, max_tries=50):
    x_key, y_key = f"x_to_predict_{PRED_SIGMA}", f"y_to_predict_{PRED_SIGMA}"
    for _ in range(max_tries):
        expr, _ = env.gen_expr(
            train=False,
            input_dimension=INPUT_DIM,
            output_dimension=1,
            n_input_points=N_FIT,
        )
        if x_key not in expr:
            continue
        x_fit = np.array(expr["X_to_fit"][0], dtype=float)
        y_fit = np.array(expr["Y_to_fit"][0], dtype=float).flatten()
        x_test = np.array(expr[x_key], dtype=float)
        y_test = np.array(expr[y_key], dtype=float).flatten()
        tree = expr["tree"]
        if x_fit.shape[0] < N_FIT or x_test.shape[0] < 50:
            continue
        ops = ops_in_tree(tree)
        info = expr["infos"]
        tags = {
            "n_binary_ops": int(info["n_binary_ops"][0]),
            "n_unary_ops": int(info["n_unary_ops"][0]),
            "d_in": int(info["d_in"][0]),
            "has_inv": int("inv" in ops),
            "has_div": int("div" in ops),
            "has_log": int("log" in ops),
            "has_sin": int("sin" in ops),
            "has_cos": int("cos" in ops),
            "ops": ",".join(sorted(ops)),
        }
        return tree, x_fit, y_fit, x_test, y_test, tags
    raise RuntimeError("could not sample a valid in-domain instance")


FIELDNAMES = [
    "id", "formula", "predicted", "r2", "exact_recovery",
    "n_binary_ops", "n_unary_ops", "d_in",
    "has_inv", "has_div", "has_log", "has_sin", "has_cos", "ops",
]


def save_rows(csv_path, rows):
    if not rows:
        return
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def bootstrap_rows_from_log(log_path, n):
    """Recover partial rows from a tee log when CSV checkpoint was missing."""
    if not os.path.exists(log_path):
        return []
    pat = re.compile(
        r"\[\s*(\d+)/\d+\]\s+R²=\s*([-\d.]+)\s+\w+\s+ops=(\d+)b\+(\d+)u"
    )
    rows = []
    for line in open(log_path):
        m = pat.search(line)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if idx >= n:
            continue
        rows.append({
            "id": idx,
            "formula": "(recovered from log)",
            "predicted": "",
            "r2": float(m.group(2)),
            "exact_recovery": 0,
            "n_binary_ops": int(m.group(3)),
            "n_unary_ops": int(m.group(4)),
            "d_in": INPUT_DIM,
            "has_inv": "",
            "has_div": "",
            "has_log": "",
            "has_sin": "",
            "has_cos": "",
            "ops": "",
        })
    rows.sort(key=lambda r: r["id"])
    return rows


def run_benchmark(n, seed, n_trees, start=0, csv_path=None):
    print(f"\nLoading transformer from {MODEL_PATH} …", flush=True)
    torch_model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    env = make_env(seed)
    est_env = torch_model.env

    rows = []
    if start > 0:
        print(f"Advancing generator to formula {start + 1}…", flush=True)
        for _ in range(start):
            sample_instance(env)
        if csv_path and os.path.exists(csv_path):
            with open(csv_path, newline="") as f:
                rows = list(csv.DictReader(f))
            for r in rows:
                for k in ("r2", "exact_recovery", "n_binary_ops", "n_unary_ops", "d_in",
                          "has_inv", "has_div", "has_log", "has_sin", "has_cos"):
                    if r.get(k) not in (None, ""):
                        r[k] = float(r[k]) if k == "r2" else int(float(r[k]))
        elif csv_path:
            log_path = os.path.join(LOGS, "transformer_indomain_benchmark_run.log")
            rows = bootstrap_rows_from_log(log_path, start)
            if rows:
                save_rows(csv_path, rows)
                print(f"  Bootstrapped {len(rows)} rows from log → {csv_path}", flush=True)

    t0 = time.time()
    for i in range(start, n):
        tree, x_fit, y_fit, x_test, y_test, tags = sample_instance(env)
        formula = tree.infix()

        est = make_estimator(torch_model, n_trees=n_trees)
        try:
            est.fit(x_fit, y_fit)
            pred_tree = est.retrieve_tree(with_infos=True)["relabed_predicted_tree"]
            pred = pred_tree.infix()
            r2 = r2_tree(est_env, pred_tree, x_test, y_test)
            recovered = int(pred_tree.infix() == formula)
        except Exception as e:
            pred = f"ERROR: {e}"
            r2 = float("nan")
            recovered = 0

        row = {
            "id": i,
            "formula": formula[:200],
            "predicted": pred[:200],
            "r2": r2,
            "exact_recovery": recovered,
            **tags,
        }
        rows.append(row)
        if csv_path:
            save_rows(csv_path, rows)

        status = "OK" if r2 >= 0.95 else "weak" if r2 >= 0.5 else "fail"
        elapsed = time.time() - t0
        done = i - start + 1
        eta = elapsed / done * (n - i - 1) if done else 0
        print(
            f"  [{i+1:3d}/{n}] R²={r2:7.4f}  {status:4s}  "
            f"ops={tags['n_binary_ops']}b+{tags['n_unary_ops']}u  "
            f"({elapsed:.0f}s elapsed, ~{eta:.0f}s left)",
            flush=True,
        )

    return rows


def summarize(rows):
    r2 = np.array([r["r2"] for r in rows], dtype=float)
    valid = np.isfinite(r2)
    r2v = r2[valid]
    exact = sum(r["exact_recovery"] for r in rows)

    lines = [
        "In-domain benchmark (transformer generator, 18-op vocab)",
        f"Total formulas: {len(rows)}",
        f"Valid R²:       {valid.sum()}",
        f"Mean R²:        {r2v.mean():.4f}" if len(r2v) else "Mean R²:        n/a",
        f"Median R²:      {np.median(r2v):.4f}" if len(r2v) else "Median R²:      n/a",
        f"R² ≥ 0.95:      {(r2v >= 0.95).sum()} ({100*(r2v >= 0.95).mean():.0f}%)" if len(r2v) else "",
        f"R² ≥ 0.50:      {(r2v >= 0.5).sum()} ({100*(r2v >= 0.5).mean():.0f}%)" if len(r2v) else "",
        f"R² < 0:         {(r2v < 0).sum()} ({100*(r2v < 0).mean():.0f}%)" if len(r2v) else "",
        f"Exact recovery: {exact} ({100*exact/len(rows):.0f}%)",
        "",
    ]

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
    if all(r.get("has_inv") not in ("", None) for r in rows):
        bucket("has inv", np.array([int(r["has_inv"]) for r in rows], bool))
        bucket("has div", np.array([int(r["has_div"]) for r in rows], bool))
        bucket(
            "no inv/div",
            np.array([not int(r["has_inv"]) and not int(r["has_div"]) for r in rows], bool),
        )
    else:
        lines.append("  (op tags missing for log-recovered rows)")

    lines.append("")
    lines.append("Worst 5:")
    worst = sorted(rows, key=lambda r: r["r2"] if np.isfinite(r["r2"]) else 999)[:5]
    for r in worst:
        lines.append(f"  R²={r['r2']:7.4f}  {r['formula'][:70]}")

    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-formulas", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-trees", type=int, default=50)
    p.add_argument("--start", type=int, default=0, help="Resume from formula index (0-based)")
    args = p.parse_args()

    ensure_dirs()
    tag = f"n{args.num_formulas}_seed{args.seed}"
    csv_path = os.path.join(TRANSFORMER_INDOMAIN, f"transformer_indomain_benchmark_{tag}.csv")
    sum_path = os.path.join(TRANSFORMER_INDOMAIN, f"transformer_indomain_benchmark_summary_{tag}.txt")

    rows = run_benchmark(
        args.num_formulas, args.seed, args.n_trees,
        start=args.start, csv_path=csv_path,
    )

    summary = summarize(rows)
    with open(sum_path, "w") as f:
        f.write(summary)

    print(f"\nSaved → {csv_path}")
    print(f"Saved → {sum_path}\n")
    print(summary)


if __name__ == "__main__":
    main()
