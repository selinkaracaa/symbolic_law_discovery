"""
Three-way benchmark on N random formulas (same generator as benchmark_transformer.py).

Per formula:
  1. DeSTrOI  — operator-ID accuracy (6 ops)
  2. Transformer alone — R² on held-out points
  3. Combined — Transformer with DeSTrOI-blocked operators — R²

Run:
    python3 benchmark_three_way.py --n 100 --seed 0
    python3 benchmark_three_way.py --n 5 --seed 0    # quick test (~5 min)

Output:
    results/three_way/three_way_benchmark_n{N}_seed{S}.csv
    results/three_way/three_way_benchmark_summary_n{N}_seed{S}.txt
"""

import os
import sys
import csv
import time
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

ROOT = os.path.dirname(os.path.abspath(__file__))
DESTORI = os.path.join(ROOT, "Symbolic-Prediction-master")
sys.path.insert(0, os.path.join(ROOT, "symbolicregression"))
sys.path.insert(0, DESTORI)

import numpy as np
import torch
import symbolicregression
from destroi_predict import OPERATOR_LIST as DESTROI_OPS_LIST, _points_to_image

from benchmark_transformer import (
    MODEL_PATH,
    formula_tags,
    r2_from_infix,
    make_estimator,
)
from results_paths import THREE_WAY, ensure_dirs
from common.symbolic_tree import (
    OPERATOR_LIST,
    generate_random_tree,
    get_value,
    get_symbolic_form,
)
from common.preset import (
    TERMINAL_PROB, CONSTANT_PROB, CONSTANT_RANGE,
    MIN_DEPTH, MAX_DEPTH, VAR_RANGE_LOW, VAR_RANGE_HIGH, Y_THRESH, NOISE,
)

OP_TOKEN = {
    "add": 42, "sub": 61, "mul": 53, "div": 47, "pow": 55,
    "inv": 51, "sqrt": 60, "log": 52, "exp": 50,
    "sin": 59, "cos": 46, "tan": 62, "abs": 41,
}
DESTROI_OPS = set(DESTROI_OPS_LIST)
THRESHOLD = 0.5
N_DESTROI = 500
N_FIT = 200
N_TEST = 500


class DeSTrOIPredictor:
    """Load DeSTrOI weights once (much faster than destroi_predict per call)."""

    def __init__(self):
        from common.channel import add_channel, remove_channel
        from model.super_resolution import EDSR
        from model.CNN import CNN

        self._add_channel = add_channel
        self._remove_channel = remove_channel
        enc_path = os.path.join(DESTORI, "weights", "superr_encoder_2.h5")
        dec_tmpl = os.path.join(DESTORI, "weights", "decoder_2_{}.h5")

        self.encoder = EDSR()
        self.encoder.load_weights(enc_path)
        self.decoder = CNN()
        self.dec_paths = [dec_tmpl.format(i) for i in range(len(DESTROI_OPS_LIST))]

    def predict(self, x1s, x2s, ys, threshold=THRESHOLD):
        x1s = np.asarray(x1s, dtype=float)
        x2s = np.asarray(x2s, dtype=float)
        ys = np.asarray(ys, dtype=float)

        lr_batch = self._add_channel(_points_to_image(x1s, x2s, ys)[np.newaxis])
        hr_img = self._remove_channel(self.encoder.predict(lr_batch, verbose=0))

        scores = {}
        for i, op in enumerate(DESTROI_OPS_LIST):
            self.decoder.load_weights(self.dec_paths[i])
            inp = self._add_channel(hr_img)
            scores[op] = float(np.array(self.decoder.predict_on_batch(inp)).flatten()[0])

        present = [op for op, p in scores.items() if p >= threshold]
        absent = [op for op, p in scores.items() if p < threshold]
        return scores, present, absent


def destroi_accuracy(scores, true_ops):
    correct = sum(
        (scores[op] >= THRESHOLD) == (op in true_ops) for op in DESTROI_OPS_LIST
    )
    return correct / len(DESTROI_OPS_LIST)


def _collect_points(root, n, rng):
    pts = []
    while len(pts) < n:
        x = rng.uniform(VAR_RANGE_LOW, VAR_RANGE_HIGH, (n * 4, 2))
        for xi in x:
            try:
                y = get_value(root, xi)
                if np.isfinite(y) and abs(y) < Y_THRESH:
                    pts.append((xi[0], xi[1], y + rng.normal(0, NOISE)))
                    if len(pts) >= n:
                        break
            except Exception:
                pass
    a = np.array(pts[:n])
    return a[:, :2], a[:, 2]


def sample_instance(rng):
    """One random tree; fit / DeSTrOI / test point pools."""
    while True:
        root = generate_random_tree(
            2, TERMINAL_PROB, CONSTANT_PROB, CONSTANT_RANGE, [MIN_DEPTH, MAX_DEPTH]
        )
        formula = get_symbolic_form(root)
        labels = np.zeros(len(OPERATOR_LIST))
        from common.symbolic_tree import get_classification_labels
        labels = get_classification_labels(root)
        ops = {OPERATOR_LIST[i] for i, v in enumerate(labels) if v > 0}

        x_dest, y_dest = _collect_points(root, N_DESTROI, rng)
        x_fit, y_fit = _collect_points(root, N_FIT, rng)
        x_test, y_test = _collect_points(root, N_TEST, rng)
        if len(x_dest) >= N_DESTROI and len(x_fit) >= N_FIT and len(x_test) >= N_TEST:
            y_clip = np.clip(y_dest, *np.percentile(y_dest, [1, 99]))
            return formula, ops, x_fit, y_fit, x_dest, y_clip, x_test, y_test


def fit_r2(est, x_fit, y_fit, x_test, y_test, forbidden=None):
    if forbidden:
        est.fit(x_fit, y_fit, forbidden_token_ids=forbidden)
    else:
        est.fit(x_fit, y_fit)
    pred = est.retrieve_tree(with_infos=True)["relabed_predicted_tree"].infix()
    return r2_from_infix(pred, x_test, y_test), pred[:120]


def summarize(rows):
    def col(k):
        return np.array([r[k] for r in rows], dtype=float)

    acc = col("destroi_acc")
    r2_t = col("r2_trans")
    r2_c = col("r2_comb")
    delta = r2_c - r2_t
    valid = np.isfinite(r2_t) & np.isfinite(r2_c)

    lines = [
        f"Total formulas: {len(rows)}",
        f"Mean DeSTrOI acc:     {acc.mean():.1%}",
        f"Mean Transformer R²:  {np.nanmean(r2_t):.4f}",
        f"Mean Combined R²:     {np.nanmean(r2_c):.4f}",
        f"Mean ΔR² (Comb-Trans): {np.nanmean(delta):+.4f}",
        "",
        f"Trans R² ≥ 0.95:  {(r2_t[valid] >= 0.95).sum()} ({100*(r2_t[valid] >= 0.95).mean():.0f}%)",
        f"Comb R² ≥ 0.95:   {(r2_c[valid] >= 0.95).sum()} ({100*(r2_c[valid] >= 0.95).mean():.0f}%)",
        f"Trans R² < 0:     {(r2_t[valid] < 0).sum()} ({100*(r2_t[valid] < 0).mean():.0f}%)",
        f"Comb R² < 0:      {(r2_c[valid] < 0).sum()} ({100*(r2_c[valid] < 0).mean():.0f}%)",
        "",
        f"Combined better (ΔR² > 0.005):  {(delta[valid] > 0.005).sum()}",
        f"Combined worse  (ΔR² < -0.005): {(delta[valid] < -0.005).sum()}",
        f"Similar         (|ΔR²| ≤ 0.005): {(np.abs(delta[valid]) <= 0.005).sum()}",
        "",
        "By feature (Transformer R² | Combined R² | fail% trans):",
    ]

    def bucket(name, mask):
        m = mask & valid
        if m.sum() == 0:
            return
        lines.append(
            f"  {name:<18s} n={m.sum():3d}  "
            f"trans={r2_t[m].mean():.3f}  comb={r2_c[m].mean():.3f}  "
            f"Δ={delta[m].mean():+.3f}  fail%={(r2_t[m] < 0.5).mean()*100:.0f}"
        )

    has_inv = np.array([r["has_inv"] for r in rows], bool)
    bucket("has inv", has_inv)
    bucket("no inv", ~has_inv)
    bucket("mul+2 unaries", np.array([r["mul_unary_pair"] for r in rows], bool))

    lines.append("")
    lines.append("Biggest Combined wins (ΔR²):")
    for r in sorted(rows, key=lambda r: r["delta_r2"], reverse=True)[:5]:
        lines.append(f"  Δ={r['delta_r2']:+.4f}  trans={r['r2_trans']:.3f}  comb={r['r2_comb']:.3f}  {r['formula'][:60]}")

    lines.append("")
    lines.append("Biggest Combined losses (ΔR²):")
    for r in sorted(rows, key=lambda r: r["delta_r2"])[:5]:
        lines.append(f"  Δ={r['delta_r2']:+.4f}  trans={r['r2_trans']:.3f}  comb={r['r2_comb']:.3f}  {r['formula'][:60]}")

    return "\n".join(lines)


def save_rows(csv_path, rows):
    if not rows:
        return
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def run(n, seed, n_trees, start=0, csv_path=None):
    print("\nLoading NeurIPS Transformer…", flush=True)
    torch_model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    print("Loading DeSTrOI weights…", flush=True)
    destroi = DeSTrOIPredictor()

    rng = np.random.default_rng(seed)
    rows = []
    if start > 0:
        print(f"Advancing RNG to formula {start + 1}…", flush=True)
        for _ in range(start):
            sample_instance(rng)
        if csv_path and os.path.exists(csv_path):
            with open(csv_path, newline="") as f:
                rows = list(csv.DictReader(f))

    t0 = time.time()

    for i in range(start, n):
        formula, ops, x_fit, y_fit, x_dest, y_clip, x_test, y_test = sample_instance(rng)
        tags = formula_tags(formula, ops)
        true_ops = ops

        scores, _, absent = destroi.predict(
            x_dest[:, 0], x_dest[:, 1], y_clip, threshold=THRESHOLD
        )
        acc = destroi_accuracy(scores, true_ops)
        blocked = sorted(set(absent) & DESTROI_OPS)
        forbidden = [OP_TOKEN[op] for op in blocked if op in OP_TOKEN]

        try:
            est = make_estimator(torch_model, n_trees=n_trees)
            r2_t, pred_t = fit_r2(est, x_fit, y_fit, x_test, y_test)
            est2 = make_estimator(torch_model, n_trees=n_trees)
            r2_c, pred_c = fit_r2(est2, x_fit, y_fit, x_test, y_test, forbidden=forbidden)
        except Exception as e:
            r2_t = r2_c = float("nan")
            pred_t = pred_c = f"ERROR: {e}"

        row = {
            "id": i,
            "formula": formula,
            "ops": tags["ops"],
            "destroi_acc": acc,
            "blocked": ",".join(blocked),
            "r2_trans": r2_t,
            "r2_comb": r2_c,
            "delta_r2": r2_c - r2_t if np.isfinite(r2_t) and np.isfinite(r2_c) else float("nan"),
            "pred_trans": pred_t,
            "pred_comb": pred_c,
            **tags,
        }
        rows.append(row)
        if csv_path:
            save_rows(csv_path, rows)

        elapsed = time.time() - t0
        done = i - start + 1
        eta = elapsed / done * (n - i - 1)
        print(
            f"  [{i+1:3d}/{n}] acc={acc:.0%}  "
            f"Trans={r2_t:7.4f}  Comb={r2_c:7.4f}  Δ={row['delta_r2']:+.4f}  "
            f"blocked={blocked or 'none'}  (~{eta:.0f}s left)",
            flush=True,
        )

    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-trees", type=int, default=50)
    p.add_argument("--start", type=int, default=0, help="Resume from formula index (0-based)")
    args = p.parse_args()

    ensure_dirs()
    tag = f"n{args.n}_seed{args.seed}"
    csv_path = os.path.join(THREE_WAY, f"three_way_benchmark_{tag}.csv")
    sum_path = os.path.join(THREE_WAY, f"three_way_benchmark_summary_{tag}.txt")

    rows = run(args.n, args.seed, args.n_trees, start=args.start, csv_path=csv_path)

    summary = summarize(rows)
    with open(sum_path, "w") as f:
        f.write(summary)

    print(f"\nSaved → {csv_path}")
    print(f"Saved → {sum_path}\n")
    print(summary)


if __name__ == "__main__":
    main()
