"""
Fair comparison: formulas built ONLY from DeSTrOI's 6 operators
(add, mul, inv, sqrt, log, sin) — matches the paper's vocabulary.

Keeps the original three_way_comparison.png unchanged.
Output: results/three_way/figures/three_way_comparison_destroi_vocab.png

Run:  python3 demo_comparison_destroi_vocab.py
"""

import sys, os, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "symbolicregression"))

import torch, numpy as np, sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import symbolicregression
from destroi_predict import destroi_predict, OPERATOR_LIST as DESTROI_OPS_LIST
from results_paths import THREE_WAY_FIGURES, ensure_dirs

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(ROOT, "symbolicregression", "model.pt")
OP_TOKEN = {
    "add": 42, "sub": 61, "mul": 53, "div": 47, "pow": 55,
    "inv": 51, "sqrt": 60, "log": 52, "exp": 50,
    "sin": 59, "cos": 46, "tan": 62, "abs": 41,
}
DESTROI_OPS = set(DESTROI_OPS_LIST)
REPLACE_SYMPY = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
N, N_FIT, THRESHOLD = 500, 200, 0.5
EPS = 1e-5

BG, PANEL = "#0d1117", "#161b22"
C_DEST, C_TRANS, C_COMB = "#a371f7", "#58a6ff", "#3fb950"
C_TP, C_TN, C_FP, C_FN = "#3fb950", "#58a6ff", "#e3b341", "#f85149"


def _inv(x):
    return 1.0 / (x + np.where(x >= 0, 1.0, -1.0) * EPS)

def _sqrt(x):
    return np.sqrt(np.abs(x))

def _log(x):
    return np.log(np.abs(x) + EPS)


DEMOS = [
    ("sin(x₁)·√|x₂|",      lambda x: np.sin(x[:, 0]) * _sqrt(x[:, 1]),
     {"mul", "sin", "sqrt"}, "mul(sin(X0),sqrt(X1))"),
    ("sin(x₁)+log(|x₂|)",   lambda x: np.sin(x[:, 0]) + _log(x[:, 1]),
     {"add", "sin", "log"}, "add(sin(X0),log(X1))"),
    ("1/(x₁·x₂)",           lambda x: _inv(x[:, 0] * x[:, 1]),
     {"inv", "mul"}, "inv(mul(X0,X1))"),
    ("x₁·x₂+sin(x₂)",       lambda x: x[:, 0] * x[:, 1] + np.sin(x[:, 1]),
     {"add", "mul", "sin"}, "add(mul(X0,X1),sin(X1))"),
    ("√(x₁²+x₂²)",          lambda x: _sqrt(x[:, 0] ** 2 + x[:, 1] ** 2),
     {"sqrt", "add", "mul"}, "sqrt(add(mul(X0,X0),mul(X1,X1)))"),
]


def get_r2(raw, x_test, y_test):
    x0s, x1s = sp.symbols("x_0 x_1")
    s = raw
    for op, rep in REPLACE_SYMPY.items():
        s = s.replace(op, rep)
    try:
        expr = sp.parse_expr(s, local_dict={"x_0": x0s, "x_1": x1s})
        f = sp.lambdify([x0s, x1s], expr, modules="numpy")
        yp = np.array(f(x_test[:, 0], x_test[:, 1]), dtype=float).flatten()
        m = np.isfinite(yp)
        if m.sum() < 10:
            return float("nan"), yp
        ss_r = np.sum((y_test[m] - yp[m]) ** 2)
        ss_t = np.sum((y_test[m] - y_test[m].mean()) ** 2)
        return float(1 - ss_r / ss_t) if ss_t > 0 else float("nan"), yp
    except Exception:
        return float("nan"), np.full(len(y_test), np.nan)


def destroi_accuracy(scores, true_ops):
    correct = 0
    details = {}
    for op in DESTROI_OPS_LIST:
        pred = scores[op] >= THRESHOLD
        actual = op in true_ops
        details[op] = "TP" if pred and actual else "TN" if not pred and not actual else "FP" if pred and not actual else "FN"
        if pred == actual:
            correct += 1
    return correct / len(DESTROI_OPS_LIST), details


def sample_data(fn, n, rng):
    pts = []
    while len(pts) < n:
        x = rng.uniform(-8, 8, (n * 4, 2))
        try:
            y = fn(x)
            ok = np.isfinite(y) & (np.abs(y) < 100)
            for xi, yi in zip(x[ok], y[ok]):
                pts.append((xi[0], xi[1], yi))
                if len(pts) >= n:
                    break
        except Exception:
            pass
    a = np.array(pts[:n])
    return a[:, :2], a[:, 2]


def style_ax(ax, title, color="white", fs=9):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=color, fontsize=fs, pad=5, fontweight="bold")
    ax.tick_params(colors="#8b949e", labelsize=7)
    for s in ax.spines.values():
        s.set_edgecolor("#30363d")


def scatter_panel(ax, y_true, y_pred, r2, title, color):
    lim = float(np.percentile(np.abs(y_true[np.isfinite(y_true)]), 95)) * 1.3
    m = np.isfinite(y_pred) & np.isfinite(y_true)
    if m.any():
        ax.scatter(y_true[m], y_pred[m], s=2, alpha=0.35, color=color)
    ax.plot([-lim, lim], [-lim, lim], "--", color="#f85149", lw=0.8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    style_ax(ax, f"{title}\nR² = {r2:.4f}", color=color)
    ax.set_xlabel("True y", color="#8b949e", fontsize=7)
    ax.set_ylabel("Pred y", color="#8b949e", fontsize=7)


def main():
    print("\nLoading NeurIPS Transformer…", end=" ", flush=True)
    torch_model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    print("done.")

    def make_est():
        return symbolicregression.model.SymbolicTransformerRegressor(
            model=torch_model, max_input_points=200, n_trees_to_refine=100, rescale=True
        )

    print("Running fair DeSTrOI-vocab experiments…")
    results = []

    for name, fn, true_ops, prefix in DEMOS:
        print(f"  {name}")
        rng = np.random.default_rng(42 + len(results))
        x_full, y_full = sample_data(fn, N, rng)
        y_clip = np.clip(y_full, *np.percentile(y_full, [1, 99]))
        x_fit, y_fit = x_full[:N_FIT], y_full[:N_FIT]
        x_test, y_test = sample_data(fn, 500, rng)

        scores, _, absent = destroi_predict(
            x_full[:, 0], x_full[:, 1], y_clip, threshold=THRESHOLD, verbose=False
        )
        acc, op_details = destroi_accuracy(scores, true_ops)
        blocked = sorted(set(absent) & DESTROI_OPS)
        forbidden = [OP_TOKEN[op] for op in blocked if op in OP_TOKEN]

        est = make_est()
        est.fit(x_fit, y_fit)
        r2_t, yp_t = get_r2(est.retrieve_tree(with_infos=True)["relabed_predicted_tree"].infix(), x_test, y_test)

        est2 = make_est()
        est2.fit(x_fit, y_fit, forbidden_token_ids=forbidden)
        r2_c, yp_c = get_r2(est2.retrieve_tree(with_infos=True)["relabed_predicted_tree"].infix(), x_test, y_test)

        results.append(dict(
            name=name, prefix=prefix, true_ops=true_ops, scores=scores,
            op_details=op_details, destroi_acc=acc, blocked=blocked,
            r2_t=r2_t, r2_c=r2_c, y_test=y_test, yp_t=yp_t, yp_c=yp_c,
        ))

    mean_acc = np.mean([r["destroi_acc"] for r in results])

    # ── Plot ──────────────────────────────────────────────────────────────────
    ensure_dirs()
    n = len(results)
    fig = plt.figure(figsize=(20, 3.4 + n * 4.8))
    fig.patch.set_facecolor(BG)
    outer = gridspec.GridSpec(1 + n, 1, figure=fig, height_ratios=[1.7] + [1] * n, hspace=0.45)

    ax_hdr = fig.add_subplot(outer[0, 0])
    ax_hdr.set_facecolor(PANEL)
    ax_hdr.axis("off")
    ax_hdr.text(0.5, 0.98, "Three-Way Comparison  ·  DeSTrOI Vocabulary Only",
                ha="center", va="top", color="white", fontsize=16, fontweight="bold",
                transform=ax_hdr.transAxes)
    ax_hdr.text(
        0.5, 0.90,
        f"All formulas use ONLY add, mul, inv, sqrt, log, sin.  "
        f"Mean DeSTrOI operator-ID accuracy: {mean_acc:.0%}  (paper: ~90%+ on their test set)",
        ha="center", va="top", color="#8b949e", fontsize=9, transform=ax_hdr.transAxes,
    )
    for x, col, hdr, body in [
        (0.02, C_DEST, "① DeSTrOI", "Predicts which of 6 operators appear."),
        (0.35, C_TRANS, "② Transformer alone", "Full formula, all 13 ops allowed."),
        (0.68, C_COMB, "③ Combined", "DeSTrOI blocks absent ops first."),
    ]:
        ax_hdr.add_patch(mpatches.FancyBboxPatch(
            (x + 0.005, 0.10), 0.30, 0.50, boxstyle="round,pad=0.01",
            facecolor="#21262d", edgecolor=col, linewidth=1.5, transform=ax_hdr.transAxes))
        ax_hdr.text(x + 0.02, 0.55, hdr, color=col, fontsize=10, fontweight="bold",
                    transform=ax_hdr.transAxes, va="top")
        ax_hdr.text(x + 0.02, 0.47, body, color="#c9d1d9", fontsize=8,
                    transform=ax_hdr.transAxes, va="top")
    ax_hdr.legend(
        handles=[mpatches.Patch(color=c, label=l) for c, l in
                 [(C_TP, "TP"), (C_TN, "TN"), (C_FP, "FP"), (C_FN, "FN")]],
        loc="lower center", ncol=4, frameon=False, labelcolor="#8b949e", fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
    )

    for i, res in enumerate(results):
        gs_row = gridspec.GridSpecFromSubplotSpec(
            1, 4, subplot_spec=outer[i + 1, 0], width_ratios=[1.1, 0.9, 0.9, 0.85], wspace=0.35
        )
        ax_d = fig.add_subplot(gs_row[0, 0])
        ops = DESTROI_OPS_LIST
        vals = [res["scores"][o] for o in ops]
        colors = [{"TP": C_TP, "TN": C_TN, "FP": C_FP, "FN": C_FN}[res["op_details"][o]] for o in ops]
        ax_d.bar(ops, vals, color=colors, edgecolor="#30363d")
        ax_d.axhline(THRESHOLD, color="white", lw=0.7, ls="--", alpha=0.4)
        ax_d.set_ylim(0, 1.12)
        for j, v in enumerate(vals):
            ax_d.text(j, v + 0.02, f"{v:.2f}", ha="center", color="white", fontsize=7)
        style_ax(ax_d, f"① DeSTrOI · {res['name']}\n"
                 f"Acc {res['destroi_acc']:.0%} | true: {', '.join(sorted(res['true_ops']))}\n"
                 f"{res['prefix']}", color=C_DEST, fs=7.5)

        scatter_panel(fig.add_subplot(gs_row[0, 1]), res["y_test"], res["yp_t"], res["r2_t"],
                      "② Transformer alone", C_TRANS)
        scatter_panel(fig.add_subplot(gs_row[0, 2]), res["y_test"], res["yp_c"], res["r2_c"],
                      "③ Combined", C_COMB)

        ax_s = fig.add_subplot(gs_row[0, 3])
        ax_s.set_facecolor(PANEL)
        labels = ["DeSTrOI\n(op ID)", "Transformer\n(R²)", "Combined\n(R²)"]
        vals_s = [res["destroi_acc"], max(0, res["r2_t"]), max(0, res["r2_c"])]
        bars = ax_s.bar(labels, vals_s, color=[C_DEST, C_TRANS, C_COMB], edgecolor="#30363d", width=0.55)
        ax_s.set_ylim(0, 1.15)
        for b, v in zip(bars, vals_s):
            ax_s.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                      ha="center", color="white", fontsize=8, fontweight="bold")
        delta = res["r2_c"] - res["r2_t"]
        verdict = "Combined better" if delta > 0.005 else "Combined worse" if delta < -0.005 else "Similar"
        vcol = C_COMB if delta > 0.005 else C_FN if delta < -0.005 else "#8b949e"
        ax_s.text(0.5, 0.28, f"ΔR² = {delta:+.4f}\n{verdict}",
                  ha="center", transform=ax_s.transAxes, color=vcol, fontsize=9, fontweight="bold")
        ax_s.text(0.5, 0.10, f"Blocked: {', '.join(res['blocked']) or 'none'}",
                  ha="center", transform=ax_s.transAxes, color="#f85149", fontsize=7)
        style_ax(ax_s, "Summary", fs=9)

    fig.text(0.5, 0.004,
             "Fair test: x² = mul(x,x). No pow, div, abs, or sub. Original mixed-vocab PNG kept separately.",
             ha="center", color="#8b949e", fontsize=8)

    out = os.path.join(THREE_WAY_FIGURES, "three_way_comparison_destroi_vocab.png")
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"\nSaved → {out}")
    print(f"Mean DeSTrOI accuracy: {mean_acc:.0%}\n")
    print(f"{'Formula':<24} {'DeSTrOI':>8} {'Trans R²':>10} {'Comb R²':>10} {'ΔR²':>8}")
    print("─" * 68)
    for r in results:
        print(f"{r['name']:<24} {r['destroi_acc']:>7.0%} {r['r2_t']:>10.4f} {r['r2_c']:>10.4f} "
              f"{r['r2_c']-r['r2_t']:>+8.4f}")


if __name__ == "__main__":
    main()
