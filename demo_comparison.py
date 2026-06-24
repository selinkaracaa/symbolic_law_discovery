"""
Three-way comparison: DeSTrOI alone  |  Transformer alone  |  Combined pipeline

Output: results/three_way/figures/three_way_comparison.png

Run:  python3 demo_comparison.py
"""

import sys, os, time, warnings
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

# ── Load Transformer ──────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(ROOT, "symbolicregression", "model.pt")
print("\nLoading NeurIPS Transformer…", end=" ", flush=True)
torch_model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
print("done.")

def make_est():
    return symbolicregression.model.SymbolicTransformerRegressor(
        model=torch_model, max_input_points=200, n_trees_to_refine=100, rescale=True
    )

OP_TOKEN = {
    "add": 42, "sub": 61, "mul": 53, "div": 47, "pow": 55,
    "inv": 51, "sqrt": 60, "log": 52, "exp": 50,
    "sin": 59, "cos": 46, "tan": 62, "abs": 41,
}
DESTROI_OPS = set(DESTROI_OPS_LIST)

DEMOS = [
    ("sin(x₁)·√|x₂|",       lambda x: np.sin(x[:,0]) * np.sqrt(np.abs(x[:,1]) + 1e-5),
     {"sin", "mul", "sqrt"}),
    ("log(|x₁|) + x₂",       lambda x: np.log(np.abs(x[:,0]) + 1e-5) + x[:,1],
     {"log", "add"}),   # abs not in DeSTrOI vocab
    ("sin(2πx₁) + x₂²",     lambda x: np.sin(2*np.pi*x[:,0]) + x[:,1]**2,
     {"sin", "add", "mul", "pow"}),
    ("x₁/(x₁²+1) + x₂",     lambda x: x[:,0] / (x[:,0]**2 + 1) + x[:,1],
     {"add", "div", "pow", "mul"}),
]

REPLACE_SYMPY = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
N, N_FIT = 500, 200
THRESHOLD = 0.5

BG, PANEL = "#0d1117", "#161b22"
C_DEST, C_TRANS, C_COMB = "#a371f7", "#58a6ff", "#3fb950"
C_TP, C_TN, C_FP, C_FN = "#3fb950", "#58a6ff", "#e3b341", "#f85149"


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


def destroi_accuracy(scores, true_ops, threshold=THRESHOLD):
    """Accuracy on DeSTrOI's 6 operators (only ops DeSTrOI knows)."""
    true_in_vocab = {op for op in true_ops if op in DESTROI_OPS}
    correct, total = 0, len(DESTROI_OPS_LIST)
    details = {}
    for op in DESTROI_OPS_LIST:
        pred = scores[op] >= threshold
        actual = op in true_in_vocab
        details[op] = "TP" if pred and actual else "TN" if not pred and not actual else "FP" if pred and not actual else "FN"
        if pred == actual:
            correct += 1
    return correct / total, details


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
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    style_ax(ax, f"{title}\nR² = {r2:.4f}", color=color)
    ax.set_xlabel("True y", color="#8b949e", fontsize=7)
    ax.set_ylabel("Pred y", color="#8b949e", fontsize=7)


# ── Run all experiments ───────────────────────────────────────────────────────
print("Running experiments…")
results = []

for name, fn, true_ops in DEMOS:
    print(f"  {name}")
    rng = np.random.default_rng(42 + len(results))
    x_full = rng.uniform(-8, 8, (N, 2))
    y_full = fn(x_full)
    clip = np.percentile(np.abs(y_full), 98)
    y_clip = np.clip(y_full, -clip, clip)
    x_fit, y_fit = x_full[:N_FIT], y_full[:N_FIT]
    x_test = rng.uniform(-8, 8, (500, 2))
    y_test = fn(x_test)

    scores, present, absent = destroi_predict(
        x_full[:, 0], x_full[:, 1], y_clip, threshold=THRESHOLD, verbose=False
    )
    acc, op_details = destroi_accuracy(scores, true_ops)
    blocked = sorted(set(absent) & DESTROI_OPS)
    forbidden = [OP_TOKEN[op] for op in blocked if op in OP_TOKEN]

    est = make_est()
    t0 = time.time()
    est.fit(x_fit, y_fit)
    r2_t, yp_t = get_r2(est.retrieve_tree(with_infos=True)["relabed_predicted_tree"].infix(), x_test, y_test)
    t_t = time.time() - t0

    est2 = make_est()
    t0 = time.time()
    est2.fit(x_fit, y_fit, forbidden_token_ids=forbidden)
    r2_c, yp_c = get_r2(est2.retrieve_tree(with_infos=True)["relabed_predicted_tree"].infix(), x_test, y_test)
    t_c = time.time() - t0

    results.append(dict(
        name=name, true_ops=true_ops, scores=scores, op_details=op_details,
        destroi_acc=acc, present=present, blocked=blocked,
        r2_t=r2_t, r2_c=r2_c, y_test=y_test, yp_t=yp_t, yp_c=yp_c,
    ))

# ── Build figure ──────────────────────────────────────────────────────────────
ensure_dirs()

n = len(results)
fig = plt.figure(figsize=(20, 3.2 + n * 4.8))
fig.patch.set_facecolor(BG)

outer = gridspec.GridSpec(
    1 + n, 1, figure=fig, height_ratios=[1.6] + [1] * n, hspace=0.45
)

# ── Header / explanation panel ──────────────────────────────────────────────
ax_hdr = fig.add_subplot(outer[0, 0])
ax_hdr.set_facecolor(PANEL)
ax_hdr.axis("off")

title = "Symbolic Regression: Three-Way Comparison"
ax_hdr.text(0.5, 0.97, title, ha="center", va="top", color="white",
            fontsize=16, fontweight="bold", transform=ax_hdr.transAxes)

cols = [
    (0.02, C_DEST, "① DeSTrOI alone",
     "Input: (x₁, x₂, y) data points\n"
     "Process: data → 50×50 image → super-resolution → CNN\n"
     "Output: operator probabilities (add, mul, sin, …)\n"
     "Does NOT predict the formula — only which operators appear\n"
     "Paper accuracy: ~90%+ on their benchmark"),
    (0.35, C_TRANS, "② Transformer alone",
     "Input: 200 data points\n"
     "Process: token-by-token formula generation (beam search)\n"
     "Output: full formula (often messy constants)\n"
     "Searches ALL operators at every step\n"
     "Strong R² but can miss unusual formulas"),
    (0.68, C_COMB, "③ Combined (DeSTrOI → Transformer)",
     "Step 1: DeSTrOI predicts operator probabilities\n"
     "Step 2: block low-probability operators from Transformer vocabulary\n"
     "Output: formula with smaller search space\n"
     "Can improve hard cases (log, div) but hard blocking is risky"),
]
for x, col, hdr, body in cols:
    ax_hdr.add_patch(mpatches.FancyBboxPatch(
        (x + 0.005, 0.05), 0.30, 0.82, boxstyle="round,pad=0.01",
        facecolor="#21262d", edgecolor=col, linewidth=1.5,
        transform=ax_hdr.transAxes, clip_on=False))
    ax_hdr.text(x + 0.02, 0.82, hdr, color=col, fontsize=10, fontweight="bold",
                transform=ax_hdr.transAxes, va="top")
    ax_hdr.text(x + 0.02, 0.72, body, color="#c9d1d9", fontsize=8,
                transform=ax_hdr.transAxes, va="top", linespacing=1.45)

legend_items = [
    mpatches.Patch(color=C_TP, label="TP — correctly detected"),
    mpatches.Patch(color=C_TN, label="TN — correctly absent"),
    mpatches.Patch(color=C_FP, label="FP — false alarm"),
    mpatches.Patch(color=C_FN, label="FN — missed operator"),
]
ax_hdr.legend(handles=legend_items, loc="lower center", ncol=4,
              frameon=False, labelcolor="#8b949e", fontsize=8,
              bbox_to_anchor=(0.5, -0.02))

# ── Per-formula rows ──────────────────────────────────────────────────────────
for i, res in enumerate(results):
    gs_row = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer[i + 1, 0], width_ratios=[1.1, 0.9, 0.9, 0.85], wspace=0.35
    )

    # Col 0 — DeSTrOI
    ax_d = fig.add_subplot(gs_row[0, 0])
    ops = DESTROI_OPS_LIST
    vals = [res["scores"][o] for o in ops]
    bar_colors = []
    for o, v in zip(ops, vals):
        d = res["op_details"][o]
        bar_colors.append({"TP": C_TP, "TN": C_TN, "FP": C_FP, "FN": C_FN}[d])
    ax_d.bar(ops, vals, color=bar_colors, edgecolor="#30363d")
    ax_d.axhline(THRESHOLD, color="white", lw=0.7, ls="--", alpha=0.4)
    ax_d.set_ylim(0, 1.12)
    for j, (o, v) in enumerate(zip(ops, vals)):
        ax_d.text(j, v + 0.02, f"{v:.2f}", ha="center", color="white", fontsize=7)
    true_str = ", ".join(sorted(res["true_ops"]))
    style_ax(ax_d, f"① DeSTrOI  ·  {res['name']}\n"
             f"Operator ID accuracy: {res['destroi_acc']:.0%}  (true: {true_str})",
             color=C_DEST, fs=8)
    ax_d.set_ylabel("Probability", color="#8b949e", fontsize=7)

    # Col 1 — Transformer alone
    ax_t = fig.add_subplot(gs_row[0, 1])
    scatter_panel(ax_t, res["y_test"], res["yp_t"], res["r2_t"],
                  "② Transformer alone", C_TRANS)

    # Col 2 — Combined
    ax_c = fig.add_subplot(gs_row[0, 2])
    scatter_panel(ax_c, res["y_test"], res["yp_c"], res["r2_c"],
                  "③ Combined", C_COMB)

    # Col 3 — summary bars + delta
    ax_s = fig.add_subplot(gs_row[0, 3])
    ax_s.set_facecolor(PANEL)
    labels = ["DeSTrOI\n(op ID)", "Transformer\n(R²)", "Combined\n(R²)"]
    # DeSTrOI: show accuracy; Transformer/Combined: show R²
    vals_s = [res["destroi_acc"], max(0, res["r2_t"]), max(0, res["r2_c"])]
    cols_s = [C_DEST, C_TRANS, C_COMB]
    bars = ax_s.bar(labels, vals_s, color=cols_s, edgecolor="#30363d", width=0.55)
    ax_s.set_ylim(0, 1.15)
    for b, v in zip(bars, vals_s):
        ax_s.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                  ha="center", color="white", fontsize=8, fontweight="bold")
    delta = res["r2_c"] - res["r2_t"]
    sign = "+" if delta >= 0 else ""
    verdict = "Combined better" if delta > 0.005 else "Combined worse" if delta < -0.005 else "Similar"
    vcol = C_COMB if delta > 0.005 else C_FN if delta < -0.005 else "#8b949e"
    blocked = ", ".join(res["blocked"]) if res["blocked"] else "none"
    ax_s.text(0.5, 0.28, f"ΔR² = {sign}{delta:.4f}\n{verdict}",
              ha="center", transform=ax_s.transAxes, color=vcol, fontsize=9, fontweight="bold")
    ax_s.text(0.5, 0.10, f"Blocked: {blocked}",
              ha="center", transform=ax_s.transAxes, color="#f85149", fontsize=7)
    style_ax(ax_s, "Summary", fs=9)
    ax_s.set_ylabel("Score", color="#8b949e", fontsize=8)

fig.text(
    0.5, 0.005,
    "Note: DeSTrOI score = operator identification accuracy (6 ops). Transformer/Combined score = R² on held-out test data. "
    "DeSTrOI does not output a formula — it only constrains the Transformer's search.",
    ha="center", color="#8b949e", fontsize=8,
)

out = os.path.join(THREE_WAY_FIGURES, "three_way_comparison.png")
plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

print(f"\nSaved → {out}\n")
print(f"{'Formula':<22} {'DeSTrOI acc':>11} {'Trans R²':>10} {'Comb R²':>10} {'ΔR²':>8}")
print("─" * 65)
for r in results:
    d = r["r2_c"] - r["r2_t"]
    print(f"{r['name']:<22} {r['destroi_acc']:>10.0%} {r['r2_t']:>10.4f} {r['r2_c']:>10.4f} {d:>+8.4f}")
