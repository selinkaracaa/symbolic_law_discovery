"""
NeurIPS 2022 Transformer Demo — "End-to-end symbolic regression with transformers"
(Kamienny et al., 2022)

Run from the symbolicregression/ subdirectory:
    cd symbolicregression
    python3 ../demo_neurips.py

Output: results/transformer/figures/neurips_transformer.png
"""

import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "symbolicregression"))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Load pretrained model ─────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "symbolicregression", "model.pt")

print("\n" + "─" * 60)
print("  NeurIPS 2022 Transformer  ·  Loading model…")

import symbolicregression
from results_paths import TRANSFORMER_FIGURES, ensure_dirs
model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
est = symbolicregression.model.SymbolicTransformerRegressor(
    model=model,
    max_input_points=200,
    n_trees_to_refine=100,
    rescale=True,
)
print("  Model loaded!")
print("─" * 60)

# ── Test formulas ─────────────────────────────────────────────────────────────
DEMOS = [
    ("x₁²  +  x₂²",          lambda x: x[:,0]**2 + x[:,1]**2),
    ("sin(2π x₁)  +  x₂²",   lambda x: np.sin(2*np.pi*x[:,0]) + x[:,1]**2),
    ("x₁ · x₂",               lambda x: x[:,0] * x[:,1]),
    ("x₁  /  (x₂² + 1)",      lambda x: x[:,0] / (x[:,1]**2 + 1)),
    ("x₁²  −  x₁  +  0.5",   lambda x: x[:,0]**2 - x[:,0] + 0.5),
]

REPLACE       = {"add": "+", "mul": "×", "sub": "−", "pow": "**", "inv": "1/"}
REPLACE_SYMPY = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
N = 200

ensure_dirs()

fig = plt.figure(figsize=(14, 3.4 * len(DEMOS)))
fig.patch.set_facecolor("#0d1117")
gs = gridspec.GridSpec(len(DEMOS), 3, figure=fig, hspace=0.7, wspace=0.35,
                       width_ratios=[1.6, 1, 1.6])

def dark_ax(ax, title):
    ax.set_facecolor("#161b22")
    ax.set_title(title, color="white", fontsize=9, pad=6)
    ax.tick_params(colors="#8b949e", labelsize=7)
    for s in ax.spines.values():
        s.set_edgecolor("#30363d")

results = []

for row, (name, fn) in enumerate(DEMOS):
    rng = np.random.default_rng(42 + row)
    x = rng.standard_normal((N, 2))
    y = fn(x)

    print(f"\n  Formula : {name}")
    est.fit(x, y)
    info = est.retrieve_tree(with_infos=True)
    raw = info["relabed_predicted_tree"].infix()
    pred = raw
    for op, rep in REPLACE.items():
        pred = pred.replace(op, rep)
    sympy_str = raw
    for op, rep in REPLACE_SYMPY.items():
        sympy_str = sympy_str.replace(op, rep)

    # R² via sympy evaluation on held-out test set
    import sympy as sp
    x0s, x1s = sp.symbols("x_0 x_1")
    x_test = rng.standard_normal((500, 2))
    y_test  = fn(x_test)
    try:
        expr   = sp.parse_expr(sympy_str, local_dict={"x_0": x0s, "x_1": x1s})
        f_eval = sp.lambdify([x0s, x1s], expr, modules="numpy")
        y_pred = np.array(f_eval(x_test[:, 0], x_test[:, 1]), dtype=float).flatten()
        mask   = np.isfinite(y_pred)
        ss_res = np.sum((y_test[mask] - y_pred[mask])**2)
        ss_tot = np.sum((y_test[mask] - y_test[mask].mean())**2)
        r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    except Exception as exc:
        print(f"  (R² eval failed: {exc})")
        y_pred = np.full(len(y_test), np.nan)
        r2     = float("nan")

    print(f"  Predicted: {pred[:80]}")
    print(f"  R²       : {r2:.4f}")
    results.append((name, pred, r2))

    # --- scatter: true vs predicted ---
    ax0 = fig.add_subplot(gs[row, 0])
    lim = np.percentile(np.abs(y_test), 95) * 1.2
    ax0.scatter(y_test, y_pred, s=4, alpha=0.4, color="#58a6ff")
    ax0.plot([-lim, lim], [-lim, lim], "--", color="#f85149", lw=1)
    dark_ax(ax0, f"True vs Predicted\n{name}")
    ax0.set_xlabel("True y", color="#8b949e", fontsize=8)
    ax0.set_ylabel("Pred y", color="#8b949e", fontsize=8)
    ax0.set_xlim(-lim, lim); ax0.set_ylim(-lim, lim)

    # --- R² gauge ---
    ax1 = fig.add_subplot(gs[row, 1])
    r2c  = max(0, min(1, r2))
    ax1.barh(0, r2c, color="#3fb950" if r2c > 0.95 else "#f0883e", height=0.4)
    ax1.barh(0, 1 - r2c, left=r2c, color="#21262d", height=0.4)
    ax1.set_xlim(0, 1); ax1.set_yticks([])
    ax1.text(0.5, 0.65, f"R² = {r2:.4f}", ha="center", va="bottom",
             color="white", fontsize=14, fontweight="bold",
             transform=ax1.transAxes)
    dark_ax(ax1, "Fit quality")

    # --- predicted formula text ---
    ax2 = fig.add_subplot(gs[row, 2])
    ax2.set_facecolor("#161b22")
    ax2.axis("off")
    wrapped = pred[:120] + ("…" if len(pred) > 120 else "")
    ax2.text(0.5, 0.6, "Predicted formula:", ha="center", va="center",
             color="#8b949e", fontsize=9, transform=ax2.transAxes)
    ax2.text(0.5, 0.35, wrapped, ha="center", va="center",
             color="#e6edf3", fontsize=8, transform=ax2.transAxes,
             wrap=True, fontfamily="monospace")
    ax2.set_title("Output", color="white", fontsize=9, pad=6)
    for s in ax2.spines.values():
        s.set_edgecolor("#30363d")

fig.suptitle(
    "NeurIPS 2022 · End-to-end Symbolic Regression with Transformers\n"
    "Data (x₁, x₂) → formula  |  no operator set given  |  pretrained model",
    color="white", fontsize=11, y=1.003
)

out = os.path.join(TRANSFORMER_FIGURES, "neurips_transformer.png")
plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

print("\n" + "─" * 60)
print(f"  Saved → {out}")
print("─" * 60)
print("\n  Summary")
print(f"  {'Formula':<30} {'R²':>8}  Predicted (truncated)")
print("  " + "─" * 70)
for nm, pr, r2 in results:
    print(f"  {nm:<30} {r2:>8.4f}  {pr[:40]}")
print()
