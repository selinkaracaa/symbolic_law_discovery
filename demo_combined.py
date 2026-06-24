"""
Combined Pipeline Demo: DeSTrOI + NeurIPS 2022 Transformer
=============================================================

Step 1  (DeSTrOI — REAL pretrained weights):
         Encodes data as image → super-resolution → CNN → operator probabilities.

Step 2  (NeurIPS 2022 Transformer):
         Predict formula token by token.
         Run TWICE per formula:
           • Unrestricted — all operators allowed (baseline)
           • DeSTrOI-restricted — low-probability operators blocked

Output: results/combined/figures/combined_pipeline_real.png

Run:  python3 demo_combined.py   (from symbolic_law_discovery/)
"""

import sys, os, time, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "symbolicregression"))

import torch, numpy as np, sympy as sp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import symbolicregression

from destroi_predict import destroi_predict
from results_paths import COMBINED_FIGURES, ensure_dirs

# ── Load NeurIPS Transformer ──────────────────────────────────────────────────
MODEL_PATH = os.path.join(ROOT, "symbolicregression", "model.pt")
print("\n" + "─"*62)
print("  DeSTrOI + Transformer  ·  Real Combined Pipeline")
print("─"*62)
print("  Loading NeurIPS Transformer…", end=" ", flush=True)
torch_model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
print("done.")

def make_est():
    return symbolicregression.model.SymbolicTransformerRegressor(
        model=torch_model, max_input_points=200, n_trees_to_refine=100, rescale=True
    )

# ── Operator vocab: NeurIPS Transformer token IDs ────────────────────────────
OP_TOKEN = {
    "add": 42, "sub": 61, "mul": 53, "div": 47, "pow": 55,
    "inv": 51, "sqrt": 60, "log": 52, "exp": 50,
    "sin": 59, "cos": 46, "tan": 62, "abs": 41,
}
# DeSTrOI only knows about these 6 operators — only block within this set
DESTROI_OPS = {"add", "mul", "inv", "sqrt", "log", "sin"}

# ── Test formulas ─────────────────────────────────────────────────────────────
# (display name, data function, ground truth operators for reference only)
DEMOS = [
    (
        "sin(x₁) · √|x₂|",
        lambda x: np.sin(x[:,0]) * np.sqrt(np.abs(x[:,1]) + 1e-5),
        {"sin", "mul", "sqrt"},
    ),
    (
        "log(|x₁|)  +  x₂",
        lambda x: np.log(np.abs(x[:,0]) + 1e-5) + x[:,1],
        {"log", "abs", "add"},
    ),
    (
        "sin(2π x₁)  +  x₂²",
        lambda x: np.sin(2*np.pi*x[:,0]) + x[:,1]**2,
        {"add", "sin", "mul", "pow"},
    ),
    (
        "x₁  /  (x₁² + 1)  +  x₂",
        lambda x: x[:,0] / (x[:,0]**2 + 1) + x[:,1],
        {"div", "add", "pow", "mul"},
    ),
]

REPLACE_DISP  = {"add":"+","mul":"×","sub":"−","pow":"**","inv":"1/"}
REPLACE_SYMPY = {"add":"+","mul":"*","sub":"-","pow":"**","inv":"1/"}

N      = 500   # data points for DeSTrOI (needs denser coverage for image)
N_FIT  = 200   # data points for Transformer


def get_r2(raw_infix, x_test, y_test):
    x0s, x1s = sp.symbols("x_0 x_1")
    sympy_str = raw_infix
    for op, rep in REPLACE_SYMPY.items():
        sympy_str = sympy_str.replace(op, rep)
    try:
        expr   = sp.parse_expr(sympy_str, local_dict={"x_0": x0s, "x_1": x1s})
        f_eval = sp.lambdify([x0s, x1s], expr, modules="numpy")
        y_pred = np.array(f_eval(x_test[:,0], x_test[:,1]), dtype=float).flatten()
        mask   = np.isfinite(y_pred)
        if mask.sum() < 10:
            return float("nan"), np.full(len(y_test), np.nan)
        ss_res = np.sum((y_test[mask] - y_pred[mask])**2)
        ss_tot = np.sum((y_test[mask] - y_test[mask].mean())**2)
        r2     = float(1 - ss_res/ss_tot) if ss_tot > 0 else float("nan")
        return r2, y_pred
    except Exception:
        return float("nan"), np.full(len(y_test), np.nan)


# ── Run experiments ───────────────────────────────────────────────────────────
results = []

for name, fn, true_ops in DEMOS:
    print(f"\n{'─'*62}")
    print(f"  Formula : {name}")
    print(f"  True ops: {sorted(true_ops)}")

    rng    = np.random.default_rng(42)
    x_full = rng.uniform(-8, 8, (N, 2))
    y_full = fn(x_full)

    # clip extreme values for cleaner image
    clip = np.percentile(np.abs(y_full), 98)
    y_clip = np.clip(y_full, -clip, clip)

    # ── Step 1: DeSTrOI predicts operators ───────────────────────────────────
    print("  Running DeSTrOI…", end=" ", flush=True)
    t_d = time.time()
    scores, predicted_present, predicted_absent = destroi_predict(
        x_full[:,0], x_full[:,1], y_clip, threshold=0.5, verbose=False
    )
    t_d = time.time() - t_d
    print(f"done ({t_d:.1f}s)")

    # Only block operators DeSTrOI knows AND predicts as absent
    ops_to_block  = set(predicted_absent) & DESTROI_OPS
    forbidden_ids = [OP_TOKEN[op] for op in ops_to_block if op in OP_TOKEN]

    print(f"  DeSTrOI predicted present : {sorted(predicted_present)}")
    print(f"  DeSTrOI predicted absent  : {sorted(predicted_absent)}")
    print(f"  Blocking token IDs        : {sorted(ops_to_block)}")

    # Data for Transformer (smaller subset)
    x_fit  = x_full[:N_FIT]
    y_fit  = y_full[:N_FIT]
    x_test = rng.uniform(-8, 8, (500, 2))
    y_test = fn(x_test)

    # ── Step 2a: Unrestricted Transformer ────────────────────────────────────
    est_u = make_est()
    t0    = time.time()
    est_u.fit(x_fit, y_fit)
    t_u   = time.time() - t0
    raw_u = est_u.retrieve_tree(with_infos=True)["relabed_predicted_tree"].infix()
    r2_u, yp_u = get_r2(raw_u, x_test, y_test)
    pred_u = raw_u
    for op, rep in REPLACE_DISP.items():
        pred_u = pred_u.replace(op, rep)
    print(f"  [Unrestricted]  R²={r2_u:.4f}  t={t_u:.1f}s")

    # ── Step 2b: DeSTrOI-restricted Transformer ──────────────────────────────
    est_r = make_est()
    t0    = time.time()
    est_r.fit(x_fit, y_fit, forbidden_token_ids=forbidden_ids)
    t_r   = time.time() - t0
    raw_r = est_r.retrieve_tree(with_infos=True)["relabed_predicted_tree"].infix()
    r2_r, yp_r = get_r2(raw_r, x_test, y_test)
    pred_r = raw_r
    for op, rep in REPLACE_DISP.items():
        pred_r = pred_r.replace(op, rep)
    print(f"  [Restricted ]   R²={r2_r:.4f}  t={t_r:.1f}s")

    results.append({
        "name": name, "true_ops": true_ops,
        "scores": scores,
        "predicted_present": predicted_present,
        "predicted_absent": predicted_absent,
        "blocked": sorted(ops_to_block),
        "r2_u": r2_u, "r2_r": r2_r,
        "t_u": t_u,   "t_r": t_r,
        "yp_u": yp_u, "yp_r": yp_r, "y_test": y_test,
    })

# ── Plot ──────────────────────────────────────────────────────────────────────
ensure_dirs()

fig = plt.figure(figsize=(18, 5.0 * len(results)))
fig.patch.set_facecolor("#0d1117")
gs  = gridspec.GridSpec(len(results), 4, figure=fig, hspace=0.8, wspace=0.4)

def dark_ax(ax, title, fs=9):
    ax.set_facecolor("#161b22")
    ax.set_title(title, color="white", fontsize=fs, pad=6)
    ax.tick_params(colors="#8b949e", labelsize=7)
    for s in ax.spines.values():
        s.set_edgecolor("#30363d")

for row, res in enumerate(results):
    y_test = res["y_test"]
    lim    = float(np.percentile(np.abs(y_test[np.isfinite(y_test)]), 95)) * 1.3

    # col 0 — DeSTrOI operator scores bar chart
    ax0 = fig.add_subplot(gs[row, 0])
    ax0.set_facecolor("#161b22")
    ops    = list(res["scores"].keys())
    vals   = [res["scores"][o] for o in ops]
    colors = []
    for o, v in zip(ops, vals):
        if o in res["true_ops"] and v >= 0.5:
            colors.append("#3fb950")   # true positive
        elif o not in res["true_ops"] and v < 0.5:
            colors.append("#58a6ff")   # true negative
        elif o in res["true_ops"] and v < 0.5:
            colors.append("#f85149")   # false negative (missed)
        else:
            colors.append("#e3b341")   # false positive
    bars = ax0.bar(ops, vals, color=colors, edgecolor="#30363d")
    ax0.axhline(0.5, color="white", lw=0.8, linestyle="--", alpha=0.5)
    ax0.set_ylim(0, 1.15)
    for b, v in zip(bars, vals):
        ax0.text(b.get_x()+b.get_width()/2, v+0.03, f"{v:.2f}",
                 ha="center", color="white", fontsize=7)
    dark_ax(ax0, f"DeSTrOI predictions\n{res['name']}", fs=8)
    ax0.set_ylabel("Probability", color="#8b949e", fontsize=8)
    ax0.tick_params(axis="x", colors="white", labelsize=7)

    # col 1 — unrestricted scatter
    ax1 = fig.add_subplot(gs[row, 1])
    yp   = res["yp_u"]
    mask = np.isfinite(yp) & np.isfinite(y_test)
    if mask.any():
        ax1.scatter(y_test[mask], yp[mask], s=3, alpha=0.35, color="#58a6ff")
    ax1.plot([-lim, lim], [-lim, lim], "--", color="#f85149", lw=1)
    ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim)
    dark_ax(ax1, f"Unrestricted\nR²={res['r2_u']:.4f}")
    ax1.set_xlabel("True y", color="#8b949e", fontsize=8)
    ax1.set_ylabel("Pred y", color="#8b949e", fontsize=8)

    # col 2 — restricted scatter
    ax2 = fig.add_subplot(gs[row, 2])
    yp   = res["yp_r"]
    mask = np.isfinite(yp) & np.isfinite(y_test)
    if mask.any():
        ax2.scatter(y_test[mask], yp[mask], s=3, alpha=0.35, color="#3fb950")
    ax2.plot([-lim, lim], [-lim, lim], "--", color="#f85149", lw=1)
    ax2.set_xlim(-lim, lim); ax2.set_ylim(-lim, lim)
    dark_ax(ax2, f"DeSTrOI-restricted\nR²={res['r2_r']:.4f}")
    ax2.set_xlabel("True y", color="#8b949e", fontsize=8)
    ax2.set_ylabel("Pred y", color="#8b949e", fontsize=8)

    # col 3 — R² comparison + blocked ops
    ax3 = fig.add_subplot(gs[row, 3])
    ax3.set_facecolor("#161b22")
    bars3 = ax3.bar(["Unrestricted", "DeSTrOI\n+Transformer"],
                    [max(0, res["r2_u"]), max(0, res["r2_r"])],
                    color=["#58a6ff", "#3fb950"], edgecolor="#30363d", width=0.5)
    ax3.set_ylim(0, 1.18)
    for b, v in zip(bars3, [res["r2_u"], res["r2_r"]]):
        ax3.text(b.get_x()+b.get_width()/2, max(0,v)+0.02,
                 f"{v:.4f}", ha="center", color="white", fontsize=9, fontweight="bold")
    # annotate blocked ops
    blocked_str = ", ".join(res["blocked"]) if res["blocked"] else "none"
    ax3.text(0.5, 0.12, f"Blocked: {blocked_str}",
             ha="center", transform=ax3.transAxes,
             color="#f85149", fontsize=8)
    dark_ax(ax3, "R² comparison", fs=9)
    ax3.tick_params(axis="x", colors="white", labelsize=8)
    ax3.set_ylabel("R²", color="#8b949e", fontsize=8)

fig.suptitle(
    "DeSTrOI  +  NeurIPS 2022 Transformer  ·  Real End-to-End Pipeline\n"
    "DeSTrOI (pretrained CNN) predicts operators from data → Transformer uses restricted vocabulary\n"
    "Green bar = TP  |  Blue bar = TN  |  Red bar = FN (missed)  |  Gold bar = FP",
    color="white", fontsize=11, y=1.003
)

out = os.path.join(COMBINED_FIGURES, "combined_pipeline_real.png")
plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "─"*68)
print(f"  {'Formula':<28} {'R²_base':>8}  {'R²_restr':>8}  {'Δ R²':>8}  Blocked")
print("  " + "─"*64)
for res in results:
    delta = res["r2_r"] - res["r2_u"]
    sign  = "+" if delta >= 0 else ""
    print(f"  {res['name']:<28} {res['r2_u']:>8.4f}  {res['r2_r']:>8.4f}  "
          f"{sign}{delta:>7.4f}  {', '.join(res['blocked']) or 'none'}")
print(f"\n  Saved → {out}")
print("─"*68 + "\n")
