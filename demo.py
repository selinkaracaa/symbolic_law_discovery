"""
DeSTrOI Demo — self-contained, no pretrained weights needed.
Shows the encoding pipeline: formula → data → naive grid → ground truth image → operator labels.

Run with:  python3 demo.py
Output: results/destroi/figures/encoding_pipeline.png
"""

import math, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from results_paths import DESTROI_FIGURES, ensure_dirs
import matplotlib.gridspec as gridspec

# ── Operator definitions (from symbolic_tree.py) ──────────────────────────
OPERATOR_LIST = ['add', 'mul', 'inv', 'sqrt', 'log', 'sin']
OPERATOR_ARGS = {'add': 2, 'mul': 2, 'inv': 1, 'sqrt': 1, 'log': 1, 'sin': 1}
EPS = 1e-5

def eval_formula(sym, x1, x2):
    """Evaluate a prefix-notation formula string at (x1, x2)."""
    sym = sym.strip()
    # terminal
    paren = sym.find('(')
    if paren == -1:
        if sym.startswith('X'):
            return x1 if sym == 'X0' else x2
        return float(sym)
    op  = sym[:paren]
    arg = sym[paren+1:-1]
    # split arguments at top-level comma
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(arg):
        if ch == '(':  depth += 1
        if ch == ')':  depth -= 1
        if ch == ',' and depth == 0:
            parts.append(arg[start:i]); start = i+1
    parts.append(arg[start:])
    a = eval_formula(parts[0], x1, x2)
    b = eval_formula(parts[1], x1, x2) if len(parts) > 1 else None
    if op == 'add':  return a + b
    if op == 'mul':  return a * b
    if op == 'inv':  return 1.0 / (a + (1 if a >= 0 else -1) * EPS)
    if op == 'sqrt': return math.sqrt(abs(a))
    if op == 'log':  return math.log(abs(a) + EPS)
    if op == 'sin':  return math.sin(a)
    raise ValueError(f"Unknown op: {op}")

def get_labels(sym):
    """Return binary label vector for which operators appear in the formula."""
    labels = np.zeros(len(OPERATOR_LIST))
    for i, op in enumerate(OPERATOR_LIST):
        if op + '(' in sym:
            labels[i] = 1
    return labels

# ── Encoding functions (from prepare_dataset.py) ──────────────────────────
def points_to_image(x1s, x2s, ys, res=50, low=-10, high=10, iters=20):
    """Build naive low-res grid from scattered data points."""
    img   = np.zeros((res, res))
    count = np.zeros((res, res))
    for x, y, z in zip(x1s, x2s, ys):
        i = int((x - low) / (high - low) * res)
        j = int((y - low) / (high - low) * res)
        if 0 <= i < res and 0 <= j < res:
            img[i, j] += z
            count[i, j] += 1
    norm = np.divide(img, count, out=np.zeros_like(img), where=count != 0)
    for _ in range(iters):
        for i in range(res):
            for j in range(res):
                if count[i, j] == 0:
                    nbrs = []
                    if i > 0:     nbrs.append(norm[i-1, j])
                    if j > 0:     nbrs.append(norm[i,   j-1])
                    if i < res-1: nbrs.append(norm[i+1, j])
                    if j < res-1: nbrs.append(norm[i,   j+1])
                    if nbrs:      norm[i, j] = np.mean(nbrs)
    return norm

def tree_to_image(sym, res=200, low=-10, high=10):
    """Evaluate formula on a dense grid — the 'ground truth' image."""
    img = np.zeros((res, res))
    for i in range(res):
        for j in range(res):
            x1 = (i + 0.5) / res * (high - low) + low
            x2 = (j + 0.5) / res * (high - low) + low
            try:
                v = eval_formula(sym, x1, x2)
                img[i, j] = v if abs(v) < 1e6 else 0.0
            except Exception:
                pass
    return img

# ── Formulas to demonstrate ────────────────────────────────────────────────
DEMOS = [
    ("sin(x₁) · √|x₂|",   "mul(sin(X0),sqrt(X1))"),
    ("x₁ · log(|x₂|)",    "mul(X0,log(X1))"),
    ("1/(x₁ · x₂)",       "inv(mul(X0,X1))"),
    ("x₁ + sin(log(x₂))", "add(X0,sin(log(X1)))"),
    ("√(x₁² + x₂²)",      "sqrt(add(mul(X0,X0),mul(X1,X1)))"),
]

N  = 3000
LR = 50
HR = 200

ensure_dirs()

# ── Build figure ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 4.2 * len(DEMOS)))
fig.patch.set_facecolor('#0d1117')
gs  = gridspec.GridSpec(len(DEMOS), 4, figure=fig, hspace=0.6, wspace=0.35)

print(f"\n{'─'*60}")
print("  DeSTrOI  ·  Encoding Pipeline Demo")
print(f"{'─'*60}")

for row, (name, sym) in enumerate(DEMOS):
    labels  = get_labels(sym)
    present = [OPERATOR_LIST[i] for i, v in enumerate(labels) if v == 1]

    # Generate data
    rng     = np.random.default_rng(42 + row)
    x1s     = rng.uniform(-10, 10, N * 4)
    x2s     = rng.uniform(-10, 10, N * 4)
    ys_raw  = []
    for x1, x2 in zip(x1s, x2s):
        try:
            v = eval_formula(sym, x1, x2)
            if np.isfinite(v) and abs(v) < 100:
                ys_raw.append((x1, x2, v + rng.normal(0, 0.01)))
        except Exception:
            pass
        if len(ys_raw) >= N:
            break
    ys_raw = np.array(ys_raw[:N])
    x1d, x2d, yd = ys_raw[:,0], ys_raw[:,1], ys_raw[:,2]

    print(f"\n  Formula : {name}")
    print(f"  Symbolic: {sym}")
    print(f"  Data pts: {len(yd):,}")
    print(f"  Present : {present}")
    print(f"  Labels  : { {op: int(v) for op, v in zip(OPERATOR_LIST, labels)} }")

    naive = points_to_image(x1d, x2d, yd, res=LR)
    gt    = tree_to_image(sym, res=HR)

    vlo, vhi = np.percentile(yd, 5), np.percentile(yd, 95)

    def dark_ax(ax, title):
        ax.set_facecolor('#161b22')
        ax.set_title(title, color='white', fontsize=9, pad=6)
        ax.tick_params(colors='#8b949e', labelsize=7)
        for s in ax.spines.values():
            s.set_edgecolor('#30363d')

    # Scatter
    ax0 = fig.add_subplot(gs[row, 0])
    ax0.scatter(x1d, x2d, c=yd, cmap='RdBu_r', s=1.5,
                alpha=0.6, vmin=vlo, vmax=vhi)
    dark_ax(ax0, f"{name}\nRaw data ({len(yd):,} pts)")
    ax0.set_xlabel("x₁", color='#8b949e', fontsize=8)
    ax0.set_ylabel("x₂", color='#8b949e', fontsize=8)

    # Naive image
    ax1 = fig.add_subplot(gs[row, 1])
    ax1.imshow(naive.T, origin='lower', cmap='RdBu_r', aspect='auto',
               extent=[-10, 10, -10, 10])
    dark_ax(ax1, f"Naive encoding ({LR}×{LR})\nnoisy, gaps filled by avg")
    ax1.set_xlabel("x₁", color='#8b949e', fontsize=8)
    ax1.set_ylabel("x₂", color='#8b949e', fontsize=8)

    # Ground truth image
    ax2 = fig.add_subplot(gs[row, 2])
    gt_clip = np.clip(gt, np.percentile(gt, 2), np.percentile(gt, 98))
    ax2.imshow(gt_clip.T, origin='lower', cmap='RdBu_r', aspect='auto',
               extent=[-10, 10, -10, 10])
    dark_ax(ax2, f"Ground truth ({HR}×{HR})\ncomputed from formula")
    ax2.set_xlabel("x₁", color='#8b949e', fontsize=8)
    ax2.set_ylabel("x₂", color='#8b949e', fontsize=8)

    # Operator labels bar chart
    ax3 = fig.add_subplot(gs[row, 3])
    colors = ['#3fb950' if v == 1 else '#f85149' for v in labels]
    bars   = ax3.bar(OPERATOR_LIST, labels, color=colors,
                     edgecolor='#30363d', linewidth=0.8)
    ax3.set_ylim(-0.15, 1.45)
    dark_ax(ax3, "Operator labels\n(green = present, red = absent)")
    for bar, lv in zip(bars, labels):
        ax3.text(bar.get_x() + bar.get_width()/2, int(lv) + 0.07,
                 str(int(lv)), ha='center', va='bottom',
                 color='white', fontsize=10, fontweight='bold')

fig.suptitle(
    "DeSTrOI  ·  Encoding Pipeline\n"
    "For each formula: raw data  →  naive 50×50 grid  →  200×200 ground truth  →  operator labels\n"
    "The super-resolution network learns to go from the noisy naive image to the clean ground truth.",
    color='white', fontsize=11, y=1.002
)

out = os.path.join(DESTROI_FIGURES, "encoding_pipeline.png")
plt.savefig(out, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()

print(f"\n{'─'*60}")
print(f"  Saved → {out}")
print(f"{'─'*60}\n")
