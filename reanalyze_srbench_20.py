"""
Re-analyze the existing SRBench 20 benchmark run with fair DeSTrOI metrics.

No rerun required — reads results/srbench/srbench_20_benchmark.csv as-is.

Two corrections applied:
  1. Operator mapping: sub→{add,mul}, div→{inv,mul}, pow→{mul} when computing
     DeSTrOI accuracy, so the ground-truth label set matches DeSTrOI's 6-op vocab.
  2. Expressible-16 subset: exclude the 4 formulas that require exp or cos,
     which genuinely cannot be expressed in the 6-op grammar.

Output:
    results/srbench/srbench_20_benchmark_summary.txt  (overwritten with new analysis)
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from results_paths import SRBENCH
from srbench_destroi_ops import (
    DESTROI_OPS,
    gt_destroi_ops_mapped,
    is_destroi_expressible,
    irreducible_ops,
    parse_ops,
)

CSV_PATH = os.path.join(SRBENCH, "srbench_20_benchmark.csv")
SUM_PATH = os.path.join(SRBENCH, "srbench_20_benchmark_summary.txt")

THRESHOLD = 0.5
R2_GOOD = 0.99
R2_OK = 0.95
DESTROI_OPS_LIST = sorted(DESTROI_OPS)   # add, inv, log, mul, sin, sqrt


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_scores(scores_str: str) -> dict[str, float]:
    """'add:0.97;mul:1.00;...' → {'add': 0.97, 'mul': 1.0, ...}"""
    result = {}
    for part in scores_str.split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            result[k.strip()] = float(v.strip())
    return result


def mapped_accuracy(scores: dict[str, float], gt_ops_str: str) -> float:
    """
    Score DeSTrOI against the mapped ground truth.

    For each of the 6 DeSTrOI ops, a prediction is correct when:
      - score >= 0.5 and the op appears in the mapped gt set, OR
      - score <  0.5 and the op does NOT appear in the mapped gt set.
    """
    true_ops = gt_destroi_ops_mapped(gt_ops_str)
    correct = sum(
        (scores.get(op, 0.0) >= THRESHOLD) == (op in true_ops)
        for op in DESTROI_OPS_LIST
    )
    return correct / len(DESTROI_OPS_LIST)


def mapped_blocked(scores: dict[str, float], gt_ops_str: str) -> list[str]:
    """Ops DeSTrOI would correctly block under the mapped scheme."""
    true_ops = gt_destroi_ops_mapped(gt_ops_str)
    # 'absent' = predicted below threshold AND genuinely not in mapped gt
    absent_and_correct = [
        op for op in DESTROI_OPS_LIST
        if scores.get(op, 0.0) < THRESHOLD and op not in true_ops
    ]
    return sorted(absent_and_correct)


def section(title: str, rows_e2e: list[dict], rows_comb: list[dict],
            acc_key: str = "destroi_acc") -> list[str]:
    lines: list[str] = []
    n = len(rows_e2e)

    for mrows, label in [(rows_e2e, "Transformer alone"),
                         (rows_comb, "DeSTrOI + Transformer")]:
        r2 = np.array([float(r["r2_test"]) for r in mrows])
        r2v = r2[np.isfinite(r2)]
        lines += [
            f"--- {label} ---",
            f"  Mean R²:   {r2v.mean():.4f}",
            f"  Median R²: {np.median(r2v):.4f}",
            f"  R² ≥ 0.99: {(r2v >= R2_GOOD).sum()} / {n}",
            f"  R² ≥ 0.95: {(r2v >= R2_OK).sum()} / {n}",
            "",
        ]

    e2e_map  = {r["problem"]: float(r["r2_test"]) for r in rows_e2e}
    comb_map = {r["problem"]: float(r["r2_test"]) for r in rows_comb}
    deltas = [
        comb_map[p] - e2e_map[p]
        for p in e2e_map
        if p in comb_map and np.isfinite(e2e_map[p]) and np.isfinite(comb_map[p])
    ]
    if deltas:
        d = np.array(deltas)
        lines += [
            "--- Head-to-head (DeSTrOI+E2E minus E2E) ---",
            f"  Mean ΔR²:          {d.mean():+.4f}",
            f"  Better (Δ> 0.005): {(d > 0.005).sum()}",
            f"  Worse  (Δ<-0.005): {(d < -0.005).sum()}",
            f"  Similar:           {(np.abs(d) <= 0.005).sum()}",
            "",
        ]

    acc = [float(r[acc_key]) for r in rows_comb if r.get(acc_key)]
    if acc:
        lines.append(f"Mean DeSTrOI operator accuracy: {np.mean(acc):.1%}")
        lines.append("")

    lines.append("Per problem:")
    for r_e, r_c in zip(rows_e2e, rows_comb):
        et, ct = float(r_e["r2_test"]), float(r_c["r2_test"])
        delta = ct - et if np.isfinite(et) and np.isfinite(ct) else float("nan")
        acc_val = float(r_c[acc_key]) if r_c.get(acc_key) else float("nan")
        lines.append(
            f"  {r_e['problem']:<28s}  E2E={et:7.4f}  Comb={ct:7.4f}  "
            f"Δ={delta:+.4f}  acc={acc_val:.0%}"
        )

    return lines


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    with open(CSV_PATH, newline="") as f:
        all_rows = list(csv.DictReader(f))

    # split by method, preserving original problem order
    e2e_rows  = [r for r in all_rows if r["method"] == "e2e"]
    comb_rows = [r for r in all_rows if r["method"] == "destroi_e2e"]

    # attach mapped accuracy to each combined row
    for r in comb_rows:
        scores = parse_scores(r.get("destroi_scores", ""))
        r["destroi_acc_mapped"] = mapped_accuracy(scores, r["gt_ops"])
        r["blocked_mapped"] = ",".join(mapped_blocked(scores, r["gt_ops"]))
        r["expressible"] = is_destroi_expressible(r["gt_ops"])

    # propagate expressibility flag to e2e rows too (for filtering)
    expr_set = {r["problem"] for r in comb_rows if r["expressible"]}
    excluded  = {r["problem"] for r in comb_rows if not r["expressible"]}

    e2e_16  = [r for r in e2e_rows  if r["problem"] in expr_set]
    comb_16 = [r for r in comb_rows if r["problem"] in expr_set]

    # ── build summary ─────────────────────────────────────────────────────────
    lines: list[str] = [
        "SRBench 20-problem benchmark — re-analysis with fair DeSTrOI metrics",
        "=" * 68,
        "",
        "Two corrections vs the original summary:",
        "  1. Operator mapping: sub→{add,mul}, div→{inv,mul}, pow→{mul}",
        "     when computing DeSTrOI accuracy against ground truth.",
        "  2. Expressible-16 subset: 4 formulas that require exp or cos",
        "     (genuinely outside the 6-op grammar) reported separately.",
        "",
        f"Excluded (irreducible ops): {sorted(excluded)}",
    ]
    for p in sorted(excluded):
        r = next(r for r in comb_rows if r["problem"] == p)
        lines.append(f"  {p}: irreducible={sorted(irreducible_ops(r['gt_ops']))}")
    lines += ["", ""]

    # ── SECTION A: all 20, original accuracy (unchanged) ──────────────────────
    lines += [
        "══ ALL 20 PROBLEMS — original accuracy (literal DeSTrOI-6 labels) ══",
        "",
    ]
    lines += section("all-20 original", e2e_rows, comb_rows, acc_key="destroi_acc")

    lines += ["", ""]

    # ── SECTION B: all 20, mapped accuracy ────────────────────────────────────
    lines += [
        "══ ALL 20 PROBLEMS — mapped accuracy (sub/div/pow rewritten) ════════",
        "",
        "Accuracy now measures: for each of DeSTrOI's 6 ops, was the",
        "present/absent prediction correct after rewriting GT into 6-op vocab?",
        "",
    ]
    lines += section("all-20 mapped", e2e_rows, comb_rows, acc_key="destroi_acc_mapped")

    lines += ["", ""]

    # ── SECTION C: expressible-16, mapped accuracy ────────────────────────────
    lines += [
        "══ EXPRESSIBLE-16 SUBSET — mapped accuracy ═══════════════════════════",
        "",
        "Formulas whose ground truth contains ONLY ops rewritable to {add, mul,",
        "inv, sqrt, log, sin}. This is the fairest test of DeSTrOI on SRBench.",
        "",
    ]
    lines += section("expr-16 mapped", e2e_16, comb_16, acc_key="destroi_acc_mapped")

    # ── by benchmark group (expressible-16) ───────────────────────────────────
    lines += ["", "--- By SRBench group (Expressible-16, E2E) ---"]
    for bench in ("feynman", "strogatz"):
        sub = [r for r in e2e_16 if r.get("benchmark") == bench]
        if not sub:
            continue
        r2v = np.array([float(r["r2_test"]) for r in sub])
        r2v = r2v[np.isfinite(r2v)]
        rate = 100 * (r2v >= R2_GOOD).mean()
        lines.append(
            f"  {bench:<10s}  n={len(r2v):2d}  "
            f"R²≥0.99: {(r2v >= R2_GOOD).sum()}/{len(r2v)} ({rate:.0f}%)  "
            f"mean R²={r2v.mean():.3f}"
        )

    lines += [
        "",
        "NOTE: Published Kamienny rates (Feynman 84.8%, Strogatz 35.7%) are over",
        "all 119+14 problems — not comparable to this 20-problem pilot.",
        "",
        "KEY TAKEAWAY:",
        "  • Original acc 57.5% was misleading — penalised DeSTrOI for not",
        "    detecting 'div'/'sub'/'pow' that are not in its vocabulary.",
        "  • Mapped acc on expressible-16 gives the honest operator-ID score.",
        "  • R² results are unchanged (no rerun); only interpretation improves.",
    ]

    summary = "\n".join(lines)
    with open(SUM_PATH, "w") as f:
        f.write(summary)
    print(summary)
    print(f"\nSaved → {SUM_PATH}")


if __name__ == "__main__":
    main()
