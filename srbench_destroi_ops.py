"""
Map SRBench ground-truth operator tags to DeSTrOI's 6-op vocabulary.

SRBench surface syntax uses div/sub/pow/exp/cos; DeSTrOI was trained on
add, mul, inv, sqrt, log, sin.  Rewrites used for fair operator-ID scoring:

    sub → add + mul
    div → inv + mul
    pow → mul   (integer powers, e.g. x**2)

Formulas with exp, cos, tan, or abs are not expressible in the 6-op grammar.
"""

from __future__ import annotations

DESTROI_OPS = frozenset({"add", "mul", "inv", "sqrt", "log", "sin"})

# SRBench tags that cannot be written with DeSTrOI's 6 operators
IRREDUCIBLE_OPS = frozenset({"exp", "cos", "tan", "abs"})


def parse_ops(ops_str: str) -> set[str]:
    if not ops_str:
        return set()
    return {o.strip() for o in ops_str.split(",") if o.strip()}


def gt_destroi_ops_literal(ops_str: str) -> set[str]:
    """Ops in ground truth that appear verbatim in DeSTrOI's vocabulary."""
    return parse_ops(ops_str) & DESTROI_OPS


def gt_destroi_ops_mapped(ops_str: str) -> set[str]:
    """Ground-truth ops rewritten into DeSTrOI's 6-op vocabulary."""
    mapped: set[str] = set()
    for op in parse_ops(ops_str):
        if op in DESTROI_OPS:
            mapped.add(op)
        elif op == "div":
            mapped |= {"inv", "mul"}
        elif op == "sub":
            mapped |= {"add", "mul"}
        elif op == "pow":
            mapped.add("mul")
        # irreducible ops contribute nothing; expressibility checked separately
    return mapped


def is_destroi_expressible(ops_str: str) -> bool:
    """True if every GT operator can be expressed with DeSTrOI's 6-op grammar."""
    ops = parse_ops(ops_str)
    if not ops:
        return True
    return not bool(ops & IRREDUCIBLE_OPS)


def irreducible_ops(ops_str: str) -> set[str]:
    return parse_ops(ops_str) & IRREDUCIBLE_OPS
