"""
SRBench ground-truth data loader (Feynman + Strogatz).

Datasets live in EpistasisLab/penn-ml-benchmarks (PMLB). Ground-truth formulas
are parsed from each problem's metadata.yaml. Tabular (X, y) data is cached
locally under datasets/srbench/cache/.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parent
SRBENCH_DIR = ROOT / "datasets" / "srbench"
PROBLEM_LIST = SRBENCH_DIR / "problem_list.json"
CACHE_DIR = SRBENCH_DIR / "cache"
PMLB_RAW = (
    "https://github.com/EpistasisLab/penn-ml-benchmarks/raw/master/datasets"
)
PMLB_META = (
    "https://raw.githubusercontent.com/EpistasisLab/penn-ml-benchmarks/master/datasets"
)

FORMULA_RE = re.compile(
    r"^\s*(?:[A-Za-z][A-Za-z0-9_']*|E_n)\s*=\s*(.+?)\s*$"
)
FUNC_OPS = (
    "sin", "cos", "tan", "cot", "sec", "csc",
    "arcsin", "arccos", "arctan",
    "exp", "log", "sqrt", "abs",
)
BINARY_OPS = ("**", "+", "-", "*", "/")


def problem_list() -> list[str]:
    with open(PROBLEM_LIST) as f:
        return json.load(f)


def benchmark_name(problem: str) -> str:
    if problem.startswith("feynman"):
        return "feynman"
    if problem.startswith("strogatz"):
        return "strogatz"
    return "other"


def _fetch(url: str, dest: Path, min_bytes: int = 100) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= min_bytes:
        if dest.suffix == ".gz" and dest.read_bytes()[:2] == b"\x1f\x8b":
            return
        if dest.suffix != ".gz":
            return
    resp = requests.get(url, allow_redirects=True, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def download_problem(problem: str) -> None:
    data_url = f"{PMLB_RAW}/{problem}/{problem}.tsv.gz"
    meta_url = f"{PMLB_META}/{problem}/metadata.yaml"
    _fetch(data_url, CACHE_DIR / problem / f"{problem}.tsv.gz", min_bytes=1000)
    _fetch(meta_url, CACHE_DIR / problem / "metadata.yaml", min_bytes=50)


def download_all(problems: list[str] | None = None) -> None:
    problems = problems or problem_list()
    for i, p in enumerate(problems, 1):
        print(f"  [{i:3d}/{len(problems)}] {p}", flush=True)
        download_problem(p)


def load_metadata(problem: str) -> dict:
    meta_path = CACHE_DIR / problem / "metadata.yaml"
    if not meta_path.exists():
        download_problem(problem)
    with open(meta_path) as f:
        return yaml.safe_load(f)


def extract_formula(meta: dict) -> str:
    desc = meta.get("description") or ""
    for line in desc.splitlines():
        m = FORMULA_RE.match(line)
        if m:
            return m.group(1).strip()
    return ""


def parenthesis_depth(formula: str) -> int:
    depth = max_depth = 0
    for ch in formula:
        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ")":
            depth = max(0, depth - 1)
    return max_depth


def count_operators(formula: str) -> dict:
    s = formula.lower()
    counts = {}
    for op in FUNC_OPS:
        counts[op] = len(re.findall(rf"\b{op}\b", s))
    counts["pow"] = s.count("**")
    counts["add"] = s.count("+")
    counts["sub"] = len(re.findall(r"(?<![eE])-(?!\d)", s))
    counts["mul"] = s.count("*") - 2 * counts["pow"]
    counts["div"] = s.count("/")
    present = {k for k, v in counts.items() if v > 0}
    return {
        "n_operators": sum(counts.values()),
        "nestedness": parenthesis_depth(formula),
        "ops": ",".join(sorted(present)),
        "has_trig": int(any(k in present for k in ("sin", "cos", "tan", "cot"))),
        "has_exp_log": int("exp" in present or "log" in present),
        "has_sqrt": int("sqrt" in present),
        "has_div": int("div" in present),
        "has_pow": int("pow" in present),
    }


def categorize_formula(formula: str, n_features: int, benchmark: str) -> dict:
    tags = count_operators(formula)
    n_ops = tags["n_operators"]
    if n_ops <= 3:
        complexity = "simple"
    elif n_ops <= 8:
        complexity = "medium"
    else:
        complexity = "complex"

    if tags["nestedness"] <= 2:
        nest_cat = "shallow"
    elif tags["nestedness"] <= 4:
        nest_cat = "medium"
    else:
        nest_cat = "deep"

    return {
        **tags,
        "n_features": n_features,
        "benchmark": benchmark,
        "complexity": complexity,
        "nest_category": nest_cat,
    }


def read_problem(problem: str) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    data_path = CACHE_DIR / problem / f"{problem}.tsv.gz"
    if not data_path.exists():
        download_problem(problem)
    meta = load_metadata(problem)
    df = pd.read_csv(data_path, sep="\t", compression="gzip")
    clean = {k: k.strip().replace(".", "_") for k in df.columns}
    df = df.rename(columns=clean)
    label = "target"
    feature_names = [c for c in df.columns if c != label]
    X = df[feature_names].values.astype(float)
    y = df[label].values.astype(float)
    formula = extract_formula(meta)
    bench = benchmark_name(problem)
    tags = categorize_formula(formula, len(feature_names), bench)
    info = {
        "problem": problem,
        "formula": formula,
        "benchmark": bench,
        "feature_names": feature_names,
        **tags,
    }
    return X, y, feature_names, info
