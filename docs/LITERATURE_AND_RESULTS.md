# Literature reviews & experimental results

Detailed paper summaries (TPSR-style) and **our** benchmark tables.  
**Repo:** [github.com/selinkaracaa/symbolic_law_discovery](https://github.com/selinkaracaa/symbolic_law_discovery)

---

## Table of contents

1. [Kamienny 2022 — E2E Transformer](#1-kamienny-2022--e2e-transformer)
2. [TPSR 2023](#2-tpsr-2023--transformer-based-planning)
3. [DGSR-MCTS 2023](#3-dgsr-mcts-2023)
4. [uDSR 2022](#4-udsr-2022)
5. [SR4MDL 2025](#5-sr4mdl-2025)
6. [DeSTrOI + our pipeline](#6-destroi--our-hybrid-pipeline)
7. [Cross-paper comparison](#7-cross-paper-comparison-table)
8. [Our experimental results](#8-our-experimental-results)
9. [SRBench taxonomy & data](#9-srbench-taxonomy--data)

---

## 1. Kamienny 2022 — E2E Transformer

**Paper:** *End-to-end symbolic regression with transformers* · NeurIPS 2022  
**Links:** [arXiv](https://arxiv.org/abs/2204.10532) · [Code](https://github.com/facebookresearch/symbolicregression) · **We use:** [`symbolicregression/`](../symbolicregression/)

### Problem (before this paper)

Classical SR (genetic programming) searches from scratch—slow, but can explore. No fast neural model could output a full formula in one pass.

### Proposed solution

Pre-train a **transformer** on millions of synthetic formulas. At test time: encode \((X,y)\) points → decode formula **token-by-token** → refine constants with BFGS.

### Method (step by step)

1. **Training:** Generate random expression trees (18 operators); model learns to predict the next token.
2. **Encoding:** Feed up to 200 \((x,y)\) pairs (with feature selection if \(D>10\)).
3. **Decoding:** Beam search produces candidate formulas.
4. **Refinement:** BFGS fits numeric constants on training data.
5. **Evaluation:** R² on held-out points (SRBench uses R² ≥ 0.99 as “solved”).

### Results (published on SRBench)

| Metric | Finding |
|--------|---------|
| Speed | **Much faster** than GP (Operon, etc.) |
| Complexity | **Lower** formula size than top GP methods |
| SRBench rank | ~**4th** overall by accuracy |
| Feynman | Strong R²≥0.99 rate |
| Strogatz | **Weak** (time-ordered ODE data ≠ training distribution) |
| Exact recovery | Limited—good fit ≠ correct formula |

### Limitations (why follow-up papers exist)

- **Blind decoding:** next token from LM probability only—no R² feedback mid-generation.
- **One shot:** no search if the first guess is wrong.
- **OOD:** performance drops on Strogatz and hard nested formulas.

---

## 2. TPSR 2023 — Transformer-based Planning

**Paper:** *Transformer-based Planning for Symbolic Regression* · NeurIPS 2023  
**Authors:** Shojaee, Meidani, Barati Farimani, Reddy  
**Links:** [arXiv](https://arxiv.org/abs/2303.06833) · [Code](https://github.com/deep-symbolic-mathematics/TPSR) · [SRBench results](https://github.com/deep-symbolic-mathematics/TPSR/tree/main/srbench_results)

### Problem with the plain transformer

**No performance feedback during creation.** The model builds equations symbol-by-symbol from token probabilities only. It does not check whether the partial equation fits the data or is unnecessarily complex until the **entire** equation is finished.

### Proposed solution

**TPSR** = Kamienny E2E transformer + **Monte Carlo Tree Search (MCTS)** at decode time. The transformer can **pause**, simulate how the formula might finish, **grade** candidates, then pick the best next symbol.

### Method (step by step)

Each time TPSR chooses the next symbol:

| Step | What happens |
|------|----------------|
| **Selection** | MCTS picks which partial equation branch to expand (balance exploit vs explore). |
| **Expansion** | Transformer proposes likely next tokens (not random). |
| **Evaluation** | Beam search **completes** partial equations for scoring. |
| **Reward** | Grade: **fitting accuracy** + **λ × complexity penalty**. |
| **Backprop** | Reward flows back to update which symbol choice was best. |

**λ (lambda):** λ=0 → accuracy only (bloated formulas); λ=1 → heavy simplicity penalty; **λ=0.1** recommended balance.

### Results (published SRBench)

#### Feynman physics

| Method | R² > 0.99 |
|--------|-----------|
| E2E Transformer | 84.8% |
| TPSR (λ=0.1) | **94.9%** |
| TPSR (λ=0) | 95.2% |

#### Strogatz ODEs (14 problems)

Time-ordered trajectories—distribution mismatch with E2E pre-training.

| Method | R² > 0.99 |
|--------|-----------|
| E2E Transformer | 35.7% |
| TPSR (λ=0) | 92.8% |
| TPSR (λ=0.1) | **82.8%** |

#### Black-box (122 problems)

| Method | Mean R² | Notes |
|--------|---------|-------|
| E2E Transformer | 0.864 | — |
| TPSR (λ=0.1) | **0.945** | Balanced |
| TPSR (λ=0) | higher fit | Avg complexity **129.85** (bloated) |

#### In-domain synthetic (400 formulas, same generator as E2E training)

| Method | Success | Avg complexity |
|--------|---------|----------------|
| E2E Transformer | 64.0% | — |
| TPSR (λ=0) | 70.2% | 67.11 |
| TPSR (λ=0.1) | **70.8%** | **40.31** |

### Takeaways

- **Better balance:** accuracy + interpretable short formulas.
- **Extrapolation:** strong on out-of-range data points.
- **Speed:** slower than greedy E2E, still faster than classical GP.
- **Same backbone** as Kamienny—improvement is **search**, not architecture.

---

## 3. DGSR-MCTS 2023

**Paper:** *Deep Generative Symbolic Regression with Monte-Carlo-Tree-Search* · ICML 2023  
**Authors:** Kamienny, Lample, Lamprier, Virgolin (**same lead author as E2E**)  
**Links:** [Proceedings](https://proceedings.mlr.press/v202/kamienny23a.html) · [Code](https://github.com/vanderschaarlab/DeepGenerativeSymbolicRegression)

### Problem

E2E transformer is **fast but weak out-of-distribution** without search. One forward pass is not enough on hard SRBench problems.

### Proposed solution

**DGSR-MCTS:** MCTS seeded with a pre-trained **mutation policy** (transformer-based). The policy is **fine-tuned online** from successful search trajectories.

### Method (step by step)

1. **Pre-train** mutation policy (like E2E) to propose expression mutations.
2. **MCTS loop:** select expression → mutate via policy → evaluate fit → backpropagate.
3. **Online update:** successful mutations improve the policy during search.
4. **Budget:** paper uses ~**500,000** expression evaluations per problem.

### Results (published)

- Claims **state-of-the-art on SRBench** at time of publication.
- Trades **speed** for **search depth** and OOD robustness.
- Official sequel from Meta team: *“transformer alone isn’t enough.”*

### vs Kamienny 2022

| | E2E | DGSR-MCTS |
|--|-----|-----------|
| Decode | One-shot | Many iterations |
| Weights at test | Fixed | Online updates |
| Speed | Fast | Slow |
| OOD / hard benchmarks | Weaker | Stronger |

---

## 4. uDSR 2022

**Paper:** *A Unified Framework for Deep Symbolic Regression* · NeurIPS 2022  
**Authors:** Landajuela et al. (LLNL / SRBench team)  
**Links:** [Paper](https://papers.nips.cc/paper_files/paper/2022/hash/dbca58f35bddc6e4003b2dd80e42f838-Abstract-Conference.html) · [Code](https://github.com/dso-org/deep-symbolic-optimization)

### Problem

Individual SR methods (GP, neural, pre-training) each fail on different problem types. No modular way to combine them.

### Proposed solution

**uDSR** = unified pipeline plugging in **5 strategies:**

| Module | Role |
|--------|------|
| Recursive simplification | Reduce problem size |
| Neural-guided search (DSR) | RL policy over expressions |
| **Large-scale pre-training (LSPT)** | Kamienny E2E transformer |
| Genetic programming | Population search |
| Linear models (`poly` token) | Fast coefficient fitting |

Kamienny’s transformer is **one interchangeable block**, not the whole system.

### Results (published SRBench)

- **Highest symbolic recovery** on ground-truth problems at time of publication.
- **1st place** GECCO 2022 SRBench real-world track.
- Black-box: on accuracy–complexity **Pareto frontier** with other top methods.

### Takeaway

*Don’t pick transformer OR GP—wire them together.*

---

## 5. SR4MDL 2025

**Paper:** *Symbolic regression via MDLformer-guided search* · ICLR 2025  
**Authors:** Yu, Ding, Li, Jin (Tsinghua)  
**Links:** [Paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/a402493de088886740b5939f666a6e56-Paper-Conference.pdf) · [Code](https://github.com/tsinghua-fib-lab/SR4MDL)

### Problem

**High R² ≠ correct formula.** Prediction error does not decrease monotonically as search approaches the true expression—search gets lost. Models “cheat” with long, high-R² formulas.

### Proposed solution

Optimize **Minimum Description Length (MDL)** instead of raw MSE:

1. Train **MDLformer** (transformer) to estimate description length of data.
2. Use MDL as **search objective** in MCTS / GP.
3. Training data uses Kamienny-style synthetic generator.

### Results (published)

| Claim | Value |
|-------|-------|
| Exact recovery on ground-truth | ~**50 / 133** problems |
| vs prior SOTA | **+43.92%** recovery rate (paper) |
| Black-box (122) | Strong accuracy–complexity tradeoff |

### Takeaway

*Fitting well ≠ finding the right formula—fix the objective, not just the search.*

Direct competitor to our goal: **short, correct structure** (like DeSTrOI pruning operators).

---

## 6. DeSTrOI + our hybrid pipeline

**Paper:** *Deep Symbolic Tree Operator Identificator* · AAAI 2021  
**Code:** [`Symbolic-Prediction-master/`](../Symbolic-Prediction-master/) · [`destroi_predict.py`](../destroi_predict.py)

### Problem we target

Transformers and MCTS still search over a **large operator alphabet**. Wrong operator choices waste search or produce bloated formulas.

### Our approach

**DeSTrOI** reads the **geometry** of \((x,y)\) as an image → predicts which of **6 operators** appear: `add, mul, inv, sqrt, log, sin`.

```
(X, y) → DeSTrOI → operator scores → forbidden_token_ids → Kamienny Transformer → formula
```

### Integration status

| Baseline | How DeSTrOI would integrate | Status |
|----------|------------------------------|--------|
| Kamienny E2E | Block absent operators in `fit()` | ✅ **Done** |
| TPSR | Mask tokens in MCTS expansion | 📋 Planned |
| uDSR | Prune `function_set` | 📋 Planned |
| DGSR-MCTS | Mask mutation vocabulary | 📋 Planned |
| SR4MDL | Restrict grammar terminals | 📋 Planned |

---

## 7. Cross-paper comparison table

| Method | Year | Search? | Uses E2E transformer? | SRBench strength | Main weakness |
|--------|------|---------|----------------------|------------------|---------------|
| Kamienny E2E | 2022 | Beam only | **Is** the model | Fast; Feynman OK | Blind decode; Strogatz |
| TPSR | 2023 | **MCTS** | Backbone | Strogatz, black-box | Slower; λ tuning |
| DGSR-MCTS | 2023 | **MCTS + online** | Mutation policy | SOTA claim | Very slow |
| uDSR | 2022 | GP + RL + LSPT | One module | Symbolic recovery | Heavy setup |
| SR4MDL | 2025 | MCTS/GP + MDL | Training data only | Exact recovery | New; compute |
| **DeSTrOI + E2E (ours)** | — | Operator prior | Constrains decode | TBD | 6-op; 2D native |

### Published SRBench headline numbers (R² ≥ 0.99 on Feynman / Strogatz)

| Method | Feynman | Strogatz | Source |
|--------|---------|----------|--------|
| E2E Transformer | 84.8% | 35.7% | TPSR paper |
| TPSR (λ=0.1) | **94.9%** | **82.8%** | TPSR paper |
| uDSR | Top symbolic recovery | — | uDSR paper |
| SR4MDL | — | — | ~50/133 **exact** recovery |
| **Our E2E (1 problem)** | 1/1 ✅ | — | [gt_benchmark.csv](../results/srbench/gt_benchmark.csv) |

---

## 8. Our experimental results

*All runs local (Mac CPU). Not yet full SRBench 133.*

### A. Three-way: DeSTrOI vs Transformer vs Combined (synthetic 6-op trees)

**Setup:** Random formulas from DeSTrOI generator · 2 vars · 6 operators · n=100 · seed=0  
**Script:** [`benchmark_three_way.py`](../benchmark_three_way.py)

| Metric | Transformer alone | DeSTrOI + Transformer |
|--------|-------------------|------------------------|
| Mean R² | −32.20 | −36.50 |
| Median R² | — | — |
| R² ≥ 0.95 | **17%** (17/100) | **22%** (22/100) |
| R² < 0 (catastrophic) | 38% | 33% |
| DeSTrOI operator accuracy | — | **74.3%** |
| Combined **better** (ΔR² > 0.005) | — | **42 / 100** |
| Combined **worse** | — | **34 / 100** |

**CSV:** [three_way_benchmark_n100_seed0.csv](../results/three_way/three_way_benchmark_n100_seed0.csv)  
**Figure:** [three_way_comparison_destroi_vocab.png](../results/three_way/figures/three_way_comparison_destroi_vocab.png)

**Pattern:** Combined wins when DeSTrOI correctly identifies operators; **hurts when DeSTrOI blocks a true operator** (e.g. formulas with `inv`).

| Subgroup | n | Transformer mean R² | Combined mean R² |
|----------|---|-------------------|------------------|
| has `inv` | 52 | −61.56 | −69.70 |
| no `inv` | 48 | −0.39 | −0.53 |

---

### B. Transformer only — DeSTrOI-vocab synthetic trees

**Setup:** Same generator as DeSTrOI (6-op) · n=100 · seed=0  
**Script:** [`benchmark_transformer.py`](../benchmark_transformer.py)

| Metric | Value |
|--------|-------|
| Mean R² | 0.16 |
| Median R² | 0.73 |
| R² ≥ 0.95 | **21%** |
| R² ≥ 0.50 | 62% |
| R² < 0 | 30% |

**CSV:** [transformer_benchmark_n100](../results/transformer/synthetic_destroi_vocab/transformer_benchmark_summary_n100.txt)

| Subgroup | Mean R² | Fail rate (R²<0.5) |
|----------|---------|---------------------|
| has `inv` (n=53) | 0.05 | 36% |
| no `inv` (n=47) | 0.29 | 40% |
| has `log` (n=61) | −0.00 | 43% |
| mul+2 unaries (n=6) | −2.09 | 83% |

---

### C. Transformer in-domain (NeurIPS 18-op generator)

**Setup:** Model’s **own** training distribution · n=100 · seed=0  
**Script:** [`benchmark_transformer_indomain.py`](../benchmark_transformer_indomain.py)

| Metric | Value |
|--------|-------|
| Mean R² | −2.42 (skewed by outliers) |
| **Median R²** | **0.999** |
| R² ≥ 0.95 | **74%** |
| R² ≥ 0.50 | 80% |
| R² < 0 | 14% |
| Exact string recovery | 0% |

**Takeaway:** Transformer is **strong in-domain** (median R² ≈ 1) but mean is dragged down by rare catastrophic failures. **Out-of-vocab / DeSTrOI trees are much harder** (Section B).

---

### D. SRBench ground-truth — Kamienny only

**Setup:** 75/25 split · seed=29910 · n_trees=10 · max_train_rows=2000  
**Script:** [`benchmark_srbench_gt.py`](../benchmark_srbench_gt.py)

| Problem | D | GT formula (short) | R²_test | ≥0.99? |
|---------|---|-------------------|---------|--------|
| feynman_III_10_19 | 4 | `mom*sqrt(Bx**2+By**2+Bz**2)` | **0.992** | ✅ |

**CSV:** [gt_benchmark.csv](../results/srbench/gt_benchmark.csv)  
**Target:** 30 problems (~3 h CPU) → then 133 overnight.

---

### E. Summary: our runs vs published TPSR (Feynman / Strogatz)

| Benchmark | Metric | E2E (published) | TPSR (published) | **Our E2E** |
|-----------|--------|-----------------|------------------|-------------|
| Feynman | R²≥0.99 | 84.8% | 94.9% | 1/1 (100%)* |
| Strogatz | R²≥0.99 | 35.7% | 82.8% | not run yet |
| DeSTrOI 6-op synthetic | R²≥0.95 | — | — | 17–22% three-way |
| In-domain 18-op | R²≥0.95 | 64%† | 70.8%† | **74%** |

\*n=1 only · †TPSR paper in-domain synthetic control (400 formulas)

---

## 9. SRBench taxonomy & data

### What is taxonomy?

A **catalog of all 133 ground-truth formulas** with labels: dimensions, operators, nestedness, complexity—**no model required**.

### Files (click to open on GitHub)

| File | Description |
|------|-------------|
| [**taxonomy.csv**](../results/srbench/taxonomy.csv) | Full table (133 rows) — open in Excel or GitHub |
| [taxonomy_summary.txt](../results/srbench/taxonomy_summary.txt) | Counts by benchmark, complexity, D |
| [problem_list.json](../datasets/srbench/problem_list.json) | All 133 problem names |

**If the link 404s:** run `git pull` — ensure `taxonomy.csv` is committed.  
**Regenerate:** `python3 build_srbench_taxonomy.py`

### Taxonomy snapshot

| | Count |
|--|-------|
| Feynman | 119 |
| Strogatz | 14 |
| Simple / medium / complex | 28 / 70 / 35 |
| D=2 / D=3 / D=4 | 29 / 37 / 32 |
| DeSTrOI-compatible (heuristic) | 129 |

### SRBench structure

| Sub-benchmark | # | Ground truth? |
|---------------|---|---------------|
| Feynman | 119 | Yes |
| Strogatz | 14 | Yes |
| Black-box | 122 | No |
| **Total ground-truth** | **133** | |

**SRBench:** [github.com/cavalab/srbench](https://github.com/cavalab/srbench)

---

*Last updated: project benchmark runs on Mac CPU. Re-run `benchmark_srbench_gt.py --n 30` to refresh Section D.*
