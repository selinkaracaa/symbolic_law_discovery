# Research roadmap & strategic questions

**Project:** DeSTrOI as an operator-identification prior for transformer-based symbolic regression  
**Repo:** [github.com/selinkaracaa/symbolic_law_discovery](https://github.com/selinkaracaa/symbolic_law_discovery)

Related docs:
- [LITERATURE_AND_RESULTS.md](LITERATURE_AND_RESULTS.md) — paper summaries + benchmark tables
- [PROFESSOR_BRIEF.md](PROFESSOR_BRIEF.md) — meeting notes
- [README.md](../README.md) — quick reference

---

## Execution timeline

```
[Phase 1: Taxonomy]  ✅ DONE
        ↓
[Phase 2: Limitation analysis]  ✅ DONE
        ↓
[Phase 3: Benchmark showdown]  ✅ 20 SRBench problems done
        ↓
[Phase 4: Proposal & paper]  📋 NEXT
```

---

## Phase status

| Phase | Goal | Status | Deliverables |
|-------|------|--------|--------------|
| **1** | Taxonomy & data for 133 SRBench ground-truth formulas | ✅ **Done** | [taxonomy.csv](../results/srbench/taxonomy.csv) · [summary](../results/srbench/taxonomy_summary.txt) · [problem_list.json](../datasets/srbench/problem_list.json) · `srbench_data.py` |
| **2** | Map SOTA limitations; position DeSTrOI | ✅ **Done** | [Limitations table §7](LITERATURE_AND_RESULTS.md#7-limitations-gaps--next-steps-all-papers) · [LITERATURE](LITERATURE_AND_RESULTS.md) |
| **3** | Run baselines ± DeSTrOI on SRBench | ✅ **20 problems** | [srbench_20_benchmark.csv](../results/srbench/srbench_20_benchmark.csv) |
| **4** | Proposal / paper write-up | 📋 **Next** | TPSR ± DeSTrOI on lab server |

**Yes — Phases 1 and 2 are complete** for a proposal or professor meeting. Phase 3 needs Phase A results (and later TPSR on a GPU server) before claiming method comparisons on SRBench.

---

## Why these 6 strategic questions?

These questions were chosen **before benchmarking**, to stress-test whether the project is novel, feasible, and worth a semester+ of work. They mirror the trajectory the SR community took after Kamienny (2022): fast transformers → OOD failures → hybrid search (TPSR, DGSR, uDSR, SR4MDL).

| # | Question | Why we asked it | Where answered |
|---|----------|-----------------|----------------|
| **1** | Is SR well-studied and **solved**? | If solved, no paper. SRBench (La Cava et al.) shows no single winner → gap remains. | [§1 below](#1-is-symbolic-regression-solved) · [LITERATURE §7](LITERATURE_AND_RESULTS.md#7-cross-paper-comparison-table) |
| **2** | Transformers alone or **combine**? | Kamienny alone is the baseline; every 2023–2025 SOTA paper adds search or hybrids. Defines our comparison set. | [§2 below](#2-transformers-alone-or-combine) |
| **3** | Is **DeSTrOI** a prerequisite before transformers? | Core novelty claim: visual operator ID **before** decode/search, vs expensive MCTS line-by-line. | [§3 below](#3-is-destroi-a-prior) · [LITERATURE §6](LITERATURE_AND_RESULTS.md#6-destroi--our-hybrid-pipeline) |
| **4** | What is **left to do**? | Turns literature gaps into our experiment list (noise, exact recovery, high-D, TPSR integration). | [§4 below](#4-whats-left-to-do) |
| **5** | Used in **physics / EE / medicine**? | Justifies real-world impact for proposal intro / related applications. | [§5 below](#5-deployment-domains) |
| **6** | How to **deploy**? Who consumes? | Forces a concrete product story (CSV in → formulas out) beyond pure accuracy tables. | [§6 below](#6-deployment) |

**Extra (professor meeting):** DeepMind FunSearch / AlphaTensor as industrial parallel (neural prior + search); recent arXiv:2511.08544 for related work.

---

## Phase 1 — Formula taxonomy & data (✅ done)

### Step 1.1 — Categorize ground-truth formulas

All **133** SRBench ground-truth problems (119 Feynman + 14 Strogatz) are catalogued in [`results/srbench/taxonomy.csv`](../results/srbench/taxonomy.csv).

| Column | Meaning |
|--------|---------|
| `n_features` | Input dimensionality (D=1 … D=9) |
| `ops` | Operators in ground-truth formula |
| `n_operators`, `nestedness`, `complexity` | Structural difficulty |
| `has_trig`, `has_exp_log`, `has_sqrt`, `has_div`, `has_pow` | Operator families |
| `destroi_compatible` | Heuristic: overlaps DeSTrOI 6-op vocab, D≤10 |

**Snapshot** ([full summary](../results/srbench/taxonomy_summary.txt)):

| Slice | Count |
|-------|-------|
| Feynman | 119 |
| Strogatz | 14 |
| Simple / medium / complex | 28 / 70 / 35 |
| D=2 / D=3 / D=4 | 29 / 37 / 32 |
| DeSTrOI-compatible (heuristic) | 129 |

**Regenerate:** `python3 build_srbench_taxonomy.py`

### Step 1.2 — SRBench data profiles

| Item | Status |
|------|--------|
| PMLB download pipeline | ✅ `srbench_data.py` |
| 133 datasets cached locally | ✅ `datasets/srbench/cache/` (gitignored) |
| Feynman = uniform random inputs | ✅ documented in taxonomy `benchmark` column |
| Strogatz = time-ordered ODE trajectories | ✅ 14 problems; known transformer failure mode |

**20-problem benchmark:** [`datasets/srbench/benchmark_subset_20.json`](../datasets/srbench/benchmark_subset_20.json) — 12 Feynman + 8 Strogatz.

---

## Phase 2 — Method limitations & DeSTrOI positioning (✅ done)

**Full limitations table (all papers):** [LITERATURE_AND_RESULTS.md §7 — Limitations, gaps & next steps](LITERATURE_AND_RESULTS.md#7-limitations-gaps--next-steps-all-papers)

### Quick view — where each method struggles on our taxonomy

| Taxonomy challenge | Kamienny E2E | TPSR | DGSR-MCTS | uDSR | SR4MDL | DeSTrOI (ours) |
|------------------|--------------|------|-----------|------|--------|----------------|
| **Strogatz** (14) | ❌ | ✅ | ✅ | ✅ | TBD | ⚠️ 2D slice only |
| **Complex** (35) | ❌ | ⚠️ | ⚠️ | ⚠️ | ✅ | ⚠️ if ops mis-ID'd |
| **High D** (D≥7, 4) | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ❌ needs MIL |
| **Exact recovery** | ❌ | ⚠️ | ✅ | ✅ | ✅ ~50/133 | 📋 not tested |
| **Speed / compute** | ✅ | ⚠️ | ❌ | ❌ | ⚠️ | ✅ O(1) CNN |

See the [master table](LITERATURE_AND_RESULTS.md#summary-table) for plain-English limitations, what's missing, and next steps per paper.

### How DeSTrOI is positioned

> **While TPSR uses MCTS to search equations symbol-by-symbol with data feedback, DeSTrOI minimizes the *operator* search space up front** by reading the geometry of \((X,y)\) as an image and predicting which of six operators (`add, mul, inv, sqrt, log, sin`) appear—**one CNN forward pass** vs thousands of MCTS rollouts.

| DeSTrOI addresses | Does not replace |
|-------------------|------------------|
| Wrong operator branches early | Need for MCTS on hard structure (Strogatz, nesting) |
| Search space explosion in operator alphabet | Constant fitting (BFGS), full formula search |
| Cheap prior usable with any baseline | Full high-D MIL projections (k>2 needs Keras fix) |

**Integration plan:**

```
(X, y) → DeSTrOI → operator mask → { Kamienny ✅ | TPSR 📋 | uDSR 📋 | SR4MDL 📋 }
```

---

## Strategic answers (literature-backed)

### 1. Is symbolic regression solved?

**Well-studied, not solved.** Post-2022 deep SR (Kamienny, TPSR, uDSR, SR4MDL) improved speed and accuracy, but SRBench shows **no single method wins everywhere**. The open bottleneck is **exact recovery under noise**—models cheat R² with bloated formulas without finding the true law.

### 2. Transformers alone or combine?

**Combine.** Pure E2E transformer = fast baseline. SOTA adds **MCTS** (TPSR, DGSR), **GP/RL hybrids** (uDSR), or **MDL objectives** (SR4MDL). Our work adds DeSTrOI as another **cheap hybrid layer**, not a replacement for search.

### 3. Is DeSTrOI a prior?

**Yes—that is the novelty.** DeSTrOI (AAAI 2021) predates Kamienny; we repurpose it as a **structural prior** before neural decode. It is not universally required (wrong operator blocks hurt—see synthetic benchmark), but when accurate it prunes the vocabulary before the transformer guesses.

### 4. What's left to do?

- High-D scaling (D>10; MIL attention path)
- **DeSTrOI + TPSR** on SRBench (fair test of hypothesis)
- Exact recovery & noise sweeps (1%, 5% Gaussian)
- Physical constraints (units, dimensional analysis)
- Deployment UX (CSV → ranked formulas)

### 5. Deployment domains

| Domain | Usage |
|--------|-------|
| **Physics** | Feynman benchmark; materials, fluids (SINDy lineage) |
| **EE / engineering** | Phenomenological fits, battery curves |
| **Medicine / pharma** | Early PK modeling; interpretability over black-box NN |

Mostly **research** today, not production SR products.

### 6. Deployment

**Consumers:** experimental scientists, engineers, analysts needing **interpretable** models.  
**Stack:** `destroi_predict.py` + `SymbolicTransformerRegressor` + SRBench loop → CSV in, ranked formulas out. Future: web UI with units metadata.

---

## Phase 3 — Testing showdown (⏳ in progress)

### Done

| Experiment | n | Result | File |
|------------|---|--------|------|
| Three-way synthetic | 100 | Comb 22% vs Trans 17% R²≥0.95 | [three_way/](../results/three_way/) |
| Transformer in-domain | 100 | Median R² 0.999 | [indomain/](../results/transformer/indomain/) |
| Taxonomy | 133 | Full catalog | [taxonomy.csv](../results/srbench/taxonomy.csv) |
| **SRBench 20** (E2E vs DeSTrOI+E2E) | 20 | E2E 7/20 R²≥0.99 · mixed ΔR² | [srbench_20_benchmark.csv](../results/srbench/srbench_20_benchmark.csv) |

**Takeaway:** DeSTrOI+E2E is mixed on 20 SRBench problems — small mean ΔR² gain, no clear win on R²≥0.99 rate. Strogatz 0/8 for E2E. Next: **DeSTrOI + TPSR** on lab GPU.

### Planned (needs lab GPU)

| Step | Work |
|------|------|
| 3.1 | Kamienny + TPSR on same 10-problem subset |
| 3.2 | DeSTrOI + TPSR vs TPSR alone |
| 3.3 | Metrics: R²≥0.99, exact recovery, complexity Pareto, noise 1%/5% |
| 3.4 | Scale to 30 → 133 problems |

---

## Phase 4 — Proposal & paper outline (📋 next)

1. **Introduction** — Transformers revolutionized SR speed; OOD, structure, and exact recovery remain open. Introduce DeSTrOI as operator prior.
2. **Related work** — Kamienny → TPSR / DGSR / uDSR / SR4MDL progression; DeSTrOI (AAAI 2021).
3. **Method** — DeSTrOI visual encoding → operator mask → baseline SR (E2E, then TPSR).
4. **Experiments** — Taxonomy breakdown (Phase 1); limitation matrix (Phase 2); SRBench ± DeSTrOI tables (Phase 3).
5. **Discussion** — When operator prior helps vs hurts; compute vs MCTS tradeoff.

---

## Quick commands

```bash
# Phase 1 — taxonomy (no GPU)
python3 build_srbench_taxonomy.py

python3 benchmark_srbench_20.py --max-train-rows 2000
```

---

*Last updated: 20-problem SRBench benchmark complete.*
