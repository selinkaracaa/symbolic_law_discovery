# Symbolic Regression × Transformers × DeSTrOI  
**Extended meeting notes** · June 2025  

> **Main documentation:**  
> - [README.md](../README.md) — roadmap & quick findings  
> - [**LITERATURE_AND_RESULTS.md**](LITERATURE_AND_RESULTS.md) — **detailed paper summaries (TPSR-style) + all result tables**

**Repo:** [github.com/selinkaracaa/symbolic_law_discovery](https://github.com/selinkaracaa/symbolic_law_discovery)

---

## Executive summary

Symbolic regression (SR) shifted after Kamienny et al. (NeurIPS 2022): transformers can propose formulas in one forward pass, but **search, recovery, and OOD data** remain open. The field’s answer is **hybrids** (MCTS, GP, MDL objectives)—not pure transformers alone.

**Our contribution (in progress):** DeSTrOI as a **visual operator-identification prior** that constrains transformer decoding *before* search—cheaper than full MCTS, complementary to TPSR/uDSR-style pipelines.

**What we have today:** working DeSTrOI + Kamienny transformer pipeline, synthetic benchmarks, **full taxonomy of 133 SRBench ground-truth problems**, and early SRBench transformer runs. **What’s next:** run Kamienny/TPSR/uDSR/SR4MDL on SRBench ± DeSTrOI (blocked on compute + per-method integration).

---

# Part 1 — Literature summaries (5 core papers)

## 0. Kamienny et al. 2022 — End-to-End Symbolic Regression with Transformers (baseline)

| | |
|--|--|
| **Venue** | NeurIPS 2022 |
| **Link** | [arXiv:2204.10532](https://arxiv.org/abs/2204.10532) · [Code](https://github.com/facebookresearch/symbolicregression) |
| **Idea** | Pre-train a transformer on synthetic formulas; at test time encode \((X,y)\) points and **decode a formula token-by-token**; refine constants with BFGS. |
| **Strength** | **Fast** inference vs GP; competitive on SRBench; **low formula complexity** vs top GP methods. |
| **Limitation** | **Greedy/beam decoding** ignores fit quality until the end; weak on **OOD** and **Strogatz** (time-ordered data); **exact recovery** still limited; ranked ~4th on SRBench overall. |
| **SRBench (published)** | ~119 Feynman: strong R²≥0.99 rate; Strogatz and black-box harder. |

**One line:** *First serious transformer SR baseline—fast but blind during decoding.*

---

## 1. TPSR — Transformer-based Planning for Symbolic Regression

| | |
|--|--|
| **Authors** | Shojaee, Meidani, Barati Farimani, Reddy |
| **Venue** | NeurIPS 2023 |
| **Link** | [arXiv:2303.06833](https://arxiv.org/abs/2303.06833) · [Code](https://github.com/deep-symbolic-mathematics/TPSR) |
| **Builds on** | Kamienny E2E transformer as **backbone**; adds **MCTS at decode time**. |

### Problem with plain transformer
- Chooses next token from **language-model probabilities only**.
- No **R²** or **complexity** feedback until the full equation is written.
- Often overfits with bloated formulas or misses the true structure.

### TPSR solution (step by step)
1. **Selection** — MCTS picks which partial expression to expand.
2. **Expansion** — Transformer proposes likely next symbols (not random).
3. **Evaluation** — Beam search **simulates** completing the equation.
4. **Reward** — Grade completed candidates: **accuracy + λ·complexity**.
5. **Backprop** — Update which symbol choices were good.

### Published SRBench results (from paper; λ = complexity weight)

| Dataset | E2E Transformer | TPSR (λ=0.1) | TPSR (λ=0) |
|---------|-----------------|--------------|------------|
| **Feynman** R²≥0.99 | 84.8% | **94.9%** | 95.2% |
| **Strogatz** R²≥0.99 | 35.7% | **82.8%** | 92.8% |
| **Black-box** mean R² | 0.864 | **0.945** | (bloated; complexity ~130) |
| **In-domain synthetic** | 64.0% | **70.8%** | 70.2% |

**One line:** *Same transformer, smarter search—especially fixes Strogatz and black-box.*

---

## 2. DGSR-MCTS — Deep Generative Symbolic Regression with MCTS

| | |
|--|--|
| **Authors** | Kamienny, Lample, Lamprier, Virgolin (same lead author as E2E) |
| **Venue** | ICML 2023 |
| **Link** | [Proceedings](https://proceedings.mlr.press/v202/kamienny23a.html) · [Code](https://github.com/vanderschaarlab/DeepGenerativeSymbolicRegression) |
| **Idea** | E2E transformer is **fast but OOD-weak** without search. DGSR-MCTS runs **MCTS** with a **neural mutation policy** (pre-trained transformer), **online fine-tuned** on successful mutations. |

### vs Kamienny 2022
| E2E Transformer | DGSR-MCTS |
|-----------------|-----------|
| One-shot decode | Many search iterations |
| Fixed weights at test | Policy updates during search |
| Fast | Slower (500k eval budget in paper) |
| OOD gap | **SOTA on SRBench** (paper claim) |

**One line:** *Authors’ official answer: “transformer alone isn’t enough on hard benchmarks—add MCTS + online learning.”*

---

## 3. uDSR — A Unified Framework for Deep Symbolic Regression

| | |
|--|--|
| **Authors** | Landajuela et al. (LLNL / SRBench team) |
| **Venue** | NeurIPS 2022 |
| **Link** | [Paper](https://papers.nips.cc/paper_files/paper/2022/hash/dbca58f35bddc6e4003b2dd80e42f838-Abstract-Conference.html) · [Code](https://github.com/dso-org/deep-symbolic-optimization) |
| **Idea** | **Modular hybrid** — plug in 5 strategies: simplification, neural-guided search (DSR), **large-scale pre-training (LSPT)**, GP, linear models. Kamienny’s transformer = LSPT module. |

### Why it matters
- Not “transformer **or** GP” but **combine** them in one pipeline.
- **Highest symbolic recovery** on SRBench ground-truth at time of publication.
- Won **1st place** GECCO 2022 SRBench real-world track.

**One line:** *Industrial-strength hybrid—transformer is one interchangeable block.*

---

## 4. SR4MDL — Symbolic Regression via MDLformer-Guided Search

| | |
|--|--|
| **Authors** | Yu, Ding, Li, Jin (Tsinghua) |
| **Venue** | ICLR 2025 |
| **Link** | [Paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/a402493de088886740b5939f666a6e56-Paper-Conference.pdf) · [Code](https://github.com/tsinghua-fib-lab/SR4MDL) |
| **Problem** | High R² ≠ **correct formula**; prediction-error landscape is non-monotonic during search. |
| **Solution** | **Minimum Description Length (MDL)** objective via **MDLformer** (transformer trained to estimate description length) + **MCTS/GP search**. Uses Kamienny-style synthetic data for training. |

### Published claim
- ~**50 / 133** ground-truth formulas **exactly recovered** (vs prior SOTA by large margin on recovery rate).
- Strong on **short, elegant** formulas—direct competitor to “don’t bloat the search space.”

**One line:** *Change the objective from “fit data” to “shortest true explanation.”*

---

# Part 2 — Strategic questions

| # | Question | Answer |
|---|----------|--------|
| **1** | Is SR well-studied and **solved**? | **Well-studied, not solved.** SRBench shows no single method wins everywhere. **Exact recovery** under noise is still hard. |
| **2** | Transformers alone or **combine**? | **Combine.** TPSR, DGSR-MCTS, uDSR, SR4MDL all add search, GP, or new objectives. Pure E2E is a **fast baseline**, not the ceiling. |
| **3** | Is DeSTrOI a **prerequisite** before transformers? | **Not universally—but a useful prior** when operator vocabulary can be pruned. We block transformer tokens via `forbidden_token_ids` when DeSTrOI predicts an operator is absent. Wrong predictions **hurt** (shown in our synthetic benchmark). Scope today: **2D + 6 operators** (extendable to k≤10 with MIL attention—code needs Keras patch). |
| **4** | What’s **left to do**? | High-D scaling; exact recovery metrics; noise sweeps; **integrate DeSTrOI into search** (TPSR mutations, uDSR `function_set`, SR4MDL MCTS); full SRBench ± DeSTrOI table; deployment UX. |
| **5** | Deployed in physics / EE / medicine? | **Physics:** Feynman benchmark + materials/fluids (SINDy, AI Feynman follow-ups). **EE:** phenomenological fits. **Medicine/pharma:** early (PK models); interpretability matters. Mostly **research**, not production SR products. |
| **7** | How to **deploy**? Who consumes? | CSV in → ranked formulas out. Consumers: experimental scientists, engineers, analysts who need **interpretable** models. Our stack: `destroi_predict.py` + `SymbolicTransformerRegressor` + SRBench loop. |
| **8** | **DeepMind** | FunSearch / AlphaTensor lineage = LLM + search for discovery; different stack but same theme: **neural prior + algorithmic search**. Worth citing as industrial parallel. |

**Recent paper to track:** [arXiv:2511.08544](https://arxiv.org/abs/2511.08544) — add to related work (not yet integrated in this repo).

---

# Part 3 — Data & benchmarks

## SRBench (gold standard)

| Source | [github.com/cavalab/srbench](https://github.com/cavalab/srbench) · [cavalab.org/srbench](https://cavalab.org/srbench) |
|--------|---|
| **Total** | 252 PMLB datasets |
| **Ground-truth** | **133** (119 Feynman + 14 Strogatz) — known formulas |
| **Black-box** | 122 — no known formula |
| **Used by** | Kamienny 2022, uDSR, TPSR, SR4MDL, DGSR-MCTS |

| Sub-benchmark | # | What it tests |
|---------------|---|---------------|
| Feynman | ~119 | Physics equations (uniform random inputs) |
| Strogatz | 14 | Chaotic ODE trajectories (**time-ordered** — hard for transformers) |
| Black-box | 122 | Real regression |

## What we did

| Deliverable | Status | Location |
|-------------|--------|----------|
| **Taxonomy (133 problems)** | ✅ Done | [`results/srbench/taxonomy.csv`](../results/srbench/taxonomy.csv) |
| Taxonomy summary | ✅ | [`results/srbench/taxonomy_summary.txt`](../results/srbench/taxonomy_summary.txt) |
| PMLB data cache | ✅ Downloaded | `datasets/srbench/cache/` (gitignored) |
| Problem list | ✅ | [`datasets/srbench/problem_list.json`](../datasets/srbench/problem_list.json) |
| Kamienny on SRBench | ⏳ 1/133 | [`results/srbench/gt_benchmark.csv`](../results/srbench/gt_benchmark.csv) |
| DeSTrOI + Transformer (synthetic) | ✅ 100 formulas | [`results/three_way/`](../results/three_way/) |

### Taxonomy at a glance

- **119** Feynman, **14** Strogatz  
- Complexity: **28** simple · **70** medium · **35** complex  
- Dimensions: D=2 (29), D=3 (37), D=4 (32), … up to D=9  
- **129/133** flagged DeSTrOI-compatible (heuristic: operator overlap with 6-op vocab)

**Open taxonomy:** `open results/srbench/taxonomy.csv` (Excel/Numbers) or view on GitHub.

---

# Part 4 — Pros & cons comparison

| Method | Year | Core mechanism | Pros | Cons |
|--------|------|----------------|------|------|
| **Kamienny E2E** | 2022 | Transformer decode + BFGS | Very fast; simple API; low complexity formulas; we have `model.pt` working | Blind decoding; weak Strogatz/OOD; ~4th on SRBench; poor exact recovery |
| **TPSR** | 2023 | E2E + **MCTS** + R²/complexity reward | Big gains on Feynman, Strogatz, black-box; same backbone | Slower than E2E; GPU helpful; MCTS hyperparams (λ, rollouts) |
| **DGSR-MCTS** | 2023 | **MCTS + online mutation policy** | SOTA SRBench in paper; fixes OOD | Heavy compute (500k evals); complex install; unofficial repo fork |
| **uDSR** | 2022 | **5-way hybrid** (GP+DSR+LSPT+…) | Best symbolic recovery on SRBench; modular | Slow; separate conda env (DSO); not end-to-end neural |
| **SR4MDL** | 2025 | **MDL objective** + MCTS | Best **exact recovery** claim; short formulas | Newest; needs checkpoint; search cost |
| **DeSTrOI (ours)** | 2021 | **Visual operator ID** from \((x,y)\) landscape | O(1) forward pass; shrinks operator alphabet; unique modality | 6-op vocab; 2D native (k≤10 with projections); mis-prediction blocks true ops |
| **DeSTrOI + E2E (ours)** | — | Operator block → transformer decode | Implemented; wins when DeSTrOI accurate | Only wired to Kamienny; not yet on TPSR/uDSR/SR4MDL |

---

# Part 5 — DeSTrOI integration plan

## What works today

```
(X, y)  →  DeSTrOI  →  operator scores  →  forbidden_token_ids  →  Kamienny Transformer  →  formula
```

Code: `destroi_predict.py`, `demo_combined.py`, `benchmark_three_way.py`

## How to integrate with each baseline

| Baseline | Integration point | DeSTrOI action |
|----------|-------------------|----------------|
| **Kamienny E2E** | `SymbolicTransformerRegressor.fit(forbidden_token_ids=…)` | ✅ **Done** |
| **TPSR** | MCTS expansion — mask invalid ops in mutation set | Prune token vocabulary per node |
| **DGSR-MCTS** | Neural mutation policy | Same as TPSR |
| **uDSR / DSO** | `function_set` in config | Drop operators DeSTrOI scores absent |
| **SR4MDL** | MCTS/GP search space | Restrict grammar terminals |

**Novelty claim:** DeSTrOI is a **cheap structural prior** (one CNN forward pass) vs expensive MCTS—orthogonal to TPSR (can combine: DeSTrOI prune **then** TPSR search).

## What we need to run each method

| Method | Repo | Weights / deps | Est. setup | SRBench 30 problems |
|--------|------|----------------|------------|---------------------|
| Kamienny | ✅ in repo | `model.pt` ✅ | Ready | ~3 h CPU (`--max-train-rows 2000`) |
| TPSR | `deep-symbolic-mathematics/TPSR` | E2E weights + GPU | 1–2 days | 1–2 days |
| uDSR | `dso-org/deep-symbolic-optimization` | conda env | 1–2 days | via SRBench harness |
| DGSR-MCTS | `vanderschaarlab/DeepGenerativeSymbolicRegression` | pretrain hours | 2–3+ days | days |
| SR4MDL | `tsinghua-fib-lab/SR4MDL` | `checkpoint.pth` | 1–2 days | hours/problem |
| DeSTrOI | ✅ in repo | `.h5` weights ✅ | Ready (2D); k>10 needs Keras patch | minutes (operator ID only) |

---

# Part 6 — Our experimental results (actual runs)

## A. Synthetic 6-op trees (DeSTrOI generator, n=100)

| Metric | Transformer alone | DeSTrOI + Transformer |
|--------|-------------------|------------------------|
| Mean R² | −32.2 (many catastrophic fails) | −36.5 |
| R² ≥ 0.95 | 17% | **22%** |
| DeSTrOI operator accuracy | — | **74.3%** |
| Combined better (ΔR²>0) | — | **42/100** |
| Combined worse | — | **34/100** (when DeSTrOI wrong) |

**Figure:** [`results/three_way/figures/three_way_comparison_destroi_vocab.png`](../results/three_way/figures/three_way_comparison_destroi_vocab.png)

**Takeaway:** DeSTrOI helps when operator ID is correct; **false negatives hurt** (blocks true operators).

## B. SRBench ground-truth (Kamienny only)

| Problem | R²_test | R²≥0.99? |
|---------|---------|----------|
| `feynman_III_10_19` (D=4) | **0.992** | ✅ |

*Full 133 not run on laptop—~6 min/problem ≈ 14 h CPU.*

## C. Literature numbers we cite (not our runs)

Use TPSR paper table for professor comparison until we reproduce:

- Feynman: E2E **84.8%** → TPSR **94.9%** (R²≥0.99)
- Strogatz: E2E **35.7%** → TPSR **82.8%**

TPSR published results: [github.com/deep-symbolic-mathematics/TPSR/tree/main/srbench_results](https://github.com/deep-symbolic-mathematics/TPSR/tree/main/srbench_results)

---

# Part 7 — Proposed full study (the plan)

```
Phase 1 ✅  Taxonomy + repo + DeSTrOI↔Kamienny prototype
Phase 2 ⏳  Kamienny on 30–133 SRBench ground-truth
Phase 3     TPSR ± DeSTrOI (operator-masked MCTS) on same subset
Phase 4     uDSR with pruned function_set ± DeSTrOI
Phase 5     SR4MDL / DGSR-MCTS on subset (compute permitting)
Phase 6     Paper: "Visual operator prior reduces search entropy before neural SR"
```

### Why we could not run all 5 × 133 × {±DeSTrOI} on laptop

- **Time:** Kamienny ~6 min/problem → 133 ≈ **14 hours** CPU.
- **Integration:** DeSTrOI only hooked to Kamienny; other methods need per-repo wiring.
- **Environments:** uDSR, DGSR, SR4MDL need separate installs + GPU for practical use.
- **Honest scope for tomorrow:** taxonomy (133) + Kamienny subset (target **30**) + synthetic DeSTrOI demo + literature table.

### Recommended 30-problem subset (balanced)

- **26 Feynman** (mix D=2,3,4, simple/medium/complex)  
- **4 Strogatz** (ODE stress test)  
- Script: `python3 benchmark_srbench_gt.py --n 30` (or custom list — see below)

**Run tonight (~3 h, plug in laptop):**

```bash
cd ~/Desktop/symbolic_law_discovery
python3 -u benchmark_srbench_gt.py --n 30 --seed 29910 --n-trees 10 --max-train-rows 2000
```

---

# Part 8 — One-page story for tomorrow

1. **Field moved from GP → transformers (2022) → hybrids + search (2023–2025).**  
2. **Gap:** decoding is blind; search is expensive; R² ≠ recovery.  
3. **Our idea:** DeSTrOI reads the **geometry** of \((X,y)\) and **prunes operators** before any transformer/search.  
4. **Evidence:** working hybrid code on GitHub; taxonomy of all 133 SRBench formulas; synthetic benchmark shows win/loss modes.  
5. **Next:** Kamienny on 30 SRBench problems; then TPSR with masked ops (cite their published SRBench numbers until reproduced).  
6. **Ask:** compute (GPU/server)? co-advise on SRBench protocol? internship alignment?

---

# Appendix — Key links

| Resource | URL |
|----------|-----|
| **Our repo** | https://github.com/selinkaracaa/symbolic_law_discovery |
| **Taxonomy CSV** | `results/srbench/taxonomy.csv` |
| SRBench | https://github.com/cavalab/srbench |
| Kamienny 2022 | https://arxiv.org/abs/2204.10532 |
| TPSR | https://arxiv.org/abs/2303.06833 |
| DGSR-MCTS | https://proceedings.mlr.press/v202/kamienny23a.html |
| uDSR | https://github.com/dso-org/deep-symbolic-optimization |
| SR4MDL | https://github.com/tsinghua-fib-lab/SR4MDL |
| Recent (to read) | https://arxiv.org/abs/2511.08544 |

---

*Generated for professor meeting. Update `gt_benchmark.csv` row count after overnight Kamienny run.*
