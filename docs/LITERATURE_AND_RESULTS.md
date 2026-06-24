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
7. [Limitations, gaps & next steps (all papers)](#7-limitations-gaps--next-steps-all-papers)
8. [Cross-paper comparison](#8-cross-paper-comparison-table)
9. [Our experimental results](#9-our-experimental-results)
10. [SRBench taxonomy & data](#10-srbench-taxonomy--data)

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

**TPSR** = Kamienny E2E transformer + **Monte Carlo Tree Search (MCTS)** at decode time.

Think of the plain transformer as **typing blind**: it picks the next symbol because it “sounds” likely, not because the partial equation already fits the data. TPSR lets the model **pause**, **try out futures**, **grade them against the real data**, and only then commit to the next symbol.

### Method (step by step)

**Every time TPSR needs to choose the next symbol**, it runs this loop:

**1. Selection**  
The AI looks at its **current partial equation** (e.g. `sin(x) + …`) and decides **which unfinished branch** is worth exploring next. It balances two instincts:
- **Exploit** paths that already look promising (high past scores)
- **Explore** paths it hasn’t tried much yet (other operators or structures)

This is MCTS “walking” a tree of possible formulas.

**2. Expansion**  
Instead of guessing randomly from *all* mathematical symbols, TPSR asks the **pre-trained transformer** to narrow the options to a few highly likely next steps — e.g. choosing between `+` vs `sin`, not every token in the vocabulary.

**3. Evaluation (the key trick)**  
You **cannot score a half-finished equation** — `sin(x) +` isn’t valid math yet. So TPSR uses **beam search** to quickly **simulate finishing** the formula: the transformer drafts several plausible completions, then each full candidate can be tested on the data.

**4. Reward**  
Those completed test equations are **graded** with a reward function:

| Component | Meaning |
|-----------|---------|
| **Fitting accuracy** | How closely the formula matches the observed \((X,y)\) points (e.g. R² or NMSE) |
| **Complexity penalty** | Penalizes bloated, overly long equations (λ controls how much) |

**Combined score** ≈ accuracy − λ × complexity

**5. Backpropagation**  
The reward is sent **backward** through the search tree: symbol choices that led to good completed formulas get reinforced; bad branches get deprioritized. Over many rollouts, TPSR learns **which next symbol was actually the best choice** given the data — not just the most probable token.

**λ (lambda) in plain terms:**
- **λ = 0** → only care about fit → often huge, ugly formulas that still score well
- **λ = 1** → heavily punish length → may sacrifice accuracy
- **λ = 0.1** → paper’s recommended **balance** (accurate *and* readable)

**One-sentence summary:** TPSR uses the transformer as a **smart proposal engine** inside a **search loop** that always checks “does this path actually fit the data?” before moving on.

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

### What we are proposing (read this first)

**We are NOT claiming that “DeSTrOI + plain transformer” beats TPSR.** Those are different things:

| | Plain Kamienny transformer | TPSR | **Our work so far** |
|--|---------------------------|------|---------------------|
| **Search** | Beam decode only (fast, blind) | MCTS + reward loop (slow, data-aware) | DeSTrOI operator mask + beam decode |
| **Uses data while building?** | No (until end) | Yes (every symbol) | Partially (DeSTrOI reads \((X,y)\) landscape once) |
| **SRBench Feynman (published)** | 84.8% R²≥0.99 | **94.9%** | **1 problem run** — not comparable yet |

**The actual hypothesis:** DeSTrOI is a **cheap add-on layer** you stack **on top of** existing methods — Kamienny, TPSR, uDSR, etc. — to **shrink the operator search space** before or during search.

```
                    ┌─────────────────────────────────────┐
  (X, y)  ────────► │  DeSTrOI: which operators appear? │  ← one CNN pass
                    └─────────────────┬───────────────────┘
                                      │ mask / prune operators
                    ┌─────────────────▼───────────────────┐
                    │  THEN pick a baseline:            │
                    │  • Kamienny E2E  ✅ we tested this │
                    │  • TPSR + MCTS   📋 not wired yet  │
                    │  • uDSR / SR4MDL 📋 planned        │
                    └───────────────────────────────────┘
```

**What we have built today:** DeSTrOI → `forbidden_token_ids` → **Kamienny transformer only** (no MCTS).

**What we have measured:** On 100 **synthetic 6-op** formulas (not SRBench):
- Transformer alone: **17%** hit R²≥0.95
- DeSTrOI + Transformer: **22%** — a **small** gain, not a TPSR-level jump
- **34/100** cases got **worse** when DeSTrOI blocked a true operator (74% operator ID accuracy)

**What we have NOT run:** DeSTrOI + TPSR on SRBench. That is the experiment that would test whether operator pruning helps MCTS — we cite TPSR’s published numbers as the **target to beat or complement**, not as something we’ve already surpassed.

### Problem we target

Even with MCTS (TPSR), search still wanders over a **large operator alphabet**. Wrong operator choices waste rollouts or produce bloated formulas. DeSTrOI tries to answer: *“Which of these 6 operators even appear in this dataset?”* before decoding starts.

### Our approach

**DeSTrOI** reads the **geometry** of \((x,y)\) as an image → predicts which of **6 operators** appear: `add, mul, inv, sqrt, log, sin`.

```
(X, y) → DeSTrOI → operator scores → forbidden_token_ids → Kamienny Transformer → formula
```

(Future: same mask fed into TPSR’s MCTS expansion step.)

### Integration status

| Baseline | How DeSTrOI would integrate | Status | Tested on SRBench? |
|----------|------------------------------|--------|-------------------|
| Kamienny E2E | Block absent operators in `fit()` | ✅ **Done** | ⏳ 1/133 |
| TPSR | Mask tokens in MCTS expansion | 📋 Planned | ❌ |
| uDSR | Prune `function_set` | 📋 Planned | ❌ |
| DGSR-MCTS | Mask mutation vocabulary | 📋 Planned | ❌ |
| SR4MDL | Restrict grammar terminals | 📋 Planned | ❌ |

---

## 7. Limitations, gaps & next steps (all papers)

Master reference for Phase 2 of the [research roadmap](RESEARCH_ROADMAP.md).  
Each row: what the method **cannot do well**, what the field still **lacks**, and the **logical next step** (from follow-up papers or our project).

### Summary table

| Paper | Year | Main limitation (plain English) | Fails or struggles on | What's still missing | Next step (field / ours) |
|-------|------|--------------------------------|----------------------|----------------------|--------------------------|
| **[Kamienny E2E](#kamienny-2022)** | 2022 | **Blind decoding** — picks tokens by language-model probability, not by whether the partial formula fits the data | Strogatz ODEs (~35.7% R²≥0.99); exact recovery; deeply nested formulas; OOD data | No search feedback during generation; one-shot guess | **TPSR / DGSR** add MCTS · **We:** stack DeSTrOI operator mask before decode |
| **[TPSR](#tpsr-2023)** | 2023 | **Heavy search cost** — MCTS + beam completion at every symbol; λ trade-off between accuracy and formula bloat | Real-time / large-scale deployment; still searches full operator alphabet; exact recovery not guaranteed | Cheap structural priors before MCTS; operator-level pruning | **SR4MDL** fixes objective · **We:** DeSTrOI masks operators **inside** TPSR expansion |
| **[DGSR-MCTS](#dgsr-mcts-2023)** | 2023 | **Extreme compute** — ~500k expression evaluations per problem; complex install & online fine-tuning | Laptop/small-lab reproduction; fast iteration; interpretable hyperparameters | Practical budget for 133 SRBench runs | Lighter hybrids (TPSR) · **We:** DeSTrOI as O(1) prior to shrink mutations |
| **[uDSR](#udsr-2022)** | 2022 | **Heavy engineering** — 5 subsystems (GP, RL, LSPT, simplification, poly); slow end-to-end | Quick prototyping; single-GPU neural pipeline; modular ablation | Unified but not simple; hard to add one new prior | Plug DeSTrOI into `function_set` pruning · reproduce via SRBench harness |
| **[SR4MDL](#sr4mdl-2025)** | 2025 | **Search + MDLformer cost**; needs checkpoint & MCTS/GP loop; newest, less reproduced | Noise robustness at scale; integration with existing E2E weights; operator-level priors | Combining MDL objective with cheap visual/structural priors | **We:** restrict grammar terminals via DeSTrOI before MDL search |
| **[DeSTrOI (original)](#destroi-aaai-2021)** | 2021 | **Operator ID only** — does not output full formulas; 6-op vocab; 2D image encoding | `sub`, `div`, `exp`, `cos`; high-D native path (k>2 needs Keras fix); no transformer integration in original paper | End-to-end SR pipeline with modern transformers | **Our project:** wire DeSTrOI → Kamienny / TPSR / uDSR |
| **[DeSTrOI + E2E (ours)](#our-work)** | — | **Modest gains so far**; mixed on 20 SRBench problems; only Kamienny wired | Full 133 benchmark; DeSTrOI + TPSR | DeSTrOI + TPSR on lab GPU |

---

### Kamienny 2022

| | |
|--|--|
| **Limitation** | Transformer decodes **without checking fit** until the full equation is written. Beam search explores token probability, not data reward. |
| **Evidence** | Strogatz R²≥0.99: **35.7%** (TPSR paper); ~4th on SRBench; poor exact recovery on ground-truth. |
| **Fails on** | Time-ordered ODE trajectories; OOD inputs; nested structure; cheating R² with long formulas. |
| **What's missing** | Any **in-loop** signal from the data (R², complexity) while building the formula. |
| **Next step** | TPSR (same backbone + MCTS). **Our step:** DeSTrOI operator mask → fewer wrong token branches before beam decode. |

---

### TPSR 2023

| | |
|--|--|
| **Limitation** | Fixes blind decoding with MCTS but is **slow** and still explores a **large operator vocabulary** unless λ is tuned carefully. |
| **Evidence** | λ=0 gives bloated formulas (complexity ~130 on black-box); λ=0.1 needed for balance; slower than plain E2E. |
| **Fails on** | Low-latency settings; problems where MCTS budget is too small; operator search space explosion. |
| **What's missing** | **Upfront** reduction of which operators to even consider — before spending rollouts. |
| **Next step** | Combine with structural priors. **Our step:** feed DeSTrOI `forbidden_token_ids` into TPSR's MCTS expansion (Phase B, lab GPU). |

---

### DGSR-MCTS 2023

| | |
|--|--|
| **Limitation** | Strongest search story but **prohibitively expensive** for student-scale replication. |
| **Evidence** | ~**500,000** expression evaluations per problem; online policy updates; unofficial repo fork. |
| **Fails on** | Full 133 SRBench on one machine; rapid hypothesis testing; fair ablation studies. |
| **What's missing** | SOTA accuracy at **TPSR-like** or lower compute budgets. |
| **Next step** | Lighter MCTS (TPSR) or better priors. **Our step:** test whether DeSTrOI reduces effective search width for mutation policies. |

---

### uDSR 2022

| | |
|--|--|
| **Limitation** | Best **symbolic recovery** in 2022 but **not a single model** — five strategies, separate conda env, long runs. |
| **Evidence** | 1st GECCO 2022 SRBench real-world; modular but heavy DSO stack. |
| **Fails on** | Fast neural-only baseline comparisons; easy "add one new idea" integration; quick professor-demo runs. |
| **What's missing** | Clean hook for a **visual operator prior** without reconfiguring entire pipeline. |
| **Next step** | Prune `function_set` from external signal. **Our step:** DeSTrOI scores → drop absent operators in uDSR config. |

---

### SR4MDL 2025

| | |
|--|--|
| **Limitation** | Best **exact recovery** push (~50/133) but optimizes **MDL**, not operators — still searches a large grammar. |
| **Evidence** | +43.92% recovery vs prior SOTA (paper); needs MDLformer + search loop. |
| **Fails on** | Noisy data (paper notes sensitivity); quick integration with Kamienny weights alone; operator-level pruning. |
| **What's missing** | Joint use of **description length** + **cheap operator identification** before search. |
| **Next step** | Restrict terminals early. **Our step:** DeSTrOI mask → smaller grammar → MDL search (future). |

---

### DeSTrOI (AAAI 2021)

| | |
|--|--|
| **Limitation** | Predicts **which operators appear**, not the full formula; trained on **6 ops** and 2D `(x,y)` landscapes. |
| **Evidence** | Original paper: operator classification + MIL for k≤10; no Kamienny/TPSR integration; `div`/`sub` not in vocab. |
| **Fails on** | Formulas needing only `cos`/`exp`; mis-prediction when landscape is ambiguous; modern Keras on k>2 (MIL bug). |
| **What's missing** | Connection to 2022–2025 transformer SR stack. |
| **Next step** | **Our entire project** — use DeSTrOI as prior for E2E, TPSR, uDSR, SR4MDL. |

---

### Our work (DeSTrOI + Kamienny today)

| | |
|--|--|
| **Limitation** | Only **plain E2E** integrated; small SRBench sample; DeSTrOI hurts when wrong (34/100 synthetic cases worse). |
| **Evidence** | Synthetic: 17% → 22% R²≥0.95; **20 SRBench:** E2E 7/20 R²≥0.99, DeSTrOI+E2E 6/20, mixed ΔR²; no TPSR yet. |
| **Fails on** | Claiming SOTA; high-D without MIL projections; `inv` vs `div` mismatch on SRBench; **SRBench operator masking often neutral or harmful**. |
| **What's missing** | DeSTrOI + TPSR on same 10 problems; noise sweep; exact recovery metric; 30–133 scale. |
| **Next step** | **TPSR ± DeSTrOI on lab GPU** (mask may help when combined with search, not blind beam decode) → noise & recovery experiments. |

---

### Limitations × SRBench taxonomy (where methods break)

| Challenge in our taxonomy | n (133 GT) | Who struggles | Why |
|---------------------------|------------|---------------|-----|
| **Strogatz** (time-ordered) | 14 | Kamienny ❌ | Training data = random points, not ODE trajectories |
| **Complex** formulas | 35 | Kamienny ❌, TPSR ⚠️ | Deep nesting needs more search budget |
| **High D** (D≥7) | 4 | All ⚠️ | Feature selection / projections; DeSTrOI 2D slice is approximate |
| **Trig-heavy** (`has_trig`) | ~40+ | Kamienny ⚠️ | Operator confusion in decode |
| **Exact ground-truth recovery** | 133 | Kamienny ❌, TPSR ⚠️ | R²≥0.99 ≠ symbolic match; SR4MDL/uDSR best here |
| **DeSTrOI vocab gap** | 4 not compatible | DeSTrOI ❌ | Formulas with only `cos`/`cot`/etc. outside 6-op set |

*Taxonomy:* [`results/srbench/taxonomy.csv`](../results/srbench/taxonomy.csv)

---

## 8. Cross-paper comparison table

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

## 9. Our experimental results

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

### D. SRBench ground-truth — 20 problems (E2E vs DeSTrOI+E2E)

**Setup:** 12 Feynman + 8 Strogatz · 75/25 split · seed=29910 · n_trees=10  
**Script:** [`benchmark_srbench_20.py`](../benchmark_srbench_20.py) · [`benchmark_subset_20.json`](../datasets/srbench/benchmark_subset_20.json)

| Metric | Transformer alone | DeSTrOI + Transformer |
|--------|-------------------|------------------------|
| Mean R² | 0.756 | 0.770 |
| Median R² | 0.967 | 0.968 |
| R² ≥ 0.99 | **7 / 20** | **6 / 20** |
| R² ≥ 0.95 | 12 / 20 | 12 / 20 |
| DeSTrOI operator accuracy | — | **58%** |

**By group (E2E):**

| Group | n | R²≥0.99 | Mean R² |
|-------|---|---------|---------|
| Feynman | 12 | **7 / 12** (58%) | 0.963 |
| Strogatz | 8 | **0 / 8** (0%) | 0.445 |

**Head-to-head:** mean ΔR² **+0.01** · better 4 · worse 4 · similar 12

Notable: DeSTrOI helped `strogatz_barmag2` (+0.33) and `shearflow2` (+0.30); hurt `strogatz_predprey1` (−0.15) and `strogatz_vdp1` (−0.27).

**CSV:** [srbench_20_benchmark.csv](../results/srbench/srbench_20_benchmark.csv) · [summary](../results/srbench/srbench_20_benchmark_summary.txt)

**Takeaway:** On this 20-problem pilot, Kamienny alone solves 7/20 at R²≥0.99 — not comparable to published 84.8% Feynman / 35.7% Strogatz (those are over all 119+14). DeSTrOI+E2E is mixed: small net gain on mean R², no clear win on success rate. Strogatz remains hard for plain E2E.

---

### D2. SRBench ground-truth — Kamienny only (legacy single run)

**Setup:** 75/25 split · seed=29910 · n_trees=10 · max_train_rows=2000  
**Script:** [`benchmark_srbench_gt.py`](../benchmark_srbench_gt.py)

| Problem | D | GT formula (short) | R²_test | ≥0.99? |
|---------|---|-------------------|---------|--------|
| feynman_III_10_19 | 4 | `mom*sqrt(Bx**2+By**2+Bz**2)` | **0.992** | ✅ |

**CSV:** [gt_benchmark.csv](../results/srbench/gt_benchmark.csv)  
**Target:** 30 problems (~3 h CPU) → then 133 overnight.

---

### E. Summary: what our numbers mean (don’t mix these up)

| Comparison | What it is | Takeaway |
|------------|------------|----------|
| **TPSR paper:** E2E vs TPSR on SRBench | Published baseline vs MCTS search | TPSR wins big (e.g. Feynman 84.8% → 94.9%) — **search matters** |
| **Our run:** Transformer vs DeSTrOI+Transformer | Same Kamienny model, ± operator mask | Synthetic: 17%→22% R²≥0.95; SRBench pilot (20): E2E 7/20 R²≥0.99, mixed ΔR² |
| **Our run vs TPSR paper** | ❌ **Not a fair comparison** | Different method (no MCTS), different data (synthetic vs SRBench) |
| **Future fair test** | DeSTrOI + TPSR vs TPSR alone on SRBench | Would actually test our hypothesis |

| Benchmark | Metric | E2E (published) | TPSR (published) | **Our E2E (20 SRBench)** | **Our DeSTrOI+E2E** |
|-----------|--------|-----------------|------------------|--------------------------|---------------------|
| Feynman | R²≥0.99 | 84.8% | 94.9% | **7/12** (58%)* | 6/20 total |
| Strogatz | R²≥0.99 | 35.7% | 82.8% | **0/8** (0%)* | — |
| Synthetic 6-op | R²≥0.95 | — | — | 17% | **22%** |
| In-domain 18-op | R²≥0.95 | 64%† | 70.8%† | **74%** | not tested |

\*20-problem pilot (subset A+B), not full 119+14 · †TPSR paper in-domain synthetic

---

## 10. SRBench taxonomy & data

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

*Last updated: 20-problem SRBench benchmark complete. See [srbench_20_benchmark_summary.txt](../results/srbench/srbench_20_benchmark_summary.txt).*
