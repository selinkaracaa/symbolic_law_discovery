# Literature reviews & experimental results

Detailed paper summaries (TPSR-style) and **our** benchmark tables.  
**Repo:** [github.com/selinkaracaa/symbolic_law_discovery](https://github.com/selinkaracaa/symbolic_law_discovery)

---

## Table of contents

1. [Kamienny 2022 — E2E Transformer](#1-kamienny-2022--e2e-transformer)
2. [uDSR 2022](#2-udsr-2022)
3. [TPSR 2023](#3-tpsr-2023--transformer-based-planning)
4. [DGSR-MCTS 2023](#4-dgsr-mcts-2023)
5. [SR4MDL 2025](#5-sr4mdl-2025)
6. [Literature overview](#6-literature-overview)
7. [Our experimental results](#7-our-experimental-results)
8. [SRBench taxonomy & data](#8-srbench-taxonomy--data)

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

## 2. uDSR 2022

**Paper:** *A Unified Framework for Deep Symbolic Regression* · NeurIPS 2022  
**Authors:** Landajuela et al. (LLNL / SRBench team)  
**Links:** [Paper](https://papers.nips.cc/paper_files/paper/2022/hash/dbca58f35bddc6e4003b2dd80e42f838-Abstract-Conference.html) · [Code](https://github.com/dso-org/deep-symbolic-optimization)

### Problem

Individual SR methods (GP, neural, pre-training) each fail on different problem types. No modular way to combine them.

### Proposed solution

The authors built a **unified assembly line**. Crucially, the Kamienny E2E Transformer is just **one interchangeable engine block** inside this machine, known as the **Large-Scale Pre-Training (LSPT)** module.

| Module | Role |
|--------|------|
| **Recursive simplification** | Takes the raw spreadsheet data and checks if it can mathematically simplify it first (e.g. stripping constants or isolating variables) to **shrink the problem size**. |
| **Large-scale pre-training (LSPT)** | Hands data to the pre-trained Transformer. The Transformer uses its intuition to instantly output a few highly likely equation **skeletons** (e.g. predicting that the equation probably looks like \(y = c_1 \cdot \sin(x) + c_2\)). |
| **Neural-guided search (DSR)** | An RL agent takes those skeletons and starts **filling in the blanks**, optimizing token selections based on rewards. |
| **Genetic programming (GP)** | If the neural components get stuck, a GP population search takes over — treating the Transformer's best guesses as **parents** and breeding them over generations to find alternative structures. |
| **Linear models (`poly` token)** | An ultra-fast linear solver instantly calculates exact numerical coefficients (\(c_1, c_2, \ldots\)) so the deep learning models don't waste time guessing precise decimals. |

Kamienny's transformer is **one interchangeable block**, not the whole system.

### Results (published SRBench)

- **Highest symbolic recovery** on ground-truth problems at time of publication.
- **1st place** GECCO 2022 SRBench real-world track.
- Black-box: on accuracy–complexity **Pareto frontier** with other top methods.

### Takeaway

*Don't pick transformer OR GP—wire them together.*

---

## 3. TPSR 2023 — Transformer-based Planning

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

## 4. DGSR-MCTS 2023

**Paper:** *Deep Generative Symbolic Regression with Monte-Carlo-Tree-Search* · ICML 2023  
**Authors:** Kamienny, Lample, Lamprier, Virgolin (**same lead author as E2E**)  
**Links:** [Proceedings](https://proceedings.mlr.press/v202/kamienny23a.html) · [Code](https://github.com/vanderschaarlab/DeepGenerativeSymbolicRegression)

### Problem

In the original **2022 E2E Transformer**, the AI worked like a person playing trivia. It looked at the dataset, made a **single forward-pass guess**, and spit out an equation in a few milliseconds. If the guess was right, great. If it was wrong, the model had **no way to fix its mistake** — it couldn't go back and edit the formula.

**Why did the authors need a search loop?** Standard transformers are heavily dependent on their training data. If you show the E2E transformer a physical pattern or a range of numbers it has never seen before in its synthetic training set, its "predictive text" logic breaks down. On complex real-world datasets (like SRBench), a single blind guess almost never finds the exact mathematical law. You need to **explore alternative formulas**.

### Proposed solution

**DGSR-MCTS** introduces an evolutionary approach. Instead of guessing the whole equation perfectly in one shot, the AI starts with a baseline guess and then **mutates (edits) the equation over and over**, keeping the good changes and throwing away the bad ones.

The clever twist: they **don't use the Transformer to write the equation from scratch**. They pre-train it to be an expert **equation editor**. Its only job is to look at an existing math formula and suggest smart mutations — such as "swap that `+` for a `*`" or "wrap this variable in a `sin()` function."

**DGSR-MCTS:** MCTS seeded with a pre-trained **mutation policy** (transformer-based). The policy is **fine-tuned online** from successful search trajectories.

### Method (step by step)

Every time the model is given a new dataset, it runs a heavy search loop (up to **500,000** equation evaluations per problem):

1. **Selection** — The algorithm looks at a tree of previously attempted equations and selects one that looks promising but still needs improvement.
2. **Mutation (the Transformer's job)** — The pre-trained Transformer takes that selected equation, looks at the target dataset, and proposes a set of intelligent mathematical mutations. It doesn't guess randomly; it uses its training history to suggest edits that actually make sense.
3. **Evaluation** — The system builds the new mutated equations, tests them against the data, and grades them based on how well they fit (R² ≥ 0.99 counts as "solved" in the paper).
4. **Backpropagation** — Good mutation paths get reinforced in the search tree; bad branches get deprioritized.
5. **Online update (the core innovation)** — If the Transformer suggests a mutation that successfully makes the equation more accurate, the system **fine-tunes the Transformer on the fly right during the test**. The AI dynamically learns what works for this specific dataset as it searches.

**MCTS loop in one line:** select expression → mutate via policy → evaluate fit → backpropagate → online update.

### Results (published SRBench)

- Claims **state-of-the-art on SRBench** at time of publication (Pareto rank 0 on both Feynman and black-box — best accuracy–simplicity trade-off).
- Trades **speed** for **search depth** and OOD robustness.
- Official sequel from Meta team: *"transformer alone isn't enough."*

#### SRBench — ground-truth Feynman (119 datasets, ≤10 features)

| Metric | E2E Transformer | DGSR-MCTS |
|--------|-----------------|-----------|
| **R² ≥ 0.99** (success rate) | **87%** | 80% |
| **Avg. expression size** (after SymPy simplify) | 121 | **33** |

DGSR-MCTS finds **shorter, simpler** formulas. Raw success rate on Feynman is actually slightly lower than E2E, but the formulas are ~3.7× smaller while still fitting well.

#### SRBench — black-box (57 datasets)

| Metric | E2E Transformer | DGSR-MCTS |
|--------|-----------------|-----------|
| **Median test R²** | 0.797 | **0.846** |
| **Avg. expression size** | 61 | **41** |

#### Synthetic stress test (paper Table 4 — not SRBench)

Before SRBench, the authors tested on **1,000 synthetic formulas** (500 like training, 500 harder "out-of-distribution" with bigger expressions). This shows **why search matters**:

| Approach | What it does | Hard OOD formulas solved (R² ≥ 0.99) |
|----------|--------------|--------------------------------------|
| One-shot decode (like E2E) | Write whole formula in one pass | **16.8%** |
| Mutation + MCTS (paper config) | Edit formula step-by-step with search | **44.0%** |

On formulas harder than training, search more than doubles success.

#### Computational cost

| Model | Evaluations per problem | Speed |
|-------|-------------------------|-------|
| **Kamienny 2022 (E2E)** | 1 (one-shot) | **~seconds** |
| **DGSR-MCTS** | up to **500,000** | **hours** (24 h cap per problem in paper) |

Why so many evaluations? DGSR-MCTS starts with a wide-open math vocabulary. Every mutation step considers probabilities over the full operator dictionary (`add`, `sub`, `mul`, `div`, `sin`, `cos`, `exp`, `log`, `sqrt`, …). That branching factor × thousands of tree nodes = hundreds of thousands of evaluations just to find one law.

### vs Kamienny 2022

| | E2E | DGSR-MCTS |
|--|-----|-----------|
| Decode | One-shot | Many iterations |
| Transformer's job | Write full formula | **Edit** existing formula |
| Weights at test | Fixed | Online updates |
| Speed | Fast (~seconds) | Slow (up to 500k evals) |
| OOD / hard benchmarks | Weaker | Stronger |

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

## 6. Literature overview

Single reference table: **published SRBench results**, **main limitations**, and **where DeSTrOI could plug in**.  
See [research roadmap](RESEARCH_ROADMAP.md) for phase details.

| Method | Year | Published SRBench results | Main limitation | How DeSTrOI could help |
|--------|------|---------------------------|-----------------|------------------------|
| **Kamienny E2E** | 2022 | Feynman **84.8%** R²≥0.99 · Strogatz **35.7%** · black-box median R² **0.864** (TPSR paper) | **Blind one-shot decode** — no fit feedback while building the formula | Mask absent operators via `forbidden_token_ids` before beam decode ✅ tested |
| **uDSR** | 2022 | Top **symbolic recovery** on ground-truth · GECCO 2022 real-world **1st** · Pareto front on black-box | **Heavy pipeline** — 5 subsystems (GP, RL, LSPT, simplification, poly); hard to extend | Prune `function_set` from DeSTrOI scores before search 📋 planned |
| **TPSR** | 2023 | Feynman **94.9%** · Strogatz **82.8%** · black-box median R² **0.945** (λ=0.1) | **Slow MCTS** — still searches full operator alphabet unless λ tuned | Mask tokens inside MCTS expansion 📋 planned (lab GPU) |
| **DGSR-MCTS** | 2023 | Feynman **80%** R²≥0.99 (simpler formulas, size **33**) · black-box median R² **0.846** · Pareto rank **0** | **Extreme compute** — up to **500k** evals/problem; online fine-tuning | Shrink mutation vocabulary before tree search 📋 planned |
| **SR4MDL** | 2025 | ~**50/133 exact** recovery (+43.9% vs prior SOTA, paper) | Optimizes **MDL**, not operators — still large grammar | Restrict terminals before MDL search 📋 planned |
| **DeSTrOI + E2E (ours)** | — | 20-problem pilot: E2E **7/20** R²≥0.99 · DeSTrOI+E2E **6/20** · synthetic 6-op: **17%→22%** R²≥0.95 | Only plain E2E wired; masking **hurts** when DeSTrOI is wrong (~34/100 synthetic) | Full project: stack prior on search methods above |

**Our 20-problem SRBench numbers are not comparable to published full-benchmark rates** (119 Feynman + 14 Strogatz).

---

## 7. Our experimental results

*All runs local (Mac CPU). Not yet full SRBench 133.*

### What each benchmark tests

| Benchmark | What it measures | Transformer in this run |
|-----------|------------------|-------------------------|
| **A — Three-way** | **Paired** test: same 100 formulas, transformer **alone vs DeSTrOI+transformer** | Kamienny on **6-op DeSTrOI trees** (out-of-distribution for 18-op model) |
| **B — Transformer only** | Same 6-op generator, **no DeSTrOI** — diagnostic baseline | Same Kamienny model, transformer only |
| **C — In-domain** | Kamienny's **own 18-op training generator** | Shows best-case transformer performance |
| **SRBench 20** | Real physics/ODE data from [taxonomy](../results/srbench/taxonomy.csv) | E2E vs DeSTrOI+E2E on 12 Feynman + 8 Strogatz |

**A vs B:** Both use the DeSTrOI 6-operator generator, but **A runs DeSTrOI+transformer on every formula** (head-to-head). **B only runs the transformer** — no masking, no operator-ID accuracy. Use **A** for "does DeSTrOI help?"; use **B** for "how hard is this generator for Kamienny alone?"

---

### A. Three-way: DeSTrOI vs Transformer vs Combined (synthetic 6-op)

**Setup:** Random formulas from DeSTrOI generator · 2 vars · 6 operators · n=100 · seed=0  
**Script:** [`benchmark_three_way.py`](../benchmark_three_way.py)

| Metric | Transformer alone | DeSTrOI + Transformer |
|--------|-------------------|------------------------|
| Mean R² | −32.20 | −36.50 |
| R² ≥ 0.95 | **17%** (17/100) | **22%** (22/100) |
| R² < 0 (catastrophic) | 38% | 33% |
| DeSTrOI operator accuracy | — | **74.3%** |
| Combined **better** (ΔR² > 0.005) | — | **42 / 100** |
| Combined **worse** | — | **34 / 100** |

**CSV:** [three_way_benchmark_n100_seed0.csv](../results/three_way/three_way_benchmark_n100_seed0.csv) · **Figure:** [three_way_comparison_destroi_vocab.png](../results/three_way/figures/three_way_comparison_destroi_vocab.png)

**When does masking help or hurt?**

| Pattern | What happens |
|---------|--------------|
| DeSTrOI **correct** (high operator accuracy) | Combined often wins — fewer wrong token branches |
| DeSTrOI **blocks a true operator** | Combined loses — transformer cannot recover the right structure |
| **`inv` in formula** (n=52) | Worst case: mean R² −61.6 (trans) / −69.7 (comb) — Kamienny uses `div`, DeSTrOI uses `inv`; mismatch amplifies failures |
| **`log` in formula** (n=57) | Mean R² −38.0 / −54.1 — nested `log`/`inv` trees are unstable for blind decode |
| **`sin` in formula** (n=51) | Mean R² −63.5 / −71.8 — deep unary nesting → catastrophic negative R² |
| **`sqrt` in formula** (n=55) | Transformer alone less bad (−0.3) but combined still dragged down (−55.7) when wrong ops blocked |
| **Nested unaries** (mul + 2 unary ops, n=5) | Hardest structure — combined worse in 2/5 cases |

---

### B. Transformer only — DeSTrOI-vocab synthetic trees

**Setup:** Same 6-op generator · n=100 · seed=0 · **no DeSTrOI**  
**Script:** [`benchmark_transformer.py`](../benchmark_transformer.py)

| Metric | Value |
|--------|-------|
| Mean R² | 0.16 |
| Median R² | 0.73 |
| R² ≥ 0.95 | **21%** |
| R² ≥ 0.50 | 62% |
| R² < 0 | 30% |

**CSV:** [transformer_benchmark_summary_n100.txt](../results/transformer/synthetic_destroi_vocab/transformer_benchmark_summary_n100.txt)

| Subgroup | Mean R² | Fail rate (R²<0.5) |
|----------|---------|---------------------|
| has `inv` (n=53) | 0.05 | 36% |
| has `log` (n=61) | −0.00 | 43% |
| mul+2 unaries (n=6) | −2.09 | 83% |

**vs published:** Kamienny reports **64%** R²≥0.95 on in-domain synthetic (TPSR paper). Our **21%** on 6-op DeSTrOI trees shows how far off-distribution this vocab is for the pretrained model.

---

### C. Transformer in-domain (NeurIPS 18-op generator)

**Setup:** Model's **own training distribution** · n=100 · seed=0  
**Script:** [`benchmark_transformer_indomain.py`](../benchmark_transformer_indomain.py)

| Metric | Our run | Published (TPSR paper, in-domain) |
|--------|---------|-----------------------------------|
| Median R² | **0.999** | — |
| R² ≥ 0.95 | **74%** | E2E **64%** · TPSR (λ=0.1) **70.8%** |
| R² ≥ 0.50 | 80% | — |
| Mean R² | −2.42 (outlier-skewed) | — |
| Exact string recovery | 0% | — |

**Takeaway:** In-domain, the transformer works as advertised (median R² ≈ 1). The gap to Section A/B is the **operator vocabulary and formula distribution**, not a broken install.

---

### SRBench 20-problem pilot (E2E vs DeSTrOI+E2E)

**Setup:** 12 Feynman + 8 Strogatz · 75/25 split · seed=29910 · n_trees=10  
**Script:** [`benchmark_srbench_20.py`](../benchmark_srbench_20.py) · problem list: [`benchmark_subset_20.json`](../datasets/srbench/benchmark_subset_20.json)

#### Results vs published

| Metric | E2E (ours) | DeSTrOI+E2E (ours) | Published E2E | Published TPSR (λ=0.1) |
|--------|------------|---------------------|---------------|------------------------|
| Mean R² (all 20) | 0.756 | 0.770 | — | — |
| Median R² | 0.967 | 0.968 | — | — |
| R² ≥ 0.99 | **7 / 20** | 6 / 20 | Feynman **84.8%** · Strogatz **35.7%** (full benchmark) | Feynman **94.9%** · Strogatz **82.8%** |
| R² ≥ 0.95 | 12 / 20 | 12 / 20 | — | — |
| DeSTrOI operator accuracy | — | **58%** | — | — |

#### By group (E2E vs DeSTrOI+E2E)

| Group | n | E2E R²≥0.99 | DeSTrOI+E2E R²≥0.99 | E2E mean R² | DeSTrOI+E2E mean R² |
|-------|---|-------------|---------------------|-------------|---------------------|
| Feynman | 12 | **7 / 12** (58%) | 6 / 12 (50%) | 0.963 | 0.967 |
| Strogatz | 8 | **0 / 8** (0%) | 0 / 8 (0%) | 0.445 | 0.473 |

**Head-to-head:** mean ΔR² **+0.014** · better 4 · worse 4 · similar 12

Notable wins: `strogatz_barmag2` (+0.33), `strogatz_shearflow2` (+0.30). Notable losses: `strogatz_predprey1` (−0.15), `strogatz_vdp1` (−0.27).

**CSV:** [srbench_20_benchmark.csv](../results/srbench/srbench_20_benchmark.csv) · [summary](../results/srbench/srbench_20_benchmark_summary.txt)

#### How the 20 problems were chosen (from taxonomy)

Problems were picked from [`taxonomy.csv`](../results/srbench/taxonomy.csv) to cover both benchmarks and a range of difficulty — not a random sample of all 133.

| Selection criterion | Count in our 20 |
|---------------------|-----------------|
| Feynman | 12 |
| Strogatz | 8 |
| Medium complexity | 13 |
| Complex | 6 |
| Simple | 1 |
| D = 2 | 11 |
| D ≥ 6 (high-D stress) | 3 |
| Has trig in ground truth | 5 |
| DeSTrOI-compatible (heuristic) | 20 / 20 |

| Problem | Benchmark | D | GT operators | Complexity |
|---------|-----------|---|--------------|------------|
| feynman_III_10_19 | Feynman | 4 | add,mul,pow,sqrt | medium |
| feynman_II_27_18 | Feynman | 2 | mul,pow | simple |
| feynman_I_14_4 | Feynman | 2 | div,mul,pow | medium |
| feynman_III_15_14 | Feynman | 3 | div,mul,pow | medium |
| feynman_II_11_3 | Feynman | 5 | div,mul,pow,sub | medium |
| feynman_III_4_32 | Feynman | 4 | div,exp,mul | medium |
| strogatz_barmag1 | Strogatz | 2 | mul,sin,sub | medium |
| strogatz_lv1 | Strogatz | 2 | mul,pow,sub | medium |
| strogatz_predprey1 | Strogatz | 2 | add,div,mul,sub | medium |
| strogatz_vdp1 | Strogatz | 2 | div,mul,pow,sub | medium |
| feynman_III_19_51 | Feynman | 5 | div,mul,pow,sub | complex |
| feynman_III_9_52 | Feynman | 6 | div,mul,pow,sin,sub | complex |
| feynman_I_9_18 | Feynman | 9 | add,div,mul,pow,sub | complex |
| feynman_II_6_15a | Feynman | 6 | add,div,mul,pow,sqrt | complex |
| feynman_I_6_2 | Feynman | 2 | div,exp,mul,pow,sqrt,sub | complex |
| feynman_I_41_16 | Feynman | 5 | div,exp,mul,pow | complex |
| strogatz_bacres1 | Strogatz | 2 | add,div,mul,pow,sub | medium |
| strogatz_barmag2 | Strogatz | 2 | mul,sin,sub | medium |
| strogatz_glider1 | Strogatz | 2 | mul,pow,sin,sub | medium |
| strogatz_shearflow2 | Strogatz | 2 | add,cos,mul,pow,sin | medium |

#### Conclusions from the 20-problem pilot

1. **E2E alone is strong on Feynman** (7/12 at R²≥0.99) but **fails all Strogatz** — matches the published pattern (Strogatz is OOD for Kamienny).
2. **DeSTrOI+E2E does not beat E2E on success rate** (6/20 vs 7/20) but slightly improves mean R² (+0.014) on hard Strogatz cases when masking is right.
3. **Operator-ID accuracy drops on real data** (58% vs 74% synthetic) — SRBench formulas use `div`/`sub`/`pow` outside DeSTrOI's 6-op vocab.
4. **Not comparable to published 84.8% / 35.7%** — those are rates over all 119+14 problems; our 20 are a taxonomy-guided stress sample.
5. **Next step:** DeSTrOI + **TPSR** (search, not blind beam) on the same 20 — the fair test of the hypothesis.

---

## 8. SRBench taxonomy & data


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

*Last updated: restructured literature overview + 20-problem SRBench pilot. See [srbench_20_benchmark_summary.txt](../results/srbench/srbench_20_benchmark_summary.txt).*
