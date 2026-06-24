# Symbolic Law Discovery

**DeSTrOI + Transformer symbolic regression on [SRBench](https://github.com/cavalab/srbench)**  
Visual operator identification as a prior for neural formula search.

> **Thesis (in progress):** After Kamienny 2022, the field moved to **hybrids** (MCTS, GP, MDL)—not pure transformers. We propose **DeSTrOI** as a cheap **operator-pruning prior** before transformer decoding, complementary to TPSR-style search.

---

## Findings so far

### What works today

| Result | Status | Link |
|--------|--------|------|
| DeSTrOI + Kamienny transformer pipeline | ✅ Implemented | [`demo_combined.py`](demo_combined.py) · [`benchmark_three_way.py`](benchmark_three_way.py) |
| Three-way comparison figure | ✅ | [`results/three_way/figures/three_way_comparison_destroi_vocab.png`](results/three_way/figures/three_way_comparison_destroi_vocab.png) |
| SRBench taxonomy (**133** Feynman + Strogatz) | ✅ | [`results/srbench/taxonomy.csv`](results/srbench/taxonomy.csv) |
| Kamienny on SRBench ground-truth | ⏳ 1/133 | [`results/srbench/gt_benchmark.csv`](results/srbench/gt_benchmark.csv) |
| TPSR / uDSR / DGSR-MCTS / SR4MDL ± DeSTrOI | 📋 Planned | See [roadmap](#roadmap) |

### Synthetic benchmark (100 random 6-op formulas)

| Metric | Transformer alone | DeSTrOI + Transformer |
|--------|-------------------|------------------------|
| Mean R² | −32.2 | −36.5 |
| R² ≥ 0.95 | 17% | **22%** |
| DeSTrOI operator accuracy | — | **74.3%** |
| Combined wins (ΔR² > 0) | — | **42 / 100** |
| Combined hurts (wrong operator ID) | — | **34 / 100** |

Full table: [`results/three_way/three_way_benchmark_n100_seed0.csv`](results/three_way/three_way_benchmark_n100_seed0.csv)

**Takeaway:** DeSTrOI helps when operator prediction is correct; **false negatives block true operators** and hurt combined performance.

### SRBench taxonomy (all 133 ground-truth problems)

| | Count |
|--|-------|
| Feynman | 119 |
| Strogatz | 14 |
| Simple / medium / complex | 28 / 70 / 35 |
| DeSTrOI-compatible (heuristic) | 129 |

Open in Excel/Numbers: [`results/srbench/taxonomy.csv`](results/srbench/taxonomy.csv)  
Summary: [`results/srbench/taxonomy_summary.txt`](results/srbench/taxonomy_summary.txt)

Columns include: ground-truth formula, `n_features`, `n_operators`, nestedness, operator flags, complexity tier.

### Early SRBench transformer result

| Problem | D | R²_test | R² ≥ 0.99 |
|---------|---|---------|-----------|
| `feynman_III_10_19` | 4 | **0.992** | ✅ |

---

## Roadmap

```
Phase 1 ✅  Repo, DeSTrOI↔Kamienny pipeline, SRBench taxonomy (133)
Phase 2 ⏳  Kamienny on 30–133 SRBench ground-truth problems
Phase 3 📋  TPSR ± DeSTrOI (operator-masked MCTS)
Phase 4 📋  uDSR with pruned function_set ± DeSTrOI
Phase 5 📋  SR4MDL / DGSR-MCTS on subset
Phase 6 📋  Paper: visual operator prior reduces search entropy
```

| Step | Task | Command / artifact |
|------|------|-------------------|
| ✅ | Build taxonomy | `python3 build_srbench_taxonomy.py` |
| ✅ | Synthetic three-way benchmark | `python3 benchmark_three_way.py --n 100` |
| ⏳ | Kamienny on SRBench (30 subset, ~3 h CPU) | `python3 benchmark_srbench_gt.py --n 30 --max-train-rows 2000` |
| 📋 | Install TPSR, cite/compare published SRBench numbers | [TPSR repo](https://github.com/deep-symbolic-mathematics/TPSR) |
| 📋 | Wire DeSTrOI → uDSR `function_set` | `dso-org/deep-symbolic-optimization` |
| 📋 | Extend DeSTrOI to k≤10 (MIL attention + Keras patch) | `Symbolic-Prediction-master/` |

**Blocked on laptop:** full 133 × 5 methods × {±DeSTrOI} needs **GPU/server** and per-method integration (~6 min/problem for Kamienny alone ≈ 14 h CPU).

---

## Literature landscape

Post-2022 symbolic regression: **transformer for speed, search for accuracy.**

| Method | Year | Mechanism | Strength | Weakness |
|--------|------|-----------|----------|----------|
| [**Kamienny E2E**](https://arxiv.org/abs/2204.10532) | 2022 | Transformer decode + BFGS | Fast; low complexity | Blind decoding; weak Strogatz/OOD |
| [**TPSR**](https://arxiv.org/abs/2303.06833) | 2023 | E2E + **MCTS** + R²/complexity reward | Fixes Strogatz & black-box | Slower; needs tuning (λ) |
| [**DGSR-MCTS**](https://proceedings.mlr.press/v202/kamienny23a.html) | 2023 | MCTS + **online mutation policy** | SOTA SRBench (paper) | Heavy compute |
| [**uDSR**](https://github.com/dso-org/deep-symbolic-optimization) | 2022 | **5-strategy hybrid** (GP+DSR+LSPT+…) | Best symbolic recovery | Slow; modular setup |
| [**SR4MDL**](https://github.com/tsinghua-fib-lab/SR4MDL) | 2025 | **MDL objective** + MCTS | Best exact recovery claim | Newest; search cost |
| **DeSTrOI + E2E (ours)** | — | **Visual operator ID** → block tokens | O(1) prior; implemented | 6-op vocab; 2D native |

### TPSR vs E2E on SRBench (published — [TPSR paper](https://arxiv.org/abs/2303.06833))

| Dataset | E2E Transformer | TPSR (λ=0.1) |
|---------|-----------------|--------------|
| Feynman R²≥0.99 | 84.8% | **94.9%** |
| Strogatz R²≥0.99 | 35.7% | **82.8%** |
| Black-box mean R² | 0.864 | **0.945** |

TPSR precomputed results: [srbench_results/](https://github.com/deep-symbolic-mathematics/TPSR/tree/main/srbench_results)

Extended write-ups: [`docs/PROFESSOR_BRIEF.md`](docs/PROFESSOR_BRIEF.md)

---

## Research questions

| Question | Answer |
|----------|--------|
| Is SR solved? | **No.** SRBench shows no single winner; exact recovery under noise is open. |
| Transformers alone? | **No.** Field consensus = **hybrids** (TPSR, uDSR, DGSR-MCTS, SR4MDL). |
| DeSTrOI before transformers? | **Useful prior** to prune operator vocabulary—not universal. Wired to Kamienny via `forbidden_token_ids`. |
| What's left? | Full SRBench runs, TPSR±DeSTrOI, k>2 DeSTrOI, exact recovery metrics, deployment. |
| Deployed in industry? | Mostly research; physics/materials PK; interpretability-driven domains. |

---

## Our approach

```
  (X, y)  →  DeSTrOI  →  operator scores  →  forbidden tokens  →  Transformer  →  formula
                ↑
         reads 2D landscape
         (6 ops: add, mul, inv, sqrt, log, sin)
```

**Novelty hook:** Cheaper than full MCTS (TPSR/DGSR)—one forward pass to shrink the search alphabet before neural decode.

### DeSTrOI integration plan (per baseline)

| Baseline | Integration | Status |
|----------|-------------|--------|
| Kamienny E2E | `forbidden_token_ids` in `fit()` | ✅ |
| TPSR | Mask ops in MCTS expansion | 📋 |
| uDSR / DSO | Prune `function_set` | 📋 |
| DGSR-MCTS | Mask mutation policy | 📋 |
| SR4MDL | Restrict search grammar | 📋 |

---

## Benchmark: SRBench

| Sub-benchmark | # problems | Ground truth? |
|---------------|------------|---------------|
| Feynman | ~119 | Yes |
| Strogatz | 14 | Yes (ODE trajectories) |
| Black-box | 122 | No |
| **Ground-truth total** | **133** | |

Data: [PMLB](https://github.com/EpistasisLab/penn-ml-benchmarks) via [`srbench_data.py`](srbench_data.py)  
Problem list: [`datasets/srbench/problem_list.json`](datasets/srbench/problem_list.json)

---

## Repository layout

```
symbolic_law_discovery/
├── README.md                 ← you are here (roadmap + findings)
├── docs/PROFESSOR_BRIEF.md   ← extended literature notes
├── destroi_predict.py        # DeSTrOI inference (2D)
├── srbench_data.py           # SRBench data loader
├── build_srbench_taxonomy.py # Generate taxonomy.csv
├── benchmark_srbench_gt.py   # Kamienny on SRBench
├── benchmark_three_way.py    # DeSTrOI + Transformer + Combined
├── demo_*.py                 # Visual demos
├── results/                  # Benchmark outputs → results/README.md
├── datasets/srbench/         # 133 problem names + cached PMLB data
├── Symbolic-Prediction-master/  # DeSTrOI (AAAI 2021)
└── symbolicregression/       # Kamienny transformer (NeurIPS 2022)
```

---

## Setup

1. **Transformer weights** — `symbolicregression/model.pt` ([upstream README](symbolicregression/README.md)) — *gitignored, download separately*
2. **DeSTrOI weights** — `Symbolic-Prediction-master/weights/*.h5` ([Google Drive](https://drive.google.com/drive/folders/1L0KR9uZQP60RYcSys1S-thXJ444FDnCh))
3. **Python deps:** `torch`, `tensorflow`, `numpy`, `pandas`, `scikit-learn`, `sympy`, `matplotlib`, `requests`, `pyyaml`

## Quick start

```bash
# DeSTrOI operator prediction
python3 destroi_predict.py

# Visual three-way comparison
python3 demo_comparison_destroi_vocab.py

# Synthetic benchmark (DeSTrOI + Transformer + Combined)
python3 benchmark_three_way.py --n 5 --seed 0

# SRBench taxonomy (no model needed)
python3 build_srbench_taxonomy.py

# Kamienny on SRBench (slow on CPU)
python3 benchmark_srbench_gt.py --n 30 --max-train-rows 2000
```

---

## Results index

| Folder | Contents |
|--------|----------|
| [`results/srbench/`](results/srbench/) | Taxonomy + Kamienny SRBench runs |
| [`results/three_way/`](results/three_way/) | DeSTrOI vs Transformer vs Combined |
| [`results/transformer/`](results/transformer/) | Transformer-only benchmarks |
| [`results/combined/`](results/combined/) | Combined pipeline figures |
| [`results/destroi/`](results/destroi/) | DeSTrOI encoding demos |

---

## References

- Kamienny et al. (2022) — [End-to-end symbolic regression with transformers](https://arxiv.org/abs/2204.10532)
- Shojaee et al. (2023) — [TPSR](https://arxiv.org/abs/2303.06833)
- Kamienny et al. (2023) — [DGSR-MCTS](https://proceedings.mlr.press/v202/kamienny23a.html)
- Landajuela et al. (2022) — [uDSR](https://papers.nips.cc/paper_files/paper/2022/hash/dbca58f35bddc6e4003b2dd80e42f838-Abstract-Conference.html)
- Yu et al. (2025) — [SR4MDL](https://proceedings.iclr.cc/paper_files/paper/2025/file/a402493de088886740b5939f666a6e56-Paper-Conference.pdf)
- La Cava et al. — [SRBench](https://github.com/cavalab/srbench)
- Xing et al. (2021) — DeSTrOI / AAAI ([`Symbolic-Prediction-master/`](Symbolic-Prediction-master/))
