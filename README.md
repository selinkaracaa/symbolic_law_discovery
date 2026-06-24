# Symbolic Law Discovery

**DeSTrOI + Transformer symbolic regression** on [SRBench](https://github.com/cavalab/srbench)  
Visual operator identification as a prior for neural formula search.

📖 **[Literature & results](docs/LITERATURE_AND_RESULTS.md)** · **[Limitations & next steps (all papers)](docs/LITERATURE_AND_RESULTS.md#7-limitations-gaps--next-steps-all-papers)** · **[Research roadmap](docs/RESEARCH_ROADMAP.md)**

---

## At a glance

| | |
|--|--|
| **Hypothesis** | DeSTrOI **on top of** existing SR methods (E2E, TPSR, uDSR) to prune operators — **not** a replacement for MCTS |
| **Implemented** | DeSTrOI → `forbidden_token_ids` → Kamienny 2022 transformer (**no TPSR yet**) |
| **Measured so far** | DeSTrOI helps plain transformer slightly on synthetic 6-op (22% vs 17% R²≥0.95) — **not** vs TPSR on SRBench |
| **Taxonomy** | All **133** SRBench ground-truth formulas classified |
| **Next** | Kamienny on 30 SRBench problems; TPSR / uDSR ± DeSTrOI |

---

## Findings so far (our runs)

| Experiment | n | Key result | Data |
|------------|---|------------|------|
| **Three-way** (DeSTrOI / Trans / Combined) | 100 | Comb R²≥0.95: **22%** vs Trans **17%**; DeSTrOI acc **74%** | [CSV](results/three_way/three_way_benchmark_n100_seed0.csv) · [Figure](results/three_way/figures/three_way_comparison_destroi_vocab.png) |
| **Transformer** (6-op synthetic) | 100 | Median R² **0.73**; R²≥0.95 **21%** | [Summary](results/transformer/synthetic_destroi_vocab/transformer_benchmark_summary_n100.txt) |
| **Transformer in-domain** (18-op) | 100 | Median R² **0.999**; R²≥0.95 **74%** | [Summary](results/transformer/indomain/transformer_indomain_benchmark_summary_n100_seed0.txt) |
| **SRBench 20** (E2E vs DeSTrOI+E2E) | 20 | E2E **7/20** R²≥0.99; Feynman 7/12; Strogatz 0/8 | [CSV](results/srbench/srbench_20_benchmark.csv) · [summary](results/srbench/srbench_20_benchmark_summary.txt) · [docs §9](docs/LITERATURE_AND_RESULTS.md#9-our-experimental-results) |
| **Taxonomy** | 133 | 119 Feynman + 14 Strogatz | [**taxonomy.csv**](results/srbench/taxonomy.csv) |

Published comparison (TPSR paper, not our runs): Feynman R²≥0.99 **84.8%** → TPSR **94.9%**; Strogatz **35.7%** → **82.8%**.  
→ [Full tables & paper summaries](docs/LITERATURE_AND_RESULTS.md)

---

## Roadmap

| Phase | Task | Status |
|-------|------|--------|
| **1** | SRBench taxonomy (133 formulas) + data pipeline | ✅ [roadmap](docs/RESEARCH_ROADMAP.md#phase-1--formula-taxonomy--data--done) |
| **2** | SOTA limitation analysis + DeSTrOI positioning | ✅ [roadmap](docs/RESEARCH_ROADMAP.md#phase-2--method-limitations--destroi-positioning--done) |
| **3** | SRBench 20-problem benchmark (E2E ± DeSTrOI) | ✅ | [srbench_20_benchmark.csv](results/srbench/srbench_20_benchmark.csv) |
| **4** | TPSR ± DeSTrOI (lab GPU) | 📋 |
| **5** | Proposal / paper | 📋 |

---

## SRBench taxonomy (133 formulas)

Open the spreadsheet:

- **GitHub:** [results/srbench/taxonomy.csv](results/srbench/taxonomy.csv) *(click → Download raw file if preview fails)*
- **Local:** `open results/srbench/taxonomy.csv`
- **Regenerate:** `python3 build_srbench_taxonomy.py`

Columns: `problem`, `gt_formula`, `n_features`, `n_operators`, `nestedness`, `complexity`, `ops`, `destroi_compatible`, …

---

## Quick start

```bash
# DeSTrOI operator prediction
python3 destroi_predict.py

# Three-way comparison figure
python3 demo_comparison_destroi_vocab.py

# Benchmarks
python3 benchmark_three_way.py --n 100 --seed 0
python3 benchmark_srbench_20.py --max-train-rows 2000   # 20 problems (~3–4 h CPU)
python3 benchmark_srbench_gt.py --n 30 --max-train-rows 2000   # ~3 h CPU
python3 build_srbench_taxonomy.py                              # no GPU needed
```

**Weights (not in repo):** `symbolicregression/model.pt` · DeSTrOI `.h5` from [Google Drive](https://drive.google.com/drive/folders/1L0KR9uZQP60RYcSys1S-thXJ444FDnCh)

---

## Repository

```
benchmark_*.py          # SRBench, three-way, transformer benchmarks
build_srbench_taxonomy.py
destroi_predict.py      # DeSTrOI inference
docs/LITERATURE_AND_RESULTS.md   # ← detailed paper reviews + results
results/                # CSVs, figures, summaries
Symbolic-Prediction-master/   # DeSTrOI (AAAI 2021)
symbolicregression/     # Kamienny transformer (NeurIPS 2022)
```

---

## References

[Kamienny 2022](https://arxiv.org/abs/2204.10532) · [TPSR 2023](https://arxiv.org/abs/2303.06833) · [DGSR-MCTS 2023](https://proceedings.mlr.press/v202/kamienny23a.html) · [uDSR 2022](https://github.com/dso-org/deep-symbolic-optimization) · [SR4MDL 2025](https://github.com/tsinghua-fib-lab/SR4MDL) · [SRBench](https://github.com/cavalab/srbench)
