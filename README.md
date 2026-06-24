# Symbolic Law Discovery

**DeSTrOI + Transformer symbolic regression** on [SRBench](https://github.com/cavalab/srbench)  
Visual operator identification as a prior for neural formula search.

📖 **[Full literature reviews & all experimental results →](docs/LITERATURE_AND_RESULTS.md)**

---

## At a glance

| | |
|--|--|
| **Hypothesis** | DeSTrOI can **prune the operator search space** before transformer decoding—cheaper than full MCTS (TPSR / DGSR). |
| **Implemented** | DeSTrOI → `forbidden_token_ids` → Kamienny 2022 transformer |
| **Taxonomy** | All **133** SRBench ground-truth formulas classified |
| **Next** | Kamienny on 30 SRBench problems; TPSR / uDSR ± DeSTrOI |

---

## Findings so far (our runs)

| Experiment | n | Key result | Data |
|------------|---|------------|------|
| **Three-way** (DeSTrOI / Trans / Combined) | 100 | Comb R²≥0.95: **22%** vs Trans **17%**; DeSTrOI acc **74%** | [CSV](results/three_way/three_way_benchmark_n100_seed0.csv) · [Figure](results/three_way/figures/three_way_comparison_destroi_vocab.png) |
| **Transformer** (6-op synthetic) | 100 | Median R² **0.73**; R²≥0.95 **21%** | [Summary](results/transformer/synthetic_destroi_vocab/transformer_benchmark_summary_n100.txt) |
| **Transformer in-domain** (18-op) | 100 | Median R² **0.999**; R²≥0.95 **74%** | [Summary](results/transformer/indomain/transformer_indomain_benchmark_summary_n100_seed0.txt) |
| **Kamienny on SRBench** | 1 | R²=**0.992** on `feynman_III_10_19` | [CSV](results/srbench/gt_benchmark.csv) |
| **Taxonomy** | 133 | 119 Feynman + 14 Strogatz | [**taxonomy.csv**](results/srbench/taxonomy.csv) |

Published comparison (TPSR paper, not our runs): Feynman R²≥0.99 **84.8%** → TPSR **94.9%**; Strogatz **35.7%** → **82.8%**.  
→ [Full tables & paper summaries](docs/LITERATURE_AND_RESULTS.md)

---

## Roadmap

| Phase | Task | Status |
|-------|------|--------|
| 1 | DeSTrOI + Kamienny pipeline, repo | ✅ |
| 2 | SRBench taxonomy (133 formulas) | ✅ |
| 3 | Synthetic benchmarks (three-way, transformer) | ✅ |
| 4 | Kamienny on 30–133 SRBench | ⏳ |
| 5 | TPSR / uDSR / SR4MDL ± DeSTrOI | 📋 |
| 6 | Paper | 📋 |

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
