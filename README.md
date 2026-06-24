# Symbolic Law Discovery

Hybrid symbolic regression: **DeSTrOI** (operator identification) + **NeurIPS 2022 Transformer** (end-to-end formula discovery), evaluated on synthetic trees and [SRBench](https://github.com/cavalab/srbench) ground-truth benchmarks.

## Repository layout

```
symbolic_law_discovery/
├── README.md
├── destroi_predict.py          # DeSTrOI inference (2D)
├── srbench_data.py             # SRBench Feynman/Strogatz data loader
├── results_paths.py            # Output path helpers
├── benchmark_*.py              # Batch benchmarks
├── demo*.py                    # Visual demos
├── results/                    # All benchmark outputs → see results/README.md
├── datasets/srbench/           # Problem list + cached PMLB data
├── Symbolic-Prediction-master/ # DeSTrOI (AAAI 2021)
└── symbolicregression/         # Transformer SR (Kamienny et al. 2022)
```

## Setup

1. **Transformer weights** — place `model.pt` in `symbolicregression/` (see [symbolicregression/README.md](symbolicregression/README.md)).
2. **DeSTrOI weights** — place `.h5` files in `Symbolic-Prediction-master/weights/` ([Google Drive](https://drive.google.com/drive/folders/1L0KR9uZQP60RYcSys1S-thXJ444FDnCh)).
3. Install Python deps: `torch`, `tensorflow`, `numpy`, `pandas`, `scikit-learn`, `sympy`, `matplotlib`, `requests`, `pyyaml`.

## Quick start

```bash
python3 destroi_predict.py
python3 demo_comparison_destroi_vocab.py
python3 benchmark_three_way.py --n 5 --seed 0
```

## Results

Committed benchmark CSVs, summaries, and figures live under [`results/`](results/README.md).
