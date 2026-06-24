# Results

Benchmark outputs and figures from DeSTrOI, Transformer, and combined pipelines.

**Project roadmap & findings:** see [README.md](../README.md) at repo root.

## Layout

| Folder | Contents |
|--------|----------|
| `destroi/` | DeSTrOI-only visualizations |
| `transformer/` | NeurIPS 2022 Transformer benchmarks |
| `transformer/synthetic_destroi_vocab/` | Random 6-op trees (DeSTrOI generator) |
| `transformer/indomain/` | NeurIPS 18-op in-domain generator |
| `three_way/` | DeSTrOI + Transformer + Combined CSVs/summaries |
| `combined/` | Combined-pipeline demo figures |
| `srbench/` | SRBench Feynman + Strogatz ground-truth runs |
| `logs/` | Run logs (gitignored by default) |

## How to reproduce

```bash
# DeSTrOI encoding demo
python3 demo.py

# Transformer demo
python3 demo_neurips.py

# Three-way comparison figure
python3 demo_comparison_destroi_vocab.py

# Benchmarks
python3 benchmark_transformer.py --n 100 --seed 0
python3 benchmark_transformer_indomain.py --n 100 --seed 0
python3 benchmark_three_way.py --n 100 --seed 0
python3 benchmark_srbench_gt.py --n 5          # SRBench smoke test
python3 benchmark_srbench_gt.py --download     # fetch PMLB data first
```

All scripts write into this tree via `results_paths.py`.
