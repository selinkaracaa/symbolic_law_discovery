"""Central paths for benchmark outputs and figures."""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")

DESTROI = os.path.join(RESULTS, "destroi")
DESTROI_FIGURES = os.path.join(DESTROI, "figures")

TRANSFORMER = os.path.join(RESULTS, "transformer")
TRANSFORMER_FIGURES = os.path.join(TRANSFORMER, "figures")
TRANSFORMER_SYNTH = os.path.join(TRANSFORMER, "synthetic_destroi_vocab")
TRANSFORMER_INDOMAIN = os.path.join(TRANSFORMER, "indomain")

THREE_WAY = os.path.join(RESULTS, "three_way")
THREE_WAY_FIGURES = os.path.join(THREE_WAY, "figures")

COMBINED = os.path.join(RESULTS, "combined")
COMBINED_FIGURES = os.path.join(COMBINED, "figures")

SRBENCH = os.path.join(RESULTS, "srbench")
LOGS = os.path.join(RESULTS, "logs")

_ALL_DIRS = (
    DESTROI_FIGURES,
    TRANSFORMER_FIGURES,
    TRANSFORMER_SYNTH,
    TRANSFORMER_INDOMAIN,
    THREE_WAY,
    THREE_WAY_FIGURES,
    COMBINED_FIGURES,
    SRBENCH,
    LOGS,
)


def ensure_dirs():
    for d in _ALL_DIRS:
        os.makedirs(d, exist_ok=True)


def join_results(*parts):
    return os.path.join(RESULTS, *parts)
