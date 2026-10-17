"""Prespecified module goals. Membership is literature; thresholds are fit later.

Scores are the mean of log1p expression after normalizing each cell to 10,000 counts.
A missing gene counts as zero, so every cell is scored on the same gene list. The
dataset's own `S_score` and `G2M_score` columns are not used: those were computed
upstream, outside this contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from cellresponse.genesets import hallmark_2020

VERSION = "hallmark-2020"

# Quantiles of the training reference-vehicle score. Down goals call the low tail
# an event; up goals and the apoptosis constraint call the high tail an event.
_GOALS = (
    {
        "goal_id": "g2m_down",
        "title": "G2/M proliferation down",
        "hallmark": "G2-M Checkpoint",
        "direction": "down",
        "role": "objective",
        "quantile": 0.10,
    },
    {
        "goal_id": "p53_up",
        "title": "p53 / DNA-damage response",
        "hallmark": "p53 Pathway",
        "direction": "up",
        "role": "objective",
        "quantile": 0.90,
    },
    {
        "goal_id": "apoptosis_constraint",
        "title": "Apoptosis constraint",
        "hallmark": "Apoptosis",
        "direction": "up",
        "role": "constraint",
        "quantile": 0.90,
    },
    {
        "goal_id": "interferon_gamma_up",
        "title": "Interferon-gamma response",
        "hallmark": "Interferon Gamma Response",
        "direction": "up",
        "role": "objective",
        "quantile": 0.90,
    },
)


@dataclass(frozen=True)
class Goal:
    goal_id: str
    title: str
    hallmark: str
    direction: str
    role: str
    quantile: float
    symbols: tuple[str, ...]

    def event_rule(self) -> str:
        return (
            f"logistic membership centered at the q{self.quantile:.2f} of protocol-A "
            "training reference-vehicle scores, width equal to their standard deviation"
        )


def goals() -> tuple[Goal, ...]:
    sets = hallmark_2020()["sets"]
    return tuple(Goal(symbols=tuple(sets[spec["hallmark"]]), **spec) for spec in _GOALS)


def vocabulary_columns(goal: Goal, genes: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Column indices of member genes present in the vocabulary, plus symbols that are absent."""
    wanted = pl.DataFrame({"gene_symbol": list(goal.symbols)})
    found = wanted.join(genes.select("gene_symbol", "column"), on="gene_symbol", how="left")
    missing = found.filter(pl.col("column").is_null())["gene_symbol"].to_list()
    columns = found.filter(pl.col("column").is_not_null())["column"].to_numpy().astype(np.int32)
    return columns, missing


def score_rows(matrix: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Mean over `columns` of a cells-by-genes matrix. An empty column list is an error."""
    if columns.size == 0:
        raise ValueError("goal has no genes in the scored space")
    return matrix[:, columns].mean(axis=1)


SCALE_FLOOR = 1e-3


def fit_soft(scores: np.ndarray, quantile: float) -> tuple[float, float]:
    """Center and width of the membership curve, from training solvent scores only."""
    threshold = float(np.quantile(scores, quantile))
    scale = max(float(np.std(scores)), SCALE_FLOOR)
    return threshold, scale


def membership(scores: np.ndarray, threshold: float, scale: float, direction: str) -> np.ndarray:
    """Smooth share. One at the goal side, zero on the other, one half at the solvent percentile."""
    scale = max(float(scale), SCALE_FLOOR)
    if direction == "down":
        z = (threshold - scores) / scale
    elif direction == "up":
        z = (scores - threshold) / scale
    else:
        raise ValueError(f"unknown goal direction {direction!r}")
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def passes_event(scores: np.ndarray, threshold: float, direction: str) -> np.ndarray:
    if direction == "down":
        return scores <= threshold
    if direction == "up":
        return scores >= threshold
    raise ValueError(f"unknown goal direction {direction!r}")
