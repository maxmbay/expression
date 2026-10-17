"""Scoring rules for predicted and observed populations, plus the synthetic checks."""

from __future__ import annotations

import numpy as np

from cellresponse.goals import membership


def pairwise_mean(left: np.ndarray, right: np.ndarray, skip_diagonal: bool) -> float:
    distance = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=-1)
    if skip_diagonal:
        distance = distance[~np.eye(distance.shape[0], distance.shape[1], dtype=bool)]
    return float(distance.mean()) if distance.size else 0.0


def energy_score(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Energy distance between two samples. Lower is better. Diagonals are excluded."""
    return (
        2 * pairwise_mean(predicted, observed, skip_diagonal=False)
        - pairwise_mean(predicted, predicted, skip_diagonal=True)
        - pairwise_mean(observed, observed, skip_diagonal=True)
    )


def mean_shift_error(predicted: np.ndarray, observed: np.ndarray) -> float:
    return float(np.linalg.norm(predicted.mean(axis=0) - observed.mean(axis=0)))


def marginal_wasserstein(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Mean of one-dimensional Wasserstein-1 distances across coordinates."""
    width = predicted.shape[1]
    total = 0.0
    for column in range(width):
        left = np.sort(predicted[:, column])
        right = np.sort(observed[:, column])
        grid = np.linspace(0, 1, max(left.size, right.size), endpoint=False)
        total += float(np.abs(np.quantile(left, grid) - np.quantile(right, grid)).mean())
    return total / width


def sliced_wasserstein(
    predicted: np.ndarray, observed: np.ndarray, projections: np.ndarray
) -> float:
    total = 0.0
    for direction in projections:
        total += marginal_wasserstein(
            (predicted @ direction).reshape(-1, 1), (observed @ direction).reshape(-1, 1)
        )
    return total / len(projections)


def covariance_discrepancy(predicted: np.ndarray, observed: np.ndarray) -> float:
    return float(np.linalg.norm(np.cov(predicted, rowvar=False) - np.cov(observed, rowvar=False)))


def population_utility(
    samples: np.ndarray,
    columns: dict[str, int],
    center: dict[str, float],
    scale: dict[str, float],
    direction: dict[str, str],
    goal_id: str,
    penalty_id: str,
    penalty_weight: float,
) -> float:
    """Mean objective membership minus the weighted apoptosis membership."""
    objective = membership(
        samples[:, columns[goal_id]], center[goal_id], scale[goal_id], direction[goal_id]
    ).mean()
    penalty = membership(
        samples[:, columns[penalty_id]],
        center[penalty_id],
        scale[penalty_id],
        direction[penalty_id],
    ).mean()
    return float(objective - penalty_weight * penalty)


def distributional(
    predicted: np.ndarray, observed: np.ndarray, projections: np.ndarray
) -> dict[str, float]:
    return {
        "energy": energy_score(predicted, observed),
        "mean_shift": mean_shift_error(predicted, observed),
        "marginal_w1": marginal_wasserstein(predicted, observed),
        "sliced_w1": sliced_wasserstein(predicted, observed, projections),
        "covariance": covariance_discrepancy(predicted, observed),
    }
