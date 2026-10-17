import numpy as np

from cellresponse.baselines.metrics import energy_score, marginal_wasserstein, mean_shift_error
from cellresponse.goals import membership


def test_equal_means_and_different_tails_change_membership():
    tight = np.full(200, -0.5)
    wide = np.concatenate([np.full(100, -3.0), np.full(100, 2.0)])
    assert abs(tight.mean() - wide.mean()) < 1e-6
    tight_share = membership(tight, 0.0, 1.0, "up").mean()
    wide_share = membership(wide, 0.0, 1.0, "up").mean()
    assert wide_share > tight_share + 0.05


def test_energy_score_prefers_the_matching_variance():
    rng = np.random.default_rng(0)
    observed = rng.normal(size=(400, 2))
    matched = rng.normal(size=(400, 2))
    wide = rng.normal(scale=3.0, size=(400, 2))
    assert energy_score(matched, observed) < energy_score(wide, observed)
    assert mean_shift_error(matched, observed) < 0.4
    assert mean_shift_error(wide, observed) < 0.4


def test_energy_score_sees_dependence_missed_by_marginals():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(500, 2))
    coupled = np.column_stack([base[:, 0], 0.9 * base[:, 0] + 0.1 * base[:, 1]])
    independent = coupled.copy()
    independent[:, 1] = rng.permutation(independent[:, 1])
    assert energy_score(coupled, coupled) < energy_score(independent, coupled)
    assert (
        marginal_wasserstein(independent, coupled) < marginal_wasserstein(coupled, coupled) + 0.15
    )
