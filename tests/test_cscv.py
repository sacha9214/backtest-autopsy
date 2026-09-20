import numpy as np
import pytest

from autopsy.cscv import pbo


def noise_matrix(n_periods=520, n_strategies=20, seed=0):
    """Strategies with no skill whatsoever: pure independent noise."""
    return np.random.default_rng(seed).normal(0, 0.01, size=(n_periods, n_strategies))


def pbo_over_seeds(n_seeds=12, edge=0.0, edge_column=3, **kwargs):
    values = []
    for seed in range(n_seeds):
        returns = noise_matrix(seed=seed, **kwargs)
        if edge:
            returns[:, edge_column] += edge
        values.append(pbo(returns, n_blocks=10).pbo)
    return np.array(values)


class TestPBO:
    def test_no_skill_selection_is_a_coin_flip_on_average(self):
        """With no real edge anywhere, the in-sample winner is random OOS.

        This only holds in expectation -- see the dispersion test below.
        """
        assert pbo_over_seeds().mean() == pytest.approx(0.5, abs=0.15)

    def test_a_single_run_is_too_noisy_to_convict(self):
        """Measured, not assumed: PBO on pure noise has a standard deviation
        around 0.2, so one run landing at 0.7 is not evidence of overfitting.
        Any report built on this number has to say so.
        """
        values = pbo_over_seeds(n_seeds=20)
        assert values.std() > 0.10
        assert values.max() - values.min() > 0.40

    def test_a_real_edge_leaves_no_ambiguity(self):
        """The one direction PBO is decisive in: a genuine edge pins it at zero."""
        values = pbo_over_seeds(n_seeds=12, edge=0.004)
        assert values.max() == 0.0

    def test_genuine_edge_is_detected(self):
        """One strategy with a persistent edge should survive every split."""
        returns = noise_matrix(seed=1)
        returns[:, 3] += 0.004
        result = pbo(returns, n_blocks=10)
        assert result.pbo < 0.1

    def test_split_count_is_n_choose_half(self):
        result = pbo(noise_matrix(), n_blocks=10)
        assert result.n_splits == 252

    def test_slope_is_mechanical_not_informative(self):
        """A dominant strategy pins the slope at -1, noise leaves it scattered.

        This is the opposite of the intuitive reading, which is why the slope
        is kept out of the rendered verdict. Regression test for that finding.
        """
        noise_slopes = np.array([
            pbo(noise_matrix(seed=s, n_strategies=44, n_periods=2000),
                n_blocks=10).performance_degradation[0]
            for s in range(6)
        ])
        edged = noise_matrix(seed=99, n_strategies=44, n_periods=2000)
        edged[:, 3] += 0.004
        edge_slope = pbo(edged, n_blocks=10).performance_degradation[0]

        assert edge_slope == pytest.approx(-1.0, abs=0.01)
        assert noise_slopes.std() > 0.1
        assert noise_slopes.mean() > edge_slope

    def test_genuine_edge_loses_money_rarely(self):
        returns = noise_matrix(seed=3)
        returns[:, 7] += 0.004
        assert pbo(returns, n_blocks=10).probability_of_loss < 0.1

    def test_ragged_periods_are_truncated_not_reweighted(self):
        clean = pbo(noise_matrix(n_periods=500, seed=4), n_blocks=10)
        ragged = pbo(noise_matrix(n_periods=507, seed=4), n_blocks=10)
        assert clean.pbo == pytest.approx(ragged.pbo, abs=0.05)

    def test_rejects_invalid_input(self):
        with pytest.raises(ValueError):
            pbo(noise_matrix(), n_blocks=7)
        with pytest.raises(ValueError):
            pbo(noise_matrix(n_strategies=1), n_blocks=10)
        with pytest.raises(ValueError):
            pbo(noise_matrix(n_periods=10), n_blocks=10)
        with pytest.raises(ValueError):
            pbo(np.zeros(100), n_blocks=10)
