import math

import numpy as np
import pytest

from autopsy.stats import (
    deflated_sharpe_ratio,
    sharpe_variance_across_trials,
    expected_max_sharpe,
    min_track_record_length,
    moments,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)


class TestExpectedMaxSharpe:
    def test_matches_published_value(self):
        """Bailey & Lopez de Prado (2014) report 3.26 for 1000 trials, V[SR]=1."""
        assert expected_max_sharpe(1000, 1.0) == pytest.approx(3.26, abs=0.01)

    def test_single_trial_needs_no_correction(self):
        assert expected_max_sharpe(1, 1.0) == 0.0

    def test_grows_with_trial_count(self):
        bars = [expected_max_sharpe(n, 1.0) for n in (2, 10, 100, 1000, 10000)]
        assert bars == sorted(bars)

    def test_scales_with_sharpe_dispersion(self):
        assert expected_max_sharpe(100, 4.0) == pytest.approx(
            2 * expected_max_sharpe(100, 1.0)
        )

    def test_zero_dispersion_means_no_bar(self):
        assert expected_max_sharpe(1000, 0.0) == 0.0

    def test_rejects_invalid_input(self):
        with pytest.raises(ValueError):
            expected_max_sharpe(0, 1.0)
        with pytest.raises(ValueError):
            expected_max_sharpe(10, -1.0)

    def test_variance_must_be_supplied(self):
        """No default: the textbook 1.0 is wrong for per-period Sharpe ratios."""
        with pytest.raises(TypeError):
            expected_max_sharpe(100)


class TestProbabilisticSharpe:
    def test_even_odds_at_the_benchmark(self):
        assert probabilistic_sharpe_ratio(0.1, 100, benchmark_sharpe=0.1) == pytest.approx(0.5)

    def test_confidence_grows_with_sample_size(self):
        short = probabilistic_sharpe_ratio(0.1, 50)
        long = probabilistic_sharpe_ratio(0.1, 5000)
        assert long > short

    def test_negative_skew_lowers_confidence(self):
        symmetric = probabilistic_sharpe_ratio(0.1, 500, skewness=0.0)
        left_tailed = probabilistic_sharpe_ratio(0.1, 500, skewness=-1.5)
        assert left_tailed < symmetric

    def test_fat_tails_lower_confidence(self):
        normal = probabilistic_sharpe_ratio(0.1, 500, kurtosis=3.0)
        fat = probabilistic_sharpe_ratio(0.1, 500, kurtosis=12.0)
        assert fat < normal

    def test_is_a_probability(self):
        for sr in (-0.5, 0.0, 0.05, 0.5):
            p = probabilistic_sharpe_ratio(sr, 200)
            assert 0.0 <= p <= 1.0


class TestDeflatedSharpe:
    def test_never_more_confident_than_psr_against_zero(self):
        """Deflating can only raise the bar, so it can only lower confidence."""
        psr = probabilistic_sharpe_ratio(0.15, 1000)
        dsr, _ = deflated_sharpe_ratio(0.15, 1000, n_trials=50, sharpe_variance=1.0)
        assert dsr <= psr

    def test_more_trials_kill_the_same_sharpe(self):
        honest, _ = deflated_sharpe_ratio(0.12, 1000, n_trials=2, sharpe_variance=1.0)
        fished, _ = deflated_sharpe_ratio(0.12, 1000, n_trials=5000, sharpe_variance=1.0)
        assert fished < honest

    def test_returns_the_bar_it_used(self):
        dsr, bar = deflated_sharpe_ratio(0.12, 1000, n_trials=100, sharpe_variance=1.0)
        assert bar == pytest.approx(expected_max_sharpe(100, 1.0))


class TestMinTrackRecordLength:
    def test_infinite_when_not_ahead_of_benchmark(self):
        assert min_track_record_length(0.05, benchmark_sharpe=0.05) == math.inf
        assert min_track_record_length(0.01, benchmark_sharpe=0.20) == math.inf

    def test_bigger_edge_needs_less_history(self):
        assert min_track_record_length(0.30) < min_track_record_length(0.05)

    def test_fat_tails_demand_more_history(self):
        assert min_track_record_length(0.1, kurtosis=12.0) > min_track_record_length(0.1)


class TestSampleStatistics:
    def test_sharpe_of_known_series(self):
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.00])
        expected = returns.mean() / returns.std(ddof=1)
        assert sharpe_ratio(returns) == pytest.approx(expected)

    def test_normal_sample_has_kurtosis_near_three(self):
        rng = np.random.default_rng(0)
        skew, kurt = moments(rng.normal(size=200_000))
        assert skew == pytest.approx(0.0, abs=0.05)
        assert kurt == pytest.approx(3.0, abs=0.05)

    def test_rejects_degenerate_series(self):
        with pytest.raises(ValueError):
            sharpe_ratio(np.ones(50))
        with pytest.raises(ValueError):
            sharpe_ratio(np.array([0.01]))


class TestSharpeVarianceInput:
    def test_estimated_from_trials_not_assumed(self):
        trials = [0.01, 0.02, 0.03, 0.04, 0.05]
        assert sharpe_variance_across_trials(trials) == pytest.approx(
            np.var(trials, ddof=1)
        )

    def test_needs_at_least_two_trials(self):
        with pytest.raises(ValueError):
            sharpe_variance_across_trials([0.02])

    def test_the_textbook_default_would_reject_everything(self):
        """Regression test for a bug this repository shipped and then caught.

        Per-period Sharpe ratios on daily data cluster around 0.03 with an
        across-trial variance near 3e-4. Using the familiar V[SR] = 1 raises the
        bar by a factor of thousands, so a genuinely strong result is rejected.
        """
        realistic = 3.4e-4
        strong_sharpe = 0.12

        honest, bar_honest = deflated_sharpe_ratio(
            strong_sharpe, 8000, n_trials=44, sharpe_variance=realistic
        )
        wrong, bar_wrong = deflated_sharpe_ratio(
            strong_sharpe, 8000, n_trials=44, sharpe_variance=1.0
        )
        assert bar_wrong > 20 * bar_honest
        assert honest > 0.95
        assert wrong < 0.01
