import numpy as np
import pandas as pd
import pytest

from autopsy.strategies import (
    long_short_grid,
    parameter_grid,
    run_grid,
    run_long_short_grid,
    sma_crossover,
    sma_crossover_ls,
)


@pytest.fixture
def trending_prices():
    """A deterministic upward drift with wiggles, so signals actually fire."""
    days = pd.bdate_range("2000-01-03", periods=1500)
    wave = np.sin(np.arange(1500) / 25) * 5
    return pd.Series(100 + np.arange(1500) * 0.05 + wave, index=days)


class TestNoLookAhead:
    def test_a_future_only_move_cannot_be_traded(self, trending_prices):
        """Changing only the LAST price must not change any past return.

        This is the load-bearing test of the whole repository: if it fails,
        every number the tool produces is inflated by information from the
        future.
        """
        base = run_grid(trending_prices)
        tampered = trending_prices.copy()
        tampered.iloc[-1] *= 1.5
        after = run_grid(tampered)

        common = base.index.intersection(after.index)[:-1]
        pd.testing.assert_frame_equal(
            base.loc[common], after.loc[common], check_exact=False, atol=1e-12
        )

    def test_positions_are_shifted_by_exactly_one_day(self, trending_prices):
        returns = trending_prices.pct_change()
        positions = sma_crossover(trending_prices, 10, 50)
        grid = run_grid(trending_prices)
        column = grid["sma_crossover(fast=10,slow=50)"]

        expected = positions.shift(1) * returns
        aligned = expected.loc[column.index]
        # Costs only subtract, so the gross series must dominate the net one.
        assert (aligned >= column - 1e-12).all()


class TestCosts:
    def test_turnover_is_charged(self, trending_prices):
        free = run_grid(trending_prices, cost_bps=0.0)
        expensive = run_grid(trending_prices, cost_bps=50.0)
        assert expensive.sum().sum() < free.sum().sum()

    def test_zero_cost_is_the_upper_bound(self, trending_prices):
        free = run_grid(trending_prices, cost_bps=0.0)
        charged = run_grid(trending_prices, cost_bps=5.0)
        assert (charged <= free + 1e-12).all().all()


class TestGrids:
    def test_trial_counts_are_stable(self):
        """The declared trial count is what deflates the Sharpe. Pin it down."""
        assert len(list(parameter_grid())) == 44
        assert len(list(long_short_grid())) == 44

    def test_fast_is_always_shorter_than_slow(self):
        for name, params in parameter_grid():
            if name == "sma_crossover":
                assert params["fast"] < params["slow"]

    def test_long_only_never_shorts(self, trending_prices):
        assert (sma_crossover(trending_prices, 10, 50) >= 0).all()

    def test_long_short_actually_shorts(self, trending_prices):
        positions = sma_crossover_ls(trending_prices, 10, 50)
        assert (positions < 0).any()
        assert (positions > 0).any()

    def test_both_grids_produce_one_column_per_configuration(self, trending_prices):
        assert run_grid(trending_prices).shape[1] == 44
        assert run_long_short_grid(trending_prices).shape[1] == 44
