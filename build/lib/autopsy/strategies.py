"""The recipes everyone repeats, plus the machinery to run a whole grid of them.

Two rules are enforced here rather than left to the caller, because both are
easy to get wrong and both manufacture profits out of nothing:

  1. Positions are shifted one day forward. A signal computed from today's
     close can only be traded from tomorrow. Skipping this is look-ahead bias
     and it is the single most common reason a backtest looks brilliant.
  2. Turnover is charged. A strategy that flips daily pays for it.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

DEFAULT_COST_BPS = 5.0


def _positions_to_returns(
    positions: pd.Series, asset_returns: pd.Series, cost_bps: float
) -> pd.Series:
    held = positions.shift(1).fillna(0.0)
    gross = held * asset_returns
    turnover = held.diff().abs().fillna(0.0)
    return gross - turnover * (cost_bps / 10_000.0)


def sma_crossover(prices: pd.Series, fast: int, slow: int) -> pd.Series:
    return (prices.rolling(fast).mean() > prices.rolling(slow).mean()).astype(float)


def rsi(prices: pd.Series, lookback: int) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / lookback, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / lookback, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def rsi_oversold(prices: pd.Series, lookback: int, threshold: float) -> pd.Series:
    return (rsi(prices, lookback) < threshold).astype(float)


def time_series_momentum(prices: pd.Series, lookback: int) -> pd.Series:
    return (prices.pct_change(lookback) > 0).astype(float)


def bollinger_reversion(prices: pd.Series, lookback: int, n_std: float) -> pd.Series:
    mean = prices.rolling(lookback).mean()
    sd = prices.rolling(lookback).std()
    return (prices < mean - n_std * sd).astype(float)


def donchian_breakout(prices: pd.Series, lookback: int) -> pd.Series:
    return (prices >= prices.rolling(lookback).max()).astype(float)


def parameter_grid() -> Iterator[tuple[str, dict]]:
    """Every configuration a retail backtester would plausibly try.

    The count matters as much as the content: this number is what gets declared
    as `n_trials` when deflating the Sharpe ratio, so it must be honest.
    """
    for fast in (5, 10, 20, 50):
        for slow in (50, 100, 150, 200):
            if fast < slow:
                yield "sma_crossover", {"fast": fast, "slow": slow}
    for lookback in (7, 14, 21):
        for threshold in (20, 30, 40):
            yield "rsi_oversold", {"lookback": lookback, "threshold": threshold}
    for lookback in (20, 60, 120, 252):
        yield "time_series_momentum", {"lookback": lookback}
    for lookback in (10, 20, 50):
        for n_std in (1.0, 1.5, 2.0, 2.5):
            yield "bollinger_reversion", {"lookback": lookback, "n_std": n_std}
    for lookback in (20, 50, 100, 200):
        yield "donchian_breakout", {"lookback": lookback}


SIGNALS = {
    "sma_crossover": sma_crossover,
    "rsi_oversold": rsi_oversold,
    "time_series_momentum": time_series_momentum,
    "bollinger_reversion": bollinger_reversion,
    "donchian_breakout": donchian_breakout,
}


def run_grid(
    prices: pd.Series, cost_bps: float = DEFAULT_COST_BPS
) -> pd.DataFrame:
    """Return a (days x configurations) frame of net strategy returns."""
    asset_returns = prices.pct_change()
    columns: dict[str, pd.Series] = {}
    for name, params in parameter_grid():
        positions = SIGNALS[name](prices, **params)
        label = name + "(" + ",".join(f"{k}={v}" for k, v in params.items()) + ")"
        columns[label] = _positions_to_returns(positions, asset_returns, cost_bps)
    frame = pd.DataFrame(columns)
    return frame.dropna(how="any")
