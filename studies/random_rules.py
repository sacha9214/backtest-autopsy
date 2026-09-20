"""Do thought-out rules beat rules invented by a coin?

Every recipe in the grid has a story: momentum persists, prices revert to a
mean, breakouts signal conviction. This study builds a control group with no
story at all -- signals that trade at the same frequency as the real ones but
decide by coin flip -- and searches both with the same effort.

If the reasoning behind the recipes is worth anything, the real grid should
win. If the two are indistinguishable, the stories were decoration.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from autopsy.data import load_prices
from autopsy.stats import annualize, sharpe_ratio
from autopsy.strategies import DEFAULT_COST_BPS, _positions_to_returns, run_grid
from _universe import UNIVERSE

warnings.filterwarnings("ignore")

N_RANDOM_GRIDS = 20


def random_grid(prices: pd.Series, n_configs: int, hold_rate: float,
                seed: int) -> pd.DataFrame:
    """Random long/flat signals matched to the real grid's time-in-market.

    Matching the holding rate matters: a rule that is long 90% of the time
    inherits 90% of the asset's return, so comparing against rules that trade
    at a different frequency would compare exposure, not skill.
    """
    rng = np.random.default_rng(seed)
    asset_returns = prices.pct_change()
    columns = {}
    for i in range(n_configs):
        raw = rng.random(len(prices)) < hold_rate
        # Smooth so positions persist like a real signal rather than flipping daily
        positions = pd.Series(raw, index=prices.index).rolling(5).mean().round()
        columns[f"random_{i}"] = _positions_to_returns(
            positions.fillna(0.0), asset_returns, DEFAULT_COST_BPS
        )
    return pd.DataFrame(columns).dropna(how="any")


def best_sharpe(frame: pd.DataFrame) -> float:
    live = frame.loc[:, frame.std() > 0]
    if live.shape[1] == 0:
        return 0.0
    return max(sharpe_ratio(live[c].values) for c in live.columns)


def main() -> None:
    rows = []
    for ticker in UNIVERSE:
        prices = load_prices(ticker)[ticker]
        real = run_grid(prices)
        real_best = annualize(best_sharpe(real), 252)

        hold_rate = float((real != 0).mean().mean())
        randoms = [
            annualize(
                best_sharpe(random_grid(prices, real.shape[1], hold_rate, seed)), 252
            )
            for seed in range(N_RANDOM_GRIDS)
        ]
        randoms = np.array(randoms)
        rows.append({
            "ticker": ticker,
            "real_best": round(real_best, 3),
            "random_best_mean": round(randoms.mean(), 3),
            "random_best_max": round(randoms.max(), 3),
            "real_wins": real_best > randoms.mean(),
            "real_beats_all_random": real_best > randoms.max(),
        })

    table = pd.DataFrame(rows)
    n = len(table)
    print(f"Real recipes vs coin-flip rules, same search effort, {n} instruments\n")
    print(table.head(12).to_string(index=False))
    print("  ...")
    print()
    print(f"  Real grid beats the average random grid : "
          f"{int(table.real_wins.sum())}/{n} ({table.real_wins.mean():.0%})")
    print(f"  Real grid beats EVERY random grid       : "
          f"{int(table.real_beats_all_random.sum())}/{n} "
          f"({table.real_beats_all_random.mean():.0%})")
    print(f"  Median real best Sharpe                 : {table.real_best.median():.3f}")
    print(f"  Median random best Sharpe               : {table.random_best_mean.median():.3f}")
    print(f"  Median advantage of having a reason     : "
          f"{(table.real_best - table.random_best_mean).median():+.3f}")
    table.to_csv("studies/random_rules_results.csv", index=False)


if __name__ == "__main__":
    main()
