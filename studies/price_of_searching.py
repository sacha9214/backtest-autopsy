"""What does searching buy you, for free, on real data?

The corrections rest on a formula for the Sharpe ratio you should expect from
the best of N worthless trials. That formula is usually taken on faith. This
study measures the thing directly: draw k configurations at random from the
grid, keep the best, repeat, and watch the curve rise with k.

Two versions are run. On real prices the winner's Sharpe includes whatever
genuine effect exists. On phase-randomised prices -- same distribution, same
autocorrelation destroyed -- there is nothing to find, so the whole curve is
the price of searching and nothing else.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from autopsy.data import load_prices
from autopsy.stats import annualize, expected_max_sharpe, sharpe_ratio
from autopsy.strategies import run_grid

warnings.filterwarnings("ignore")

DRAWS = 400
TICKER = "SPY"


def scramble(prices: pd.Series, seed: int) -> pd.Series:
    """Shuffle daily returns: same distribution, every pattern destroyed.

    Any strategy edge that survives this is impossible by construction, so
    whatever a search still finds here is pure selection.
    """
    rng = np.random.default_rng(seed)
    returns = prices.pct_change().dropna().values
    shuffled = rng.permutation(returns)
    return pd.Series(
        prices.iloc[0] * np.cumprod(1 + shuffled),
        index=prices.index[1:],
        name=prices.name,
    )


def best_of_k(sharpes: np.ndarray, k: int, rng: np.random.Generator) -> float:
    picks = rng.choice(sharpes.size, size=k, replace=False)
    return float(sharpes[picks].max())


def curve(grid: pd.DataFrame, label: str) -> pd.DataFrame:
    live = grid.loc[:, grid.std() > 0]
    sharpes = np.array([sharpe_ratio(live[c].values) for c in live.columns])
    variance = float(sharpes.var(ddof=1))
    rng = np.random.default_rng(0)

    rows = []
    for k in (1, 2, 3, 5, 8, 12, 20, 30, min(43, sharpes.size)):
        if k > sharpes.size:
            continue
        observed = np.array([best_of_k(sharpes, k, rng) for _ in range(DRAWS)])
        rows.append({
            "source": label,
            "trials": k,
            "best_sharpe": round(annualize(observed.mean(), 252), 3),
            "predicted_gain": round(annualize(expected_max_sharpe(k, variance), 252), 3),
            "actual_gain": round(annualize(observed.mean() - sharpes.mean(), 252), 3),
        })
    return pd.DataFrame(rows)


def main() -> None:
    prices = load_prices(TICKER)[TICKER]
    real = curve(run_grid(prices), "real prices")
    fake = curve(run_grid(scramble(prices, seed=7)), "scrambled")

    table = pd.concat([real, fake], ignore_index=True)
    print(f"Best-of-k Sharpe on {TICKER}, {DRAWS} random draws per k\n")
    for source in ("real prices", "scrambled"):
        part = table[table.source == source]
        print(f"  {source}")
        print(f"  {'trials':>7} {'best Sharpe':>12} {'gain vs avg':>12} {'formula says':>13}")
        for _, r in part.iterrows():
            print(f"  {r.trials:>7} {r.best_sharpe:>12.3f} "
                  f"{r.actual_gain:>12.3f} {r.predicted_gain:>13.3f}")
        print()

    scrambled = table[table.source == "scrambled"]
    err = (scrambled.actual_gain - scrambled.predicted_gain).abs().mean()
    print(f"  On data with nothing to find, searching 43 configurations still")
    print(f"  produces a Sharpe of "
          f"{scrambled[scrambled.trials == scrambled.trials.max()].best_sharpe.iloc[0]:.2f}.")
    print(f"  Mean gap between measured gain and the formula: {err:.3f} Sharpe.")
    table.to_csv("studies/price_of_searching_results.csv", index=False)


if __name__ == "__main__":
    main()
