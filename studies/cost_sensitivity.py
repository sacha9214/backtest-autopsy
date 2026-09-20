"""At what trading cost does the whole thing collapse?

Backtests are usually run at zero cost or at a cost the author picked. This
sweeps it instead, and reports the level at which the best strategy stops
beating buy and hold. If that level is below what anyone actually pays, the
strategy was never tradeable.
"""

from __future__ import annotations

import warnings

import pandas as pd

from autopsy.data import load_prices
from autopsy.stats import annualize, sharpe_ratio
from autopsy.strategies import run_grid
from _universe import UNIVERSE

warnings.filterwarnings("ignore")

COST_LEVELS = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]


def best(frame: pd.DataFrame) -> float:
    live = frame.loc[:, frame.std() > 0]
    return max(sharpe_ratio(live[c].values) for c in live.columns)


def main() -> None:
    rows = []
    for ticker in UNIVERSE:
        prices = load_prices(ticker)[ticker]
        reference = run_grid(prices, cost_bps=0.0)
        bh = annualize(
            sharpe_ratio(prices.pct_change().loc[reference.index].dropna().values), 252
        )
        row = {"ticker": ticker, "buy_hold": round(bh, 3)}
        for cost in COST_LEVELS:
            row[f"bps_{cost:g}"] = round(annualize(best(run_grid(prices, cost)), 252), 3)
        rows.append(row)

    table = pd.DataFrame(rows)
    print("Best-in-grid Sharpe as trading cost rises (annualized)\n")
    print(f"  {'cost (bps)':>11} {'median best':>12} {'beats B&H':>11} {'vs 0 bps':>10}")
    zero = table["bps_0"].median()
    for cost in COST_LEVELS:
        col = table[f"bps_{cost:g}"]
        beats = int((col > table.buy_hold).sum())
        print(f"  {cost:>11.0f} {col.median():>12.3f} "
              f"{beats:>7}/{len(table)} {col.median() - zero:>10.3f}")
    print()
    print("  A round trip on a liquid US ETF costs a few bps today, and cost")
    print("  far more for most of the history these backtests cover.")
    table.to_csv("studies/cost_sensitivity_results.csv", index=False)


if __name__ == "__main__":
    main()
