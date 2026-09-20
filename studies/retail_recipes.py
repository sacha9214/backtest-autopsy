"""Do the strategies everyone repeats survive a correction for multiple testing?

Runs the same 44-configuration grid across ten liquid US instruments and
deflates each winner by the number of trials it took to find it.
"""

from __future__ import annotations

import warnings
from collections import Counter

import pandas as pd

from autopsy.data import load_prices
from autopsy.report import audit
from autopsy.strategies import run_grid

warnings.filterwarnings("ignore")

TICKERS = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "GLD", "TLT", "AAPL"]


def main() -> None:
    rows = []
    families = Counter()

    for ticker in TICKERS:
        prices = load_prices(ticker)[ticker]
        grid = run_grid(prices)
        result = audit(
            grid,
            label=ticker,
            benchmark_returns=prices.pct_change().loc[grid.index],
        )
        families[result.winner.split("(")[0]] += 1
        rows.append(
            {
                "ticker": ticker,
                "years": round(result.n_observations / 252, 1),
                "winner": result.winner.split("(")[0],
                "sharpe": round(result.sharpe_annualized, 2),
                "bar": round(result.benchmark_annualized, 2),
                "dsr": round(result.dsr, 3),
                "survives": result.survives,
                "buy_hold": round(result.benchmark_asset_sharpe, 2),
                "beats_bh": result.sharpe_annualized > result.benchmark_asset_sharpe,
                "pbo": round(result.cscv.pbo, 3),
            }
        )

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    print()
    survivors = int(table["survives"].sum())
    beats = int(table["beats_bh"].sum())
    both = int((table["survives"] & table["beats_bh"]).sum())
    print(f"  Survive deflation           : {survivors}/{len(table)}")
    print(f"  Beat buy and hold           : {beats}/{len(table)}")
    print(f"  BOTH (the only ones that count): {both}/{len(table)}")
    print(f"  Median Sharpe / bar         : {table.sharpe.median():.2f} / {table.bar.median():.2f}")
    print(f"  Winning family              : {dict(families)}")
    table.to_csv("studies/retail_recipes_results.csv", index=False)


if __name__ == "__main__":
    main()
