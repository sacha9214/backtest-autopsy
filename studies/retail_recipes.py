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
from _universe import UNIVERSE, group_of

warnings.filterwarnings("ignore")

TICKERS = UNIVERSE


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
                "group": group_of(ticker),
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
    n = len(table)

    print("By asset class:")
    print(f"  {'group':<15} {'n':>3} {'survive':>8} {'beat B&H':>9} {'both':>6} "
          f"{'med Sharpe':>11} {'med bar':>8}")
    for group, part in table.groupby("group", sort=False):
        print(f"  {group:<15} {len(part):>3} "
              f"{int(part.survives.sum()):>8} {int(part.beats_bh.sum()):>9} "
              f"{int((part.survives & part.beats_bh).sum()):>6} "
              f"{part.sharpe.median():>11.2f} {part.bar.median():>8.2f}")

    survivors = int(table["survives"].sum())
    beats = int(table["beats_bh"].sum())
    both = int((table["survives"] & table["beats_bh"]).sum())
    print()
    print(f"  Instruments tested             : {n}")
    print(f"  Survive deflation              : {survivors}/{n} ({survivors/n:.0%})")
    print(f"  Beat buy and hold              : {beats}/{n} ({beats/n:.0%})")
    print(f"  BOTH (the only ones that count): {both}/{n} ({both/n:.0%})")
    print(f"  Median Sharpe / bar            : {table.sharpe.median():.2f} / {table.bar.median():.2f}")
    print(f"  Winning family                 : {dict(families)}")
    if both:
        print()
        print("  Survivors that also beat buy and hold:")
        for _, r in table[table.survives & table.beats_bh].iterrows():
            print(f"    {r.ticker:6} {r.winner:<22} Sharpe {r.sharpe:.2f} "
                  f"vs B&H {r.buy_hold:.2f}, DSR {r.dsr:.3f}")
    table.to_csv("studies/retail_recipes_results.csv", index=False)


if __name__ == "__main__":
    main()
