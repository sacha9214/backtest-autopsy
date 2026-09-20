"""Did these recipes ever work, and did publishing them kill them?

Anomalies tend to shrink after they are written up -- crowding, or the fact
that they were never there. This splits the longest histories into eras and
deflates each one separately, so a recipe that genuinely worked in 1970 and
stopped in 2000 is visible as such rather than averaged into nothing.

Each era is deflated on its own trial count, so a short era is held to the same
standard as a long one and simply needs a bigger edge to clear it.
"""

from __future__ import annotations

import warnings

import pandas as pd

from autopsy.data import load_prices
from autopsy.report import audit
from autopsy.strategies import run_grid
from _universe import UNIVERSE

warnings.filterwarnings("ignore")

ERAS = [
    ("1962-1979", "1962-01-01", "1979-12-31"),
    ("1980-1994", "1980-01-01", "1994-12-31"),
    ("1995-2009", "1995-01-01", "2009-12-31"),
    ("2010-2026", "2010-01-01", "2026-12-31"),
]
MIN_DAYS = 756  # three years: below this an era cannot say anything


def main() -> None:
    rows = []
    for ticker in UNIVERSE:
        prices = load_prices(ticker)[ticker]
        for name, start, end in ERAS:
            window = prices.loc[start:end]
            if len(window) < MIN_DAYS:
                continue
            grid = run_grid(window)
            if grid.empty or (grid.std() > 0).sum() < 2:
                continue
            result = audit(
                grid, label=f"{ticker} {name}",
                benchmark_returns=window.pct_change().loc[grid.index],
            )
            rows.append({
                "era": name,
                "ticker": ticker,
                "sharpe": round(result.sharpe_annualized, 3),
                "bar": round(result.benchmark_annualized, 3),
                "dsr": round(result.dsr, 3),
                "survives": result.survives,
                "buy_hold": round(result.benchmark_asset_sharpe, 3),
                "beats_bh": result.sharpe_annualized > result.benchmark_asset_sharpe,
            })

    table = pd.DataFrame(rows)
    print("Deflated results by era, each era judged on its own\n")
    print(f"  {'era':<12} {'n':>3} {'survive':>8} {'beat B&H':>9} {'both':>6} "
          f"{'med Sharpe':>11} {'med B&H':>9}")
    for era, _, _ in ERAS:
        part = table[table.era == era]
        if part.empty:
            continue
        print(f"  {era:<12} {len(part):>3} "
              f"{int(part.survives.sum()):>8} {int(part.beats_bh.sum()):>9} "
              f"{int((part.survives & part.beats_bh).sum()):>6} "
              f"{part.sharpe.median():>11.2f} {part.buy_hold.median():>9.2f}")
    print()
    print(f"  Total instrument-eras tested : {len(table)}")
    print(f"  Survived deflation           : {int(table.survives.sum())} "
          f"({table.survives.mean():.0%})")
    print(f"  Expected by chance at 95%    : {0.05 * len(table):.1f}")
    table.to_csv("studies/by_era_results.csv", index=False)


if __name__ == "__main__":
    main()
