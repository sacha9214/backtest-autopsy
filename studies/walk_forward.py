"""The test that settles it: pick on the first half, measure on the second.

Deflation says whether a result could be luck. Walk-forward says whether the
choice you made actually paid. A strategy that passes the first and fails the
second was fitted, full stop.
"""

from __future__ import annotations

import warnings

import pandas as pd

from autopsy.data import load_prices
from autopsy.stats import annualize, sharpe_ratio
from autopsy.strategies import run_grid

warnings.filterwarnings("ignore")

from _universe import UNIVERSE

TICKERS = UNIVERSE


def realised_sharpe(returns) -> float:
    """Sharpe of a strategy as actually experienced.

    A configuration that never triggers earns exactly nothing, which is a
    Sharpe of zero -- not an error and not an exclusion. Dropping such columns
    after seeing the out-of-sample half would be look-ahead bias through the
    back door: the selection must not know what happens next.
    """
    try:
        return sharpe_ratio(returns)
    except ValueError:
        return 0.0


def main() -> None:
    rows = []
    for ticker in TICKERS:
        prices = load_prices(ticker)[ticker]
        grid = run_grid(prices)
        split = len(grid) // 2
        in_sample, out_sample = grid.iloc[:split], grid.iloc[split:]
        # Filter on the first half only -- the selection cannot see the second.
        in_sample = in_sample.loc[:, in_sample.std() > 0]
        out_sample = out_sample[in_sample.columns]

        chosen = max(
            in_sample.columns, key=lambda c: realised_sharpe(in_sample[c].values)
        )
        is_sr = annualize(realised_sharpe(in_sample[chosen].values), 252)
        oos_sr = annualize(realised_sharpe(out_sample[chosen].values), 252)

        # What the best OOS strategy would have been, for context only
        best_oos = max(
            out_sample.columns, key=lambda c: realised_sharpe(out_sample[c].values)
        )
        # Long-only strategies on rising assets inherit the market's Sharpe.
        # The honest comparison is against holding it over the SAME window.
        bh_oos = annualize(
            realised_sharpe(
                prices.pct_change().loc[out_sample.index].dropna().values
            ),
            252,
        )
        rows.append({
            "ticker": ticker,
            "chosen_on_1st_half": chosen.split("(")[0],
            "sharpe_is": round(is_sr, 2),
            "sharpe_oos": round(oos_sr, 2),
            "buy_hold_oos": round(bh_oos, 2),
            "vs_bh": round(oos_sr - bh_oos, 2),
            "decay": round(oos_sr - is_sr, 2),
            "still_picked_oos": chosen == best_oos,
        })

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    print()
    print(f"  Still positive out-of-sample   : {int((table.sharpe_oos > 0).sum())}/{len(table)}")
    print(f"  BEAT BUY AND HOLD out-of-sample: {int((table.vs_bh > 0).sum())}/{len(table)}"
          "   <- the only comparison that matters")
    print(f"  Median decay vs in-sample      : {table.decay.median():+.2f} Sharpe")
    print(f"  Median edge over buy and hold  : {table.vs_bh.median():+.2f} Sharpe")
    print(f"  Was still the best choice OOS  : {int(table.still_picked_oos.sum())}/{len(table)}")
    table.to_csv("studies/walk_forward_results.csv", index=False)


if __name__ == "__main__":
    main()
