"""Does the conclusion survive once the market's beta is taken out?

The long-only study is vulnerable to one objection: on assets that rose for
decades, "sometimes long" is nearly "always long", so of course buy and hold
wins. These strategies can be short, so that objection does not apply. If the
recipes still fail here, it is not because they were competing against a rising
tide.
"""

from __future__ import annotations

import warnings

import pandas as pd

from autopsy.data import load_prices
from autopsy.report import audit
from autopsy.stats import annualize, sharpe_ratio
from autopsy.strategies import run_long_short_grid

warnings.filterwarnings("ignore")

TICKERS = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "GLD", "TLT", "AAPL"]


def realised_sharpe(values) -> float:
    try:
        return sharpe_ratio(values)
    except ValueError:
        return 0.0


def main() -> None:
    rows = []
    for ticker in TICKERS:
        prices = load_prices(ticker)[ticker]
        grid = run_long_short_grid(prices)
        result = audit(
            grid,
            label=ticker,
            benchmark_returns=prices.pct_change().loc[grid.index],
        )

        split = len(grid) // 2
        in_sample = grid.iloc[:split]
        in_sample = in_sample.loc[:, in_sample.std() > 0]
        out_sample = grid.iloc[split:][in_sample.columns]
        chosen = max(in_sample.columns, key=lambda c: realised_sharpe(in_sample[c].values))
        oos_sr = annualize(realised_sharpe(out_sample[chosen].values), 252)
        bh_oos = annualize(
            realised_sharpe(prices.pct_change().loc[out_sample.index].dropna().values), 252
        )

        rows.append({
            "ticker": ticker,
            "winner": result.winner.split("(")[0],
            "sharpe": round(result.sharpe_annualized, 2),
            "bar": round(result.benchmark_annualized, 2),
            "dsr": round(result.dsr, 3),
            "survives": result.survives,
            "buy_hold": round(result.benchmark_asset_sharpe, 2),
            "sharpe_oos": round(oos_sr, 2),
            "bh_oos": round(bh_oos, 2),
            "beats_bh_oos": oos_sr > bh_oos,
            "positive_oos": oos_sr > 0,
        })

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    print()
    n = len(table)
    print(f"  Survive deflation ({table.iloc[0].name and 44 or 44} trials) : "
          f"{int(table.survives.sum())}/{n}")
    print(f"  Positive out-of-sample                 : {int(table.positive_oos.sum())}/{n}")
    print(f"  BEAT BUY AND HOLD out-of-sample        : {int(table.beats_bh_oos.sum())}/{n}")
    print(f"  Median Sharpe / bar                    : "
          f"{table.sharpe.median():.2f} / {table.bar.median():.2f}")
    print(f"  Median OOS Sharpe                      : {table.sharpe_oos.median():+.2f}")
    table.to_csv("studies/long_short_results.csv", index=False)


if __name__ == "__main__":
    main()
