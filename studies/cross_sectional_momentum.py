"""The one anomaly that has survived thirty years of hostile examination.

Everything else here is timing: when to be in one instrument. Cross-sectional
momentum is a different question -- which instruments to hold, ranked against
each other -- and it is the anomaly with the strongest published record
(Jegadeesh & Titman 1993, and a long line of attempts to kill it since).

Held to exactly the same standard as the blog recipes: an honest trial count,
deflation, and a walk-forward test. If the corrections reject everything
indiscriminately, they should reject this too.

IMPORTANT LIMIT. This does NOT test the published anomaly and cannot refute it.
Jegadeesh & Titman rank hundreds of individual stocks; this ranks 49 ETFs and
large caps, which is a different and much smaller cross-section -- fewer names
to spread across, and most of them already diversified baskets that dilute the
very dispersion momentum feeds on. A negative result here means the effect does
not survive in THIS universe at THIS trial count. Nothing more.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator

import numpy as np
import pandas as pd

from autopsy.data import load_prices
from autopsy.report import audit
from autopsy.stats import annualize, sharpe_ratio
from _universe import UNIVERSE

warnings.filterwarnings("ignore")

COST_BPS = 5.0
REBALANCE = 21  # trading days, roughly monthly


def momentum_portfolio(
    prices: pd.DataFrame, lookback: int, skip: int, n_hold: int, short_side: bool
) -> pd.Series:
    """Rank on past return, hold the winners, optionally short the losers."""
    signal = prices.shift(skip).pct_change(lookback)
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for i in range(0, len(prices), REBALANCE):
        date = prices.index[i]
        scores = signal.loc[date].dropna()
        if len(scores) < 2 * n_hold:
            continue
        ranked = scores.sort_values(ascending=False)
        winners = ranked.index[:n_hold]
        weights.loc[date:, :] = 0.0
        weights.loc[date:, winners] = 1.0 / n_hold
        if short_side:
            weights.loc[date:, ranked.index[-n_hold:]] = -1.0 / n_hold

    asset_returns = prices.pct_change()
    held = weights.shift(1).fillna(0.0)
    gross = (held * asset_returns).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    return (gross - turnover * COST_BPS / 10_000.0).dropna()


def grid() -> Iterator[tuple[str, dict]]:
    for lookback in (63, 126, 189, 252):
        for skip in (0, 21):
            for n_hold in (3, 5, 10):
                for short_side in (False, True):
                    yield (
                        f"mom(lb={lookback},skip={skip},n={n_hold},"
                        f"{'ls' if short_side else 'lo'})",
                        {"lookback": lookback, "skip": skip,
                         "n_hold": n_hold, "short_side": short_side},
                    )


def main() -> None:
    prices = pd.DataFrame({t: load_prices(t)[t] for t in UNIVERSE}).sort_index()
    # Start once enough instruments exist to rank meaningfully.
    prices = prices.loc[prices.notna().sum(axis=1) >= 20]
    print(f"Universe: {prices.shape[1]} instruments, "
          f"{prices.index[0].date()} -> {prices.index[-1].date()} "
          f"({len(prices)} days)\n")

    configs = list(grid())
    columns = {
        name: momentum_portfolio(prices, **params) for name, params in configs
    }
    frame = pd.DataFrame(columns).dropna(how="any")

    equal_weight = prices.pct_change().mean(axis=1).loc[frame.index]
    result = audit(frame, label="Cross-sectional momentum",
                   benchmark_returns=equal_weight)
    print(result.render())

    split = len(frame) // 2
    in_sample = frame.iloc[:split]
    in_sample = in_sample.loc[:, in_sample.std() > 0]
    out_sample = frame.iloc[split:][in_sample.columns]
    chosen = max(in_sample.columns, key=lambda c: sharpe_ratio(in_sample[c].values))
    oos = annualize(sharpe_ratio(out_sample[chosen].values), 252)
    bh_oos = annualize(sharpe_ratio(equal_weight.loc[out_sample.index].values), 252)

    print()
    print(f"  Walk-forward: picked {chosen} on the first half")
    print(f"    out-of-sample Sharpe   {oos:>7.2f}")
    print(f"    equal-weight benchmark {bh_oos:>7.2f}")
    print(f"    {'BEATS' if oos > bh_oos else 'LOSES TO'} the benchmark out-of-sample")
    print()
    print("  Caveat: 49 ETFs and large caps is not the cross-section the published")
    print("  anomaly was measured on. This says the effect does not survive here,")
    print("  not that the anomaly is false.")

    pd.DataFrame([{
        "trials": len(configs), "winner": result.winner,
        "sharpe": round(result.sharpe_annualized, 3),
        "bar": round(result.benchmark_annualized, 3),
        "dsr": round(result.dsr, 3), "survives": result.survives,
        "benchmark": round(result.benchmark_asset_sharpe, 3),
        "sharpe_oos": round(oos, 3), "benchmark_oos": round(bh_oos, 3),
    }]).to_csv("studies/cross_sectional_momentum_results.csv", index=False)


if __name__ == "__main__":
    main()
