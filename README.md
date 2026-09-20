# backtest-autopsy

**Was that Sharpe ratio skill, or the best of N coin flips?**

Test fifty variants of a strategy, keep the best one, and the number you are
left with is not a measurement of performance. It is a maximum over fifty
draws — and a maximum over fifty draws is large even when every variant is
worthless. Run 1000 backtests of pure noise and the winner is *expected* to
show a Sharpe ratio of **3.26**.

This repository implements the standard corrections for that problem, and then
uses them on the strategies people actually repeat.

## The result

Forty-four configurations that every trading blog recommends — moving-average
crossovers, RSI oversold, time-series momentum, Bollinger reversion, Donchian
breakouts — run across ten liquid US instruments, 22 to 46 years of adjusted
daily data each, with transaction costs and a one-day execution lag.

| Question | Answer |
|---|---|
| Beat buy and hold in-sample? | **9 / 10** |
| Survive deflation for 44 trials? | **1 / 10** |
| Beat buy and hold *out-of-sample*? | **1 / 10** |
| Still the best choice out-of-sample? | **1 / 10** |

The single out-of-sample winner is TLT — long-dated Treasuries, the only
instrument in the set that did not rise over the test window (its own Sharpe
was 0.06). On a flat asset, being occasionally out of the market helps. That is
not a strategy, it is a description of the period.

XLK survived deflation (DSR 0.964) and still lost to simply holding it: 0.89
against 0.99.

```
  SPY 1993-2026, 44 retail recipes
  Winner            sma_crossover(fast=10,slow=200)
  Sharpe (ann.)           0.80
  Trials declared           44
  Bar for  44 trials      0.65   <- Sharpe expected from luck alone
  DEFLATED SHARPE       0.8070   FAIL
  MinTRL                 30471   (121 years, +22005 vs what you have)
  Buy and hold            0.65   PASS

  VERDICT: not distinguishable from the best of 44 coin flips.
```

Thirty-three years of data is not enough to tell this strategy apart from luck.
It would take 121.

## What it computes

| Measure | Question it answers |
|---|---|
| **PSR** | Given this much history and these fat tails, is the true Sharpe above zero? |
| **Expected max Sharpe** | What would the best of N worthless strategies have scored? |
| **DSR** | Is this Sharpe above *that* bar, rather than above zero? |
| **MinTRL** | How much history would it take before this number means something? |
| **PBO (CSCV)** | Across every way of splitting history, how often does the in-sample winner land in the bottom half afterwards? |

Sources: Bailey & López de Prado (2012, 2014); Bailey, Borwein, López de Prado
& Zhu (2015).

## Two findings about the tools themselves

Both were measured here, not assumed, and both cut against how these metrics
are usually presented.

**1. A single PBO value proves very little.** On pure noise, PBO averages 0.50
as advertised — but with a standard deviation near **0.2**. Individual runs of
strategies with no skill whatsoever came back anywhere from 0.01 to 0.96. PBO
is decisive in one direction only: a genuine edge pins it at 0.000 with zero
variance. A mid-range value is not evidence of overfitting.

**2. The in-sample-to-out-of-sample slope is mechanical, and its intuitive
reading is backwards.** CSCV splits are complementary, so `mean(IS) + mean(OOS)`
is pinned to twice the full-sample mean, which forces a negative slope. Measured
on 44 strategies over 2000 periods:

| | Slope |
|---|---|
| Pure noise, no skill | −0.46 ± 0.31 |
| One genuine edge | **−1.00 ± 0.00** |

A slope near −1 indicates a *dominant* strategy — the opposite of "degradation".
The metric is exposed in the API with this warning attached and deliberately
kept out of the rendered verdict.

## Install and run

```bash
pip install -e ".[data,dev]"
python studies/retail_recipes.py     # the deflation table
python studies/walk_forward.py       # the out-of-sample test
pytest                               # 29 tests, no network
```

Prices are cached as CSV under `data/cache/` on first download, so every study
re-runs offline. The test suite never touches the network and never reads the
cache: it runs on synthetic data with known properties.

## Use it on your own backtest

```python
from autopsy.report import audit

result = audit(returns_frame, n_trials=42)   # periods x configurations
print(result.render())
```

Declare `n_trials` honestly. It is every configuration you tried, including the
ones you abandoned — that is the whole point, and nothing in the data can
check it for you.

## Two rules enforced in code

Both are easy to get wrong and both manufacture profits out of nothing:

1. **Positions are shifted one day forward.** A signal computed from today's
   close can only be traded tomorrow.
2. **Turnover is charged**, 5 bps per position change by default.

A configuration that never triggers earns a Sharpe of zero rather than being
dropped — silently removing it after seeing the out-of-sample half would be
look-ahead bias through the back door.

## Prior art

The corrections themselves are not new, and several implementations exist —
[pypbo](https://github.com/esvhd/pypbo), vectorbt's `deflated_sharpe_ratio`,
skfolio. What this repository adds is the study: the same honest trial count
applied across ten instruments and four decades, the walk-forward test that
settles it, and the two measured caveats about the metrics above.

## License

MIT
