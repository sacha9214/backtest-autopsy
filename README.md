# backtest-autopsy

**Was that Sharpe ratio skill, or the best of N coin flips?**

Test fifty variants of a strategy, keep the best one, and the number you are
left with is not a measurement of performance. It is a maximum over fifty
draws — and a maximum over fifty draws is large even when every variant is
worthless. Run 1000 backtests of pure noise and the winner is *expected* to
show a Sharpe ratio of **3.26**.

This repository implements the standard corrections for that, and then uses
them on the strategies people actually repeat.

## The result

Forty-four configurations that every trading blog recommends — moving-average
crossovers, RSI oversold, time-series momentum, Bollinger reversion, Donchian
breakouts — across ten liquid US instruments, 22 to 46 years of adjusted daily
data each, with transaction costs and a one-day execution lag. Then the same
grid again in long/short form, so the results cannot simply be inheriting the
market's rise.

| | Long-only | Long/short |
|---|---|---|
| Beat buy and hold in-sample | 9 / 10 | 3 / 10 |
| Survive deflation for 44 trials | **1 / 10** | **0 / 10** |
| Positive out-of-sample | 9 / 10 | 5 / 10 |
| Beat buy and hold out-of-sample | **1 / 10** | **1 / 10** |
| Median out-of-sample Sharpe | — | **−0.01** |

The long-only column has an obvious objection: on assets that rose for thirty
years, "sometimes long" is nearly "always long". The long/short column removes
that objection and the answer gets worse, not better. Stripped of the market's
beta, these recipes produce a median out-of-sample Sharpe of **−0.01**.

The single out-of-sample winner in both studies is TLT — long-dated Treasuries,
the only instrument in the set that did not rise over the test window (its own
Sharpe was 0.06). On a flat asset, being occasionally out of the market helps.
Its DSR is 0.556, so it does not clear the bar either: it beats a weak
benchmark, not chance.

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

## Three findings about the tools themselves

All three were measured here rather than assumed, and all three cut against how
these metrics are usually presented or implemented.

**1. A single PBO value proves very little.** On pure noise PBO averages 0.50 as
advertised — with a standard deviation near **0.2**. Individual runs of
strategies with no skill whatsoever came back anywhere from 0.01 to 0.96. It is
decisive in one direction only: a genuine edge pins it at 0.000 with zero
variance. A mid-range value is not evidence of overfitting.

**2. The in-sample-to-out-of-sample slope is mechanical, and its intuitive
reading is backwards.** CSCV splits are complementary, so `mean(IS) + mean(OOS)`
is pinned to twice the full-sample mean (verified to 1e-18), forcing a negative
slope. Measured on 44 strategies over 2000 periods:

| | Slope |
|---|---|
| Pure noise, no skill | −0.46 ± 0.31 |
| One genuine edge | **−1.00 ± 0.00** |

A slope near −1 indicates a *dominant* strategy — the opposite of "degradation".
The metric is kept in the API with this warning attached and out of the verdict.

**3. `V[SR] = 1` is a trap, and this repository fell into it.** The familiar
default comes from the paper, where it suits *annualized* Sharpe ratios. Applied
to per-period ratios it is catastrophic: a real daily search here produced an
across-trial variance of **3.4e-4**, so the default is roughly 3000× too large
and rejects everything — silently, and with an air of rigour. An early version
of the retrospective below "caught" three out of three false positives that way,
which proved nothing at all. `sharpe_variance` is now a required argument with
no default, and `sharpe_variance_across_trials` estimates it from your own
trials. There is a regression test for it.

## Turning the tool on its author

Before any of this existed I spent three weeks looking for an edge on Polymarket
sports markets and produced results that looked real and were not.
`studies/polymarket_retrospective.py` feeds them back through the corrections
using only what was known at the time. Since the across-trial variance was never
recorded, it solves for the value at which each verdict would flip instead of
inventing one:

| Claim | Textbook verdict | DSR | Flips at |
|---|---|---|---|
| Price zone 0.50-0.75: +7.61%, t = 2.41 | **significant** (PSR 0.992) | 0.919 rejected | 0.57× reference — *close* |
| Pair assembly: +3.05% | not significant | 0.818 rejected | never |
| Ladderbot residual: +6.9% | not significant | 0.247 rejected | never |

The first is the interesting one, in both directions. Every standard test said
yes — t of 2.41, PSR above 0.99 — and out-of-sample it delivered +0.06% (t =
0.03) on 1404 fresh observations. What killed it was counting the eight
strategies tried before it. But the rejection is close to the flip point, so it
depends on a parameter that was never written down: had the search been less
dispersed, the correction would have let it through. Counting your trials is not
optional, and the count cannot be reconstructed after the fact.

## Install and run

```bash
pip install -e ".[data,dev]"
python studies/retail_recipes.py            # long-only deflation table
python studies/long_short.py                # same grid, beta removed
python studies/walk_forward.py              # the out-of-sample test
python studies/polymarket_retrospective.py  # the tool against its author
pytest                                      # 42 tests, no network
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
ones you abandoned — that is the whole point, and nothing in the data can check
it for you.

## Three rules enforced in code

Each is easy to get wrong and each manufactures profits out of nothing:

1. **Positions are shifted one day forward.** A signal computed from today's
   close can only be traded tomorrow. There is a test that perturbs the *last*
   price and asserts no past return moves.
2. **Turnover is charged**, 5 bps per position change by default.
3. **A configuration that never triggers earns a Sharpe of zero** rather than
   being dropped. Removing it after seeing the out-of-sample half would be
   look-ahead bias through the back door.

## Prior art

The corrections are not new, and several implementations exist —
[pypbo](https://github.com/esvhd/pypbo), vectorbt's `deflated_sharpe_ratio`,
skfolio. What this repository adds is the study: an honest trial count applied
across ten instruments and four decades, in both long-only and long/short form,
the walk-forward test that settles it, and the three measured caveats above.

## License

MIT
