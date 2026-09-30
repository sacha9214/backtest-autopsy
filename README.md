# backtest-autopsy

[![tests](https://github.com/sacha9214/backtest-autopsy/actions/workflows/tests.yml/badge.svg)](https://github.com/sacha9214/backtest-autopsy/actions/workflows/tests.yml)

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
breakouts — across **49 instruments** spanning US broad market, sectors,
international, fixed income, commodities, factor ETFs and large caps, with up to
64 years of adjusted daily data each. Transaction costs charged, positions
lagged one day.

| | Long-only | Long/short |
|---|---|---|
| Survive deflation for 44 trials | **2 / 49** | **0 / 49** |
| Beat buy and hold in-sample | 33 / 49 | 16 / 49 |
| Beat buy and hold *out-of-sample* | **6 / 49** | **4 / 49** |
| Still the best choice out-of-sample | 2 / 49 | — |
| Median out-of-sample edge vs buy and hold | **−0.19** | **−0.54** |

Two survivors out of 49 is what a 95% threshold produces by chance: 49 × 0.05 =
2.45 expected false positives. The observed count is indistinguishable from
there being nothing at all. Splitting history into four eras and deflating each
separately gives the same answer from the other direction — **2 survivors out of
108 instrument-eras, against 5.4 expected by chance.** Fewer discoveries than
noise alone would generate. (These tests are not independent, since the
instruments are correlated, so treat both expectations as approximate.)

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

### But the recipes are not pure noise

Against a control group of coin-flip rules trading at the same frequency, given
the same search effort, the real grid wins on **45 of 49 instruments**, by a
median of **+0.18 Sharpe**. Momentum, reversion and breakout logic do capture
something real.

That something is simply too small to survive the cost of having searched for
it, and too small to beat holding the asset. Both statements are true at once,
and the second is the one that matters for anyone deciding what to do with
money. *(Caveat: trend-following rules are long when markets rise, so part of
that +0.18 is timing the market's own drift rather than skill.)*

### What searching is worth, measured

Scrambling SPY's daily returns destroys every pattern, so nothing is findable by
construction. Searching 43 configurations on that scrambled series still
produces a Sharpe of **0.64**. On the real series the best is 0.80.

The gap — **0.16 Sharpe** — is everything the signal was actually worth. The
rest was the price of looking. The measured gain tracks the theoretical bar to
within 0.10 Sharpe, with the formula running slightly high: grid configurations
are correlated (mean pairwise 0.35), so 44 nominal trials are fewer than 44
independent ones, and declaring the raw count is conservative.

### Costs are not the culprit

Sweeping transaction costs from 0 to 50 bps degrades the median best Sharpe from
0.63 to 0.49, and 19 of 49 instruments still beat buy and hold at 50 bps. The
decay is gradual. Whatever kills these strategies, it is not fees — it is
selection.

### The published anomaly fares no better here

Cross-sectional momentum — the anomaly with the strongest survival record in the
literature — ranked across the same 49 instruments, 48 configurations, judged by
the same standard: Sharpe 0.76 against a bar of 0.69, **DSR 0.630, rejected**,
and out-of-sample it returns 0.83 against 0.86 for an equal-weight basket.
MinTRL: 683 years.

This does **not** refute the published anomaly. Jegadeesh & Titman rank hundreds
of individual stocks; 49 mostly-diversified ETFs are a different and much
smaller cross-section. It says the effect does not survive *in this universe at
this trial count*, and nothing more.

## The studies

| Script | Question |
|---|---|
| `retail_recipes.py` | Do the recipes survive deflation? (49 instruments, by asset class) |
| `long_short.py` | Same, with the market's beta removed |
| `walk_forward.py` | Does the choice you made actually pay, on data it never saw? |
| `price_of_searching.py` | What does searching buy on data with nothing to find? |
| `random_rules.py` | Do reasoned rules beat coin-flip rules at equal search effort? |
| `cost_sensitivity.py` | At what trading cost does it collapse? |
| `by_era.py` | Did they ever work, in any era? |
| `cross_sectional_momentum.py` | Does the strongest published anomaly clear the same bar? |

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
to per-period ratios it is catastrophic: the SPY grid in this repository has an
across-trial variance of **3.4e-4**, so the default is roughly 3000× too large.
It rejects everything — silently, and with an air of rigour, which is the worst
way for a tool like this to be wrong. `sharpe_variance` is now a required
argument with no default, and `sharpe_variance_across_trials` estimates it from
your own trials. There is a regression test pinning the behaviour.

## Install and run

```bash
pip install -e ".[data,dev]"
python studies/retail_recipes.py             # start here: the deflation table
python studies/price_of_searching.py         # what searching is worth
python studies/random_rules.py               # vs coin-flip rules
pytest                                       # 42 tests, no network
```

The first run downloads 49 instruments, which takes a few minutes. Every run
afterwards reads the cache.

The first study you run downloads its prices from Yahoo Finance and caches them
as CSV under `data/cache/`; every run after that is offline. The cache is not
committed — Yahoo's terms do not allow redistributing their data.

The test suite never touches the network and never reads the cache: it runs on
synthetic data with known properties, so `pytest` works on a fresh clone with no
downloads at all.

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
