"""Probability of Backtest Overfitting via Combinatorially Symmetric Cross-Validation.

Bailey, Borwein, Lopez de Prado & Zhu (2015).

Where the Deflated Sharpe Ratio judges one strategy, CSCV judges the *selection
procedure*. It asks: across every way of splitting history in half, how often
does the configuration that won in-sample land in the bottom half out-of-sample?

A PBO above 0.5 means picking the in-sample winner is worse than picking at
random, which is the signature of a search that fit noise.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .stats import sharpe_ratio


@dataclass(frozen=True)
class CSCVResult:
    pbo: float
    """Fraction of splits where the in-sample winner ranked below the OOS median."""

    logits: np.ndarray
    """One logit per split. Negative means the winner underperformed."""

    is_performance: np.ndarray
    """In-sample performance of the selected strategy, per split."""

    oos_performance: np.ndarray
    """Out-of-sample performance of that same strategy, per split."""

    n_splits: int
    n_strategies: int

    @property
    def performance_degradation(self) -> tuple[float, float]:
        """(slope, intercept) of OOS performance regressed on IS performance.

        WARNING -- do not read this as a quality signal. Because CSCV splits are
        complementary, mean(IS) + mean(OOS) is pinned to twice the full-sample
        mean, which induces a mechanical negative slope. Measured on synthetic
        data with 44 strategies over 2000 periods:

            pure noise, no skill : -0.46 +/- 0.31
            one genuine edge     : -1.00 +/- 0.00

        A slope near -1 therefore indicates a *dominant* strategy, the opposite
        of what "degradation" suggests. It is exposed for completeness and
        deliberately kept out of the rendered verdict.
        """
        slope, intercept = np.polyfit(self.is_performance, self.oos_performance, 1)
        return float(slope), float(intercept)

    @property
    def probability_of_loss(self) -> float:
        """Fraction of splits where the selected strategy lost money out-of-sample."""
        return float((self.oos_performance <= 0).mean())


def _default_metric(column: np.ndarray) -> float:
    try:
        return sharpe_ratio(column)
    except ValueError:
        return 0.0


def pbo(
    returns: np.ndarray,
    n_blocks: int = 16,
    metric: Callable[[np.ndarray], float] | None = None,
) -> CSCVResult:
    """Run CSCV on a (periods x strategies) matrix of returns.

    `n_blocks` must be even. The number of splits is C(n_blocks, n_blocks/2),
    so 16 blocks means 12870 evaluations -- high enough to be stable, low
    enough to stay fast. Rows beyond the last whole block are dropped, since
    unequal blocks would weight some periods more than others.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 2:
        raise ValueError("returns must be a 2-D (periods x strategies) array")
    if n_blocks % 2 != 0:
        raise ValueError("n_blocks must be even")

    n_periods, n_strategies = returns.shape
    if n_strategies < 2:
        raise ValueError("need at least 2 strategies to rank one against another")
    if n_periods < n_blocks * 2:
        raise ValueError(
            f"need at least {n_blocks * 2} periods for {n_blocks} blocks, got {n_periods}"
        )

    metric = metric or _default_metric
    usable = n_periods - (n_periods % n_blocks)
    blocks = np.split(returns[:usable], n_blocks, axis=0)

    all_indices = set(range(n_blocks))
    logits: list[float] = []
    is_perf: list[float] = []
    oos_perf: list[float] = []

    for chosen in combinations(range(n_blocks), n_blocks // 2):
        rest = sorted(all_indices - set(chosen))
        in_sample = np.concatenate([blocks[i] for i in chosen])
        out_sample = np.concatenate([blocks[i] for i in rest])

        is_scores = np.array([metric(in_sample[:, j]) for j in range(n_strategies)])
        oos_scores = np.array([metric(out_sample[:, j]) for j in range(n_strategies)])

        winner = int(np.argmax(is_scores))
        # Rank of the winner among OOS scores: 1 = worst, n_strategies = best.
        rank = float((oos_scores <= oos_scores[winner]).sum())
        omega = rank / (n_strategies + 1)
        omega = min(max(omega, 1e-12), 1 - 1e-12)

        logits.append(math.log(omega / (1 - omega)))
        is_perf.append(float(is_scores[winner]))
        oos_perf.append(float(oos_scores[winner]))

    logits_arr = np.array(logits)
    return CSCVResult(
        pbo=float((logits_arr <= 0).mean()),
        logits=logits_arr,
        is_performance=np.array(is_perf),
        oos_performance=np.array(oos_perf),
        n_splits=logits_arr.size,
        n_strategies=n_strategies,
    )
