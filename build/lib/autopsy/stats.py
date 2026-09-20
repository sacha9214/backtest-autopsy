"""Selection-bias corrections for Sharpe ratios.

Implements Bailey & Lopez de Prado (2012, 2014):
  - Probabilistic Sharpe Ratio (PSR)
  - Expected maximum Sharpe ratio under the null of no skill
  - Deflated Sharpe Ratio (DSR)
  - Minimum Track Record Length (MinTRL)

Every Sharpe ratio in this module is expressed PER PERIOD, never annualized.
Mixing the two is the most common way to get these formulas wrong: annualizing
multiplies the ratio by sqrt(periods_per_year) but leaves skewness and kurtosis
untouched, so the corrections silently stop applying.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Per-period Sharpe ratio. Uses the sample standard deviation (ddof=1)."""
    returns = np.asarray(returns, dtype=float)
    if returns.size < 2:
        raise ValueError("need at least 2 observations")
    excess = returns - risk_free
    sd = excess.std(ddof=1)
    if sd == 0:
        raise ValueError("zero variance: Sharpe ratio is undefined")
    return float(excess.mean() / sd)


def annualize(sharpe_per_period: float, periods_per_year: int) -> float:
    return sharpe_per_period * math.sqrt(periods_per_year)


def moments(returns: np.ndarray) -> tuple[float, float]:
    """Sample skewness and NON-excess kurtosis (3.0 for a normal distribution).

    The PSR formula takes non-excess kurtosis. Passing excess kurtosis here is
    the second classic mistake and it makes the denominator too small, which
    inflates confidence exactly when returns are fat-tailed.
    """
    returns = np.asarray(returns, dtype=float)
    n = returns.size
    centered = returns - returns.mean()
    m2 = (centered**2).sum() / n
    if m2 == 0:
        raise ValueError("zero variance: moments are undefined")
    m3 = (centered**3).sum() / n
    m4 = (centered**4).sum() / n
    return float(m3 / m2**1.5), float(m4 / m2**2)


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    benchmark_sharpe: float = 0.0,
) -> float:
    """P(true Sharpe > benchmark), correcting for sample size and non-normality.

    Bailey & Lopez de Prado (2012), eq. 4.
    """
    if n_observations < 2:
        raise ValueError("need at least 2 observations")
    variance = (
        1.0
        - skewness * observed_sharpe
        + (kurtosis - 1.0) / 4.0 * observed_sharpe**2
    )
    if variance <= 0:
        raise ValueError(
            f"non-positive Sharpe variance ({variance:.4f}): the skewness/kurtosis "
            "combination is inconsistent with this Sharpe ratio"
        )
    z = (observed_sharpe - benchmark_sharpe) * math.sqrt(n_observations - 1)
    return float(norm.cdf(z / math.sqrt(variance)))


def expected_max_sharpe(n_trials: int, sharpe_variance: float = 1.0) -> float:
    """Expected maximum per-period Sharpe over n_trials skill-less strategies.

    Bailey & Lopez de Prado (2014), eq. 5. This is the bar a backtest must clear
    to mean anything: with 1000 independent trials of pure noise, the winner is
    expected to show a Sharpe of about 3.26.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if sharpe_variance < 0:
        raise ValueError("sharpe_variance must be >= 0")
    if n_trials == 1:
        return 0.0
    sd = math.sqrt(sharpe_variance)
    a = norm.ppf(1.0 - 1.0 / n_trials)
    b = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(sd * ((1.0 - EULER_MASCHERONI) * a + EULER_MASCHERONI * b))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_observations: int,
    n_trials: int,
    sharpe_variance: float = 1.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> tuple[float, float]:
    """Return (DSR, benchmark) where DSR is the PSR against the expected maximum.

    DSR is a probability: below 0.95 the observed Sharpe is not distinguishable
    from the best of n_trials coin flips.
    """
    benchmark = expected_max_sharpe(n_trials, sharpe_variance)
    dsr = probabilistic_sharpe_ratio(
        observed_sharpe, n_observations, skewness, kurtosis, benchmark
    )
    return dsr, benchmark


def min_track_record_length(
    observed_sharpe: float,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    benchmark_sharpe: float = 0.0,
    confidence: float = 0.95,
) -> float:
    """Observations needed before the Sharpe clears the benchmark at `confidence`.

    Returns math.inf when the observed Sharpe is at or below the benchmark: no
    amount of extra data rescues a track record that is not ahead to begin with.
    """
    edge = observed_sharpe - benchmark_sharpe
    if edge <= 0:
        return math.inf
    variance = (
        1.0
        - skewness * observed_sharpe
        + (kurtosis - 1.0) / 4.0 * observed_sharpe**2
    )
    if variance <= 0:
        raise ValueError("non-positive Sharpe variance")
    return float(1.0 + variance * (norm.ppf(confidence) / edge) ** 2)
