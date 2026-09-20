"""Turn a grid of backtested strategies into a verdict.

The point of this module is that a number without a bar next to it means
nothing. Every figure it prints is paired with what it has to beat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .cscv import CSCVResult, pbo
from .stats import (
    annualize,
    deflated_sharpe_ratio,
    min_track_record_length,
    moments,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)

DSR_THRESHOLD = 0.95


@dataclass
class Autopsy:
    label: str
    winner: str
    sharpe: float
    sharpe_annualized: float
    skewness: float
    kurtosis: float
    n_observations: int
    n_trials: int
    n_degenerate: int
    sharpe_variance: float
    psr: float
    benchmark: float
    benchmark_annualized: float
    dsr: float
    min_track_record: float
    periods_per_year: int
    cscv: CSCVResult | None = None
    benchmark_asset_sharpe: float | None = None
    ranking: list[tuple[str, float]] = field(default_factory=list)

    @property
    def survives(self) -> bool:
        return self.dsr > DSR_THRESHOLD

    def render(self) -> str:
        mark = lambda ok: "PASS" if ok else "FAIL"
        years = self.n_observations / self.periods_per_year
        lines = [
            f"  {self.label}",
            f"  {'-' * len(self.label)}",
            f"  Winner            {self.winner}",
            f"  Sharpe (ann.)     {self.sharpe_annualized:>10.2f}",
            f"  Skew / Kurtosis   {self.skewness:>10.2f} / {self.kurtosis:.1f}",
            f"  Observations      {self.n_observations:>10}  ({years:.1f} years)",
            f"  Trials declared   {self.n_trials:>10}"
            + (f"  ({self.n_degenerate} never traded)" if self.n_degenerate else ""),
            "",
            f"  PSR vs zero       {self.psr:>10.4f}",
            f"  Bar for {self.n_trials:>3} trials  {self.benchmark_annualized:>10.2f}"
            "   <- Sharpe expected from luck alone",
            f"  DEFLATED SHARPE   {self.dsr:>10.4f}   {mark(self.survives)}",
        ]
        if math.isinf(self.min_track_record):
            lines.append(
                f"  MinTRL                  never   <- below the bar; more data will not help"
            )
        else:
            lines.append(
                f"  MinTRL            {self.min_track_record:>10.0f}"
                f"   ({self.min_track_record / self.periods_per_year:.0f} years,"
                f" {self.min_track_record - self.n_observations:+.0f} vs what you have)"
            )
        if self.benchmark_asset_sharpe is not None:
            beats = self.sharpe_annualized > self.benchmark_asset_sharpe
            lines += [
                "",
                f"  Buy and hold      {self.benchmark_asset_sharpe:>10.2f}"
                f"   {mark(beats)}  <- before any of this matters",
            ]
        if self.cscv is not None:
            lines += [
                "",
                f"  PBO (CSCV)        {self.cscv.pbo:>10.3f}"
                f"   over {self.cscv.n_splits} splits",
                f"  P(OOS loss)       {self.cscv.probability_of_loss:>10.3f}",
                "  Note: PBO on pure noise has a standard deviation near 0.2, so a",
                "        single mid-range value is not evidence either way. The",
                "        IS->OOS slope is deliberately not shown; see CSCVResult.",
            ]
        lines += ["", f"  VERDICT: {self.verdict()}"]
        return "\n".join(lines)

    def verdict(self) -> str:
        if not self.survives:
            return (
                f"not distinguishable from the best of {self.n_trials} coin flips. "
                f"The bar was {self.benchmark_annualized:.2f} annualized; "
                f"this scored {self.sharpe_annualized:.2f}."
            )
        if self.benchmark_asset_sharpe is not None and (
            self.sharpe_annualized <= self.benchmark_asset_sharpe
        ):
            return (
                "survives deflation, but does not beat holding the asset. "
                "Statistically real, practically pointless."
            )
        return (
            f"survives {self.n_trials} trials (DSR {self.dsr:.3f}). "
            "Worth a walk-forward test on data this search never touched."
        )


def audit(
    returns: pd.DataFrame,
    label: str = "audit",
    n_trials: int | None = None,
    periods_per_year: int = 252,
    n_blocks: int = 10,
    benchmark_returns: pd.Series | None = None,
) -> Autopsy:
    """Audit a (periods x configurations) frame of strategy returns."""
    if returns.shape[1] < 2:
        raise ValueError("need at least 2 configurations to talk about selection")

    declared = n_trials if n_trials is not None else returns.shape[1]
    live = returns.loc[:, returns.std() > 0]
    n_degenerate = returns.shape[1] - live.shape[1]
    if live.shape[1] < 2:
        raise ValueError("fewer than 2 configurations ever took a position")

    sharpes = {c: sharpe_ratio(live[c].values) for c in live.columns}
    winner = max(sharpes, key=sharpes.get)
    sr = sharpes[winner]
    n = len(live)
    skew, kurt = moments(live[winner].values)
    var_sr = float(np.var(list(sharpes.values()), ddof=1))

    dsr, bar = deflated_sharpe_ratio(sr, n, declared, var_sr, skew, kurt)

    result = Autopsy(
        label=label,
        winner=winner,
        sharpe=sr,
        sharpe_annualized=annualize(sr, periods_per_year),
        skewness=skew,
        kurtosis=kurt,
        n_observations=n,
        n_trials=declared,
        n_degenerate=n_degenerate,
        sharpe_variance=var_sr,
        psr=probabilistic_sharpe_ratio(sr, n, skew, kurt),
        benchmark=bar,
        benchmark_annualized=annualize(bar, periods_per_year),
        dsr=dsr,
        min_track_record=min_track_record_length(sr, skew, kurt, benchmark_sharpe=bar),
        periods_per_year=periods_per_year,
        cscv=pbo(live.values, n_blocks=n_blocks),
        benchmark_asset_sharpe=(
            annualize(sharpe_ratio(benchmark_returns.dropna().values), periods_per_year)
            if benchmark_returns is not None
            else None
        ),
        ranking=sorted(
            ((c, annualize(s, periods_per_year)) for c, s in sharpes.items()),
            key=lambda kv: kv[1],
            reverse=True,
        ),
    )
    return result
