"""Would this tool have caught my own false positives?

Before building any of this I spent three weeks searching for an edge on
Polymarket sports markets and produced results that looked real and were not.
This script feeds them back through the corrections using only what was known
at the time.

One input is missing from the published figures: the variance of the Sharpe
ratios across the trials that were run. Rather than invent it, this script
solves for the value at which each verdict would flip, and asks whether that
value is plausible. That is a stronger statement than a single number, because
it does not depend on a parameter nobody recorded.
"""

from __future__ import annotations

import math

from autopsy.stats import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)

CASES = [
    {
        "name": "Price zone 0.50-0.75",
        "t_stat": 2.41,
        "n_obs": 1404,
        "n_trials": 8,
        "claimed": "+7.61% per entry, t = 2.41 -- significant by the textbook",
        "truth": "+0.06% out-of-sample, t = 0.03, on 1404 fresh observations",
    },
    {
        "name": "Pair assembly",
        "t_stat": 1.10,
        "n_obs": 30,
        "n_trials": 12,
        "claimed": "+3.05% per pair",
        "truth": "-7.25% over 30 days, 0 positive days out of 30",
    },
    {
        "name": "Ladderbot residual",
        "t_stat": -0.46,
        "n_obs": 90,
        "n_trials": 6,
        "claimed": "+6.9% on the naked residual",
        "truth": "95% CI [-31.9%, +19.8%]; the figure wandered 5.7 -> 13.5 -> 2.2 -> 4.5",
    },
]

# Measured on this repository's own SPY grid: 43 live configurations on daily
# data gave an across-trial Sharpe variance of 3.4e-4. That is the order of
# magnitude a real search produces, and the reference point used below.
TYPICAL_DAILY_VARIANCE = 3.4e-4


def sharpe_from_t(t_stat: float, n_obs: int) -> float:
    return t_stat / math.sqrt(n_obs)


def critical_variance(sharpe: float, n_obs: int, n_trials: int) -> float | None:
    """Largest across-trial variance at which this result still passes at 0.95.

    Returns None if it fails even with a vanishing bar, meaning the result was
    never significant regardless of how little searching went on.
    """
    if probabilistic_sharpe_ratio(sharpe, n_obs) <= 0.95:
        return None
    low, high = 1e-12, 10.0
    if deflated_sharpe_ratio(sharpe, n_obs, n_trials, high)[0] > 0.95:
        return high
    for _ in range(200):
        mid = math.sqrt(low * high)
        if deflated_sharpe_ratio(sharpe, n_obs, n_trials, mid)[0] > 0.95:
            low = mid
        else:
            high = mid
    return low


def main() -> None:
    print("Would the corrections have caught these before the fact?")
    print(f"(reference: a real daily search here gave V[SR] = {TYPICAL_DAILY_VARIANCE:.1e})\n")

    caught = 0
    for case in CASES:
        sharpe = sharpe_from_t(case["t_stat"], case["n_obs"])
        psr = probabilistic_sharpe_ratio(sharpe, case["n_obs"])
        dsr, bar = deflated_sharpe_ratio(
            sharpe, case["n_obs"], case["n_trials"], TYPICAL_DAILY_VARIANCE
        )
        critical = critical_variance(sharpe, case["n_obs"], case["n_trials"])
        rejected = dsr <= 0.95
        caught += rejected

        print(f"  {case['name']}")
        print(f"    claimed             : {case['claimed']}")
        print(f"    actually            : {case['truth']}")
        print(f"    per-period Sharpe   : {sharpe:>9.4f}  "
              f"(t = {case['t_stat']:+.2f}, n = {case['n_obs']}, "
              f"{case['n_trials']} trials)")
        print(f"    PSR vs zero         : {psr:>9.4f}  "
              f"{'<- textbook says significant' if psr > 0.95 else ''}")
        print(f"    DSR at V[SR]={TYPICAL_DAILY_VARIANCE:.0e}  : {dsr:>9.4f}  "
              f"{'REJECTED' if rejected else 'SURVIVES'}")
        if critical is None:
            print("    flips at V[SR]      :     never  "
                  "(fails even with no search at all)")
        else:
            ratio = critical / TYPICAL_DAILY_VARIANCE
            print(f"    flips at V[SR]      : {critical:>9.2e}  "
                  f"({ratio:.2f}x the reference)")
        print()

    print(f"  Caught before the fact: {caught}/{len(CASES)}")
    print()
    print("  The point is not that the tool is clever. The zone 0.50-0.75 result")
    print("  had a textbook t of 2.41 and a PSR above 0.99: every standard test")
    print("  said yes. What killed it was counting the eight strategies tried")
    print("  before it -- information no amount of extra data can recover once")
    print("  the count is lost.")


if __name__ == "__main__":
    main()
