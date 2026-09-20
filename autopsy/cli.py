"""Command line entry point: autopsy returns.csv --trials 42"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .report import audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="autopsy",
        description="Deflate a backtest by the number of trials it took to find it.",
    )
    parser.add_argument("csv", help="CSV of returns: one column per configuration")
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="configurations tried, INCLUDING abandoned ones "
        "(default: number of columns, which undercounts if you discarded any)",
    )
    parser.add_argument("--periods-per-year", type=int, default=252)
    parser.add_argument(
        "--blocks", type=int, default=10, help="CSCV blocks, must be even"
    )
    parser.add_argument("--top", type=int, default=0, help="also list the top N")
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.csv, index_col=0).apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(how="all", axis=1).dropna(how="any")
    if frame.empty:
        print(f"error: no usable numeric data in {args.csv}", file=sys.stderr)
        return 2

    try:
        result = audit(
            frame,
            label=args.csv,
            n_trials=args.trials,
            periods_per_year=args.periods_per_year,
            n_blocks=args.blocks,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(result.render())
    if args.top:
        print("\n  Top in-sample:")
        for name, sharpe in result.ranking[: args.top]:
            print(f"    {sharpe:>6.2f}  {name}")
    return 0 if result.survives else 1


if __name__ == "__main__":
    raise SystemExit(main())
