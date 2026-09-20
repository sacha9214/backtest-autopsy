"""Price loading with an on-disk cache.

Yahoo rate-limits aggressively (plain HTTP requests get 429 after a handful of
calls), and a study nobody can re-run is not a study. So every download is
cached as CSV: the first run needs the network, every later run does not.
Tests never touch either.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"


def _cache_path(ticker: str, cache_dir: Path) -> Path:
    safe = ticker.replace("^", "_index_").replace("/", "_")
    return cache_dir / f"{safe}.csv"


def load_prices(
    tickers: str | list[str],
    start: str | None = None,
    end: str | None = None,
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Adjusted daily closes, one column per ticker, indexed by date.

    Prices are adjusted for dividends and splits. Using raw closes instead is a
    silent way to invent a strategy: dividend drops look like losses, and a
    split looks like a crash.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    cache_dir = cache_dir or DEFAULT_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)

    series: dict[str, pd.Series] = {}
    for ticker in tickers:
        path = _cache_path(ticker, cache_dir)
        if path.exists() and not refresh:
            frame = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            frame = _download(ticker)
            frame.to_csv(path)
        series[ticker] = frame["close"]

    prices = pd.DataFrame(series).sort_index()
    if start is not None:
        prices = prices.loc[prices.index >= pd.Timestamp(start)]
    if end is not None:
        prices = prices.loc[prices.index <= pd.Timestamp(end)]
    return prices.dropna(how="all")


def _download(ticker: str) -> pd.DataFrame:
    import yfinance as yf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            ticker, period="max", interval="1d",
            auto_adjust=True, progress=False, threads=False,
        )
    if raw is None or raw.empty:
        raise RuntimeError(
            f"no data returned for {ticker!r}. Yahoo may be rate-limiting this "
            "IP; wait a few minutes and retry, or populate data/cache/ by hand."
        )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    out = raw[["Close"]].rename(columns={"Close": "close"})
    out.index.name = "date"
    return out


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns. The first row is dropped, not filled with zero."""
    return prices.pct_change().iloc[1:]
