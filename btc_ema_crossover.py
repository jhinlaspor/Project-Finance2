"""
BTC/USD EMA-Crossover Strategy – Phase 1
================================================
Author: Autonomous Trading Agent

This script implements Phase 1 of the multi-phase crypto trading workflow
specified by the user.  It focuses on *strategy design* only – no orders are
submitted here.

Key responsibilities in this phase
----------------------------------
1. Pull 1-minute BTC/USD OHLCV history covering the last 180 days from the
   Alpaca Market-Data v2 crypto API.
2. Engineer indicators:
      • 50-period Exponential Moving Average (EMA-50)
      • 200-period Exponential Moving Average (EMA-200)
      • 14-period Average True Range (ATR-14)
3. Provide helper utilities for later phases (back-test, live trading).

Usage
-----
Execute directly to download the data and cache it locally::

    python btc_ema_crossover.py  # saves data to "btc_usd_1min.csv"

The script respects the standard Alpaca environment variables:
`APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`, and `APCA_API_BASE_URL`.

All functions use a lightweight retry mechanism with exponential back-off as
mandated.  External API calls will be attempted twice before surfacing an
exception.
"""

from __future__ import annotations

import functools
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, TypeVar

import numpy as np
import pandas as pd

try:
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.timeframe import TimeFrame
except ImportError as e:  # pragma: no cover
    sys.stderr.write("alpaca-py not found – install with `pip install alpaca-py`\n")
    raise e

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def retry(max_attempts: int = 2, backoff: float = 1.5):
    """Simple exponential back-off retry decorator.

    Parameters
    ----------
    max_attempts : int
        Total number of attempts *including* the first.
    backoff : float
        Base back-off multiplier in seconds.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            delay = backoff
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    if attempt >= max_attempts - 1:
                        raise  # re-raise after final attempt
                    sys.stderr.write(
                        f"Error in {func.__name__}: {exc}. Retrying in {delay:.1f}s…\n"
                    )
                    time.sleep(delay)
                    delay *= 2  # exponential step
            # Unreachable – satisfies static analyzers
            raise RuntimeError("Unreachable")

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------

@retry()
def fetch_crypto_bars(
    symbol: str = "BTC/USD",
    days: int = 180,
    timeframe: TimeFrame = TimeFrame.Minute,
) -> pd.DataFrame:
    """Download 1-minute OHLCV data for *symbol* covering the last *days* days.

    Returns a *pandas* ``DataFrame`` indexed by timestamp (UTC).
    """

    client = CryptoHistoricalDataClient()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    bars = client.get_crypto_bars(symbol, timeframe, start=start, end=end).df

    if bars.empty:
        raise RuntimeError("Received no data from Alpaca crypto API")

    # The returned DataFrame is multi-indexed by [symbol, timestamp]. Flatten.
    bars = bars.droplevel(level=0)
    bars = bars.sort_index()

    # Rename columns to conventional OHLCV lower-case naming
    bars = bars.rename(
        columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
    )

    return bars


# ---------------------------------------------------------------------------
# Indicator engineering
# ---------------------------------------------------------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute EMA-50, EMA-200 and ATR-14 and append as new columns."""

    if df.isnull().values.any():
        df = df.dropna().copy()  # ensure clean data for rolling calcs

    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    # True Range (TR)
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr_components = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    tr = tr_components.max(axis=1)

    df["atr14"] = tr.rolling(window=14).mean()

    return df


# ---------------------------------------------------------------------------
# Signal generation (for later phases)
# ---------------------------------------------------------------------------

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add *signal* column: +1 for long entry, 0 otherwise.

    Exit signals or stop logic are handled in later phases; here we only mark
    entry points per the strategy specification.
    """

    long_cross = (df["ema50"].shift(1) <= df["ema200"].shift(1)) & (df["ema50"] > df["ema200"])
    price_above = df["close"] > (df["ema200"] + df["atr14"])

    df["signal"] = 0
    df.loc[long_cross & price_above, "signal"] = 1

    return df


# ---------------------------------------------------------------------------
# Main routine – Phase 1 execution
# ---------------------------------------------------------------------------

def main() -> None:
    print("[Phase 1] Downloading BTC/USD 1-minute data for the last 180 days…")
    data = fetch_crypto_bars()
    print(f"Data fetched: {len(data):,} bars")

    print("Computing indicators…")
    data = add_indicators(data)
    data = generate_signals(data)

    out_path = "btc_usd_1min.csv"
    data.to_csv(out_path)
    print(f"Saved enriched data to {out_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # pragma: no cover
        print("Interrupted by user – exiting…", file=sys.stderr)
        sys.exit(130)