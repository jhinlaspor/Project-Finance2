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
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.enums import CryptoFeed
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

    request_params = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
    )

    bars = client.get_crypto_bars(request_params, feed=CryptoFeed.US).df

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

def run_phase1() -> None:
    print("[Phase 1] Downloading BTC/USD 1-minute data for the last 180 days…")
    data = fetch_crypto_bars()
    print(f"Data fetched: {len(data):,} bars")

    print("Computing indicators…")
    data = add_indicators(data)
    data = generate_signals(data)

    out_path = "btc_usd_1min.csv"
    data.to_csv(out_path)
    print(f"Saved enriched data to {out_path}")


# ---------------------------------------------------------------------------
# Phase 2 – Back-test & risk filter
# ---------------------------------------------------------------------------

import json
from itertools import product

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # graceful degradation


def compute_equity_curve(
    df: pd.DataFrame,
    atr_mult: float = 1.5,
) -> pd.Series:
    """Simulate strategy returns and return equity curve (indexed by df index)."""
    cash = 1.0  # start with 1 unit equity
    qty = 0.0
    in_pos = False
    trailing_stop = np.nan
    equity = []

    for ts, row in df.iterrows():
        price = row["close"]
        if not in_pos and row["signal"] == 1:
            # Enter long
            qty = cash / price
            cash = 0.0
            in_pos = True
            trailing_stop = price - atr_mult * row["atr14"]
        elif in_pos:
            trailing_stop = max(trailing_stop, price - atr_mult * row["atr14"])
            exit_cross = row["ema50"] < row["ema200"]
            exit_trail = price < trailing_stop
            if exit_cross or exit_trail:
                cash = qty * price
                qty = 0.0
                in_pos = False
                trailing_stop = np.nan
        # Mark equity
        equity_val = cash + qty * price
        equity.append(equity_val)

    equity_series = pd.Series(equity, index=df.index, name="equity")
    return equity_series


def metrics_from_equity(equity: pd.Series) -> dict[str, float]:
    """Compute CAGR, Sharpe (daily), and max drawdown from an equity curve."""
    if len(equity) < 2:
        raise ValueError("Equity curve too short for metrics")

    # Daily returns
    daily_equity = equity.resample("1D").last().dropna()
    daily_returns = daily_equity.pct_change().dropna()
    mean_daily = daily_returns.mean()
    std_daily = daily_returns.std(ddof=0)
    sharpe = (mean_daily / std_daily) * np.sqrt(252) if std_daily > 0 else 0.0

    # CAGR
    days = (equity.index[-1] - equity.index[0]).days or 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (365 / days) - 1

    # Max drawdown
    rolling_max = equity.cummax()
    drawdowns = (equity - rolling_max) / rolling_max
    max_dd = abs(drawdowns.min())

    return {
        "CAGR": float(cagr),
        "Sharpe": float(sharpe),
        "MaxDrawdown": float(max_dd),
    }


def backtest_strategy(
    short_ma: int = 50,
    long_ma: int = 200,
    atr_mult: float = 1.5,
    csv_path: str = "btc_usd_1min.csv",
) -> tuple[dict[str, float], pd.Series]:
    """Run backtest with supplied params and return (metrics, equity_curve)."""
    if not os.path.exists(csv_path):
        run_phase1()  # generate data on-demand
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # Recompute indicators with custom parameters
    df["ema_short"] = df["close"].ewm(span=short_ma, adjust=False).mean()
    df["ema_long"] = df["close"].ewm(span=long_ma, adjust=False).mean()
    df["ema50"] = df["ema_short"]  # for compatibility with exit logic
    df["ema200"] = df["ema_long"]
    # ATR remains the same (14)
    # Adjust signals
    long_cross = (
        (df["ema_short"].shift(1) <= df["ema_long"].shift(1))
        & (df["ema_short"] > df["ema_long"])
    )
    price_above = df["close"] > (df["ema_long"] + df["atr14"])
    df["signal"] = 0
    df.loc[long_cross & price_above, "signal"] = 1

    equity = compute_equity_curve(df, atr_mult=atr_mult)
    metrics = metrics_from_equity(equity)
    return metrics, equity


def run_phase2() -> None:
    print("[Phase 2] Back-testing and risk filter…")
    param_options = {
        "short_ma": [40, 50, 60],
        "long_ma": [150, 200, 250],
        "atr_mult": [1.5, 1.0, 2.0],
    }
    attempts = 0
    best_metrics: dict[str, float] | None = None
    best_equity: pd.Series | None = None
    best_params: dict[str, float] | None = None

    for short_ma, long_ma, atr_mult in product(
        param_options["short_ma"],
        param_options["long_ma"],
        param_options["atr_mult"],
    ):
        if attempts >= 5:
            break
        attempts += 1
        print(f"Attempt {attempts}: short={short_ma} long={long_ma} atr_mult={atr_mult}")
        metrics, equity = backtest_strategy(short_ma, long_ma, atr_mult)
        print(
            f"  → Sharpe={metrics['Sharpe']:.2f}, MaxDD={metrics['MaxDrawdown']:.2%}, CAGR={metrics['CAGR']:.2%}"
        )
        if (
            metrics["Sharpe"] >= 1.0
            and metrics["MaxDrawdown"] <= 0.10
        ):
            best_metrics, best_equity, best_params = metrics, equity, {
                "short_ma": short_ma,
                "long_ma": long_ma,
                "atr_mult": atr_mult,
            }
            print("  → PASS criteria met ✔")
            break
        # keep track of best Sharpe even if fail
        if best_metrics is None or metrics["Sharpe"] > best_metrics["Sharpe"]:
            best_metrics, best_equity, best_params = metrics, equity, {
                "short_ma": short_ma,
                "long_ma": long_ma,
                "atr_mult": atr_mult,
            }
    else:
        print("Max attempts reached without passing criteria.")

    # Persist results
    print("Saving backtest_results.json…")
    results = {
        "params": best_params,
        "metrics": best_metrics,
        "pass": bool(
            best_metrics["Sharpe"] >= 1.0 and best_metrics["MaxDrawdown"] <= 0.10
        ),
        "equity_curve": best_equity.resample("1H").last().fillna(method="ffill").tolist(),
    }
    with open("backtest_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print("backtest_results.json written.")

    # Initialize / update agent_status.yaml
    if yaml is not None:
        status = {
            "mode": "PAPER",
            "last_trade_time": None,
            "cumulative_pnl": 0.0,
        }
        with open("agent_status.yaml", "w", encoding="utf-8") as fh:
            yaml.safe_dump(status, fh)
        print("agent_status.yaml initialized.")
    else:
        print("PyYAML not installed – agent_status.yaml not written.")

    # Summarize
    print("\n=== Back-test Summary ===")
    for k, v in best_metrics.items():
        if k == "MaxDrawdown":
            print(f"{k}: {v:.2%}")
        else:
            print(f"{k}: {v:.4f}")
    print(f"Params: {best_params}")
    if results["pass"]:
        print("Strategy meets risk criteria – ready for Phase 3 (paper trading).")
    else:
        print("Strategy does NOT meet criteria after 5 attempts. Awaiting guidance.")


# ---------------------------------------------------------------------------
# CLI routing
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="BTC EMA crossover trading agent")
    parser.add_argument(
        "--phase",
        choices=["1", "2"],
        default="1",
        help="Which phase to execute (1=design/data, 2=backtest)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = _parse_args()
        if args.phase == "1":
            run_phase1()
        elif args.phase == "2":
            run_phase2()
    except KeyboardInterrupt:  # pragma: no cover
        print("Interrupted by user – exiting…", file=sys.stderr)
        sys.exit(130)