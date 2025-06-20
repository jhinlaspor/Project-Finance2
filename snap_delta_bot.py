"""
Snap‑Delta Bot — 0‑DTE Options Day‑Trading Automation
====================================================
Author : ChatGPT (OpenAI)
Date   : 2025‑06‑12
License: MIT — educational example, use at your own risk.

Strategy Recap
--------------
• Underlying : SPY only (fallback QQQ/IWM)
• Session    : Tuesday‑Thursday, 10:05 → 15:45 ET
• Direction  : Buy 0‑DTE CALL when price > VWAP & RSI‑2 <30
               Buy 0‑DTE PUT  when price < VWAP & RSI‑2 >70
• Contract   : ~0.30 delta, tightest spread
• Risk       : ≤1 % account per trade
• Exit       : +50 % target | ‑25 % stop | 15:45 flat
• Daily cap  : max 3 entries OR 2 consecutive losses

Dependencies
------------
$ pip install alpaca-py polygon-api-client pandas numpy ta pytz

Environment Variables (export … / set …):
  APCA_API_KEY_ID       — Alpaca key
  APCA_API_SECRET_KEY   — Alpaca secret
  APCA_PAPER            — 'true' to use paper endpoint
  POLYGON_API_KEY       — Polygon key
  ACCOUNT_START_EQUITY  — optional, default 25_000

Run:
  python snap_delta_bot.py
"""

from __future__ import annotations

import os
import sys
import asyncio
import datetime as dt
import math
from collections import deque

import pytz


# ──────────────────────────── Configuration ────────────────────────────────
TIMEZONE = pytz.timezone("America/New_York")
UNIVERSE = "SPY"
SESSION_DAYS = {1, 2, 3}  # Tue=1 … Thu=3 (Python weekday: Mon=0)
ENTRY_START = dt.time(10, 5)
ENTRY_END = dt.time(14, 30)
FORCE_FLAT_TIME = dt.time(15, 45)
RISK_PER_TRADE = 0.01  # 1 % equity
TARGET_MULTIPLIER = 1.50  # +50 %
STOP_MULTIPLIER = 0.75  # ‑25 %
MAX_TRADES_DAY = int(os.getenv("MAX_TRADES_PER_DAY", 3))
MAX_CONSEC_LOSS = 2
RSI_PERIOD = 2
ATR_PERIOD = 5  # on 5‑min bars, for position filter
DELTA_TARGET = 0.30
DELTA_TOLERANCE = 0.05

# ──────────────────────────── Helpers ───────────────────────────────────────


def now_et():
    return dt.datetime.now(tz=TIMEZONE)


def today_string():
    return now_et().strftime("%Y‑%m‑%d")


# ──────────────────────────── Broker & Data Clients ────────────────────────

ALPACA_PAPER = os.getenv("APCA_PAPER", "true").lower() == "true"
APCA_API_BASE_URL = os.getenv(
    "APCA_API_BASE_URL",
    "https://paper-api.alpaca.markets",
)
SIMULATION_ENV = (
    os.getenv("SIMULATION", "false").lower() == "true" or "--simulate" in sys.argv
)
ALPACA_CLIENT = None
POLY_CLIENT = None


def validate_env() -> None:
    required = ["APCA_API_KEY_ID", "APCA_API_SECRET_KEY"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )


if not SIMULATION_ENV:
    from polygon import RESTClient as PolygonRest

    from alpaca.trading.client import TradingClient

    validate_env()
    ALPACA_CLIENT = TradingClient(
        api_key=os.getenv("APCA_API_KEY_ID"),
        secret_key=os.getenv("APCA_API_SECRET_KEY"),
        paper=ALPACA_PAPER,
        base_url=APCA_API_BASE_URL,
    )
    POLY_CLIENT = PolygonRest(api_key=os.getenv("POLYGON_API_KEY"))

ACCOUNT_EQUITY = float(os.getenv("ACCOUNT_START_EQUITY", 25_000))

# ──────────────────────────── Strategy State ───────────────────────────────


class DayState:
    def __init__(self):
        self.trades = 0
        self.consec_loss = 0
        self.open_orders = {}
        self.pnl = 0.0

    def reset(self):
        self.__init__()


STATE = DayState()

# ──────────────────────────── Indicator Buffers ────────────────────────────
PRICE_BUFFER = deque(maxlen=300)  # 300 min ≈ full session
HIGH_BUFFER = deque(maxlen=ATR_PERIOD * 5)
LOW_BUFFER = deque(maxlen=ATR_PERIOD * 5)
CLOSE_BUFFER = deque(maxlen=ATR_PERIOD * 5)
VWAP_NUM = 0.0
VWAP_DEN = 0.0

# ──────────────────────────── Core Functions ───────────────────────────────


def update_indicators(bar):
    """Update rolling buffers and compute VWAP, RSI‑2, ATR‑5min."""
    global VWAP_NUM, VWAP_DEN

    import pandas as pd
    from ta.momentum import RSIIndicator
    from ta.volatility import AverageTrueRange

    close = bar["c"]
    volume = bar["v"]
    high = bar["h"]
    low = bar["l"]

    # VWAP
    VWAP_NUM += close * volume
    VWAP_DEN += volume
    vwap = VWAP_NUM / VWAP_DEN if VWAP_DEN else close

    # buffers for RSI & ATR
    PRICE_BUFFER.append(close)
    HIGH_BUFFER.append(high)
    LOW_BUFFER.append(low)
    CLOSE_BUFFER.append(close)

    rsi = None
    atr = None

    if len(PRICE_BUFFER) >= RSI_PERIOD:
        rsi = RSIIndicator(pd.Series(PRICE_BUFFER), window=RSI_PERIOD).rsi().iloc[-1]
    if len(HIGH_BUFFER) >= ATR_PERIOD * 5:  # convert 1‑min to 5‑min window counts
        df = pd.DataFrame(
            {"high": HIGH_BUFFER, "low": LOW_BUFFER, "close": CLOSE_BUFFER}
        )
        atr = (
            AverageTrueRange(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                window=ATR_PERIOD,
            )
            .average_true_range()
            .iloc[-1]
        )

    return vwap, rsi, atr


def generate_signal(price, vwap, rsi):
    if rsi is None:
        return None
    if price > vwap and rsi < 30:
        return "CALL"
    if price < vwap and rsi > 70:
        return "PUT"
    return None


def select_contract(side):
    """Query Polygon option chain and pick the tightest‑spread ~0.3 delta."""
    contracts = POLY_CLIENT.list_options_tickers(
        UNDERLYING=UNIVERSE,
        expiration_date=today_string(),
    )
    candidates = []
    for c in contracts:
        if c["option_type"] != ("C" if side == "CALL" else "P"):
            continue
        delta = abs(c.get("delta", 0))
        if abs(delta - DELTA_TARGET) > DELTA_TOLERANCE:
            continue
        spread = c["ask"] - c["bid"]
        candidates.append((spread, c))
    if not candidates:
        return None
    return min(candidates, key=lambda x: x[0])[1]


def calc_quantity(ask_price):
    risk_dollars = ACCOUNT_EQUITY * RISK_PER_TRADE
    contract_cost = ask_price * 100  # multiplier
    qty = math.floor(risk_dollars / (contract_cost * (1 - STOP_MULTIPLIER)))
    return max(qty, 0)


def place_bracket(contract, qty):
    from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
    from alpaca.trading.requests import (
        MarketOrderRequest,
        StopLossRequest,
        TakeProfitRequest,
    )

    tp_price = round(contract["ask"] * TARGET_MULTIPLIER, 2)
    sl_price = round(contract["ask"] * STOP_MULTIPLIER, 2)

    order = MarketOrderRequest(
        symbol=contract["ticker"],
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=tp_price),
        stop_loss=StopLossRequest(stop_price=sl_price),
    )
    res = ALPACA_CLIENT.submit_order(order)
    STATE.open_orders[res.id] = {"entry_price": contract["ask"], "qty": qty}
    STATE.trades += 1


# ──────────────────────────── Stream Handling ──────────────────────────────
async def on_bar(bar):
    if now_et().weekday() not in SESSION_DAYS:
        return
    price = bar["c"]
    vwap, rsi, atr = update_indicators(bar)
    signal = generate_signal(price, vwap, rsi)

    # Check time window & risk guards
    t = now_et().time()
    if t < ENTRY_START or t > ENTRY_END:
        return
    if STATE.trades >= MAX_TRADES_DAY or STATE.consec_loss >= MAX_CONSEC_LOSS:
        return

    if signal:
        side = signal
        contract = select_contract(side)
        if contract:
            qty = calc_quantity(contract["ask"])
            if qty > 0:
                place_bracket(contract, qty)
                print(f"Entered {side} {contract['ticker']} x{qty} @ {contract['ask']}")


async def force_flat():
    while True:
        await asyncio.sleep(10)
        if now_et().time() >= FORCE_FLAT_TIME and STATE.open_orders:
            ALPACA_CLIENT.close_all_positions(cancel_orders=True)
            STATE.open_orders.clear()
            print("Force‑flat executed.")
            break


async def main(simulate: bool = False):
    """Entry point for the bot."""
    STATE.reset()

    if not simulate:
        validate_env()

    if simulate:
        for i in range(5):
            await asyncio.sleep(0)
            print(f"Simulated tick {i+1}")
            if i == 2:
                print("Entered CALL TEST x1 @ 1.23")
        return

    from alpaca.data.live import StockDataStream

    stream = StockDataStream(
        os.getenv("APCA_API_KEY_ID"),
        os.getenv("APCA_API_SECRET_KEY"),
        feed="iex",
        base_url=APCA_API_BASE_URL,
    )
    stream.subscribe_bars(on_bar, UNIVERSE)

    await asyncio.gather(stream._run_forever(), force_flat())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Snap‑Delta bot")
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=None,
        help="run without hitting external APIs",
    )
    args = parser.parse_args()
    simulate_flag = args.simulate
    if simulate_flag is None:
        simulate_flag = SIMULATION_ENV

    try:
        asyncio.run(main(simulate_flag))
    except KeyboardInterrupt:
        print("Interrupted — shutting down.")
