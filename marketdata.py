"""Price downloads from Alpaca's free data, shared by the bot, backtest and screener."""

import pickle
from datetime import date, datetime, timedelta, timezone

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from broker import ROOT, data_client

ALL_COLUMNS = ("open", "high", "low", "close", "volume")


def fetch_daily_bars(symbols: list[str], days: int,
                     columns: tuple[str, ...] = ("open", "high", "low", "close")) -> dict[str, pd.DataFrame]:
    """Daily bars per symbol, split-adjusted, oldest first."""
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=days)
    client = data_client()
    last_error = None
    for feed in (DataFeed.SIP, DataFeed.IEX):
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbols, timeframe=TimeFrame.Day, start=start, end=end,
                adjustment=Adjustment.SPLIT, feed=feed,
            )
            frame = client.get_stock_bars(request).df
            break
        except Exception as error:  # the free plan may refuse one feed; try the other
            last_error = error
    else:
        raise RuntimeError(f"Could not download prices: {last_error}")
    bars = {}
    if frame.empty:
        return bars
    present = set(frame.index.get_level_values(0))
    for symbol in symbols:
        if symbol in present:
            one = frame.xs(symbol, level=0)[list(columns)].copy()
            one.index = pd.to_datetime(one.index).tz_convert("America/New_York").normalize().tz_localize(None)
            bars[symbol] = one
    return bars


def fetch_many(symbols: list[str], days: int, progress: bool = False) -> dict[str, pd.DataFrame]:
    """Daily bars for a large list of stocks, kept on disk for the rest of the day
    so a second run does not download everything again."""
    cache_dir = ROOT / "state" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"bars_{date.today().isoformat()}_{days}d_{len(symbols)}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    bars = {}
    for i in range(0, len(symbols), 200):
        bars |= fetch_daily_bars(symbols[i:i + 200], days, columns=ALL_COLUMNS)
        if progress:
            print(f"  downloaded {min(i + 200, len(symbols))} of {len(symbols)} stocks", end="\r", flush=True)
    if progress:
        print()
    for old in cache_dir.glob("bars_*.pkl"):
        if date.today().isoformat() not in old.name:  # drop earlier days' downloads
            old.unlink()
    cache.write_bytes(pickle.dumps(bars))
    return bars


def average_range(df: pd.DataFrame, days: int = 14) -> pd.Series:
    """Average true range: the stock's typical daily move in dollars."""
    prev_close = df["close"].shift(1)
    true_range = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(),
                            (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    return true_range.rolling(days).mean()
