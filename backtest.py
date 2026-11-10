"""Replays the rules in strategy.py over past daily prices.

    .venv/bin/python backtest.py              # last 2 years, settings from config.py
    .venv/bin/python backtest.py --years 1
    .venv/bin/python backtest.py --compare    # also show the old bot's rule and other variants

Assumptions, all leaning pessimistic:
  * Daily bars only. If a day's range touches both a stop and a profit target, the stop is assumed first.
  * If the price opens beyond a level (a gap), the fill is at the open, not the level.
  * 0.1% is lost to slippage on every buy and every sell.
  * Sale proceeds cannot be spent until the next trading day.
"""

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

import strategy
from broker import data_client
from config import SETTINGS, WATCHLIST, Settings

SLIPPAGE = 0.001


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
        raise SystemExit(f"Could not download prices: {last_error}")
    bars = {}
    for symbol in symbols:
        if symbol in frame.index.get_level_values(0):
            one = frame.xs(symbol, level=0)[list(columns)].copy()
            one.index = pd.to_datetime(one.index).tz_convert("America/New_York").normalize().tz_localize(None)
            bars[symbol] = one
    return bars


def average_range(df: pd.DataFrame, days: int = 14) -> pd.Series:
    """Average true range: the stock's typical daily move in dollars."""
    prev_close = df["close"].shift(1)
    true_range = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(),
                            (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    return true_range.rolling(days).mean()


@dataclass
class Trade:
    symbol: str
    bought: pd.Timestamp
    sold: pd.Timestamp
    entry: float
    exit: float
    dollars: float
    reason: str

    @property
    def profit(self) -> float:
        return self.dollars * (self.exit / self.entry - 1)


def run(bars: dict[str, pd.DataFrame], s: Settings, start: pd.Timestamp, legacy_exit: bool = False):
    """Returns (equity curve, list of trades). legacy_exit reproduces the old bot:
    sell at 1% above yesterday's high and never cut a loss."""
    atrs = {sym: average_range(df) for sym, df in bars.items()}
    trend = {sym: df["close"].rolling(s.trend_days).mean() if s.trend_days > 0 else None for sym, df in bars.items()}
    days = sorted({d for df in bars.values() for d in df.index if d >= start})
    cash = s.budget
    holding: dict[str, dict] = {}
    blocked_until: dict[str, pd.Timestamp] = {}
    trades: list[Trade] = []
    realized = 0.0
    curve = []

    for day in days:
        unsettled = 0.0  # proceeds from today's sells; spendable tomorrow

        def sell(symbol: str, price: float, reason: str):
            nonlocal cash, unsettled, realized
            pos = holding.pop(symbol)
            price *= 1 - SLIPPAGE
            trade = Trade(symbol, pos["bought"], day, pos["entry"], price, pos["dollars"], reason)
            trades.append(trade)
            proceeds = pos["dollars"] * price / pos["entry"]
            cash += proceeds
            unsettled += proceeds
            realized += trade.profit
            if trade.profit < 0 and s.loss_cooldown_days > 0:
                blocked_until[symbol] = day + pd.Timedelta(days=s.loss_cooldown_days)

        # 1. Exits on positions held coming into today.
        for symbol in list(holding):
            df = bars[symbol]
            if day not in df.index:
                continue
            bar, pos = df.loc[day], holding[symbol]
            if legacy_exit:
                i = df.index.get_loc(day)
                target = df["high"].iloc[i - 1] * 1.01
                if bar["high"] >= target:
                    sell(symbol, max(bar["open"], target), "target")
                continue
            stop = strategy.stop_level(pos["entry"], pos["peak"], s, pos["atr"])
            target = strategy.target_level(pos["entry"], s)
            if bar["low"] <= stop:
                reason = "stop-loss" if stop < pos["entry"] else "trailing stop"
                sell(symbol, min(bar["open"], stop), reason)
            elif target is not None and bar["high"] >= target:
                sell(symbol, max(bar["open"], target), "target")
            else:
                pos["peak"] = max(pos["peak"], bar["high"])

        # 2. Entries.
        for symbol in WATCHLIST:
            df = bars.get(symbol)
            if df is None or day not in df.index or symbol in holding:
                continue
            if len(holding) >= s.max_positions or realized <= -s.max_total_loss:
                break
            if symbol in blocked_until and day < blocked_until[symbol]:
                continue
            if any(t.symbol == symbol and t.sold == day for t in trades):
                continue  # no rebuying a stock on the day it was sold
            i = df.index.get_loc(day)
            if i == 0:
                continue
            prev, bar = df.iloc[i - 1], df.iloc[i]
            trend_avg = None if trend[symbol] is None else trend[symbol].iloc[i - 1]
            if trend_avg is not None and pd.isna(trend_avg):
                trend_avg = None
            if not strategy.trend_ok(prev["close"], trend_avg, s):
                continue
            level = strategy.entry_level(prev["low"], s)
            if bar["low"] > level:
                continue
            entry = min(bar["open"], level) * (1 + SLIPPAGE)
            atr = atrs[symbol].iloc[i - 1]
            atr = None if pd.isna(atr) else float(atr)
            size = strategy.position_dollars(s, entry, atr)
            spendable = cash - (unsettled if s.settled_cash_only else 0.0)
            if spendable < size or size < 1:
                continue
            cash -= size
            holding[symbol] = {"entry": entry, "peak": entry, "dollars": size, "bought": day, "atr": atr}
            # The stop sits under the fill, so a low beneath it today happened after we bought.
            first_stop = strategy.stop_level(entry, entry, s, atr)
            if not legacy_exit and bar["low"] <= first_stop:
                sell(symbol, first_stop, "stop-loss")
            else:
                holding[symbol]["peak"] = max(entry, bar["close"])

        value = cash
        for symbol, pos in holding.items():
            df = bars[symbol]
            close = df["close"].asof(day)
            value += pos["dollars"] * close / pos["entry"]
        curve.append((day, value))

    return pd.Series(dict(curve)), trades


def buy_and_hold(bars: dict[str, pd.DataFrame], budget: float, start: pd.Timestamp) -> pd.Series:
    """Budget split evenly across the watchlist on day one and never touched."""
    closes = pd.DataFrame({sym: df["close"] for sym, df in bars.items()})
    closes = closes[closes.index >= start].ffill().dropna(axis=1, how="all")
    first = closes.bfill().iloc[0]
    return (closes / first).mean(axis=1) * budget


def summarize(name: str, curve: pd.Series, trades: list[Trade] | None, budget: float) -> dict:
    drawdown = (curve / curve.cummax() - 1).min()
    row = {"rule": name, "end $": round(curve.iloc[-1], 2), "return": f"{curve.iloc[-1] / budget - 1:+.1%}",
           "worst drop": f"{drawdown:.1%}", "lowest $": round(curve.min(), 2)}
    if trades is not None:
        wins = [t.profit for t in trades if t.profit > 0]
        losses = [t.profit for t in trades if t.profit <= 0]
        row |= {"trades": len(trades),
                "win rate": f"{len(wins) / len(trades):.0%}" if trades else "-",
                "avg win $": round(sum(wins) / len(wins), 2) if wins else 0,
                "avg loss $": round(sum(losses) / len(losses), 2) if losses else 0,
                "worst trade $": round(min((t.profit for t in trades), default=0), 2)}
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=float, default=2.0)
    parser.add_argument("--compare", action="store_true", help="also test the old bot's rule and other variants")
    parser.add_argument("--trades", action="store_true", help="list every trade for the config.py settings")
    args = parser.parse_args()

    test_days = int(args.years * 365)
    bars = fetch_daily_bars(WATCHLIST, test_days + 120)  # extra history warms up the trend average
    start = pd.Timestamp(datetime.now().date() - timedelta(days=test_days))
    missing = [s for s in WATCHLIST if s not in bars]
    if missing:
        print("No price data for:", ", ".join(missing))

    s = SETTINGS
    curve, trades = run(bars, s, start)
    rows = [summarize("config.py settings", curve, trades, s.budget)]

    if args.compare:
        variants = {
            "old bot (no stop, no filter)": (replace(s, trend_days=0, loss_cooldown_days=0, risk_per_trade=0), True),
            "fixed 8% stop, 5% trail, $25 each": (replace(s, atr_mult=0, risk_per_trade=0), False),
            "volatility stop 2x": (replace(s, atr_mult=2.0), False),
            "volatility stop 4x": (replace(s, atr_mult=4.0), False),
            "no trend filter": (replace(s, trend_days=0), False),
            "no loss cooldown": (replace(s, loss_cooldown_days=0), False),
            "fixed +10% profit target": (replace(s, take_profit_pct=0.10), False),
            "deeper dip (3% under low)": (replace(s, dip_pct=0.03), False),
            "full $25 positions (no risk sizing)": (replace(s, risk_per_trade=0), False),
        }
        for name, (variant, legacy) in variants.items():
            c, t = run(bars, variant, start, legacy_exit=legacy)
            rows.append(summarize(name, c, t, s.budget))
    rows.append(summarize("buy and hold the watchlist", buy_and_hold(bars, s.budget, start), None, s.budget))

    print(f"\nBacktest {curve.index[0].date()} to {curve.index[-1].date()}, starting with ${s.budget:.0f}\n")
    print(pd.DataFrame(rows).astype(object).fillna("").to_string(index=False))

    if args.trades:
        print("\nTrades (config.py settings):")
        for t in trades:
            print(f"  {t.bought.date()} -> {t.sold.date()}  {t.symbol:5} in {t.entry:8.2f} out {t.exit:8.2f} "
                  f"{t.profit:+7.2f}  {t.reason}")
        by_symbol = pd.Series({sym: sum(t.profit for t in trades if t.symbol == sym) for sym in WATCHLIST})
        print("\nProfit by stock ($):")
        print(by_symbol.round(2).to_string())


if __name__ == "__main__":
    main()
