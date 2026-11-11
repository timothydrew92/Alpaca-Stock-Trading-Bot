"""Replays the bot's rules over past daily prices.

    .venv/bin/python backtest.py                # last 2 years
    .venv/bin/python backtest.py --years 1
    .venv/bin/python backtest.py --compare      # also show other rule variants
    .venv/bin/python backtest.py --trades       # list every trade
    .venv/bin/python backtest.py --symbols IONQ RGTI NVDA   # test a fixed list instead

By default the watchlist is rebuilt every week exactly as the bot does it, using only
prices that existed at that point, so the test cannot pick stocks with hindsight.

Assumptions, mostly leaning pessimistic:
  * Daily bars only. If a day's range touches both a stop and a profit target, the stop is assumed first.
  * If the price opens beyond a level (a gap), the fill is at the open, not the level.
  * 0.1% is lost to slippage on every buy and every sell.
  * Sale proceeds cannot be spent until the next trading day.
One assumption leans optimistic: only stocks still listed today are scanned, so
companies that went bust or were delisted along the way are missing.
"""

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

import pandas as pd

import screen
import strategy
from config import MY_PICKS, SCREEN, SETTINGS, Settings
from marketdata import average_range, fetch_daily_bars, fetch_many

SLIPPAGE = 0.001


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


def run(bars: dict[str, pd.DataFrame], s: Settings, start: pd.Timestamp,
        watchlists: dict[pd.Timestamp, list[str]], legacy_exit: bool = False):
    """Returns (equity curve, list of trades). `watchlists` maps a date to the list
    in force from that date on. legacy_exit reproduces the old bot: sell at 1% above
    yesterday's high and never cut a loss."""
    used = {sym for symbols in watchlists.values() for sym in symbols}
    bars = {sym: df for sym, df in bars.items() if sym in used}
    changes = sorted(watchlists)
    watchlist: list[str] = []
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
        while changes and changes[0] <= day:
            watchlist = watchlists[changes.pop(0)]

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
        for symbol in watchlist:
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
            size = strategy.position_dollars(s, entry, atr, realized)
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


def buy_and_hold(bars: dict[str, pd.DataFrame], symbols: list[str], budget: float, start: pd.Timestamp) -> pd.Series:
    """Budget split evenly across the given stocks on day one and never touched."""
    closes = pd.DataFrame({sym: bars[sym]["close"] for sym in symbols if sym in bars})
    closes = closes[closes.index >= start].ffill().dropna(axis=1, how="all")
    first = closes.bfill().iloc[0]
    return (closes / first).mean(axis=1) * budget


def weekly_watchlists(bars: dict[str, pd.DataFrame], names: dict[str, str], start: pd.Timestamp) -> dict:
    """The screener's list as it would have been at the start of each week."""
    m = screen.measures(bars)
    days = [d for d in m["price"].index if d >= start]
    lists = {}
    for day in days[::5]:
        ranked = screen.ranking(m, names, SCREEN, before=day)
        lists[day] = MY_PICKS + [s for s in ranked.head(SCREEN.size).index if s not in MY_PICKS]
    return lists


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
    parser.add_argument("--compare", action="store_true", help="also test other rule variants")
    parser.add_argument("--trades", action="store_true", help="list every trade for the config.py settings")
    parser.add_argument("--symbols", nargs="+", metavar="SYM", help="test this fixed list instead of the weekly screen")
    args = parser.parse_args()

    s = SETTINGS
    test_days = int(args.years * 365)
    start = pd.Timestamp(datetime.now().date() - timedelta(days=test_days))
    market = fetch_daily_bars(["SPY"], test_days + 10)

    if args.symbols:
        symbols = [x.upper() for x in args.symbols]
        bars = fetch_daily_bars(symbols, test_days + 120)  # extra history warms up the averages
        missing = [x for x in symbols if x not in bars]
        if missing:
            print("No price data for:", ", ".join(missing))
        watchlists = {start: symbols}
        label = "fixed list: " + " ".join(symbols)
    else:
        names = screen.universe()
        load_days = max(test_days, 730) + 420  # one size for most runs, so the day's download is reused
        print(f"Loading {load_days} days of prices for {len(names)} stocks "
              "(several minutes the first time each day)...")
        bars = fetch_many(sorted(names), load_days, progress=True)
        watchlists = weekly_watchlists(bars, names, start)
        label = f"weekly screen, top {SCREEN.size}"

    curve, trades = run(bars, s, start, watchlists)
    rows = [summarize("config.py settings", curve, trades, s.budget)]

    if args.compare:
        variants = {
            "fixed 8% stop, 5% trail, $25 each": replace(s, atr_mult=0, risk_per_trade=0),
            "volatility stop 2x": replace(s, atr_mult=2.0),
            "volatility stop 4x": replace(s, atr_mult=4.0),
            "no trend filter": replace(s, trend_days=0),
            "no loss cooldown": replace(s, loss_cooldown_days=0),
            "fixed +10% profit target": replace(s, take_profit_pct=0.10),
            "deeper dip (3% under low)": replace(s, dip_pct=0.03),
            "full $25 positions (no risk sizing)": replace(s, risk_per_trade=0),
            "no loss limit": replace(s, max_total_loss=1e9),
            "profits not reinvested" if s.reinvest_profits else "profits reinvested":
                replace(s, reinvest_profits=not s.reinvest_profits),
        }
        for name, variant in variants.items():
            c, t = run(bars, variant, start, watchlists)
            rows.append(summarize(name, c, t, s.budget))
        if args.symbols:
            old = replace(s, trend_days=0, loss_cooldown_days=0, risk_per_trade=0)
            c, t = run(bars, old, start, watchlists, legacy_exit=True)
            rows.append(summarize("2024 bot (no stop, no filter)", c, t, s.budget))
    if args.symbols:
        rows.append(summarize("buy and hold this list", buy_and_hold(bars, symbols, s.budget, start), None, s.budget))
    rows.append(summarize("buy and hold S&P 500 (SPY)", buy_and_hold(market, ["SPY"], s.budget, start), None, s.budget))

    print(f"\nBacktest {curve.index[0].date()} to {curve.index[-1].date()}, starting with ${s.budget:.0f} ({label})\n")
    print(pd.DataFrame(rows).astype(object).fillna("").to_string(index=False))

    if not args.symbols:
        ever = {sym for symbols in watchlists.values() for sym in symbols}
        print(f"\n{len(ever)} different stocks appeared on the weekly list; {len({t.symbol for t in trades})} were traded.")
    if args.trades:
        print("\nTrades (config.py settings):")
        for t in trades:
            print(f"  {t.bought.date()} -> {t.sold.date()}  {t.symbol:5} in {t.entry:8.2f} out {t.exit:8.2f} "
                  f"{t.profit:+7.2f}  {t.reason}")
        by_symbol = pd.Series({sym: sum(t.profit for t in trades if t.symbol == sym)
                               for sym in sorted({t.symbol for t in trades})})
        print("\nProfit by stock ($), best and worst:")
        ordered = by_symbol.sort_values()
        print(pd.concat([ordered.head(8), ordered.tail(8)]).round(2).to_string())


if __name__ == "__main__":
    main()
