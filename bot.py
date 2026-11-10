"""Swing bot: buys dips in up-trending stocks, cuts losers, lets winners run.

    .venv/bin/python bot.py --status          # show the account and what the bot holds
    .venv/bin/python bot.py --dry-run --once  # one pass, prints what it would do, places no orders
    .venv/bin/python bot.py                   # run on the PAPER account until stopped (Ctrl+C)
    .venv/bin/python bot.py --live            # run with REAL money (asks you to type LIVE first)
    .venv/bin/python bot.py --close-all       # sell everything the bot holds, then exit

Rules live in strategy.py, numbers in config.py. Once a week the bot rebuilds its own
watchlist with screen.py and adds any stocks you list in MY_PICKS. It only ever sells
shares it bought itself (tracked in state/<mode>.json); anything else is left alone.
"""

import argparse
import json
import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.trading.enums import OrderSide, OrderStatus, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

import screen
import strategy
from broker import ROOT, data_client, trading_client
from config import EXCLUDE, MY_PICKS, SCREEN, SETTINGS
from marketdata import average_range, fetch_daily_bars

NEW_YORK = ZoneInfo("America/New_York")
DONE = {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED}
log = logging.getLogger("bot")


def today_ny() -> date:
    return datetime.now(NEW_YORK).date()


class Bot:
    def __init__(self, mode: str, dry_run: bool):
        self.mode, self.dry_run, self.s = mode, dry_run, SETTINGS
        self.trading = trading_client(mode)
        self.data = data_client(mode)
        self.state_file = ROOT / "state" / f"{mode}.json"
        self.state = {"positions": {}, "pending": {}, "realized": 0.0, "blocked_until": {}, "last_sold": {}}
        if self.state_file.exists():
            self.state |= json.loads(self.state_file.read_text())
        self.watchlist_file = ROOT / "state" / "watchlist.json"
        self.levels: dict[str, dict] = {}
        self.levels_day: date | None = None
        self.tradable: set[str] = set()

    # ---------- state ----------

    def save(self):
        if self.dry_run:
            return
        self.state_file.parent.mkdir(exist_ok=True)
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2))
        tmp.replace(self.state_file)

    # ---------- watchlist ----------

    def screened(self) -> dict:
        if self.watchlist_file.exists():
            return json.loads(self.watchlist_file.read_text())
        return {"date": None, "symbols": []}

    def watchlist(self) -> list[str]:
        """Your picks first, then the screener's list."""
        picks = [s.upper() for s in MY_PICKS]
        skip = {s.upper() for s in EXCLUDE}
        return [s for s in picks + [s for s in self.screened()["symbols"] if s not in picks] if s not in skip]

    def watchlist_age_days(self) -> float:
        made = self.screened()["date"]
        return float("inf") if made is None else (today_ny() - date.fromisoformat(made)).days

    def refresh_watchlist(self):
        """Re-run the screen. Takes a few minutes, so it normally runs while the market is closed."""
        log.info("Rebuilding the watchlist (scanning all stocks, a few minutes)...")
        old = set(self.screened()["symbols"])
        symbols = screen.auto_watchlist()
        self.watchlist_file.parent.mkdir(exist_ok=True)
        self.watchlist_file.write_text(json.dumps({"date": today_ny().isoformat(), "symbols": symbols}, indent=2))
        self.levels_day = None
        added, dropped = [s for s in symbols if s not in old], sorted(old - set(symbols))
        log.info("Watchlist: %s", ", ".join(self.watchlist()))
        if old:
            log.info("New this week: %s. Dropped: %s.", ", ".join(added) or "none", ", ".join(dropped) or "none")

    # ---------- market data ----------

    def refresh_levels(self):
        """Once per day: yesterday's low, the trend average and the typical daily range per stock."""
        today = today_ny()
        if self.levels_day == today:
            return
        bars = fetch_daily_bars(self.watchlist(), 150)
        levels = {}
        for symbol, df in bars.items():
            atr_series = average_range(df)
            df = df[df.index.date < today]  # completed days only
            if len(df) < max(self.s.trend_days, 15):
                log.warning("%s: not enough price history, skipping", symbol)
                continue
            trend_avg = df["close"].tail(self.s.trend_days).mean() if self.s.trend_days > 0 else None
            levels[symbol] = {
                "prev_low": float(df["low"].iloc[-1]),
                "prev_close": float(df["close"].iloc[-1]),
                "trend_avg": None if trend_avg is None else float(trend_avg),
                "atr": float(atr_series.loc[df.index[-1]]),
            }
        self.levels, self.levels_day = levels, today
        self.tradable = set()
        for symbol in levels:
            asset = self.trading.get_asset(symbol)
            if asset.tradable and asset.fractionable:
                self.tradable.add(symbol)
            else:
                log.warning("%s: not tradable in fractions at Alpaca, skipping", symbol)
        log.info("Levels refreshed for %s: %d stocks usable", today, len(self.tradable))

    def prices(self, symbols: list[str]) -> dict[str, float]:
        """Latest trade per stock. The free live feed (IEX) sees only part of the market, so
        when its last trade is old we fall back to the full-market feed, which runs 15 minutes
        behind. A stock with no usable price is left out of this pass."""
        if not symbols:
            return {}
        now, fresh = datetime.now(timezone.utc), {}
        for feed, max_age in ((DataFeed.IEX, timedelta(minutes=3)), (DataFeed.DELAYED_SIP, timedelta(minutes=25))):
            missing = [s for s in symbols if s not in fresh]
            if not missing:
                break
            try:
                trades = self.data.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=missing, feed=feed))
            except Exception as error:
                log.warning("Price feed %s failed: %s", feed.value, error)
                continue
            for symbol, trade in trades.items():
                if now - trade.timestamp <= max_age:
                    fresh[symbol] = float(trade.price)
        for symbol in symbols:
            if symbol not in fresh:
                log.warning("%s: no recent price, skipping this pass", symbol)
        return fresh

    # ---------- account rules ----------

    def spendable_cash(self) -> float:
        """Cash the bot may spend right now. Sale proceeds from today are excluded because
        they have not settled (T+1); spending them is what causes good-faith violations.
        Any error here is raised, which blocks buying for this pass."""
        account = self.trading.get_account()
        if account.trading_blocked or account.account_blocked:
            raise RuntimeError("Alpaca reports this account is blocked from trading")
        funds = float(account.cash)
        if self.mode == "paper" and funds < 0:
            # A paper account already borrowing on margin has no meaningful cash figure.
            funds = float(account.buying_power)
        if self.s.settled_cash_only:
            fills = self.trading.get("/account/activities/FILL", {"date": today_ny().isoformat()})
            unsettled = sum(float(f["qty"]) * float(f["price"]) for f in fills if f["side"].startswith("sell"))
            funds -= unsettled
        invested = sum(p["qty"] * p["entry"] for p in self.state["positions"].values())
        return min(funds, self.s.budget - invested)

    def symbols_with_open_orders(self) -> set[str]:
        orders = self.trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))
        return {o.symbol for o in orders}

    # ---------- orders ----------

    def submit(self, symbol: str, side: OrderSide, *, notional: float | None = None,
               qty: float | None = None, reason: str = ""):
        what = f"${notional:.2f}" if notional else f"{qty} shares"
        if self.dry_run:
            log.info("DRY RUN: would %s %s of %s (%s)", side.value, what, symbol, reason)
            return
        order = self.trading.submit_order(MarketOrderRequest(
            symbol=symbol, side=side, notional=notional, qty=qty, time_in_force=TimeInForce.DAY,
            client_order_id=f"swingbot-{symbol}-{int(time.time())}"))
        self.state["pending"][str(order.id)] = {"symbol": symbol, "side": side.value, "reason": reason}
        self.save()
        log.info("%s %s of %s submitted (%s)", side.value.upper(), what, symbol, reason)

    def settle_pending(self, wait_seconds: int = 0):
        """Record fills for orders the bot placed. Called every pass, so a slow fill is never lost."""
        deadline = time.time() + wait_seconds
        while self.state["pending"]:
            for order_id, info in list(self.state["pending"].items()):
                order = self.trading.get_order_by_id(order_id)
                if order.status not in DONE:
                    continue
                filled = float(order.filled_qty or 0)
                if filled > 0:
                    self.record_fill(info, filled, float(order.filled_avg_price))
                else:
                    log.warning("%s order for %s ended as %s with nothing filled",
                                info["side"], info["symbol"], order.status.value)
                del self.state["pending"][order_id]
                self.save()
            if time.time() >= deadline:
                break
            time.sleep(2)

    def record_fill(self, info: dict, qty: float, price: float):
        symbol, positions = info["symbol"], self.state["positions"]
        if info["side"] == "buy":
            positions[symbol] = {"qty": qty, "entry": price, "peak": price,
                                 "atr": self.levels.get(symbol, {}).get("atr"),
                                 "bought": today_ny().isoformat()}
            log.info("BOUGHT %.6f %s at $%.2f ($%.2f)", qty, symbol, price, qty * price)
            return
        pos = positions.get(symbol)
        if pos is None:
            return
        profit = (price - pos["entry"]) * qty
        self.state["realized"] += profit
        self.state["last_sold"][symbol] = today_ny().isoformat()
        if profit < 0 and self.s.loss_cooldown_days > 0:
            self.state["blocked_until"][symbol] = (
                today_ny() + timedelta(days=self.s.loss_cooldown_days)).isoformat()
        pos["qty"] -= qty
        if pos["qty"] <= 1e-9:
            del positions[symbol]
        log.info("SOLD %.6f %s at $%.2f, profit $%+.2f (%s). Bot total so far: $%+.2f",
                 qty, symbol, price, profit, info["reason"], self.state["realized"])

    def reconcile(self):
        """If shares the bot thinks it holds are gone from the account, trust the account."""
        held = {p.symbol: float(p.qty) for p in self.trading.get_all_positions()}
        for symbol, pos in list(self.state["positions"].items()):
            actual = held.get(symbol, 0.0)
            if actual < pos["qty"] - 1e-6:
                log.warning("%s: bot recorded %.6f shares but the account has %.6f; adjusting",
                            symbol, pos["qty"], actual)
                if actual <= 1e-9:
                    del self.state["positions"][symbol]
                else:
                    pos["qty"] = actual
                self.save()

    # ---------- one pass ----------

    def run_once(self):
        self.settle_pending()
        self.refresh_levels()
        self.reconcile()
        busy = self.symbols_with_open_orders() | {p["symbol"] for p in self.state["pending"].values()}
        positions = self.state["positions"]
        price = self.prices(sorted(self.tradable | set(positions)))
        today = today_ny().isoformat()

        # Exits first.
        for symbol, pos in list(positions.items()):
            if symbol in busy or symbol not in price:
                continue
            now = price[symbol]
            stop = strategy.stop_level(pos["entry"], pos["peak"], self.s, pos.get("atr"))
            target = strategy.target_level(pos["entry"], self.s)
            if now <= stop:
                reason = "stop-loss" if stop < pos["entry"] else "trailing stop"
                self.submit(symbol, OrderSide.SELL, qty=pos["qty"], reason=reason)
            elif target is not None and now >= target:
                self.submit(symbol, OrderSide.SELL, qty=pos["qty"], reason="profit target")
            elif now > pos["peak"]:
                pos["peak"] = now
                self.save()
        self.settle_pending(wait_seconds=20)

        # Entries.
        if self.state["realized"] <= -self.s.max_total_loss:
            log.warning("Loss limit reached (bot total $%+.2f). Not buying anything more.", self.state["realized"])
            return
        try:
            cash = self.spendable_cash()
        except Exception as error:
            log.error("Could not confirm settled cash, so not buying this pass: %s", error)
            return
        for symbol in self.watchlist():
            if len(positions) + sum(1 for p in self.state["pending"].values() if p["side"] == "buy") >= self.s.max_positions:
                break
            if symbol not in self.tradable or symbol in positions or symbol in busy or symbol not in price:
                continue
            if self.state["last_sold"].get(symbol) == today:
                continue
            if self.state["blocked_until"].get(symbol, "") > today:
                continue
            lv, now = self.levels[symbol], price[symbol]
            if not strategy.trend_ok(lv["prev_close"], lv["trend_avg"], self.s):
                continue
            if now > strategy.entry_level(lv["prev_low"], self.s):
                continue
            size = strategy.position_dollars(self.s, now, lv["atr"])
            if size < 1 or cash < size:
                log.info("%s is at a buy level ($%.2f) but only $%.2f of settled cash is free", symbol, now, max(cash, 0))
                continue
            self.submit(symbol, OrderSide.BUY, notional=size, reason=f"dip to ${now:.2f}")
            cash -= size
        self.settle_pending(wait_seconds=20)

    # ---------- commands ----------

    def status(self):
        account = self.trading.get_account()
        print(f"\n{self.mode.upper()} account: cash ${float(account.cash):,.2f}, equity ${float(account.equity):,.2f}")
        print(f"Bot realized profit so far: ${self.state['realized']:+.2f} "
              f"(stops buying at ${-self.s.max_total_loss:.2f})")
        positions = self.state["positions"]
        if not positions:
            print("Bot holds nothing.")
        else:
            price = self.prices(list(positions))
            for symbol, pos in positions.items():
                now = price.get(symbol)
                stop = strategy.stop_level(pos["entry"], pos["peak"], self.s, pos.get("atr"))
                gain = "" if now is None else f", now ${now:.2f} ({(now / pos['entry'] - 1):+.1%})"
                print(f"  {symbol}: {pos['qty']:.4f} shares bought {pos['bought']} at ${pos['entry']:.2f}{gain}, "
                      f"sells at or below ${stop:.2f}")
        made = self.screened()["date"]
        print(f"Watching ({'screened ' + made if made else 'no screen yet'}): {', '.join(self.watchlist()) or 'nothing'}")
        blocked = {s: d for s, d in self.state["blocked_until"].items() if d > today_ny().isoformat()}
        if blocked:
            print("Not rebuying after a loss until:", ", ".join(f"{s} {d}" for s, d in blocked.items()))

    def close_all(self):
        if not self.trading.get_clock().is_open:
            raise SystemExit("Market is closed; run this again during market hours.")
        self.settle_pending()
        self.reconcile()
        for symbol, pos in list(self.state["positions"].items()):
            self.submit(symbol, OrderSide.SELL, qty=pos["qty"], reason="close-all")
        self.settle_pending(wait_seconds=30)
        self.status()

    def run_forever(self):
        log.info("Bot started in %s mode%s. Budget $%.0f, loss limit $%.0f.", self.mode.upper(),
                 " (dry run)" if self.dry_run else "", self.s.budget, self.s.max_total_loss)
        while True:
            try:
                clock = self.trading.get_clock()
                age = self.watchlist_age_days()
                # Rebuild weekly while the market is closed; during market hours only if badly overdue.
                if age >= SCREEN.refresh_days and (not clock.is_open or age >= 2 * SCREEN.refresh_days):
                    self.refresh_watchlist()
                    continue
                if clock.is_open:
                    self.run_once()
                    time.sleep(self.s.poll_seconds)
                else:
                    wait = (clock.next_open - clock.timestamp).total_seconds()
                    log.info("Market closed. Next open %s.", clock.next_open.astimezone(NEW_YORK).strftime("%a %b %d %H:%M ET"))
                    time.sleep(max(60, min(wait, 1800)))
            except KeyboardInterrupt:
                log.info("Stopped by user. Positions are NOT sold and NOT protected while the bot is off.")
                return
            except Exception:
                log.exception("Pass failed; trying again in a minute")
                time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="use the real-money account (default is paper)")
    parser.add_argument("--dry-run", action="store_true", help="decide but place no orders")
    parser.add_argument("--once", action="store_true", help="run a single pass and exit")
    parser.add_argument("--status", action="store_true", help="show account and bot positions, then exit")
    parser.add_argument("--close-all", action="store_true", help="sell everything the bot holds, then exit")
    args = parser.parse_args()
    mode = "live" if args.live else "paper"

    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(ROOT / "logs" / f"{mode}.log")])
    logging.getLogger("httpx").setLevel(logging.WARNING)

    bot = Bot(mode, args.dry_run)
    if args.status:
        return bot.status()
    if mode == "live" and not args.dry_run:
        answer = input(f"This will trade REAL money (up to ${SETTINGS.budget:.0f}). Type LIVE to continue: ")
        if answer.strip() != "LIVE":
            raise SystemExit("Cancelled.")
    if args.close_all:
        return bot.close_all()
    if args.once:
        if bot.watchlist_age_days() >= SCREEN.refresh_days:
            bot.refresh_watchlist()
        if not bot.trading.get_clock().is_open:
            raise SystemExit("Market is closed; nothing to trade right now.")
        return bot.run_once()
    bot.run_forever()


if __name__ == "__main__":
    main()
