# Alpaca Swing Bot

A small Python bot for [Alpaca](https://alpaca.markets) that buys dips in stocks that are trending up, cuts losing positions quickly, and lets winning ones run. It is built to be tried with a small amount of money (default budget: $100) and has a paper-trading mode and a backtest.

This is a personal experiment, not financial advice. The backtest does not show a reliable edge, and you can lose money.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Create two files in this folder (both are ignored by git) and paste in your keys from the Alpaca dashboard:

`.env.paper`
```
APCA_API_KEY_ID=your-paper-key
APCA_API_SECRET_KEY=your-paper-secret
```

`.env.live`
```
APCA_API_KEY_ID=your-live-key
APCA_API_SECRET_KEY=your-live-secret
```

## Commands

| Command | What it does |
|---|---|
| `.venv/bin/python backtest.py` | Replays the rules over the last two years |
| `.venv/bin/python backtest.py --compare` | Same, next to other rule variants and buy-and-hold |
| `.venv/bin/python screen.py` | Scans all fractionable stocks and lists watchlist candidates (prints only) |
| `.venv/bin/python bot.py --status` | Shows the paper account and what the bot holds |
| `.venv/bin/python bot.py --dry-run --once` | One pass that prints decisions and places no orders |
| `.venv/bin/python bot.py` | Runs on the paper account until you press Ctrl+C |
| `.venv/bin/python bot.py --live` | Runs with real money; asks you to type `LIVE` first |
| `.venv/bin/python bot.py --close-all` | Sells everything the bot holds |

Add `--live` to `--status` or `--close-all` to use the real account.

## The rules

All numbers are in [config.py](config.py); the logic is in [strategy.py](strategy.py) and is shared by the bot and the backtest.

**Buy** a stock from the watchlist when all of these are true:
- Its price is 1% or more below yesterday's low.
- Yesterday's close was above its 50-day average (it is in an uptrend).
- The bot holds fewer than 4 stocks and has settled cash available.

**Position size** is whatever amount would lose about $2 if the stop is hit, capped at $25. Jumpier stocks get smaller positions.

**Sell** when the price falls to the stop. The stop starts three "average daily ranges" below the buy price. Once the price has risen by that same distance, the stop follows that far behind the highest price reached, so gains are kept without capping them.

## Safety rules

- **Budget cap:** never more than $100 invested at once.
- **Loss limit:** after $25 of realized losses the bot stops buying permanently.
- **Settled cash only:** money from a sale is not spent until the next trading day, which avoids good-faith violations in small accounts. If the bot cannot confirm settled cash, it does not buy.
- **No rebuying after a loss:** a stock sold at a loss is not bought again for 31 days, which avoids IRS wash sales.
- **One order at a time per stock:** the bot skips any stock that already has an open order.
- **Hands off everything else:** the bot only sells shares it bought itself, recorded in `state/`.
- **Fractional market orders, regular hours only.**

## Limits to know about

- The stop only works while the bot is running. If the bot is off, nothing sells.
- A stock can open far below the stop (after bad news overnight). The sale then happens at the open, and the loss is larger than $2.
- Prices come from Alpaca's free IEX feed, which can differ slightly from the full market.
- The backtest uses daily bars and a short, unusual period for these stocks. Treat its results as a sanity check, not a forecast.
