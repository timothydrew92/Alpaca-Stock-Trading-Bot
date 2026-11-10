"""Finds candidate stocks for the watchlist using free daily prices.

    .venv/bin/python screen.py                 # top 20 candidates
    .venv/bin/python screen.py --max-price 20  # only stocks under $20
    .venv/bin/python screen.py --top 40

It scans every stock Alpaca can trade in fractions and keeps those that are:
  * liquid      - at least $20M traded per day on average, so orders fill at fair prices
  * not pennies - price of $2 or more (below that, spreads and manipulation dominate)
  * in an uptrend - above both their 50-day and 200-day averages
  * swingy enough - typical daily move between 2.5% and 10%
and ranks them by 6-month price gain (stocks that have been rising tend to keep
outperforming for a while; it is a tendency, not a promise).

It only prints a list. To trade a stock, add it to WATCHLIST in config.py yourself.
"""

import argparse

import pandas as pd
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import GetAssetsRequest

from backtest import average_range, fetch_daily_bars
from broker import trading_client
from config import WATCHLIST

FUND_WORDS = ("ETF", "ETN", "FUND", "TRUST", "PROSHARES", "DIREXION", "ISHARES", "SPDR", "VANGUARD",
              "INVESCO", "2X", "3X", "LEVERAGED", "INVERSE", "BULL ", "BEAR ", "SHARES ", "INDEX",
              "WARRANT", "UNIT", "RIGHT", "PREFERRED", "DEPOSITARY SHARES", "NOTES DUE")
MIN_DOLLAR_VOLUME = 20_000_000
MIN_PRICE = 2.0


def candidate_symbols() -> dict[str, str]:
    """Symbol -> company name for ordinary stocks Alpaca can trade in fractions."""
    assets = trading_client("paper").get_all_assets(
        GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY))
    names = {}
    for a in assets:
        name = (a.name or "").upper()
        if a.tradable and a.fractionable and a.symbol.isalpha() and not any(w in name for w in FUND_WORDS):
            names[a.symbol] = a.name
    return names


def measure(df: pd.DataFrame) -> dict | None:
    if len(df) < 200:
        return None
    close = df["close"]
    price = float(close.iloc[-1])
    dollar_volume = float((close * df["volume"]).tail(20).mean())
    avg50, avg200 = float(close.tail(50).mean()), float(close.tail(200).mean())
    daily_move = float(average_range(df).iloc[-1]) / price
    return {
        "price": round(price, 2),
        "6mo gain": float(price / close.iloc[-126] - 1),
        "1mo gain": float(price / close.iloc[-21] - 1),
        "daily move": daily_move,
        "$M traded/day": round(dollar_volume / 1e6),
        "above 50d": price > avg50,
        "above 200d": price > avg200 and avg50 > avg200,
        "from 1y high": float(price / close.tail(252).max() - 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--max-price", type=float, default=None, help="only show stocks at or under this price")
    args = parser.parse_args()

    names = candidate_symbols()
    symbols = sorted(names)
    print(f"Scanning {len(symbols)} stocks (takes a minute or two)...")
    rows = {}
    for i in range(0, len(symbols), 200):
        bars = fetch_daily_bars(symbols[i:i + 200], 420, columns=("open", "high", "low", "close", "volume"))
        for symbol, df in bars.items():
            stats = measure(df)
            if stats:
                rows[symbol] = stats
    table = pd.DataFrame.from_dict(rows, orient="index")

    passing = table[(table["$M traded/day"] >= MIN_DOLLAR_VOLUME / 1e6) & (table["price"] >= MIN_PRICE)
                    & table["above 50d"] & table["above 200d"] & table["daily move"].between(0.025, 0.10)]
    if args.max_price:
        passing = passing[passing["price"] <= args.max_price]
    passing = passing.sort_values("6mo gain", ascending=False).head(args.top)

    def show(frame: pd.DataFrame) -> str:
        out = frame.drop(columns=["above 50d", "above 200d"]).copy()
        for col in ("6mo gain", "1mo gain", "daily move", "from 1y high"):
            out[col] = out[col].map(lambda v: f"{v:+.0%}" if col != "daily move" else f"{v:.1%}")
        out.insert(0, "company", [names.get(s, "")[:28] for s in out.index])
        return out.to_string()

    print(f"\n{len(rows)} stocks had enough history. Top {len(passing)} that pass every filter, by 6-month gain:\n")
    print(show(passing))

    current = table.reindex(WATCHLIST).dropna(how="all")
    if not current.empty:
        current = current.copy()
        current["uptrend"] = (current["above 50d"] & current["above 200d"]).map({True: "yes", False: "no"})
        print("\nYour current watchlist for comparison:\n")
        print(show(current.drop(columns=["uptrend"])))
        print("\nIn an uptrend now:", ", ".join(s for s in current.index if current.loc[s, "uptrend"] == "yes") or "none")


if __name__ == "__main__":
    main()
