"""Finds stocks worth watching, using free daily prices. The bot runs this itself
once a week; you can also run it by hand to see the current list.

    .venv/bin/python screen.py            # show the list the bot would use today
    .venv/bin/python screen.py --top 40   # show more of the ranking

Every stock Alpaca can trade in fractions is scanned. A stock qualifies when it is:
  * liquid        - enough dollars traded per day that small orders fill at fair prices
  * not a penny   - below a few dollars, spreads and manipulation dominate
  * in an uptrend - above its 50-day average, which is above its 200-day average
  * swingy, not wild - a typical daily move inside a set band
Qualifiers are ranked by 6-month price gain. Stocks that have been rising tend to
keep outperforming for a while; that is a tendency, not a promise.

Thresholds are in config.py (SCREEN). Nothing here places orders.
"""

import argparse

import pandas as pd
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import GetAssetsRequest

from broker import trading_client
from config import MY_PICKS, SCREEN, ScreenSettings
from marketdata import fetch_many

NOT_A_COMPANY = ("ETF", "ETN", "FUND", "TRUST", "PROSHARES", "DIREXION", "ISHARES", "SPDR", "VANGUARD",
                 "INVESCO", "2X", "3X", "LEVERAGED", "INVERSE", "BULL ", "BEAR ", "SHARES ", "INDEX",
                 "WARRANT", "UNIT", "RIGHT", "PREFERRED", "DEPOSITARY SHARES", "NOTES DUE")
# Drug developers can halve overnight on one trial result, which no stop-loss can dodge.
DRUG_DEVELOPER = ("THERAPEUTIC", "PHARMA", "BIO", "ONCOLOGY", "GENOMIC", "GENETIC", "MEDICINES",
                  "SCIENCES", "MODERNA", "VACCINE", "RX", "LABORATORIES")
HISTORY_DAYS = 420


def universe() -> dict[str, str]:
    """Symbol -> company name for ordinary stocks Alpaca can trade in fractions."""
    assets = trading_client("paper").get_all_assets(
        GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY))
    names = {}
    for a in assets:
        name = (a.name or "").upper()
        if a.tradable and a.fractionable and a.symbol.isalpha() and not any(w in name for w in NOT_A_COMPANY):
            names[a.symbol] = a.name
    return names


def measures(bars: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """One table per measure, with a row per day and a column per stock, so the
    list can be worked out as of any past date as well as today."""
    def wide(column):
        return pd.concat({s: df[column] for s, df in bars.items()}, axis=1).sort_index()
    close, high, low, volume = wide("close"), wide("high"), wide("low"), wide("volume")
    prev_close = close.shift(1)
    true_range = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()]).groupby(level=0).max()
    avg50, avg200 = close.rolling(50).mean(), close.rolling(200).mean()
    return {
        "price": close,
        "6mo gain": close / close.shift(126) - 1,
        "1mo gain": close / close.shift(21) - 1,
        "daily move": true_range.rolling(14).mean() / close,
        "$M traded/day": (close * volume).rolling(20).mean() / 1e6,
        "uptrend": (close > avg50) & (avg50 > avg200),
        "from 1y high": close / close.rolling(252, min_periods=200).max() - 1,
    }


def ranking(m: dict[str, pd.DataFrame], names: dict[str, str], cfg: ScreenSettings = SCREEN,
            before: pd.Timestamp | None = None) -> pd.DataFrame:
    """Qualifying stocks, best first, using only days before `before` (default: all days)."""
    days = m["price"].index
    if before is not None:
        days = days[days < before]
    if len(days) == 0:
        return pd.DataFrame()
    table = pd.DataFrame({name: frame.loc[days[-1]] for name, frame in m.items()}).dropna()
    keep = ((table["$M traded/day"] >= cfg.min_dollar_volume / 1e6) & (table["price"] >= cfg.min_price)
            & table["uptrend"].astype(bool) & table["daily move"].between(cfg.min_daily_move, cfg.max_daily_move))
    if cfg.avoid_drug_developers:
        keep &= ~table.index.map(lambda s: any(w in (names.get(s) or "").upper() for w in DRUG_DEVELOPER))
    return table[keep].sort_values("6mo gain", ascending=False)


def auto_watchlist() -> list[str]:
    """Today's automatic list. Takes a few minutes the first time each day."""
    names = universe()
    bars = fetch_many(sorted(names), HISTORY_DAYS)
    return list(ranking(measures(bars), names).head(SCREEN.size).index)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top", type=int, default=SCREEN.size)
    args = parser.parse_args()

    names = universe()
    print(f"Scanning {len(names)} stocks (a few minutes the first time each day)...")
    bars = fetch_many(sorted(names), HISTORY_DAYS, progress=True)
    ranked = ranking(measures(bars), names)
    out = ranked.head(args.top).drop(columns=["uptrend"]).copy()
    for col in ("6mo gain", "1mo gain", "from 1y high"):
        out[col] = out[col].map("{:+.0%}".format)
    out["daily move"] = out["daily move"].map("{:.1%}".format)
    out["price"] = out["price"].round(2)
    out["$M traded/day"] = out["$M traded/day"].round().astype(int)
    out.insert(0, "company", [(names.get(s) or "")[:28] for s in out.index])
    print(f"\n{len(ranked)} stocks qualify today. The bot watches the top {SCREEN.size}"
          + (f" plus your picks ({', '.join(MY_PICKS)})" if MY_PICKS else "") + ".\n")
    print(out.to_string())


if __name__ == "__main__":
    main()
