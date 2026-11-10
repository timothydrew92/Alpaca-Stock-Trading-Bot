"""Settings for the swing bot. The backtest and the bot both read this file,
so a number changed here changes both."""

from dataclasses import dataclass

# Stocks the bot is allowed to trade.
WATCHLIST = ["IONQ", "RGTI", "LUNR", "RKLB", "SMR", "SMCI", "SPCE", "ASTS", "TTD", "NVDA"]


@dataclass(frozen=True)
class Settings:
    # --- Money ---
    budget: float = 100.0          # Most the bot will ever have invested at once.
    max_positions: int = 4         # No more than this many stocks held at once, at most $25 each.
    risk_per_trade: float = 2.0    # Each position is sized so a normal stop-out loses about $2.
    max_total_loss: float = 25.0   # Bot stops buying for good once its realized losses reach this.

    # --- When to buy ---
    dip_pct: float = 0.01          # Buy when price falls 1% below yesterday's low...
    trend_days: int = 50           # ...but only if yesterday's close was above its 50-day average.

    # --- When to sell ---
    atr_mult: float = 3.0          # Stop-loss sits 3 "average daily ranges" below the buy price, so
                                   # jumpy stocks get more room and calm ones less. Once the price is
                                   # up by that same distance, the stop trails that far behind the peak.
    take_profit_pct: float = 0.0   # Fixed profit target; 0 = off (the trailing stop takes profits).

    # Only used if atr_mult is set to 0 (fixed-percent stops instead of volatility-based):
    stop_loss_pct: float = 0.08
    trail_start_pct: float = 0.05
    trail_pct: float = 0.05

    # --- Rules that keep the account out of trouble ---
    settled_cash_only: bool = True  # Never spend sale proceeds until the next trading day (T+1).
    loss_cooldown_days: int = 31    # After selling a stock at a loss, don't rebuy it for 31 days
                                    # (avoids IRS wash sales and re-buying a falling stock).

    # --- Bot timing ---
    poll_seconds: int = 60


SETTINGS = Settings()
