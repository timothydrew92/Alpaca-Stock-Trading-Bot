"""Settings for the swing bot. The backtest and the bot both read this file,
so a number changed here changes both."""

from dataclasses import dataclass

# Stocks you want the bot to watch no matter what the screener says, e.g. ["RKLB", "NVDA"].
# They still have to pass the buy rules (uptrend + dip) before the bot buys them.
MY_PICKS: list[str] = []

# Stocks the bot must never buy, even if the screener finds them.
EXCLUDE: list[str] = []


@dataclass(frozen=True)
class ScreenSettings:
    """How the bot finds its own stocks. See screen.py."""
    size: int = 15                         # How many screened stocks to watch at once.
    refresh_days: int = 7                  # Re-run the screen this often.
    min_dollar_volume: float = 50_000_000  # Average dollars traded per day.
    min_price: float = 5.0                 # Skip stocks cheaper than this.
    min_daily_move: float = 0.025          # Typical daily move must be at least 2.5%...
    max_daily_move: float = 0.08           # ...and no more than 8%.
    avoid_drug_developers: bool = True     # Skip biotech/pharma names (overnight trial-result risk).


SCREEN = ScreenSettings()


@dataclass(frozen=True)
class Settings:
    # --- Money ---
    budget: float = 100.0          # Starting money the bot may have invested at once.
    reinvest_profits: bool = True  # Add the bot's own profits to the budget (and take its losses off),
                                   # so position sizes grow and shrink with results. False = always $100.
    max_positions: int = 4         # No more than this many stocks held at once, a quarter of the budget each.
    risk_per_trade: float = 2.0    # Each position is sized so a normal stop-out loses about $2
                                   # per $100 of budget (2%).
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
