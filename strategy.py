"""The trading rules, with no broker or network code. The backtest and the live
bot both call these functions so they cannot drift apart."""

from config import Settings


def entry_level(prev_low: float, s: Settings) -> float:
    """Price at or below which we buy today."""
    return prev_low * (1 - s.dip_pct)


def trend_ok(prev_close: float, trend_avg: float | None, s: Settings) -> bool:
    """Only buy dips in stocks that are still above their longer-term average."""
    if s.trend_days <= 0:
        return True
    return trend_avg is not None and prev_close > trend_avg


def stop_distance(entry_price: float, atr: float | None, s: Settings) -> float:
    """How far below the entry price the stop-loss sits, in dollars per share."""
    if s.atr_mult > 0 and atr:
        return s.atr_mult * atr
    return entry_price * s.stop_loss_pct


def stop_level(entry_price: float, peak_price: float, s: Settings, atr: float | None = None) -> float:
    """Price at or below which we sell. Starts as the stop-loss and ratchets up
    behind the peak once the position is far enough in profit."""
    level = entry_price - stop_distance(entry_price, atr, s)
    if s.atr_mult > 0 and atr:
        start, trail = entry_price + s.atr_mult * atr, s.atr_mult * atr
    else:
        start, trail = entry_price * (1 + s.trail_start_pct), peak_price * s.trail_pct
    if s.trail_pct > 0 and peak_price >= start:
        level = max(level, peak_price - trail)
    return level


def target_level(entry_price: float, s: Settings) -> float | None:
    """Fixed profit target, or None when it is turned off."""
    if s.take_profit_pct <= 0:
        return None
    return entry_price * (1 + s.take_profit_pct)


def position_dollars(s: Settings, entry_price: float | None = None, atr: float | None = None) -> float:
    """Dollars to put into one position: an even split of the budget, shrunk for
    jumpier stocks when risk_per_trade is set."""
    size = s.budget / s.max_positions
    if s.risk_per_trade > 0 and s.atr_mult > 0 and atr and entry_price:
        size = min(size, s.risk_per_trade * entry_price / stop_distance(entry_price, atr, s))
    return round(size, 2)
