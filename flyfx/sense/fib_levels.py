"""Fibonacci retracements of the latest trend impulse.

The 55-bar high/low range stays on the indicator for the structure
channel. This is the ordered swing: low then high is an uptrend, high
then low is a downtrend. Trade levels are the pullbacks of that leg
(38.2, 50, 61.8, 78.6), not the 23.6 which is already a continuation.
"""

from __future__ import annotations

FIB_RATIOS = (0.382, 0.500, 0.618, 0.786)


def impulse_swing(
    highs: list[float] | tuple[float, ...],
    lows: list[float] | tuple[float, ...],
) -> tuple[float, float, bool] | None:
    """Return ``(high, low, up)`` for the impulse inside this window."""
    n = len(highs)
    if n < 8 or len(lows) != n:
        return None
    hi_i = max(range(n), key=lambda i: float(highs[i]))
    lo_i = min(range(n), key=lambda i: float(lows[i]))
    if hi_i == lo_i:
        return None
    if hi_i > lo_i:
        lo = min(float(x) for x in lows[: hi_i + 1])
        hi = float(highs[hi_i])
        if hi <= lo:
            return None
        return hi, lo, True
    hi = max(float(x) for x in highs[: lo_i + 1])
    lo = float(lows[lo_i])
    if hi <= lo:
        return None
    return hi, lo, False


def retracement_prices(hi: float, lo: float, up: bool) -> dict[str, float]:
    """Price of each retracement, measured back from the impulse extreme."""
    span = float(hi) - float(lo)
    out: dict[str, float] = {}
    for ratio in FIB_RATIOS:
        px = (float(hi) - ratio * span) if up else (float(lo) + ratio * span)
        out[f"{ratio:.3f}"] = float(px)
    return out


def held_level(
    high: float,
    low: float,
    close: float,
    atr: float,
    hi: float,
    lo: float,
    up: bool,
) -> str:
    """Level name when this bar reached it and the close held it.

    A close through the level is a break, not a trend entry. A swing
    smaller than 1.5 ATR is noise and returns no level.
    """
    span = float(hi) - float(lo)
    atr_n = float(atr)
    if span <= 0.0 or atr_n <= 0.0 or span < 1.5 * atr_n:
        return ""
    tol = max(0.25 * atr_n, 0.06 * span)
    best_name = ""
    best_dist = 1e18
    for name, px in retracement_prices(hi, lo, up).items():
        if float(low) > px + tol or float(high) < px - tol:
            continue
        if up and float(close) < px - 0.20 * tol:
            continue
        if (not up) and float(close) > px + 0.20 * tol:
            continue
        dist = abs(float(close) - px)
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name
