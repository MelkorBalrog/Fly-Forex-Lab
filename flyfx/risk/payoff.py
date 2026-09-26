"""Day bias and a payoff stop that is the same on every pair.

The local regime can turn up for an hour inside a down day. Entries have to
agree with the slow EMA. The stop does not trail until the trade has earned
twice its initial risk, so a small winner is not cashed against a full loss.
"""

from __future__ import annotations

BIAS_BARS = 288
BIAS_ATR = 0.25
BE_R = 1.0
TRAIL_R = 2.0
TRAIL_KEEP = 0.50
MAX_COST_RATIO = 0.40


def tape_bias(close: float, ema_day: float | None, atr: float, bars_seen: int) -> str:
    """UP, DOWN, or FLAT versus the slow EMA. FLAT until that EMA has a day of bars."""
    if bars_seen < BIAS_BARS or ema_day is None:
        return "FLAT"
    span = max(float(atr), 1e-12)
    gap = (float(close) - float(ema_day)) / span
    if gap >= BIAS_ATR:
        return "UP"
    if gap <= -BIAS_ATR:
        return "DOWN"
    return "FLAT"


def bias_trade(bias: str, trend: str, fade: str, flow: str, impulse: float) -> tuple[str, str]:
    """Sponsor a with-day trade when the pullback setup itself is silent.

    Both flies opposing the day is a skip. A trend fly or the flow algo
    in the day's direction is enough. Impulse has to agree and not be spent.
    """
    imp = float(impulse or 0.0)
    if bias == "UP" and 0.35 <= imp < 0.92:
        if trend == "SELL" and fade == "SELL":
            return "HOLD", ""
        if trend == "BUY" or flow == "BUY":
            return "BUY", "bias-up"
    if bias == "DOWN" and -0.92 < imp <= -0.35:
        if trend == "BUY" and fade == "BUY":
            return "HOLD", ""
        if trend == "SELL" or flow == "SELL":
            return "SELL", "bias-down"
    return "HOLD", ""


def allows_side(side: str, bias: str) -> bool:
    """With a day bias, only trade that way. FLAT means no day filter yet."""
    if bias in ("", "FLAT", "flat", "none", "off"):
        return True
    if bias == "UP" and side == "BUY":
        return True
    if bias == "DOWN" and side == "SELL":
        return True
    return False


def cost_ratio(spread: float, slip: float, stop_dist: float, commission_px: float) -> float:
    stop = max(float(stop_dist), 1e-12)
    cost = max(float(spread), 0.0) + 2.0 * max(float(slip), 0.0) + max(float(commission_px), 0.0)
    return cost / stop


def usd_to_price(to_usd, usd: float, mid: float) -> float:
    """Invert a one-lot price-to-USD converter. ``to_usd(dx, 1, mid)`` is the probe."""
    probe = max(abs(float(mid)) * 1e-4, 1e-8)
    per = abs(float(to_usd(probe, 1.0, mid)))
    if per <= 1e-12:
        return 0.0
    return abs(float(usd)) * probe / per


def payoff_stop(
    side: str,
    anchor: float,
    stop: float,
    unreal: float,
    risk_px: float,
    extreme: float,
) -> float:
    """Lock entry at 1R. Trail half an R behind the extreme only after 2R."""
    r = max(float(risk_px), 1e-12)
    held = float(stop)
    if float(unreal) >= TRAIL_R * r:
        if side == "BUY":
            return max(held, float(extreme) - TRAIL_KEEP * r)
        if side == "SELL":
            return min(held, float(extreme) + TRAIL_KEEP * r)
    if float(unreal) >= BE_R * r:
        if side == "BUY":
            return max(held, float(anchor))
        if side == "SELL":
            return min(held, float(anchor))
    return held
