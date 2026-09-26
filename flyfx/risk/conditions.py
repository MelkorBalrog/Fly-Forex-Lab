"""Live scalp risk. Same function for every pair.

Size follows the bar in front of the book, not a coefficient stored on
``brains/{PAIR}.json``. Five conditions, each in price or clock units:

- cost: round-trip spread and slip versus the stop distance
- vol: ATR divided by its slow ATR (dead tape, usable tape, spike)
- liquidity: UTC session, London/New York overlap is full size
- chase: how stretched the impulse already is
- heat: consecutive losses, size comes down

The product never exceeds 1, so a hot tape cannot raise the stated risk percent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


def _clip(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _lerp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    if x <= x0:
        return y0
    if x >= x1:
        return y1
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


# M5: 6 bars is half an hour, 12 bars is an hour.
SCRATCH_BARS = 6
SCRATCH_HARD_BARS = 12


def round_trip_px(spread: float, slip: float) -> float:
    return max(float(spread), 0.0) + 2.0 * max(float(slip), 0.0)


def cost_factor(spread: float, slip: float, stop_dist: float) -> float:
    """Full size when the stop is many times the spread. Cut as the spread eats it."""
    stop = max(float(stop_dist), 1e-12)
    ratio = round_trip_px(spread, slip) / stop
    return _lerp(ratio, 0.08, 0.40, 1.0, 0.35)


def vol_factor_regime(atr_ratio: float, uncert: float = 0.0) -> float:
    """Dead tape and volatility spikes both get less size. The middle is full."""
    ratio = float(atr_ratio or 1.0)
    if ratio < 0.72:
        vol = 0.55
    elif ratio < 1.25:
        vol = 1.0
    elif ratio < 1.60:
        vol = 0.78
    else:
        vol = 0.48
    if float(uncert or 0.0) >= 0.75:
        vol *= 0.80
    return vol


def liquidity_factor(stamp: int) -> float:
    """London/New York overlap is the liquid hour. Shoulders and Tokyo are smaller.

    A missing clock stays at 1 so a caller without a bar time is not punished.
    """
    if not stamp:
        return 1.0
    g = time.gmtime(int(stamp))
    if g.tm_wday >= 5:
        return 0.50
    hour = g.tm_hour
    if 12 <= hour < 16:
        return 1.0
    if 8 <= hour < 12:
        return 0.90
    if 7 <= hour < 8:
        return 0.75
    if 0 <= hour < 4:
        return 0.65
    return 0.70


def chase_factor(impulse: float) -> float:
    """Impulse is tanh-scaled. Near saturation the move is already gone."""
    stretched = abs(float(impulse or 0.0))
    return _lerp(stretched, 0.78, 0.95, 1.0, 0.62)


def heat_factor(consec_losses: int) -> float:
    """Cut size after losses. A scratch does not buy a larger next bet."""
    n = int(consec_losses or 0)
    if n <= 0:
        return 1.0
    if n == 1:
        return 0.70
    if n == 2:
        return 0.50
    return 0.35


@dataclass(frozen=True)
class TapeRisk:
    mult: float
    cost: float
    vol: float
    liquidity: float
    chase: float
    heat: float

    def note(self) -> str:
        return (
            f"tape×{self.mult:.2f}  cost×{self.cost:.2f}  vol×{self.vol:.2f}  "
            f"liq×{self.liquidity:.2f}  chase×{self.chase:.2f}  heat×{self.heat:.2f}"
        )


def condition_scale(
    *,
    spread: float,
    slip: float,
    stop_dist: float,
    atr_ratio: float = 1.0,
    uncert: float = 0.0,
    stamp: int = 0,
    impulse: float = 0.0,
    consec_losses: int = 0,
) -> TapeRisk:
    cost = cost_factor(spread, slip, stop_dist)
    vol = vol_factor_regime(atr_ratio, uncert)
    liquidity = liquidity_factor(stamp)
    chase = chase_factor(impulse)
    heat = heat_factor(consec_losses)
    mult = _clip(cost * vol * liquidity * chase * heat, 0.25, 1.0)
    return TapeRisk(
        mult=mult,
        cost=cost,
        vol=vol,
        liquidity=liquidity,
        chase=chase,
        heat=heat,
    )


def cost_lock_stop(
    side: str,
    anchor: float,
    stop: float,
    unreal: float,
    spread: float,
    slip: float,
    atr: float,
) -> float:
    """Once open profit has paid the round trip, park the stop at entry."""
    need = max(1.5 * round_trip_px(spread, slip), 0.20 * max(float(atr), 0.0))
    if float(unreal) < need:
        return float(stop)
    if str(side) == "BUY":
        return max(float(stop), float(anchor))
    if str(side) == "SELL":
        return min(float(stop), float(anchor))
    return float(stop)


def dead_scalp(
    bars_held: int,
    unreal: float,
    spread: float,
    slip: float,
    *,
    regime_with: bool,
    impulse_with: bool,
) -> bool:
    """Scratch a trade that has not paid its spread.

    Half an hour is enough when the regime or the impulse has left.
    An hour with no pay is enough even if the regime is still with the trade.
    """
    if float(unreal) >= round_trip_px(spread, slip):
        return False
    held = int(bars_held)
    if held >= SCRATCH_HARD_BARS:
        return True
    if held >= SCRATCH_BARS and not (regime_with and impulse_with):
        return True
    return False
