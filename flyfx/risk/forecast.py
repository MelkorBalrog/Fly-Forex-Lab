"""A timed price forecast for an open trade. Same rule on every pair.

At the fill, the next hour of M5 bars is the window. The target is the
recent cycle and the impulse, in ATR, and it has to clear the round trip.
If that price prints inside the window and the trade is in profit, close.
If price runs the other way by about half the initial risk, close before
the full stop takes the money.
"""

from __future__ import annotations

from dataclasses import dataclass

HORIZON = 12
DAMAGE_R = 0.45
MIN_DRIFT_ATR = 0.45
MAX_DRIFT_ATR = 1.20


@dataclass(frozen=True)
class Forecast:
    side: str
    target: float
    damage_px: float
    horizon: int
    drift_atr: float

    def note(self) -> str:
        return (
            f"forecast {self.drift_atr:.2f} ATR in {self.horizon} bars  "
            f"cut {DAMAGE_R:.2f}R"
        )


def project(
    side: str,
    entry: float,
    atr: float,
    roc: float,
    impulse: float,
    risk_px: float,
    cost_px: float,
) -> Forecast:
    """Project a with-trade target from the cycle and the impulse."""
    span = max(float(atr), 1e-12)
    sign = 1.0 if str(side) == "BUY" else -1.0
    signed_roc = sign * float(roc or 0.0)
    signed_imp = sign * float(impulse or 0.0)
    pace = 0.70 * max(signed_roc, 0.0) + 0.55 * max(signed_imp, 0.0)
    drift_atr = min(MAX_DRIFT_ATR, max(MIN_DRIFT_ATR, pace))
    drift_px = max(drift_atr * span, 1.8 * max(float(cost_px), 0.0))
    risk = max(float(risk_px), 1e-12)
    damage = max(2.0 * max(float(cost_px), 0.0), DAMAGE_R * risk, 0.35 * span)
    damage = min(damage, 0.55 * risk)
    return Forecast(
        side=str(side),
        target=float(entry) + sign * drift_px,
        damage_px=damage,
        horizon=HORIZON,
        drift_atr=drift_px / span,
    )


def forecast_exit(
    forecast: Forecast | None,
    side: str,
    mark: float,
    unreal: float,
    bars_held: int,
    cost_px: float,
) -> str | None:
    """Close on the forecast, or on an adverse move inside its window."""
    if forecast is None or int(bars_held) > int(forecast.horizon):
        return None
    if int(bars_held) >= 2 and float(unreal) <= -float(forecast.damage_px):
        return "prediction fail"
    paid = float(unreal) > max(float(cost_px), 0.0)
    if not paid:
        return None
    if str(side) == "BUY" and float(mark) >= float(forecast.target):
        return "prediction hit"
    if str(side) == "SELL" and float(mark) <= float(forecast.target):
        return "prediction hit"
    return None
