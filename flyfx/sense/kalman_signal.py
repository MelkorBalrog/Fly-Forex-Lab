"""Kalman primitives and the legacy EMA/RSI/ATR mixer.

Live decision fusion is per-indicator-category (see ``category_kalman.py``).
``IndicatorKalmanFusion`` remains for the ``--legacy-2oo3`` path and as a
debug dump under ``feat['kalman']['legacy']``.
"""

from __future__ import annotations

import math

import numpy as np


def is_measurement(z: object) -> bool:
    try:
        v = float(z)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


class Kalman1D:
    """Scalar random-walk Kalman. ``step(None)`` is predict-only (P grows)."""

    def __init__(self, q: float, r: float, x: float = 0.0, p: float = 1.0):
        self.q = q
        self.r = r
        self.x = x
        self.P = p

    def step(self, z: float | None = None) -> tuple[float, float]:
        # Match snapshots/eurusd-snowball-lock Kalman1D: uncapped P growth on
        # the legacy path so impulse / regime stay freeze-identical.
        self.P = self.P + self.q
        if not is_measurement(z):
            self.P = min(self.P, 1.0e6)
            return self.x, self.P
        zf = float(z)
        k = self.P / (self.P + self.r)
        self.x = self.x + k * (zf - self.x)
        self.P = (1.0 - k) * self.P
        return self.x, self.P


class KalmanCV:
    """Constant-velocity filter: state = [level, slope]."""

    def __init__(self, dt: float = 1.0, q: float = 1e-5, r: float = 4e-3):
        self.dt = dt
        self.q = q
        self.r = r
        self.x = np.zeros(2, dtype=float)
        self.P = np.eye(2, dtype=float)
        self._started = False

    def step(self, z: float | None = None) -> np.ndarray:
        # Freeze-identical order: seed level before the first predict when
        # the measurement is live (snapshots/eurusd-snowball-lock-20260919).
        if is_measurement(z) and not self._started:
            self.x[0] = float(z)
            self._started = True
        f = np.array([[1.0, self.dt], [0.0, 1.0]])
        q = self.q * np.array(
            [
                [self.dt ** 3 / 3.0, self.dt ** 2 / 2.0],
                [self.dt ** 2 / 2.0, self.dt],
            ]
        )
        self.x = f @ self.x
        self.P = f @ self.P @ f.T + q
        if not is_measurement(z):
            np.clip(self.P, -1.0e6, 1.0e6, out=self.P)
            return self.x
        h = np.array([1.0, 0.0])
        y = float(z) - float(h @ self.x)
        s = float(h @ self.P @ h + self.r)
        k = (self.P @ h) / max(s, 1e-12)
        self.x = self.x + k * y
        self.P = self.P - np.outer(k, h) @ self.P
        return self.x


class IndicatorKalmanFusion:
    """Legacy mixer: EMA gap + momentum + RSI + price slope → one impulse.

    Kept for ``--legacy-2oo3`` / nest / koo9 encodings. Default trading
    decisions use ``CategoryKalmanBank`` instead.
    """

    def __init__(self):
        self.kf_trend = Kalman1D(q=2.0e-4, r=8.0e-3)
        self.kf_mom = Kalman1D(q=5.0e-4, r=2.5e-2)
        self.kf_rsi = Kalman1D(q=3.5e-4, r=3.5e-2)
        self.kf_price = KalmanCV(q=8.0e-6, r=5.0e-3)
        self.anchor: float | None = None

    def update(
        self,
        close: float,
        atr: float,
        ema_fast: float,
        ema_slow: float,
        rsi: float,
        mom: float,
    ) -> dict:
        atr = max(float(atr), 1e-8)
        if self.anchor is None:
            self.anchor = close
        z_trend = float(np.clip((ema_fast - ema_slow) / atr, -4.0, 4.0))
        z_mom = float(np.clip(mom, -4.0, 4.0))
        z_rsi = float(np.clip((rsi - 50.0) / 12.5, -4.0, 4.0))
        z_px = float(np.clip((close - self.anchor) / atr, -8.0, 8.0))

        xt, pt = self.kf_trend.step(z_trend)
        xm, pm = self.kf_mom.step(z_mom)
        xr, pr = self.kf_rsi.step(z_rsi)
        level, slope = self.kf_price.step(z_px)
        xslope = float(np.clip(slope * 10.0, -4.0, 4.0))
        residual = float(np.clip(z_px - float(level), -4.0, 4.0))

        weights = np.array(
            [
                1.0 / max(pt, 1e-9),
                1.0 / max(pm, 1e-9),
                1.0 / max(pr, 1e-9),
                4.0,
            ]
        )
        xs = np.array([xt, xm, xr, xslope])
        fused = float(np.dot(weights, xs) / weights.sum())
        impulse = float(np.tanh(0.9 * fused))
        if fused > 0.28:
            regime = "UP"
        elif fused < -0.28:
            regime = "DOWN"
        else:
            regime = "CHOP"
        return {
            "impulse": impulse,
            "fused": fused,
            "kalman_regime": regime,
            "slope": slope,
            "level": level,
            "residual": residual,
            "parts": {"trend": xt, "mom": xm, "rsi": xr, "slope": xslope},
        }
