"""Cycle rate-of-change. Same clock on every pair.

A cycle is six M5 bars, half an hour. Rate of change is that move in ATR.
The jump is this cycle minus the previous one. A jump with the trade
increases size in proportion. A jump the other way, once the trade is
in profit, is a reason to get flat.
"""

from __future__ import annotations

CYCLE = 6
ROC_MIN = 0.20
JUMP_MIN = 0.30
ADD_JUMP = 0.45
FLIP_ROC = 0.15
FLIP_JUMP = 0.35
THRUST_CAP = 2.0


class CycleTape:
    """Rolling close tape. Push once per bar, in order."""

    def __init__(self) -> None:
        self._closes: list[float] = []
        self.roc = 0.0
        self.prev_roc = 0.0

    def push(self, close: float, atr: float) -> None:
        self._closes.append(float(close))
        keep = CYCLE + 1
        if len(self._closes) > keep:
            del self._closes[: len(self._closes) - keep]
        if len(self._closes) <= CYCLE:
            return
        span = max(float(atr), 1e-12)
        raw = (self._closes[-1] - self._closes[0]) / span
        self.prev_roc = self.roc
        self.roc = raw

    @property
    def jump(self) -> float:
        return self.roc - self.prev_roc


def _signed(side: str, value: float) -> float:
    return float(value) if str(side) == "BUY" else -float(value)


def thrust_mult(side: str, roc: float, jump: float) -> float:
    """1 when the cycle is quiet. Up to 2 when it accelerates with the trade."""
    if _signed(side, roc) < ROC_MIN or _signed(side, jump) < JUMP_MIN:
        return 1.0
    extra = (_signed(side, jump) - JUMP_MIN) / 1.0
    return min(THRUST_CAP, 1.0 + max(extra, 0.0))


def thrust_add_frac(side: str, roc: float, jump: float) -> float:
    """Share of the original lots to add. 0 until the jump is sharp."""
    accel = _signed(side, jump)
    if _signed(side, roc) < ROC_MIN or accel < ADD_JUMP:
        return 0.0
    return min(0.80, 0.25 + (accel - ADD_JUMP) * 0.55)


def roc_flip(side: str, roc: float, jump: float, unreal: float, cost_px: float) -> bool:
    """True when a profitable trade's cycle has turned against it."""
    if float(unreal) <= max(float(cost_px), 0.0):
        return False
    turned = _signed(side, roc) <= -FLIP_ROC
    sudden = _signed(side, jump) <= -FLIP_JUMP
    return turned and sudden
