"""Trade-rate governor: loosen entry gates when the book is under-trading.

``appetite`` > 1 means behind the target rate → lower impulse floors and allow
soft 2oo3 when the NN abstains (HOLD). Appetite never bypasses costs_ok, rails
circuit, fly veto, or kill DD.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace

WINDOW_DEFAULT = 200
TARGET_DEFAULT = 8.0  # sealed entries per 1000 bars
APPETITE_LO = 0.70
APPETITE_HI = 1.60
SOFT_AT = 1.12  # appetite above this → soft 2oo3 on NN HOLD


def _clip(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def parse_trade_rate(spec) -> float | None:
    """None / blank / auto → None (caller chooses default). 0 → off."""
    if spec is None:
        return None
    text = str(spec).strip().lower()
    if text in ("", "auto", "none", "off"):
        return None
    try:
        n = float(text)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    return _clip(n, 1.0, 40.0)


@dataclass(frozen=True)
class AppetitePlan:
    appetite: float
    soft_nn_hold: bool
    min_impulse: float
    cont_impulse: float
    rate: float
    target: float
    note: str

    def params_view(self, base) -> SimpleNamespace:
        """AdaptiveParams-like view with appetite-adjusted impulse floors."""
        snap = base.snapshot() if hasattr(base, "snapshot") else {}
        if not snap:
            for name in (
                "min_impulse",
                "cont_impulse",
                "rsi_buy_arm",
                "rsi_buy_fire",
                "rsi_sell_arm",
                "rsi_sell_fire",
                "rsi_cont_lo",
                "rsi_cont_hi",
            ):
                snap[name] = float(getattr(base, name))
        snap["min_impulse"] = float(self.min_impulse)
        snap["cont_impulse"] = float(self.cont_impulse)
        return SimpleNamespace(**snap)


class TradeAppetite:
    """Rolling entry-rate controller."""

    def __init__(
        self,
        *,
        target_per_1000: float = TARGET_DEFAULT,
        window: int = WINDOW_DEFAULT,
        enabled: bool = True,
    ) -> None:
        self.enabled = bool(enabled) and float(target_per_1000 or 0.0) > 0.0
        self.target = float(target_per_1000) if self.enabled else TARGET_DEFAULT
        self.window = max(40, int(window))
        self._flags: deque[int] = deque(maxlen=self.window)
        self._plan = AppetitePlan(
            appetite=1.0,
            soft_nn_hold=False,
            min_impulse=0.50,
            cont_impulse=0.75,
            rate=0.0,
            target=self.target,
            note="appetite off" if not self.enabled else "appetite warming",
        )
        self.last_plan = self._plan

    @property
    def bars_seen(self) -> int:
        return len(self._flags)

    @property
    def entries_seen(self) -> int:
        return int(sum(self._flags))

    def plan(self, base_min: float, base_cont: float) -> AppetitePlan:
        """Compute this bar's gates from the rolling window (before decide)."""
        if not self.enabled:
            self._plan = AppetitePlan(
                appetite=1.0,
                soft_nn_hold=False,
                min_impulse=float(base_min),
                cont_impulse=float(base_cont),
                rate=0.0,
                target=self.target,
                note="appetite off",
            )
            self.last_plan = self._plan
            return self._plan

        n = len(self._flags)
        entries = int(sum(self._flags))
        rate = (entries / max(n, 1)) * 1000.0
        # Warm-up: do not loosen until we have half a window of tape.
        if n < self.window // 2:
            appetite = 1.0
            note = f"appetite warming  n={n}/{self.window}  target {self.target:.0f}/1k"
        else:
            appetite = _clip(self.target / max(rate, 0.35), APPETITE_LO, APPETITE_HI)
            note = (
                f"appetite {appetite:.2f}  rate {rate:.1f}/1k  "
                f"target {self.target:.0f}/1k  entries {entries}/{n}"
            )

        # appetite>1 → behind target → lower floors (scale < 1)
        floor_scale = _clip(1.0 / max(appetite, 1e-6), 0.72, 1.15)
        min_i = _clip(float(base_min) * floor_scale, 0.35, 0.72)
        cont_i = _clip(float(base_cont) * floor_scale, 0.55, 0.90)
        soft = bool(appetite >= SOFT_AT)
        if soft:
            note = f"{note}  soft-2oo3 on nn HOLD"

        self._plan = AppetitePlan(
            appetite=float(appetite),
            soft_nn_hold=soft,
            min_impulse=float(min_i),
            cont_impulse=float(cont_i),
            rate=float(rate),
            target=float(self.target),
            note=note,
        )
        self.last_plan = self._plan
        return self._plan

    def end_bar(self, entered: bool) -> None:
        """Record whether this bar opened a new sealed entry."""
        if not self.enabled:
            return
        self._flags.append(1 if entered else 0)

    def pack(self) -> dict:
        p = self.last_plan
        return {
            "enabled": bool(self.enabled),
            "appetite": round(float(p.appetite), 3),
            "soft_nn_hold": bool(p.soft_nn_hold),
            "rate": round(float(p.rate), 2),
            "target": round(float(p.target), 1),
            "min_impulse": round(float(p.min_impulse), 3),
            "cont_impulse": round(float(p.cont_impulse), 3),
            "note": str(p.note),
        }
