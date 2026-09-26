"""Sugar GRNs = food present. Satiety tank = already full. Appetite = eat or not.

The 23 MaleCNS sugar receptors never vote BUY/SELL.

When sugar feed is **off**, they still get the 4k-era lamp: ``0.45 × nice`` on
the risk pass. Eat every 2oo3 (no skip / size / flatten).

When sugar feed is **on**, a satiety tank mixes with a local GRN stamp and may
size, skip, pyramid, or spit out a winner.
"""

from __future__ import annotations

import math

IDLE = {
    "feeding": 0.50,
    "satiety": 0.50,
    "hunger": 0.50,
    "appetite": 1.0,
    "nutrition": 0.50,
    "dose": 0.0,
    "dyn": False,
    "dyn_state": "OFF",
    "dyn_reason": "manual",
    "size_mult": 1.0,
    "skip": False,
    "snowball": True,
    "flatten": False,
    "reason": "off",
}

# Pre-feed overlay (4k-era): 23 GRNs still get 0.45×nice on the risk pass.
# Eat every 2oo3 — no skip / size / flatten. Never votes BUY/SELL.
NATIVE_LAMP = {
    **IDLE,
    "reason": "native lamp",
    "dyn_reason": "native",
}

SUGAR_AMT_MIN = 0.0
SUGAR_AMT_MAX = 2.0
SUGAR_AMT_DEFAULT = 1.0


def parse_sugar_amt(val, default: float = SUGAR_AMT_DEFAULT) -> float:
    """How much sugar to put on the tongue when feed is on.

    ``0`` = none (eat like sugar off). ``1`` = full (today's default). ``2`` = extra.
    Values above 2 are treated as percent (``100`` → 1.0, ``50`` → 0.5).
    """
    if val is None or val == "":
        return float(default)
    try:
        x = float(val)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(x):
        return float(default)
    if 10.0 - 1e-9 <= x <= 200.0 + 1e-9:
        x = x / 100.0
    return float(min(max(x, SUGAR_AMT_MIN), SUGAR_AMT_MAX))


def food_nutrition(
    tag: str,
    *,
    banc_mult: float = 1.0,
    uncertainty: float = 0.5,
    agreement: float = 0.5,
    n_agree: int = 0,
    in_trade: bool = False,
    trend_with: bool = False,
) -> float:
    """How sweet the current picture is (food on the tongue). Not satiety."""
    t = str(tag or "").lower()
    if in_trade and trend_with:
        tag_score = 0.88
    elif "3oo3" in t or n_agree >= 3:
        tag_score = 1.0
    elif "2oo3" in t or n_agree >= 2:
        tag_score = 0.74
    elif "bounce" in t:
        tag_score = 0.46
    elif in_trade:
        tag_score = 0.28
    else:
        tag_score = 0.12
    banc_ok = float(min(max(banc_mult, 0.0), 1.20)) / 1.20
    clean = 0.50 * (1.0 - float(min(max(uncertainty, 0.0), 1.0))) + 0.50 * float(
        min(max(agreement, 0.0), 1.0)
    )
    return float(min(max(0.50 * tag_score + 0.28 * banc_ok + 0.22 * clean, 0.0), 1.0))


def compute_satiety(
    *,
    equity: float,
    peak_eq: float,
    start: float,
    day_start_eq: float,
    floating: float,
    day_trades: int,
    day_trade_cap: int,
) -> float:
    """Internal sugar: 0 starving, 1 stuffed. From the book, not the GRNs."""
    peak = max(float(peak_eq or 0.0), float(equity or 0.0), 1e-9)
    dd = max(0.0, (peak - float(equity or 0.0)) / peak)
    hunger_dd = min(dd / 0.025, 1.0)
    base = max(float(start or 0.0), 1e-9)
    day_gain = (float(equity or 0.0) - float(day_start_eq or start or 0.0)) / base
    full_day = min(max(day_gain / 0.015, 0.0), 1.0)
    float_frac = float(floating or 0.0) / max(float(equity or 0.0), 1.0)
    full_float = min(max(float_frac / 0.006, 0.0), 1.0)
    cap = max(int(day_trade_cap or 1), 1)
    trade_fill = min(max(int(day_trades or 0) / cap, 0.0), 1.0)
    sat = (
        0.38 * (1.0 - hunger_dd)
        + 0.32 * full_day
        + 0.20 * full_float
        + 0.10 * trade_fill
    )
    return float(min(max(sat, 0.0), 1.0))


def apply_sugar_dose(pack: dict, dose: float = SUGAR_AMT_DEFAULT) -> dict:
    """Blend full sugar policy toward eat-everything. ``1`` is identity."""
    d = parse_sugar_amt(dose)
    out = dict(pack or {})
    out["dose"] = round(d, 3)
    if abs(d - 1.0) <= 1e-12:
        return out
    w = min(d, 1.0)
    extra = max(d - 1.0, 0.0)
    size = 1.0 + w * (float(out.get("size_mult") or 1.0) - 1.0)
    if extra:
        size *= 1.0 + 0.22 * extra
    out["size_mult"] = round(float(min(max(size, 0.50), 1.40)), 4)
    if w <= 1e-12:
        out["skip"] = False
        out["snowball"] = True
        out["flatten"] = False
        out["reason"] = "dose 0"
        return out
    if w < 0.45:
        out["skip"] = False
        out["flatten"] = False
        out["snowball"] = True
        reason = str(out.get("reason") or "")
        if reason in {"stuffed", "full, food weak", "no appetite", "sugar full"}:
            out["reason"] = "forage"
    return out


def mix_appetite(
    feeding: float,
    satiety: float,
    *,
    tag: str = "",
    bars_held: int = 0,
    min_hold: int = 4,
    unreal: float = 0.0,
    atr: float = 0.0,
    dose: float = SUGAR_AMT_DEFAULT,
) -> dict:
    """Hungry × sweet → eat. Full → skip / no pyramid / spit out a winner.

    ``dose`` (0..2) is how much sugar is on the tongue. ``1`` is the full
    policy; ``0`` eats like sugar off; ``2`` is extra sweet (size up).
    """
    feed = float(min(max(feeding, 0.0), 1.0))
    sat = float(min(max(satiety, 0.0), 1.0))
    hunger = 1.0 - sat
    appetite = float(min(max(feed * (0.25 + 0.75 * hunger), 0.0), 1.0))
    t = str(tag or "").lower()
    feast = "3oo3" in t
    skip = False
    reason = "forage"
    if sat >= 0.92:
        skip = True
        reason = "stuffed"
    elif sat >= 0.80 and feed < 0.62 and not feast:
        skip = True
        reason = "full, food weak"
    elif appetite < 0.18 and not feast:
        skip = True
        reason = "no appetite"
    snowball = sat < 0.72 and appetite >= 0.35
    atr = max(float(atr or 0.0), 1e-12)
    flatten = (
        sat >= 0.84
        and int(bars_held) >= int(min_hold)
        and float(unreal) > 0.40 * atr
    )
    size_mult = float(min(max(0.62 + 0.70 * appetite, 0.55), 1.22))
    pack = {
        "feeding": round(feed, 4),
        "satiety": round(sat, 4),
        "hunger": round(hunger, 4),
        "appetite": round(appetite, 4),
        "size_mult": round(size_mult, 4),
        "skip": bool(skip),
        "snowball": bool(snowball),
        "flatten": bool(flatten),
        "reason": "sugar full" if flatten else reason,
    }
    return apply_sugar_dose(pack, dose)


def feeding_from_raw(raw: float, rest: float) -> float:
    """Map GRN mean |x| minus rest into 0..1 sweet."""
    score = float(raw) - float(rest)
    return float(min(max(0.5 + 0.5 * math.tanh(3.2 * score), 0.0), 1.0))


def native_lamp_pack(raw: float, rest: float) -> dict:
    """Telemetry for the 4k-era GRN lamp. Does not skip, size, or flatten."""
    feed = feeding_from_raw(raw, rest)
    try:
        nut = float(raw)
    except (TypeError, ValueError):
        nut = 0.0
    if not math.isfinite(nut):
        nut = 0.0
    out = dict(NATIVE_LAMP)
    out["feeding"] = round(feed, 4)
    out["nutrition"] = round(float(min(max(nut, 0.0), 1.0)), 4)
    return out


SUGAR_STATES = ("FEAST", "FORAGE", "NIBBLE", "FAST")
DOSE_FOR_STATE = {
    "FEAST": 1.70,
    "FORAGE": 1.00,
    "NIBBLE": 0.55,
    "FAST": 0.50,
}
SUGAR_HOLD_BARS = 6


def _clip01(val, default: float = 0.5) -> float:
    try:
        x = float(val)
    except (TypeError, ValueError):
        x = float(default)
    if not math.isfinite(x):
        x = float(default)
    return float(min(max(x, 0.0), 1.0))


def propose_sugar_state(cond: dict | None) -> tuple[str, str]:
    """Pure classifier (no hysteresis). How much sugar to pour this bar."""
    c = cond or {}
    sat = _clip01(c.get("satiety"), 0.5)
    hunger = 1.0 - sat
    tag = str(c.get("tag") or "").lower()
    try:
        n_agree = int(c.get("n_agree") or 0)
    except (TypeError, ValueError):
        n_agree = 0
    try:
        banc = float(c.get("banc_mult") if c.get("banc_mult") is not None else 1.0)
    except (TypeError, ValueError):
        banc = 1.0
    uncert = _clip01(c.get("uncertainty"), 0.5)
    agr = _clip01(c.get("agreement"), 0.5)
    regime = str(c.get("regime") or "CHOP").upper()
    try:
        atr_ratio = float(c.get("atr_ratio") if c.get("atr_ratio") is not None else 1.0)
    except (TypeError, ValueError):
        atr_ratio = 1.0
    nutrition = _clip01(c.get("nutrition"), 0.5)
    in_trade = bool(c.get("in_trade"))
    kill = bool(c.get("rails_kill"))
    feast = "3oo3" in tag or n_agree >= 3
    two = "2oo3" in tag or n_agree >= 2

    if kill or banc < 0.58 or uncert >= 0.82:
        why = f"threat BANC×{banc:.2f} uncert {uncert:.2f}"
        if kill:
            why += " kill"
        return "FAST", why
    if sat >= 0.80:
        return "FORAGE", f"sat {sat:.2f} keep skip/flatten policy"
    clean = banc >= 0.88 and uncert <= 0.48 and agr >= 0.45
    if hunger >= 0.42 and feast and clean:
        return "FEAST", f"hungry {hunger:.2f} 3oo3 BANC×{banc:.2f}"
    if hunger >= 0.55 and two and banc >= 0.95 and uncert <= 0.40:
        return "FEAST", f"starved {hunger:.2f} 2oo3"
    weak = ((not two) and not in_trade) or nutrition < 0.38 or (regime == "CHOP" and not in_trade)
    filling = sat >= 0.62
    threatish = banc < 0.78 or atr_ratio >= 1.85
    if weak or filling or threatish:
        bits = []
        if weak:
            bits.append("chop" if regime == "CHOP" and not two else "weak food")
        if filling:
            bits.append(f"filling sat {sat:.2f}")
        if threatish:
            bits.append(f"BANC×{banc:.2f}" if banc < 0.78 else f"ATR×{atr_ratio:.2f}")
        return "NIBBLE", "  ".join(bits) or "nibble"
    return "FORAGE", f"{'2oo3' if two else 'hold'} sat {sat:.2f}"


def may_leave_sugar(state: str, want: str, bars_in: int, hold_bars: int, *, satiety: float = 0.5) -> bool:
    if want == state:
        return False
    if want == "FAST":
        return True
    if want == "FORAGE" and float(satiety) >= 0.80:
        return True
    if bars_in < hold_bars:
        return False
    return True


class SugarDoseMachine:
    """Hysteretic sugar pour. One instance per trade loop. Never votes BUY/SELL.

    FEAST  extra sugar (hungry + 3oo3 + clean sensors).
    FORAGE full dose (including stuffed — so skip/flatten still fire).
    NIBBLE half sugar (chop, weak food, filling up).
    FAST   austere (BANC threat, kill, dirty Kalman). Floor keeps policy alive.

    Operator ``amt`` is a scale: 1 = machine as designed, 2 = extra headroom.
    """

    def __init__(self, hold_bars: int = SUGAR_HOLD_BARS):
        self.hold_bars = max(1, int(hold_bars))
        self.state = "FORAGE"
        self.bars_in = 0
        self.reason = "init"
        self.want = "FORAGE"
        self.dose = 1.0
        self._primed = False

    def step(self, cond: dict | None = None, scale: float = SUGAR_AMT_DEFAULT) -> dict:
        cond = cond or {}
        want, why = propose_sugar_state(cond)
        self.want = want
        sat = _clip01(cond.get("satiety"), 0.5)
        if not self._primed:
            self.state = want
            self.bars_in = 1
            self.reason = why
            self._primed = True
        elif want == self.state:
            self.bars_in += 1
            self.reason = why
        elif may_leave_sugar(self.state, want, self.bars_in, self.hold_bars, satiety=sat):
            self.state = want
            self.bars_in = 1
            self.reason = why
        else:
            self.bars_in += 1
            self.reason = f"hold {self.state} ({self.bars_in}/{self.hold_bars}) want {want}"
        base = float(DOSE_FOR_STATE.get(self.state) or 1.0)
        sc = parse_sugar_amt(scale)
        self.dose = parse_sugar_amt(base * sc)
        return {
            "dynamic": True,
            "state": self.state,
            "want": self.want,
            "reason": self.reason,
            "dose": round(self.dose, 3),
            "base": round(base, 3),
            "scale": round(sc, 3),
            "bars_in": self.bars_in,
        }
