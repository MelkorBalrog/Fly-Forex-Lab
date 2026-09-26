"""Regime state machine that picks which category Kalmans to mix.

The 3-voter 2oo3 stays fixed. Only the impulse mix changes with market
conditions. Overlay families that never measured stay out — they are not
faked. Hysteresis stops the mix from chattering every bar.

States
------
TREND  ADX is alive and the trend Kalman agrees with the EMA regime.
       Mix: trend + structure  (+ volume if live).
RANGE  ADX is dead and oscillators are stretched.
       Mix: momentum + structure  (+ volume if live).
BREAK  Bollinger squeeze then ATR expansion.
       Mix: volatility + trend  (+ volume if live).
QUIET  None of the above. Conservative trend-only mix (CHOP still HOLDs).

The checked FUSE boxes / ``--fuse-cats`` / MT4 Inputs are an *allowlist*
the machine intersects. Empty intersection falls back to trend, then the
first allowed name.
"""

from __future__ import annotations

from flyfx.sense.category_kalman import CATEGORY_NAMES, SPARSE_CATEGORIES, parse_fuse_cats

STATES = ("TREND", "RANGE", "BREAK", "QUIET")

FUSE_FOR_STATE: dict[str, tuple[str, ...]] = {
    "TREND": ("trend", "structure"),
    "RANGE": ("momentum", "structure"),
    "BREAK": ("volatility", "trend"),
    "QUIET": ("trend",),
}

ADX_TREND_IN = 22.0
ADX_TREND_OUT = 16.0
ADX_RANGE_IN = 18.0
ADX_RANGE_OUT = 22.0
RSI_STRETCH_LO = 32.0
RSI_STRETCH_HI = 68.0
TREND_FUSED = 0.25
MOM_FUSED = 0.28
SQUEEZE_BW = 0.78
SQUEEZE_BARS = 3
EXPAND_ATR = 1.12
EXPAND_BW = 0.95
HOLD_BARS = 8
VOLUME_FUSED = 0.20


def _cat(cats: dict, name: str) -> dict:
    return (cats or {}).get(name) or {}


def _fused(cats: dict, name: str) -> float:
    st = _cat(cats, name)
    if not st.get("available"):
        return 0.0
    try:
        return abs(float(st.get("fused") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _clip_allow(wanted: tuple[str, ...], allow: tuple[str, ...], cats: dict) -> tuple[str, ...]:
    allow_set = set(allow) if allow else set(CATEGORY_NAMES)
    out: list[str] = []
    for name in wanted:
        if name not in allow_set:
            continue
        st = _cat(cats, name)
        if name in SPARSE_CATEGORIES and not st.get("available"):
            continue
        if name not in out:
            out.append(name)
    vol = _cat(cats, "volume")
    if (
        "volume" in allow_set
        and vol.get("available")
        and _fused(cats, "volume") >= VOLUME_FUSED
        and "volume" not in out
    ):
        out.append("volume")
    if out:
        return tuple(out)
    if "trend" in allow_set:
        return ("trend",)
    return (allow[0],) if allow else ("trend",)


def propose_state(raw: dict, cats: dict, squeeze_bars: int) -> tuple[str, str, int]:
    """Pure classifier (no hysteresis). Returns state, reason, new squeeze_bars."""
    adx = float(raw.get("adx") or 0.0)
    rsi = float(raw.get("rsi") or 50.0)
    atr_ratio = float(raw.get("atr_ratio") or 1.0)
    bb_bw = float(raw.get("bb_bw") or 0.0)
    bb_bw_ma = float(raw.get("bb_bw_ma") or 0.0)
    ema_reg = str(raw.get("regime") or "CHOP")
    bw_ratio = bb_bw / max(bb_bw_ma, 1e-9) if bb_bw_ma > 0 else 1.0

    sq = squeeze_bars
    squeezed = bw_ratio < SQUEEZE_BW
    armed = squeeze_bars >= SQUEEZE_BARS
    if squeezed:
        sq = squeeze_bars + 1
    else:
        sq = max(0, squeeze_bars - 1)

    expanding = atr_ratio >= EXPAND_ATR and bw_ratio >= EXPAND_BW
    if armed and expanding:
        return "BREAK", f"squeeze {squeeze_bars} bars then ATR×{atr_ratio:.2f} bw {bw_ratio:.2f}", sq

    trend_ok = (
        adx >= ADX_TREND_IN
        and _fused(cats, "trend") >= TREND_FUSED
        and ema_reg in ("UP", "DOWN")
    )
    if trend_ok:
        return "TREND", f"ADX {adx:.1f}  |trend fused| {_fused(cats, 'trend'):.2f}  EMA {ema_reg}", sq

    stretched = rsi <= RSI_STRETCH_LO or rsi >= RSI_STRETCH_HI or _fused(cats, "momentum") >= MOM_FUSED
    if adx <= ADX_RANGE_IN and stretched:
        return "RANGE", f"ADX {adx:.1f}  RSI {rsi:.0f}  |mom fused| {_fused(cats, 'momentum'):.2f}", sq

    return "QUIET", f"ADX {adx:.1f}  EMA {ema_reg}  no stretch/squeeze", sq


def may_leave(state: str, want: str, raw: dict, bars_in: int, hold_bars: int) -> bool:
    if want == state:
        return False
    if bars_in < hold_bars and not (state == "BREAK" and want in ("TREND", "QUIET")):
        return False
    adx = float(raw.get("adx") or 0.0)
    if state == "TREND" and want != "BREAK" and adx >= ADX_TREND_OUT:
        return False
    if state == "RANGE" and want != "BREAK" and adx <= ADX_RANGE_OUT:
        return False
    return True


class FuseRegimeMachine:
    """Hysteretic mixer picker. One instance per Indicators / trade loop."""

    def __init__(self, hold_bars: int = HOLD_BARS):
        self.hold_bars = max(1, int(hold_bars))
        self.state = "QUIET"
        self.bars_in = 0
        self.squeeze_bars = 0
        self.reason = "init"
        self.include: tuple[str, ...] = FUSE_FOR_STATE["QUIET"]
        self.want = "QUIET"
        self._primed = False

    def step(self, raw: dict | None, cats: dict | None, allow: tuple[str, ...] | list[str] | None = None) -> dict:
        raw = raw or {}
        cats = cats or {}
        try:
            allowed = parse_fuse_cats(allow)
        except ValueError:
            allowed = tuple(CATEGORY_NAMES)
        want, why, self.squeeze_bars = propose_state(raw, cats, self.squeeze_bars)
        self.want = want
        if not self._primed:
            self.state = want
            self.bars_in = 1
            self.reason = why
            self._primed = True
        elif want == self.state:
            self.bars_in += 1
            self.reason = why
        elif may_leave(self.state, want, raw, self.bars_in, self.hold_bars):
            self.state = want
            self.bars_in = 1
            self.reason = why
        else:
            self.bars_in += 1
            self.reason = f"hold {self.state} ({self.bars_in}/{self.hold_bars}) want {want}"
        wanted = FUSE_FOR_STATE.get(self.state) or FUSE_FOR_STATE["QUIET"]
        self.include = _clip_allow(wanted, allowed, cats)
        return {
            "dynamic": True,
            "state": self.state,
            "want": self.want,
            "reason": self.reason,
            "include": list(self.include),
            "bars_in": self.bars_in,
            "squeeze_bars": self.squeeze_bars,
            "allow": list(allowed),
        }


def idle_fuse_plan(include: tuple[str, ...] | list[str] | None = None) -> dict:
    names = list(include) if include else list(CATEGORY_NAMES)
    return {
        "dynamic": False,
        "state": "OFF",
        "want": "OFF",
        "reason": "manual mix",
        "include": names,
        "bars_in": 0,
        "squeeze_bars": 0,
        "allow": names,
    }
