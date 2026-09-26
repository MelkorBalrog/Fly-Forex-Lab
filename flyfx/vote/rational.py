"""Plausibility and rationality for a bull or bear decision.

Five trend readings and three momentum readings vote independently.
A side is plausible when both groups support it and the opposition is
clearly smaller. It is irrational when RSI, stochastic, and Bollinger
position are all pinned, which is a chase, not a fresh entry.
A missing reading abstains. The same thresholds on every pair.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Verdict:
    ok: bool
    why: str
    mult: float
    trend_for: int
    mom_for: int
    against: int

    def note(self) -> str:
        return self.why


def assess(side: str, feat: dict) -> Verdict:
    """Accept, shrink, or refuse ``side`` from the indicator tape in ``feat``."""
    want = str(side)
    if want not in ("BUY", "SELL"):
        return Verdict(False, "no side", 0.0, 0, 0, 0)
    if _stretched(want, feat):
        return Verdict(
            False,
            (
                f"irrational {want} stretch  "
                f"rsi {float(feat.get('rsi') or 50):.0f}  "
                f"stoch {float(feat.get('stoch') or 0.5):.2f}  "
                f"bb {float(feat.get('bb_pct') or 0.5):.2f}"
            ),
            0.0,
            0,
            0,
            0,
        )
    votes = {
        "ma": _ma(feat),
        "di": _di(feat),
        "sar": _sar(feat),
        "supertrend": _supertrend(feat),
        "cloud": _cloud(feat),
        "macd": _macd(feat),
        "rsi": _rsi(feat),
        "osc": _osc(feat),
    }
    trend_names = ("ma", "di", "sar", "supertrend", "cloud")
    mom_names = ("macd", "rsi", "osc")
    trend_for = sum(1 for n in trend_names if votes[n] == want)
    mom_for = sum(1 for n in mom_names if votes[n] == want)
    against_names = [n for n, v in votes.items() if v in ("BUY", "SELL") and v != want]
    against = len(against_names)
    for_n = trend_for + mom_for
    plausible = trend_for >= 2 and mom_for >= 1 and for_n >= against + 2
    if not plausible:
        opposed = ",".join(against_names) or "-"
        return Verdict(
            False,
            (
                f"implausible {want}  trend {trend_for}/5  mom {mom_for}/3  "
                f"against {against} ({opposed})"
            ),
            0.0,
            trend_for,
            mom_for,
            against,
        )
    if against == 0 and trend_for >= 3 and mom_for >= 2:
        mult = 1.0
    elif against == 0:
        mult = 0.85
    else:
        mult = 0.70
    return Verdict(
        True,
        f"rational {want}  trend {trend_for}/5  mom {mom_for}/3  against {against}  ×{mult:.2f}",
        mult,
        trend_for,
        mom_for,
        against,
    )


def _stretched(side: str, feat: dict) -> bool:
    rsi = float(feat.get("rsi") or 50.0)
    stoch = float(feat.get("stoch") or 0.5)
    bb = float(feat.get("bb_pct") or 0.5)
    if side == "BUY":
        return rsi >= 76.0 and stoch >= 0.85 and bb >= 0.90
    return rsi <= 24.0 and stoch <= 0.15 and bb <= 0.10


def _atr(feat: dict) -> float:
    return max(float(feat.get("atr") or 0.0), 1e-12)


def _ma(feat: dict) -> str:
    fast = feat.get("ema_fast")
    slow = feat.get("ema_slow")
    if fast is None or slow is None:
        return ""
    gap = (float(fast) - float(slow)) / _atr(feat)
    if gap >= 0.10:
        return "BUY"
    if gap <= -0.10:
        return "SELL"
    return ""


def _di(feat: dict) -> str:
    if float(feat.get("adx") or 0.0) < 18.0:
        return ""
    plus = float(feat.get("plus_di") or 0.0)
    minus = float(feat.get("minus_di") or 0.0)
    if plus >= minus + 2.0:
        return "BUY"
    if minus >= plus + 2.0:
        return "SELL"
    return ""


def _sar(feat: dict) -> str:
    sar = feat.get("sar")
    close = feat.get("close")
    if sar is None or close is None:
        return ""
    gap = (float(close) - float(sar)) / _atr(feat)
    if gap >= 0.05:
        return "BUY"
    if gap <= -0.05:
        return "SELL"
    return ""


def _supertrend(feat: dict) -> str:
    if int(feat.get("bars_seen") or 0) < 30:
        return ""
    direction = int(feat.get("st_dir") or 0)
    if direction > 0:
        return "BUY"
    if direction < 0:
        return "SELL"
    return ""


def _cloud(feat: dict) -> str:
    ichi = feat.get("ichi") or {}
    cloud = float(ichi.get("cloud") or 0.0)
    if cloud > 0:
        return "BUY"
    if cloud < 0:
        return "SELL"
    return ""


def _macd(feat: dict) -> str:
    hist = float(feat.get("macd_hist") or 0.0)
    if hist >= 0.05:
        return "BUY"
    if hist <= -0.05:
        return "SELL"
    return ""


def _rsi(feat: dict) -> str:
    rsi = float(feat.get("rsi") or 50.0)
    if rsi >= 55.0:
        return "BUY"
    if rsi <= 45.0:
        return "SELL"
    return ""


def _osc(feat: dict) -> str:
    leans: list[str] = []
    stoch = feat.get("stoch")
    if stoch is not None:
        s = float(stoch)
        if s >= 0.62:
            leans.append("BUY")
        elif s <= 0.38:
            leans.append("SELL")
    willr = feat.get("willr")
    if willr is not None:
        w = float(willr)
        if w >= -40.0:
            leans.append("BUY")
        elif w <= -60.0:
            leans.append("SELL")
    cci = feat.get("cci")
    if cci is not None:
        c = float(cci)
        if c >= 40.0:
            leans.append("BUY")
        elif c <= -40.0:
            leans.append("SELL")
    buy_n = sum(1 for v in leans if v == "BUY")
    sell_n = sum(1 for v in leans if v == "SELL")
    if buy_n >= 2 and buy_n > sell_n:
        return "BUY"
    if sell_n >= 2 and sell_n > buy_n:
        return "SELL"
    return ""
