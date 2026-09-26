"""Higher-timeframe readings on the same base bar.

Closed 10, 15, 30, and 60 minute bars each keep their own indicators.
A base bar only sees buckets that have already closed, so the higher
timeframe does not look into the future.
"""

from __future__ import annotations

MTF_MINUTES = (10, 15, 30, 60)


def _snap(feat: dict) -> dict:
    return {
        "impulse": float(feat.get("impulse") or 0.0),
        "regime": str(feat.get("regime") or "CHOP"),
        "ready": bool(feat.get("ready")),
    }


def _blank() -> dict:
    return {"impulse": 0.0, "regime": "CHOP", "ready": False}


class MultiTimeframe:
    def __init__(self, pip: float = 0.0001, mixer: str = "cat_fuse") -> None:
        from flyfx.trader import Indicators

        self.inds = {m: Indicators(pip=pip, mixer=mixer) for m in MTF_MINUTES}
        self.buckets: dict[int, dict | None] = {m: None for m in MTF_MINUTES}
        self.state = {m: _blank() for m in MTF_MINUTES}

    def update(self, bar: dict) -> dict[int, dict]:
        try:
            stamp = int(bar.get("time") or 0)
        except (TypeError, ValueError):
            stamp = 0
        if stamp <= 0:
            return dict(self.state)
        high = float(bar.get("high") or bar.get("close") or 0.0)
        low = float(bar.get("low") or bar.get("close") or 0.0)
        close = float(bar.get("close") or 0.0)
        open_ = float(bar.get("open") or close)
        vol = bar.get("volume")
        for minutes in MTF_MINUTES:
            width = minutes * 60
            bucket = (stamp // width) * width
            acc = self.buckets[minutes]
            if acc is None:
                self.buckets[minutes] = {
                    "time": bucket, "open": open_, "high": high, "low": low, "close": close,
                    "volume": float(vol or 0.0),
                }
                continue
            if int(acc["time"]) != bucket:
                done = acc
                feat = self.inds[minutes].update(
                    done["high"], done["low"], done["close"],
                    volume=done.get("volume"), open_=done.get("open"), stamp=int(done["time"]),
                )
                self.state[minutes] = _snap(feat)
                self.buckets[minutes] = {
                    "time": bucket, "open": open_, "high": high, "low": low, "close": close,
                    "volume": float(vol or 0.0),
                }
            else:
                acc["high"] = max(float(acc["high"]), high)
                acc["low"] = min(float(acc["low"]), low)
                acc["close"] = close
                if vol:
                    acc["volume"] = float(acc.get("volume") or 0.0) + float(vol)
        return dict(self.state)


def mtf_agree(feat: dict | None, side: str) -> tuple[int, int]:
    """How many ready higher timeframes share ``side`` (BUY↔UP, SELL↔DOWN)."""
    want = "UP" if side == "BUY" else ("DOWN" if side == "SELL" else "")
    rows = (feat or {}).get("mtf") or {}
    ready = 0
    agree = 0
    for row in rows.values():
        if not isinstance(row, dict) or not row.get("ready"):
            continue
        ready += 1
        if want and str(row.get("regime") or "") == want:
            agree += 1
    return agree, ready


def mtf_allows(feat: dict | None, side: str) -> bool:
    """Majority of ready higher timeframes must match. Too few readings: allow."""
    agree, ready = mtf_agree(feat, side)
    if ready < 2:
        return True
    return agree * 2 >= ready


def mtf_scale(feat: dict | None, side: str) -> float:
    """1 when higher timeframes agree, down toward 0.4 when they do not."""
    agree, ready = mtf_agree(feat, side)
    if ready < 2:
        return 1.0
    frac = agree / ready
    return float(0.40 + 0.60 * frac)
