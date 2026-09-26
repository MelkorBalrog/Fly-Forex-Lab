"""Nine diverse voters: three 3oo3 families, then 2oo3 of the family votes.

Families (each must be unanimous to speak):
  structure  — pullback timing, EMA hold, rejection candle
  momentum   — fly-trend, Kalman flow, impulse sign
  fade       — fly-fade, RSI stretch, Kalman residual

The outer 2oo3 is the same majority+veto rule as the old {tech, trend, fade}
committee, but each seat is harder to fill. Any of the nine nodes voting the
opposite side still vetoes (a HOLD family does not). Tags stay `2oo3` / `3oo3`
so sizing, martingale, and the wide-spread gate keep working.
"""

from __future__ import annotations


def unanimous3(a: str, b: str, c: str) -> str:
    if a in ("BUY", "SELL") and a == b == c:
        return a
    return "HOLD"


def vote_ema_structure(feat: dict, close: float) -> str:
    """Price holding the slow EMA in the Kalman regime (trend structure)."""
    ema_s = float(feat.get("ema_slow") or close)
    regime = feat.get("regime")
    if regime == "UP" and close >= ema_s:
        return "BUY"
    if regime == "DOWN" and close <= ema_s:
        return "SELL"
    return "HOLD"


def vote_candle(feat: dict, close: float) -> str:
    """Close location in the bar: buyers defended the low, or sellers the high."""
    high = float(feat.get("bar_high") or close)
    low = float(feat.get("bar_low") or close)
    rng = high - low
    if rng <= 1e-12:
        return "HOLD"
    pos = (close - low) / rng
    regime = feat.get("regime")
    if regime == "UP" and pos >= 0.55:
        return "BUY"
    if regime == "DOWN" and pos <= 0.45:
        return "SELL"
    return "HOLD"


def vote_impulse(feat: dict) -> str:
    """Raw Kalman impulse sign — no regime check (unlike flow_algo)."""
    k = float(feat.get("impulse") or 0.0)
    if k >= 0.40:
        return "BUY"
    if k <= -0.40:
        return "SELL"
    return "HOLD"


def vote_rsi_stretch(feat: dict, params=None) -> str:
    """RSI extreme: buy stretch-down, sell stretch-up (fade family)."""
    rsi = float(feat.get("rsi") or 50.0)
    buy_arm = float(getattr(params, "rsi_buy_arm", 52.0) if params is not None else 52.0)
    sell_arm = float(getattr(params, "rsi_sell_arm", 48.0) if params is not None else 48.0)
    if rsi <= min(42.0, buy_arm):
        return "BUY"
    if rsi >= max(58.0, sell_arm):
        return "SELL"
    return "HOLD"


def vote_residual(feat: dict) -> str:
    """Kalman residual: buy when price is below the filter, sell above."""
    r = float((feat.get("kalman") or {}).get("residual") or 0.0)
    if r <= -0.10:
        return "BUY"
    if r >= 0.10:
        return "SELL"
    return "HOLD"


def pack_nodes(
    feat: dict,
    close: float,
    tech: str,
    fly_trend: str,
    fly_fade: str,
    flow: str,
    params=None,
) -> dict[str, str]:
    return {
        "tech": tech if tech in ("BUY", "SELL", "HOLD") else "HOLD",
        "ema": vote_ema_structure(feat, close),
        "candle": vote_candle(feat, close),
        "trend": fly_trend if fly_trend in ("BUY", "SELL", "HOLD") else "HOLD",
        "flow": flow if flow in ("BUY", "SELL", "HOLD") else "HOLD",
        "impulse": vote_impulse(feat),
        "fade": fly_fade if fly_fade in ("BUY", "SELL", "HOLD") else "HOLD",
        "rsi": vote_rsi_stretch(feat, params),
        "residual": vote_residual(feat),
    }


def nested_committee(nodes: dict, kind: str, flow: str) -> tuple[str, str, int, dict]:
    """3 families of 3oo3, then 2oo3 of {structure, momentum, fade}."""
    structure = unanimous3(nodes["tech"], nodes["ema"], nodes["candle"])
    momentum = unanimous3(nodes["trend"], nodes["flow"], nodes["impulse"])
    fade = unanimous3(nodes["fade"], nodes["rsi"], nodes["residual"])
    families = {"structure": structure, "momentum": momentum, "fade": fade}
    pack = {"families": families, "nodes": dict(nodes)}

    buy_n = sum(1 for v in families.values() if v == "BUY")
    sell_n = sum(1 for v in families.values() if v == "SELL")
    if buy_n >= 1 and sell_n >= 1:
        opp = "SELL" if buy_n >= sell_n else "BUY"
        return "HOLD", f"veto family {opp}", 0, pack

    side = "BUY" if buy_n >= sell_n else "SELL"
    agree = buy_n if side == "BUY" else sell_n
    if agree >= 2:
        opp = "SELL" if side == "BUY" else "BUY"
        dissent = [name for name, v in nodes.items() if v == opp]
        if dissent:
            return "HOLD", f"veto {opp} ({','.join(dissent)})", 0, pack
    if agree >= 3:
        tag = "3oo3+flow" if flow == side else "3oo3"
        return side, tag, agree, pack
    if agree >= 2:
        tag = "2oo3+flow" if flow == side else "2oo3"
        return side, tag, agree, pack
    if structure in ("BUY", "SELL") and (kind == "bounce" or str(kind).startswith("fib-")):
        tag = "1oo3-fib" if str(kind).startswith("fib-") else "1oo3-bounce"
        return structure, tag, 1, pack
    if structure in ("BUY", "SELL"):
        return "HOLD", f"1oo3-skip ({kind})", 1, pack
    return "HOLD", "no-family", 0, pack
