"""Technical strategy node per indicator category.

Each node reads the fused Kalman state plus the underlying raw indicators
and emits BUY / SELL / HOLD with a reason and strength in [0, 1].
Unavailable overlay categories (breadth, sentiment, volume without tick
volume) stay NEUTRAL with strength 0 — they never invent a side.
"""

from __future__ import annotations

from flyfx.sense.category_kalman import CATEGORY_NAMES


def _vote(side: str, reason: str, strength: float) -> dict:
    if side not in ("BUY", "SELL"):
        side = "HOLD"
    return {
        "vote": side,
        "reason": reason,
        "strength": float(max(0.0, min(1.0, strength))),
    }


def _hold(reason: str, strength: float = 0.0) -> dict:
    return _vote("HOLD", reason, strength)


def _cat(feat: dict, name: str) -> dict:
    return ((feat.get("categories") or {}).get(name)) or {}


def tech_trend(feat: dict) -> dict:
    """Trade with the prevailing trend. Stay in while ADX/supertrend hold; skip chop."""
    st = _cat(feat, "trend")
    adx = float(feat.get("adx") or 0.0)
    impulse = float(st.get("impulse") or 0.0)
    fused = float(st.get("fused") or 0.0)
    st_dir = int(feat.get("st_dir") or 0)
    ema_f = feat.get("ema_fast")
    ema_s = feat.get("ema_slow")
    sma_f = feat.get("sma_fast")
    sma_s = feat.get("sma_slow")
    regime = st.get("regime") or feat.get("regime") or "CHOP"
    if adx < 18.0 and abs(impulse) < 0.22:
        return _hold("trend-chop", 0.15)
    crossed_up = False
    crossed_dn = False
    if ema_f is not None and ema_s is not None:
        crossed_up = float(ema_f) > float(ema_s) and fused > 0.0
        crossed_dn = float(ema_f) < float(ema_s) and fused < 0.0
    if sma_f is not None and sma_s is not None:
        if float(sma_f) > float(sma_s):
            crossed_up = crossed_up or fused > 0.12
        if float(sma_f) < float(sma_s):
            crossed_dn = crossed_dn or fused < -0.12
    strength = min(1.0, 0.35 * abs(impulse) + 0.35 * min(adx / 40.0, 1.0) + 0.30 * min(abs(fused) / 1.2, 1.0))
    if (regime == "UP" or st_dir > 0 or crossed_up) and impulse > 0.12:
        if fused > 0.08 or crossed_up:
            return _vote("BUY", "trend-follow", max(0.35, strength))
    if (regime == "DOWN" or st_dir < 0 or crossed_dn) and impulse < -0.12:
        if fused < -0.08 or crossed_dn:
            return _vote("SELL", "trend-follow", max(0.35, strength))
    fib = feat.get("fib") if isinstance(feat.get("fib"), dict) else {}
    lvl = str((fib or {}).get("near") or "")
    if feat.get("fib_trade", True) and lvl and "swing_up" in (fib or {}):
        bar_pos = float(feat.get("bar_pos") or 0.5)
        if regime == "UP" and bool(fib.get("swing_up")) and bar_pos >= 0.55:
            return _vote("BUY", f"trend-fib-{lvl}", max(0.42, strength))
        if regime == "DOWN" and not bool(fib.get("swing_up")) and bar_pos <= 0.45:
            return _vote("SELL", f"trend-fib-{lvl}", max(0.42, strength))
    return _hold("trend-wait", 0.20)


def tech_momentum(feat: dict) -> dict:
    """Reversal when stretched; divergence vs price; prefers range-bound ADX."""
    st = _cat(feat, "momentum")
    if not st.get("available") and float(st.get("n_live") or 0) <= 0:
        return _hold("momentum-cold", 0.05)
    rsi = float(feat.get("rsi") or 50.0)
    stoch = float(feat.get("stoch") or 0.50)
    willr = float(feat.get("willr") or -50.0)
    cci = float(feat.get("cci") or 0.0)
    fused = float(st.get("fused") or 0.0)
    residual = float(st.get("residual") or 0.0)
    adx = float(feat.get("adx") or 20.0)
    range_fit = max(0.25, min(1.0, (32.0 - adx) / 20.0))
    oversold = rsi <= 32.0 or stoch <= 0.20 or willr <= -80.0 or cci <= -100.0
    overbought = rsi >= 68.0 or stoch >= 0.80 or willr >= -20.0 or cci >= 100.0
    close = float(feat.get("close") or 0.0)
    hi55 = feat.get("fib", {}).get("hi") if isinstance(feat.get("fib"), dict) else None
    lo55 = feat.get("fib", {}).get("lo") if isinstance(feat.get("fib"), dict) else None
    diverg_sell = False
    diverg_buy = False
    if hi55 is not None and close >= float(hi55) * 0.999 and fused < 0.05:
        diverg_sell = True
    if lo55 is not None and close <= float(lo55) * 1.001 and fused > -0.05:
        diverg_buy = True
    strength = min(1.0, range_fit * (0.45 + 0.35 * abs(fused) + 0.20 * abs(residual)))
    if (oversold or diverg_buy or residual < -0.18) and fused <= 0.10:
        return _vote("BUY", "momentum-oversold" if oversold else "momentum-divergence", max(0.30, strength))
    if (overbought or diverg_sell or residual > 0.18) and fused >= -0.10:
        return _vote("SELL", "momentum-overbought" if overbought else "momentum-divergence", max(0.30, strength))
    return _hold("momentum-mid", 0.20 * range_fit)


def tech_volatility(feat: dict) -> dict:
    """Breakout after a squeeze; stand aside when ATR is already extreme."""
    st = _cat(feat, "volatility")
    atr_ratio = float(feat.get("atr_ratio") or 1.0)
    if atr_ratio >= 1.85:
        return _hold("vol-extreme", 0.10)
    bb_pct = float(feat.get("bb_pct") or 0.50)
    bb_bw = float(feat.get("bb_bw") or 0.0)
    bb_bw_ma = float(feat.get("bb_bw_ma") or bb_bw or 1.0)
    don_pct = float(feat.get("donchian_pct") or 0.50)
    fused = float(st.get("fused") or 0.0)
    squeeze = bb_bw_ma > 0 and bb_bw <= 0.78 * bb_bw_ma
    strength = min(1.0, 0.40 * abs(fused) + 0.35 * (1.0 if squeeze else 0.25) + 0.25 * min(atr_ratio, 1.4) / 1.4)
    if squeeze and (bb_pct >= 0.82 or don_pct >= 0.92) and fused >= 0.0:
        return _vote("BUY", "vol-squeeze-break", max(0.40, strength))
    if squeeze and (bb_pct <= 0.18 or don_pct <= 0.08) and fused <= 0.0:
        return _vote("SELL", "vol-squeeze-break", max(0.40, strength))
    if bb_pct >= 0.92 and fused > 0.20 and atr_ratio < 1.55:
        return _vote("BUY", "vol-band-walk", 0.45 * strength)
    if bb_pct <= 0.08 and fused < -0.20 and atr_ratio < 1.55:
        return _vote("SELL", "vol-band-walk", 0.45 * strength)
    return _hold("vol-wait", 0.15)


def tech_volume(feat: dict) -> dict:
    """Confirm or invalidate the price move. Signed vote, but a filter by role."""
    st = _cat(feat, "volume")
    if not st.get("available"):
        return _hold("volume-unavailable", 0.0)
    fused = float(st.get("fused") or 0.0)
    impulse = float(st.get("impulse") or 0.0)
    # price direction from signed_vol / mom
    px = float(feat.get("signed_vol") or 0.0)
    if px == 0.0:
        px = float(feat.get("mom") or 0.0)
    cmf = float(feat.get("cmf") or 0.0)
    strength = min(1.0, 0.50 * abs(impulse) + 0.30 * abs(fused) + 0.20 * min(abs(cmf) / 0.25, 1.0))
    if px > 0.08 and fused > 0.08:
        return _vote("BUY", "volume-confirm-up", max(0.30, strength))
    if px < -0.08 and fused < -0.08:
        return _vote("SELL", "volume-confirm-down", max(0.30, strength))
    if px > 0.12 and fused < -0.05:
        return _vote("SELL", "volume-fail-up", max(0.28, 0.70 * strength))
    if px < -0.12 and fused > 0.05:
        return _vote("BUY", "volume-fail-down", max(0.28, 0.70 * strength))
    return _hold("volume-quiet", 0.15)


def tech_breadth(feat: dict) -> dict:
    """Broad participation confirms; headline-only moves are caution/sell."""
    st = _cat(feat, "breadth")
    if not st.get("available"):
        return _hold("breadth-unavailable", 0.0)
    fused = float(st.get("fused") or 0.0)
    impulse = float(st.get("impulse") or 0.0)
    strength = min(1.0, 0.55 * abs(impulse) + 0.45 * min(abs(fused) / 1.0, 1.0))
    if impulse > 0.18:
        return _vote("BUY", "breadth-participate", max(0.30, strength))
    if impulse < -0.18:
        return _vote("SELL", "breadth-narrow", max(0.30, strength))
    return _hold("breadth-flat", 0.12)


def tech_structure(feat: dict) -> dict:
    """Bounce at support / reject at resistance, or a confirmed break."""
    st = _cat(feat, "structure")
    piv = feat.get("pivot") or {}
    fib = feat.get("fib") or {}
    close = float(feat.get("close") or 0.0)
    atr = max(float(feat.get("atr") or 1e-9), 1e-9)
    bar_pos = float(feat.get("bar_pos") or 0.50)
    fused = float(st.get("fused") or 0.0)
    impulse = float(st.get("impulse") or 0.0)
    near = str(piv.get("near") or "")
    dist = float(piv.get("dist_atr") or 9.0)
    fib_pos = fib.get("pos")
    vol_st = _cat(feat, "volume")
    mom_st = _cat(feat, "momentum")
    confirm = 0.0
    if vol_st.get("available"):
        confirm += 0.12 * (1.0 if (float(vol_st.get("impulse") or 0.0)) * (1.0 if fused >= 0 else -1.0) > 0 else -0.4)
    if mom_st.get("available"):
        confirm += 0.08 * (1.0 if (float(mom_st.get("impulse") or 0.0)) * (1.0 if fused >= 0 else -1.0) > 0 else 0.0)
    strength = min(1.0, 0.40 * abs(impulse) + 0.30 * max(0.0, 1.0 - dist / 1.2) + 0.30 * abs(fused) + confirm)

    s1 = piv.get("s1")
    r1 = piv.get("r1")
    bounce_buy = near.startswith("s") and dist <= 0.45 and bar_pos >= 0.55
    bounce_sell = near.startswith("r") and dist <= 0.45 and bar_pos <= 0.45
    if isinstance(fib_pos, (int, float)):
        if float(fib_pos) <= 0.40 and bar_pos >= 0.55:
            bounce_buy = True
        if float(fib_pos) >= 0.60 and bar_pos <= 0.45:
            bounce_sell = True
    break_up = r1 is not None and close > float(r1) + 0.12 * atr and bar_pos >= 0.60 and fused > 0.05
    break_dn = s1 is not None and close < float(s1) - 0.12 * atr and bar_pos <= 0.40 and fused < -0.05
    if bounce_buy or break_up:
        why = "structure-break-up" if break_up else "structure-bounce"
        return _vote("BUY", why, max(0.32, strength))
    if bounce_sell or break_dn:
        why = "structure-break-down" if break_dn else "structure-reject"
        return _vote("SELL", why, max(0.32, strength))
    if abs(impulse) > 0.28 and abs(fused) > 0.22:
        return _vote("BUY" if impulse > 0 else "SELL", "structure-channel", 0.40 * strength)
    return _hold("structure-mid", 0.15)


def tech_sentiment(feat: dict) -> dict:
    """Contrarian at extremes: buy fear, sell greed. Macro overlay."""
    st = _cat(feat, "sentiment")
    if not st.get("available"):
        return _hold("sentiment-unavailable", 0.0)
    fused = float(st.get("fused") or 0.0)
    impulse = float(st.get("impulse") or 0.0)
    strength = min(1.0, 0.50 * abs(impulse) + 0.50 * min(abs(fused), 1.0))
    # measurements are already signed contrarian (high VIX / put-call → +z → BUY)
    if impulse > 0.22:
        return _vote("BUY", "sentiment-fear", max(0.30, strength))
    if impulse < -0.22:
        return _vote("SELL", "sentiment-greed", max(0.30, strength))
    return _hold("sentiment-neutral", 0.10)


TECH_FNS = {
    "trend": tech_trend,
    "momentum": tech_momentum,
    "volatility": tech_volatility,
    "volume": tech_volume,
    "breadth": tech_breadth,
    "structure": tech_structure,
    "sentiment": tech_sentiment,
}


def run_category_tech(feat: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name in CATEGORY_NAMES:
        out[name] = TECH_FNS[name](feat)
    return out


def vol_size_mult(feat: dict, latent=None) -> float:
    """Defensive lot scale when volatility is already extreme (ATR path stays)."""
    from flyfx.risk.latent import vol_factor

    st = _cat(feat, "volatility")
    return vol_factor(
        float(feat.get("atr_ratio") or 1.0),
        float(st.get("uncertainty") or 0.0),
        latent,
    )
