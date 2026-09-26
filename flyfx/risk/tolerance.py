"""Operator risk-tolerance profiles that change how open trades get closed.

``balanced`` is the factory geometry (same SL/TP/trail/hold as AdaptiveParams
defaults). Conservative and aggressive overlay those numbers and turn extra
risk-factor exits on or off. ``scalp`` is the M5 retail companion that restores
the ~4k cat-fuse book (factory geometry + snowball + ~12% VOL) — it does not
change the ``--legacy-2oo3 --risk-tol balanced`` freeze path.
"""

from __future__ import annotations

from flyfx.risk.bayes_sizer import PARAM_BOUNDS

NAMES = ("conservative", "balanced", "aggressive", "scalp")

GEOM_KEYS = (
    "sl_atr",
    "tp_atr",
    "trail_arm_atr",
    "trail_gap_atr",
    "be_atr",
    "min_hold_bars",
    "max_hold_bars",
)

# Bounds for scalp overlay — factory-range so the 4k companion geom is not clamped.
SCALP_BOUNDS: dict[str, tuple[float, float]] = {
    "sl_atr": (1.80, 3.20),
    "tp_atr": (8.0, 15.0),
    "trail_arm_atr": (2.0, 4.50),
    "trail_gap_atr": (1.20, 2.50),
    "be_atr": (1.0, 2.0),
    "min_hold_bars": (3.0, 8.0),
    "max_hold_bars": (40.0, 80.0),
}

# Companion package applied when RISK = scalp (CLI/GUI may still override after).
# Restores the cat-fuse blank-FUSE ~4k EURUSD M5 book: factory impulse/cooldown,
# snowball on, bank/appetite off, stop floor 8 pips. VOL profile is 12%.
SCALP_MODE: dict = {
    "cooldown_bars": 8.0,
    "min_impulse": 0.50,
    "cont_impulse": 0.75,
    "lock_win_pips": 8.0,
    "bank_pct": 0.0,
    "hold_risk": True,  # giveback flatten only when bank is 0 (same as factory CLI)
    "hold_risk_sens": 1.0,
    "no_snowball": False,
    "entry_appetite": False,
    "trade_rate": None,
    "min_stop_pips": 8.0,
}

_ALIASES = {
    "conservative": "conservative",
    "cons": "conservative",
    "tight": "conservative",
    "low": "conservative",
    "safe": "conservative",
    "balanced": "balanced",
    "balance": "balanced",
    "medium": "balanced",
    "normal": "balanced",
    "default": "balanced",
    "mid": "balanced",
    "aggressive": "aggressive",
    "agg": "aggressive",
    "high": "aggressive",
    "loose": "aggressive",
    "risk-on": "aggressive",
    "riskon": "aggressive",
    "scalp": "scalp",
    "scalping": "scalp",
    "fast": "scalp",
    "m5-scalp": "scalp",
    "m5scalp": "scalp",
}

PROFILES: dict[str, dict] = {
    "conservative": {
        "name": "conservative",
        "sl_atr": 1.90,
        "tp_atr": 8.0,
        "trail_arm_atr": 2.30,
        "trail_gap_atr": 1.30,
        "be_atr": 1.10,
        "min_hold_bars": 3.0,
        "max_hold_bars": 40.0,
        "chop_unreal_atr": 0.40,
        "banc_exit": 0.72,
        "uncert_exit": 0.78,
        "fade_exit": True,
        "atr_blow_exit": 1.85,
        "kill_flatten": True,
        "risk_scale": 0.70,
        "kill_dd": 0.015,
        "protect_dd": 0.008,
        "day_losses": 1,
        "day_trades": 2,
    },
    "balanced": {
        "name": "balanced",
        "sl_atr": 2.40,
        "tp_atr": 12.0,
        "trail_arm_atr": 3.20,
        "trail_gap_atr": 1.80,
        "be_atr": 1.50,
        "min_hold_bars": 4.0,
        "max_hold_bars": 64.0,
        "chop_unreal_atr": 0.25,
        "banc_exit": 0.0,
        "uncert_exit": 0.0,
        "fade_exit": False,
        "atr_blow_exit": 0.0,
        "kill_flatten": False,
        "risk_scale": 1.0,
        "kill_dd": 0.025,
        "protect_dd": 0.010,
        "day_losses": 2,
        "day_trades": 3,
    },
    "aggressive": {
        "name": "aggressive",
        "sl_atr": 3.10,
        "tp_atr": 15.0,
        "trail_arm_atr": 4.20,
        "trail_gap_atr": 2.40,
        "be_atr": 2.00,
        "min_hold_bars": 6.0,
        "max_hold_bars": 80.0,
        "chop_unreal_atr": 0.10,
        "banc_exit": 0.0,
        "uncert_exit": 0.0,
        "fade_exit": False,
        "atr_blow_exit": 0.0,
        "kill_flatten": False,
        "risk_scale": 1.25,
        "kill_dd": 0.040,
        "protect_dd": 0.018,
        "day_losses": 3,
        "day_trades": 5,
    },
    # 4k cat-fuse companion: same exit geometry / kill rails as balanced factory.
    # VOL profile (12%) + snowball companions live in SCALP_MODE / VOLUME_BY_RISK.
    "scalp": {
        "name": "scalp",
        "sl_atr": 2.40,
        "tp_atr": 12.0,
        "trail_arm_atr": 3.20,
        "trail_gap_atr": 1.80,
        "be_atr": 1.50,
        "min_hold_bars": 4.0,
        "max_hold_bars": 64.0,
        "chop_unreal_atr": 0.25,
        "banc_exit": 0.0,
        "uncert_exit": 0.0,
        "fade_exit": False,
        "atr_blow_exit": 0.0,
        "kill_flatten": False,
        "risk_scale": 1.0,
        "kill_dd": 0.025,
        "protect_dd": 0.010,
        "day_losses": 2,
        "day_trades": 6,
    },
}


def parse_risk_tol(spec: str | None) -> str:
    text = str(spec or "").strip().lower().replace("_", "-")
    if not text:
        return "balanced"
    name = _ALIASES.get(text, "")
    if not name:
        raise ValueError(
            f"unknown risk tolerance {spec!r}; use conservative, balanced, aggressive, scalp"
        )
    return name


def risk_profile(spec: str | None) -> dict:
    try:
        name = parse_risk_tol(spec)
    except ValueError:
        name = "balanced"
    return dict(PROFILES[name])


def overlay_geom(geom: dict | None, profile: dict | None) -> dict:
    """Replace exit geometry when not balanced. Balanced leaves AdaptiveParams as-is."""
    out = dict(geom or {})
    prof = profile or PROFILES["balanced"]
    name = str(prof.get("name") or "balanced")
    if name == "balanced":
        return out
    bounds = SCALP_BOUNDS if name == "scalp" else PARAM_BOUNDS
    for key in GEOM_KEYS:
        if key not in prof:
            continue
        lo, hi = bounds.get(key, (float(prof[key]), float(prof[key])))
        val = float(prof[key])
        if val < lo:
            val = lo
        if val > hi:
            val = hi
        out[key] = val
    return out


def apply_scalp_companions(args, params) -> None:
    """Fill scalp package onto Namespace + AdaptiveParams (idempotent)."""
    if bool(getattr(args, "_scalp_companions", False)):
        return
    mode = SCALP_MODE
    from flyfx.risk.bank import parse_bank_pct, parse_hold_risk_sens
    from flyfx.risk.appetite import parse_trade_rate

    if parse_bank_pct(getattr(args, "bank_pct", 0.0)) <= 0:
        args.bank_pct = float(mode["bank_pct"])
    if getattr(args, "hold_risk", None) is None:
        args.hold_risk = bool(mode["hold_risk"])
    else:
        args.hold_risk = bool(args.hold_risk)
    if abs(parse_hold_risk_sens(getattr(args, "hold_risk_sens", 1.0)) - 1.0) < 1e-9:
        args.hold_risk_sens = float(mode["hold_risk_sens"])
    # Snowball stays on (4k book) unless CLI already asked for --no-snowball.
    if not bool(getattr(args, "no_snowball", False)):
        args.no_snowball = bool(mode["no_snowball"])
    if not bool(getattr(args, "entry_appetite", False)):
        args.entry_appetite = bool(mode["entry_appetite"])
    if mode.get("trade_rate") is not None and parse_trade_rate(
        getattr(args, "trade_rate", None)
    ) is None:
        args.trade_rate = float(mode["trade_rate"])
    if params is not None:
        # Bypass PARAM_BOUNDS floors via direct setattr for scalp geom.
        for key, val in (
            ("cooldown_bars", mode["cooldown_bars"]),
            ("min_impulse", mode["min_impulse"]),
            ("cont_impulse", mode["cont_impulse"]),
            ("lock_win_pips", mode["lock_win_pips"]),
            ("min_hold_bars", PROFILES["scalp"]["min_hold_bars"]),
            ("max_hold_bars", PROFILES["scalp"]["max_hold_bars"]),
            ("sl_atr", PROFILES["scalp"]["sl_atr"]),
            ("tp_atr", PROFILES["scalp"]["tp_atr"]),
            ("trail_arm_atr", PROFILES["scalp"]["trail_arm_atr"]),
            ("trail_gap_atr", PROFILES["scalp"]["trail_gap_atr"]),
            ("be_atr", PROFILES["scalp"]["be_atr"]),
        ):
            setattr(params, key, float(val))
    args._scalp_companions = True


def risk_close_reason(
    profile: dict | None,
    *,
    side: str,
    bars_held: int,
    feat: dict | None,
    banc_mult: float,
    fade: str,
    rails_kill: bool,
    unreal: float,
    atr: float,
    equity: float = 0.0,
    peak_eq: float = 0.0,
) -> str | None:
    """Discretionary flatten from BANC / fade / uncertainty / vol / kill.

    Returns a close reason or None. Never fires on the first ``min_hold`` bars
    except kill-flatten. Balanced profile returns None (old close set only).
    """
    prof = profile or PROFILES["balanced"]
    feat = feat or {}
    atr = max(float(atr or 0.0), 1e-12)
    min_hold = int(round(float(prof.get("min_hold_bars") or 4.0)))
    if bool(prof.get("kill_flatten")):
        if rails_kill:
            return "risk kill"
        lim = float(prof.get("kill_dd") or 0.0)
        pk = float(peak_eq or 0.0)
        eq = float(equity or 0.0)
        if lim > 0 and pk > 0 and eq <= pk * (1.0 - lim):
            return "risk kill"
    if bars_held < min_hold:
        return None
    banc_bar = float(prof.get("banc_exit") or 0.0)
    if banc_bar > 0 and float(banc_mult) < banc_bar:
        return f"risk BANC {float(banc_mult):.2f}<{banc_bar:.2f}"
    if bool(prof.get("fade_exit")):
        if side == "BUY" and str(fade).upper() == "SELL":
            return "risk fade"
        if side == "SELL" and str(fade).upper() == "BUY":
            return "risk fade"
    uncert_bar = float(prof.get("uncert_exit") or 0.0)
    uncert = float((feat.get("kalman") or {}).get("uncertainty") or 0.0)
    be_atr = float(prof.get("be_atr") or 1.5)
    if uncert_bar > 0 and uncert >= uncert_bar and float(unreal) < be_atr * atr:
        return f"risk uncertainty {uncert:.2f}"
    blow = float(prof.get("atr_blow_exit") or 0.0)
    atr_ratio = float(feat.get("atr_ratio") or 1.0)
    if blow > 0 and atr_ratio >= blow and float(unreal) < 0.50 * atr:
        return f"risk vol spike ATR×{atr_ratio:.2f}"
    return None
