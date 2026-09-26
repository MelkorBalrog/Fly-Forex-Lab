"""Factory size, circuit, and snowball numbers.

Live size does not come from a per-pair fit unless ``--apply-latent-live``.
``flyfx.risk.conditions`` scales the stated risk from the current spread,
volatility, session, chase, and heat. Sim Replay applies fingerprint-matched
``risk_factors`` (from auto/explicit recalibrate) via ``fit_latent``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, replace

MIN_TRADES = 6


def _clip(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@dataclass
class LatentRisk:
    """Factory values match the literals that shipped before this fitter."""

    bayes_lo: float = 0.35
    bayes_hi: float = 1.00
    kelly_div: float = 0.06
    edge_base: float = 0.45
    edge_gain: float = 0.70
    rr_div: float = 1.35
    rr_lo: float = 0.40
    rr_hi: float = 1.15
    conf_base: float = 0.78
    conf_slope: float = 0.010
    conf_lo: float = 0.72
    conf_hi: float = 1.22
    cal_bar: float = 0.38
    cal_haircut: float = 0.85
    similar_w: float = 1.20
    similar_bar: float = 0.38
    mult_3: float = 1.18
    flow_3: float = 1.06
    mult_2: float = 1.00
    flow_2: float = 1.05
    mult_bounce: float = 0.72
    bounce_conf: float = 52.0
    vol_soft_atr: float = 1.45
    vol_hard_atr: float = 1.85
    vol_soft: float = 0.88
    vol_hard: float = 0.75
    vol_uncert: float = 0.85
    size_floor: float = 0.45
    banc_floor: float = 0.50
    banc_soft: float = 0.90
    banc_cut: float = 1.00
    recover_floor: float = 0.50
    admit_floor: float = 0.45
    sugar_floor: float = 0.55
    risk_cap: float = 1.80
    lock_keep: float = 0.90
    recover_dd: float = 0.020
    recover_banc: float = 0.90
    recover_freeze: float = 0.85
    scratch_r: float = 0.45
    mart_1: float = 1.20
    mart_2: float = 1.28
    circuit_n: int = 3
    circuit_bars: int = 12
    admit_n: int = 2
    admit_pf: float = 1.00
    admit_mult: float = 0.60
    snow_frac: float = 0.55
    snow_banc: float = 0.85
    snow_cushion: float = 1.40
    snow_impulse: float = 0.45
    snow_pause: int = 4
    snow_max: int = 2
    cost_stop: float = 0.45
    cost_tp: float = 1.80
    note: str = ""
    n_fit: int = 0


_INTS = {"circuit_n", "circuit_bars", "admit_n", "snow_pause", "snow_max", "n_fit"}

_BOUNDS: dict[str, tuple[float, float]] = {
    "bayes_lo": (0.15, 0.60),
    "bayes_hi": (0.50, 1.25),
    "kelly_div": (0.04, 0.10),
    "edge_base": (0.30, 0.60),
    "edge_gain": (0.40, 1.00),
    "rr_div": (1.0, 2.0),
    "rr_lo": (0.25, 0.60),
    "rr_hi": (1.0, 1.40),
    "conf_base": (0.60, 0.95),
    "conf_slope": (0.004, 0.020),
    "conf_lo": (0.40, 0.90),
    "conf_hi": (1.00, 1.40),
    "cal_bar": (0.25, 0.55),
    "cal_haircut": (0.60, 1.00),
    "similar_w": (0.60, 2.00),
    "similar_bar": (0.22, 0.55),
    "mult_3": (0.70, 1.50),
    "flow_3": (0.90, 1.25),
    "mult_2": (0.40, 1.30),
    "flow_2": (0.90, 1.20),
    "mult_bounce": (0.35, 1.00),
    "bounce_conf": (45.0, 70.0),
    "vol_soft_atr": (1.20, 1.80),
    "vol_hard_atr": (1.50, 2.40),
    "vol_soft": (0.60, 1.00),
    "vol_hard": (0.45, 0.95),
    "vol_uncert": (0.70, 0.95),
    "size_floor": (0.15, 0.70),
    "banc_floor": (0.30, 0.80),
    "banc_soft": (0.70, 1.05),
    "banc_cut": (0.45, 1.20),
    "recover_floor": (0.30, 0.80),
    "admit_floor": (0.25, 0.80),
    "sugar_floor": (0.35, 0.80),
    "risk_cap": (0.80, 2.50),
    "lock_keep": (0.40, 1.00),
    "recover_dd": (0.008, 0.050),
    "recover_banc": (0.70, 1.05),
    "recover_freeze": (0.60, 1.00),
    "scratch_r": (0.15, 0.80),
    "mart_1": (1.00, 1.45),
    "mart_2": (1.00, 1.55),
    "circuit_n": (2, 6),
    "circuit_bars": (4, 36),
    "admit_n": (1, 6),
    "admit_pf": (0.70, 1.40),
    "admit_mult": (0.30, 1.00),
    "snow_frac": (0.25, 0.80),
    "snow_banc": (0.70, 1.05),
    "snow_cushion": (1.05, 2.50),
    "snow_impulse": (0.25, 0.75),
    "snow_pause": (2, 12),
    "snow_max": (1, 3),
    "cost_stop": (0.25, 0.70),
    "cost_tp": (1.2, 3.0),
}


FACTORY = LatentRisk()


def _bound(name: str, value: float) -> float:
    lo, hi = _BOUNDS.get(name, (value, value))
    out = _clip(float(value), lo, hi)
    if name in _INTS and name != "n_fit":
        return int(round(out))
    return out


def clamp_latent(lat: LatentRisk) -> LatentRisk:
    data = asdict(lat)
    for name in _BOUNDS:
        data[name] = _bound(name, data[name])
    if data["bayes_lo"] >= data["bayes_hi"]:
        data["bayes_lo"] = min(data["bayes_lo"], data["bayes_hi"] - 0.05)
        data["bayes_lo"] = _bound("bayes_lo", data["bayes_lo"])
    if data["vol_soft_atr"] > data["vol_hard_atr"]:
        data["vol_soft_atr"] = data["vol_hard_atr"]
    if data["mart_2"] < data["mart_1"]:
        data["mart_2"] = data["mart_1"]
    data["note"] = str(lat.note or "")
    data["n_fit"] = int(lat.n_fit or 0)
    return LatentRisk(**data)


def is_factory(lat: LatentRisk | None) -> bool:
    if lat is None:
        return True
    for f in fields(LatentRisk):
        if f.name in ("note", "n_fit"):
            continue
        a = getattr(lat, f.name)
        b = getattr(FACTORY, f.name)
        if isinstance(b, float):
            if abs(float(a) - float(b)) > 1e-9:
                return False
        elif a != b:
            return False
    return True


def adopt(dst: LatentRisk, src: LatentRisk) -> None:
    for f in fields(LatentRisk):
        setattr(dst, f.name, getattr(src, f.name))


def dump_latent(lat: LatentRisk) -> dict:
    out = asdict(lat)
    out["n_fit"] = int(lat.n_fit or 0)
    return out


def load_latent(data: dict | None) -> LatentRisk:
    if not isinstance(data, dict):
        return LatentRisk()
    base = asdict(FACTORY)
    for name in base:
        if name not in data:
            continue
        base[name] = data[name]
    try:
        lat = LatentRisk(**base)
    except TypeError:
        return LatentRisk()
    return clamp_latent(lat)


def describe_latent(lat: LatentRisk) -> str:
    if is_factory(lat):
        return "factory"
    return (
        f"2oo3×{lat.mult_2:.2f}  3oo3×{lat.mult_3:.2f}  "
        f"bayes {lat.bayes_lo:.2f}-{lat.bayes_hi:.2f}  "
        f"circuit {int(lat.circuit_n)}/{int(lat.circuit_bars)}  "
        f"banc-cut {lat.banc_cut:.2f}  snow {lat.snow_frac:.2f}  "
        f"n={int(lat.n_fit)}"
    )


def vol_factor(atr_ratio: float, uncert: float = 0.0, lat: LatentRisk | None = None) -> float:
    """Lot haircut when ATR is already stretched. Factory: 0.75 / 0.88 / 1."""
    lat = lat or FACTORY
    ratio = float(atr_ratio or 1.0)
    if ratio >= float(lat.vol_hard_atr) or float(uncert or 0.0) > float(lat.vol_uncert):
        return float(lat.vol_hard)
    if ratio >= float(lat.vol_soft_atr):
        return float(lat.vol_soft)
    return 1.0


def bayes_scale_from_parts(parts: dict | None, lat: LatentRisk | None = None) -> float:
    """Outer half-Kelly scale. Factory clip is [0.35, 1.00]."""
    lat = lat or FACTORY
    parts = parts or {}
    if "kelly_raw" not in parts:
        return float(parts.get("bayes_scale") or 1.0)
    kelly = _clip(float(parts["kelly_raw"]), lat.bayes_lo, lat.bayes_hi)
    rr = _clip(float(parts.get("rr_raw") or 1.0), lat.rr_lo, lat.rr_hi)
    edge = float(parts.get("edge") or 1.0)
    return _clip(kelly * edge * rr, lat.bayes_lo, lat.bayes_hi)


def hidden_size(
    scored: dict,
    tag: str,
    lat: LatentRisk | None = None,
    *,
    threshold: float = 45.0,
) -> tuple[bool, str, float]:
    """Committee size multiplier. Same branches as the factory gate."""
    lat = lat or FACTORY
    tag = str(tag or "")
    conf = float(scored.get("conf") or 0.0)
    cal = float(scored.get("cal_p", 0.5) or 0.5)
    n = int(scored.get("n", 0) or 0)
    conf_mult = _clip(lat.conf_base + (conf - 45.0) * lat.conf_slope, lat.conf_lo, lat.conf_hi)
    if n >= 6 and cal < lat.cal_bar:
        conf_mult *= lat.cal_haircut
    similar = float(scored.get("similar", 0.5) or 0.5)
    similar_w = float(scored.get("similar_w", 0.0) or 0.0)
    if n >= 5 and similar_w >= lat.similar_w and similar < lat.similar_bar:
        return False, f"skip {tag}: similar setups win {100.0 * similar:.0f}%", 0.0
    if tag.startswith("3oo3") or tag.startswith("4oo4") or tag.startswith("3oo4"):
        # 3oo4 / 4oo4 use the strong-agreement size tier (same as 3oo3).
        mult = lat.mult_3 * conf_mult
        if tag.endswith("+flow"):
            mult *= lat.flow_3
        extra = _clip(float(scored.get("ens_mult") or 1.0), 0.70, 1.40)
        return True, f"{tag} size×{mult * extra:.2f}  conf {conf:.0f}%", mult * extra
    if "oo9" in tag and tag[:1].isdigit():
        k = int(tag[0])
        if k < 5:
            return False, f"skip {tag}: need 5oo9–9oo9", 0.0
        k_mult = {5: 0.72, 6: 0.95, 7: 1.15, 8: 1.32, 9: 1.50}.get(k, 0.72)
        extra = float(scored.get("koo9_mult") or k_mult)
        mult = extra * conf_mult
        return True, f"{tag} size×{mult:.2f}  conf {conf:.0f}%", mult
    if tag.startswith("2oo3") or tag.startswith("2oo4"):
        mult = lat.mult_2 * conf_mult
        if tag.endswith("+flow"):
            mult *= lat.flow_2
        extra = _clip(float(scored.get("ens_mult") or 1.0), 0.70, 1.40)
        return True, f"{tag} size×{mult * extra:.2f}  conf {conf:.0f}%", mult * extra
    if (tag == "1oo3-bounce" or tag == "2oo4-bounce") and conf >= max(
        lat.bounce_conf, float(threshold) * 0.85
    ):
        mult = lat.mult_bounce * conf_mult
        return True, f"{tag} size×{mult:.2f}  conf {conf:.0f}%", mult
    return False, f"skip {tag}: conf {conf:.0f}% (need 2oo3/3oo4 or a high-conf bounce)", 0.0


def _scored_from_hx(hx: dict, lat: LatentRisk) -> dict:
    ens = float(hx.get("ens") or 1.0)
    if hx.get("vol_baked"):
        ens = vol_factor(float(hx.get("atr_ratio") or 1.0), float(hx.get("uncert") or 0.0), lat)
    return {
        "conf": float(hx.get("conf") or 0.0),
        "cal_p": float(hx.get("cal") or 0.5),
        "n": int(hx.get("n") or 0),
        "similar": float(hx.get("similar") or 0.5),
        "similar_w": float(hx.get("similar_w") or 0.0),
        "ens_mult": ens,
        "koo9_mult": float(hx.get("koo9") or 0.0),
    }


def extra_vol_mult(hx: dict, lat: LatentRisk) -> float:
    """Vol haircut outside the gate. Cat path bakes it into ens; factory default does not."""
    if hx.get("vol_baked"):
        return 1.0
    if is_factory(lat):
        return 1.0
    return vol_factor(float(hx.get("atr_ratio") or 1.0), float(hx.get("uncert") or 0.0), lat)


def risk_product(
    lat: LatentRisk | None,
    *,
    risk_pct: float,
    bayes_scale: float,
    size_mult: float,
    vol_mult: float,
    banc: float,
    recover: float,
    admit: float,
    sugar: float,
) -> float:
    """Risk percent after hidden floors. Same order as size_lots."""
    lat = lat or FACTORY
    used = float(risk_pct) * float(bayes_scale)
    used *= max(float(lat.size_floor), float(size_mult) * float(vol_mult))
    banc_term = max(float(lat.banc_floor), float(banc))
    if float(banc) < float(lat.banc_soft):
        banc_term *= float(lat.banc_cut)
    used *= banc_term
    used *= max(float(lat.recover_floor), float(recover))
    used *= max(float(lat.admit_floor), float(admit))
    used *= max(float(lat.sugar_floor), float(sugar))
    return float(min(used, float(lat.risk_cap)))


def position_used(hx: dict, lat: LatentRisk | None, *, admit: float, recover: float) -> float:
    """Risk percent this entry would have used under ``lat``. 0 means skip."""
    lat = lat or FACTORY
    tag = str(hx.get("tag") or "")
    ok, _note, gate = hidden_size(
        _scored_from_hx(hx, lat),
        tag,
        lat,
        threshold=float(hx.get("threshold") or 45.0),
    )
    if not ok:
        return 0.0
    book = float(hx.get("book") or 1.0)
    parts = {
        "kelly_raw": hx.get("kelly_raw"),
        "edge": hx.get("edge"),
        "rr_raw": hx.get("rr_raw"),
        "bayes_scale": hx.get("bayes_scale"),
    }
    if parts["kelly_raw"] is None:
        scale = float(hx.get("bayes_scale") or 1.0)
    else:
        scale = bayes_scale_from_parts(parts, lat)
    return risk_product(
        lat,
        risk_pct=float(hx.get("risk_pct") or 1.0),
        bayes_scale=scale,
        size_mult=gate * book,
        vol_mult=extra_vol_mult(hx, lat),
        banc=float(hx.get("banc") or 1.0),
        recover=float(recover),
        admit=float(admit),
        sugar=float(hx.get("sugar") or 1.0),
    )


def _stats(rows: list[dict]) -> tuple[int, float, float]:
    usd = [float(r.get("usd") or 0.0) for r in rows]
    if not usd:
        return 0, 1.0, 0.0
    gain = sum(x for x in usd if x > 0.0)
    loss = -sum(x for x in usd if x <= 0.0)
    if loss <= 1e-9:
        pf = 99.0 if gain > 0.0 else 1.0
    else:
        pf = gain / loss
    return len(usd), float(pf), float(sum(usd))


def _nudge(default: float, lo: float, hi: float, pf: float, n: int, *, press_hi: bool = True) -> float:
    """Shrink a knob toward the bound that matches this slice's profit factor."""
    if n <= 0:
        return float(default)
    w = n / (n + 6.0)
    t = math.tanh(abs(pf - 1.0) * 1.25)
    if pf >= 1.0:
        target = default + (hi - default) * t if press_hi else default - (default - lo) * t
    else:
        target = default - (default - lo) * t if press_hi else default + (hi - default) * t
    return _clip((1.0 - w) * default + w * target, lo, hi)


def _row_hx(trades: list[dict], *, sealed: bool) -> list[dict]:
    out = []
    for row in trades or []:
        hx = row.get("hx")
        if not isinstance(hx, dict) or not hx.get("tag"):
            continue
        if sealed and float(hx.get("live_used") or 0.0) <= 0.0:
            continue
        out.append(row)
    return out


def _eligible(trades: list[dict]) -> list[dict]:
    return _row_hx(trades, sealed=True)


def _tag(row: dict) -> str:
    return str((row.get("hx") or {}).get("tag") or "")


def _hx(row: dict) -> dict:
    return row.get("hx") or {}


def _propose(base: LatentRisk, trades: list[dict]) -> LatentRisk:
    lat = replace(base)
    rows = _eligible(trades)
    n_all, pf_all, _net = _stats(rows)

    def sl(pred):
        return _stats([r for r in rows if pred(r)])

    n2, pf2, _ = sl(
        lambda r: (_tag(r).startswith("2oo3") or _tag(r).startswith("2oo4"))
        and "flow" not in _tag(r)
        and "bounce" not in _tag(r)
    )
    n3, pf3, _ = sl(
        lambda r: _tag(r).startswith("3oo3")
        or _tag(r).startswith("3oo4")
        or _tag(r).startswith("4oo4")
    )
    nf, pff, _ = sl(lambda r: "flow" in _tag(r))
    nb, pfb, _ = sl(lambda r: "bounce" in _tag(r))
    nv, pfv, _ = sl(lambda r: float(_hx(r).get("atr_ratio") or 1.0) >= base.vol_soft_atr)
    nk, pfk, _ = sl(lambda r: float(_hx(r).get("banc") or 1.0) < base.banc_soft)
    na, pfa, _ = sl(lambda r: int(_hx(r).get("streak") or 0) >= 1)
    ns, pfs, _ = sl(lambda r: int(r.get("snowball_adds") or 0) > 0)
    nsim, pfsim, _ = sl(
        lambda r: float(_hx(r).get("similar") or 0.5) < base.similar_bar + 0.04
        and float(_hx(r).get("similar_w") or 0.0) >= 0.8
    )
    nlow, pflow, _ = sl(lambda r: float(_hx(r).get("conf") or 0.0) < 50.0)

    lo, hi = _BOUNDS["mult_2"]
    lat.mult_2 = _nudge(base.mult_2, lo, hi, pf2, n2)
    lo, hi = _BOUNDS["mult_3"]
    lat.mult_3 = _nudge(base.mult_3, lo, hi, pf3, n3)
    lo, hi = _BOUNDS["flow_2"]
    lat.flow_2 = _nudge(base.flow_2, lo, hi, pff, nf)
    lo, hi = _BOUNDS["flow_3"]
    lat.flow_3 = _nudge(base.flow_3, lo, hi, pff, nf)
    lo, hi = _BOUNDS["mult_bounce"]
    lat.mult_bounce = _nudge(base.mult_bounce, lo, hi, pfb, nb)
    lo, hi = _BOUNDS["similar_bar"]
    lat.similar_bar = _nudge(base.similar_bar, lo, hi, pfsim, nsim, press_hi=False)
    lo, hi = _BOUNDS["conf_lo"]
    lat.conf_lo = _nudge(base.conf_lo, lo, hi, pflow, nlow, press_hi=False)
    lo, hi = _BOUNDS["vol_hard"]
    lat.vol_hard = _nudge(base.vol_hard, lo, hi, pfv, nv, press_hi=False)
    lo, hi = _BOUNDS["vol_soft"]
    lat.vol_soft = _nudge(base.vol_soft, lo, hi, pfv, nv, press_hi=False)
    lo, hi = _BOUNDS["banc_cut"]
    lat.banc_cut = _nudge(base.banc_cut, lo, hi, pfk, nk, press_hi=False)
    lo, hi = _BOUNDS["bayes_hi"]
    lat.bayes_hi = _nudge(base.bayes_hi, lo, hi, pf_all, n_all)
    lo, hi = _BOUNDS["bayes_lo"]
    lat.bayes_lo = _nudge(base.bayes_lo, lo, hi, pf_all, n_all, press_hi=False)
    if lat.bayes_lo >= lat.bayes_hi:
        lat.bayes_lo = min(lat.bayes_lo, lat.bayes_hi - 0.05)
    lo, hi = _BOUNDS["size_floor"]
    worst = min((pf for n, pf in ((n2, pf2), (n3, pf3), (nv, pfv), (nk, pfk)) if n >= 3), default=1.0)
    lat.size_floor = _nudge(base.size_floor, lo, hi, worst, n_all, press_hi=False)
    lo, hi = _BOUNDS["mart_1"]
    lat.mart_1 = _nudge(base.mart_1, lo, hi, pfa, na)
    lo, hi = _BOUNDS["mart_2"]
    lat.mart_2 = _nudge(base.mart_2, lo, hi, pfa, na)
    if lat.mart_2 < lat.mart_1:
        lat.mart_2 = lat.mart_1
    lo, hi = _BOUNDS["scratch_r"]
    lat.scratch_r = _nudge(base.scratch_r, lo, hi, pfa, na, press_hi=False)
    lo, hi = _BOUNDS["recover_dd"]
    lat.recover_dd = _nudge(base.recover_dd, lo, hi, pfa, na, press_hi=False)
    lo, hi = _BOUNDS["snow_frac"]
    lat.snow_frac = _nudge(base.snow_frac, lo, hi, pfs, ns)
    lo, hi = _BOUNDS["snow_banc"]
    lat.snow_banc = _nudge(base.snow_banc, lo, hi, pfs, ns, press_hi=False)
    lo, hi = _BOUNDS["snow_cushion"]
    lat.snow_cushion = _nudge(base.snow_cushion, lo, hi, pfs, ns, press_hi=False)
    n_streak, pf_streak, _ = sl(lambda r: int(_hx(r).get("streak") or 0) >= 2)
    if n_streak >= 2 and pf_streak < 0.90:
        lat.circuit_n = 2
        lat.circuit_bars = int(round(_nudge(float(base.circuit_bars), 8, 28, pf_streak, n_streak, press_hi=False)))
    lo, hi = _BOUNDS["admit_mult"]
    n_early, pf_early, _ = sl(lambda r: int(_hx(r).get("admit_n_seen") or 0) < int(base.admit_n))
    lat.admit_mult = _nudge(base.admit_mult, lo, hi, pf_early, n_early)
    lo, hi = _BOUNDS["lock_keep"]
    n_lock, pf_lock, _ = sl(lambda r: bool(_hx(r).get("locked")))
    lat.lock_keep = _nudge(base.lock_keep, lo, hi, pf_lock, n_lock, press_hi=False)
    return clamp_latent(lat)


def _defensive(fitted: LatentRisk, base: LatentRisk, trades: list[dict]) -> LatentRisk:
    lat = replace(fitted)
    rows = _eligible(trades)

    def pf_of(pred) -> tuple[int, float]:
        n, pf, _ = _stats([r for r in rows if pred(r)])
        return n, pf

    for attr, pred in (
        ("mult_2", lambda r: _tag(r).startswith("2oo3") or _tag(r).startswith("2oo4")),
        (
            "mult_3",
            lambda r: _tag(r).startswith("3oo3")
            or _tag(r).startswith("3oo4")
            or _tag(r).startswith("4oo4"),
        ),
        ("mult_bounce", lambda r: "bounce" in _tag(r)),
    ):
        n, pf = pf_of(pred)
        if n >= 3 and pf < 0.95:
            lo = _BOUNDS[attr][0]
            cur = float(getattr(lat, attr))
            setattr(lat, attr, _clip(cur * 0.85, lo, cur))
    _n, pf_all, _ = _stats(rows)
    if pf_all < 1.0:
        lat.bayes_hi = min(float(lat.bayes_hi), float(base.bayes_hi) * 0.90)
        lat.size_floor = min(float(lat.size_floor), 0.30)
        lat.circuit_n = min(int(lat.circuit_n), 2)
    return clamp_latent(lat)


def recover_mult(
    consec_losses: int,
    equity: float,
    peak_eq: float,
    banc_mult: float,
    enabled: bool,
    tag: str = "",
    last_loss_r: float = 0.0,
    admit_mult: float = 1.0,
    lat: LatentRisk | None = None,
) -> tuple[float, str]:
    """Scratch martingale. Factory: ×1.20 then ×1.28, freeze on 2% DD or BANC < 0.90."""
    lat = lat or FACTORY
    if not enabled:
        return 1.0, ""
    peak = max(float(peak_eq), float(equity))
    dd = (peak - float(equity)) / max(peak, 1.0)
    if dd > float(lat.recover_dd) or float(banc_mult) < float(lat.recover_banc):
        return (
            float(lat.recover_freeze),
            f"recovery freeze DD={100.0 * dd:.1f}% BANC×{float(banc_mult):.2f}",
        )
    if float(admit_mult) < 0.99:
        return 1.0, "no martingale until pair admits"
    if int(consec_losses) <= 0:
        return 1.0, ""
    label = str(tag or "")
    if not (
        label.startswith("2oo3")
        or label.startswith("3oo3")
        or ("oo9" in label and label[:1] in "6789")
    ):
        return 1.0, "no martingale on bounce"
    if float(last_loss_r) >= float(lat.scratch_r):
        return 1.0, "no martingale after a full stop"
    if int(consec_losses) == 1:
        return float(lat.mart_1), f"martingale ×{float(lat.mart_1):.2f} (scratch)"
    return float(lat.mart_2), f"martingale ×{float(lat.mart_2):.2f} (2 scratches, cap)"


def _walk(trades: list[dict], lat: LatentRisk, start_eq: float, *, seal: bool = False):
    """Yield each taken row with the risk percent ``lat`` would have used."""
    from flyfx.risk.fly_rails import BookRails

    rows = _row_hx(trades, sealed=not seal)
    rails = BookRails(
        kill_dd=1.0,
        protect_dd=1.0,
        circuit_n=int(lat.circuit_n),
        circuit_bars=int(lat.circuit_bars),
        admit_n=int(lat.admit_n),
        admit_pf=float(lat.admit_pf),
        admit_mult=float(lat.admit_mult),
    )
    eq = float(start_eq or 100_000.0)
    peak = eq
    consec = 0
    last_loss_r = 0.0
    for row in rows:
        hx = _hx(row)
        step = int(hx.get("step") or 0)
        if rails.blocked(step):
            continue
        admit, _note = rails.admit()
        rec, _why = recover_mult(
            consec,
            eq,
            peak,
            float(hx.get("banc") or 1.0),
            bool(hx.get("mart_on", True)),
            str(hx.get("tag") or ""),
            last_loss_r=last_loss_r,
            admit_mult=admit,
            lat=lat,
        )
        used = position_used(hx, lat, admit=admit, recover=rec)
        if used <= 0.0:
            continue
        if seal:
            hx = dict(hx)
            hx["live_used"] = float(used)
            hx["snow_frac"] = float(lat.snow_frac)
            hx["lock_keep"] = float(lat.lock_keep)
            row = dict(row)
            row["hx"] = hx
        yield row, used, eq, peak
        live = float(hx.get("live_used") or used)
        usd = float(row.get("usd") or 0.0) * (used / live) if live > 0.0 else 0.0
        eq += usd
        if eq > peak:
            peak = eq
        rails.on_close(usd, abs(float(hx.get("r_mult") or 0.0)), step)
        if usd <= 0.0:
            consec += 1
            last_loss_r = abs(float(hx.get("r_mult") or 0.0))
        else:
            consec = 0
            last_loss_r = 0.0


def score_tape(trades: list[dict], lat: LatentRisk, start_eq: float = 100_000.0) -> dict:
    """Rescale the tape under ``lat``. Skips and a longer circuit drop trades."""
    rows = _eligible(trades)
    net = 0.0
    max_dd = 0.0
    n_taken = 0
    peak_pnl = 0.0
    cum = 0.0
    for row, used, _eq, _peak in _walk(trades, lat, start_eq):
        hx = _hx(row)
        live = float(hx.get("live_used") or 0.0)
        if live <= 0.0:
            continue
        usd = float(row.get("usd") or 0.0) * (used / live)
        adds = int(row.get("snowball_adds") or 0)
        if adds > 0:
            share = min(0.70, 0.34 * adds)
            live_frac = max(float(hx.get("snow_frac") or FACTORY.snow_frac), 1e-6)
            usd *= (1.0 - share) + share * (float(lat.snow_frac) / live_frac)
        if hx.get("locked"):
            live_keep = max(float(hx.get("lock_keep") or FACTORY.lock_keep), 1e-6)
            room = 1.0 / max(float(hx.get("lock_ratio") or 1.0), 1e-6)
            usd *= min(room, float(lat.lock_keep) / live_keep)
        cum += usd
        if cum > peak_pnl:
            peak_pnl = cum
        max_dd = max(max_dd, peak_pnl - cum)
        net += usd
        n_taken += 1
    need = max(3, int(0.25 * len(rows))) if rows else 3
    score = net - 0.65 * max_dd
    if n_taken < need:
        score -= 1.0e6
    return {
        "net": float(net),
        "max_dd": float(max_dd),
        "n": int(n_taken),
        "score": float(score),
    }


def fit_latent(trades: list[dict] | None, current: LatentRisk | None = None) -> LatentRisk:
    """Pick hidden factors that improve tape net minus drawdown.

    Fewer than ``MIN_TRADES`` sealed rows leaves ``current`` (or factory) alone.
    The genome is not an input and is not an output.
    """
    base = current if current is not None else FACTORY
    rows = _eligible(trades or [])
    if len(rows) < MIN_TRADES:
        return current if current is not None else LatentRisk()
    start = float(_hx(rows[0]).get("start_eq") or 100_000.0)
    fitted = _propose(base, rows)
    defensive = _defensive(fitted, base, rows)
    cands: list[LatentRisk] = [base, fitted, defensive]
    if not is_factory(base):
        cands.append(FACTORY)
    best = base
    best_score = score_tape(rows, base, start)["score"]
    best_rep = score_tape(rows, base, start)
    for cand in cands[1:]:
        rep = score_tape(rows, cand, start)
        if rep["score"] > best_score + 25.0:
            best = cand
            best_score = rep["score"]
            best_rep = rep
    if best is base:
        return base
    raw = score_tape(rows, base, start)
    best = replace(best)
    best.n_fit = len(rows)
    best.note = (
        f"tape ${raw['net']:+.0f} → ${best_rep['net']:+.0f}  "
        f"dd ${raw['max_dd']:.0f} → ${best_rep['max_dd']:.0f}  "
        f"{describe_latent(best)}"
    )
    return clamp_latent(best)


def stamp_tape(trades: list[dict], lat: LatentRisk | None = None, start_eq: float = 100_000.0) -> list[dict]:
    """Write ``live_used`` from ``lat`` so a later rescore has a denominator."""
    lat = lat or FACTORY
    return [row for row, _used, _eq, _peak in _walk(trades, lat, start_eq, seal=True)]
