"""Per-category Kalman fusion of all live indicators in that family.

Each category keeps one latent signed state (inverse-variance mix of
Kalman1D channels) plus a constant-velocity [level, slope] on that mix.
Missing / NaN measurements are predict-only and drop out of the mix so
breadth and sentiment still run when only a subset (or none) of the
overlay series is present.
"""

from __future__ import annotations

import math
import re

import numpy as np

from flyfx.sense.kalman_signal import Kalman1D, KalmanCV, is_measurement

CATEGORY_NAMES = (
    "trend",
    "momentum",
    "volatility",
    "volume",
    "breadth",
    "structure",
    "sentiment",
)

# Overlay / participation families: NEUTRAL when empty, not a fake oscillator.
SPARSE_CATEGORIES = frozenset({"volume", "breadth", "sentiment"})

FUSE_ALIASES = {
    "trend": "trend",
    "momentum": "momentum",
    "mom": "momentum",
    "osc": "momentum",
    "rsi": "momentum",
    "volatility": "volatility",
    "vola": "volatility",
    "atr": "volatility",
    "volume": "volume",
    "vol": "volume",
    "flow": "volume",
    "breadth": "breadth",
    "ad": "breadth",
    "structure": "structure",
    "struct": "structure",
    "sr": "structure",
    "sentiment": "sentiment",
    "sent": "sentiment",
}


CATEGORY_CHANNELS = {
    "trend": ("sma", "ema", "wma", "macd", "adx", "sar", "ichimoku", "supertrend"),
    "momentum": ("rsi", "stoch", "cci", "willr", "roc"),
    "volatility": ("bb", "atr", "keltner", "stdev", "donchian"),
    "volume": ("obv", "cmf", "vwap", "ad", "profile"),
    "breadth": ("ad_line", "mcclellan", "trin"),
    "structure": ("pivot", "fib", "channel"),
    "sentiment": ("put_call", "vix", "cot"),
}

CHANNEL_TO_CAT: dict[str, str] = {
    ch: cat for cat, chs in CATEGORY_CHANNELS.items() for ch in chs
}

CHANNEL_ALIASES = {
    **{ch: ch for ch in CHANNEL_TO_CAT},
    "ichi": "ichimoku",
    "ichimoku": "ichimoku",
    "st": "supertrend",
    "super": "supertrend",
    "supertrend": "supertrend",
    "williams": "willr",
    "will": "willr",
    "wr": "willr",
    "%r": "willr",
    "stochastic": "stoch",
    "stoch": "stoch",
    "kel": "keltner",
    "keltner": "keltner",
    "std": "stdev",
    "stddev": "stdev",
    "stdev": "stdev",
    "don": "donchian",
    "donchian": "donchian",
    "pc": "put_call",
    "putcall": "put_call",
    "put_call": "put_call",
    "mcc": "mcclellan",
    "mcclellan": "mcclellan",
    "adline": "ad_line",
    "ad_line": "ad_line",
    "advdec": "ad_line",
    "accum": "ad",
    "vp": "profile",
    "volprofile": "profile",
}


def _full_inds() -> dict[str, tuple[str, ...]]:
    return {n: tuple(chs) for n, chs in CATEGORY_CHANNELS.items()}


def parse_fuse_inds(spec: str | None | list | tuple | dict | None) -> dict[str, tuple[str, ...]]:
    """Which indicator channels enter each category Kalman.

    Empty / ``all`` → every channel. ``none`` → none.
    Flat ``ema,macd,rsi`` keeps only those names (unlisted stay out).
    A category token in a flat list (``trend``) means every channel in it.
    Grouped ``trend:ema+macd;momentum:rsi`` constrains those families;
    unmentioned families keep every channel.
    """
    if isinstance(spec, dict):
        out = _full_inds()
        unknown: list[str] = []
        for key, val in spec.items():
            cat = FUSE_ALIASES.get(str(key).strip().lower(), "")
            if not cat:
                unknown.append(str(key))
                continue
            if isinstance(val, str):
                chs = _parse_channel_tokens(val, allow_empty=True)
            else:
                chs = _parse_channel_tokens(
                    ",".join(str(x) for x in (val or ())), allow_empty=True
                )
            out[cat] = chs
        if unknown:
            raise ValueError(f"unknown fuse category {unknown!r}")
        return out
    if isinstance(spec, (list, tuple)):
        text = ",".join(str(x) for x in spec).strip().lower()
    else:
        text = str(spec or "").strip().lower()
    if not text or text in ("all", "*", "default"):
        return _full_inds()
    if text in ("plausibility", "plausible", "4k"):
        return _flat_inds(",".join(PLAUSIBILITY_INDS))
    if text in ("gui-default", "lab-default", "lab"):
        return _flat_inds(",".join(LAB_DEFAULT_INDS))
    if text in ("none", "off", "-"):
        return {n: () for n in CATEGORY_NAMES}
    if ":" in text:
        out = _full_inds()
        for chunk in re.split(r"[;]+", text):
            chunk = chunk.strip()
            if not chunk:
                continue
            if ":" not in chunk:
                for cat, chs in _flat_inds(chunk).items():
                    if chs:
                        out[cat] = chs
                continue
            cat_tok, rest = chunk.split(":", 1)
            cat = FUSE_ALIASES.get(cat_tok.strip(), "")
            if not cat:
                raise ValueError(f"unknown fuse category {cat_tok!r}")
            rest = rest.strip()
            if rest in ("", "-", "none", "off"):
                out[cat] = ()
            elif rest in ("*", "all"):
                out[cat] = tuple(CATEGORY_CHANNELS[cat])
            else:
                chs = _parse_channel_tokens(rest, allow_empty=False)
                own = tuple(ch for ch in chs if CHANNEL_TO_CAT.get(ch) == cat)
                if not own:
                    raise ValueError(f"no {cat} indicators in {rest!r}")
                out[cat] = own
        return out
    return _flat_inds(text)


def _parse_channel_tokens(text: str, allow_empty: bool = False) -> tuple[str, ...]:
    raw = [p.strip() for p in re.split(r"[,+\s|/]+", str(text or "").lower()) if p.strip()]
    out: list[str] = []
    unknown: list[str] = []
    for tok in raw:
        name = CHANNEL_ALIASES.get(tok, "")
        if not name:
            unknown.append(tok)
            continue
        if name not in out:
            out.append(name)
    if unknown:
        raise ValueError(
            f"unknown fuse indicator {unknown!r}; use {', '.join(CHANNEL_TO_CAT)}"
        )
    if not out and not allow_empty:
        raise ValueError("fuse-inds resolved to empty")
    return tuple(out)


def _flat_inds(text: str) -> dict[str, tuple[str, ...]]:
    raw = [p.strip() for p in re.split(r"[,+\s|/]+", text) if p.strip()]
    picked: list[str] = []
    unknown: list[str] = []
    for tok in raw:
        name = CHANNEL_ALIASES.get(tok, "")
        if name:
            if name not in picked:
                picked.append(name)
            continue
        cat = FUSE_ALIASES.get(tok, tok if tok in CATEGORY_NAMES else "")
        if cat:
            for ch in CATEGORY_CHANNELS[cat]:
                if ch not in picked:
                    picked.append(ch)
            continue
        unknown.append(tok)
    if unknown:
        raise ValueError(
            f"unknown fuse indicator {unknown!r}; use {', '.join(CHANNEL_TO_CAT)}"
        )
    out: dict[str, list[str]] = {n: [] for n in CATEGORY_NAMES}
    for name in picked:
        out[CHANNEL_TO_CAT[name]].append(name)
    return {n: tuple(out[n]) for n in CATEGORY_NAMES}


# Older 4k plausibility subset (CLI --fuse-inds plausibility).
PLAUSIBILITY_INDS: tuple[str, ...] = (
    "ema",
    "macd",
    "adx",
    "sar",
    "ichimoku",
    "supertrend",
    "rsi",
    "stoch",
    "cci",
    "willr",
    "bb",
)
# Lab GUI default (operator screenshot / CFG default).
LAB_DEFAULT_INDS: tuple[str, ...] = (
    "macd",
    "sar",
    "ichimoku",
    "supertrend",
    "rsi",
    "stoch",
    "bb",
    "keltner",
    "obv",
    "vwap",
    "ad_line",
    "pivot",
    "fib",
    "put_call",
)


def encode_fuse_inds(mapping: dict[str, tuple[str, ...] | list[str]] | None) -> str:
    blob = parse_fuse_inds(mapping) if mapping is not None else _full_inds()
    if blob == _full_inds():
        return "all"
    if blob == parse_fuse_inds("plausibility"):
        return "plausibility"
    if blob == parse_fuse_inds("lab-default"):
        return "lab-default"
    names: list[str] = []
    for cat in CATEGORY_NAMES:
        names.extend(blob.get(cat) or ())
    return ",".join(names) if names else "none"


def adapt_fuse_inds(
    inds: dict | str | None,
    measurements: dict | None,
    side: str,
    won: bool,
) -> tuple[dict[str, tuple[str, ...]], str]:
    """Swap one fused indicator after a close.

    A winner drops the channel that opposed the trade and turns on one that
    agreed. A loser drops a channel that agreed with the bad trade and turns
    on one that warned. A category never loses its last channel.
    """
    sign = 1.0 if str(side).upper() == "BUY" else (-1.0 if str(side).upper() == "SELL" else 0.0)
    cur = {cat: list(chs) for cat, chs in parse_fuse_inds(inds).items()}
    if sign == 0.0:
        return {k: tuple(v) for k, v in cur.items()}, ""
    meas = measurements or {}
    scored: list[tuple[float, str, str, bool]] = []
    for cat, channels in CATEGORY_CHANNELS.items():
        on = set(cur.get(cat) or ())
        cat_m = meas.get(cat) if isinstance(meas.get(cat), dict) else {}
        for name in channels:
            raw = cat_m.get(name)
            if raw is None:
                continue
            try:
                align = sign * float(raw)
            except (TypeError, ValueError):
                continue
            if align != align:
                continue
            scored.append((align, cat, name, name in on))
    if won:
        drop = sorted((s for s in scored if s[3] and s[0] < -0.25), key=lambda s: s[0])
        add = sorted((s for s in scored if (not s[3]) and s[0] > 0.25), key=lambda s: -s[0])
    else:
        drop = sorted((s for s in scored if s[3] and s[0] > 0.25), key=lambda s: -s[0])
        add = sorted((s for s in scored if (not s[3]) and s[0] < -0.25), key=lambda s: s[0])
    notes: list[str] = []
    if drop:
        _align, cat, name, _on = drop[0]
        if len(cur.get(cat) or []) > 1:
            cur[cat] = [ch for ch in cur[cat] if ch != name]
            notes.append(f"{name} off")
    if add:
        _align, cat, name, _on = add[0]
        if name not in (cur.get(cat) or []):
            cur.setdefault(cat, []).append(name)
            notes.append(f"{name} on")
    out = {cat: tuple(chs) for cat, chs in cur.items()}
    if not notes:
        return out, ""
    return out, "fuse  " + "  ".join(notes)


def parse_fuse_cats(spec: str | None | list | tuple) -> tuple[str, ...]:
    """Parse ``trend,volume`` / ``trend+mom`` / ``all``. Empty → all seven."""
    if isinstance(spec, (list, tuple)):
        text = ",".join(str(x) for x in spec).strip().lower()
    else:
        text = str(spec or "").strip().lower()
    if not text or text in ("all", "*", "default"):
        return tuple(CATEGORY_NAMES)
    raw = [p.strip() for p in re.split(r"[,+\s|/]+", text) if p.strip()]
    out: list[str] = []
    unknown: list[str] = []
    for tok in raw:
        name = FUSE_ALIASES.get(tok, tok if tok in CATEGORY_NAMES else "")
        if not name:
            unknown.append(tok)
            continue
        if name not in out:
            out.append(name)
    if unknown:
        raise ValueError(
            f"unknown fuse category {unknown!r}; use {', '.join(CATEGORY_NAMES)}"
        )
    if not out:
        raise ValueError("fuse-cats resolved to empty")
    return tuple(out)

_NOISE = {
    "trend": {"q": 2.2e-4, "r": 1.4e-2, "fuse_q": 2.8e-4, "fuse_r": 1.6e-2},
    "momentum": {"q": 5.0e-4, "r": 2.8e-2, "fuse_q": 4.0e-4, "fuse_r": 2.2e-2},
    "volatility": {"q": 3.5e-4, "r": 2.2e-2, "fuse_q": 3.2e-4, "fuse_r": 1.8e-2},
    "volume": {"q": 4.0e-4, "r": 3.0e-2, "fuse_q": 3.5e-4, "fuse_r": 2.4e-2},
    "breadth": {"q": 2.5e-4, "r": 3.5e-2, "fuse_q": 2.5e-4, "fuse_r": 2.8e-2},
    "structure": {"q": 3.0e-4, "r": 2.0e-2, "fuse_q": 3.0e-4, "fuse_r": 1.8e-2},
    "sentiment": {"q": 2.0e-4, "r": 4.0e-2, "fuse_q": 2.2e-4, "fuse_r": 3.2e-2},
}


def fuse_category_latents(
    states: dict[str, dict] | None,
    include: tuple[str, ...] | list[str] | None = None,
    *,
    isolate: bool = True,
    health_out: list | None = None,
) -> dict:
    """Inverse-variance mix of selected, available category fused latents.

    Mixes signed Kalman states, never BUY/SELL. Sparse overlays that never
    measured (volume / breadth / sentiment) stay out — they are not faked.
    ``include`` limits which families enter the mix (default: all seven).

    When ``isolate`` is True (default), uncertain / outlier families are
    dropped and no single family may dominate the weight — so one broken
    Kalman cannot cascade into every voter that reads ``impulse``.
    """
    from flyfx.risk.isolation import (
        disagree_circuit,
        isolate_category_states,
    )

    blob = states or {}
    try:
        selected = parse_fuse_cats(include)
    except ValueError:
        selected = tuple(CATEGORY_NAMES)
    empty = {
        "impulse": 0.0,
        "fused": 0.0,
        "residual": 0.0,
        "P": 1.0,
        "uncertainty": 1.0,
        "agreement": 0.0,
        "n_live": 0,
        "n_expected": len(selected),
        "available": False,
        "warmup": True,
        "regime": "CHOP",
        "parts": {},
        "names": [],
        "selected": list(selected),
        "isolation": "",
    }
    if isolate:
        rows, health = isolate_category_states(blob, selected)
        if health_out is not None:
            health_out.clear()
            health_out.append(health)
        if not rows:
            empty["isolation"] = health.note()
            return empty
        names = [n for n, _z, _r, _w in rows]
        zs = [z for _n, z, _r, _w in rows]
        rs = [r for _n, _z, r, _w in rows]
        ws = [w for _n, _z, _r, w in rows]
        parts = {n: round(z, 4) for n, z, _r, _w in rows}
        iso_note = health.note()
    else:
        zs = []
        ws = []
        rs = []
        parts = {}
        names = []
        iso_note = ""
        for name in selected:
            st = blob.get(name) or {}
            if not st.get("available"):
                continue
            p = float(st.get("P") or 1.0)
            w = 1.0 / max(p, 1e-9)
            z = float(st.get("fused") or 0.0)
            zs.append(z)
            ws.append(w)
            rs.append(float(st.get("residual") or 0.0))
            parts[name] = round(z, 4)
            names.append(name)
        if not zs:
            return empty

    w = np.array(ws, dtype=float)
    z = np.array(zs, dtype=float)
    mix = float(np.dot(w, z) / w.sum())
    rmix = float(np.dot(w, np.array(rs, dtype=float)) / w.sum())
    p_mix = float(1.0 / w.sum())
    agr = float(1.0 / (1.0 + np.std(z))) if len(z) >= 2 else 0.55
    # Annotate hard disagreement for sizing / dash — do not zero a healthy
    # impulse (that would break the 4k-era cat-fuse path on borderline bars).
    if isolate and disagree_circuit(list(z), agr):
        iso_note = (iso_note + "  disagree-circuit").strip()
        if health_out:
            health_out[0].disagree_circuit = True
    # Sep-20 4k-era: hot enough that all-channel fuse clears cont_impulse on
    # Sep-10 13:50 / Sep-14 08:05, but not so hot that 13:40 fires first and
    # poisons the book (scale 1.20 did). Bracket is ~1.01–1.08 on this HST.
    impulse = float(np.tanh(1.05 * mix))
    if mix > 0.28:
        regime = "UP"
    elif mix < -0.28:
        regime = "DOWN"
    else:
        regime = "CHOP"
    return {
        "impulse": float(impulse),
        "fused": round(mix, 4),
        "residual": round(float(np.clip(rmix, -4.0, 4.0)), 4),
        "P": round(p_mix, 5),
        "uncertainty": round(float(min(1.0, math.sqrt(max(p_mix, 1e-9)))), 4),
        "agreement": round(agr, 4),
        "n_live": len(zs),
        "n_expected": len(selected),
        "available": True,
        "warmup": False,
        "regime": regime,
        "parts": parts,
        "names": names,
        "selected": list(selected),
        "isolation": iso_note,
        "disagree": "disagree-circuit" in iso_note,
    }


def _empty_state(name: str, n_expected: int) -> dict:
    return {
        "name": name,
        "impulse": 0.0,
        "fused": 0.0,
        "residual": 0.0,
        "slope": 0.0,
        "level": 0.0,
        "P": 1.0,
        "uncertainty": 1.0,
        "innovation": 0.0,
        "agreement": 0.0,
        "n_live": 0,
        "n_expected": n_expected,
        "available": False,
        "warmup": True,
        "regime": "CHOP",
        "parts": {},
        "snr": 0.0,
        "channels": [],
    }


class CategoryKalman:
    """Fuse every live channel in one indicator category into one latent."""

    def __init__(self, name: str, expected: tuple[str, ...] | list[str]):
        self.name = name
        self.expected = tuple(expected)
        noise = _NOISE.get(name, _NOISE["trend"])
        self.q = float(noise["q"])
        self.r = float(noise["r"])
        self.filters: dict[str, Kalman1D] = {k: Kalman1D(q=self.q, r=self.r) for k in self.expected}
        self.kf_fuse = Kalman1D(q=float(noise["fuse_q"]), r=float(noise["fuse_r"]))
        self.kf_cv = KalmanCV(q=8.0e-6, r=5.0e-3)
        self._steps = 0
        self._live_steps = 0

    def update(
        self,
        measurements: dict[str, float | None] | None,
        include: tuple[str, ...] | list[str] | None = None,
    ) -> dict:
        src = measurements or {}
        if include is None:
            selected = set(self.expected)
        else:
            selected = {k for k in include if k in self.filters}
        zs: list[float] = []
        ps: list[float] = []
        parts: dict[str, float] = {}
        live_raw: list[float] = []
        n_live = 0
        for key in self.expected:
            z = src.get(key)
            kf = self.filters[key]
            if is_measurement(z):
                x, p = kf.step(float(z))
                if key in selected:
                    zs.append(x)
                    ps.append(p)
                    parts[key] = round(float(x), 4)
                    live_raw.append(float(z))
                    n_live += 1
            else:
                kf.step(None)
        self._steps += 1
        n_exp = max(len(selected), 1)
        if not zs:
            fused, pf = self.kf_fuse.step(None)
            level, slope = self.kf_cv.step(None)
            out = _empty_state(self.name, n_exp if selected else 0)
            out["fused"] = round(float(fused), 4)
            out["P"] = round(float(pf), 5)
            out["uncertainty"] = round(float(min(1.0, math.sqrt(max(pf, 1e-9)) * (n_exp ** 0.35))), 4)
            out["level"] = round(float(level), 4)
            out["slope"] = round(float(slope), 6)
            out["warmup"] = self._live_steps < 3
            out["channels"] = list(selected)
            return out

        w = 1.0 / np.maximum(np.array(ps, dtype=float), 1e-9)
        mix = float(np.dot(w, np.array(zs, dtype=float)) / w.sum())
        prior = float(self.kf_fuse.x)
        fused, pf = self.kf_fuse.step(mix)
        level, slope = self.kf_cv.step(mix)
        residual = float(np.clip(mix - float(level), -4.0, 4.0))
        innovation = float(mix - prior)
        if len(live_raw) >= 2:
            agr = float(1.0 / (1.0 + np.std(np.array(live_raw, dtype=float))))
        else:
            agr = 0.55
        snr = abs(mix) / max(math.sqrt(max(pf, 1e-9)), 0.08)
        impulse = float(np.tanh(0.9 * fused))
        if fused > 0.28:
            regime = "UP"
        elif fused < -0.28:
            regime = "DOWN"
        else:
            regime = "CHOP"
        self._live_steps += 1
        uncertainty = float(min(1.0, math.sqrt(max(pf, 1e-9)) * (n_exp / max(n_live, 1)) ** 0.35))
        return {
            "name": self.name,
            "impulse": float(impulse),
            "fused": round(float(fused), 4),
            "residual": round(residual, 4),
            "slope": round(float(slope), 6),
            "level": round(float(level), 4),
            "P": round(float(pf), 5),
            "uncertainty": round(uncertainty, 4),
            "innovation": round(float(innovation), 4),
            "agreement": round(float(agr), 4),
            "n_live": int(n_live),
            "n_expected": int(n_exp),
            "available": True,
            "warmup": self._live_steps < 3,
            "regime": regime,
            "parts": parts,
            "snr": round(float(snr), 3),
            "channels": list(selected),
        }


class CategoryKalmanBank:
    """Seven independent category filters plus one inverse-variance mix."""

    CHANNELS = CATEGORY_CHANNELS

    def __init__(self, include: tuple[str, ...] | list[str] | None = None) -> None:
        self.filters = {name: CategoryKalman(name, keys) for name, keys in CATEGORY_CHANNELS.items()}
        self.last: dict[str, dict] = {n: _empty_state(n, len(k)) for n, k in CATEGORY_CHANNELS.items()}
        try:
            self.include: tuple[str, ...] = parse_fuse_cats(include)
        except ValueError:
            self.include = tuple(CATEGORY_NAMES)
        self.channel_include: dict[str, tuple[str, ...]] = _full_inds()
        self.mix: dict = fuse_category_latents({}, include=self.include)

    def update(self, measurements: dict[str, dict] | None) -> dict[str, dict]:
        blob = measurements or {}
        out: dict[str, dict] = {}
        for name, kf in self.filters.items():
            chs = self.channel_include.get(name)
            out[name] = kf.update(blob.get(name) or {}, include=chs)
        self.last = out
        self.mix = fuse_category_latents(out, include=self.include)
        return out
