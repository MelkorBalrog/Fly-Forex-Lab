"""Per-fly sugar tanks + attributed dopamine on the 3-head FlySwarm.

Each named head (trend, fade, conf, risk) keeps a virtual crop: satiety from
attributed R, its own mix_appetite, and a results×risk governor that sets
sugar dose and DA gain. Sugar still never votes BUY/SELL — a full fly is
muted to HOLD, then 2oo3 runs as today.

W stays frozen. Opt-in via --fly-crops (requires sugar feed).
"""

from __future__ import annotations

import math

from flyfx.brain.sugar_feed import (
    DOSE_FOR_STATE,
    SUGAR_HOLD_BARS,
    compute_satiety,
    feeding_from_raw,
    food_nutrition,
    may_leave_sugar,
    mix_appetite,
    parse_sugar_amt,
)

CROP_NAMES = ("trend", "fade", "conf", "risk")
CROP_START = 10_000.0
R_USD = 80.0
CROP_DAY_CAP = 8
PF_WINDOW = 4
POOR_PF = 0.80
LEAK_FEED = 0.15  # matches trader LEAK; feeding_from_raw rest of tanh(nutrition)


def _clip(val, lo: float, hi: float, default: float = 0.0) -> float:
    try:
        x = float(val)
    except (TypeError, ValueError):
        x = float(default)
    if not math.isfinite(x):
        x = float(default)
    return float(min(max(x, lo), hi))


def _clip01(val, default: float = 0.5) -> float:
    return _clip(val, 0.0, 1.0, default)


def profit_factor(pnls: list[float]) -> float:
    g = sum(x for x in pnls if x > 0)
    l = -sum(x for x in pnls if x <= 0)
    if l <= 1e-9:
        return 99.0 if g > 0 else 0.0
    return float(g / l)


def hit_rate(pnls: list[float]) -> float:
    if not pnls:
        return 0.5
    wins = sum(1 for x in pnls if x > 0)
    return float(wins / len(pnls))


def scaled_r(attributed_r: float, hunger: float, da_mult: float) -> float:
    """Hunger × governor DA gain. Stuffed + threatened → almost no write."""
    h = _clip01(hunger, 0.5)
    d = _clip(da_mult, 0.05, 1.80, 1.0)
    return float(attributed_r) * (0.35 + 0.65 * h) * d


def vote_tag(vote: str, tech: str, other: str) -> tuple[str, int]:
    """How sweet this fly's ballot looks vs the other fused voters (pre-committee)."""
    v = str(vote or "HOLD")
    if v not in ("BUY", "SELL"):
        return "HOLD", 0
    n = 1
    if str(tech or "") == v:
        n += 1
    if str(other or "") == v:
        n += 1
    if n >= 3:
        return "3oo3", 3
    if n >= 2:
        return "2oo3", 2
    return "HOLD", 1


def head_nutrition(
    name: str,
    vote: str,
    *,
    tech: str = "HOLD",
    other: str = "HOLD",
    banc_mult: float = 1.0,
    uncertainty: float = 0.5,
    agreement: float = 0.5,
    in_trade: bool = False,
    trend_with: bool = False,
) -> float:
    """Sweetness of this fly's picture. Conf = sensors; risk = BANC; others = vote tag."""
    n = str(name or "")
    if n == "conf":
        return float(
            min(
                max(0.50 * (1.0 - _clip01(uncertainty)) + 0.50 * _clip01(agreement), 0.0),
                1.0,
            )
        )
    if n == "risk":
        return float(min(max(float(banc_mult) / 1.20, 0.0), 1.0))
    tag, n_agree = vote_tag(vote, tech, other)
    with_side = bool(trend_with) and str(vote) in ("BUY", "SELL")
    return food_nutrition(
        tag,
        banc_mult=banc_mult,
        uncertainty=uncertainty,
        agreement=agreement,
        n_agree=n_agree,
        in_trade=in_trade,
        trend_with=with_side,
    )


def attributed_r(name: str, vote: str, side: str, signed_r: float) -> float:
    """Credit this head for a fill (or shadow) with signed R on the book side."""
    if side not in ("BUY", "SELL") or signed_r == 0.0:
        return 0.0
    mag = abs(float(signed_r))
    v = str(vote or "HOLD")
    n = str(name or "")
    if n == "risk":
        return float(signed_r)
    if n in ("trend", "fade"):
        if v == side:
            return float(signed_r)
        if v in ("BUY", "SELL"):
            return float(-signed_r)
        return 0.0
    if n == "conf":
        if v == side:
            return float(signed_r)
        if v not in ("BUY", "SELL"):
            return mag if signed_r < 0 else 0.0
        return float(-signed_r)
    return 0.0


def credits_from_votes(votes: dict | None, side: str, signed_r: float) -> dict[str, float]:
    v = votes or {}
    return {
        name: attributed_r(name, str(v.get(name) or "HOLD"), side, signed_r)
        for name in CROP_NAMES
    }


def drive_from_results_risk(
    results: dict | None,
    risk: dict | None,
    amt: float = 1.0,
) -> dict:
    """Per-head sugar dose + DA gain from that crop's results and book risk.

    Poor + clean → NIBBLE, learn harder.
    Poor + dangerous → FAST, almost no DA write.
    Hot + clean → FEAST/FORAGE, moderate DA.
    Hot + dangerous → NIBBLE/FAST, protect.
    Kill / low BANC / dirty Kalman → FAST + da_mult floor.
    """
    res = results or {}
    rk = risk or {}
    n = 0
    try:
        n = int(res.get("n") or 0)
    except (TypeError, ValueError):
        n = 0
    pf = _clip(res.get("pf"), 0.0, 99.0, 0.0)
    last_r = _clip(res.get("last_r"), -8.0, 8.0, 0.0)
    hit = _clip01(res.get("hit_rate"), 0.5)
    dd = _clip01(res.get("crop_dd"), 0.0)
    banc = _clip(rk.get("banc_mult"), 0.0, 1.5, 1.0)
    uncert = _clip01(rk.get("uncertainty"), 0.5)
    atr = _clip(rk.get("atr_ratio"), 0.0, 8.0, 1.0)
    last_loss = _clip(rk.get("last_loss_r"), 0.0, 8.0, 0.0)
    kill = bool(rk.get("rails_kill"))
    tol = str(rk.get("risk_tol") or "balanced").strip().lower()
    if tol not in ("conservative", "balanced", "aggressive"):
        tol = "balanced"

    poor = (
        (n >= PF_WINDOW and pf < POOR_PF)
        or (n >= 2 and hit < 0.35)
        or last_r < -0.80
        or dd >= 0.025
    )
    good = (n >= 2 and pf >= 1.20 and last_r >= 0.0) or (
        n >= PF_WINDOW and pf >= 1.0 and hit >= 0.55
    )
    high_risk = (
        kill
        or banc < 0.58
        or uncert >= 0.82
        or atr >= 1.85
        or last_loss >= 1.20
    )
    low_risk = banc >= 0.88 and uncert <= 0.48 and (not kill) and atr < 1.40

    if kill or banc < 0.58 or uncert >= 0.82:
        state = "FAST"
        da_mult = 0.15
        why = f"threat BANC×{banc:.2f} uncert {uncert:.2f}"
        if kill:
            why += " kill"
    elif poor and high_risk:
        state = "FAST"
        da_mult = 0.15
        why = f"cold PF={pf:.2f} high risk BANC×{banc:.2f}"
    elif poor and not high_risk:
        state = "NIBBLE"
        da_mult = 1.25
        why = f"cold PF={pf:.2f} clean BANC×{banc:.2f} relearn"
    elif good and high_risk:
        state = "FAST" if (kill or banc < 0.70) else "NIBBLE"
        da_mult = 0.35
        why = f"hot PF={pf:.2f} protect BANC×{banc:.2f}"
    elif good and low_risk:
        if pf >= 1.50 and hit >= 0.55:
            state = "FEAST"
            why = f"hot PF={pf:.2f} feast BANC×{banc:.2f}"
        else:
            state = "FORAGE"
            why = f"hot PF={pf:.2f} forage BANC×{banc:.2f}"
        da_mult = 0.70
    else:
        state = "FORAGE"
        da_mult = 1.0
        why = f"mid PF={pf:.2f} BANC×{banc:.2f}"

    dose_tilt = 1.0
    da_tilt = 1.0
    if tol == "conservative":
        dose_tilt, da_tilt = 0.85, 0.75
        if state == "FEAST":
            state = "FORAGE"
            why += " conservative"
        elif state == "FORAGE":
            state = "NIBBLE"
            why += " conservative"
    elif tol == "aggressive":
        dose_tilt, da_tilt = 1.15, 1.15
        if state == "NIBBLE" and not high_risk:
            state = "FORAGE"
            why += " aggressive"
        elif state == "FORAGE" and good:
            state = "FEAST"
            why += " aggressive"

    scale = parse_sugar_amt(amt)
    base = float(DOSE_FOR_STATE.get(state) or 1.0)
    dose = parse_sugar_amt(base * scale * dose_tilt)
    da_mult = _clip(da_mult * da_tilt, 0.05, 1.80, 1.0)
    return {
        "state": state,
        "dose": round(dose, 3),
        "da_mult": round(da_mult, 3),
        "why": why,
        "poor": bool(poor),
        "good": bool(good),
        "high_risk": bool(high_risk),
        "low_risk": bool(low_risk),
        "base": round(base, 3),
        "scale": round(scale, 3),
        "want": state,
    }


def mute_votes(trend: str, fade: str, conf: str, packs: dict | None) -> tuple[str, str, str]:
    """Full direction/conf flies do not cast; risk skip never mutes a side."""
    p = packs or {}
    t, f, c = str(trend or "HOLD"), str(fade or "HOLD"), str(conf or "HOLD")
    if (p.get("trend") or {}).get("skip"):
        t = "HOLD"
    if (p.get("fade") or {}).get("skip"):
        f = "HOLD"
    if (p.get("conf") or {}).get("skip"):
        c = "HOLD"
    return t, f, c


def mix_crop_packs(
    decision: str,
    packs: dict | None,
    *,
    tag: str = "",
    bars_held: int = 0,
    min_hold: int = 4,
    unreal: float = 0.0,
    atr: float = 0.0,
) -> dict:
    """One book appetite from the flies that still agree with the fill side."""
    p = packs or {}
    risk_p = dict(p.get("risk") or {})
    agreeing: list[dict] = []
    side = str(decision or "HOLD")
    if side in ("BUY", "SELL"):
        for name in ("trend", "fade"):
            row = dict(p.get(name) or {})
            if str(row.get("vote") or "") == side:
                agreeing.append(row)
    skip = bool(agreeing) and all(bool(row.get("skip")) for row in agreeing)
    if side in ("BUY", "SELL") and not agreeing:
        skip = True
    if agreeing:
        size = sum(float(row.get("size_mult") or 1.0) for row in agreeing) / float(len(agreeing))
        size *= float(risk_p.get("size_mult") or 1.0)
        feed = sum(float(row.get("feeding") or 0.5) for row in agreeing) / float(len(agreeing))
        sat = sum(float(row.get("satiety") or 0.5) for row in agreeing) / float(len(agreeing))
        nut = sum(float(row.get("nutrition") or 0.5) for row in agreeing) / float(len(agreeing))
        dose = sum(float(row.get("dose") or 1.0) for row in agreeing) / float(len(agreeing))
    else:
        size = float(risk_p.get("size_mult") or 1.0)
        feed = float(risk_p.get("feeding") or 0.5)
        sat = float(risk_p.get("satiety") or 0.5)
        nut = float(risk_p.get("nutrition") or 0.5)
        dose = float(risk_p.get("dose") or 1.0)
    if risk_p.get("skip"):
        size *= 0.55
    size = float(min(max(size, 0.55), 1.22))
    trend_p = dict(p.get("trend") or {})
    snowball = bool(trend_p.get("snowball")) and not bool(risk_p.get("skip"))
    atr = max(float(atr or 0.0), 1e-12)
    flatten = bool(risk_p.get("flatten")) or (
        sat >= 0.84
        and int(bars_held) >= int(min_hold)
        and float(unreal) > 0.40 * atr
    )
    reason = "sugar-crops"
    if flatten:
        reason = "sugar full"
    elif skip:
        reason = "sugar-crops"
    elif risk_p.get("skip"):
        reason = "risk nibble"
    else:
        reason = str(trend_p.get("reason") or risk_p.get("reason") or "forage")
    hunger = 1.0 - sat
    return {
        "feeding": round(feed, 4),
        "satiety": round(sat, 4),
        "hunger": round(hunger, 4),
        "appetite": round(feed * (0.25 + 0.75 * hunger), 4),
        "nutrition": round(nut, 4),
        "dose": round(dose, 3),
        "size_mult": round(size, 4),
        "skip": bool(skip),
        "snowball": bool(snowball),
        "flatten": bool(flatten),
        "reason": reason,
        "dyn": True,
        "dyn_state": str((p.get("trend") or {}).get("dyn_state") or "FORAGE"),
        "dyn_reason": "crops",
        "crops": {name: dict(p.get(name) or {}) for name in CROP_NAMES},
        "tag": str(tag or ""),
    }


class FlyCrop:
    """Virtual satiety tank for one FlySwarm head."""

    def __init__(self, name: str, start: float = CROP_START) -> None:
        self.name = str(name)
        self.start = float(start)
        self.equity = float(start)
        self.peak = float(start)
        self.day_start = float(start)
        self.day_key = ""
        self.day_trades = 0
        self.n_trades = 0
        self.pnls: list[float] = []
        self.last_r = 0.0
        self.last_vote = "HOLD"
        self.last_pack: dict = {}
        self.da_mult = 1.0
        self.dose = 1.0
        self.state = "FORAGE"
        self.want = "FORAGE"
        self.why = "init"
        self.bars_in = 0
        self._primed = False
        self.hold_bars = SUGAR_HOLD_BARS

    def _roll_day(self, day_key: str) -> None:
        day = str(day_key or "")
        if day and day != self.day_key:
            self.day_key = day
            self.day_start = float(self.equity)
            self.day_trades = 0

    def results(self) -> dict:
        window = self.pnls[-PF_WINDOW:]
        peak = max(float(self.peak), float(self.equity), 1e-9)
        dd = max(0.0, (peak - float(self.equity)) / peak)
        return {
            "n": len(window),
            "pf": profit_factor(window),
            "hit_rate": hit_rate(window),
            "last_r": self.last_r,
            "crop_dd": dd,
        }

    def _hold_state(self, want: str, satiety: float, *, force: bool = False) -> str:
        if not self._primed:
            self.state = want
            self.bars_in = 1
            self._primed = True
            return self.state
        if want == self.state:
            self.bars_in += 1
            return self.state
        if force or may_leave_sugar(self.state, want, self.bars_in, self.hold_bars, satiety=satiety):
            self.state = want
            self.bars_in = 1
            return self.state
        self.bars_in += 1
        self.why = f"hold {self.state} ({self.bars_in}/{self.hold_bars}) want {want}"
        return self.state

    def on_bar(
        self,
        vote: str,
        *,
        nutrition: float,
        floating: float,
        risk: dict,
        amt: float = 1.0,
        sugar_dyn: bool = True,
        tag: str = "",
        bars_held: int = 0,
        min_hold: int = 4,
        unreal: float = 0.0,
        atr: float = 0.0,
        day_key: str = "",
    ) -> dict:
        self._roll_day(day_key)
        self.last_vote = str(vote or "HOLD")
        sat = compute_satiety(
            equity=self.equity,
            peak_eq=self.peak,
            start=self.start,
            day_start_eq=self.day_start,
            floating=float(floating or 0.0),
            day_trades=self.day_trades,
            day_trade_cap=CROP_DAY_CAP,
        )
        drive = drive_from_results_risk(self.results(), risk, amt=amt)
        self.want = str(drive.get("want") or drive.get("state") or "FORAGE")
        self.da_mult = float(drive.get("da_mult") or 1.0)
        self.why = str(drive.get("why") or "")
        kill = bool((risk or {}).get("rails_kill"))
        banc = _clip((risk or {}).get("banc_mult"), 0.0, 1.5, 1.0)
        hard_fast = kill or banc < 0.58 or _clip01((risk or {}).get("uncertainty"), 0.5) >= 0.82
        if sugar_dyn:
            held = self._hold_state(self.want, sat, force=hard_fast)
            if held != self.want:
                drive = dict(drive)
                drive["state"] = held
                base = float(DOSE_FOR_STATE.get(held) or 1.0)
                drive["dose"] = parse_sugar_amt(base * parse_sugar_amt(amt))
            self.dose = float(drive["dose"])
            self.state = str(drive["state"])
        else:
            if hard_fast:
                self.state = "FAST"
                self.dose = parse_sugar_amt(float(DOSE_FOR_STATE["FAST"]) * parse_sugar_amt(amt))
                self._hold_state("FAST", sat, force=True)
            else:
                self.state = "FORAGE"
                self.dose = parse_sugar_amt(amt)
            self.why = str(drive.get("why") or "manual") + " dyn-off"
        mag = float(min(max(float(nutrition), 0.0), 1.0))
        raw = abs((1.0 - LEAK_FEED) * math.tanh(mag))
        rest = abs((1.0 - LEAK_FEED) * math.tanh(0.45))
        feed = feeding_from_raw(raw, rest)
        pack = mix_appetite(
            feed,
            sat,
            tag=tag,
            bars_held=bars_held,
            min_hold=min_hold,
            unreal=unreal,
            atr=atr,
            dose=self.dose,
        )
        pack["vote"] = self.last_vote
        pack["nutrition"] = round(mag, 4)
        pack["dose"] = round(float(self.dose), 3)
        pack["da_mult"] = round(float(self.da_mult), 3)
        pack["dyn"] = bool(sugar_dyn)
        pack["dyn_state"] = self.state
        pack["dyn_reason"] = self.why
        pack["last_r"] = round(float(self.last_r), 3)
        pack["pf"] = round(float(self.results()["pf"]), 3)
        pack["equity"] = round(float(self.equity), 2)
        pack["name"] = self.name
        self.last_pack = pack
        return pack

    def on_close(self, signed_r: float, vote: str, side: str) -> float:
        ar = attributed_r(self.name, vote, side, signed_r)
        self.last_r = float(ar)
        if ar == 0.0:
            return 0.0
        usd = float(ar) * R_USD
        self.equity = float(self.equity + usd)
        self.peak = max(float(self.peak), float(self.equity))
        self.pnls.append(usd)
        self.pnls = self.pnls[-40:]
        self.n_trades += 1
        self.day_trades += 1
        return float(ar)

    def dump(self) -> dict:
        return {
            "name": self.name,
            "equity": self.equity,
            "peak": self.peak,
            "day_start": self.day_start,
            "day_key": self.day_key,
            "day_trades": self.day_trades,
            "n_trades": self.n_trades,
            "pnls": list(self.pnls[-40:]),
            "last_r": self.last_r,
            "state": self.state,
            "bars_in": self.bars_in,
        }

    def load(self, data: dict | None) -> None:
        if not isinstance(data, dict):
            return
        self.equity = float(data.get("equity", self.equity) or self.equity)
        self.peak = float(data.get("peak", self.peak) or self.peak)
        self.day_start = float(data.get("day_start", self.day_start) or self.day_start)
        self.day_key = str(data.get("day_key") or self.day_key)
        self.day_trades = int(data.get("day_trades") or 0)
        self.n_trades = int(data.get("n_trades") or 0)
        raw = data.get("pnls") or []
        if isinstance(raw, list):
            self.pnls = [float(x) for x in raw][-40:]
        self.last_r = float(data.get("last_r") or 0.0)
        self.state = str(data.get("state") or self.state)
        self.bars_in = int(data.get("bars_in") or 0)
        self._primed = True


class FlyCropPool:
    """Four FlyCrop tanks + mix into one book pack."""

    def __init__(self) -> None:
        self.crops = {name: FlyCrop(name) for name in CROP_NAMES}
        self.last_packs: dict = {}
        self.last_book: dict = {}

    def step(
        self,
        votes: dict,
        *,
        tech: str = "HOLD",
        banc_mult: float = 1.0,
        uncertainty: float = 0.5,
        agreement: float = 0.5,
        in_trade: bool = False,
        trend_with: bool = False,
        floating: float = 0.0,
        entry_agreed: list[str] | tuple[str, ...] | None = None,
        risk: dict | None = None,
        amt: float = 1.0,
        sugar_dyn: bool = True,
        tag: str = "",
        bars_held: int = 0,
        min_hold: int = 4,
        unreal: float = 0.0,
        atr: float = 0.0,
        day_key: str = "",
    ) -> dict:
        rk = dict(risk or {})
        rk.setdefault("banc_mult", banc_mult)
        rk.setdefault("uncertainty", uncertainty)
        agreed = set(str(x) for x in (entry_agreed or ()))
        packs: dict = {}
        t_vote = str((votes or {}).get("trend") or "HOLD")
        f_vote = str((votes or {}).get("fade") or "HOLD")
        for name in CROP_NAMES:
            vote = str((votes or {}).get(name) or "HOLD")
            other = f_vote if name == "trend" else t_vote
            nut = head_nutrition(
                name,
                vote,
                tech=tech,
                other=other,
                banc_mult=banc_mult,
                uncertainty=uncertainty,
                agreement=agreement,
                in_trade=in_trade,
                trend_with=trend_with if name == "trend" else False,
            )
            float_usd = 0.0
            if in_trade and (name == "risk" or name in agreed):
                float_usd = float(floating or 0.0)
            packs[name] = self.crops[name].on_bar(
                vote,
                nutrition=nut,
                floating=float_usd,
                risk=rk,
                amt=amt,
                sugar_dyn=sugar_dyn,
                tag=tag,
                bars_held=bars_held,
                min_hold=min_hold,
                unreal=unreal if (name in agreed or not in_trade) else 0.0,
                atr=atr,
                day_key=day_key,
            )
        self.last_packs = packs
        return packs

    def mix(
        self,
        decision: str,
        *,
        tag: str = "",
        bars_held: int = 0,
        min_hold: int = 4,
        unreal: float = 0.0,
        atr: float = 0.0,
    ) -> dict:
        book = mix_crop_packs(
            decision,
            self.last_packs,
            tag=tag,
            bars_held=bars_held,
            min_hold=min_hold,
            unreal=unreal,
            atr=atr,
        )
        self.last_book = book
        return book

    def mean_nutrition(self, names: list[str] | None = None) -> float:
        keys = list(names or ("trend", "fade"))
        vals = [float((self.last_packs.get(n) or {}).get("nutrition") or 0.5) for n in keys]
        if not vals:
            return 0.5
        return float(sum(vals) / len(vals))

    def on_close(self, side: str, signed_r: float, votes: dict | None) -> dict[str, float]:
        v = votes or {}
        out: dict[str, float] = {}
        for name, crop in self.crops.items():
            out[name] = crop.on_close(signed_r, str(v.get(name) or crop.last_vote), side)
        return out

    def da_scale(self) -> dict[str, float]:
        return {name: float(c.da_mult) for name, c in self.crops.items()}

    def hunger(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, c in self.crops.items():
            sat = float((c.last_pack or {}).get("satiety") or 0.5)
            out[name] = float(max(0.0, min(1.0, 1.0 - sat)))
        return out

    def snapshot(self) -> dict:
        return {name: dict(self.last_packs.get(name) or {}) for name in CROP_NAMES}

    def dump(self) -> dict:
        return {name: crop.dump() for name, crop in self.crops.items()}

    def load(self, data: dict | None) -> None:
        if not isinstance(data, dict):
            return
        for name, crop in self.crops.items():
            blob = data.get(name)
            if isinstance(blob, dict):
                crop.load(blob)
