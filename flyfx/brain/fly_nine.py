"""Nine MaleCNS flies on one W, diverse indicator encodings, k-out-of-9 size.

Each head is a real leaky-tanh pass through the shared connectome. The retina
drive is what differs: trend, fade, Bollinger, activity/volume-proxy, RSI,
MACD, Kalman flow, a mixed fusion, and pullback structure.

5oo9 opens small, 9oo9 opens large. A 5-4 split does not trade.
"""

from __future__ import annotations

import numpy as np

from flyfx.brain.plasticity import HeadPlasticity

NODE_NAMES = ("trend", "fade", "boll", "vol", "rsi", "macd", "flow", "mix", "struct")

KOO9_SIZE = {5: 0.72, 6: 0.95, 7: 1.15, 8: 1.32, 9: 1.50}

TAU = {
    "trend": 0.55,
    "fade": 0.85,
    "boll": 0.52,
    "vol": 0.52,
    "rsi": 0.52,
    "macd": 0.52,
    "flow": 0.50,
    "mix": 0.50,
    "struct": 0.55,
}

SUBSTEPS_NINE = 6


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _drive(core, bull: float, bear: float, plastic: HeadPlasticity | None, floor: float = 0.05):
    out = np.zeros(core.n, dtype=np.float32)
    core.drive_into(out, bull, bear, plastic, floor=floor)
    return out


def encode_trend(core, feat: dict, plastic=None):
    impulse = float(feat.get("impulse", 0.0))
    residual = float(feat.get("kalman", {}).get("residual", 0.0))
    bull = max(0.0, impulse) * 0.75 + max(0.0, -residual) * 0.25
    bear = max(0.0, -impulse) * 0.75 + max(0.0, residual) * 0.25
    return _drive(core, bull, bear, plastic)


def encode_fade(core, feat: dict, plastic=None):
    impulse = float(feat.get("impulse", 0.0))
    residual = float(feat.get("kalman", {}).get("residual", 0.0))
    bull = max(0.0, -residual) * 0.80 + max(0.0, -impulse) * 0.20
    bear = max(0.0, residual) * 0.80 + max(0.0, impulse) * 0.20
    return _drive(core, bull, bear, plastic, floor=0.08)


def encode_boll(core, feat: dict, plastic=None):
    """Bollinger breakout: buy above the mid toward the upper band, sell below."""
    pct = float(feat.get("bb_pct", 0.5))
    bw = float(feat.get("bb_bw", 0.0))
    mag = _clip01(0.45 + 0.55 * min(bw / 0.04, 1.0))
    bull = mag * _clip01((pct - 0.50) / 0.45)
    bear = mag * _clip01((0.50 - pct) / 0.45)
    return _drive(core, bull, bear, plastic)


def encode_vol(core, feat: dict, plastic=None):
    """Activity proxy (true-range / ATR) signed by close vs prior close.

    Yahoo/HST 5m has no reliable tick volume; range expansion is the fly's
    'how loud is this bar' channel.
    """
    signed = float(feat.get("signed_vol", 0.0))
    rel = float(feat.get("vol_rel", 1.0))
    mag = _clip01((rel - 0.70) / 1.10)
    bull = mag * _clip01(signed)
    bear = mag * _clip01(-signed)
    return _drive(core, bull, bear, plastic)


def encode_rsi(core, feat: dict, plastic=None):
    """RSI momentum: >50 bull, <50 bear (not the fade stretch)."""
    rsi = float(feat.get("rsi", 50.0))
    bull = _clip01((rsi - 50.0) / 28.0)
    bear = _clip01((50.0 - rsi) / 28.0)
    return _drive(core, bull, bear, plastic)


def encode_macd(core, feat: dict, plastic=None):
    """EMA-spread MACD vs its signal line."""
    line = float(feat.get("macd", 0.0))
    sig = float(feat.get("macd_sig", 0.0))
    hist = line - sig
    bull = _clip01(max(line, 0.0) * 0.55 + max(hist, 0.0) * 0.70)
    bear = _clip01(max(-line, 0.0) * 0.55 + max(-hist, 0.0) * 0.70)
    return _drive(core, bull, bear, plastic)


def encode_flow(core, feat: dict, plastic=None):
    """Kalman fused impulse + regime — combination trend/flow node."""
    fused = float(feat.get("fused", 0.0))
    impulse = float(feat.get("impulse", 0.0))
    regime = feat.get("regime", "CHOP")
    mix = 0.55 * fused + 0.45 * impulse
    if regime == "CHOP":
        mix *= 0.35
    bull = _clip01(mix)
    bear = _clip01(-mix)
    return _drive(core, bull, bear, plastic)


def encode_mix(core, feat: dict, plastic=None):
    """Fusion node: Bollinger %b + RSI + MACD histogram, one fly."""
    pct = float(feat.get("bb_pct", 0.5))
    rsi = float(feat.get("rsi", 50.0))
    hist = float(feat.get("macd", 0.0)) - float(feat.get("macd_sig", 0.0))
    stoch = float(feat.get("stoch", 0.5))
    bull = _clip01(
        0.30 * (pct - 0.5) * 2.0
        + 0.25 * (rsi - 50.0) / 30.0
        + 0.25 * hist
        + 0.20 * (stoch - 0.5) * 2.0
    )
    bear = _clip01(
        0.30 * (0.5 - pct) * 2.0
        + 0.25 * (50.0 - rsi) / 30.0
        + 0.25 * (-hist)
        + 0.20 * (0.5 - stoch) * 2.0
    )
    return _drive(core, bull, bear, plastic)


def encode_struct(core, feat: dict, plastic=None):
    """Pullback structure: trend MA + rejection candle + RSI dip/rally."""
    close = float(feat.get("close") or 0.0)
    ema_s = float(feat.get("ema_slow") or close)
    rsi = float(feat.get("rsi", 50.0))
    pos = float(feat.get("bar_pos", 0.5))
    regime = feat.get("regime", "CHOP")
    atr = max(float(feat.get("atr") or 1e-9), 1e-9)
    hold = (close - ema_s) / atr
    bull = bear = 0.0
    if regime == "UP":
        bull = _clip01(0.40 * max(hold, 0.0) + 0.35 * pos + 0.25 * (52.0 - min(rsi, 52.0)) / 22.0)
    elif regime == "DOWN":
        bear = _clip01(0.40 * max(-hold, 0.0) + 0.35 * (1.0 - pos) + 0.25 * (max(rsi, 48.0) - 48.0) / 22.0)
    return _drive(core, bull, bear, plastic)


ENCODERS = {
    "trend": encode_trend,
    "fade": encode_fade,
    "boll": encode_boll,
    "vol": encode_vol,
    "rsi": encode_rsi,
    "macd": encode_macd,
    "flow": encode_flow,
    "mix": encode_mix,
    "struct": encode_struct,
}


def koo9_committee(votes: dict[str, str]) -> tuple[str, str, int, float, dict]:
    """Majority of 9. Need >=5 on one side and net >=2 (no 5-4 coin flip)."""
    buy_n = sum(1 for v in votes.values() if v == "BUY")
    sell_n = sum(1 for v in votes.values() if v == "SELL")
    hold_n = 9 - buy_n - sell_n
    pack = {"votes": dict(votes), "buy": buy_n, "sell": sell_n, "hold": hold_n}
    if buy_n >= 5 and buy_n > sell_n:
        side, k, opp = "BUY", buy_n, sell_n
    elif sell_n >= 5 and sell_n > buy_n:
        side, k, opp = "SELL", sell_n, buy_n
    else:
        return "HOLD", f"hold {buy_n}B/{sell_n}S", max(buy_n, sell_n), 0.0, pack
    if k - opp < 2:
        return "HOLD", f"thin {k}-{opp}", k, 0.0, pack
    k = int(min(9, max(5, k)))
    tag = f"{k}oo9"
    net_frac = (k - opp) / max(k, 1)
    mult = float(KOO9_SIZE[k]) * float(max(0.80, net_frac))
    pack["mult"] = round(mult, 3)
    return side, tag, k, mult, pack


class FlyNine:
    """Nine state vectors, one shared MaleCNS W."""

    def __init__(self, core, plastic: bool = True) -> None:
        self.core = core
        n = core.n
        n_desc = int(core.descending.size)
        self.plastic_on = bool(plastic)
        self.x = {name: np.zeros(n, dtype=np.float32) for name in NODE_NAMES}
        self.r = {name: np.zeros(n, dtype=np.float32) for name in NODE_NAMES}
        self.p = {
            name: (HeadPlasticity(n_desc) if self.plastic_on else None) for name in NODE_NAMES
        }
        self.last: dict = {"votes": {n: "HOLD" for n in NODE_NAMES}, "buy": 0, "sell": 0}

    def vote(self, feat: dict, substeps: int | None = None) -> dict:
        nstep = int(substeps) if substeps else SUBSTEPS_NINE
        nstep = max(3, min(nstep, 10))
        xs = []
        drives = []
        rewards = []
        scales = []
        plastics = []
        bulls = []
        bears = []
        for name in NODE_NAMES:
            enc = ENCODERS[name](self.core, feat, self.p[name])
            xs.append(self.x[name])
            drives.append(enc)
            rewards.append(self.r[name])
            scales.append(TAU[name])
            plastics.append(self.p[name])
            bulls.append(self.core.last_bull)
            bears.append(self.core.last_bear)
        results = self.core.run_windows_batch(
            xs, drives, rewards, nstep, scales, plastics, bulls, bears
        )
        votes: dict[str, str] = {}
        scores: dict[str, float] = {}
        nfire = 0
        for name, (vote, score, _, nf) in zip(NODE_NAMES, results):
            votes[name] = vote
            scores[name] = round(float(score), 2)
            nfire += int(nf)
        side, tag, k, mult, pack = koo9_committee(votes)
        pack["scores"] = scores
        pack["nfire"] = nfire
        pack["side"] = side
        pack["tag"] = tag
        pack["k"] = k
        pack["mult"] = round(float(mult), 3)
        self.last = pack
        return pack

    def apply_reward(self, profit: float, side: str = "") -> list[str]:
        notes: list[str] = []
        if side not in ("BUY", "SELL"):
            return notes
        for name in NODE_NAMES:
            self.core.deposit_reward(self.r[name], profit)
            head = self.p[name]
            if self.plastic_on and head is not None:
                notes.append(f"{name} {head.learn(profit, side)}")
        return notes

    def apply_reward_r(self, r_mult: float, side: str = "") -> list[str]:
        if r_mult == 0 or side not in ("BUY", "SELL"):
            return []
        proxy = 400.0 * float(np.tanh(r_mult))
        notes: list[str] = []
        for name in NODE_NAMES:
            self.core.deposit_reward(self.r[name], proxy)
            head = self.p[name]
            if self.plastic_on and head is not None:
                notes.append(f"{name} {head.learn_r(r_mult, side)}")
        return notes

    def dump_plastic(self) -> dict:
        return {name: head.dump() for name, head in self.p.items() if head is not None}

    def load_plastic(self, data: dict) -> None:
        if not isinstance(data, dict):
            return
        for name, head in self.p.items():
            if head is not None and name in data:
                head.load(data[name])

    def rollback_plastic(self) -> None:
        for head in self.p.values():
            if head is not None:
                head.rollback()
