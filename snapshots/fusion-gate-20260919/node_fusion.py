"""Per-node Kalman + Bayesian fusion for 2oo3 voters.

Each committee node (tech, fly-trend, fly-fade) keeps its native vote
(pullback fire or fly head). Several evidence channels are Kalman-filtered,
inverse-variance mixed, and scored with a Beta track record. Fusion and the
confidence fly may only *abstain* a node — they never flip its side. That
keeps one-bar pullback timing while dropping casts the mix clearly opposes.
"""

from __future__ import annotations

import math

import numpy as np

from kalman_signal import Kalman1D


def _clip(z: float, lo: float = -1.5, hi: float = 1.5) -> float:
    return float(np.clip(z, lo, hi))


def _side_to_z(side: str) -> float:
    if side == "BUY":
        return 1.0
    if side == "SELL":
        return -1.0
    return 0.0


class FusedNode:
    """One decision node: Kalman-fused evidence + Beta robustness."""

    def __init__(self, name: str, vote_thresh: float = 0.26) -> None:
        self.name = name
        self.vote_thresh = float(vote_thresh)
        self.filters: dict[str, Kalman1D] = {}
        self.kf_fuse = Kalman1D(q=3.5e-4, r=1.8e-2)
        # Beta prior ~50%, modest strength
        self.a = 3.0
        self.b = 3.0
        self.last: dict = {}

    @property
    def p_win(self) -> float:
        return float(self.a / max(self.a + self.b, 1e-9))

    def observe(self, channels: dict[str, float]) -> dict:
        if not channels:
            self.last = {"vote": "HOLD", "fused": 0.0, "p_cast": 0.0, "p_robust": 0.0}
            return self.last
        FAST = {"fire", "bounce", "fly_hard"}
        zs: list[float] = []
        ps: list[float] = []
        parts: dict[str, float] = {}
        for key, raw in channels.items():
            z = _clip(float(raw))
            kf = self.filters.get(key)
            if kf is None:
                if key in FAST:
                    kf = Kalman1D(q=2.5e-2, r=8.0e-3)
                else:
                    kf = Kalman1D(q=4.5e-4, r=2.8e-2)
                self.filters[key] = kf
            x, p = kf.step(z)
            zs.append(x)
            ps.append(p)
            parts[key] = round(x, 3)
        w = 1.0 / np.maximum(np.array(ps, dtype=float), 1e-9)
        mix = float(np.dot(w, np.array(zs, dtype=float)) / w.sum())
        fused, pf = self.kf_fuse.step(mix)
        # Vote off the mix (tracks a one-bar pullback). SNR still uses the smoother.
        snr = abs(mix) / max(math.sqrt(max(pf, 1e-9)), 0.08)
        p_robust = float(1.0 / (1.0 + math.exp(-(snr - 0.85) * 2.2)))
        n = max(self.a + self.b - 6.0, 0.0)
        shrink = n / (n + 8.0)
        # After the node has a record, demand that its Beta p_win support the cast
        p_cast = (1.0 - shrink) * p_robust + shrink * p_robust * (0.40 + 0.60 * self.p_win)
        vote = "HOLD"
        if mix > self.vote_thresh:
            vote = "BUY"
        elif mix < -self.vote_thresh:
            vote = "SELL"
        self.last = {
            "vote": vote,
            "fused": round(float(mix), 4),
            "smooth": round(float(fused), 4),
            "p_cast": round(float(p_cast), 3),
            "p_robust": round(float(p_robust), 3),
            "p_win": round(self.p_win, 3),
            "snr": round(float(snr), 2),
            "parts": parts,
        }
        return self.last

    def remember(self, voted: str, won: bool) -> None:
        if voted not in ("BUY", "SELL"):
            return
        if won:
            self.a += 1.0
        else:
            self.b += 1.0

    def dump(self) -> dict:
        return {"name": self.name, "a": self.a, "b": self.b, "vote_thresh": self.vote_thresh}

    def load(self, data: dict) -> None:
        self.a = float(data.get("a", self.a))
        self.b = float(data.get("b", self.b))
        self.vote_thresh = float(data.get("vote_thresh", self.vote_thresh))


def conf_bar(conf_vote: str, tech_side: str) -> float:
    """p_cast bar the confidence fly demands when it does not agree with the node."""
    if tech_side in ("BUY", "SELL") and conf_vote == tech_side:
        return 0.32
    if conf_vote == "HOLD":
        return 0.42
    if conf_vote in ("BUY", "SELL") and tech_side in ("BUY", "SELL") and conf_vote != tech_side:
        return 0.58
    return 0.40


def gate_vote(
    native: str,
    node_out: dict,
    conf_vote: str,
    bar: float,
    *,
    oppose_drop: float = -0.12,
    use_conf: bool = True,
) -> str:
    """Cast the native vote, or HOLD. Fusion never flips a node.

    Drop the vote when the Kalman mix clearly opposes it. Optionally also
    drop when the confidence fly votes the other way and p_cast is below bar.
    Fade vetoes pass `use_conf=False` so a dump-fade still blocks chase BUYs.
    """
    if native not in ("BUY", "SELL"):
        return "HOLD"
    mix = float(node_out.get("fused", 0.0))
    p_cast = float(node_out.get("p_cast", 0.0))
    support = mix * _side_to_z(native)
    if support <= oppose_drop:
        return "HOLD"
    if use_conf and conf_vote in ("BUY", "SELL") and conf_vote != native and p_cast < bar:
        return "HOLD"
    return native


def apply_robustness(node_out: dict, bar: float, lock_side: str | None = None) -> str:
    """Legacy helper: mix-vote + p_cast bar. Prefer gate_vote()."""
    vote = str(node_out.get("vote", "HOLD"))
    if vote == "HOLD":
        return "HOLD"
    if float(node_out.get("p_cast", 0.0)) < bar:
        return "HOLD"
    if lock_side in ("BUY", "SELL") and vote != lock_side:
        return "HOLD"
    return vote


class NodeBoard:
    def __init__(self) -> None:
        self.tech = FusedNode("tech", vote_thresh=0.22)
        self.trend = FusedNode("trend", vote_thresh=0.24)
        self.fade = FusedNode("fade", vote_thresh=0.30)

    def remember_votes(self, votes: dict[str, str], won: bool) -> None:
        self.tech.remember(votes.get("tech", "HOLD"), won)
        self.trend.remember(votes.get("trend", "HOLD"), won)
        self.fade.remember(votes.get("fade", "HOLD"), won)

    def dump(self) -> dict:
        return {"tech": self.tech.dump(), "trend": self.trend.dump(), "fade": self.fade.dump()}

    def load(self, data: dict) -> None:
        if not isinstance(data, dict):
            return
        if "tech" in data:
            self.tech.load(data["tech"])
        if "trend" in data:
            self.trend.load(data["trend"])
        if "fade" in data:
            self.fade.load(data["fade"])


def tech_channels(feat: dict, raw_tech: str, kind: str) -> dict[str, float]:
    rsi = float(feat.get("rsi", 50.0))
    impulse = float(feat.get("impulse", 0.0))
    sep = float(feat.get("sep", 0.0))
    sign = 1.0 if raw_tech == "BUY" else -1.0 if raw_tech == "SELL" else 0.0
    rsi_pull = 0.0
    if raw_tech == "BUY":
        rsi_pull = (52.0 - rsi) / 22.0
    elif raw_tech == "SELL":
        rsi_pull = (rsi - 48.0) / 22.0
    bounce = 0.0
    if kind == "bounce":
        bounce = 0.85 * sign
    elif kind == "cont":
        bounce = 0.40 * sign
    return {
        "fire": sign,
        "rsi_pull": _clip(rsi_pull),
        "impulse": _clip(impulse),
        "sep": _clip(sign * min(sep / 0.40, 1.5) if sign else sep / 0.40),
        "bounce": bounce,
    }


def trend_channels(feat: dict, fly_vote: str, fly_score: float, tau: float, flow: str) -> dict[str, float]:
    impulse = float(feat.get("impulse", 0.0))
    sep = float(feat.get("sep", 0.0))
    ema_sign = 1.0 if feat.get("ema_fast", 0) >= feat.get("ema_slow", 0) else -1.0
    return {
        "fly": _clip(fly_score / max(tau, 1e-6)),
        "fly_hard": _side_to_z(fly_vote),
        "impulse": _clip(impulse),
        "flow": _side_to_z(flow),
        "ema": _clip(ema_sign * min(sep / 0.40, 1.5)),
    }


def fade_channels(feat: dict, fly_vote: str, fly_score: float, tau: float) -> dict[str, float]:
    impulse = float(feat.get("impulse", 0.0))
    residual = float(feat.get("kalman", {}).get("residual", 0.0))
    rsi = float(feat.get("rsi", 50.0))
    return {
        "fly": _clip(fly_score / max(tau, 1e-6)),
        "fly_hard": _side_to_z(fly_vote),
        "residual": _clip(-residual),  # bull when price is below the Kalman level
        "rsi_ext": _clip((50.0 - rsi) / 25.0),
        "anti_impulse": _clip(-0.55 * impulse),
    }
