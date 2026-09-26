"""Seven categories × three MaleCNS heads (confidence, trend, fade).

Shares the same connectome ``W`` and the same ``encode_trend`` / ``encode_fade``
/ ``encode_conf`` retina maps as ``FlySwarm``. Each category only differs by
the fused Kalman features driven into those encoders. Unavailable overlay
categories skip the network and return HOLD with conf_norm=0.
"""

from __future__ import annotations

import numpy as np

from flyfx.brain.plasticity import HeadPlasticity
from flyfx.sense.category_kalman import CATEGORY_NAMES, SPARSE_CATEGORIES
from flyfx.ui.dash import pack_head
from flyfx.sense.mtf import mtf_scale
from flyfx.vote.node_fusion import confidence_drive

ROLES = ("trend", "fade", "conf")
SUBSTEPS_CAT = 6
TAU = {"trend": 0.55, "fade": 1.0, "conf": 0.70}


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def category_feat(name: str, feat: dict, state: dict) -> dict:
    """Shim so existing FlyBrain encoders see this category as 'the' Kalman."""
    out = dict(feat)
    out["impulse"] = float(state.get("impulse") or 0.0)
    out["fused"] = float(state.get("fused") or 0.0)
    kalman = dict(feat.get("kalman") or {})
    kalman["residual"] = float(state.get("residual") or 0.0)
    kalman["kalman_regime"] = state.get("regime") or "CHOP"
    out["kalman"] = kalman
    out["regime"] = state.get("regime") or feat.get("regime") or "CHOP"
    return out


def category_conf_pack(name: str, feat: dict, state: dict, tech: dict) -> dict:
    """Robustness features for encode_conf — not price. Regime-fit included."""
    agr = float(state.get("agreement") or 0.0)
    unc = float(state.get("uncertainty") or 1.0)
    n_live = float(state.get("n_live") or 0.0)
    n_exp = max(float(state.get("n_expected") or 1.0), 1.0)
    nfrac = n_live / n_exp
    adx = float(feat.get("adx") or 20.0)
    regime = str(feat.get("regime") or "CHOP")
    if name == "trend":
        fit = _clip01((adx - 18.0) / 25.0) * (0.35 if regime == "CHOP" else 1.0)
    elif name == "momentum":
        fit = _clip01((30.0 - adx) / 22.0)
    elif name == "volatility":
        bw = float(feat.get("bb_bw") or 0.0)
        ma = float(feat.get("bb_bw_ma") or bw or 1.0)
        fit = 0.55 if ma <= 0 else _clip01(0.35 + 0.65 * (1.0 if bw <= 0.85 * ma else 0.40))
    elif name == "structure":
        dist = float((feat.get("pivot") or {}).get("dist_atr") or 1.0)
        fit = _clip01(1.0 - dist / 1.4) * 0.50 + 0.35
    elif name in SPARSE_CATEGORIES:
        fit = 0.70 if state.get("available") else 0.05
    else:
        fit = 0.50
    p_edge = _clip01(0.40 * agr + 0.35 * (1.0 - unc) + 0.25 * nfrac)
    conf_proxy = _clip01(0.50 * p_edge + 0.50 * fit)
    sign = float(tech.get("strength") or 0.0)
    if tech.get("vote") == "SELL":
        sign = -sign
    elif tech.get("vote") != "BUY":
        sign = 0.0
    fused = float(state.get("fused") or 0.0)
    return {
        "p_edge": p_edge,
        "conf_proxy": conf_proxy,
        "p_trend": _clip01(0.50 + 0.50 * abs(fused)),
        "p_fade": _clip01(0.50 + 0.50 * abs(float(state.get("residual") or 0.0))),
        "tech_fused": sign,
        "trend_fused": fused,
        "regime_fit": fit,
    }


class CategoryFlyPool:
    """21 state vectors (7×3) on one MaleCNS W. Same network size as FlySwarm."""

    def __init__(self, core, plastic: bool = True) -> None:
        self.core = core
        n = core.n
        n_desc = int(core.descending.size)
        self.plastic_on = bool(plastic)
        self.x = {(c, r): np.zeros(n, dtype=np.float32) for c in CATEGORY_NAMES for r in ROLES}
        self.r = {(c, r): np.zeros(n, dtype=np.float32) for c in CATEGORY_NAMES for r in ROLES}
        self.p = {
            (c, r): (HeadPlasticity(n_desc) if self.plastic_on else None)
            for c in CATEGORY_NAMES
            for r in ROLES
        }
        self.last: dict[str, dict] = {
            c: {"trend": "HOLD", "fade": "HOLD", "conf": "HOLD", "conf_norm": 0.0, "nfire": 0}
            for c in CATEGORY_NAMES
        }

    def vote(
        self,
        feat: dict,
        cat_states: dict[str, dict],
        tech: dict[str, dict],
        substeps: int | None = None,
        *,
        viz_category: str | None = None,
        nn_conf: float | None = None,
        nn_sign: float = 0.0,
    ) -> dict[str, dict]:
        nstep = int(substeps) if substeps else SUBSTEPS_CAT
        nstep = max(3, min(nstep, 10))
        tau0 = max(float(self.core.tau), 1e-6)
        out: dict[str, dict] = {}
        live: list[str] = []
        xs: list = []
        drives: list = []
        rewards: list = []
        scales: list[float] = []
        plastics: list = []
        bulls: list[float] = []
        bears: list[float] = []
        roles_live: list[str] = []
        for name in CATEGORY_NAMES:
            st = cat_states.get(name) or {}
            if not st.get("available"):
                out[name] = {
                    "trend": "HOLD",
                    "fade": "HOLD",
                    "conf": "HOLD",
                    "trend_score": 0.0,
                    "fade_score": 0.0,
                    "conf_score": 0.0,
                    "conf_norm": 0.0,
                    "nfire": 0,
                    "skipped": True,
                }
                continue
            live.append(name)
            cfeat = category_feat(name, feat, st)
            pack = category_conf_pack(name, feat, st, tech.get(name) or {})
            if nn_conf is not None:
                pack["nn_conf"] = float(nn_conf)
                pack["nn_sign"] = float(nn_sign)
            for role in ROLES:
                I = np.zeros(self.core.n, dtype=np.float32)
                if role == "trend":
                    impulse = float(cfeat.get("impulse", 0.0))
                    residual = float(cfeat.get("kalman", {}).get("residual", 0.0))
                    bull = max(0.0, impulse) * 0.75 + max(0.0, -residual) * 0.25
                    bear = max(0.0, -impulse) * 0.75 + max(0.0, residual) * 0.25
                    scale = mtf_scale(cfeat, "BUY" if bull >= bear else "SELL")
                    bull *= scale
                    bear *= scale
                    floor = 0.05
                elif role == "fade":
                    impulse = float(cfeat.get("impulse", 0.0))
                    residual = float(cfeat.get("kalman", {}).get("residual", 0.0))
                    bull = max(0.0, -residual) * 0.80 + max(0.0, -impulse) * 0.20
                    bear = max(0.0, residual) * 0.80 + max(0.0, impulse) * 0.20
                    scale = mtf_scale(cfeat, "BUY" if bull >= bear else "SELL")
                    bull *= scale
                    bear *= scale
                    floor = 0.08
                else:
                    bull, bear = confidence_drive(cfeat, pack)
                    floor = 0.06
                b, e = self.core.drive_into(I, bull, bear, self.p[(name, role)], floor=floor)
                xs.append(self.x[(name, role)])
                drives.append(I)
                rewards.append(self.r[(name, role)])
                scales.append(TAU[role])
                plastics.append(self.p[(name, role)])
                bulls.append(b)
                bears.append(e)
                roles_live.append(role)
            out[name] = {"pack": pack, "heads": {}}
        results = self.core.run_windows_batch(
            xs, drives, rewards, nstep, scales, plastics, bulls, bears
        )
        idx = 0
        for name in live:
            votes: dict[str, str] = {}
            scores: dict[str, float] = {}
            nfire = 0
            for role in ROLES:
                vote, score, _, nf = results[idx]
                idx += 1
                votes[role] = vote
                scores[role] = float(score)
                nfire += int(nf)
            conf_tau = tau0 * TAU["conf"]
            conf_norm = abs(float(scores["conf"])) / max(conf_tau, 1e-6)
            pack = out[name].get("pack") or {}
            out[name] = {
                "trend": votes["trend"],
                "fade": votes["fade"],
                "conf": votes["conf"],
                "trend_score": round(scores["trend"], 3),
                "fade_score": round(scores["fade"], 3),
                "conf_score": round(scores["conf"], 3),
                "conf_norm": round(float(conf_norm), 4),
                "nfire": nfire,
                "skipped": False,
                "heads": {},
                "pack": pack,
            }
        self.last = out
        return out

    def snapshot_nodes(self, flies: dict, tech: dict, cat_pack: dict) -> dict:
        """Pack all 7×4 nodes (tech + 3 flies) for the dashboard."""
        tau0 = max(float(self.core.tau), 1e-6)
        winner = str(cat_pack.get("winner") or "")
        rows = cat_pack.get("categories") or {}
        board: dict = {}
        for name in CATEGORY_NAMES:
            f = flies.get(name) or {}
            t = tech.get(name) or {}
            row = rows.get(name) or {}
            heads = {}
            for role in ROLES:
                heads[role] = pack_head(
                    self.core,
                    self.x[(name, role)],
                    str(f.get(role) or "HOLD"),
                    float(f.get(f"{role}_score") or 0.0),
                    int(f.get("nfire") or 0),
                    tau0 * TAU[role],
                    f"{name}-{role}",
                    self.p[(name, role)],
                    viz=True,
                    compact=True,
                )
            board[name] = {
                "tech": {
                    "vote": str(t.get("vote") or "HOLD"),
                    "reason": str(t.get("reason") or ""),
                    "strength": round(float(t.get("strength") or 0.0), 3),
                },
                "heads": heads,
                "decision": str(row.get("decision") or "HOLD"),
                "vote_type": str(row.get("vote_type") or "HOLD"),
                "inhibited": bool(row.get("inhibited")),
                "available": bool(row.get("available")),
                "winner": winner == name,
                "impulse": float(row.get("impulse") or 0.0),
                "conf_norm": float(f.get("conf_norm") or row.get("conf_norm") or 0.0),
                "n_agree": int(row.get("n_agree") or 0),
                "inhibit_reason": str(row.get("inhibit_reason") or ""),
            }
        return board

    def apply_reward(self, profit: float, side: str = "", winner: str = "") -> list[str]:
        notes: list[str] = []
        if side not in ("BUY", "SELL"):
            return notes
        names = [winner] if winner in CATEGORY_NAMES else list(CATEGORY_NAMES)
        for name in names:
            for role in ROLES:
                self.core.deposit_reward(self.r[(name, role)], profit)
                head = self.p[(name, role)]
                if self.plastic_on and head is not None:
                    notes.append(f"{name}-{role} {head.learn(profit, side)}")
        return notes

    def apply_reward_r(self, r_mult: float, side: str = "", winner: str = "") -> list[str]:
        if r_mult == 0 or side not in ("BUY", "SELL"):
            return []
        proxy = 400.0 * float(np.tanh(r_mult))
        notes: list[str] = []
        names = [winner] if winner in CATEGORY_NAMES else list(CATEGORY_NAMES)
        for name in names:
            for role in ROLES:
                self.core.deposit_reward(self.r[(name, role)], proxy)
                head = self.p[(name, role)]
                if self.plastic_on and head is not None:
                    notes.append(f"{name}-{role} {head.learn_r(r_mult, side)}")
        return notes

    def dump_plastic(self) -> dict:
        out: dict = {}
        for name in CATEGORY_NAMES:
            blob: dict = {}
            for role in ROLES:
                head = self.p[(name, role)]
                if head is not None:
                    blob[role] = head.dump()
            if blob:
                out[name] = blob
        return out

    def da_updates(self) -> int:
        n = 0
        for head in self.p.values():
            if head is not None:
                n += int(getattr(head, "n_updates", 0) or 0)
        return n

    def load_plastic(self, data: dict) -> None:
        if not isinstance(data, dict):
            return
        for name, blob in data.items():
            if not isinstance(blob, dict):
                continue
            for role, rec in blob.items():
                head = self.p.get((name, role))
                if head is not None and isinstance(rec, dict):
                    head.load(rec)

    def rollback_plastic(self) -> None:
        for head in self.p.values():
            if head is not None:
                head.rollback()
