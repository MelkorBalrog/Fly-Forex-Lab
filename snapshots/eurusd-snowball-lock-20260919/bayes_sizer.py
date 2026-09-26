"""Bayesian edge tracker, evolving confidence, and adaptive trade geometry.

Confidence is a plausibility check: we only enter when the current posterior,
similar-setup kernel, and calibrated P(win) all say the setup is live.
Weights, the confidence bar, and SL/TP/trail/RSI/impulse walk after each close.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class BayesianSizer:
    def __init__(self, rng: np.random.Generator | None = None) -> None:
        # Beta prior ≈ 42% win rate, modest strength
        self.a = 3.5
        self.b = 4.8
        # Normal-known-variance update for mean win / |loss| pips
        self.w_mu, self.w_prec = 12.0, 0.12  # prior: winners larger than losers
        self.l_mu, self.l_prec = 7.0, 0.12
        self.obs_prec = 1.0 / 36.0  # σ ≈ 6 pips
        self.n_win = 0
        self.n_loss = 0
        self.rng = rng or np.random.default_rng(7)
        self.last: dict = {}

    def update(self, pips: float) -> None:
        if pips > 0:
            self.a += 1.0
            self.n_win += 1
            self.w_mu, self.w_prec = self._update_mean(self.w_mu, self.w_prec, pips)
        else:
            self.b += 1.0
            self.n_loss += 1
            self.l_mu, self.l_prec = self._update_mean(self.l_mu, self.l_prec, abs(pips))

    def _update_mean(self, mu: float, prec: float, x: float) -> tuple[float, float]:
        post_prec = prec + self.obs_prec
        post_mu = (prec * mu + self.obs_prec * x) / post_prec
        return float(post_mu), float(post_prec)

    @property
    def p_win(self) -> float:
        return float(self.a / (self.a + self.b))

    @property
    def payoff(self) -> float:
        return float(max(self.w_mu, 0.4) / max(self.l_mu, 0.4))

    def posterior(self, draws: int = 2500) -> dict:
        p = self.rng.beta(self.a, self.b, size=draws)
        w = self.rng.normal(self.w_mu, 1.0 / np.sqrt(self.w_prec), size=draws)
        l = np.abs(self.rng.normal(self.l_mu, 1.0 / np.sqrt(self.l_prec), size=draws))
        w = np.clip(w, 0.5, None)
        l = np.clip(l, 0.5, None)
        expectancy = p * w - (1.0 - p) * l
        b = w / l
        kelly = p - (1.0 - p) / b
        kelly = np.clip(kelly, 0.0, 0.4)
        out = {
            "p_win": float(p.mean()),
            "e_win": float(w.mean()),
            "e_loss": float(l.mean()),
            "payoff": float(b.mean()),
            "expectancy": float(expectancy.mean()),
            "p_edge": float((expectancy > 0.0).mean()),
            "kelly": float(np.median(kelly)),
        }
        self.last = out
        return out

    def risk_scale(self, base_risk: float) -> tuple[float, str]:
        """Scale base risk% by half-Kelly and P(edge>0). Stays in [0.35, 1.00]."""
        if base_risk <= 0:
            return 0.0, "fixed"
        post = self.posterior()
        half = 0.5 * post["kelly"]
        kelly_term = float(np.clip(half / 0.06, 0.35, 1.00))
        edge_term = 0.45 + 0.70 * post["p_edge"]
        rr_term = float(np.clip(post["payoff"] / 1.35, 0.40, 1.15))
        scale = float(np.clip(kelly_term * edge_term * rr_term, 0.35, 1.00))
        risk = base_risk * scale
        note = (
            f"bayes P(edge)={post['p_edge']:.2f}  p={post['p_win']:.2f}  "
            f"R={post['payoff']:.2f}  ½K={half:.3f}  scale={scale:.2f}  "
            f"risk {risk:.2f}%"
        )
        return risk, note

    def dump(self) -> dict:
        return {
            "a": self.a,
            "b": self.b,
            "w_mu": self.w_mu,
            "w_prec": self.w_prec,
            "l_mu": self.l_mu,
            "l_prec": self.l_prec,
            "n_win": self.n_win,
            "n_loss": self.n_loss,
        }

    def load(self, data: dict) -> None:
        self.a = float(data.get("a", self.a))
        self.b = float(data.get("b", self.b))
        self.w_mu = float(data.get("w_mu", self.w_mu))
        self.w_prec = float(data.get("w_prec", self.w_prec))
        self.l_mu = float(data.get("l_mu", self.l_mu))
        self.l_prec = float(data.get("l_prec", self.l_prec))
        self.n_win = int(data.get("n_win", self.n_win))
        self.n_loss = int(data.get("n_loss", self.n_loss))


# Default geometry (matches fly_forex.py module constants at first launch).
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "sl_atr": (1.8, 3.4),
    "tp_atr": (7.0, 16.0),
    "trail_arm_atr": (2.2, 4.6),
    "trail_gap_atr": (1.2, 2.8),
    "be_atr": (1.0, 2.2),
    "rsi_buy_arm": (46.0, 54.0),
    "rsi_buy_fire": (56.0, 66.0),
    "rsi_sell_arm": (46.0, 54.0),
    "rsi_sell_fire": (34.0, 44.0),
    "rsi_cont_lo": (28.0, 38.0),
    "rsi_cont_hi": (38.0, 46.0),
    "min_impulse": (0.40, 0.72),
    "cont_impulse": (0.65, 0.90),
    "cooldown_bars": (5.0, 14.0),
    "lock_win_pips": (6.0, 12.0),
    "max_hold_bars": (40.0, 80.0),
    "min_hold_bars": (3.0, 8.0),
}

PARAM_DEFAULTS: dict[str, float] = {
    "sl_atr": 2.4,
    "tp_atr": 12.0,
    "trail_arm_atr": 3.2,
    "trail_gap_atr": 1.8,
    "be_atr": 1.5,
    "rsi_buy_arm": 52.0,
    "rsi_buy_fire": 62.0,
    "rsi_sell_arm": 48.0,
    "rsi_sell_fire": 38.0,
    "rsi_cont_lo": 32.0,
    "rsi_cont_hi": 42.0,
    "min_impulse": 0.50,
    "cont_impulse": 0.75,
    "cooldown_bars": 8.0,
    "lock_win_pips": 8.0,
    "max_hold_bars": 64.0,
    "min_hold_bars": 4.0,
}


class AdaptiveParams:
    """SL/TP/trail/RSI/impulse that walk toward what actually paid."""

    def __init__(self) -> None:
        for name, value in PARAM_DEFAULTS.items():
            setattr(self, name, float(value))
        self.n_updates = 0
        self.log: list[str] = []

    def snapshot(self) -> dict:
        return {name: float(getattr(self, name)) for name in PARAM_DEFAULTS}

    def apply(self, data: dict) -> None:
        for name in PARAM_DEFAULTS:
            if name in data:
                lo, hi = PARAM_BOUNDS[name]
                setattr(self, name, float(np.clip(float(data[name]), lo, hi)))

    def dump(self) -> dict:
        out = self.snapshot()
        out["n_updates"] = self.n_updates
        return out

    def load(self, data: dict) -> None:
        self.apply(data)
        self.n_updates = int(data.get("n_updates", self.n_updates))

    def _set(self, name: str, value: float) -> str | None:
        lo, hi = PARAM_BOUNDS[name]
        old = float(getattr(self, name))
        new = float(np.clip(value, lo, hi))
        if abs(new - old) < 1e-4:
            return None
        setattr(self, name, new)
        return f"{name} {old:.2f}->{new:.2f}"

    def update(self, row: dict) -> str:
        """Error-driven step from exit reason, pips, and entry RSI/impulse."""
        self.n_updates += 1
        n = self.n_updates
        lr = 0.11 * (n / (n + 5.0))
        won = float(row.get("usd", 0.0)) > 0.0
        pips = float(row.get("pips", 0.0))
        conf01 = float(row.get("conf", 50.0)) / 100.0
        surprise = abs((1.0 if won else 0.0) - conf01)
        step = lr * (0.50 + 0.90 * surprise)
        reason = str(row.get("reason", "")).lower()
        notes: list[str] = []

        def nudge(name: str, delta: float) -> None:
            msg = self._set(name, float(getattr(self, name)) + step * delta)
            if msg:
                notes.append(msg)

        if "trail" in reason and won:
            if pips < 10.0:
                nudge("trail_gap_atr", 0.55)
                nudge("trail_arm_atr", 0.25)
            else:
                nudge("trail_gap_atr", -0.20)
            nudge("tp_atr", 0.20)
        elif "stop" in reason and "trail" not in reason and not won:
            nudge("sl_atr", 0.85)
            nudge("cooldown_bars", 0.80)
        elif "target" in reason:
            nudge("tp_atr", 0.45)
            nudge("trail_arm_atr", -0.25)
        elif "time" in reason:
            nudge("tp_atr", -1.10)
            nudge("max_hold_bars", -8.0)
            nudge("trail_arm_atr", -0.45)
        elif "trend flip" in reason:
            nudge("trail_arm_atr", -0.70)
            nudge("be_atr", -0.40)
        elif "chop" in reason:
            nudge("trail_arm_atr", -0.20)

        # Entry filters stay put — walking min_impulse/RSI was shrinking the book.

        line = f"params n={n}  lr={step:.3f}  " + (", ".join(notes) if notes else "held")
        self.log.append(line)
        if len(self.log) > 40:
            self.log = self.log[-40:]
        return line


class SetupScorer:
    """Per-trade confidence that is a plausibility check, not a decoration.

    Empirical weight grows with n. Component weights, the confidence bar, and
    a 3-bin Beta calibration all update after every settled trade.
    """

    def __init__(self, sizer: BayesianSizer, min_conf: float = 45.0) -> None:
        self.sizer = sizer
        self.history: list[dict] = []
        self.base_min_conf = float(min_conf)
        self.threshold = float(min_conf)
        self.p_edge_floor = 0.38
        # edge, similar, recent, ta, risk — renormalized after every close
        self.w = np.array([0.28, 0.16, 0.16, 0.24, 0.16], dtype=float)
        # Beta(2,2) = weak 50% prior in each confidence band
        self.cal = {"low": [2.0, 2.0], "mid": [2.0, 2.0], "high": [2.0, 2.0]}
        self.skips = 0
        self.skip_reasons: list[str] = []

    @staticmethod
    def _bin(conf: float) -> str:
        if conf < 50.0:
            return "low"
        if conf < 65.0:
            return "mid"
        return "high"

    def calibrated_p(self, conf: float) -> float:
        a, b = self.cal[self._bin(conf)]
        return float(a / max(a + b, 1e-9))

    def adaptive_threshold(self) -> float:
        n = self.sizer.n_win + self.sizer.n_loss
        if n < 3 or not self.history:
            self.threshold = float(np.clip(self.base_min_conf, 38.0, 72.0))
            return self.threshold
        claimed = float(np.mean([float(r.get("conf", 50.0)) / 100.0 for r in self.history[-16:]]))
        realized = float(np.mean([1.0 if float(r.get("usd", 0.0)) > 0 else 0.0 for r in self.history[-16:]]))
        overconf = claimed - realized
        shrink = n / (n + 8.0)
        ha, hb = self.cal["high"]
        high_n = ha + hb - 4.0
        high_wr = ha / max(ha + hb, 1e-9)
        thresh = self.base_min_conf
        thresh += 18.0 * max(0.0, overconf) * shrink
        thresh += 8.0 * max(0.0, 0.50 - high_wr) * min(1.0, max(high_n, 0.0) / 4.0)
        thresh -= 8.0 * max(0.0, -overconf) * shrink
        last = self.history[-5:]
        last_wr = sum(1.0 for r in last if float(r.get("usd", 0.0)) > 0) / len(last)
        if last_wr <= 0.25 and n >= 4:
            thresh += 4.0
        elif last_wr >= 0.70 and n >= 4:
            thresh -= 3.0
        self.threshold = float(np.clip(thresh, 38.0, 72.0))
        floor = 0.38 + 0.16 * max(0.0, overconf) * shrink - 0.06 * max(0.0, -overconf) * shrink
        if last_wr <= 0.25 and n >= 4:
            floor += 0.04
        self.p_edge_floor = float(np.clip(floor, 0.32, 0.52))
        return self.threshold

    def remember(self, row: dict) -> None:
        self.history.append(row)
        if len(self.history) > 80:
            self.history = self.history[-80:]
        self.sizer.update(float(row.get("pips", 0.0)))
        won = float(row.get("usd", 0.0)) > 0.0
        conf = float(row.get("conf", 50.0))
        key = self._bin(conf)
        if won:
            self.cal[key][0] += 1.0
        else:
            self.cal[key][1] += 1.0
        comps = np.array(
            [
                float(row.get("p_edge", 0.5) or 0.5),
                float(row.get("similar", 0.5) or 0.5),
                float(row.get("recent_wr", 0.5) or 0.5),
                float(row.get("ta", 0.5) or 0.5),
                float(row.get("risk", 0.5) or 0.5),
            ],
            dtype=float,
        )
        comps = np.clip(comps, 0.0, 1.0)
        if won:
            self.w *= 1.0 + 0.07 * comps
        else:
            self.w *= 1.0 - 0.09 * comps
        self.w = np.clip(self.w, 0.07, 0.42)
        self.w /= self.w.sum()
        self.adaptive_threshold()

    def _similar_win_rate(self, side: str, impulse: float, rsi: float) -> tuple[float, float]:
        """Recency-weighted kernel among past trades with like side/impulse/RSI."""
        if not self.history:
            return 0.50, 0.0
        num = 0.0
        den = 0.0
        n = len(self.history)
        for i, row in enumerate(self.history):
            recency = 0.92 ** (n - 1 - i)
            w = 0.35 * recency
            if row.get("side") == side:
                w += 0.40 * recency
            di = abs(float(row.get("impulse", 0.0)) - impulse) / 0.35
            w *= float(np.exp(-0.5 * di * di))
            dr = abs(float(row.get("rsi", 50.0)) - rsi) / 12.0
            w *= float(np.exp(-0.5 * dr * dr))
            if w < 0.04:
                continue
            num += w * (1.0 if float(row.get("usd", 0.0)) > 0 else 0.0)
            den += w
        if den < 0.20:
            return 0.50, den
        return float(num / den), float(den)

    def _ta_score(self, side: str, feat: dict, fly: str) -> float:
        impulse = abs(float(feat.get("impulse", 0.0)))
        rsi = float(feat.get("rsi", 50.0))
        residual = float(feat.get("kalman", {}).get("residual", 0.0))
        sep = float(feat.get("sep", 0.0))
        regime = feat.get("regime", "CHOP")
        aligned = 1.0 if regime == ("UP" if side == "BUY" else "DOWN") else 0.25
        imp_s = float(np.clip((impulse - 0.40) / 0.55, 0.0, 1.0))
        if side == "BUY":
            rsi_s = 1.0 - abs(rsi - 46.0) / 22.0
            dip_s = float(np.clip(-residual / 0.25, 0.0, 1.0))
        else:
            rsi_s = 1.0 - abs(rsi - 54.0) / 22.0
            dip_s = float(np.clip(residual / 0.25, 0.0, 1.0))
        rsi_s = float(np.clip(rsi_s, 0.0, 1.0))
        if fly == side:
            fly_s = 1.0
        elif fly == "HOLD":
            fly_s = 0.72
        else:
            fly_s = 0.15
        vol_s = float(np.clip(sep / 0.40, 0.0, 1.0))
        return float(
            np.clip(
                0.28 * aligned + 0.22 * imp_s + 0.18 * rsi_s + 0.14 * dip_s + 0.10 * fly_s + 0.08 * vol_s,
                0.0,
                1.0,
            )
        )

    def _risk_score(self, equity: float, risk_usd: float, stop_pips: float, post: dict) -> float:
        if equity <= 0:
            return 0.5
        frac = risk_usd / equity
        size_s = 1.0 - min(abs(frac - 0.007) / 0.012, 1.0)
        stop_s = float(np.clip((stop_pips - 4.0) / 6.0, 0.0, 1.0))
        cover = post["e_win"] / max(stop_pips, 1.0)
        rr_s = float(np.clip(cover / 1.3, 0.0, 1.0))
        return float(np.clip(0.40 * size_s + 0.25 * stop_s + 0.35 * rr_s, 0.0, 1.0))

    def score(
        self,
        side: str,
        feat: dict,
        fly: str,
        equity: float,
        risk_usd: float,
        stop_pips: float,
    ) -> dict:
        post = self.sizer.posterior()
        n = self.sizer.n_win + self.sizer.n_loss
        shrink = n / (n + 8.0)
        impulse = float(feat.get("impulse", 0.0))
        rsi = float(feat.get("rsi", 50.0))
        similar, similar_w = self._similar_win_rate(side, impulse, rsi)
        recent = self.history[-8:]
        if recent:
            recent_wr = sum(1.0 for r in recent if r.get("usd", 0.0) > 0) / len(recent)
        else:
            recent_wr = post["p_win"]
        ta = self._ta_score(side, feat, fly)
        risk = self._risk_score(equity, risk_usd, stop_pips, post)
        comps = np.array([post["p_edge"], similar, recent_wr, ta, risk], dtype=float)
        prior_w = np.array([0.32, 0.14, 0.14, 0.22, 0.18], dtype=float)
        prior = float(np.dot(prior_w, np.array([post["p_edge"], 0.50, post["p_win"], ta, risk])))
        empirical = float(np.dot(self.w, comps))
        conf01 = (1.0 - shrink) * prior + shrink * empirical
        conf = float(np.clip(100.0 * conf01, 1.0, 99.0))
        cal = self.calibrated_p(conf)
        thresh = self.adaptive_threshold()
        return {
            "conf": round(conf, 1),
            "cal_p": round(cal, 3),
            "threshold": round(thresh, 1),
            "p_edge": round(post["p_edge"], 3),
            "p_win": round(post["p_win"], 3),
            "similar": round(similar, 3),
            "similar_w": round(similar_w, 3),
            "recent_wr": round(recent_wr, 3),
            "ta": round(ta, 3),
            "risk": round(risk, 3),
            "n": n,
            "shrink": round(shrink, 3),
            "impulse": round(impulse, 3),
            "rsi": round(rsi, 1),
            "w_edge": round(float(self.w[0]), 3),
            "w_ta": round(float(self.w[3]), 3),
            "note": (
                f"conf {conf:.0f}%  bar {thresh:.0f}%  cal P(win)={cal:.2f}  "
                f"n={n}  P(edge)={post['p_edge']:.2f}  similar={similar:.2f}  "
                f"TA={ta:.2f}  data-weight={shrink:.2f}"
            ),
        }

    def gate(self, scored: dict, tag: str) -> tuple[bool, str, float]:
        """Confidence scales size. Committee decides go/no-go.

        2oo3 / 3oo3 always enter (unless confidence is garbage). 1oo3 bounce
        needs a higher bar. Evolving threshold damps size, it does not skip
        majority votes — that was the profit leak.
        """
        conf = float(scored["conf"])
        cal = float(scored.get("cal_p", 0.5) or 0.5)
        n = int(scored.get("n", 0))
        conf_mult = float(np.clip(0.78 + (conf - 45.0) * 0.010, 0.72, 1.22))
        if n >= 6 and cal < 0.38:
            conf_mult *= 0.85
        similar = float(scored.get("similar", 0.5) or 0.5)
        similar_w = float(scored.get("similar_w", 0.0) or 0.0)
        if n >= 5 and similar_w >= 1.20 and similar < 0.38:
            self.skips += 1
            note = f"skip {tag}: similar setups win {100.0 * similar:.0f}%"
            self.skip_reasons.append(note)
            return False, note, 0.0
        if tag.startswith("3oo3"):
            mult = 1.18 * conf_mult
            if tag.endswith("+flow"):
                mult *= 1.06
            return True, f"{tag} size×{mult:.2f}  conf {conf:.0f}%", mult
        if tag.startswith("2oo3"):
            mult = 1.00 * conf_mult
            if tag.endswith("+flow"):
                mult *= 1.05
            return True, f"{tag} size×{mult:.2f}  conf {conf:.0f}%", mult
        if tag == "1oo3-bounce" and conf >= max(52.0, self.threshold * 0.85):
            mult = 0.72 * conf_mult
            return True, f"{tag} size×{mult:.2f}  conf {conf:.0f}%", mult
        self.skips += 1
        note = f"skip {tag}: conf {conf:.0f}% (need 2oo3 or a high-conf bounce)"
        self.skip_reasons.append(note)
        if len(self.skip_reasons) > 40:
            self.skip_reasons = self.skip_reasons[-40:]
        return False, note, 0.0

    def plausible(self, scored: dict, fly: str, side: str) -> tuple[bool, str]:
        ok, why, _mult = self.gate(scored, "2oo3")
        _ = fly, side
        return ok, why

    def dump(self) -> dict:
        return {
            "history": self.history[-80:],
            "base_min_conf": self.base_min_conf,
            "threshold": self.threshold,
            "p_edge_floor": self.p_edge_floor,
            "w": self.w.tolist(),
            "cal": self.cal,
            "skips": self.skips,
        }

    def load(self, data: dict) -> None:
        self.history = list(data.get("history", []))[-80:]
        self.base_min_conf = float(data.get("base_min_conf", self.base_min_conf))
        self.threshold = float(data.get("threshold", self.threshold))
        self.p_edge_floor = float(data.get("p_edge_floor", self.p_edge_floor))
        w = data.get("w")
        if w is not None and len(w) == 5:
            self.w = np.array(w, dtype=float)
            self.w = np.clip(self.w, 0.07, 0.42)
            self.w /= self.w.sum()
        cal = data.get("cal")
        if isinstance(cal, dict):
            for key in ("low", "mid", "high"):
                if key in cal and len(cal[key]) == 2:
                    self.cal[key] = [float(cal[key][0]), float(cal[key][1])]
        self.skips = int(data.get("skips", 0))


def save_adapt_state(
    path: Path,
    sizer: BayesianSizer,
    scorer: SetupScorer,
    params: AdaptiveParams,
    extra: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sizer": sizer.dump(),
        "scorer": scorer.dump(),
        "params": params.dump(),
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_adapt_state(
    path: Path,
    sizer: BayesianSizer,
    scorer: SetupScorer,
    params: AdaptiveParams,
    extra: dict | None = None,
) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    if "sizer" in data:
        sizer.load(data["sizer"])
    if "scorer" in data:
        scorer.load(data["scorer"])
    if "params" in data:
        params.load(data["params"])
    if extra is not None:
        extra.update({k: v for k, v in data.items() if k not in ("sizer", "scorer", "params")})
    return True


def estimate_tax(net_usd: float, rate_pct: float, model: str = "ordinary") -> dict:
    """Estimate tax on realized net P&L. Not tax advice; rate is jurisdiction-specific."""
    model = (model or "ordinary").lower()
    rate = max(rate_pct, 0.0) / 100.0
    ltcg = 0.15
    if net_usd <= 0:
        return {
            "due": 0.0,
            "after": net_usd,
            "effective": 0.0,
            "model": model,
            "rate": rate_pct,
            "note": "no tax due on a net loss; the loss may carry forward depending on local rules",
        }
    if model in ("1256", "section1256", "60-40"):
        due = net_usd * (0.60 * ltcg + 0.40 * rate)
        label = "IRC 1256 60/40 (futures-style; 60% at 15% LTCG, 40% at ordinary rate)"
    else:
        due = net_usd * rate
        label = f"ordinary income / short-term gains at {rate_pct:.1f}%"
    return {
        "due": round(due, 2),
        "after": round(net_usd - due, 2),
        "effective": round(100.0 * due / net_usd, 2),
        "model": model,
        "rate": rate_pct,
        "note": label,
    }
