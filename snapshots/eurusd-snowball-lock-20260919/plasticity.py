"""Reward-modulated plasticity on each fly head.

The connectome matrix W stays frozen (measured wiring). What learns is a
small three-factor rule on top of it, the same idea as PAM/PPL dopamine in
the fly: an eligibility trace of what the head just did, then a dopamine
pulse on close that nudges the descending readout and the sensory gains.

Δreadout ∝ dopamine × eligibility(descending)
Δgain    ∝ dopamine × eligibility(bull/bear drive)
"""

from __future__ import annotations

import math

import numpy as np


class HeadPlasticity:
    def __init__(self, n_desc: int, lr: float = 0.018, decay: float = 0.90) -> None:
        self.n_desc = int(n_desc)
        self.lr = float(lr)
        self.decay = float(decay)
        self.delta = np.zeros(self.n_desc, dtype=np.float32)
        self.e_desc = np.zeros(self.n_desc, dtype=np.float32)
        self.e_pos = 0.0
        self.e_neg = 0.0
        self.gain_pos = 1.0
        self.gain_neg = 1.0
        self.n_updates = 0
        self.last_r = 0.0

    def mix_readout(self, base: np.ndarray) -> np.ndarray:
        out = np.asarray(base, dtype=np.float32) + self.delta
        cap = float(max(np.max(np.abs(base)), 1.0) * 2.5)
        return np.clip(out, -cap, cap)

    def scale_drive(self, bull: float, bear: float) -> tuple[float, float]:
        return bull * self.gain_pos, bear * self.gain_neg

    def trace(self, desc: np.ndarray, bull: float, bear: float) -> None:
        d = np.asarray(desc, dtype=np.float32).ravel()
        if d.size != self.n_desc:
            d = np.resize(d, self.n_desc)
        a = 1.0 - self.decay
        self.e_desc *= self.decay
        self.e_desc += np.float32(a) * d
        self.e_pos = self.decay * self.e_pos + a * float(bull)
        self.e_neg = self.decay * self.e_neg + a * float(bear)

    def learn(self, profit: float, side: str) -> str:
        if side not in ("BUY", "SELL") or profit == 0:
            return "plastic skip"
        r = float(np.tanh(profit / 400.0))
        self.last_r = r
        sign = 1.0 if side == "BUY" else -1.0
        self.delta += np.float32(self.lr * r * sign) * self.e_desc
        self.delta *= np.float32(0.996)
        np.clip(self.delta, -2.5, 2.5, out=self.delta)
        if side == "BUY":
            self.gain_pos += self.lr * r * self.e_pos
            self.gain_neg -= 0.45 * self.lr * r * self.e_neg
        else:
            self.gain_neg += self.lr * r * self.e_neg
            self.gain_pos -= 0.45 * self.lr * r * self.e_pos
        self.gain_pos = float(np.clip(self.gain_pos, 0.62, 1.55))
        self.gain_neg = float(np.clip(self.gain_neg, 0.62, 1.55))
        self.n_updates += 1
        return (
            f"DA r={r:+.2f}  Δ||={float(np.linalg.norm(self.delta)):.3f}  "
            f"g+={self.gain_pos:.2f} g-={self.gain_neg:.2f}"
        )

    def rollback(self) -> None:
        """fly-trader shadow: freeze D back to the measured readout (W never moved)."""
        self.delta[:] = 0
        self.gain_pos = 1.0
        self.gain_neg = 1.0
        self.e_desc[:] = 0
        self.e_pos = 0.0
        self.e_neg = 0.0
        self.last_r = 0.0

    def dump(self) -> dict:
        return {
            "delta": self.delta.astype(float).tolist(),
            "gain_pos": self.gain_pos,
            "gain_neg": self.gain_neg,
            "n_updates": self.n_updates,
        }

    def load(self, data: dict) -> None:
        if not isinstance(data, dict):
            return
        d = data.get("delta")
        if isinstance(d, list) and len(d) == self.n_desc:
            self.delta = np.array(d, dtype=np.float32)
        self.gain_pos = float(np.clip(data.get("gain_pos", self.gain_pos), 0.62, 1.55))
        self.gain_neg = float(np.clip(data.get("gain_neg", self.gain_neg), 0.62, 1.55))
        self.n_updates = int(data.get("n_updates", self.n_updates))


def dopamine_kick(profit: float) -> float:
    if profit > 0:
        return 0.55 * math.tanh(profit / 350.0)
    return -0.35 * math.tanh(abs(profit) / 350.0)
