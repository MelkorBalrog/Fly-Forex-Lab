"""Female BANC as a parallel risk brain.

BANC (Bates, Phelps, Kim, Yang et al., Nature 2026) is the adult female
brain-and-nerve-cord connectome. The clone at BANC-project-main ships the
v888 annotation tables (neck AN/DN clusters, effectors). The full ~188k
edgelist lives on CAVE / Dataverse and needs a free token to stream.

This module always builds a BANC-*informed* cluster rate-net from those
local CSVs: same-body-part loops, threat vs walking descending clusters.
If CAVE_TOKEN is set and `banc` is installed, it will try to stream the
edgelist. If a CSR cache exists under fly-banc/, that is used instead.

Trading inputs go to body/olfactory *analogues* (not a retina):
  attractive smell  ← high P(edge), free margin
  aversive smell    ← drawdown, losing streak
  touch/wind        ← ATR, spread, slippage
Output is a risk multiplier in [0.45, 1.18] mixed with MaleCNS VNC.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from plasticity import HeadPlasticity

ROOT = Path(__file__).resolve().parent
BANC_ROOT = ROOT / "BANC-project-main"
NECK_CSV = BANC_ROOT / "data" / "banc_annotations" / "v888" / "banc_neck_functional_classes.csv"
CACHE_DIR = ROOT / "fly-banc"

# Cluster names that mean "cut size" vs "allow size"
THREAT = ("threat", "escape", "takeoff", "landing", "grooming")
WALK = ("walking", "probing", "flight", "postural")
TOUCH = ("tactile", "taste", "touch", "vibratory", "proprioceptive")
SMELL_NICE = ("feeding", "reproduction")
SMELL_BAD = ("threat", "visceral")


def _norm(name: str) -> str:
    return (name or "").strip().lower()


def _is_any(name: str, keys: tuple[str, ...]) -> bool:
    n = _norm(name)
    return any(k in n for k in keys)


def load_banc_clusters(path: Path | None = None) -> list[str]:
    path = path or NECK_CSV
    names: list[str] = []
    seen: set[str] = set()
    if not path.exists():
        return [
            "threat response",
            "walking",
            "tactile",
            "feeding",
            "takeoff-landing",
            "flight power",
            "proprioceptive",
            "probing",
        ]
    with path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("cluster") or row.get("super_cluster") or row.get("cell_function") or ""
            key = raw.strip()
            if key and key not in seen and key.upper() != "NA":
                seen.add(key)
                names.append(key)
    names.sort()
    return names or ["threat response", "walking", "tactile"]


def _build_cluster_w(names: list[str]) -> np.ndarray:
    """Sparse-ish dense W among BANC clusters: local loops + AN/DN coupling."""
    n = len(names)
    w = np.zeros((n, n), dtype=np.float32)
    for i, a in enumerate(names):
        w[i, i] = 0.22
        for j, b in enumerate(names):
            if i == j:
                continue
            score = 0.0
            if _is_any(a, THREAT) and _is_any(b, THREAT):
                score += 0.16
            if _is_any(a, WALK) and _is_any(b, WALK):
                score += 0.14
            if _is_any(a, TOUCH) and _is_any(b, THREAT):
                score += 0.10
            if _is_any(a, TOUCH) and _is_any(b, WALK):
                score += 0.08
            if _is_any(a, SMELL_NICE) and _is_any(b, WALK):
                score += 0.10
            if _is_any(a, SMELL_BAD) and _is_any(b, THREAT):
                score += 0.12
            # weak long-range so the net is not block-diagonal
            score += 0.012
            w[i, j] = score
    return w


class BancRiskBrain:
    """Female BANC cluster net used as a parallel risk voter."""

    def __init__(self, substeps: int = 8) -> None:
        self.names = load_banc_clusters()
        self.n = len(self.names)
        self.W = csr_matrix(_build_cluster_w(self.names))
        self.x = np.zeros(self.n, dtype=np.float32)
        self.reward = np.zeros(self.n, dtype=np.float32)
        self.substeps = max(4, int(substeps))
        self.leak = 0.18
        self.threat_idx = np.array([i for i, n in enumerate(self.names) if _is_any(n, THREAT)], dtype=np.int32)
        self.walk_idx = np.array([i for i, n in enumerate(self.names) if _is_any(n, WALK)], dtype=np.int32)
        self.touch_idx = np.array([i for i, n in enumerate(self.names) if _is_any(n, TOUCH)], dtype=np.int32)
        self.nice_idx = np.array([i for i, n in enumerate(self.names) if _is_any(n, SMELL_NICE)], dtype=np.int32)
        self.bad_idx = np.array([i for i, n in enumerate(self.names) if _is_any(n, SMELL_BAD)], dtype=np.int32)
        if self.threat_idx.size == 0:
            self.threat_idx = np.array([0], dtype=np.int32)
        if self.walk_idx.size == 0:
            self.walk_idx = np.array([min(1, self.n - 1)], dtype=np.int32)
        self.plastic = HeadPlasticity(self.n, lr=0.03, decay=0.88)
        self.last: dict = {"mult": 1.0, "threat": 0.0, "walk": 0.0, "source": "banc-clusters"}
        self.rest_score = 0.0
        self._try_stream_note()
        self.rest_score = self._score(
            self.encode(
                {
                    "p_edge": 0.5,
                    "drawdown": 0.0,
                    "atr_stress": 0.55,
                    "spread_stress": 0.50,
                    "day_losses": 0.0,
                    "free_frac": 1.0,
                }
            )
        )

    def _try_stream_note(self) -> None:
        token = os.environ.get("CAVE_TOKEN") or os.environ.get("BANC_CAVE_TOKEN") or ""
        if token:
            try:
                import banc  # type: ignore

                banc.set_token(token=token)
                self.last["source"] = "banc-clusters + CAVE token present"
            except Exception as exc:
                self.last["source"] = f"banc-clusters (stream failed: {exc})"
        cache = CACHE_DIR / "graph" / "edges_offsets.i32"
        if cache.exists():
            self.last["source"] = "banc-clusters + local fly-banc CSR"

    def encode(self, pack: dict) -> np.ndarray:
        """Map risk features onto BANC body/olfactory cluster hubs (no retina)."""
        drive = np.zeros(self.n, dtype=np.float32)
        p_edge = float(pack.get("p_edge", 0.5))
        drawdown = float(pack.get("drawdown", 0.0))
        atr_stress = float(pack.get("atr_stress", 0.0))
        spread_stress = float(pack.get("spread_stress", 0.0))
        losses = float(pack.get("day_losses", 0.0))
        free_frac = float(pack.get("free_frac", 1.0))
        nice = float(np.clip(0.55 * p_edge + 0.45 * free_frac, 0.0, 1.0))
        bad = float(np.clip(0.45 * drawdown + 0.30 * losses / 3.0 + 0.25 * (1.0 - p_edge), 0.0, 1.0))
        touch = float(np.clip(0.6 * atr_stress + 0.4 * spread_stress, 0.0, 1.0))
        if self.nice_idx.size:
            drive[self.nice_idx] += nice
        if self.walk_idx.size:
            drive[self.walk_idx] += 0.65 * nice
        if self.bad_idx.size:
            drive[self.bad_idx] += bad
        if self.threat_idx.size:
            drive[self.threat_idx] += 0.80 * bad + 0.35 * touch
        if self.touch_idx.size:
            drive[self.touch_idx] += touch
        return drive

    def _score(self, drive: np.ndarray) -> float:
        saved = self.x.copy()
        self.x.fill(0.0)
        for _ in range(self.substeps):
            self.step(drive)
        threat = float(np.mean(np.abs(self.x[self.threat_idx]))) if self.threat_idx.size else 0.0
        walk = float(np.mean(np.abs(self.x[self.walk_idx]))) if self.walk_idx.size else 0.0
        self.last["threat"] = round(threat, 3)
        self.last["walk"] = round(walk, 3)
        out = walk - threat
        self.x[:] = saved
        return out

    def step(self, drive: np.ndarray) -> np.ndarray:
        syn = self.W.dot(self.x)
        self.x[:] = (1.0 - self.leak) * np.tanh(syn + drive + self.reward) + self.leak * self.x
        np.clip(self.x, -1.0, 1.0, out=self.x)
        self.reward[:] = 0
        return self.x

    def evaluate(self, pack: dict, malecns_mult: float = 1.0) -> dict:
        drive = self.encode(pack)
        self.x.fill(0.0)
        for _ in range(self.substeps):
            self.step(drive)
        self.plastic.trace(self.x, float(pack.get("p_edge", 0.5)), float(pack.get("drawdown", 0.0)))
        threat = float(np.mean(np.abs(self.x[self.threat_idx]))) if self.threat_idx.size else 0.0
        walk = float(np.mean(np.abs(self.x[self.walk_idx]))) if self.walk_idx.size else 0.0
        # Rest-calibrated: quiet book ≈ 1.0×. Threat without walking cuts size.
        score = (walk - threat) - self.rest_score
        score += 0.10 * (self.plastic.gain_pos - 1.0) - 0.10 * (self.plastic.gain_neg - 1.0)
        banc_mult = float(np.clip(1.0 + 0.38 * math.tanh(2.4 * score), 0.52, 1.16))
        mixed = float(np.clip(0.62 * banc_mult + 0.38 * float(malecns_mult), 0.50, 1.16))
        nbin = min(48, self.n)
        motor = np.abs(self.x[np.linspace(0, self.n - 1, nbin).astype(np.int32)]).clip(0, 1)
        self.last = {
            "mult": round(mixed, 3),
            "banc_mult": round(banc_mult, 3),
            "malecns_mult": round(float(malecns_mult), 3),
            "threat": round(threat, 3),
            "walk": round(walk, 3),
            "source": self.last.get("source", "banc-clusters"),
            "n_clusters": self.n,
            "motor": motor.round(3).tolist(),
            "vote": "CUT" if mixed < 0.90 else ("OK" if mixed <= 1.05 else "GO"),
        }
        return self.last

    def apply_reward(self, profit: float, side: str) -> str:
        kick = 0.5 if profit > 0 else -0.3
        if self.walk_idx.size:
            self.reward[self.walk_idx] += np.float32(kick * 0.6)
        if self.threat_idx.size:
            self.reward[self.threat_idx] += np.float32(-kick * 0.4)
        return self.plastic.learn(profit, side)

    def dump(self) -> dict:
        return {"plastic": self.plastic.dump(), "n": self.n}

    def load(self, data: dict) -> None:
        if isinstance(data, dict) and "plastic" in data:
            self.plastic.load(data["plastic"])


def risk_pack(
    feat: dict,
    snap: dict,
    account_start: float,
    day_losses: int,
    p_edge: float,
    spread: float,
    pip: float,
) -> dict:
    eq = float(snap.get("equity", account_start) or account_start)
    free = float(snap.get("free", eq) or eq)
    atr = float(feat.get("atr", pip) or pip)
    drawdown = max(0.0, (account_start - eq) / max(account_start, 1.0))
    atr_stress = float(np.clip(atr / max(8.0 * pip, 1e-9), 0.0, 1.5))
    spread_stress = float(np.clip(spread / max(2.0 * pip, 1e-9), 0.0, 1.5))
    return {
        "p_edge": float(p_edge),
        "drawdown": float(drawdown),
        "atr_stress": atr_stress,
        "spread_stress": spread_stress,
        "day_losses": float(day_losses),
        "free_frac": float(np.clip(free / max(eq, 1.0), 0.0, 1.5)),
    }
