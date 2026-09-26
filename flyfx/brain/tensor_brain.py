"""CUDA/tensor committee voter (4th vote → 3oo4).

Default voter is a Bayesian model average of 10 feedforward nets with
different depth, width, and activation. They share one supervised set
(forward-return labels on the same feature rows). A chronological tail is
held out; each net's posterior is a tempered score of that tail (mean
validation NLL, plus a light log-parameter tie-break). The 3oo4 vote is the
weight-averaged class distribution.

PyTorch is used when available (CUDA if a probe succeeds, else CPU). Without
torch, the same 10 structures train in numpy. Weights: ``brains/{PAIR}.bma.pt``
/ ``.bma.npz``. A legacy single MLP still loads from ``.nn.pt`` / ``.nn.npz``.

Retrain (default) transfers knowledge: warm-start prior weights, distill soft
policy from a frozen teacher, mix ``brains/{PAIR}.nn_mem.npz`` experience
replay, EWC-style L2 pull, and blend the BMA posterior. Pass ``transfer=False``
/ ``--nn-fresh`` to wipe and train from scratch.

Training also uses feature-space adversarial examples (FGSM / short PGD) by
default so the 4th voter stays reliable under noisy tape. Disable with
``adversarial=False`` / ``--no-nn-adv``.

``vision=True`` keeps the older single chart-CNN path. A full VLM is not used.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from flyfx.paths import BRAIN_DIR

FEATURE_NAMES = (
    "impulse",
    "residual",
    "uncertainty",
    "agreement",
    "rsi",
    "atr_ratio",
    "sep",
    "mom",
    "macd_hist",
    "bb_pct",
    "stoch",
    "adx",
    "di_plus",
    "di_minus",
    "sar_side",
    "supertrend_side",
    "ret1",
    "ret3",
    "ret6",
    "mtf10",
    "mtf15",
    "mtf30",
    "mtf60",
)
N_FEATURES = len(FEATURE_NAMES)
# Labels: 0=HOLD, 1=BUY, 2=SELL. Hold bins are bars to keep a trade.
N_CLASSES = 3
HOLD_BINS = (4, 8, 12, 16, 24, 32, 48)
N_HOLD = len(HOLD_BINS)
N_OUT = N_CLASSES + N_HOLD
RL_COST_PIPS = 1.2
HORIZON = 6
RET_EDGE = 0.00035  # ~3.5 pips on EURUSD for a supervised label
CHART_BARS = 32
CHART_H = 24


def nn_path(symbol: str, torch_fmt: bool = True) -> Path:
    tag = str(symbol or "PAIR").upper()
    return BRAIN_DIR / (f"{tag}.nn.pt" if torch_fmt else f"{tag}.nn.npz")


def bma_path(symbol: str, torch_fmt: bool = True) -> Path:
    tag = str(symbol or "PAIR").upper()
    return BRAIN_DIR / (f"{tag}.bma.pt" if torch_fmt else f"{tag}.bma.npz")


def mem_path(symbol: str) -> Path:
    """Replay buffer of past supervised rows so retrain does not forget."""
    tag = str(symbol or "PAIR").upper()
    return BRAIN_DIR / f"{tag}.nn_mem.npz"


# Continual-learning defaults (knowledge transfer on retrain).
MEM_CAP = 8_000
DISTILL_W = 0.45  # soft KL toward the prior net
EWC_W = 2e-4  # L2 pull toward prior weights
TRANSFER_LR = 0.35  # fine-tune LR scale vs cold start
WEIGHT_BLEND = 0.35  # keep this fraction of prior BMA posterior
# Adversarial training (feature-space FGSM / short PGD) for reliability.
ADV_EPS = 0.06  # L∞ ball on packed features
ADV_STEPS = 2  # 1 = FGSM, >1 = PGD
ADV_MIX = 0.50  # weight of adversarial loss vs clean
ADV_NOISE = 0.015  # free Gaussian feature noise each step


# Ten structures, one label set. Posterior temperature is on per-row
# validation NLL so a better holdout fit is heavier without a MAP collapse.
N_ENSEMBLE = 10
BMA_TAU = 0.05
VAL_FRAC = 0.20
ENSEMBLE_SPECS: tuple[dict[str, Any], ...] = (
    {"name": "shallow32", "hidden": (32,), "act": "relu", "dropout": 0.0},
    {"name": "shallow128", "hidden": (128,), "act": "relu", "dropout": 0.0},
    {"name": "mlp64", "hidden": (64, 64), "act": "relu", "dropout": 0.0},
    {"name": "mlp128", "hidden": (128, 128), "act": "relu", "dropout": 0.0},
    {"name": "deep32", "hidden": (32, 32, 32), "act": "relu", "dropout": 0.0},
    {"name": "wide_narrow", "hidden": (256, 32), "act": "relu", "dropout": 0.0},
    {"name": "gelu64", "hidden": (64, 64), "act": "gelu", "dropout": 0.0},
    {"name": "tanh48", "hidden": (48, 48), "act": "tanh", "dropout": 0.0},
    {"name": "bottleneck", "hidden": (96, 16, 96), "act": "relu", "dropout": 0.0},
    {"name": "drop64", "hidden": (64, 64), "act": "relu", "dropout": 0.15},
)


def spec_by_name(name: str) -> dict[str, Any]:
    for spec in ENSEMBLE_SPECS:
        if spec["name"] == name:
            return spec
    raise KeyError(name)


def _f(feat: dict, key: str, default: float = 0.0) -> float:
    try:
        v = feat.get(key, default)
        if v is None:
            return float(default)
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def pack_features(feat: dict | None, closes: list[float] | None = None) -> np.ndarray:
    """Build a fixed-length float32 vector from a live indicator feat dict."""
    feat = feat or {}
    kal = feat.get("kalman") if isinstance(feat.get("kalman"), dict) else {}
    x = np.zeros(N_FEATURES, dtype=np.float32)
    x[0] = _f(feat, "impulse")
    x[1] = _f(kal, "residual", _f(feat, "residual"))
    x[2] = _f(kal, "uncertainty", 0.5)
    x[3] = _f(kal, "agreement", 0.5)
    x[4] = (_f(feat, "rsi", 50.0) - 50.0) / 50.0
    x[5] = _f(feat, "atr_ratio", 1.0) - 1.0
    x[6] = _f(feat, "sep")
    x[7] = _f(feat, "mom")
    x[8] = _f(feat, "macd_hist")
    x[9] = _f(feat, "bb_pct", 0.5) - 0.5
    x[10] = (_f(feat, "stoch", 50.0) - 50.0) / 50.0
    x[11] = _f(feat, "adx", 20.0) / 50.0
    x[12] = _f(feat, "di_plus", 20.0) / 50.0
    x[13] = _f(feat, "di_minus", 20.0) / 50.0
    close = _f(feat, "close", 0.0)
    if close <= 0 and closes:
        close = float(closes[-1])
    sar = _f(feat, "sar", close)
    x[14] = 1.0 if sar < close else (-1.0 if sar > close else 0.0)
    st = _f(feat, "supertrend", close)
    x[15] = 1.0 if close > st else (-1.0 if close < st else 0.0)
    series = list(closes or [])
    if close > 0:
        series = series + [close]
    def _ret(n: int) -> float:
        if len(series) <= n or series[-1 - n] <= 0:
            return 0.0
        return float(series[-1] / series[-1 - n] - 1.0)
    x[16] = _ret(1)
    x[17] = _ret(3)
    x[18] = _ret(6)
    mtf = feat.get("mtf") if isinstance(feat.get("mtf"), dict) else {}
    for i, minutes in enumerate((10, 15, 30, 60)):
        row = mtf.get(minutes) or mtf.get(str(minutes)) or {}
        if not row.get("ready"):
            continue
        regime = str(row.get("regime") or "")
        sign = 1.0 if regime == "UP" else (-1.0 if regime == "DOWN" else 0.0)
        x[19 + i] = sign * abs(_f(row, "impulse"))
    np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(x, -8.0, 8.0)


def pack_chart(
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    *,
    width: int = CHART_BARS,
    height: int = CHART_H,
) -> np.ndarray:
    """Rasterize the last ``width`` OHLC bars into a (1, H, W) float32 image.

    This is the lightweight vision path (not a language VLM): price range is
    normalized per window; body + wick paint the column.
    """
    img = np.zeros((1, height, width), dtype=np.float32)
    n = min(width, len(closes), len(opens), len(highs), len(lows))
    if n < 4:
        return img
    o = np.asarray(opens[-n:], dtype=np.float64)
    h = np.asarray(highs[-n:], dtype=np.float64)
    l = np.asarray(lows[-n:], dtype=np.float64)
    c = np.asarray(closes[-n:], dtype=np.float64)
    lo = float(np.min(l))
    hi = float(np.max(h))
    span = max(hi - lo, 1e-9)
    pad = width - n
    for i in range(n):
        col = pad + i
        y_hi = int(round((1.0 - (h[i] - lo) / span) * (height - 1)))
        y_lo = int(round((1.0 - (l[i] - lo) / span) * (height - 1)))
        y_o = int(round((1.0 - (o[i] - lo) / span) * (height - 1)))
        y_c = int(round((1.0 - (c[i] - lo) / span) * (height - 1)))
        y0, y1 = sorted((max(0, min(height - 1, y_hi)), max(0, min(height - 1, y_lo))))
        img[0, y0 : y1 + 1, col] = 0.35
        b0, b1 = sorted((max(0, min(height - 1, y_o)), max(0, min(height - 1, y_c))))
        img[0, b0 : b1 + 1, col] = 1.0 if c[i] >= o[i] else 0.55
    return img


def _torch_mod():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError:
        return None, None


def _act_np(z: np.ndarray, kind: str) -> np.ndarray:
    if kind == "tanh":
        return np.tanh(z)
    if kind == "gelu":
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * z * (1.0 + np.tanh(c * (z + 0.044715 * z * z * z)))
    return np.maximum(0.0, z)


def _act_grad_np(z: np.ndarray, kind: str) -> np.ndarray:
    if kind == "tanh":
        t = np.tanh(z)
        return 1.0 - t * t
    if kind == "gelu":
        # d/dz of the tanh approximation.
        c = math.sqrt(2.0 / math.pi)
        u = c * (z + 0.044715 * z * z * z)
        tanh_u = np.tanh(u)
        sech2 = 1.0 - tanh_u * tanh_u
        du = c * (1.0 + 3.0 * 0.044715 * z * z)
        return 0.5 * (1.0 + tanh_u) + 0.5 * z * sech2 * du
    return (z > 0).astype(np.float32)


def posterior_weights(
    val_nll: np.ndarray,
    n_params: np.ndarray,
    n_train: int,
    n_val: int,
    *,
    tau: float = BMA_TAU,
) -> np.ndarray:
    """Tempered posterior from per-row validation NLL.

    A full BIC penalty (½ k log n) swamps neural nets and always picks the
    smallest one. The score is mean validation NLL, plus a light log(k) Occam
    term that only breaks near-ties. Empty validation → uniform weights.
    """
    k = int(len(val_nll))
    if k == 0:
        return np.zeros(0, dtype=np.float64)
    if n_val <= 0:
        return np.full(k, 1.0 / k, dtype=np.float64)
    mean_nll = np.asarray(val_nll, dtype=np.float64) / float(n_val)
    params = np.maximum(np.asarray(n_params, dtype=np.float64), 1.0)
    occam = np.log(params) / float(n_val)
    score = -(mean_nll + occam)
    z = score / max(float(tau), 1e-6)
    z = z - float(np.max(z))
    w = np.exp(np.clip(z, -60.0, 0.0))
    total = float(w.sum())
    if total <= 0 or not np.isfinite(total):
        return np.full(k, 1.0 / k, dtype=np.float64)
    return w / total


def bayes_average(probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``probs`` is (M, 3), ``weights`` sums to 1. Returns a length-3 distribution."""
    p = np.asarray(probs, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    if w.size != p.shape[0] or w.size == 0:
        return p.mean(axis=0)
    out = w @ p
    s = float(out.sum())
    if s <= 0 or not np.isfinite(s):
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return out / s


def _split_chrono(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Earlier rows train; the last slice scores the posterior. Same cut for every net."""
    if n < 40:
        return np.arange(n), np.zeros(0, dtype=np.int64)
    n_val = max(12, int(round(n * VAL_FRAC)))
    n_val = min(n_val, n // 4)
    if n - n_val < 24:
        return np.arange(n), np.zeros(0, dtype=np.int64)
    return np.arange(0, n - n_val), np.arange(n - n_val, n)


def pick_device(prefer_cuda: bool = True) -> str:
    """Prefer CUDA when a tiny probe succeeds; else CPU torch; else numpy.

    Blackwell (sm_120) cards may report cuda.is_available() while kernels fail —
    we probe and fall back so training still runs.
    """
    torch, _ = _torch_mod()
    if torch is None:
        return "numpy"
    if prefer_cuda and torch.cuda.is_available():
        try:
            x = torch.zeros(1, device="cuda")
            _ = (x + 1).sum().item()
            del x
            torch.cuda.empty_cache()
            return "cuda"
        except Exception:
            return "cpu"
    return "cpu"


class _TorchMLP:
    def __init__(self, hidden: int = 64, device: str = "cpu"):
        torch, nn = _torch_mod()
        assert torch is not None and nn is not None
        self.torch = torch
        self.device = torch.device(device)
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N_CLASSES),
        ).to(self.device)
        self.hidden = hidden
        self.vision = False

    def forward_logits(self, x: np.ndarray, chart: np.ndarray | None = None):
        t = self.torch.from_numpy(np.asarray(x, dtype=np.float32)).to(self.device)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        return self.net(t)

    def probs(self, x: np.ndarray, chart: np.ndarray | None = None) -> np.ndarray:
        with self.torch.no_grad():
            logits = self.forward_logits(x, chart)
            p = self.torch.softmax(logits, dim=-1).detach().cpu().numpy()
        return p[0] if p.shape[0] == 1 else p

    def state_dict(self) -> dict:
        return {k: v.detach().cpu().numpy() for k, v in self.net.state_dict().items()}

    def load_state_dict(self, blob: dict) -> None:
        sd = {k: self.torch.tensor(np.asarray(v)) for k, v in blob.items()}
        self.net.load_state_dict(sd)
        self.net.to(self.device)


class _TorchChartNet:
    """Feature MLP + tiny Conv2d on the OHLC chart raster (vision tensor path)."""

    def __init__(self, hidden: int = 64, device: str = "cpu"):
        torch, nn = _torch_mod()
        assert torch is not None and nn is not None
        self.torch = torch
        self.device = torch.device(device)
        self.vision = True
        self.hidden = hidden
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        ).to(self.device)
        chart_dim = 16 * 4 * 4
        self.head = nn.Sequential(
            nn.Linear(N_FEATURES + chart_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N_CLASSES),
        ).to(self.device)

    def parameters(self):
        return list(self.conv.parameters()) + list(self.head.parameters())

    def forward_logits(self, x: np.ndarray, chart: np.ndarray | None = None):
        t = self.torch.from_numpy(np.asarray(x, dtype=np.float32)).to(self.device)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        if chart is None:
            c = self.torch.zeros((t.shape[0], 1, CHART_H, CHART_BARS), device=self.device)
        else:
            c = self.torch.from_numpy(np.asarray(chart, dtype=np.float32)).to(self.device)
            if c.ndim == 3:
                c = c.unsqueeze(0)
            if c.shape[0] != t.shape[0]:
                c = c.expand(t.shape[0], -1, -1, -1)
        emb = self.conv(c).flatten(1)
        return self.head(self.torch.cat([t, emb], dim=1))

    def probs(self, x: np.ndarray, chart: np.ndarray | None = None) -> np.ndarray:
        with self.torch.no_grad():
            logits = self.forward_logits(x, chart)
            p = self.torch.softmax(logits, dim=-1).detach().cpu().numpy()
        return p[0] if p.shape[0] == 1 else p

    def train(self) -> None:
        self.conv.train()
        self.head.train()

    def eval(self) -> None:
        self.conv.eval()
        self.head.eval()

    def state_dict(self) -> dict:
        out = {f"conv.{k}": v.detach().cpu().numpy() for k, v in self.conv.state_dict().items()}
        out.update({f"head.{k}": v.detach().cpu().numpy() for k, v in self.head.state_dict().items()})
        out["vision"] = np.array([1], dtype=np.int32)
        return out

    def load_state_dict(self, blob: dict) -> None:
        conv_sd = {
            k[len("conv.") :]: self.torch.tensor(np.asarray(v))
            for k, v in blob.items()
            if str(k).startswith("conv.")
        }
        head_sd = {
            k[len("head.") :]: self.torch.tensor(np.asarray(v))
            for k, v in blob.items()
            if str(k).startswith("head.")
        }
        if conv_sd:
            self.conv.load_state_dict(conv_sd)
        if head_sd:
            self.head.load_state_dict(head_sd)
        self.conv.to(self.device)
        self.head.to(self.device)


class _NumpyMLP:
    """Tiny ReLU MLP using numpy 'tensors' when torch is unavailable."""

    def __init__(self, hidden: int = 64, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng(0)
        self.hidden = hidden
        scale1 = math.sqrt(2.0 / N_FEATURES)
        scale2 = math.sqrt(2.0 / hidden)
        self.w1 = (rng.normal(0, scale1, (N_FEATURES, hidden))).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.w2 = (rng.normal(0, scale2, (hidden, hidden))).astype(np.float32)
        self.b2 = np.zeros(hidden, dtype=np.float32)
        self.w3 = (rng.normal(0, scale2, (hidden, N_CLASSES))).astype(np.float32)
        self.b3 = np.zeros(N_CLASSES, dtype=np.float32)

    def forward_logits(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        h1 = np.maximum(0.0, x @ self.w1 + self.b1)
        h2 = np.maximum(0.0, h1 @ self.w2 + self.b2)
        logits = h2 @ self.w3 + self.b3
        return logits[0] if single else logits

    def probs(self, x: np.ndarray, chart: np.ndarray | None = None) -> np.ndarray:
        logits = self.forward_logits(x)
        if logits.ndim == 1:
            z = logits - logits.max()
            e = np.exp(z)
            return e / max(float(e.sum()), 1e-9)
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-9)

    def state_dict(self) -> dict:
        return {
            "w1": self.w1, "b1": self.b1,
            "w2": self.w2, "b2": self.b2,
            "w3": self.w3, "b3": self.b3,
            "hidden": self.hidden,
            "backend": "numpy",
            "vision": False,
        }

    def load_state_dict(self, blob: dict) -> None:
        self.w1 = np.asarray(blob["w1"], dtype=np.float32)
        self.b1 = np.asarray(blob["b1"], dtype=np.float32)
        self.w2 = np.asarray(blob["w2"], dtype=np.float32)
        self.b2 = np.asarray(blob["b2"], dtype=np.float32)
        self.w3 = np.asarray(blob["w3"], dtype=np.float32)
        self.b3 = np.asarray(blob["b3"], dtype=np.float32)
        self.hidden = int(blob.get("hidden") or self.w1.shape[1])


def _torch_activation(nn, kind: str):
    if kind == "gelu":
        return nn.GELU()
    if kind == "tanh":
        return nn.Tanh()
    return nn.ReLU()


class _TorchFlex:
    """Feedforward net built from an ensemble spec."""

    def __init__(self, spec: dict, device: str = "cpu"):
        torch, nn = _torch_mod()
        assert torch is not None and nn is not None
        self.torch = torch
        self.spec = spec
        self.name = str(spec["name"])
        self.device = torch.device(device)
        self.vision = False
        layers: list[Any] = []
        din = N_FEATURES
        drop = float(spec.get("dropout") or 0.0)
        for width in spec["hidden"]:
            layers.append(nn.Linear(din, int(width)))
            layers.append(_torch_activation(nn, str(spec.get("act") or "relu")))
            if drop > 0:
                layers.append(nn.Dropout(drop))
            din = int(width)
        layers.append(nn.Linear(din, N_OUT))
        self.net = nn.Sequential(*layers).to(self.device)

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))

    def forward_logits(self, x: np.ndarray, chart: np.ndarray | None = None):
        t = self.torch.from_numpy(np.asarray(x, dtype=np.float32)).to(self.device)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        return self.net(t)

    def _split_probs(self, logits):
        side = self.torch.softmax(logits[:, :N_CLASSES], dim=-1)
        hold = self.torch.softmax(logits[:, N_CLASSES:], dim=-1)
        return side, hold

    def probs(self, x: np.ndarray, chart: np.ndarray | None = None) -> np.ndarray:
        was = self.net.training
        self.net.eval()
        with self.torch.no_grad():
            side, _hold = self._split_probs(self.forward_logits(x, chart))
            p = side.detach().cpu().numpy()
        if was:
            self.net.train()
        return p[0] if p.shape[0] == 1 else p

    def hold_probs(self, x: np.ndarray) -> np.ndarray:
        was = self.net.training
        self.net.eval()
        with self.torch.no_grad():
            _side, hold = self._split_probs(self.forward_logits(x, None))
            p = hold.detach().cpu().numpy()
        if was:
            self.net.train()
        return p[0] if p.shape[0] == 1 else p

    def state_dict(self) -> dict:
        return {k: v.detach().cpu().numpy() for k, v in self.net.state_dict().items()}

    def load_state_dict(self, blob: dict) -> None:
        sd = {k: self.torch.tensor(np.asarray(v)) for k, v in blob.items() if k != "vision"}
        self.net.load_state_dict(sd)
        self.net.to(self.device)


class _NumpyFlex:
    """Variable-depth numpy MLP matching an ensemble spec."""

    def __init__(self, spec: dict, rng: np.random.Generator | None = None):
        self.spec = spec
        self.name = str(spec["name"])
        self.act = str(spec.get("act") or "relu")
        self.dropout = float(spec.get("dropout") or 0.0)
        self.vision = False
        rng = rng or np.random.default_rng(0)
        dims = [N_FEATURES, *[int(h) for h in spec["hidden"]], N_OUT]
        self.ws: list[np.ndarray] = []
        self.bs: list[np.ndarray] = []
        for din, dout in zip(dims[:-1], dims[1:]):
            scale = math.sqrt(2.0 / max(din, 1))
            self.ws.append(rng.normal(0, scale, (din, dout)).astype(np.float32))
            self.bs.append(np.zeros(dout, dtype=np.float32))

    def n_params(self) -> int:
        return int(sum(w.size + b.size for w, b in zip(self.ws, self.bs)))

    def forward_logits(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        h = x
        last = len(self.ws) - 1
        for i, (w, b) in enumerate(zip(self.ws, self.bs)):
            z = h @ w + b
            if i == last:
                return z[0] if single else z
            h = _act_np(z, self.act).astype(np.float32)
        return h[0] if single else h

    def _softmax_rows(self, logits: np.ndarray) -> np.ndarray:
        z = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-9)

    def probs(self, x: np.ndarray, chart: np.ndarray | None = None) -> np.ndarray:
        logits = np.atleast_2d(self.forward_logits(x))
        p = self._softmax_rows(logits[:, :N_CLASSES])
        return p[0] if np.asarray(x).ndim == 1 else p

    def hold_probs(self, x: np.ndarray) -> np.ndarray:
        logits = np.atleast_2d(self.forward_logits(x))
        p = self._softmax_rows(logits[:, N_CLASSES:])
        return p[0] if np.asarray(x).ndim == 1 else p

    def state_dict(self) -> dict:
        blob: dict[str, Any] = {}
        for i, (w, b) in enumerate(zip(self.ws, self.bs)):
            blob[f"w{i}"] = w
            blob[f"b{i}"] = b
        blob["n_layers"] = np.array([len(self.ws)], dtype=np.int32)
        return blob

    def load_state_dict(self, blob: dict) -> None:
        n = int(np.asarray(blob["n_layers"]).reshape(-1)[0]) if "n_layers" in blob else len(self.ws)
        self.ws = [np.asarray(blob[f"w{i}"], dtype=np.float32) for i in range(n)]
        self.bs = [np.asarray(blob[f"b{i}"], dtype=np.float32) for i in range(n)]


def _mean_nll(model: Any, x: np.ndarray, y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    p = np.asarray(model.probs(x), dtype=np.float64)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    picked = p[np.arange(len(y)), y]
    return float(-np.log(np.clip(picked, 1e-9, 1.0)).sum())


def _mean_profit(model: Any, x: np.ndarray, reward: np.ndarray) -> float:
    """Mean pips of the greedy side and hold on these rows."""
    if len(x) == 0 or len(reward) == 0:
        return 0.0
    side = np.asarray(model.probs(x), dtype=np.float64)
    hold = np.asarray(model.hold_probs(x), dtype=np.float64)
    if side.ndim == 1:
        side = side.reshape(1, -1)
        hold = hold.reshape(1, -1)
    s = np.argmax(side, axis=1)
    h = np.argmax(hold, axis=1)
    picked = reward[np.arange(len(s)), s, h]
    return float(np.mean(picked))


def _policy_grad(side: np.ndarray, hold: np.ndarray, reward: np.ndarray) -> np.ndarray:
    """Gradient of minus expected profit. Profit raises the chosen side and hold."""
    # side (B,3), hold (B,H), reward (B,3,H) in pips. HOLD column is 0.
    r_side = np.einsum("bh,bsh->bs", hold, reward)
    r_hold = np.einsum("bs,bsh->bh", side, reward)
    expect = np.einsum("bs,bs->b", side, r_side)
    g_side = side * (expect[:, None] - r_side)
    g_hold = hold * (expect[:, None] - r_hold)
    return np.concatenate([g_side, g_hold], axis=1).astype(np.float32)


def load_nn_memory(symbol: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    path = mem_path(symbol)
    if not path.exists():
        return None, None
    try:
        data = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None, None
    if "X" not in data.files or "R" not in data.files:
        return None, None
    X = np.asarray(data["X"], dtype=np.float32)
    R = np.asarray(data["R"], dtype=np.float32)
    if X.ndim != 2 or R.ndim != 3 or len(X) != len(R):
        return None, None
    if X.shape[1] != N_FEATURES or R.shape[1:] != (N_CLASSES, N_HOLD):
        return None, None
    return X, R


def save_nn_memory(symbol: str, X: np.ndarray, R: np.ndarray) -> Path:
    BRAIN_DIR.mkdir(parents=True, exist_ok=True)
    path = mem_path(symbol)
    X = np.asarray(X, dtype=np.float32)
    R = np.asarray(R, dtype=np.float32)
    if len(X) > MEM_CAP:
        X = X[-MEM_CAP:]
        R = R[-MEM_CAP:]
    np.savez_compressed(path, X=X, R=R, n=np.array([len(X)], dtype=np.int32))
    return path


def mix_with_memory(
    X: np.ndarray,
    R: np.ndarray,
    mem_X: np.ndarray | None,
    mem_R: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Prepend past rows so retrain sees old regimes + new window."""
    if mem_X is None or mem_R is None or len(mem_X) == 0:
        return X, R, 0
    if mem_X.shape[1:] != X.shape[1:] or mem_R.shape[1:] != R.shape[1:]:
        return X, R, 0
    n_new = len(X)
    # Keep up to as many old rows as new (or half MEM_CAP), preferring recent memory.
    n_old = min(len(mem_X), max(n_new, MEM_CAP // 2))
    old_X = mem_X[-n_old:]
    old_R = mem_R[-n_old:]
    return (
        np.concatenate([old_X, X], axis=0),
        np.concatenate([old_R, R], axis=0),
        int(n_old),
    )


def _clone_torch_flex(src: "_TorchFlex") -> "_TorchFlex":
    dst = _TorchFlex(src.spec, device=str(src.device))
    dst.load_state_dict(src.state_dict())
    dst.net.eval()
    for p in dst.net.parameters():
        p.requires_grad_(False)
    return dst


def _clone_numpy_flex(src: "_NumpyFlex") -> "_NumpyFlex":
    dst = _NumpyFlex(src.spec, rng=np.random.default_rng(0))
    dst.load_state_dict(src.state_dict())
    return dst


def _torch_ewc_penalty(model: "_TorchFlex", anchor: dict[str, Any], weight: float):
    if weight <= 0 or not anchor:
        torch, _ = _torch_mod()
        assert torch is not None
        return torch.zeros((), device=model.device)
    torch, _ = _torch_mod()
    assert torch is not None
    pen = torch.zeros((), device=model.device)
    for name, param in model.net.named_parameters():
        if name not in anchor:
            continue
        a = anchor[name].to(device=param.device, dtype=param.dtype)
        pen = pen + ((param - a) ** 2).sum()
    return weight * pen


def _numpy_ewc_penalty(model: "_NumpyFlex", anchor_ws: list, anchor_bs: list, weight: float) -> float:
    if weight <= 0 or not anchor_ws:
        return 0.0
    total = 0.0
    for w, aw, b, ab in zip(model.ws, anchor_ws, model.bs, anchor_bs):
        total += float(np.sum((w - aw) ** 2) + np.sum((b - ab) ** 2))
    return float(weight * total)


def _numpy_input_grad(
    model: "_NumpyFlex",
    xb: np.ndarray,
    logits_grad: np.ndarray,
    masks: list[np.ndarray | None],
    zs: list[np.ndarray],
    keep: float,
) -> np.ndarray:
    """Backprop ``dL/dlogits`` to ``dL/dx`` for FGSM on numpy flex nets."""
    grad = np.asarray(logits_grad, dtype=np.float32)
    last = len(model.ws) - 1
    for li in range(last, -1, -1):
        if li == 0:
            return (grad @ model.ws[0].T).astype(np.float32)
        dh = grad @ model.ws[li].T
        if masks[li - 1] is not None:
            dh = dh * masks[li - 1] / keep
        dh = dh * _act_grad_np(zs[li - 1], model.act).astype(np.float32)
        grad = dh.astype(np.float32)
    return np.zeros_like(xb, dtype=np.float32)


def _fgsm_numpy(x: np.ndarray, grad_x: np.ndarray, eps: float) -> np.ndarray:
    if eps <= 0:
        return x
    return (x + float(eps) * np.sign(grad_x)).astype(np.float32)


def _project_linf(x_adv: np.ndarray, x: np.ndarray, eps: float) -> np.ndarray:
    return np.clip(x_adv, x - eps, x + eps).astype(np.float32)


def _train_torch_flex(
    model: _TorchFlex,
    x: np.ndarray,
    reward: np.ndarray,
    *,
    epochs: int,
    batch: int,
    lr: float,
    teacher: _TorchFlex | None = None,
    distill_w: float = 0.0,
    ewc_w: float = 0.0,
    adversarial: bool = True,
    adv_eps: float = ADV_EPS,
    adv_steps: int = ADV_STEPS,
    adv_mix: float = ADV_MIX,
) -> None:
    torch, _nn = _torch_mod()
    assert torch is not None
    opt = torch.optim.Adam(model.net.parameters(), lr=lr)
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(model.device)
    rt = torch.from_numpy(np.asarray(reward, dtype=np.float32)).to(model.device)
    n = int(len(x))
    anchor = {k: v.detach().clone() for k, v in model.net.named_parameters()} if ewc_w > 0 else {}
    eps = float(adv_eps) if adversarial else 0.0
    steps = max(1, int(adv_steps))
    mix = float(np.clip(adv_mix, 0.0, 1.0)) if eps > 0 else 0.0
    step_size = eps / float(steps)

    def _rl_loss(xb, rb, *, with_distill: bool):
        logits = model.net(xb)
        side = torch.softmax(logits[:, :N_CLASSES], dim=-1)
        hold = torch.softmax(logits[:, N_CLASSES:], dim=-1)
        expect = (side[:, :, None] * hold[:, None, :] * rb).sum(dim=(1, 2))
        loss = -expect.mean()
        if with_distill and teacher is not None and distill_w > 0:
            with torch.no_grad():
                t_logits = teacher.net(xb)
                t_side = torch.softmax(t_logits[:, :N_CLASSES], dim=-1)
                t_hold = torch.softmax(t_logits[:, N_CLASSES:], dim=-1)
            kl_s = (t_side * (t_side.clamp_min(1e-8).log() - side.clamp_min(1e-8).log())).sum(-1).mean()
            kl_h = (t_hold * (t_hold.clamp_min(1e-8).log() - hold.clamp_min(1e-8).log())).sum(-1).mean()
            loss = loss + float(distill_w) * (kl_s + kl_h)
        return loss

    def _make_adv(x_clean, rb):
        """Short PGD / FGSM in feature space: maximize RL loss (hurt expected pips)."""
        x_adv = x_clean.detach()
        if ADV_NOISE > 0:
            x_adv = x_adv + ADV_NOISE * torch.randn_like(x_adv)
        for _ in range(steps):
            x_var = x_adv.detach().requires_grad_(True)
            loss_atk = _rl_loss(x_var, rb, with_distill=False)
            grad = torch.autograd.grad(loss_atk, x_var, retain_graph=False, create_graph=False)[0]
            x_adv = x_adv + step_size * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x_clean + eps), x_clean - eps)
        return x_adv.detach()

    model.net.train()
    for _ep in range(max(1, epochs)):
        perm = torch.randperm(n, device=model.device)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            x_clean = xt[idx]
            rb = rt[idx]
            if ADV_NOISE > 0 and eps <= 0:
                x_clean = x_clean + ADV_NOISE * torch.randn_like(x_clean)
            loss_clean = _rl_loss(x_clean, rb, with_distill=True)
            if mix > 0 and eps > 0:
                x_adv = _make_adv(xt[idx], rb)
                loss_adv = _rl_loss(x_adv, rb, with_distill=True)
                loss = (1.0 - mix) * loss_clean + mix * loss_adv
            else:
                loss = loss_clean
            if ewc_w > 0:
                loss = loss + _torch_ewc_penalty(model, anchor, float(ewc_w))
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.net.eval()


def _train_numpy_flex(
    model: _NumpyFlex,
    x: np.ndarray,
    reward: np.ndarray,
    *,
    epochs: int,
    batch: int,
    lr: float,
    seed: int,
    teacher: _NumpyFlex | None = None,
    distill_w: float = 0.0,
    ewc_w: float = 0.0,
    adversarial: bool = True,
    adv_eps: float = ADV_EPS,
    adv_steps: int = ADV_STEPS,
    adv_mix: float = ADV_MIX,
) -> None:
    rng = np.random.default_rng(seed)
    n = int(len(x))
    x = np.asarray(x, dtype=np.float32)
    reward = np.asarray(reward, dtype=np.float32)
    keep = 1.0 - model.dropout
    anchor_ws = [w.copy() for w in model.ws] if ewc_w > 0 else []
    anchor_bs = [b.copy() for b in model.bs] if ewc_w > 0 else []
    eps = float(adv_eps) if adversarial else 0.0
    steps = max(1, int(adv_steps))
    mix = float(np.clip(adv_mix, 0.0, 1.0)) if eps > 0 else 0.0
    step_size = eps / float(steps)

    def _forward_batch(xb: np.ndarray, train_drop: bool):
        hs: list[np.ndarray] = []
        zs: list[np.ndarray] = []
        masks: list[np.ndarray | None] = []
        h = xb
        last = len(model.ws) - 1
        for li, (w, b) in enumerate(zip(model.ws, model.bs)):
            z = h @ w + b
            hs.append(h)
            zs.append(z)
            if li == last:
                masks.append(None)
                break
            a = _act_np(z, model.act).astype(np.float32)
            if train_drop and model.dropout > 0 and keep > 0:
                mask = (rng.random(a.shape) >= model.dropout).astype(np.float32)
                a = a * mask / keep
                masks.append(mask)
            else:
                masks.append(None)
            h = a
        logits = zs[-1]
        side = model._softmax_rows(logits[:, :N_CLASSES])
        hold = model._softmax_rows(logits[:, N_CLASSES:])
        return hs, zs, masks, logits, side, hold

    def _logits_grad(side, hold, rb, xb):
        grad = _policy_grad(side, hold, rb)
        if teacher is not None and distill_w > 0:
            t_logits = np.atleast_2d(teacher.forward_logits(xb))
            t_side = teacher._softmax_rows(t_logits[:, :N_CLASSES])
            t_hold = teacher._softmax_rows(t_logits[:, N_CLASSES:])
            grad_side = (side - t_side) * float(distill_w)
            grad_hold = (hold - t_hold) * float(distill_w)
            grad = grad + np.concatenate([grad_side, grad_hold], axis=1).astype(np.float32)
        return grad

    def _apply_grad(hs, zs, masks, grad, scale: float):
        last = len(model.ws) - 1
        g = grad
        for li in range(last, -1, -1):
            dw = hs[li].T @ g
            db = g.sum(axis=0)
            if ewc_w > 0 and anchor_ws:
                dw = dw + float(ewc_w) * (model.ws[li] - anchor_ws[li])
                db = db + float(ewc_w) * (model.bs[li] - anchor_bs[li])
            if li > 0:
                dh = g @ model.ws[li].T
                if masks[li - 1] is not None:
                    dh = dh * masks[li - 1] / keep
                dh = dh * _act_grad_np(zs[li - 1], model.act).astype(np.float32)
                g = dh.astype(np.float32)
            model.ws[li] -= (lr * scale * dw).astype(np.float32)
            model.bs[li] -= (lr * scale * db).astype(np.float32)

    def _make_adv(xb, rb):
        x_adv = xb.copy()
        if ADV_NOISE > 0:
            x_adv = x_adv + ADV_NOISE * rng.standard_normal(x_adv.shape).astype(np.float32)
        for _ in range(steps):
            _hs, zs, masks, _logits, side, hold = _forward_batch(x_adv, train_drop=False)
            # Attack maximizes RL loss ≡ uses policy grad direction on inputs.
            g_logits = _policy_grad(side, hold, rb)
            g_x = _numpy_input_grad(model, x_adv, g_logits, masks, zs, keep)
            x_adv = _project_linf(x_adv + step_size * np.sign(g_x), xb, eps)
        return x_adv

    for _ep in range(max(1, epochs)):
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            xb = x[idx]
            rb = reward[idx]
            if ADV_NOISE > 0 and eps <= 0:
                xb = xb + ADV_NOISE * rng.standard_normal(xb.shape).astype(np.float32)
            hs, zs, masks, _logits, side, hold = _forward_batch(xb, train_drop=True)
            g_clean = _logits_grad(side, hold, rb, xb)
            if mix > 0 and eps > 0:
                x_adv = _make_adv(x[idx], rb)
                hs_a, zs_a, masks_a, _la, side_a, hold_a = _forward_batch(x_adv, train_drop=True)
                g_adv = _logits_grad(side_a, hold_a, rb, x_adv)
                _apply_grad(hs, zs, masks, g_clean, scale=1.0 - mix)
                _apply_grad(hs_a, zs_a, masks_a, g_adv, scale=mix)
            else:
                _apply_grad(hs, zs, masks, g_clean, scale=1.0)


@dataclass
class TensorVote:
    side: str  # BUY / SELL / HOLD
    conf: float
    probs: tuple[float, float, float]
    device: str
    weights: tuple[float, ...] = ()
    hold_bars: int = 0


class TensorBrain:
    """4th voter: Bayesian average of 10 nets, or a single chart CNN when vision is on."""

    def __init__(
        self,
        symbol: str = "EURUSD",
        *,
        prefer_cuda: bool = True,
        hidden: int = 64,
        min_conf: float = 0.45,
        vision: bool = False,
        ensemble: bool = False,
    ):
        self.symbol = str(symbol or "EURUSD").upper()
        self.prefer_cuda = bool(prefer_cuda)
        self.hidden = int(hidden)
        self.min_conf = float(min_conf)
        self.vision = bool(vision)
        self.device = pick_device(prefer_cuda)
        self.backend = "torch" if self.device in ("cuda", "cpu") and _torch_mod()[0] is not None else "numpy"
        # Chart CNN needs torch. Without it, the 4th voter is the 10-net average.
        if self.vision and self.backend != "torch":
            self.vision = False
            ensemble = True
        self.ensemble = bool(ensemble) and not self.vision
        self.members: list[Any] = []
        self.member_names: list[str] = []
        self.weights = np.ones(N_ENSEMBLE, dtype=np.float64) / N_ENSEMBLE
        self.model: Any = None
        if self.ensemble:
            self._build_members()
            self.model = self.members[0]
        elif self.backend == "torch" and self.vision:
            self.model = _TorchChartNet(hidden=self.hidden, device=self.device)
        elif self.backend == "torch":
            self.model = _TorchMLP(hidden=self.hidden, device=self.device)
        else:
            self.device = "numpy"
            self.vision = False  # chart CNN needs torch
            self.model = _NumpyMLP(hidden=self.hidden)
        self.n_train = 0
        self.trained_at = ""
        tag = f"bma×{len(self.members)}" if self.ensemble else ("+vision" if self.vision else "")
        self.note = f"untrained {self.backend}/{self.device} {tag}".strip()
        self._opens: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._closes: list[float] = []

    def _build_members(self, names: list[str] | None = None) -> None:
        specs = [spec_by_name(n) for n in names] if names else list(ENSEMBLE_SPECS)
        self.members = []
        self.member_names = []
        for i, spec in enumerate(specs):
            if self.backend == "torch":
                self.members.append(_TorchFlex(spec, device=self.device))
            else:
                self.members.append(_NumpyFlex(spec, rng=np.random.default_rng(1000 + i * 97)))
            self.member_names.append(str(spec["name"]))
        self.weights = np.ones(len(self.members), dtype=np.float64) / max(len(self.members), 1)

    def observe(self, bar: dict | None) -> None:
        """Feed live/sim OHLC so the vision path can rasterize a chart."""
        if not bar:
            return
        try:
            self._opens.append(float(bar.get("open") or bar.get("close") or 0.0))
            self._highs.append(float(bar.get("high") or bar.get("close") or 0.0))
            self._lows.append(float(bar.get("low") or bar.get("close") or 0.0))
            self._closes.append(float(bar.get("close") or 0.0))
        except (TypeError, ValueError):
            return
        keep = CHART_BARS + 8
        if len(self._closes) > keep:
            self._opens = self._opens[-keep:]
            self._highs = self._highs[-keep:]
            self._lows = self._lows[-keep:]
            self._closes = self._closes[-keep:]

    def _chart_now(self) -> np.ndarray | None:
        if not self.vision:
            return None
        return pack_chart(self._opens, self._highs, self._lows, self._closes)

    def _predict_probs(self, x: np.ndarray, chart: np.ndarray | None) -> np.ndarray:
        if self.ensemble and self.members:
            rows = []
            holds = []
            for member in self.members:
                p = np.asarray(member.probs(x, None), dtype=np.float64).reshape(-1)
                if p.size < 3:
                    p = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                rows.append(p[:3])
                h = np.asarray(member.hold_probs(x), dtype=np.float64).reshape(-1)
                if h.size < N_HOLD:
                    h = np.full(N_HOLD, 1.0 / N_HOLD)
                holds.append(h[:N_HOLD])
            self._hold_dist = bayes_average(np.stack(holds, axis=0), self.weights)
            return bayes_average(np.stack(rows, axis=0), self.weights)
        p = np.asarray(self.model.probs(x, chart), dtype=np.float64).reshape(-1)
        if p.size < 3:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return p[:3]

    def vote(self, feat: dict | None, closes: list[float] | None = None) -> TensorVote:
        x = pack_features(feat, closes if closes is not None else self._closes)
        chart = self._chart_now()
        p = self._predict_probs(x, chart)
        idx = int(np.argmax(p))
        conf = float(p[idx])
        side = ("HOLD", "BUY", "SELL")[idx]
        if conf < self.min_conf:
            side = "HOLD"
        w = tuple(float(v) for v in self.weights) if self.ensemble else ()
        dist = getattr(self, "_hold_dist", None)
        if dist is None and hasattr(self.model, "hold_probs"):
            dist = np.asarray(self.model.hold_probs(x), dtype=np.float64).reshape(-1)
        hold_bars = 0
        if dist is not None and dist.size == N_HOLD:
            hold_bars = int(round(float(np.dot(dist, HOLD_BINS))))
        return TensorVote(
            side=side,
            conf=conf,
            probs=(float(p[0]), float(p[1]), float(p[2])),
            device=self.device,
            weights=w,
            hold_bars=hold_bars,
        )

    def _meta(self) -> dict:
        return {
            "symbol": self.symbol,
            "backend": self.backend,
            "device": self.device,
            "hidden": self.hidden,
            "n_train": self.n_train,
            "trained_at": self.trained_at,
            "note": self.note,
            "features": list(FEATURE_NAMES),
            "vision": bool(self.vision),
            "ensemble": bool(self.ensemble),
            "names": list(self.member_names),
            "weights": [float(v) for v in self.weights],
        }

    def save(self) -> Path:
        BRAIN_DIR.mkdir(parents=True, exist_ok=True)
        meta = self._meta()
        if self.ensemble and self.members:
            members = [{"name": m.name, "state": m.state_dict()} for m in self.members]
            if self.backend == "torch":
                path = bma_path(self.symbol, torch_fmt=True)
                torch, _ = _torch_mod()
                assert torch is not None
                torch.save({"meta": meta, "members": members}, path)
                flat: dict[str, Any] = {"meta": json.dumps(meta), "weights": np.asarray(self.weights, dtype=np.float64)}
                for i, member in enumerate(self.members):
                    for key, val in member.state_dict().items():
                        if key == "vision":
                            continue
                        flat[f"m{i}_{key}"] = val
                np.savez_compressed(bma_path(self.symbol, torch_fmt=False), **flat)
                return path
            path = bma_path(self.symbol, torch_fmt=False)
            flat = {"meta": json.dumps(meta), "weights": np.asarray(self.weights, dtype=np.float64)}
            for i, member in enumerate(self.members):
                for key, val in member.state_dict().items():
                    flat[f"m{i}_{key}"] = val
            np.savez_compressed(path, **flat)
            return path
        if self.backend == "torch":
            path = nn_path(self.symbol, torch_fmt=True)
            torch, _ = _torch_mod()
            assert torch is not None
            torch.save({"meta": meta, "state": self.model.state_dict()}, path)
            np.savez_compressed(
                nn_path(self.symbol, torch_fmt=False),
                meta=json.dumps(meta),
                **{k: v for k, v in self.model.state_dict().items() if k != "vision"},
            )
            return path
        path = nn_path(self.symbol, torch_fmt=False)
        blob = self.model.state_dict()
        np.savez_compressed(path, meta=json.dumps(meta), **{k: v for k, v in blob.items() if k != "vision"})
        return path

    def _apply_meta(self, meta: dict) -> None:
        self.n_train = int(meta.get("n_train") or 0)
        self.trained_at = str(meta.get("trained_at") or "")
        self.note = str(meta.get("note") or "loaded")
        weights = meta.get("weights") or []
        if weights and len(weights) == len(self.members):
            self.weights = np.asarray(weights, dtype=np.float64)

    def _load_bma(self) -> bool:
        pt = bma_path(self.symbol, torch_fmt=True)
        npz = bma_path(self.symbol, torch_fmt=False)
        if self.backend == "torch" and pt.exists():
            torch, _ = _torch_mod()
            assert torch is not None
            try:
                blob = torch.load(pt, map_location="cpu", weights_only=False)
            except TypeError:
                blob = torch.load(pt, map_location="cpu")
            meta = blob.get("meta") or {}
            names = list(meta.get("names") or [])
            self.ensemble = True
            self.vision = False
            self._build_members(names or None)
            saved = blob.get("members") or []
            for member, item in zip(self.members, saved):
                member.load_state_dict(item.get("state") or {})
            self._apply_meta(meta)
            return True
        if not npz.exists():
            return False
        data = np.load(npz, allow_pickle=False)
        meta: dict = {}
        if "meta" in data.files:
            try:
                meta = json.loads(str(data["meta"]))
            except Exception:
                meta = {}
        names = list(meta.get("names") or [])
        if self.backend == "torch":
            self.ensemble = True
            self.vision = False
            self._build_members(names or None)
        else:
            self.backend = "numpy"
            self.device = "numpy"
            self.ensemble = True
            self.vision = False
            self._build_members(names or None)
        for i, member in enumerate(self.members):
            prefix = f"m{i}_"
            state = {k[len(prefix):]: data[k] for k in data.files if k.startswith(prefix)}
            if state:
                member.load_state_dict(state)
        if "weights" in data.files:
            self.weights = np.asarray(data["weights"], dtype=np.float64)
        self._apply_meta(meta)
        return True

    def load(self) -> bool:
        if self._load_bma():
            return True
        pt = nn_path(self.symbol, torch_fmt=True)
        npz = nn_path(self.symbol, torch_fmt=False)
        if self.backend == "torch" and pt.exists():
            torch, _ = _torch_mod()
            assert torch is not None
            try:
                blob = torch.load(pt, map_location="cpu", weights_only=False)
            except TypeError:
                blob = torch.load(pt, map_location="cpu")
            meta = blob.get("meta") or {}
            want_vision = bool(meta.get("vision")) or any(
                str(k).startswith("conv.") for k in (blob.get("state") or {})
            )
            if want_vision and not isinstance(self.model, _TorchChartNet):
                self.vision = True
                self.model = _TorchChartNet(hidden=int(meta.get("hidden") or self.hidden), device=self.device)
            self.model.load_state_dict(blob["state"])
            self.ensemble = False
            self.members = []
            self.n_train = int(meta.get("n_train") or 0)
            self.trained_at = str(meta.get("trained_at") or "")
            self.note = str(meta.get("note") or "loaded torch")
            self.vision = bool(meta.get("vision") or getattr(self.model, "vision", False))
            return True
        if npz.exists():
            data = np.load(npz, allow_pickle=False)
            meta = {}
            if "meta" in data.files:
                try:
                    meta = json.loads(str(data["meta"]))
                except Exception:
                    meta = {}
            if self.backend == "torch":
                keys = [k for k in data.files if k != "meta"]
                if keys and all(k.startswith("net.") or k in ("w1", "w2", "w3") or k.startswith("conv.") or k.startswith("head.") for k in keys):
                    if "w1" in data.files:
                        self.backend = "numpy"
                        self.device = "numpy"
                        self.vision = False
                        self.model = _NumpyMLP(hidden=int(meta.get("hidden") or self.hidden))
                        self.model.load_state_dict({k: data[k] for k in ("w1", "b1", "w2", "b2", "w3", "b3") if k in data.files})
                    else:
                        if any(k.startswith("conv.") for k in keys) and not isinstance(self.model, _TorchChartNet):
                            self.vision = True
                            self.model = _TorchChartNet(hidden=int(meta.get("hidden") or self.hidden), device=self.device)
                        self.model.load_state_dict({k: data[k] for k in keys})
            else:
                self.model.load_state_dict({k: data[k] for k in ("w1", "b1", "w2", "b2", "w3", "b3") if k in data.files})
            self.ensemble = False
            self.members = []
            self.n_train = int(meta.get("n_train") or 0)
            self.trained_at = str(meta.get("trained_at") or "")
            self.note = str(meta.get("note") or "loaded npz")
            return True
        return False


def build_supervised_set(
    bars: list[dict],
    *,
    mixer: str = "cat_fuse",
    fuse_cats: str = "",
    fuse_inds: str = "",
    horizon: int = HORIZON,
    edge: float = RET_EDGE,
    vision: bool = False,
    with_reward: bool = False,
) -> tuple:
    """Walk history with Indicators; label by forward return sign.

    When ``vision`` is True also returns chart tensors ``C`` shaped (N, 1, H, W).
    """
    from flyfx.sense.category_kalman import parse_fuse_inds
    from flyfx.trader import Indicators

    empty_x = np.zeros((0, N_FEATURES), dtype=np.float32)
    empty_y = np.zeros((0,), dtype=np.int64)
    empty_c = np.zeros((0, 1, CHART_H, CHART_BARS), dtype=np.float32)
    if not bars or len(bars) < horizon + 40:
        return (empty_x, empty_c, empty_y) if vision else (empty_x, empty_y)

    from flyfx.sense.mtf import MultiTimeframe

    ind = Indicators(pip=0.0001, mixer=mixer)
    mtf = MultiTimeframe(pip=0.0001, mixer=mixer)
    if mixer == "cat_fuse":
        ind.fuse_inds = parse_fuse_inds(fuse_inds or "")
        ind.cats.channel_include = ind.fuse_inds
    xs: list[np.ndarray] = []
    cs: list[np.ndarray] = []
    ys: list[int] = []
    rewards: list[np.ndarray] = []
    max_hold = max(HOLD_BINS)
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    feats: list[dict] = []
    for bar in bars:
        feat = ind.update(bar["high"], bar["low"], bar["close"], bar.get("volume"))
        feat = dict(feat)
        feat["close"] = float(bar["close"])
        feat["mtf"] = mtf.update(bar)
        opens.append(float(bar.get("open") or bar["close"]))
        highs.append(float(bar["high"]))
        lows.append(float(bar["low"]))
        closes.append(float(bar["close"]))
        feats.append(feat)
    ahead = max(horizon, max_hold if with_reward else horizon)
    for i in range(len(feats) - ahead):
        if not feats[i].get("ready"):
            continue
        c0 = closes[i]
        c1 = closes[i + horizon]
        if c0 <= 0:
            continue
        ret = c1 / c0 - 1.0
        if ret >= edge:
            y = 1  # BUY
        elif ret <= -edge:
            y = 2  # SELL
        else:
            y = 0  # HOLD
        xs.append(pack_features(feats[i], closes[: i + 1]))
        if vision:
            cs.append(
                pack_chart(
                    opens[: i + 1],
                    highs[: i + 1],
                    lows[: i + 1],
                    closes[: i + 1],
                )
            )
        ys.append(y)
        if with_reward:
            pip = 0.01 if c0 >= 20.0 else 0.0001
            table = np.zeros((N_CLASSES, N_HOLD), dtype=np.float32)
            for hi, hold in enumerate(HOLD_BINS):
                j = i + hold
                if j >= len(closes):
                    continue
                move = (closes[j] - c0) / pip
                table[1, hi] = np.float32(move - RL_COST_PIPS)
                table[2, hi] = np.float32(-move - RL_COST_PIPS)
            rewards.append(table)
    if not xs:
        return (empty_x, empty_c, empty_y) if vision else (empty_x, empty_y)
    X = np.stack(xs, axis=0)
    y = np.asarray(ys, dtype=np.int64)
    if with_reward:
        R = np.stack(rewards, axis=0) if rewards else np.zeros((0, N_CLASSES, N_HOLD), dtype=np.float32)
        if vision:
            return X, np.stack(cs, axis=0), y, R
        return X, y, R
    if vision:
        return X, np.stack(cs, axis=0), y
    return X, y


def train_tensor_brain(
    bars: list[dict],
    symbol: str,
    *,
    epochs: int = 25,
    batch: int = 128,
    lr: float = 1e-3,
    prefer_cuda: bool = True,
    mixer: str = "cat_fuse",
    fuse_cats: str = "",
    fuse_inds: str = "",
    quiet: bool = False,
    vision: bool = False,
    transfer: bool = True,
    adversarial: bool = True,
) -> TensorBrain:
    """Supervise on historic forward returns; save weights under brains/.

    Without ``vision``, trains all 10 ensemble nets on the same rows and sets
    Bayesian weights from a chronological validation tail.

    When ``transfer`` is True (default) and prior weights exist, retrain
    warm-starts from those nets, distills soft policy from a frozen teacher,
    mixes an experience buffer of past rows, and blends the BMA posterior —
    so recalibrate does not wipe what the ANNs already learned.

    When ``adversarial`` is True (default), each step also trains on FGSM/PGD
    feature-space attacks so the voter stays reliable under tape noise.
    """
    brain = TensorBrain(symbol, prefer_cuda=prefer_cuda, vision=vision, ensemble=not vision)
    transferred = False
    prior_n = 0
    prior_weights = None
    if transfer and not vision:
        if brain.load():
            transferred = True
            prior_n = int(brain.n_train or 0)
            prior_weights = np.asarray(brain.weights, dtype=np.float64).copy()
            if not quiet:
                print(
                    f"nn-brain  transfer  warm-start from prior  "
                    f"n_prior={prior_n}  bins={len(brain.members)}  "
                    f"{brain.backend}/{brain.device}"
                )
    elif transfer and vision and brain.load():
        transferred = True
        prior_n = int(brain.n_train or 0)
        if not quiet:
            print(f"nn-brain  transfer  warm-start vision net  n_prior={prior_n}")

    packed = build_supervised_set(
        bars,
        mixer=mixer,
        fuse_cats=fuse_cats,
        fuse_inds=fuse_inds,
        vision=brain.vision,
        with_reward=bool(brain.ensemble),
    )
    R = None
    if brain.vision and brain.ensemble:
        X, C, y, R = packed  # type: ignore[misc]
    elif brain.vision:
        X, C, y = packed  # type: ignore[misc]
    elif brain.ensemble:
        X, y, R = packed  # type: ignore[misc]
        C = None
    else:
        X, y = packed  # type: ignore[misc]
        C = None
    if len(y) < 40:
        brain.note = f"too few labels ({len(y)}) — untrained"
        if not quiet:
            print(f"nn-brain  skip train  {brain.note}")
        return brain

    train_lr = float(lr) * (TRANSFER_LR if transferred else 1.0)
    distill_w = DISTILL_W if transferred else 0.0
    ewc_w = EWC_W if transferred else 0.0
    use_adv = bool(adversarial)
    n_replay = 0

    if brain.ensemble and brain.members:
        assert R is not None
        mem_X, mem_R = (None, None)
        if transferred:
            mem_X, mem_R = load_nn_memory(symbol)
            X_mix, R_mix, n_replay = mix_with_memory(X, R, mem_X, mem_R)
            if n_replay and not quiet:
                print(
                    f"nn-brain  transfer  replay {n_replay} prior rows + {len(X)} new  "
                    f"(mem {0 if mem_X is None else len(mem_X)})"
                )
        else:
            X_mix, R_mix = X, R

        # Chrono split on the *new* window only for validation honesty; train on mix.
        train_idx, val_idx = _split_chrono(len(y))
        x_tr = X_mix
        r_tr = R_mix
        # Validation stays on the new chronological tail (no replay leakage).
        r_va = R[val_idx] if len(val_idx) else R[:0]
        x_va = X[val_idx] if len(val_idx) else X[:0]
        # Also keep a train slice of new labels for reporting.
        n_new_train = int(len(train_idx))

        teachers: list[Any] = []
        if transferred:
            if brain.backend == "torch":
                teachers = [_clone_torch_flex(m) for m in brain.members]
            else:
                teachers = [_clone_numpy_flex(m) for m in brain.members]

        if use_adv and not quiet:
            print(
                f"nn-brain  adversarial  FGSM/PGD  eps={ADV_EPS:.3f}  "
                f"steps={ADV_STEPS}  mix={ADV_MIX:.2f}  noise={ADV_NOISE:.3f}"
            )
        nlls = []
        nps = []
        for i, member in enumerate(brain.members):
            teacher = teachers[i] if teachers else None
            if brain.backend == "torch":
                _train_torch_flex(
                    member,
                    x_tr,
                    r_tr,
                    epochs=epochs,
                    batch=batch,
                    lr=train_lr,
                    teacher=teacher,
                    distill_w=distill_w,
                    ewc_w=ewc_w,
                    adversarial=use_adv,
                )
            else:
                _train_numpy_flex(
                    member,
                    x_tr,
                    r_tr,
                    epochs=epochs,
                    batch=batch,
                    lr=train_lr,
                    seed=1000 + i * 97,
                    teacher=teacher,
                    distill_w=distill_w,
                    ewc_w=ewc_w,
                    adversarial=use_adv,
                )
            score = _mean_profit(member, x_va, r_va) if len(r_va) else _mean_profit(member, x_tr, r_tr)
            nlls.append(-score)
            nps.append(member.n_params())
            if not quiet:
                tag = "xfer" if transferred else "fresh"
                if use_adv:
                    tag = f"{tag}+adv"
                print(
                    f"nn-brain  {member.name:12}  val_pips {score:.2f}  params {nps[-1]}  "
                    f"{brain.backend}/{brain.device}  {tag}"
                )
        n_score = int(len(r_va) or len(r_tr))
        new_w = posterior_weights(
            np.asarray(nlls, dtype=np.float64),
            np.asarray(nps, dtype=np.float64),
            n_train=int(len(x_tr)),
            n_val=n_score,
        )
        if transferred and prior_weights is not None and len(prior_weights) == len(new_w):
            blended = (1.0 - WEIGHT_BLEND) * new_w + WEIGHT_BLEND * prior_weights
            blended = np.clip(blended, 1e-9, None)
            blended /= blended.sum()
            brain.weights = blended
        else:
            brain.weights = new_w
        brain.n_train = int(prior_n + n_new_train) if transferred else int(n_new_train)
        brain.trained_at = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        top = int(np.argmax(brain.weights))
        order = np.argsort(-brain.weights)
        rank = "  ".join(f"{brain.member_names[j]} {brain.weights[j]:.2f}" for j in order[:4])
        buys = int((y == 1).sum())
        sells = int((y == 2).sum())
        holds = int((y == 0).sum())
        xfer_note = (
            f"transfer warm+distill+replay={n_replay}  lr×{TRANSFER_LR:.2f}  "
            if transferred
            else ""
        )
        adv_note = f"adv eps={ADV_EPS:.2f}/{ADV_STEPS}  " if use_adv else ""
        brain.note = (
            f"rl bma {xfer_note}{adv_note}n={brain.n_train} val={len(r_va)}  "
            f"BUY {buys} SELL {sells} HOLD {holds}  "
            f"top {brain.member_names[top]}  {rank}  {brain.backend}/{brain.device}"
        )
        # Grow experience memory: prior mem + this window's rows.
        try:
            if mem_X is not None and mem_R is not None and len(mem_X):
                save_nn_memory(symbol, np.concatenate([mem_X, X], axis=0), np.concatenate([mem_R, R], axis=0))
            else:
                save_nn_memory(symbol, X, R)
        except OSError as exc:
            if not quiet:
                print(f"nn-brain  mem save failed: {exc}")
        path = brain.save()
        if not quiet:
            print(f"nn-brain  saved  {path.name}  {brain.note}")
            if transferred:
                print(f"nn-brain  memory  {mem_path(symbol).name}  (kept for next retrain)")
        return brain

    if brain.backend == "torch":
        torch, nn = _torch_mod()
        assert torch is not None and nn is not None
        params = brain.model.parameters() if brain.vision else brain.model.net.parameters()
        opt = torch.optim.Adam(params, lr=train_lr)
        loss_fn = nn.CrossEntropyLoss()
        xt = torch.from_numpy(X).to(brain.model.device)
        yt = torch.from_numpy(y).to(brain.model.device)
        ct = None
        if brain.vision and C is not None:
            ct = torch.from_numpy(C).to(brain.model.device)
        # Teacher soft labels for transfer (single net / vision).
        teacher_state = None
        if transferred:
            teacher_state = {
                k: v.detach().clone()
                for k, v in (
                    brain.model.state_dict().items()
                    if not brain.vision
                    else {}
                )
            }
            if brain.vision:
                # Snapshot logits via a frozen forward — rebuild is heavy; use EWC on params only.
                teacher_state = None
        n = len(y)
        if brain.vision:
            brain.model.train()
        else:
            brain.model.net.train()
        anchor = None
        if transferred and ewc_w > 0 and not brain.vision:
            anchor = {k: v.detach().clone() for k, v in brain.model.net.named_parameters()}
        for ep in range(max(1, epochs)):
            perm = torch.randperm(n, device=brain.model.device)
            total = 0.0
            steps = 0
            for i in range(0, n, batch):
                idx = perm[i : i + batch]
                x_clean = xt[idx]
                if use_adv and not brain.vision and ADV_EPS > 0:
                    x_var = x_clean.detach().clone().requires_grad_(True)
                    logits_atk = brain.model.net(x_var)
                    loss_atk = loss_fn(logits_atk, yt[idx])
                    g = torch.autograd.grad(loss_atk, x_var, retain_graph=False, create_graph=False)[0]
                    x_adv = torch.max(
                        torch.min(x_clean + float(ADV_EPS) * g.sign(), x_clean + float(ADV_EPS)),
                        x_clean - float(ADV_EPS),
                    ).detach()
                    logits_c = brain.model.net(x_clean)
                    logits_a = brain.model.net(x_adv)
                    loss = (1.0 - ADV_MIX) * loss_fn(logits_c, yt[idx]) + ADV_MIX * loss_fn(logits_a, yt[idx])
                elif brain.vision:
                    emb = brain.model.conv(ct[idx]).flatten(1)
                    logits = brain.model.head(torch.cat([x_clean, emb], dim=1))
                    loss = loss_fn(logits, yt[idx])
                else:
                    if ADV_NOISE > 0:
                        x_clean = x_clean + ADV_NOISE * torch.randn_like(x_clean)
                    logits = brain.model.net(x_clean)
                    loss = loss_fn(logits, yt[idx])
                if transferred and ewc_w > 0 and anchor is not None:
                    pen = torch.zeros((), device=brain.model.device)
                    for name, param in brain.model.net.named_parameters():
                        if name in anchor:
                            pen = pen + ((param - anchor[name]) ** 2).sum()
                    loss = loss + float(ewc_w) * pen
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.item())
                steps += 1
            if not quiet and (ep == 0 or ep == epochs - 1 or ep % 5 == 0):
                tag = "vision" if brain.vision else "mlp"
                xfer = " xfer" if transferred else ""
                adv = " adv" if use_adv and not brain.vision else ""
                print(
                    f"nn-brain  epoch {ep+1}/{epochs}  loss {total/max(steps,1):.4f}  "
                    f"device {brain.device}  {tag}{xfer}{adv}"
                )
        if brain.vision:
            brain.model.eval()
        else:
            brain.model.net.eval()
    else:
        # Numpy SGD (feature MLP only)
        rng = np.random.default_rng(0)
        n = len(y)
        anchor_w = None
        if transferred and ewc_w > 0:
            anchor_w = {
                "w1": brain.model.w1.copy(),
                "b1": brain.model.b1.copy(),
                "w2": brain.model.w2.copy(),
                "b2": brain.model.b2.copy(),
                "w3": brain.model.w3.copy(),
                "b3": brain.model.b3.copy(),
            }
        for ep in range(max(1, epochs)):
            perm = rng.permutation(n)
            total = 0.0
            steps = 0
            for i in range(0, n, batch):
                idx = perm[i : i + batch]
                xb = X[idx].astype(np.float32)
                yb = y[idx]
                if use_adv and ADV_EPS > 0:
                    # One-step FGSM on CE: dL/dx ≈ (p - onehot) backprop through MLP.
                    h1 = np.maximum(0.0, xb @ brain.model.w1 + brain.model.b1)
                    h2 = np.maximum(0.0, h1 @ brain.model.w2 + brain.model.b2)
                    logits = h2 @ brain.model.w3 + brain.model.b3
                    z = logits - logits.max(axis=1, keepdims=True)
                    exp = np.exp(z)
                    p = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-9)
                    row = np.arange(len(yb))
                    dlogits = p.copy()
                    dlogits[row, yb] -= 1.0
                    dlogits /= max(len(yb), 1)
                    dh2 = dlogits @ brain.model.w3.T
                    dh2 *= (h2 > 0).astype(np.float32)
                    dh1 = dh2 @ brain.model.w2.T
                    dh1 *= (h1 > 0).astype(np.float32)
                    dx = (dh1 @ brain.model.w1.T).astype(np.float32)
                    x_adv = _project_linf(xb + float(ADV_EPS) * np.sign(dx), xb, float(ADV_EPS))
                    # Train mix on clean + adv (reuse the update block twice via scale).
                    batches = ((xb, 1.0 - ADV_MIX), (x_adv, ADV_MIX))
                else:
                    if ADV_NOISE > 0:
                        xb = xb + ADV_NOISE * rng.standard_normal(xb.shape).astype(np.float32)
                    batches = ((xb, 1.0),)

                loss = 0.0
                for x_use, scale in batches:
                    h1 = np.maximum(0.0, x_use @ brain.model.w1 + brain.model.b1)
                    h2 = np.maximum(0.0, h1 @ brain.model.w2 + brain.model.b2)
                    logits = h2 @ brain.model.w3 + brain.model.b3
                    z = logits - logits.max(axis=1, keepdims=True)
                    exp = np.exp(z)
                    p = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-9)
                    row = np.arange(len(yb))
                    loss += float(scale) * float(-np.log(np.maximum(p[row, yb], 1e-9)).mean())
                    dlogits = p.copy()
                    dlogits[row, yb] -= 1.0
                    dlogits /= max(len(yb), 1)
                    dw3 = h2.T @ dlogits
                    db3 = dlogits.sum(axis=0)
                    dh2 = dlogits @ brain.model.w3.T
                    dh2 *= (h2 > 0).astype(np.float32)
                    dw2 = h1.T @ dh2
                    db2 = dh2.sum(axis=0)
                    dh1 = dh2 @ brain.model.w2.T
                    dh1 *= (h1 > 0).astype(np.float32)
                    dw1 = x_use.T @ dh1
                    db1 = dh1.sum(axis=0)
                    if anchor_w is not None:
                        dw3 = dw3 + ewc_w * (brain.model.w3 - anchor_w["w3"])
                        db3 = db3 + ewc_w * (brain.model.b3 - anchor_w["b3"])
                        dw2 = dw2 + ewc_w * (brain.model.w2 - anchor_w["w2"])
                        db2 = db2 + ewc_w * (brain.model.b2 - anchor_w["b2"])
                        dw1 = dw1 + ewc_w * (brain.model.w1 - anchor_w["w1"])
                        db1 = db1 + ewc_w * (brain.model.b1 - anchor_w["b1"])
                    brain.model.w3 -= (train_lr * scale * dw3).astype(np.float32)
                    brain.model.b3 -= (train_lr * scale * db3).astype(np.float32)
                    brain.model.w2 -= (train_lr * scale * dw2).astype(np.float32)
                    brain.model.b2 -= (train_lr * scale * db2).astype(np.float32)
                    brain.model.w1 -= (train_lr * scale * dw1).astype(np.float32)
                    brain.model.b1 -= (train_lr * scale * db1).astype(np.float32)
                total += float(loss)
                steps += 1
            if not quiet and (ep == 0 or ep == epochs - 1 or ep % 5 == 0):
                xfer = " xfer" if transferred else ""
                print(
                    f"nn-brain  epoch {ep+1}/{epochs}  loss {total/max(steps,1):.4f}  "
                    f"device numpy{xfer}"
                )

    brain.n_train = int(prior_n + len(y)) if transferred else int(len(y))
    brain.trained_at = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    buys = int((y == 1).sum())
    sells = int((y == 2).sum())
    holds = int((y == 0).sum())
    brain.note = (
        f"{'transfer ' if transferred else ''}{'adv ' if use_adv else ''}trained n={brain.n_train}  "
        f"BUY {buys} SELL {sells} HOLD {holds}  "
        f"{brain.backend}/{brain.device}"
        + ("+vision" if brain.vision else "")
    )
    path = brain.save()
    if not quiet:
        print(f"nn-brain  saved  {path.name}  {brain.note}")
    return brain


def load_tensor_brain(
    symbol: str, *, prefer_cuda: bool = True, vision: bool = False
) -> TensorBrain:
    # Prefer the 10-net average when those weights exist. Vision stays single-net.
    if not vision and (bma_path(symbol, True).exists() or bma_path(symbol, False).exists()):
        brain = TensorBrain(symbol, prefer_cuda=prefer_cuda, ensemble=True)
    else:
        brain = TensorBrain(symbol, prefer_cuda=prefer_cuda, vision=vision)
    brain.load()
    return brain
