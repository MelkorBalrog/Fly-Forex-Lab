"""Twenty parallel nets that trim trade settings as the book wears.

Each net has a different depth, width, and activation. They all see the same
bar: selected indicator channels, the fused category latents, every node
output, and the current dials (the actuator positions). A Bayesian average
of their proposals is applied as a small step, the way a vehicle trims brake
pressure and steering as pads, tires, and alignment wear — the plant changes,
the controller follows what still pays.

Weights live in ``brains/{PAIR}.setann.npz``. Posterior mass tracks which
structure has been predicting the profitable trim. Output heads start at zero
so a fresh fleet holds the dials until closes teach it.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from flyfx.paths import BRAIN_DIR
from flyfx.risk.bayes_sizer import PARAM_BOUNDS, PARAM_DEFAULTS
from flyfx.sense.category_kalman import (
    CATEGORY_CHANNELS,
    CATEGORY_NAMES,
    CHANNEL_TO_CAT,
    parse_fuse_inds,
)

PARAM_NAMES: tuple[str, ...] = tuple(PARAM_DEFAULTS)
N_SETTINGS = len(PARAM_NAMES)
INDICATOR_NAMES: tuple[str, ...] = tuple(
    ch for cat in CATEGORY_NAMES for ch in CATEGORY_CHANNELS[cat]
)
N_IND = len(INDICATOR_NAMES)
FUSE_FIELDS = ("impulse", "fused", "residual", "uncertainty", "agreement", "slope")
N_FUSE_CAT = len(FUSE_FIELDS)
MIX_FIELDS = ("impulse", "fused", "residual", "uncertainty", "agreement")
NODE_NAMES = ("tech", "trend", "fade", "conf", "nn")
NODE_FIELDS = ("side", "fused", "p_cast", "p_win")
N_SCORES = 3
WEAR_NAMES = (
    "signed_r",
    "win_rate",
    "stop_rate",
    "trail_rate",
    "time_rate",
    "target_rate",
    "loss_streak",
    "atr_stretch",
)
N_WEAR = len(WEAR_NAMES)

OFF_MASK = N_IND
OFF_FUSE = OFF_MASK + N_IND
N_FUSE = len(CATEGORY_NAMES) * N_FUSE_CAT + len(MIX_FIELDS)
OFF_NODES = OFF_FUSE + N_FUSE
N_NODES = len(NODE_NAMES) * len(NODE_FIELDS) + N_SCORES
OFF_SETTINGS = OFF_NODES + N_NODES
OFF_WEAR = OFF_SETTINGS + N_SETTINGS
N_IN = OFF_WEAR + N_WEAR

N_NETS = 20
# Posterior temperature on recent squared error. Lower → sharper model pick.
BMA_TAU = 0.015
# How far a saturated head may pull the normalized dials in one forward.
DELTA_GAIN = 0.50
# Max normalized step applied after a close (~8% of each dial's range).
MAX_STEP = 0.14
WEAR_ALPHA = 0.18

# Twenty structures, one input, one dial vector. Evaluated independently.
SPECS: tuple[dict[str, Any], ...] = (
    {"name": "linear", "hidden": (), "act": "linear"},
    {"name": "shallow16", "hidden": (16,), "act": "relu"},
    {"name": "shallow32", "hidden": (32,), "act": "relu"},
    {"name": "shallow64", "hidden": (64,), "act": "tanh"},
    {"name": "silu24", "hidden": (24,), "act": "silu"},
    {"name": "mlp16", "hidden": (16, 16), "act": "relu"},
    {"name": "mlp32", "hidden": (32, 32), "act": "relu"},
    {"name": "mlp48", "hidden": (48, 48), "act": "gelu"},
    {"name": "gelu24", "hidden": (24, 24), "act": "gelu"},
    {"name": "tanh24", "hidden": (24, 24), "act": "tanh"},
    {"name": "deep12", "hidden": (12, 12, 12), "act": "relu"},
    {"name": "deep20", "hidden": (20, 20, 20), "act": "tanh"},
    {"name": "deep8", "hidden": (8, 8, 8, 8), "act": "silu"},
    {"name": "wide_narrow", "hidden": (72, 12), "act": "relu"},
    {"name": "narrow_wide", "hidden": (12, 64), "act": "gelu"},
    {"name": "bottleneck", "hidden": (48, 8, 48), "act": "relu"},
    {"name": "skip32", "hidden": (32, 32), "act": "relu", "residual": True},
    {"name": "asym", "hidden": (40, 14), "act": "tanh"},
    {"name": "controller", "hidden": (20, 20), "act": "tanh"},
    {"name": "drop32", "hidden": (32, 32), "act": "relu", "dropout": 0.10},
)


def settings_ann_path(symbol: str) -> Path:
    tag = str(symbol or "PAIR").upper()
    return BRAIN_DIR / f"{tag}.setann.npz"


def _clip(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(min(hi, max(lo, v)))


def _finite(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(x):
        return float(default)
    return x


def _side(vote: Any) -> float:
    text = str(vote or "HOLD").upper()
    if text == "BUY":
        return 1.0
    if text == "SELL":
        return -1.0
    return 0.0


def default_settings() -> np.ndarray:
    out = np.zeros(N_SETTINGS, dtype=np.float64)
    for i, name in enumerate(PARAM_NAMES):
        out[i] = normalize_param(name, PARAM_DEFAULTS[name])
    return out


def normalize_param(name: str, value: float) -> float:
    lo, hi = PARAM_BOUNDS[name]
    span = max(hi - lo, 1e-9)
    return _clip(2.0 * (_finite(value) - lo) / span - 1.0)


def denormalize_param(name: str, z: float) -> float:
    lo, hi = PARAM_BOUNDS[name]
    u = (_clip(z) + 1.0) * 0.5
    return float(lo + u * (hi - lo))


def settings_vector(params) -> np.ndarray:
    out = np.zeros(N_SETTINGS, dtype=np.float64)
    for i, name in enumerate(PARAM_NAMES):
        out[i] = normalize_param(name, getattr(params, name))
    return out


def _allow_map(fuse_inds) -> dict[str, tuple[str, ...]]:
    if fuse_inds is None:
        return parse_fuse_inds("")
    if isinstance(fuse_inds, dict):
        return {str(k): tuple(v) for k, v in fuse_inds.items()}
    return parse_fuse_inds(fuse_inds)


def pack_state(
    feat: dict | None,
    nodes: dict | None,
    params,
    fuse_inds=None,
    wear: np.ndarray | None = None,
    scores: dict | None = None,
) -> np.ndarray:
    """Fixed-length controller input. Unselected indicators stay at 0."""
    feat = feat or {}
    nodes = nodes or {}
    scores = scores or {}
    x = np.zeros(N_IN, dtype=np.float64)
    measurements = feat.get("measurements") if isinstance(feat.get("measurements"), dict) else {}
    allow = _allow_map(fuse_inds)
    for i, name in enumerate(INDICATOR_NAMES):
        cat = CHANNEL_TO_CAT[name]
        selected = name in allow.get(cat, ())
        if not selected:
            continue
        x[OFF_MASK + i] = 1.0
        raw = (measurements.get(cat) or {}).get(name) if isinstance(measurements.get(cat), dict) else None
        if raw is None:
            continue
        x[i] = _clip(_finite(raw), -4.0, 4.0)

    cats = feat.get("categories") if isinstance(feat.get("categories"), dict) else {}
    cursor = OFF_FUSE
    for cat in CATEGORY_NAMES:
        row = cats.get(cat) if isinstance(cats.get(cat), dict) else {}
        for field in FUSE_FIELDS:
            x[cursor] = _clip(_finite(row.get(field)), -4.0, 4.0)
            cursor += 1
    kal = feat.get("kalman") if isinstance(feat.get("kalman"), dict) else {}
    mix_src = {
        "impulse": feat.get("impulse", kal.get("impulse")),
        "fused": feat.get("fused", kal.get("fused")),
        "residual": kal.get("residual"),
        "uncertainty": kal.get("uncertainty"),
        "agreement": kal.get("agreement"),
    }
    for field in MIX_FIELDS:
        x[cursor] = _clip(_finite(mix_src.get(field)), -4.0, 4.0)
        cursor += 1

    for name in NODE_NAMES:
        row = nodes.get(name) if isinstance(nodes.get(name), dict) else {}
        x[cursor] = _side(row.get("vote") if row.get("vote") is not None else row.get("side"))
        cursor += 1
        x[cursor] = _clip(_finite(row.get("fused")), -4.0, 4.0)
        cursor += 1
        x[cursor] = _clip(_finite(row.get("p_cast"), 0.5), 0.0, 1.0)
        cursor += 1
        x[cursor] = _clip(_finite(row.get("p_win"), 0.5), 0.0, 1.0)
        cursor += 1
    for key in ("trend", "fade", "conf"):
        x[cursor] = math.tanh(_finite(scores.get(key)))
        cursor += 1
    if cursor != OFF_SETTINGS:
        raise RuntimeError(f"settings input layout drifted: {cursor} != {OFF_SETTINGS}")

    dials = settings_vector(params)
    x[OFF_SETTINGS:OFF_WEAR] = dials
    if wear is not None:
        w = np.asarray(wear, dtype=np.float64).reshape(-1)
        n = min(N_WEAR, w.size)
        x[OFF_WEAR:OFF_WEAR + n] = w[:n]
    np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def posterior_weights(nll: np.ndarray) -> np.ndarray:
    """Tempered softmax of recent error, with a floor so a net can recover."""
    z = -np.asarray(nll, dtype=np.float64) / BMA_TAU
    z -= np.max(z)
    e = np.exp(np.clip(z, -60.0, 60.0))
    s = e / max(float(e.sum()), 1e-12)
    floor = 0.004
    w = floor + (1.0 - floor * s.size) * s
    w /= w.sum()
    return w


def _act_fwd(name: str, z: np.ndarray) -> np.ndarray:
    if name == "relu":
        return np.maximum(z, 0.0)
    if name == "tanh":
        return np.tanh(z)
    if name == "silu":
        return z / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
    if name == "gelu":
        c = 0.7978845608028654
        u = c * (z + 0.044715 * z * z * z)
        return 0.5 * z * (1.0 + np.tanh(u))
    return z


def _act_bwd(name: str, z: np.ndarray, a: np.ndarray) -> np.ndarray:
    if name == "relu":
        return (z > 0.0).astype(np.float64)
    if name == "tanh":
        return 1.0 - a * a
    if name == "silu":
        sig = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
        return sig + z * sig * (1.0 - sig)
    if name == "gelu":
        c = 0.7978845608028654
        u = c * (z + 0.044715 * z * z * z)
        t = np.tanh(u)
        du = c * (1.0 + 3.0 * 0.044715 * z * z)
        dt = (1.0 - t * t) * du
        return 0.5 * (1.0 + t) + 0.5 * z * dt
    return np.ones_like(z)


class _Net:
    """One feedforward trim head. Output is a residual on the current dials."""

    def __init__(self, spec: dict, n_in: int, n_out: int, rng: np.random.Generator) -> None:
        self.name = str(spec["name"])
        self.act = str(spec.get("act") or "relu")
        self.dropout = float(spec.get("dropout") or 0.0)
        self.residual = bool(spec.get("residual"))
        hidden = tuple(int(h) for h in spec.get("hidden") or ())
        dims = [n_in, *hidden, n_out]
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for i in range(len(dims) - 1):
            fan = max(dims[i], 1)
            if self.act in ("relu", "silu", "gelu"):
                scale = math.sqrt(2.0 / fan)
            else:
                scale = math.sqrt(1.0 / fan)
            if i == len(dims) - 2:
                scale *= 0.01
            self.W.append(rng.normal(0.0, scale, size=(dims[i + 1], dims[i])).astype(np.float64))
            self.b.append(np.zeros(dims[i + 1], dtype=np.float64))
        self.nll = 0.05
        self.n_fit = 0

    def n_params(self) -> int:
        return int(sum(int(W.size) + int(b.size) for W, b in zip(self.W, self.b)))

    def forward(
        self,
        x: np.ndarray,
        settings: np.ndarray,
        *,
        train: bool = False,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, dict]:
        h = np.asarray(x, dtype=np.float64)
        zs: list[np.ndarray] = []
        hs: list[np.ndarray] = [h]
        masks: list[np.ndarray | None] = []
        acts: list[str] = []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = W @ h + b
            last = i == len(self.W) - 1
            if last:
                delta = np.tanh(z)
                y = np.clip(settings + DELTA_GAIN * delta, -1.0, 1.0)
                cache = {"hs": hs, "zs": zs, "masks": masks, "acts": acts, "z_out": z, "delta": delta, "y": y}
                return y, cache
            a = _act_fwd(self.act, z)
            if self.residual and a.shape == h.shape:
                a = a + h
            mask = None
            if train and self.dropout > 0.0 and rng is not None:
                keep = 1.0 - self.dropout
                mask = (rng.random(a.shape) < keep).astype(np.float64)
                a = a * mask / max(keep, 1e-6)
            zs.append(z)
            acts.append(self.act)
            masks.append(mask)
            hs.append(a)
            h = a
        raise RuntimeError("net has no output layer")

    def train_step(
        self,
        x: np.ndarray,
        settings: np.ndarray,
        target: np.ndarray,
        lr: float,
        rng: np.random.Generator,
    ) -> float:
        """One SGD step toward ``target`` dials. Returns pre-step squared error."""
        y, cache = self.forward(x, settings, train=True, rng=rng)
        err = float(np.mean((y - target) ** 2))
        delta = cache["delta"]
        delta_star = np.clip((target - settings) / max(DELTA_GAIN, 1e-6), -1.0, 1.0)
        # d(0.5 ||delta - delta*||^2) / d delta, through tanh.
        grad = (delta - delta_star) * (1.0 - delta * delta)
        hs = cache["hs"]
        zs = cache["zs"]
        masks = cache["masks"]
        last = len(self.W) - 1
        for i in range(last, -1, -1):
            W = self.W[i]
            h_in = hs[i]
            skip = None
            if i == last:
                grad_z = grad
            else:
                g = grad
                if masks[i] is not None:
                    keep = max(1.0 - self.dropout, 1e-6)
                    g = g * masks[i] / keep
                if self.residual and g.shape == h_in.shape:
                    skip = g
                grad_z = g * _act_bwd(self.act, zs[i], _act_fwd(self.act, zs[i]))
            grad_z = np.clip(grad_z, -1.0, 1.0)
            d_in = W.T @ grad_z
            if skip is not None:
                d_in = d_in + skip
            step = lr * (3.0 if i == last else 1.0)
            self.W[i] = W - step * np.outer(grad_z, h_in)
            self.b[i] = self.b[i] - step * grad_z
            if i == last:
                self.W[i] *= 1.0 - 1e-4
            grad = np.clip(d_in, -1.0, 1.0)
        self.n_fit += 1
        return err

    def dump_arrays(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            out[f"w{i}"] = W
            out[f"b{i}"] = b
        return out

    def load_arrays(self, blob: dict[str, np.ndarray]) -> bool:
        try:
            for i in range(len(self.W)):
                W = np.asarray(blob[f"w{i}"], dtype=np.float64)
                b = np.asarray(blob[f"b{i}"], dtype=np.float64)
                if W.shape != self.W[i].shape or b.shape != self.b[i].shape:
                    return False
            for i in range(len(self.W)):
                self.W[i] = np.asarray(blob[f"w{i}"], dtype=np.float64)
                self.b[i] = np.asarray(blob[f"b{i}"], dtype=np.float64)
        except (KeyError, ValueError, TypeError):
            return False
        return True


SETANN_MEM_CAP = 4_000
SETANN_DISTILL = 0.45
SETANN_TRANSFER_LR = 0.35


def _setann_mem_path(symbol: str) -> Path:
    tag = str(symbol or "PAIR").upper()
    return BRAIN_DIR / f"{tag}.setann_mem.npz"


def _load_setann_mem(symbol: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    path = _setann_mem_path(symbol)
    if not path.exists():
        return None, None
    data = None
    try:
        data = np.load(path, allow_pickle=False)
        X = np.asarray(data["X"], dtype=np.float64)
        T = np.asarray(data["T"], dtype=np.float64)
    except (OSError, ValueError, KeyError):
        return None, None
    finally:
        if data is not None:
            data.close()
    if X.ndim != 2 or T.ndim != 2 or len(X) != len(T):
        return None, None
    if X.shape[1] != N_IN or T.shape[1] != N_SETTINGS or len(X) == 0:
        return None, None
    return X, T


def _save_setann_mem(symbol: str, X: np.ndarray, T: np.ndarray) -> None:
    if len(X) == 0:
        return
    if len(X) > SETANN_MEM_CAP:
        X = X[-SETANN_MEM_CAP:]
        T = T[-SETANN_MEM_CAP:]
    dest = _setann_mem_path(symbol)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, X=np.asarray(X, dtype=np.float64), T=np.asarray(T, dtype=np.float64))


def _clone_settings_teacher(fleet: SettingsFleet) -> SettingsFleet:
    teacher = SettingsFleet(fleet.symbol, seed=11, parallel=False)
    for src, dst in zip(fleet.nets, teacher.nets):
        dst.W = [np.array(w, dtype=np.float64, copy=True) for w in src.W]
        dst.b = [np.array(b, dtype=np.float64, copy=True) for b in src.b]
        dst.nll = float(src.nll)
        dst.n_fit = int(src.n_fit)
    return teacher


def train_settings_fleet(
    bars: list[dict],
    symbol: str,
    *,
    fuse_inds: str = "",
    quiet: bool = False,
    horizon: int = 12,
    transfer: bool = True,
) -> SettingsFleet | None:
    """SGD the 20 controllers on this window and print one line per net.

    Each sample is the indicator state at a bar and the trend's forward
    result ``horizon`` bars later. The teacher is the same dial target the
    live book uses after a close. Wear and the profit anchor are cleared
    afterward so the sim still starts from the live book, with the weights
    already fit.
    """
    if len(bars) < 80:
        if not quiet:
            print(f"settings-ann  skip train  need 80 bars, have {len(bars)}")
        return None
    from flyfx.exec.account_sim import pip_size
    from flyfx.risk.bayes_sizer import AdaptiveParams
    from flyfx.sense.category_indicators import CategoryIndicatorEngine

    fleet = SettingsFleet(symbol)
    teacher = None
    if transfer and fleet.load():
        teacher = _clone_settings_teacher(fleet)
        if not quiet:
            print(
                f"settings-ann  transfer  warm-start from prior  "
                f"n={fleet.n_updates}  distill {SETANN_DISTILL:.2f}"
            )
    mem_x, mem_t = _load_setann_mem(symbol) if teacher is not None else (None, None)
    if teacher is not None and mem_x is not None and not quiet:
        print(f"settings-ann  transfer  replay {len(mem_x)} prior rows")
    fresh_x: list[np.ndarray] = []
    fresh_t: list[np.ndarray] = []
    params = AdaptiveParams()
    settings = settings_vector(params)
    eng = CategoryIndicatorEngine(pip=pip_size(float(bars[0].get("close") or 1.0)))
    packed: list[tuple[np.ndarray, float, str]] = []
    for bar in bars:
        close = float(bar.get("close") or 0.0)
        if close <= 0.0:
            continue
        feat = eng.update(
            float(bar.get("high") or close),
            float(bar.get("low") or close),
            close,
            volume=bar.get("volume"),
            open_=bar.get("open"),
            extras=bar if isinstance(bar, dict) else None,
            stamp=int(bar.get("time") or 0),
        )
        if not feat.get("ready"):
            continue
        packed.append((pack_state(feat, {}, params, fuse_inds, None, {}), close, str(feat.get("regime") or "CHOP")))
    h = max(int(horizon), 4)
    if len(packed) <= h + 4:
        if not quiet:
            print(f"settings-ann  skip train  only {len(packed)} ready bars")
        return None
    if not quiet:
        print(
            f"settings-ann  training 20 controllers on this window  "
            f"{len(packed) - h} samples  horizon {h} bars…"
        )
    pip = pip_size(packed[0][1])
    err_sum = np.zeros(len(fleet.nets), dtype=np.float64)
    n_steps = 0
    fleet._anchored = True
    fleet.peak_dials = settings.copy()
    for i in range(len(packed) - h):
        x, now, regime = packed[i]
        fut = packed[i + h][1]
        pips = (fut - now) / max(pip, 1e-12)
        if regime == "DOWN":
            pips = -pips
        elif regime != "UP":
            continue
        if pips >= 6.0:
            reason = "ATR target"
        elif pips > 0.0:
            reason = "trail stop"
        elif pips <= -4.0:
            reason = "ATR stop"
        else:
            reason = "time stop"
        row = {"usd": pips * 10.0, "pips": pips, "reason": reason}
        target = fleet._teach(settings, row)
        fresh_x.append(np.asarray(x, dtype=np.float64).copy())
        fresh_t.append(np.asarray(target, dtype=np.float64).copy())
        fit_target = target
        if teacher is not None:
            taught, _w = teacher._average(x, settings)
            fit_target = (1.0 - SETANN_DISTILL) * target + SETANN_DISTILL * taught
        lr = fleet._lr() * (SETANN_TRANSFER_LR if teacher is not None else 1.0)
        errs = np.zeros(len(fleet.nets), dtype=np.float64)
        for k, net in enumerate(fleet.nets):
            errs[k] = net.train_step(x, settings, fit_target, lr, fleet.rng)
        if teacher is not None and mem_x is not None and len(mem_x) > 0:
            j = int(fleet.rng.integers(0, len(mem_x)))
            replay_t = mem_t[j]
            taught, _w = teacher._average(mem_x[j], settings)
            replay_t = (1.0 - SETANN_DISTILL) * replay_t + SETANN_DISTILL * taught
            for k, net in enumerate(fleet.nets):
                net.train_step(mem_x[j], settings, replay_t, lr, fleet.rng)
        fleet._blend_nll(errs)
        err_sum += errs
        fleet.n_updates += 1
        n_steps += 1
    if n_steps <= 0:
        if not quiet:
            print("settings-ann  skip train  no trend samples in the window")
        return None
    mean_err = err_sum / float(n_steps)
    w = posterior_weights(np.array([n.nll for n in fleet.nets], dtype=np.float64))
    if not quiet:
        for k, net in enumerate(fleet.nets):
            print(
                f"settings-ann  {net.name:12}  err {mean_err[k]:.4f}  "
                f"params {net.n_params()}  steps {net.n_fit}  w {w[k]:.2f}"
            )
        top = int(np.argmax(w))
        order = np.argsort(-w)
        rank = "  ".join(f"{fleet.nets[j].name} {w[j]:.2f}" for j in order[:4])
        print(
            f"settings-ann  saved  {settings_ann_path(symbol).name}  "
            f"n={fleet.n_updates}  top {fleet.nets[top].name}  {rank}"
        )
    fleet.wear = np.zeros(N_WEAR, dtype=np.float64)
    fleet.wear[1] = 0.45
    fleet._streak = 0
    fleet.paid = default_settings()
    fleet.paid_n = 0
    fleet.peak_dials = default_settings()
    fleet.cum_usd = 0.0
    fleet.peak_usd = 0.0
    fleet._anchored = False
    fleet._last_lost = False
    fleet._armed_x = None
    fleet._armed_s = None
    fleet.save()
    if fresh_x:
        stacked_x = np.stack(fresh_x, axis=0)
        stacked_t = np.stack(fresh_t, axis=0)
        if mem_x is not None and len(mem_x):
            stacked_x = np.concatenate([mem_x, stacked_x], axis=0)
            stacked_t = np.concatenate([mem_t, stacked_t], axis=0)
        _save_setann_mem(symbol, stacked_x, stacked_t)
    return fleet


def _predict_one(pack: tuple[_Net, np.ndarray, np.ndarray]) -> np.ndarray:
    net, x, settings = pack
    y, _ = net.forward(x, settings, train=False)
    return y


class SettingsFleet:
    """Bayesian average of 20 setting controllers. Learns on each close."""

    def __init__(self, symbol: str, *, seed: int = 7, parallel: bool = True) -> None:
        self.symbol = str(symbol or "PAIR").upper()
        self.parallel = bool(parallel)
        self.rng = np.random.default_rng(seed)
        self.nets = [_Net(spec, N_IN, N_SETTINGS, self.rng) for spec in SPECS]
        self.n_updates = 0
        self.wear = np.zeros(N_WEAR, dtype=np.float64)
        self.wear[1] = 0.45
        self._streak = 0
        self.paid = default_settings()
        self.paid_n = 0
        self.peak_dials = default_settings()
        self.cum_usd = 0.0
        self.peak_usd = 0.0
        self._anchored = False
        self._last_lost = False
        self._armed_x: np.ndarray | None = None
        self._armed_s: np.ndarray | None = None
        self.last_note = ""
        self.last_moved = "held"
        self._pool: ThreadPoolExecutor | None = None

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _pool_get(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=min(8, len(self.nets)),
                thread_name_prefix="setann",
            )
        return self._pool

    def pack(self, ctx: dict | None, params) -> np.ndarray:
        ctx = ctx or {}
        nodes = ctx.get("nodes")
        if not isinstance(nodes, dict):
            nodes = _nodes_from_ctx(ctx)
        scores = ctx.get("scores") if isinstance(ctx.get("scores"), dict) else _scores_from_ctx(ctx)
        return pack_state(
            ctx.get("feat") if isinstance(ctx.get("feat"), dict) else ctx.get("feat"),
            nodes,
            params,
            ctx.get("fuse_inds"),
            self.wear,
            scores,
        )

    def arm(self, ctx: dict | None, params) -> None:
        """Remember the plant state that opened the trade."""
        self._armed_x = self.pack(ctx, params)
        self._armed_s = settings_vector(params)
        if not self._anchored:
            self.peak_dials = self._armed_s.copy()
            self.paid = self._armed_s.copy()
            self._anchored = True

    def propose(self, ctx: dict | None, params) -> tuple[np.ndarray, np.ndarray]:
        """Bayesian-averaged normalized dials, and the posterior weights."""
        x = self.pack(ctx, params)
        settings = settings_vector(params)
        return self._average(x, settings)

    def on_close(self, row: dict, params, ctx: dict | None = None) -> str:
        """Learn from the close, then trim the dials for the next trade."""
        self._update_wear(row, (ctx or {}).get("feat") if isinstance(ctx, dict) else None)
        if self._armed_x is None or self._armed_s is None:
            self.arm(ctx, params)
        assert self._armed_x is not None and self._armed_s is not None
        self._mark_profit(self._armed_s, row)
        target = self._teach(self._armed_s, row)
        lr = self._lr()
        errs = np.zeros(len(self.nets), dtype=np.float64)
        for i, net in enumerate(self.nets):
            errs[i] = net.train_step(self._armed_x, self._armed_s, target, lr, self.rng)
        self._blend_nll(errs)
        y_bma, w = self._average(self._armed_x, self._armed_s)
        # The 20-net average is what moves the live dials. Mix in the wear
        # target so a loss still trims before the nets have caught up.
        # _apply keeps a loss from widening the stop.
        live = np.clip(0.5 * np.asarray(target, dtype=np.float64) + 0.5 * np.asarray(y_bma, dtype=np.float64), -1.0, 1.0)
        notes = self._apply(params, live)
        self.n_updates += 1
        self._armed_x = None
        self._armed_s = None
        params.n_updates = int(getattr(params, "n_updates", 0)) + 1
        top = int(np.argmax(w))
        line = (
            f"settings-ann n={self.n_updates}  lr={lr:.3f}  "
            f"bma {self.nets[top].name} {w[top]:.2f}  "
            + (", ".join(notes) if notes else "held")
        )
        self.last_moved = ", ".join(notes) if notes else "held"
        self.last_note = line
        log = getattr(params, "log", None)
        if isinstance(log, list):
            log.append(line)
            if len(log) > 40:
                del log[:-40]
        return line

    def dump(self) -> dict:
        w = posterior_weights(np.array([n.nll for n in self.nets], dtype=np.float64))
        top = int(np.argmax(w))
        return {
            "on": True,
            "n": int(self.n_updates),
            "top": f"{self.nets[top].name} {float(w[top]):.2f}",
            "weights": {net.name: round(float(w[i]), 4) for i, net in enumerate(self.nets)},
            "wear": {WEAR_NAMES[i]: round(float(self.wear[i]), 3) for i in range(N_WEAR)},
            "paid_n": int(self.paid_n),
            "moved": self.last_moved,
            "note": self.last_note,
        }

    def save(self, path: Path | None = None) -> Path:
        dest = Path(path) if path is not None else settings_ann_path(self.symbol)
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob: dict[str, np.ndarray] = {
            "nll": np.array([n.nll for n in self.nets], dtype=np.float64),
            "n_fit": np.array([n.n_fit for n in self.nets], dtype=np.int32),
            "wear": self.wear.astype(np.float64),
            "paid": self.paid.astype(np.float64),
            "meta": np.array(
                [self.n_updates, self.paid_n, self._streak, int(self._anchored)],
                dtype=np.int32,
            ),
            "peak_dials": self.peak_dials.astype(np.float64),
            "profit": np.array([self.cum_usd, self.peak_usd], dtype=np.float64),
        }
        for i, net in enumerate(self.nets):
            for key, arr in net.dump_arrays().items():
                blob[f"n{i}{key}"] = arr
        np.savez_compressed(dest, **blob)
        return dest

    def load(self, path: Path | None = None) -> bool:
        dest = Path(path) if path is not None else settings_ann_path(self.symbol)
        if not dest.exists():
            return False
        try:
            data = np.load(dest, allow_pickle=False)
        except (OSError, ValueError):
            return False
        try:
            nll = np.asarray(data["nll"], dtype=np.float64)
            if nll.shape != (len(self.nets),):
                return False
            loaded: list[dict[str, np.ndarray]] = []
            for i, net in enumerate(self.nets):
                blob = {key: np.asarray(data[f"n{i}{key}"]) for key in net.dump_arrays()}
                if any(np.asarray(blob[f"w{k}"]).shape != net.W[k].shape for k in range(len(net.W))):
                    return False
                loaded.append(blob)
            for i, net in enumerate(self.nets):
                if not net.load_arrays(loaded[i]):
                    return False
                net.nll = float(nll[i])
                net.n_fit = int(np.asarray(data["n_fit"])[i]) if "n_fit" in data.files else 0
            self.wear = np.asarray(data["wear"], dtype=np.float64).reshape(N_WEAR)
            self.paid = np.asarray(data["paid"], dtype=np.float64).reshape(N_SETTINGS)
            meta = np.asarray(data["meta"], dtype=np.int32)
            self.n_updates = int(meta[0])
            self.paid_n = int(meta[1])
            self._streak = int(meta[2])
            if meta.size > 3:
                self._anchored = bool(int(meta[3]))
            if "peak_dials" in data.files:
                peak = np.asarray(data["peak_dials"], dtype=np.float64).reshape(-1)
                if peak.size == N_SETTINGS:
                    self.peak_dials = peak
            if "profit" in data.files:
                profit = np.asarray(data["profit"], dtype=np.float64).reshape(-1)
                if profit.size >= 2:
                    self.cum_usd = float(profit[0])
                    self.peak_usd = float(profit[1])
        except (KeyError, ValueError, IndexError):
            return False
        finally:
            data.close()
        return True

    def _average(self, x: np.ndarray, settings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w = posterior_weights(np.array([n.nll for n in self.nets], dtype=np.float64))
        if self.parallel and len(self.nets) > 1:
            packs = [(net, x, settings) for net in self.nets]
            ys = list(self._pool_get().map(_predict_one, packs))
        else:
            ys = [_predict_one((net, x, settings)) for net in self.nets]
        acc = np.zeros(N_SETTINGS, dtype=np.float64)
        for wi, y in zip(w, ys):
            acc += float(wi) * y
        return np.clip(acc, -1.0, 1.0), w

    def _lr(self) -> float:
        n = self.n_updates + 1
        return 0.18 * (0.55 + 0.45 * n / (n + 6.0))

    def _blend_nll(self, errs: np.ndarray) -> None:
        for net, err in zip(self.nets, errs):
            net.nll = 0.82 * float(net.nll) + 0.18 * float(err)

    def _update_wear(self, row: dict, feat: dict | None) -> None:
        won = _finite(row.get("usd")) > 0.0
        pips = _finite(row.get("pips"))
        reason = str(row.get("reason") or "").lower()
        a = WEAR_ALPHA
        stop = ("stop" in reason) and ("trail" not in reason) and (not won)
        self.wear[0] = (1.0 - a) * self.wear[0] + a * math.tanh(pips / 12.0)
        self.wear[1] = (1.0 - a) * self.wear[1] + a * (1.0 if won else 0.0)
        self.wear[2] = (1.0 - a) * self.wear[2] + a * (1.0 if stop else 0.0)
        self.wear[3] = (1.0 - a) * self.wear[3] + a * (1.0 if "trail" in reason else 0.0)
        self.wear[4] = (1.0 - a) * self.wear[4] + a * (1.0 if "time" in reason else 0.0)
        self.wear[5] = (1.0 - a) * self.wear[5] + a * (1.0 if "target" in reason else 0.0)
        if won:
            self._streak = 0
        else:
            self._streak += 1
        self.wear[6] = min(self._streak / 6.0, 1.5)
        atr = _finite((feat or {}).get("atr_ratio"), 1.0)
        self.wear[7] = (1.0 - a) * self.wear[7] + a * _clip(atr - 1.0, -1.5, 2.5)

    def _mark_profit(self, settings: np.ndarray, row: dict) -> None:
        """Remember the dials that were on when equity made a new high."""
        usd = _finite(row.get("usd"))
        self.cum_usd += usd
        self._last_lost = usd <= 0.0
        if not self._anchored:
            self.peak_dials = np.asarray(settings, dtype=np.float64).copy()
            self._anchored = True
        if usd > 0.0 and self.cum_usd >= self.peak_usd:
            self.peak_usd = self.cum_usd
            self.peak_dials = np.asarray(settings, dtype=np.float64).copy()
            self.paid = self.peak_dials.copy()
            self.paid_n += 1

    def _teach(self, current: np.ndarray, row: dict) -> np.ndarray:
        """Wins keep the dials that paid. Losses step back toward that snapshot.

        The stop is never given more room after a loss — a wider stop at the
        same size is a larger dollar loss.
        """
        cur = np.asarray(current, dtype=np.float64)
        anchor = np.asarray(self.peak_dials, dtype=np.float64)
        idx = {name: i for i, name in enumerate(PARAM_NAMES)}
        won = _finite(row.get("usd")) > 0.0
        if won:
            tgt = cur.copy()
            # A clean target hit can keep a little more of the move. Still no SL widen.
            if "target" in str(row.get("reason") or "").lower():
                tgt[idx["tp_atr"]] = min(1.0, float(tgt[idx["tp_atr"]]) + 0.04)
        else:
            tgt = np.clip(0.20 * cur + 0.80 * anchor, -1.0, 1.0)
            tgt[idx["sl_atr"]] = min(float(tgt[idx["sl_atr"]]), float(anchor[idx["sl_atr"]]), float(cur[idx["sl_atr"]]))
            tgt[idx["cooldown_bars"]] = min(1.0, max(float(tgt[idx["cooldown_bars"]]), float(anchor[idx["cooldown_bars"]]) + 0.05))
            tgt[idx["max_hold_bars"]] = max(-1.0, min(float(tgt[idx["max_hold_bars"]]), float(anchor[idx["max_hold_bars"]]) - 0.04))
            tgt[idx["trail_arm_atr"]] = max(-1.0, min(float(tgt[idx["trail_arm_atr"]]), float(anchor[idx["trail_arm_atr"]]) - 0.03))
        return np.clip(tgt, -1.0, 1.0)

    def _apply(self, params, y: np.ndarray) -> list[str]:
        cur = settings_vector(params)
        y = np.asarray(y, dtype=np.float64).copy()
        i_sl = PARAM_NAMES.index("sl_atr")
        if self._last_lost:
            y[i_sl] = min(float(y[i_sl]), float(cur[i_sl]), float(self.peak_dials[i_sl]))
        step = np.clip(y - cur, -MAX_STEP, MAX_STEP)
        nxt = np.clip(cur + step, -1.0, 1.0)
        if self._last_lost:
            nxt[i_sl] = min(float(nxt[i_sl]), float(cur[i_sl]))
        notes: list[str] = []
        for i, name in enumerate(PARAM_NAMES):
            msg = params._set(name, denormalize_param(name, float(nxt[i])))
            if msg:
                notes.append(msg)
        return notes


def _nodes_from_ctx(ctx: dict) -> dict:
    def row(obs, vote_key: str | None = None) -> dict:
        src = obs if isinstance(obs, dict) else {}
        vote = src.get("vote")
        if vote is None and vote_key:
            vote = ctx.get(vote_key)
        return {
            "vote": vote if vote is not None else "HOLD",
            "fused": src.get("fused", 0.0),
            "p_cast": src.get("p_cast", 0.5),
            "p_win": src.get("p_win", 0.5),
        }

    nn_conf = ctx.get("nn_conf")
    nn_side = ctx.get("nn_side") or "HOLD"
    nn_p = _finite(nn_conf, 0.0) if nn_conf is not None else 0.0
    return {
        "tech": row(ctx.get("tech")),
        "trend": row(ctx.get("trend")),
        "fade": row(ctx.get("fade")),
        "conf": {
            "vote": ctx.get("conf_vote") or "HOLD",
            "fused": math.tanh(_finite(ctx.get("conf_score"))),
            "p_cast": _clip(abs(_finite(ctx.get("conf_score"))), 0.0, 1.0),
            "p_win": 0.5,
        },
        "nn": {
            "vote": nn_side,
            "fused": 1.0 if nn_side == "BUY" else (-1.0 if nn_side == "SELL" else 0.0),
            "p_cast": _clip(nn_p, 0.0, 1.0),
            "p_win": _clip(nn_p, 0.0, 1.0),
        },
    }


def _scores_from_ctx(ctx: dict) -> dict:
    return {
        "trend": ctx.get("t_score") or 0.0,
        "fade": ctx.get("f_score") or 0.0,
        "conf": ctx.get("conf_score") or 0.0,
    }
