"""Two small Bayesian-average fleets for the profit-recycle book.

Size fleet: after a sealed win the next trade's money is 1.5× that win.
These nets may raise that budget when the setup still agrees with the trade.
A fresh fleet holds the 1.5× cap. A loss teaches them to stop raising it.

Close fleet: a reflex on the open profit. The hard rule (an already-real
float that almost doubles in one bar) lives in BankGuard. These nets learn
to close a high open gain on a sharp jump before that double, and to leave
a trail winner alone.

Weights: ``brains/{PAIR}.reflex.npz``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from flyfx.paths import BRAIN_DIR

N_IN = 8
N_NETS = 8
BMA_TAU = 0.04
CLOSE_SCORE = 0.72
# Open profit must already be real before a reflex close. A $1k snowball
# start is not this. The hard 2× spike in BankGuard still fires at $1k.
CLOSE_FLOOR = 2_000.0
CLOSE_JUMP = 1.55

SPECS: tuple[dict, ...] = (
    {"name": "linear", "hidden": ()},
    {"name": "shallow8", "hidden": (8,)},
    {"name": "shallow16", "hidden": (16,)},
    {"name": "silu12", "hidden": (12,), "act": "silu"},
    {"name": "tanh12", "hidden": (12,), "act": "tanh"},
    {"name": "deep8", "hidden": (8, 8)},
    {"name": "wide24", "hidden": (24,)},
    {"name": "bottleneck", "hidden": (16, 4, 16)},
)


def _act(name: str, z: np.ndarray) -> np.ndarray:
    if name == "tanh":
        return np.tanh(z)
    if name == "silu":
        return z / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
    return np.maximum(z, 0.0)


def _act_bwd(name: str, z: np.ndarray, a: np.ndarray) -> np.ndarray:
    if name == "tanh":
        return 1.0 - a * a
    if name == "silu":
        sig = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
        return sig + z * sig * (1.0 - sig)
    return (z > 0.0).astype(np.float64)


WORTH_IMPULSE = 0.92


def worth_budget(
    budget: float,
    equity: float,
    impulse_abs: float,
    *,
    impulse_min: float = WORTH_IMPULSE,
    frac0: float = 0.70,
    span: float = 0.30,
) -> float:
    """Raise a 1.5× profit budget toward the account when impulse is strong.

    At the impulse floor the budget is ``frac0`` of equity. At impulse 1 it
    adds ``span``. The margin cap still limits lots. A milder impulse stays
    on 1.5×.
    """
    base = float(budget)
    eq = float(equity)
    mag = float(impulse_abs)
    bar = float(impulse_min)
    if mag < bar or eq <= base or bar >= 1.0:
        return base
    t = min(1.0, (mag - bar) / (1.0 - bar))
    frac = float(frac0) + float(span) * t
    return max(base, eq * frac)


def budget_lift(score: float) -> float:
    """Map a size-net score in [-1, 1] onto a budget multiplier of 1..2.

    Zero (a fresh net) stays at 1.5× the latest profit. A positive score
    raises that budget, up to twice it, when the risk has been paying.
    """
    s = float(score)
    if s < 0.0:
        s = 0.0
    if s > 1.0:
        s = 1.0
    return 1.0 + s


def reflex_close(score: float, prev: float, now: float, *, top: str = "") -> str | None:
    """Close a high open gain on a sharp one-bar jump when the reflex agrees."""
    if prev < CLOSE_FLOOR or now < CLOSE_FLOOR:
        return None
    if now < prev * CLOSE_JUMP:
        return None
    if float(score) < CLOSE_SCORE:
        return None
    who = f"  bma {top}" if top else ""
    return (
        f"reflex close  ${prev:,.0f} → ${now:,.0f}  "
        f"({now / max(prev, 1.0):.2f}×)  score {float(score):.2f}{who}"
    )


def pack_features(
    feat: dict | None,
    *,
    side: str = "",
    prev_usd: float = 0.0,
    now_usd: float = 0.0,
    unreal_r: float = 0.0,
    balance: float = 100_000.0,
) -> np.ndarray:
    feat = feat or {}
    impulse = float(feat.get("impulse") or 0.0)
    sign = 1.0 if str(side).upper() == "BUY" else (-1.0 if str(side).upper() == "SELL" else 0.0)
    agree = float(np.clip(sign * impulse, -1.0, 1.0))
    atr_ratio = float(feat.get("atr_ratio") or 1.0)
    rsi = float(feat.get("rsi") or 50.0)
    kal = feat.get("kalman") if isinstance(feat.get("kalman"), dict) else {}
    uncertainty = float((kal or {}).get("uncertainty") or 0.0)
    ratio = 0.0
    if prev_usd > 50.0 and now_usd > 0.0:
        ratio = float(np.tanh(math.log(max(now_usd, 1.0) / max(prev_usd, 1.0))))
    bal = max(float(balance), 1.0)
    dollars = float(np.clip(now_usd / bal / 0.03, -1.0, 2.0))
    x = np.array(
        [
            agree,
            float(np.clip(atr_ratio - 1.0, -1.0, 2.0)),
            float(np.clip((rsi - 50.0) / 50.0, -1.0, 1.0)),
            float(np.clip(unreal_r / 3.0, -1.0, 2.0)),
            ratio,
            dollars,
            float(np.clip(1.0 - uncertainty, 0.0, 1.0)),
            float(np.clip(float(feat.get("bb_bw") or 0.0) / max(float(feat.get("bb_bw_ma") or 0.01), 1e-6), 0.0, 3.0)),
        ],
        dtype=np.float64,
    )
    return x


class _Net:
    def __init__(self, spec: dict, rng: np.random.Generator, *, out_bias: float) -> None:
        self.name = str(spec["name"])
        self.act = str(spec.get("act") or "relu")
        hidden = tuple(int(h) for h in spec.get("hidden") or ())
        dims = [N_IN, *hidden, 1]
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for i in range(len(dims) - 1):
            fan = max(dims[i], 1)
            scale = math.sqrt(2.0 / fan) if self.act != "tanh" else math.sqrt(1.0 / fan)
            if i == len(dims) - 2:
                scale *= 0.01
            self.W.append(rng.normal(0.0, scale, size=(dims[i + 1], dims[i])).astype(np.float64))
            self.b.append(np.zeros(dims[i + 1], dtype=np.float64))
        self.b[-1][:] = float(out_bias)
        self.nll = 0.05
        self.n_fit = 0

    def forward(self, x: np.ndarray) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
        h = np.asarray(x, dtype=np.float64)
        hs: list[np.ndarray] = [h]
        zs: list[np.ndarray] = []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = W @ h + b
            zs.append(z)
            if i == len(self.W) - 1:
                return float(np.tanh(z[0])), hs, zs
            h = _act(self.act, z)
            hs.append(h)
        return 0.0, hs, zs

    def train_step(self, x: np.ndarray, target: float, lr: float) -> float:
        y, hs, zs = self.forward(x)
        err = (y - float(target)) ** 2
        grad = np.array([(y - float(target)) * (1.0 - y * y)], dtype=np.float64)
        for i in range(len(self.W) - 1, -1, -1):
            h_in = hs[i]
            grad = np.clip(grad, -1.0, 1.0)
            w_prev = self.W[i]
            self.W[i] = w_prev - lr * np.outer(grad, h_in)
            self.b[i] = self.b[i] - lr * grad
            grad = w_prev.T @ grad
            if i > 0:
                z = zs[i - 1]
                a = _act(self.act, z)
                grad = grad * _act_bwd(self.act, z, a)
        self.n_fit += 1
        self.nll = 0.85 * self.nll + 0.15 * float(err)
        return float(err)


class _Fleet:
    def __init__(self, rng: np.random.Generator, *, out_bias: float) -> None:
        self.nets = [_Net(spec, rng, out_bias=out_bias) for spec in SPECS]

    def bma(self, x: np.ndarray) -> tuple[float, str, float]:
        ys = []
        weights = []
        for net in self.nets:
            y, _hs, _zs = net.forward(x)
            ys.append(y)
            weights.append(math.exp(-net.nll / BMA_TAU))
        w = np.asarray(weights, dtype=np.float64)
        w = w / max(float(w.sum()), 1e-12)
        score = float(np.dot(w, np.asarray(ys, dtype=np.float64)))
        top_i = int(np.argmax(w))
        return score, self.nets[top_i].name, float(w[top_i])

    def teach(self, x: np.ndarray, target: float, lr: float = 0.05) -> None:
        for net in self.nets:
            net.train_step(x, target, lr)


REFLEX_DISTILL = 0.45
REFLEX_MEM_CAP = 512


def _clone_fleet(src: _Fleet, rng: np.random.Generator) -> _Fleet:
    dst = _Fleet(rng, out_bias=0.0)
    for a, b in zip(dst.nets, src.nets):
        a.W = [np.array(w, dtype=np.float64, copy=True) for w in b.W]
        a.b = [np.array(v, dtype=np.float64, copy=True) for v in b.b]
        a.nll = float(b.nll)
        a.n_fit = int(b.n_fit)
    return dst


class ProfitReflex:
    """Size lift and close reflex. Off when profit recycle is off."""

    def __init__(self, symbol: str, *, enabled: bool = True, transfer: bool = True) -> None:
        self.enabled = bool(enabled)
        self.symbol = str(symbol or "EURUSD").upper()
        self.transfer = bool(transfer)
        self.persist = True
        self.worth_impulse = WORTH_IMPULSE
        self.worth_frac = 0.70
        self.worth_span = 0.30
        rng = np.random.default_rng(7)
        # Size head is centered at 0 so a fresh book does not raise the budget.
        self.size = _Fleet(rng, out_bias=0.0)
        # Close head starts negative so a fresh reflex does not flatten.
        self.close = _Fleet(rng, out_bias=-1.6)
        self._entry_x: np.ndarray | None = None
        self._live_x: np.ndarray | None = None
        self._size_teacher: _Fleet | None = None
        self._close_teacher: _Fleet | None = None
        self._mem_x: np.ndarray | None = None
        self._mem_size: np.ndarray | None = None
        self._mem_close: np.ndarray | None = None
        self.lift = 1.0
        self.close_score = 0.0
        self.top = ""
        self.note = "reflex off" if not self.enabled else "reflex fresh"
        if self.enabled and self.transfer and self.load():
            self._size_teacher = _clone_fleet(self.size, rng)
            self._close_teacher = _clone_fleet(self.close, rng)
            self._load_mem()
            self.note = "reflex transfer"

    def path(self) -> Path:
        return BRAIN_DIR / f"{self.symbol}.reflex.npz"

    def mem_path(self) -> Path:
        return BRAIN_DIR / f"{self.symbol}.reflex_mem.npz"

    def save(self) -> None:
        dest = self.path()
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob: dict[str, np.ndarray] = {}
        for prefix, fleet in (("s", self.size), ("c", self.close)):
            for i, net in enumerate(fleet.nets):
                for k, (w, b) in enumerate(zip(net.W, net.b)):
                    blob[f"{prefix}{i}w{k}"] = np.asarray(w, dtype=np.float64)
                    blob[f"{prefix}{i}b{k}"] = np.asarray(b, dtype=np.float64)
                blob[f"{prefix}{i}nll"] = np.array([net.nll, float(net.n_fit)], dtype=np.float64)
        np.savez_compressed(dest, **blob)

    def load(self) -> bool:
        dest = self.path()
        if not dest.exists():
            return False
        data = None
        try:
            data = np.load(dest, allow_pickle=False)
            files = set(data.files)
            for prefix, fleet in (("s", self.size), ("c", self.close)):
                for i, net in enumerate(fleet.nets):
                    for k in range(len(net.W)):
                        key_w = f"{prefix}{i}w{k}"
                        key_b = f"{prefix}{i}b{k}"
                        if key_w not in files or key_b not in files:
                            return False
                        w = np.asarray(data[key_w], dtype=np.float64)
                        b = np.asarray(data[key_b], dtype=np.float64)
                        if w.shape != net.W[k].shape or b.shape != net.b[k].shape:
                            return False
            for prefix, fleet in (("s", self.size), ("c", self.close)):
                for i, net in enumerate(fleet.nets):
                    for k in range(len(net.W)):
                        net.W[k] = np.asarray(data[f"{prefix}{i}w{k}"], dtype=np.float64)
                        net.b[k] = np.asarray(data[f"{prefix}{i}b{k}"], dtype=np.float64)
                    nll = np.asarray(data[f"{prefix}{i}nll"], dtype=np.float64)
                    net.nll = float(nll[0])
                    if nll.size > 1:
                        net.n_fit = int(nll[1])
        except (OSError, ValueError, KeyError):
            return False
        finally:
            if data is not None:
                data.close()
        return True

    def _load_mem(self) -> None:
        path = self.mem_path()
        if not path.exists():
            return
        data = None
        try:
            data = np.load(path, allow_pickle=False)
            x = np.asarray(data["X"], dtype=np.float64)
            s = np.asarray(data["size"], dtype=np.float64)
            c = np.asarray(data["close"], dtype=np.float64)
        except (OSError, ValueError, KeyError):
            return
        finally:
            if data is not None:
                data.close()
        if x.ndim != 2 or x.shape[1] != N_IN or len(x) != len(s) or len(x) != len(c) or len(x) == 0:
            return
        self._mem_x = x
        self._mem_size = s
        self._mem_close = c

    def _remember(self, x: np.ndarray | None, size_t: float | None, close_t: float | None) -> None:
        if x is None or size_t is None:
            return
        row = np.asarray(x, dtype=np.float64).reshape(1, N_IN)
        s = np.array([float(size_t)], dtype=np.float64)
        c = np.array([float(0.0 if close_t is None else close_t)], dtype=np.float64)
        if self._mem_x is None:
            self._mem_x, self._mem_size, self._mem_close = row, s, c
        else:
            self._mem_x = np.concatenate([self._mem_x, row], axis=0)
            self._mem_size = np.concatenate([self._mem_size, s], axis=0)
            self._mem_close = np.concatenate([self._mem_close, c], axis=0)
        if len(self._mem_x) > REFLEX_MEM_CAP:
            self._mem_x = self._mem_x[-REFLEX_MEM_CAP:]
            self._mem_size = self._mem_size[-REFLEX_MEM_CAP:]
            self._mem_close = self._mem_close[-REFLEX_MEM_CAP:]
        dest = self.mem_path()
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dest,
            X=self._mem_x,
            size=self._mem_size,
            close=self._mem_close,
        )

    def lift_budget(
        self,
        budget: float,
        feat: dict | None,
        side: str,
        equity: float | None = None,
    ) -> tuple[float, str]:
        if not self.enabled or budget <= 0:
            self.lift = 1.0
            return float(budget), ""
        x = pack_features(feat, side=side)
        self._entry_x = x
        score, top, w = self.size.bma(x)
        impulse = float((feat or {}).get("impulse") or 0.0)
        with_trade = (str(side).upper() == "BUY" and impulse >= 0.50) or (
            str(side).upper() == "SELL" and impulse <= -0.50
        )
        lift = budget_lift(score) if with_trade else 1.0
        raised = float(budget) * lift
        tape = (
            worth_budget(
                float(budget),
                float(equity or 0.0),
                abs(impulse),
                impulse_min=self.worth_impulse,
                frac0=self.worth_frac,
                span=self.worth_span,
            )
            if with_trade
            else float(budget)
        )
        if tape > raised:
            raised = tape
        self.lift = raised / float(budget) if budget else 1.0
        self.top = top
        if self.lift <= 1.02:
            self.note = f"size×{N_NETS} held 1.5×  bma {top} {w:.2f}"
            return float(budget), self.note
        self.note = (
            f"size×{N_NETS} lift ×{self.lift:.2f}  budget ${raised:,.0f}  bma {top} {w:.2f}"
        )
        return raised, self.note

    def consider(
        self,
        prev_usd: float | None,
        now_usd: float,
        feat: dict | None,
        unreal_r: float | None,
        *,
        side: str,
        balance: float,
    ) -> str | None:
        if not self.enabled or prev_usd is None:
            return None
        x = pack_features(
            feat,
            side=side,
            prev_usd=float(prev_usd),
            now_usd=float(now_usd),
            unreal_r=float(unreal_r or 0.0),
            balance=balance,
        )
        self._live_x = x
        score, top, _w = self.close.bma(x)
        # tanh output, shift into 0..1. Fresh bias -1.6 → tanh ≈ -0.92 → score ≈ 0.04.
        prob = 0.5 * (float(score) + 1.0)
        self.close_score = prob
        self.top = top
        return reflex_close(prob, float(prev_usd), float(now_usd), top=top)

    def on_close(self, usd: float, peak_usd: float) -> str:
        if not self.enabled:
            return ""
        notes: list[str] = []
        size_raw: float | None = None
        close_raw: float | None = None
        if self._entry_x is not None:
            if float(usd) > 0.0:
                # A paid win may press the next similar setup. Cap the lesson.
                target = float(np.clip(float(usd) / max(float(peak_usd), 1.0), 0.0, 0.65))
            else:
                target = 0.0
            size_raw = target
            if self._size_teacher is not None:
                prior, _top, _w = self._size_teacher.bma(self._entry_x)
                target = (1.0 - REFLEX_DISTILL) * target + REFLEX_DISTILL * float(prior)
                notes.append("size transfer")
            self.size.teach(self._entry_x, target)
            notes.append(f"size teach {target:.2f}")
        if self._live_x is not None and float(peak_usd) >= CLOSE_FLOOR:
            if float(usd) < 0.75 * float(peak_usd):
                target_c = 0.85
            elif float(usd) >= 0.85 * float(peak_usd):
                target_c = -0.85
            else:
                target_c = 0.0
            close_raw = target_c
            if self._close_teacher is not None:
                prior_c, _top, _w = self._close_teacher.bma(self._live_x)
                target_c = (1.0 - REFLEX_DISTILL) * target_c + REFLEX_DISTILL * float(prior_c)
                notes.append("close transfer")
            self.close.teach(self._live_x, target_c)
            notes.append(f"close teach {target_c:.2f}")
        if self._mem_x is not None and len(self._mem_x) > 0:
            j = int(np.random.default_rng(self.size.nets[0].n_fit + 1).integers(0, len(self._mem_x)))
            self.size.teach(self._mem_x[j], float(self._mem_size[j]))
            if self._live_x is not None:
                self.close.teach(self._mem_x[j], float(self._mem_close[j]))
            notes.append("replay")
        if size_raw is not None and self.persist:
            self._remember(self._entry_x, size_raw, close_raw)
            self.save()
        self._entry_x = None
        if not notes:
            return ""
        return "reflex  " + "  ".join(notes)

    def pack(self) -> dict:
        return {
            "on": bool(self.enabled),
            "lift": round(float(self.lift), 3),
            "close": round(float(self.close_score), 3),
            "top": self.top,
            "note": self.note,
            "n": N_NETS,
        }
