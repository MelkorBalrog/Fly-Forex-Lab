"""Eight nets that read a currency triangle and score the traded pair's trend.

The triangle for a pair is that pair, another USD leg, and the cross that
closes the three. EURUSD is EURUSD, AUDUSD, and EURAUD. The Bayesian average
of the eight nets is one more indicator: it is mixed into every category
node, and those nodes are what the flies and the committee already read.

A prediction at this bar uses only closes up to now. The label is the traded
pair's return after the horizon, so the weight file learns once that move
has happened. Weights: ``brains/{PAIR}.tri.npz``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from flyfx.paths import BRAIN_DIR

# Traded pair, partner, cross. The cross is partner/traded or traded*partner
# as noted in IMPLIED when a downloaded cross is missing.
TRIPLETS: dict[str, tuple[str, str, str]] = {
    "EURUSD": ("EURUSD", "AUDUSD", "EURAUD"),
    "GBPUSD": ("GBPUSD", "EURUSD", "EURGBP"),
    "AUDUSD": ("AUDUSD", "NZDUSD", "AUDNZD"),
    "NZDUSD": ("NZDUSD", "AUDUSD", "AUDNZD"),
    "USDCAD": ("USDCAD", "AUDUSD", "AUDCAD"),
}

# cross = a/b, or a*b when the quote is the product of the two USD legs.
IMPLIED: dict[str, tuple[str, str, str]] = {
    "EURAUD": ("EURUSD", "AUDUSD", "div"),
    "EURGBP": ("EURUSD", "GBPUSD", "div"),
    "AUDNZD": ("AUDUSD", "NZDUSD", "div"),
    "AUDCAD": ("AUDUSD", "USDCAD", "mul"),
}

HORIZON = 12
N_IN = 10
N_NETS = 8
MIN_FIT = 24
BLEND = 0.18
_SPECS = (
    ("linear", (), "linear"),
    ("relu8", (8,), "relu"),
    ("relu16", (16,), "relu"),
    ("tanh12", (12,), "tanh"),
    ("relu24", (24,), "relu"),
    ("deep", (12, 8), "relu"),
    ("wide", (32,), "tanh"),
    ("bottleneck", (8, 8), "relu"),
)


def triplet_for(symbol: str) -> tuple[str, str, str] | None:
    return TRIPLETS.get(str(symbol or "").upper())


class _Net:
    def __init__(self, hidden: tuple[int, ...], act: str, rng: np.random.Generator) -> None:
        dims = (N_IN, *hidden, 1)
        self.w: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for i in range(len(dims) - 1):
            scale = 1.0 / math.sqrt(dims[i])
            self.w.append(rng.normal(0.0, scale, size=(dims[i + 1], dims[i])))
            self.b.append(np.zeros(dims[i + 1], dtype=np.float64))
        self.act = act
        self.n_fit = 0
        self._acts: list[np.ndarray] = []
        self._pres: list[np.ndarray] = []

    def forward(self, x: np.ndarray, train: bool = False) -> float:
        h = np.asarray(x, dtype=np.float64)
        acts = [h]
        pres: list[np.ndarray] = []
        last = len(self.w) - 1
        for i, (w, b) in enumerate(zip(self.w, self.b)):
            z = w @ h + b
            pres.append(z)
            if i == last or self.act == "tanh":
                h = np.tanh(z)
            elif self.act == "relu":
                h = np.maximum(z, 0.0)
            else:
                h = z
            acts.append(h)
        if train:
            self._acts = acts
            self._pres = pres
        return float(h[0])

    def train_step(self, x: np.ndarray, target: float, lr: float = 0.04) -> float:
        y = self.forward(x, train=True)
        delta = np.array([(y - target) * (1.0 - y * y)], dtype=np.float64)
        for i in reversed(range(len(self.w))):
            h = self._acts[i]
            self.w[i] -= lr * np.outer(delta, h)
            self.b[i] -= lr * delta
            if i == 0:
                break
            delta = self.w[i].T @ delta
            z = self._pres[i - 1]
            if self.act == "relu":
                delta = delta * (z > 0.0)
            elif self.act == "tanh":
                th = np.tanh(z)
                delta = delta * (1.0 - th * th)
        self.n_fit += 1
        return (y - target) ** 2


class TripletBook:
    """One pair's triangle and the eight nets that read it."""

    def __init__(self, symbol: str) -> None:
        self.symbol = str(symbol or "").upper()
        self.legs = TRIPLETS[self.symbol]
        self.series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        seed = sum(ord(ch) for ch in self.symbol) % (2**32)
        rng = np.random.default_rng(seed)
        self.nets = [_Net(hidden, act, rng) for _name, hidden, act in _SPECS]
        self.nll = np.ones(N_NETS, dtype=np.float64)
        self.n_updates = 0
        self._pending: list[tuple[np.ndarray, float]] = []
        self.persist = True

    @classmethod
    def open(cls, symbol: str, traded_bars: list[dict], load_bars, *, fresh: bool = False) -> TripletBook | None:
        """Attach the traded bars and the other two legs. None if the triangle is missing."""
        if triplet_for(symbol) is None:
            return None
        book = cls(symbol)
        if not fresh:
            book.load()
        book.attach(book.symbol, traded_bars)
        for leg in book.legs:
            if leg == book.symbol or leg in book.series:
                continue
            bars: list[dict] = []
            try:
                loaded, _src = load_bars(leg, "M5", 0, None, None)
                bars = list(loaded or [])
            except Exception:
                bars = []
            if bars:
                book.attach(leg, bars)
        book.ensure_cross()
        if not book.ready():
            return None
        return book

    def path(self) -> Path:
        return BRAIN_DIR / f"{self.symbol}.tri.npz"

    def attach(self, symbol: str, bars: list[dict]) -> None:
        times: list[int] = []
        closes: list[float] = []
        for bar in bars or []:
            try:
                stamp = int(bar.get("time") or 0)
                close = float(bar.get("close") or 0.0)
            except (TypeError, ValueError):
                continue
            if stamp <= 0 or close <= 0:
                continue
            times.append(stamp)
            closes.append(close)
        if len(times) < 2:
            return
        order = np.argsort(np.asarray(times), kind="stable")
        t = np.asarray(times, dtype=np.int64)[order]
        c = np.asarray(closes, dtype=np.float64)[order]
        last = np.ones(len(t), dtype=bool)
        last[:-1] = t[1:] != t[:-1]
        self.series[str(symbol).upper()] = (t[last], c[last])

    def ensure_cross(self) -> None:
        cross = self.legs[2]
        if cross in self.series:
            return
        spec = IMPLIED.get(cross)
        if spec is None:
            return
        a, b, how = spec
        if a not in self.series or b not in self.series:
            return
        ta, ca = self.series[a]
        tb, cb = self.series[b]
        times: list[int] = []
        closes: list[float] = []
        for stamp, px in zip(ta, ca):
            j = int(np.searchsorted(tb, int(stamp), side="right")) - 1
            if j < 0:
                continue
            if int(stamp) - int(tb[j]) > 3 * 300:
                continue
            other = float(cb[j])
            if other <= 0 or px <= 0:
                continue
            times.append(int(stamp))
            closes.append(px / other if how == "div" else px * other)
        if len(times) >= 60:
            self.series[cross] = (np.asarray(times, dtype=np.int64), np.asarray(closes, dtype=np.float64))

    def ready(self) -> bool:
        return all(leg in self.series and len(self.series[leg][0]) >= 60 for leg in self.legs)

    def _at(self, symbol: str, stamp: int, back: int) -> float | None:
        times, closes = self.series[symbol]
        i = int(np.searchsorted(times, int(stamp), side="right")) - 1
        j = i - int(back)
        if i < 0 or j < 0:
            return None
        return float(closes[j])

    def features(self, stamp: int) -> np.ndarray | None:
        if not self.ready():
            return None
        short: list[float] = []
        long: list[float] = []
        for leg in self.legs:
            now = self._at(leg, stamp, 0)
            s = self._at(leg, stamp, 12)
            m = self._at(leg, stamp, 48)
            if now is None or s is None or m is None or s <= 0 or m <= 0:
                return None
            short.append(math.tanh((now - s) / s / 0.004))
            long.append(math.tanh((now - m) / m / 0.008))
        agree = [1.0 if a * b > 0 else -1.0 for a, b in zip(short, long)]
        residual = 0.0
        spec = IMPLIED.get(self.legs[2])
        if spec is not None:
            a, b, how = spec
            pa, pb, pc = (self._at(a, stamp, 0), self._at(b, stamp, 0), self._at(self.legs[2], stamp, 0))
            if pa and pb and pc and pa > 0 and pb > 0 and pc > 0:
                implied = pa / pb if how == "div" else pa * pb
                residual = math.tanh((pc / implied - 1.0) / 0.002)
        vec = np.asarray([*short, *long, *agree, residual], dtype=np.float64)
        if vec.shape != (N_IN,):
            return None
        return vec

    def _weights(self) -> np.ndarray:
        z = -self.nll / 0.05
        z = z - float(z.max())
        e = np.exp(z)
        return e / max(float(e.sum()), 1e-12)

    def _bma(self, x: np.ndarray) -> float:
        w = self._weights()
        score = 0.0
        for i, net in enumerate(self.nets):
            score += float(w[i]) * net.forward(x)
        return float(max(-1.0, min(1.0, score)))

    def predict(self, stamp: int) -> float:
        if self.n_updates < MIN_FIT:
            return 0.0
        x = self.features(stamp)
        if x is None:
            return 0.0
        return self._bma(x)

    def observe(self, stamp: int, close: float) -> float:
        """Score this bar, then teach the sample from one horizon ago."""
        score = self.predict(int(stamp))
        x = self.features(int(stamp))
        if x is not None and close > 0:
            self._pending.append((x, float(close)))
        if len(self._pending) > HORIZON:
            old_x, old_close = self._pending.pop(0)
            if old_close > 0 and close > 0:
                target = math.tanh((float(close) - old_close) / old_close / 0.004)
                self._teach(old_x, target)
        return score

    def _teach(self, x: np.ndarray, target: float) -> None:
        for i, net in enumerate(self.nets):
            err = net.train_step(x, target)
            self.nll[i] = 0.94 * float(self.nll[i]) + 0.06 * float(err)
        self.n_updates += 1

    def save(self) -> Path | None:
        if not self.persist:
            return None
        dest = self.path()
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob: dict[str, np.ndarray] = {
            "nll": self.nll.astype(np.float64),
            "meta": np.array([self.n_updates], dtype=np.int32),
        }
        for i, net in enumerate(self.nets):
            blob[f"n{i}fit"] = np.array([net.n_fit], dtype=np.int32)
            for k, w in enumerate(net.w):
                blob[f"n{i}w{k}"] = w
            for k, b in enumerate(net.b):
                blob[f"n{i}b{k}"] = b
        np.savez_compressed(dest, **blob)
        return dest

    def load(self) -> bool:
        path = self.path()
        if not path.exists():
            return False
        try:
            raw = np.load(path, allow_pickle=False)
        except (OSError, ValueError):
            return False
        try:
            self.n_updates = int(raw["meta"][0])
            self.nll = np.asarray(raw["nll"], dtype=np.float64)
            for i, net in enumerate(self.nets):
                net.n_fit = int(raw[f"n{i}fit"][0])
                for k in range(len(net.w)):
                    net.w[k] = np.asarray(raw[f"n{i}w{k}"], dtype=np.float64)
                    net.b[k] = np.asarray(raw[f"n{i}b{k}"], dtype=np.float64)
        except (KeyError, ValueError, IndexError):
            return False
        return True


def apply_triplet(feat: dict, score: float) -> dict:
    """Mix the triangle score into every category, then into the shared impulse."""
    z = float(max(-1.0, min(1.0, float(score or 0.0))))
    feat["triplet"] = z
    if abs(z) < 0.02:
        return feat
    cats = feat.get("categories") or {}
    touched = False
    for st in cats.values():
        if not isinstance(st, dict) or not st.get("available"):
            continue
        fused = (1.0 - BLEND) * float(st.get("fused") or 0.0) + BLEND * z
        st["fused"] = round(fused, 4)
        st["impulse"] = float(np.tanh(1.05 * fused))
        parts = dict(st.get("parts") or {})
        parts["triplet"] = round(z, 4)
        st["parts"] = parts
        touched = True
    if not touched:
        return feat
    from flyfx.sense.category_kalman import fuse_category_latents

    mix = fuse_category_latents(cats)
    if not mix.get("available"):
        return feat
    feat["impulse"] = float(mix.get("impulse") or feat.get("impulse") or 0.0)
    feat["fused"] = float(mix.get("fused") or feat.get("fused") or 0.0)
    kal = dict(feat.get("kalman") or {})
    kal["impulse"] = feat["impulse"]
    kal["fused"] = feat["fused"]
    kal["triplet"] = z
    feat["kalman"] = kal
    return feat
