"""Per-pair book knobs saved beside the brain.

``brains/{PAIR}.book.json`` holds the spike close, the strong-impulse size
lift, the day lock, and the tape floor. Replay reuses that file. The tune
checkbox is the only path that searches new values, and it stops once the
pair's return target is met (default 7%).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from flyfx.paths import BRAIN_DIR


@dataclass
class BookCfg:
    """Knobs that turned the EURUSD week from a small book into 7%+."""

    spike_mult: float = 1.85
    spike_floor: float = 2000.0
    worth_impulse: float = 0.92
    worth_frac: float = 0.70
    worth_span: float = 0.30
    day_lock_usd: float = 1000.0
    tape_floor: float = 0.90
    target_return: float = 0.07
    # 0 keeps the pair on the live geometry / the existing entry rule.
    # flat_hour 18 is the daily session flatten. 24 turns that hour off.
    entry_impulse: float = 0.0
    be_atr: float = 0.0
    trail_arm_atr: float = 0.0
    trail_gap_atr: float = 0.0
    max_hold_bars: float = 0.0
    flat_hour: float = 18.0
    peak_bank: float = 0.030
    # 0 keeps the risk profile's kill switch. A pair book may widen it.
    kill_dd: float = 0.0
    # 0 keeps the live TP. A pair book may place a farther target.
    tp_atr: float = 0.0
    # 1 follows the lab hold-risk switch. 0 keeps a pair from banking a dip.
    hold_risk: float = 1.0

    def brief(self) -> str:
        return (
            f"spike ×{self.spike_mult:.2f} from ${self.spike_floor:,.0f}  "
            f"worth |k|≥{self.worth_impulse:.2f} → {self.worth_frac:.0%}+ of equity  "
            f"day-lock ${self.day_lock_usd:,.0f}  tape floor {self.tape_floor:.2f}  "
            f"entry |k|≥{self.entry_impulse:.2f}  hold {self.max_hold_bars:.0f}  "
            f"flat {self.flat_hour:.0f}  target {self.target_return:.0%}"
        )


def book_path(symbol: str) -> Path:
    tag = str(symbol or "EURUSD").upper()
    return BRAIN_DIR / f"{tag}.book.json"


def load_book(symbol: str) -> BookCfg | None:
    path = book_path(symbol)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    base = BookCfg()
    data = raw.get("cfg") if isinstance(raw.get("cfg"), dict) else raw
    fields = asdict(base)
    for key in fields:
        if key in data and data[key] is not None:
            try:
                fields[key] = float(data[key])
            except (TypeError, ValueError):
                pass
    return BookCfg(**fields)


def save_book(
    symbol: str,
    cfg: BookCfg,
    *,
    net_usd: float | None = None,
    balance: float | None = None,
) -> Path:
    path = book_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbol": str(symbol or "EURUSD").upper(),
        "kind": "book",
        "cfg": asdict(cfg),
    }
    if net_usd is not None:
        payload["net_usd"] = round(float(net_usd), 2)
    if balance is not None and float(balance) > 0 and net_usd is not None:
        payload["return"] = round(float(net_usd) / float(balance), 4)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# One quiet replay of the saved book, then at most one proposal from each
# of the eight tuner nets. The search always ends and the simulation starts.
MAX_TUNE_EVALS = 9
N_TUNE_NETS = 8

# Each knob's ladder. A proposal moves one rung. The sign is the cold-start
# direction: loosen a gate that blocks entries, keep a tape floor, lower the
# day lock. Teaching can reverse a net after a step that did not pay.
_LADDERS: dict[str, tuple[float, ...]] = {
    "spike_mult": (1.50, 1.70, 1.85, 2.00, 2.30),
    "spike_floor": (500.0, 1000.0, 1500.0, 2000.0, 2500.0, 4000.0),
    "worth_impulse": (0.50, 0.70, 0.85, 0.92, 0.96),
    "worth_frac": (0.40, 0.55, 0.70, 0.85, 1.00),
    "worth_span": (0.10, 0.30, 0.50, 0.80),
    "day_lock_usd": (200.0, 500.0, 1000.0, 2000.0, 5000.0),
    "tape_floor": (0.50, 0.75, 0.90, 1.00),
    "entry_impulse": (0.0, 0.50, 0.70, 0.88),
    "be_atr": (0.0, 1.50, 8.0, 30.0),
    "trail_arm_atr": (0.0, 3.20, 8.0, 12.0),
    "trail_gap_atr": (0.0, 1.80, 6.0, 12.0),
    "max_hold_bars": (0.0, 64.0, 256.0, 700.0),
    "flat_hour": (18.0, 21.0, 24.0),
    "peak_bank": (0.030, 0.06, 0.12),
}

_SEEK: dict[str, int] = {
    "spike_mult": -1,
    "spike_floor": 1,
    "worth_impulse": 1,
    "worth_frac": 1,
    "worth_span": 1,
    "day_lock_usd": -1,
    "tape_floor": 1,
    "entry_impulse": -1,
    "be_atr": -1,
    "trail_arm_atr": -1,
    "trail_gap_atr": -1,
    "max_hold_bars": -1,
    "flat_hour": -1,
    "peak_bank": -1,
}

_KEYS: tuple[str, ...] = tuple(_LADDERS)


def _signature(trial: BookCfg) -> tuple[float, ...]:
    data = asdict(trial)
    return tuple(round(float(data[key]), 6) for key in _KEYS)


def _rung(key: str, value: float) -> int:
    options = _LADDERS[key]
    return min(range(len(options)), key=lambda i: abs(float(options[i]) - float(value)))


def _step(cfg: BookCfg, key: str, direction: int) -> BookCfg | None:
    options = _LADDERS[key]
    nxt = _rung(key, float(getattr(cfg, key))) + (1 if int(direction) >= 0 else -1)
    if nxt < 0 or nxt >= len(options):
        return None
    val = float(options[nxt])
    if abs(val - float(getattr(cfg, key))) <= 1e-9:
        return None
    return replace(cfg, **{key: val})


def _pack(cfg: BookCfg, ret_frac: float) -> np.ndarray:
    xs: list[float] = []
    for key in _KEYS:
        span = max(len(_LADDERS[key]) - 1, 1)
        xs.append(_rung(key, float(getattr(cfg, key))) / span)
    xs.append(float(np.clip(ret_frac, -1.0, 1.0)))
    xs.append(float(np.clip(float(cfg.target_return) - ret_frac, -1.0, 1.0)))
    return np.asarray(xs, dtype=np.float64)


class _TuneNet:
    """One small net. Hidden width differs across the group."""

    def __init__(self, rng: np.random.Generator, n_in: int, n_out: int, hidden: int) -> None:
        self.W1 = rng.normal(0.0, 0.02, size=(hidden, n_in))
        self.b1 = np.zeros(hidden, dtype=np.float64)
        self.W2 = rng.normal(0.0, 0.02, size=(n_out, hidden))
        self.b2 = np.zeros(n_out, dtype=np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = np.asarray(x, dtype=np.float64)
        self._h = np.tanh(self.W1 @ self._x + self.b1)
        self._y = np.tanh(self.W2 @ self._h + self.b2)
        return self._y

    def train(self, x: np.ndarray, target: np.ndarray, lr: float = 0.25) -> float:
        y = self.forward(x)
        err = y - np.asarray(target, dtype=np.float64)
        # d(tanh)/dpre = 1 - y^2, then one step back through the hidden tanh.
        dy = err * (1.0 - y * y)
        h = self._h
        dh = (self.W2.T @ dy) * (1.0 - h * h)
        self.W2 -= lr * np.outer(dy, h)
        self.b2 -= lr * dy
        self.W1 -= lr * np.outer(dh, self._x)
        self.b1 -= lr * dh
        return float(np.mean(err * err))


class BookTuneFleet:
    """Eight nets, each owning one knob. The group picks the next book to replay.

    Posterior weight rises when that net's step improves the book and falls
    when it does not. A later step is applied on top of the best book so two
    knobs can combine. The fleet does not write a brain file.
    """

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng(7)
        n_in = len(_KEYS) + 2
        n_out = len(_KEYS)
        self.specialty = [_KEYS[i % len(_KEYS)] for i in range(N_TUNE_NETS)]
        widths = (4, 8, 12, 6, 10, 8, 5, 16)
        self.nets = [
            _TuneNet(self.rng, n_in, n_out, widths[i]) for i in range(N_TUNE_NETS)
        ]
        for i, key in enumerate(self.specialty):
            self.nets[i].b2[_KEYS.index(key)] = 1.5 * float(_SEEK[key])
        self.nll = np.zeros(N_TUNE_NETS, dtype=np.float64)
        self.last_ret = 0.0

    def weights(self) -> np.ndarray:
        z = -self.nll
        z -= float(np.max(z))
        w = np.exp(np.clip(z, -20.0, 20.0))
        w = np.maximum(w, 1e-3)
        return w / float(np.sum(w))

    def propose(self, cfg: BookCfg, seen: set[tuple[float, ...]], cursor: int) -> tuple[BookCfg, int, int] | None:
        """Next unseen one-rung step at or after ``cursor``. None when the pass is over."""
        x = _pack(cfg, self.last_ret)
        start = max(int(cursor), 0)
        for i in range(start, N_TUNE_NETS):
            key = self.specialty[i]
            y = self.nets[i].forward(x)
            slot = _KEYS.index(key)
            direction = 1 if float(y[slot]) >= 0.0 else -1
            trial = _step(cfg, key, direction)
            if trial is None or _signature(trial) in seen:
                trial = _step(cfg, key, -direction)
            if trial is None or _signature(trial) in seen:
                continue
            return trial, i, i + 1
        return None

    def teach(self, cfg: BookCfg, net_index: int, improved: bool, direction: int) -> None:
        key = self.specialty[int(net_index)]
        slot = _KEYS.index(key)
        target = np.zeros(len(_KEYS), dtype=np.float64)
        if improved:
            target[slot] = 1.0 if int(direction) >= 0 else -1.0
            self.nll[int(net_index)] -= 1.0
        else:
            self.nll[int(net_index)] += 1.0
        x = _pack(cfg, self.last_ret)
        self.nets[int(net_index)].train(x, target)


def search_book(cfg: BookCfg, balance: float, evaluate) -> tuple[BookCfg, float]:
    """Replay the saved book, then let the eight nets propose a few changes.

    ``evaluate(cfg)`` returns net USD. The first call is the saved
    configuration. If that already meets the target, nothing else is replayed.
    Otherwise each net proposes one rung on top of the best book so far.
    The search stops when the target is met, the eight proposals are done,
    or ``MAX_TUNE_EVALS`` is reached. It does not keep walking after that.
    """
    bal = max(float(balance), 1.0)
    target = float(cfg.target_return)
    seen: set[tuple[float, ...]] = set()
    fleet = BookTuneFleet()

    def run(trial: BookCfg) -> float:
        seen.add(_signature(trial))
        return float(evaluate(trial))

    best = cfg
    best_net = run(best)
    fleet.last_ret = best_net / bal
    if best_net / bal >= target:
        return best, best_net
    cursor = 0
    evals = 1
    while evals < MAX_TUNE_EVALS:
        found = fleet.propose(best, seen, cursor)
        if found is None:
            break
        trial, net_index, cursor = found
        key = fleet.specialty[net_index]
        before = float(getattr(best, key))
        after = float(getattr(trial, key))
        direction = 1 if after >= before else -1
        net = run(trial)
        evals += 1
        fleet.last_ret = net / bal
        fleet.teach(best, net_index, net > best_net, direction)
        if net > best_net:
            best, best_net = trial, net
            fleet.last_ret = best_net / bal
        if best_net / bal >= target:
            return best, best_net
    return best, best_net
