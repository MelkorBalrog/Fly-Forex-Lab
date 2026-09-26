"""Per-pair book knobs saved beside the brain.

``brains/{PAIR}.book.json`` holds one pair's book. Replay reuses that file.
The tune checkbox runs one diagnostic replay. Eight nets read the whole week
plus those trades, and a Bayesian average of their outputs decides the book
changes. A pair whose saved return is already at the target is not replayed
again. The tuner does not walk knobs in a loop.
"""

from __future__ import annotations

import calendar
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
    # 0 trades both sides. 1 is buy only, -1 is sell only.
    side: float = 0.0
    # 0 does not cap the number of trades. 1 keeps the first trade only.
    max_trades: float = 0.0
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
    # 0 keeps the live stop. A pair book may set a wider one.
    sl_atr: float = 0.0
    # 1 follows the lab hold-risk switch. 0 keeps a pair from banking a dip.
    hold_risk: float = 1.0
    # 0 keeps the wide-spread veto. 1 lets a pair take the sell or buy anyway.
    wide_ok: float = 0.0
    # 1 keeps the Friday 17:00 flatten. 0 lets a pair hold through the weekend.
    friday_flat: float = 1.0
    # Multiplies entry size (1.0 = unchanged). Use for a small leverage-retune bump.
    size_boost: float = 1.0

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


def saved_return(symbol: str) -> float | None:
    """Return stored on the book file, if a finished replay wrote one."""
    path = book_path(symbol)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict) or raw.get("return") is None:
        return None
    try:
        return float(raw["return"])
    except (TypeError, ValueError):
        return None


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


def _open_stamp(text: str) -> int | None:
    raw = str(text or "")[:16]
    try:
        return int(calendar.timegm(time_struct(raw)))
    except (TypeError, ValueError, OverflowError):
        return None


def time_struct(text: str):
    import time as _time

    return _time.strptime(text, "%Y-%m-%d %H:%M")


def favorable_pips(side: str, entry: float, bars: list[dict], pip: float, opened: int) -> float:
    """Best pips available in the trade's direction from the entry to the end of the window."""
    px = float(entry)
    best = px
    step = max(float(pip), 1e-9)
    for bar in bars:
        try:
            stamp = int(bar.get("time") or 0)
        except (TypeError, ValueError):
            continue
        if stamp < int(opened):
            continue
        if str(side).upper() == "SELL":
            px_bar = bar.get("low") if bar.get("low") is not None else bar.get("close")
            best = min(best, float(px_bar if px_bar is not None else best))
        else:
            px_bar = bar.get("high") if bar.get("high") is not None else bar.get("close")
            best = max(best, float(px_bar if px_bar is not None else best))
    if str(side).upper() == "SELL":
        return (px - best) / step
    return (best - px) / step


def trend_run_book(cfg: BookCfg) -> BookCfg:
    """The book that held AUDUSD's sell through the rest of the drop.

    Farther target, no hold-risk bank on the dip, a longer hold, no session
    flatten, a trail that waits for the move, and a kill switch wide enough
    that one stopped-out trade does not end the week.
    """
    return replace(
        cfg,
        spike_mult=2.30,
        spike_floor=25000.0,
        worth_impulse=min(float(cfg.worth_impulse), 0.80),
        day_lock_usd=max(float(cfg.day_lock_usd), 8000.0),
        be_atr=30.0,
        trail_arm_atr=10.0,
        trail_gap_atr=6.0,
        max_hold_bars=max(float(cfg.max_hold_bars), 400.0),
        flat_hour=24.0,
        peak_bank=max(float(cfg.peak_bank), 0.20),
        kill_dd=max(float(cfg.kill_dd), 0.08),
        tp_atr=max(float(cfg.tp_atr), 40.0),
        hold_risk=0.0,
    )


_EARLY_EXIT = (
    "atr target",
    "hold-risk",
    "time stop",
    "session flatten",
    "chop scratch",
    "trend flip",
    "nn hold",
    "giveback",
    "bb closing",
)


# Week path, then what the diagnostic trades did with it.
# 0 drift, 1 range, 2 high-at, 3 low-at, 4 down, 5 up,
# 6 n, 7 win rate, 8 net return, 9 pips left, 10 winner side,
# 11 stop fraction, 12 gave the win back, 13 early cut, 14 kept pips, 15 gap to target
N_WEEK = 16
N_ADJ = 8
_ADJ_HEADS = ("side", "hold", "stop", "one", "trail", "worth", "kill", "gate")


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def week_vector(bars: list[dict], trades: list[dict], balance: float, pip: float) -> np.ndarray:
    """One vector for the whole window: the path first, then the trades."""
    step = max(float(pip), 1e-9)
    x = np.zeros(N_WEEK, dtype=np.float64)
    closes: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    for bar in bars or []:
        try:
            c = float(bar.get("close") or 0.0)
        except (TypeError, ValueError):
            continue
        if c <= 0:
            continue
        closes.append(c)
        try:
            highs.append(float(bar.get("high") if bar.get("high") is not None else c))
            lows.append(float(bar.get("low") if bar.get("low") is not None else c))
        except (TypeError, ValueError):
            highs.append(c)
            lows.append(c)
    n = len(closes)
    if n:
        drift = (closes[-1] - closes[0]) / step
        span = (max(highs) - min(lows)) / step
        x[0] = _clip(drift / 100.0, -1.5, 1.5)
        x[1] = _clip(span / 150.0, 0.0, 2.0)
        x[2] = highs.index(max(highs)) / max(n - 1, 1)
        x[3] = lows.index(min(lows)) / max(n - 1, 1)
        x[4] = 1.0 if drift <= -30.0 else 0.0
        x[5] = 1.0 if drift >= 30.0 else 0.0
    wins = 0
    losses = 0
    stops = 0
    net = 0.0
    best_win = 0.0
    winner_side = 0.0
    left = 0.0
    kept_best = 0.0
    gave = 0.0
    early = 0.0
    for trade in trades or []:
        try:
            usd = float(trade.get("usd") or 0.0)
            kept = abs(float(trade.get("pips") or 0.0))
            entry = float(trade.get("entry") or 0.0)
        except (TypeError, ValueError):
            continue
        net += usd
        reason = str(trade.get("reason") or "").lower()
        side = str(trade.get("side") or "").upper()
        if usd > 0:
            wins += 1
            if usd >= best_win:
                best_win = usd
                winner_side = -1.0 if side == "SELL" else 1.0
                kept_best = kept
            opened = _open_stamp(str(trade.get("open_time") or ""))
            if opened is not None and entry > 0:
                mfe = favorable_pips(side, entry, bars, pip, opened)
                extra = mfe - kept
                if extra > left:
                    left = extra
                cut = any(key in reason for key in _EARLY_EXIT) or kept < 0.55 * mfe
                if extra >= 20.0 and cut:
                    early = 1.0
        elif usd < 0:
            losses += 1
            if "stop" in reason:
                stops += 1
            if best_win > 0 and -usd > best_win * 0.5:
                gave = 1.0
    booked = wins + losses
    bal = max(float(balance), 1.0)
    x[6] = _clip(booked / 6.0, 0.0, 1.5)
    x[7] = (wins / booked) if booked else 0.0
    x[8] = _clip(net / bal, -0.2, 0.2) * 5.0
    x[9] = _clip(left / 50.0, 0.0, 2.0)
    x[10] = winner_side
    x[11] = (stops / losses) if losses else 0.0
    x[12] = gave
    x[13] = early
    x[14] = _clip(kept_best / 40.0, 0.0, 2.0)
    x[15] = _clip(0.07 - (net / bal), -0.2, 0.3) * 4.0
    return x


class _AdjustNet:
    """One small net. Hidden width differs; one hidden unit reads one week feature."""

    def __init__(
        self,
        rng: np.random.Generator,
        hidden: int,
        read_at: int,
        read_w: float,
        head: int,
        head_w: float,
    ) -> None:
        self.W1 = rng.normal(0.0, 0.01, size=(hidden, N_WEEK))
        self.b1 = np.zeros(hidden, dtype=np.float64)
        self.W2 = rng.normal(0.0, 0.01, size=(N_ADJ, hidden))
        self.b2 = np.zeros(N_ADJ, dtype=np.float64)
        self.W1[0, int(read_at)] = float(read_w)
        self.W2[int(head), 0] = float(head_w)
        self.head = int(head)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(self.W1 @ np.asarray(x, dtype=np.float64) + self.b1)
        return np.tanh(self.W2 @ h + self.b2)


class BookAdjustFleet:
    """Eight nets. Each one reads the week. Bayesian weights favor the net that owns that knob."""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        rng = rng or np.random.default_rng(11)
        # (hidden, feature index, feature weight, output head, output weight)
        specs = (
            (4, 0, 2.2, 0, 3.0),    # week drift → side
            (8, 13, 2.4, 1, 3.0),   # winner cut early → hold the move
            (12, 11, 2.2, 2, 3.0),  # stops → wider stop
            (6, 12, 2.6, 3, 3.0),   # a later trade gave the win back → one trade
            (10, 12, 2.4, 4, 3.0),  # same give-back → tighter trail
            (8, 13, 1.8, 5, 2.4),   # early cut → size the impulse
            (5, 4, 1.8, 6, 2.2),    # down week → don't let one loss end it
            (16, 11, 2.2, 7, 3.0),  # stops on the week → require a stronger impulse
        )
        self.nets = [
            _AdjustNet(rng, hidden, read_at, read_w, head, head_w)
            for hidden, read_at, read_w, head, head_w in specs
        ]
        self.prior = np.full((len(self.nets), N_ADJ), 0.04, dtype=np.float64)
        for i, net in enumerate(self.nets):
            self.prior[i, net.head] = 1.0

    def decide(self, x: np.ndarray) -> np.ndarray:
        ys = np.stack([net.forward(x) for net in self.nets], axis=0)
        w = self.prior
        return (w * ys).sum(axis=0) / np.maximum(w.sum(axis=0), 1e-9)


def prescribe_book(
    cfg: BookCfg,
    trades: list[dict],
    bars: list[dict],
    balance: float,
    pip: float,
) -> BookCfg:
    """Let the eight nets read the whole week and set this pair's book.

    A saved book is only changed when the averaged outputs ask for it.
    A winner that already kept the move leaves the book as it is.
    """
    x = week_vector(bars, trades, balance, pip)
    y = BookAdjustFleet().decide(x)
    out = cfg
    if float(y[1]) > 0.35:
        out = trend_run_book(out)
    if float(y[0]) <= -0.35 and float(x[4]) > 0.5:
        out = replace(out, side=-1.0)
    elif float(y[0]) >= 0.35 and float(x[5]) > 0.5:
        out = replace(out, side=1.0)
    if float(y[3]) > 0.45:
        out = replace(out, max_trades=max(float(out.max_trades), 1.0))
    if float(y[2]) > 0.45:
        out = replace(out, sl_atr=max(float(out.sl_atr), 8.0))
    if float(y[4]) > 0.45:
        out = replace(out, trail_arm_atr=8.0, trail_gap_atr=2.0)
    if float(y[7]) > 0.45 and float(out.entry_impulse) < 0.80:
        out = replace(out, entry_impulse=0.80)
    if float(y[6]) > 0.45:
        out = replace(out, kill_dd=max(float(out.kill_dd), 0.08))
    return out
