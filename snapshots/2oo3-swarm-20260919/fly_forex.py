"""
fly_forex.py — Fly-brain forex trader

Loads the MaleCNS 166k connectome, runs a leaky tanh rate network, and trades
through the FlyTrader.mq4 ZeroMQ bridge.

Dependencies: numpy, scipy, pyzmq, huggingface_hub
  pip install numpy scipy pyzmq huggingface_hub
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
import zmq
from scipy.sparse import csr_matrix

from account_sim import AccountSim
from bayes_sizer import (
    AdaptiveParams,
    BayesianSizer,
    SetupScorer,
    estimate_tax,
    load_adapt_state,
    save_adapt_state,
)
from kalman_signal import IndicatorKalmanFusion

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ─── defaults ─────────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parent
CONNECTOME_DIR = ROOT / "fly-connectome"
HF_REPO        = "fernandofernandes/fly-connectome-malecns-166k"
N_NEURONS      = 166_700
SYMBOL         = "EURUSD"
LOTS           = 0.01
DECISION_MS    = 500
SUBSTEPS       = 10
SYN_GAIN       = 0.02          # keeps tanh in a usable range for raw synaptic counts
LEAK           = 0.15          # x <- (1-LEAK)*tanh(...) + LEAK*x
ZMQ_REP        = "tcp://127.0.0.1:5555"
HISTORY        = 80
EMA_FAST       = 21
EMA_SLOW       = 55
RSI_PERIOD     = 14
ATR_PERIOD     = 14
MIN_HOLD_BARS  = 4
SL_ATR         = 2.4
TP_ATR         = 12.0
TRAIL_ARM_ATR  = 3.2
TRAIL_GAP_ATR  = 1.8
BE_ATR         = 1.5
CHOP_ATR_SEP   = 0.18
SPREAD         = 0.00010
WARMUP_BARS    = 55
FLAT_HOUR      = 18
COOLDOWN_BARS  = 8
MAX_HOLD_BARS  = 64
MAX_DAY_LOSSES = 2
MAX_DAY_TRADES = 2
LOCK_WIN_PIPS  = 8.0
RISK_PCT       = 1.0
MIN_LOTS       = 0.01
MAX_LOTS       = 20.0
LOT_STEP       = 0.01
MARGIN_CAP_PCT = 25.0
MIN_STOP_PIPS  = 8.0
MIN_CONF       = 45.0
TAX_RATE       = 24.0

MT4_ERRORS = {
    1: "no result",
    2: "common error",
    64: "account disabled",
    65: "invalid account",
    128: "trade timeout",
    129: "invalid price",
    130: "invalid stops",
    131: "invalid trade volume",
    132: "market closed",
    133: "trade disabled",
    134: "not enough money",
    135: "price changed",
    136: "off quotes",
    138: "requote",
    139: "order locked",
    141: "too many requests",
    145: "modification denied",
    146: "trade context busy",
    147: "expiration denied",
    148: "too many orders",
}


def pip_size(price: float) -> float:
    return 0.01 if price > 20 else 0.0001


def _ema(prev: float | None, price: float, period: int) -> float:
    k = 2.0 / (period + 1.0)
    return price if prev is None else prev + k * (price - prev)


class Indicators:
    """EMA trend, RSI, and ATR from incoming OHLC bars."""

    def __init__(self):
        self.closes: list[float] = []
        self.ema_fast: float | None = None
        self.ema_slow: float | None = None
        self.prev_close: float | None = None
        self.avg_gain: float | None = None
        self.avg_loss: float | None = None
        self.atr: float | None = None
        self.rsi = 50.0
        self.n = 0
        self.fusion = IndicatorKalmanFusion()

    def update(self, high: float, low: float, close: float) -> dict:
        self.n += 1
        self.closes.append(close)
        if len(self.closes) > 200:
            self.closes.pop(0)

        self.ema_fast = _ema(self.ema_fast, close, EMA_FAST)
        self.ema_slow = _ema(self.ema_slow, close, EMA_SLOW)

        if self.prev_close is None:
            tr = max(high - low, pip_size(close))
        else:
            tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))
            change = close - self.prev_close
            gain, loss = max(change, 0.0), max(-change, 0.0)
            if self.avg_gain is None:
                self.avg_gain, self.avg_loss = gain, loss
            else:
                p = RSI_PERIOD
                self.avg_gain = (self.avg_gain * (p - 1) + gain) / p
                self.avg_loss = (self.avg_loss * (p - 1) + loss) / p
            if self.avg_loss and self.avg_loss > 1e-12:
                rs = self.avg_gain / self.avg_loss
                self.rsi = 100.0 - 100.0 / (1.0 + rs)
            elif self.avg_gain and self.avg_gain > 0:
                self.rsi = 100.0

        if self.atr is None:
            self.atr = max(tr, pip_size(close))
        else:
            self.atr = (self.atr * (ATR_PERIOD - 1) + tr) / ATR_PERIOD

        self.prev_close = close
        atr = max(self.atr, pip_size(close))
        sep = abs(self.ema_fast - self.ema_slow) / atr
        if self.ema_fast > self.ema_slow and close > self.ema_slow and sep > CHOP_ATR_SEP:
            regime = "UP"
        elif self.ema_fast < self.ema_slow and close < self.ema_slow and sep > CHOP_ATR_SEP:
            regime = "DOWN"
        else:
            regime = "CHOP"

        mom = 0.0
        if len(self.closes) >= 6:
            mom = (self.closes[-1] - self.closes[-6]) / atr

        kalman = self.fusion.update(close, atr, self.ema_fast, self.ema_slow, self.rsi, mom)
        if kalman["kalman_regime"] == "CHOP":
            regime = "CHOP"
        elif kalman["kalman_regime"] != regime:
            regime = kalman["kalman_regime"] if abs(kalman["fused"]) > 0.45 else "CHOP"

        return {
            "ema_fast": self.ema_fast,
            "ema_slow": self.ema_slow,
            "rsi": self.rsi,
            "atr": atr,
            "regime": regime,
            "impulse": kalman["impulse"],
            "fused": kalman["fused"],
            "ready": self.n >= WARMUP_BARS,
            "sep": sep,
            "kalman": kalman,
        }


def _explain(reply: str) -> str:
    if not reply.startswith("ERR|"):
        return reply
    parts = reply.split("|")
    if len(parts) >= 2 and parts[1].isdigit():
        code = int(parts[1])
        return f"{reply} ({MT4_ERRORS.get(code, 'unknown')})"
    return reply


# ─── connectome ───────────────────────────────────────────────────────────
def ensure_connectome(root: Path) -> Path:
    offsets = root / "graph" / "edges_offsets.i32"
    source  = root / "graph" / "edges_source.i32"
    weight  = root / "graph" / "edges_weight.f32"
    needed  = (666_804, 102_331_752, 102_331_752)
    files   = (offsets, source, weight)
    if all(p.exists() and p.stat().st_size == sz for p, sz in zip(files, needed)):
        return root

    print("Connectome graph binaries are missing or incomplete.")
    print(f"Downloading {HF_REPO} into {root} ...")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=str(root),
        local_dir_use_symlinks=False,
    )
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError("Download finished but still missing: " + ", ".join(missing))
    return root


def load_population(root: Path, name: str) -> np.ndarray:
    path = root / "populations" / f"{name}.i32"
    if not path.exists():
        return np.array([], dtype=np.int32)
    return np.fromfile(path, dtype=np.int32)


def load_dopamine_indices(root: Path) -> np.ndarray:
    labels_path = root / "neurons" / "cell_type_labels.json"
    index_path  = root / "neurons" / "cell_type_index.i32"
    if not labels_path.exists() or not index_path.exists():
        return np.array([], dtype=np.int32)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    da_ids = [
        i for i, name in enumerate(labels)
        if isinstance(name, str) and (name.startswith("PAM") or name.startswith("PPL"))
    ]
    if not da_ids:
        return np.array([], dtype=np.int32)
    index = np.fromfile(index_path, dtype=np.int32)
    return np.flatnonzero(np.isin(index, np.array(da_ids, dtype=np.int32))).astype(np.int32)


class FlyBrain:
    """MaleCNS rate network: destination-major CSR, leaky tanh units.

    Fly photoreceptors are inhibitory, so a binary spike LIF with no tonic
    drive never leaves the retina. The connectome authors' own example is a
    leaky tanh rate model; that is what we run, then read motor intent out of
    the descending neurons with a calibrated linear decoder.
    """

    def __init__(self, root: Path, substeps: int = SUBSTEPS):
        print("Loading connectome...")
        offsets = np.fromfile(root / "graph" / "edges_offsets.i32", dtype=np.int32)
        source  = np.fromfile(root / "graph" / "edges_source.i32",  dtype=np.int32)
        weight  = np.fromfile(root / "graph" / "edges_weight.f32",  dtype=np.float32)
        n = int(offsets.size - 1)
        if n != N_NEURONS:
            print(f"  warning: expected {N_NEURONS} neurons, file has {n}")
        self.n = n
        self.W = csr_matrix(
            ((weight * np.float32(SYN_GAIN)).astype(np.float32), source, offsets),
            shape=(n, n),
        )
        print(f"  {n} neurons, {source.size:,} synapses")

        self.retina = load_population(root, "retina")
        self.descending = load_population(root, "descending")
        self.da = load_dopamine_indices(root)
        if self.retina.size == 0:
            self.retina = np.arange(200, 1200, dtype=np.int32)
        if self.descending.size == 0:
            self.descending = np.arange(0, 200, dtype=np.int32)
        print(f"  retina={self.retina.size}  descending={self.descending.size}  DA={self.da.size}")

        self.x = np.zeros(n, dtype=np.float32)
        self.reward = np.zeros(n, dtype=np.float32)
        self.rng = np.random.default_rng(7)
        self.readout = np.zeros(self.descending.size, dtype=np.float32)
        self.tau = 10.0
        self.substeps = max(1, substeps)
        self._calibrate_readout()

    def _drive_half(self, which: str, magnitude: float = 1.0) -> np.ndarray:
        currents = np.zeros(self.n, dtype=np.float32)
        mid = self.retina.size // 2
        if which == "UP":
            currents[self.retina[:mid]] = magnitude
        else:
            currents[self.retina[mid:]] = magnitude
        return currents

    def _calibrate_readout(self) -> None:
        print("  calibrating descending readout...")
        saved = self.x.copy()

        def probe(which: str) -> np.ndarray:
            self.x.fill(0.0)
            I = self._drive_half(which)
            for _ in range(self.substeps):
                self.step(I)
            return self.x[self.descending].copy()

        up, down = probe("UP"), probe("DOWN")
        self.readout = (up - down).astype(np.float32)
        score = float(np.dot(up, self.readout))
        self.tau = max(8.0, 0.25 * abs(score))
        self.x = saved
        print(f"  readout tau={self.tau:.2f}  (UP score {score:.1f})")

    def _retina_noise(self) -> np.ndarray:
        currents = np.zeros(self.n, dtype=np.float32)
        if self.retina.size:
            currents[self.retina] = 0.03 * self.rng.random(self.retina.size, dtype=np.float32)
        return currents

    def encode_trend(self, feat: dict) -> np.ndarray:
        """Follow Kalman impulse (motor intent with the trend)."""
        currents = self._retina_noise()
        impulse = float(feat.get("impulse", 0.0))
        residual = float(feat.get("kalman", {}).get("residual", 0.0))
        bull = max(0.0, impulse) * 0.75 + max(0.0, -residual) * 0.25
        bear = max(0.0, -impulse) * 0.75 + max(0.0, residual) * 0.25
        if bull > 0.05:
            currents += self._drive_half("UP", min(bull, 1.0))
        if bear > 0.05:
            currents += self._drive_half("DOWN", min(bear, 1.0))
        return currents

    def encode_fade(self, feat: dict) -> np.ndarray:
        """Fade Kalman residual (mean-reversion / anti-extension)."""
        currents = self._retina_noise()
        impulse = float(feat.get("impulse", 0.0))
        residual = float(feat.get("kalman", {}).get("residual", 0.0))
        bull = max(0.0, -residual) * 0.80 + max(0.0, -impulse) * 0.20
        bear = max(0.0, residual) * 0.80 + max(0.0, impulse) * 0.20
        if bull > 0.08:
            currents += self._drive_half("UP", min(bull, 1.0))
        if bear > 0.08:
            currents += self._drive_half("DOWN", min(bear, 1.0))
        return currents

    def features_to_input(self, feat: dict) -> np.ndarray:
        return self.encode_trend(feat)

    def price_to_input(self, price_history: list[float]) -> np.ndarray:
        currents = self._retina_noise()
        if len(price_history) < 2:
            return currents
        trend = price_history[-1] - price_history[0]
        mag = float(min(abs(trend) * 1000.0, 1.0))
        if trend > 0:
            currents += self._drive_half("UP", mag)
        elif trend < 0:
            currents += self._drive_half("DOWN", mag)
        return currents

    def step(self, currents: np.ndarray) -> np.ndarray:
        return self.step_state(self.x, currents, self.reward)

    def step_state(self, x: np.ndarray, currents: np.ndarray, reward: np.ndarray) -> np.ndarray:
        syn = self.W.dot(x)
        x[:] = (1.0 - LEAK) * np.tanh(syn + currents + reward) + LEAK * x
        np.clip(x, -1.0, 1.0, out=x)
        reward[:] = 0
        return x

    def run_window(self, currents: np.ndarray, substeps: int) -> tuple[str, float, float, int]:
        return self.run_window_state(self.x, currents, self.reward, substeps, tau_scale=1.0)

    def run_window_state(
        self,
        x: np.ndarray,
        currents: np.ndarray,
        reward: np.ndarray,
        substeps: int,
        tau_scale: float = 1.0,
    ) -> tuple[str, float, float, int]:
        x.fill(0.0)
        score = 0.0
        nfire = 0
        nstep = max(1, int(substeps) if substeps else self.substeps)
        for _ in range(nstep):
            self.step_state(x, currents, reward)
            score = float(np.dot(x[self.descending], self.readout))
            nfire = int(np.count_nonzero(np.abs(x) > 0.5))
        tau = self.tau * max(0.15, float(tau_scale))
        if score > tau:
            return "BUY", score, -score, nfire
        if score < -tau:
            return "SELL", score, -score, nfire
        return "HOLD", score, -score, nfire

    def deposit_reward(self, reward: np.ndarray, profit: float) -> None:
        if self.da.size == 0 or profit == 0:
            return
        kick = 0.5 if profit > 0 else -0.3
        reward[self.da] += np.float32(kick)

    def apply_reward(self, profit: float) -> None:
        self.deposit_reward(self.reward, profit)


class FlySwarm:
    """Two MaleCNS instances sharing W: trend follower + fade/reversion."""

    def __init__(self, core: FlyBrain):
        self.core = core
        n = core.n
        self.trend_x = np.zeros(n, dtype=np.float32)
        self.fade_x = np.zeros(n, dtype=np.float32)
        self.trend_r = np.zeros(n, dtype=np.float32)
        self.fade_r = np.zeros(n, dtype=np.float32)

    def vote(self, feat: dict, substeps: int) -> dict:
        t_vote, t_score, _, t_n = self.core.run_window_state(
            self.trend_x, self.core.encode_trend(feat), self.trend_r, substeps, tau_scale=0.55
        )
        f_vote, f_score, _, f_n = self.core.run_window_state(
            self.fade_x, self.core.encode_fade(feat), self.fade_r, substeps, tau_scale=1.0
        )
        return {
            "trend": t_vote,
            "trend_score": t_score,
            "fade": f_vote,
            "fade_score": f_score,
            "nfire": t_n + f_n,
        }

    def apply_reward(self, profit: float) -> None:
        self.core.deposit_reward(self.trend_r, profit)
        self.core.deposit_reward(self.fade_r, profit)


def flow_algo(feat: dict) -> str:
    """Cheap parallel voter: Kalman regime with a minimum impulse."""
    imp = float(feat.get("impulse", 0.0))
    regime = feat.get("regime", "CHOP")
    if regime == "UP" and imp >= 0.40:
        return "BUY"
    if regime == "DOWN" and imp <= -0.40:
        return "SELL"
    return "HOLD"


def committee(tech: str, kind: str, fly_trend: str, fly_fade: str, flow: str) -> tuple[str, str, int]:
    """2-out-of-3 on {tech, fly_trend, fly_fade}. Opposite vote is a veto.

    `flow` is a parallel algorithm: it cannot create a trade by itself, but a
    3oo3 plus flow agreement sizes up, and flow dissent without a fade veto
    still allows 2oo3 (diversity, not a fourth hard vote).
    Continuation shorts need 2oo3; a silent-fly bounce may pass as 1oo3.
    """
    if tech not in ("BUY", "SELL"):
        return "HOLD", "no-tech", 0
    votes = {"tech": tech, "fly_trend": fly_trend, "fly_fade": fly_fade}
    buy_n = sum(1 for v in votes.values() if v == "BUY")
    sell_n = sum(1 for v in votes.values() if v == "SELL")
    opp = "SELL" if tech == "BUY" else "BUY"
    if votes["fly_trend"] == opp or votes["fly_fade"] == opp:
        return "HOLD", f"veto {opp} (t={fly_trend} f={fly_fade} tech={tech})", 0
    agree = buy_n if tech == "BUY" else sell_n
    if agree >= 3:
        tag = "3oo3"
        if flow == tech:
            tag = "3oo3+flow"
        return tech, tag, agree
    if agree >= 2:
        tag = "2oo3"
        if flow == tech:
            tag = "2oo3+flow"
        return tech, tag, agree
    if kind == "bounce":
        return tech, "1oo3-bounce", 1
    return "HOLD", f"1oo3-skip ({kind} t={fly_trend} f={fly_fade})", 1


# ─── MT4 bridge ───────────────────────────────────────────────────────────
class MT4Bridge:
    def __init__(self, endpoint: str, timeout_ms: int = 5000):
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self.ctx = zmq.Context.instance()
        self.sock = None
        self._connect()

    def _connect(self) -> None:
        if self.sock is not None:
            self.sock.close(linger=0)
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.sock.connect(self.endpoint)

    def request(self, cmd: str) -> str:
        try:
            self.sock.send_string(cmd)
            return self.sock.recv_string()
        except zmq.ZMQError:
            self._connect()
            raise

    def ping(self) -> str:
        return self.request("PING")

    def rates(self, symbol: str) -> tuple[float, float]:
        reply = self.request(f"RATES|{symbol}")
        if reply.startswith("ERR"):
            raise RuntimeError(_explain(reply))
        parts = reply.split("|")
        return float(parts[1]), float(parts[2])

    def trade(self, side: str, symbol: str, lots: float) -> str:
        order_type = 0 if side == "BUY" else 1
        return self.request(f"TRADE|OPEN|{order_type}|{symbol}|{lots}|0|0")

    def close(self, ticket: int) -> str:
        return self.request(f"CLOSE|{ticket}")

    def close_all(self) -> str:
        return self.request("CLOSE|ALL")

    def history(self, symbol: str, timeframe: int, count: int) -> tuple[int, list[dict]]:
        reply = self.request(f"HISTORY|{symbol}|{timeframe}|{count}")
        if reply.startswith("ERR"):
            raise RuntimeError(_explain(reply))
        parts = reply.split("|")
        used_tf = timeframe
        chunks = parts[2:]
        if chunks and "," not in chunks[0] and chunks[0].isdigit():
            used_tf = int(chunks[0])
            chunks = chunks[1:]
        bars = []
        for chunk in chunks:
            if not chunk or "," not in chunk:
                continue
            fields = chunk.split(",")
            if len(fields) < 5:
                continue
            bars.append({
                "time": int(fields[0]),
                "open": float(fields[1]),
                "high": float(fields[2]),
                "low": float(fields[3]),
                "close": float(fields[4]),
            })
        return used_tf, bars


class PaperBroker:
    """Same interface as MT4Bridge, but never sends orders."""

    def __init__(self, inner: MT4Bridge | None):
        self.inner = inner
        self._next = 1000

    def ping(self) -> str:
        return self.inner.ping() if self.inner else "PONG|PAPER|0"

    def rates(self, symbol: str) -> tuple[float, float]:
        if self.inner:
            return self.inner.rates(symbol)
        t = time.time()
        mid = 1.08000 + 0.0015 * np.sin(t / 20.0)
        return float(mid - 0.00005), float(mid + 0.00005)

    def trade(self, side: str, symbol: str, lots: float) -> str:
        self._next += 1
        return f"OK|{self._next}|PAPER"

    def close(self, ticket: int) -> str:
        return f"OK|{ticket}"

    def close_all(self) -> str:
        return "OK|closed=0|failed=0"

    def history(self, symbol: str, timeframe: int, count: int) -> tuple[int, list[dict]]:
        if self.inner:
            return self.inner.history(symbol, timeframe, count)
        raise RuntimeError("No MT4 connection; cannot load history")


def wait_for_mt4(bridge: MT4Bridge, retries: int = 30) -> None:
    for i in range(retries):
        try:
            reply = bridge.ping()
            print(f"MT4 bridge: {reply}")
            return
        except zmq.Again:
            print(f"Waiting for FlyTrader.mq4 on {bridge.endpoint} ({i+1}/{retries})...")
            time.sleep(1)
        except zmq.ZMQError as exc:
            print(f"ZMQ error: {exc}; retrying...")
            time.sleep(1)
    raise SystemExit(
        "Could not reach MT4. Attach FlyTrader.mq4 to a chart, enable "
        "'Allow DLL imports' and 'Allow live trading', then retry."
    )


TIMEFRAMES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}


def cmd_ping(args: argparse.Namespace) -> None:
    bridge = MT4Bridge(args.endpoint)
    wait_for_mt4(bridge)
    print("account:", _explain(bridge.request("ACCOUNT")))
    try:
        bid, ask = bridge.rates(args.symbol)
        print(f"rates:   {args.symbol} {bid:.5f} / {ask:.5f}")
    except RuntimeError as exc:
        print("rates:  ", exc)
    print("ZMQ bridge is working. The market can still be closed for live orders.")


def _write_trade_report(
    trades: list[dict],
    path: Path,
    lots: float,
    account: AccountSim,
    bid: float,
    ask: float,
    tax_rate: float = TAX_RATE,
    tax_model: str = "ordinary",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "n", "side", "lots", "conf", "threshold", "cal_p", "open_time", "close_time",
        "entry", "exit", "pips", "gross", "commission", "slippage", "swap", "usd",
        "reason", "equity",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in trades:
            w.writerow(row)
    wins = [t for t in trades if t["usd"] > 0]
    losses = [t for t in trades if t["usd"] <= 0]
    net_usd = sum(t["usd"] for t in trades)
    net_pips = sum(t["pips"] for t in trades)
    snap = account.snapshot(bid, ask)
    print("\n──────── trade report (costs included) ────────")
    print(f"{'#':>3} {'side':<4} {'lots':>6} {'conf':>5} {'entry':>8} {'exit':>8} {'pips':>7} {'USD':>10}  reason")
    for t in trades:
        print(
            f"{t['n']:3d} {t['side']:<4} {t.get('lots', 0):6.2f} {t.get('conf', 0):5.0f} "
            f"{t['entry']:8.5f} {t['exit']:8.5f} "
            f"{t['pips']:+7.1f} {t['usd']:+10.2f}  {t['reason']}"
        )
    wr = (100.0 * len(wins) / len(trades)) if trades else 0.0
    print("──────────────────────────────────────────────")
    if not trades:
        print("No setups passed the session / pullback / cost filters.")
    print(
        f"{len(trades)} trades  {len(wins)}W/{len(losses)}L  win rate {wr:.0f}%  "
        f"net {net_pips:+.1f} pips  net ${net_usd:+.2f}"
    )
    print(
        f"costs  spread≈${account.cost_spread:.2f}  commission=${account.cost_commission:.2f}  "
        f"slippage=${account.cost_slippage:.2f}  swap=${account.cost_swap:.2f}"
    )
    print(
        f"account  start ${account.start_balance:,.2f}  balance ${snap['balance']:,.2f}  "
        f"equity ${snap['equity']:,.2f}  margin ${snap['margin']:.2f}  "
        f"free ${snap['free']:,.2f}  level {snap['level'] if snap['level'] != float('inf') else 'n/a'}"
        f"{'' if snap['level'] == float('inf') else '%'}"
    )
    ret = (snap["equity"] / account.start_balance - 1.0) * 100.0 if account.start_balance else 0.0
    avg_lots = (sum(t.get("lots", 0.0) for t in trades) / len(trades)) if trades else lots
    print(
        f"leverage 1:{account.leverage:.0f}  rejected {account.rejected}  "
        f"stop-outs {account.stopouts}  avg {avg_lots:.2f} lots  return {ret:+.3f}%"
    )
    hi = [t for t in trades if t.get("conf", 0) >= 60]
    lo = [t for t in trades if t.get("conf", 0) < 60]
    if hi or lo:
        def _wr(rows: list[dict]) -> str:
            if not rows:
                return "n/a"
            w = sum(1 for r in rows if r["usd"] > 0)
            return f"{w}/{len(rows)} ({100.0 * w / len(rows):.0f}%)"

        print(f"confidence  high>=60 {_wr(hi)}  low<60 {_wr(lo)}  (should separate as n grows)")
    if trades:
        bars = [float(t.get("threshold", 0) or 0) for t in trades]
        cals = [float(t.get("cal_p", 0) or 0) for t in trades]
        print(
            f"plausibility  bar {bars[0]:.0f}% -> {bars[-1]:.0f}%  "
            f"mean cal P(win) {sum(cals) / len(cals):.2f}"
        )
    tax = estimate_tax(net_usd, tax_rate, tax_model)
    print(
        f"tax  model={tax['note']}  estimated due ${tax['due']:,.2f}  "
        f"after-tax ${tax['after']:,.2f}  effective {tax['effective']:.1f}%"
    )
    print("      estimate only - not tax advice; set --tax-rate / --tax-model for your jurisdiction")
    print(f"saved {path}")


def in_session(ts: int) -> bool:
    """London through NY overlap; no weekend."""
    g = time.gmtime(ts)
    if g.tm_wday >= 5:
        return False
    return 8 <= g.tm_hour < 15


def should_flatten(ts: int) -> bool:
    g = time.gmtime(ts)
    if g.tm_wday == 4 and g.tm_hour >= 17:
        return True
    return g.tm_hour >= FLAT_HOUR


class PullbackSetup:
    """Arm on a dip/rally against the Kalman trend, fire on the bounce.

    The fly can veto (opposite vote) but HOLD does not block a clean setup.
    Strong trends may also fire a continuation after a one-bar pause.
    """

    def __init__(self):
        self.armed: str | None = None
        self.armed_at = -1
        self.prev_rsi = 50.0
        self.prev_close = None
        self.cooldown_until = 0
        self.lock_dir: str | None = None

    def cool(self, step: int, bars: int = COOLDOWN_BARS) -> None:
        self.cooldown_until = step + bars
        self.armed = None
        self.armed_at = -1

    def decide(self, feat: dict, close: float, step: int, p: AdaptiveParams) -> tuple[str, str]:
        if not feat["ready"] or step < self.cooldown_until:
            self.armed = None
            self.armed_at = -1
            return "HOLD", "hold"
        if feat["regime"] == "CHOP":
            self.armed = None
            self.armed_at = -1
            self.lock_dir = None
            self.prev_rsi, self.prev_close = feat["rsi"], close
            return "HOLD", "hold"

        rsi = feat["rsi"]
        impulse = abs(float(feat.get("impulse", 0.0)))
        bounced_up = rsi >= self.prev_rsi and (self.prev_close is None or close >= self.prev_close)
        bounced_dn = rsi <= self.prev_rsi and (self.prev_close is None or close <= self.prev_close)

        action = "HOLD"
        kind = "hold"
        if feat["regime"] == "UP":
            if self.armed != "BUY" and rsi < p.rsi_buy_arm:
                self.armed = "BUY"
                self.armed_at = step
            if (
                self.armed == "BUY"
                and step > self.armed_at
                and bounced_up
                and rsi < p.rsi_buy_fire
            ):
                action = "BUY"
                kind = "bounce"
                self.armed = None
                self.armed_at = -1
        elif feat["regime"] == "DOWN":
            if self.armed != "SELL" and rsi > p.rsi_sell_arm:
                self.armed = "SELL"
                self.armed_at = step
            if (
                self.armed == "SELL"
                and step > self.armed_at
                and bounced_dn
                and rsi > p.rsi_sell_fire
            ):
                action = "SELL"
                kind = "bounce"
                self.armed = None
                self.armed_at = -1
            elif (
                action == "HOLD"
                and impulse >= p.cont_impulse
                and p.rsi_cont_lo <= rsi <= p.rsi_cont_hi
                and bounced_dn
            ):
                action = "SELL"
                kind = "cont"

        if action != "HOLD" and abs(float(feat.get("impulse", 0.0))) < p.min_impulse:
            action, kind = "HOLD", "hold"
        if action == self.lock_dir:
            action, kind = "HOLD", "hold"

        self.prev_rsi, self.prev_close = rsi, close
        return action, kind


def costs_ok(
    account: AccountSim,
    lots: float,
    bid: float,
    ask: float,
    slip: float,
    atr: float,
    sl_atr: float,
    tp_atr: float,
) -> bool:
    spread = max(ask - bid, 0.0)
    pip = pip_size((bid + ask) / 2.0)
    if spread > 1.5 * pip:
        return False
    comm_px = account.commission_rt_per_lot / max(account.contract, 1.0)
    cost_px = spread + 2.0 * slip + comm_px
    if tp_atr * atr < 1.8 * cost_px:
        return False
    if sl_atr * atr < 1.1 * cost_px:
        return False
    return True


def round_lot(lots: float, step: float = LOT_STEP) -> float:
    if lots <= 0 or step <= 0:
        return 0.0
    return math.floor(lots / step + 1e-12) * step


def size_lots(
    account: AccountSim,
    equity: float,
    free: float,
    price: float,
    atr: float,
    risk_pct: float,
    fixed_lots: float,
    min_lots: float,
    max_lots: float,
    margin_cap_pct: float,
    bayes: BayesianSizer | None = None,
    best_win_usd: float = 0.0,
    sl_atr: float = SL_ATR,
    size_mult: float = 1.0,
    best_win_pips: float = 0.0,
    lock_pips: float = LOCK_WIN_PIPS,
) -> tuple[float, str]:
    """Risk a fraction of equity on the ATR stop; Bayesian scale when available."""
    pip = pip_size(price)
    stop_px = max(sl_atr * atr, MIN_STOP_PIPS * pip)
    stop_pips = stop_px / pip
    if risk_pct <= 0:
        lots = round_lot(min(max(fixed_lots, min_lots), max_lots))
        return lots, f"fixed {lots:.2f} lots  SL {stop_pips:.1f} pips"

    used_risk = risk_pct
    bayes_note = ""
    if bayes is not None:
        used_risk, bayes_note = bayes.risk_scale(risk_pct)

    used_risk *= max(0.45, float(size_mult))
    risk_usd = max(equity, 0.0) * (used_risk / 100.0)
    if best_win_usd > 0 and best_win_pips >= lock_pips:
        # Only cap 1R after a *meaningful* win so a tiny first profit cannot starve later trades.
        risk_usd = min(risk_usd, 0.90 * best_win_usd)
    raw = risk_usd / (stop_px * account.contract)
    margin_per_lot = account.required_margin(1.0, price)
    cap_margin = (max(free, 0.0) * (margin_cap_pct / 100.0) / margin_per_lot) if margin_per_lot > 0 else 0.0
    lots = round_lot(min(raw, max_lots, cap_margin))
    if lots < min_lots:
        min_risk = stop_px * account.contract * min_lots
        if min_risk > 2.0 * risk_usd:
            return 0.0, "size 0 (min lot risks more than 2x the cap)"
        lots = min_lots
    note = (
        f"{lots:.2f} lots  risk ${risk_usd:.0f} ({used_risk:.2f}% eq)  "
        f"SL {stop_pips:.1f} pips  ~${stop_px * account.contract * lots:.0f} if stopped"
    )
    if bayes_note:
        note = f"{note}  {bayes_note}"
    return lots, note


def trade_loop(brain: FlyBrain, prices, args: argparse.Namespace, broker, delay: float) -> None:
    indicators = Indicators()
    account = AccountSim(
        balance=getattr(args, "balance", 100_000.0),
        leverage=getattr(args, "leverage", 100.0),
        spread=getattr(args, "spread", SPREAD),
        commission_rt_per_lot=getattr(args, "commission", 7.0),
        slippage_pips=getattr(args, "slippage_pips", 0.2),
        swap_long_per_lot=getattr(args, "swap_long", -0.72),
        swap_short_per_lot=getattr(args, "swap_short", 0.12),
        stop_out=getattr(args, "stop_out", 50.0) / 100.0,
    )
    print(
        f"sim account ${account.start_balance:,.0f}  leverage 1:{account.leverage:.0f}  "
        f"commission ${account.commission_rt_per_lot:.2f}/lot RT  "
        f"slip {account.slippage_pips:.1f} pip  spread {account.spread:.5f}"
    )
    risk_pct = float(getattr(args, "risk", RISK_PCT))
    min_lots = float(getattr(args, "min_lots", MIN_LOTS))
    max_lots = float(getattr(args, "max_lots", MAX_LOTS))
    margin_cap = float(getattr(args, "margin_cap", MARGIN_CAP_PCT))
    if risk_pct > 0:
        print(
            f"sizing  {risk_pct:.2f}% of equity per trade, scaled by Bayesian half-Kelly  "
            f"lot {min_lots:.2f}-{max_lots:.2f}  margin cap {margin_cap:.0f}% of free"
        )
    else:
        print(f"sizing  fixed {args.lots:.2f} lots")
    bayes = BayesianSizer()
    params = AdaptiveParams()
    scorer = SetupScorer(bayes, min_conf=float(getattr(args, "min_conf", MIN_CONF)))
    min_conf = float(getattr(args, "min_conf", MIN_CONF))
    tax_rate = float(getattr(args, "tax_rate", TAX_RATE))
    tax_model = str(getattr(args, "tax_model", "ordinary"))
    state_path = Path(getattr(args, "adapt_state", "") or (ROOT / "reports" / "adapt_state.json"))
    if getattr(args, "reset_adapt", False) and state_path.exists():
        state_path.unlink()
        print(f"adapt  reset {state_path}")
    elif load_adapt_state(state_path, bayes, scorer, params):
        n_hist = bayes.n_win + bayes.n_loss
        print(
            f"adapt  loaded {state_path}  n={n_hist}  bar {scorer.threshold:.0f}%  "
            f"SL {params.sl_atr:.2f} ATR  TP {params.tp_atr:.1f} ATR  "
            f"impulse>={params.min_impulse:.2f}"
        )
    print(
        f"confidence  plausibility bar starts {scorer.threshold:.0f}% "
        f"(base {min_conf:.0f}%; evolves with calibration)"
    )
    print(
        f"params  SL {params.sl_atr:.2f} ATR  TP {params.tp_atr:.1f} ATR  "
        f"trail {params.trail_arm_atr:.2f}/{params.trail_gap_atr:.2f}  "
        f"RSI arm {params.rsi_buy_arm:.0f}/{params.rsi_sell_arm:.0f}  "
        f"min |impulse| {params.min_impulse:.2f}"
    )
    print(f"tax  {tax_rate:.1f}%  model={tax_model}  (estimate at report time)")
    swarm = FlySwarm(brain)
    print("committee  2oo3  voters: tech pullback | fly-trend | fly-fade   parallel: flow-algo")
    ticket = None
    open_side = None
    open_time = ""
    bars_held = 0
    sl = tp = 0.0
    trades: list[dict] = []
    step = 0
    lots = args.lots
    bid = ask = 0.0
    slip = 0.0
    stamp = 0
    setup = PullbackSetup()
    peak = 0.0
    day_key = None
    day_losses = 0
    day_trades = 0
    best_win_usd = 0.0
    best_win_pips = 0.0
    open_score: dict | None = None
    open_geom: dict | None = None

    def close_now(reason: str) -> None:
        nonlocal ticket, open_side, open_time, bars_held, sl, tp, peak, day_losses, best_win_usd, best_win_pips, open_score, open_geom
        reason_l = reason.lower()
        if "stop" in reason_l:
            setup.cool(step, bars=int(round(params.cooldown_bars)))
        res = account.close(bid, ask, slip, stamp, reason)
        if not res.ok:
            if ticket is not None:
                print(_explain(broker.close(ticket)))
            ticket = None
            open_side = None
            return
        pip = pip_size(res.exit_price)
        snap = account.snapshot(bid, ask)
        if reason == "ATR stop" and res.net > 0:
            reason = "trail stop"
        scored = open_score or {}
        row = {
            "n": len(trades) + 1,
            "side": open_side,
            "open_time": open_time,
            "close_time": time.strftime("%Y-%m-%d %H:%M", time.gmtime(stamp)) if stamp else str(step),
            "entry": round(res.entry_price, 5),
            "exit": round(res.exit_price, 5),
            "pips": round(res.net / (pip * account.contract * lots), 1) if lots else 0.0,
            "gross": round(res.gross, 2),
            "commission": round(res.commission, 2),
            "slippage": round(res.slippage, 2),
            "swap": round(res.swap, 2),
            "usd": round(res.net, 2),
            "lots": round(lots, 2),
            "conf": round(float(scored.get("conf", 0.0)), 1),
            "threshold": round(float(scored.get("threshold", scorer.threshold)), 1),
            "cal_p": scored.get("cal_p", ""),
            "p_edge": scored.get("p_edge", ""),
            "similar": scored.get("similar", ""),
            "recent_wr": scored.get("recent_wr", ""),
            "ta": scored.get("ta", ""),
            "risk": scored.get("risk", ""),
            "reason": reason,
            "equity": round(snap["equity"], 2),
            "impulse": scored.get("impulse", 0.0),
            "rsi": scored.get("rsi", 50.0),
            "votes": scored.get("votes", ""),
        }
        trades.append(row)
        mark = "WIN" if row["usd"] > 0 else "LOSS"
        print(
            f"[{row['close_time']}] CLOSE {open_side} {mark}  {row['lots']:.2f} lots  "
            f"conf {row['conf']:.0f}%  bar {row['threshold']:.0f}%  "
            f"{row['pips']:+.1f} pips  ${row['usd']:+.2f}  "
            f"comm ${row['commission']:.2f} slip ${row['slippage']:.2f} "
            f"swap ${row['swap']:.2f}  ({reason})"
        )
        if ticket is not None:
            print(f"       → {_explain(broker.close(ticket))}")
        brain.apply_reward(res.net)
        swarm.apply_reward(res.net)
        if row["usd"] <= 0:
            day_losses += 1
        elif row["pips"] >= params.lock_win_pips:
            setup.lock_dir = open_side
        if row["usd"] > best_win_usd:
            best_win_usd = row["usd"]
        if row["usd"] > 0 and row["pips"] > best_win_pips:
            best_win_pips = row["pips"]
        scorer.remember(row)
        param_note = params.update(row)
        save_adapt_state(state_path, bayes, scorer, params)
        print(
            f"       adapt  bar {scorer.threshold:.0f}%  cal P(win)={scorer.calibrated_p(row['conf']):.2f}  "
            f"P(edge) floor {scorer.p_edge_floor:.2f}  {param_note}"
        )
        ticket = None
        open_side = None
        bars_held = 0
        peak = 0.0
        open_score = None
        open_geom = None

    try:
        for bar in prices:
            high, low, close = bar["high"], bar["low"], bar["close"]
            stamp = bar["time"]
            label = time.strftime("%Y-%m-%d %H:%M", time.gmtime(stamp)) if stamp else str(step)
            feat = indicators.update(high, low, close)
            bid, ask, slip = account.quotes(close, stamp, high, low, feat["atr"])
            stop = account.mark(bid, ask, stamp)
            key = time.gmtime(stamp).tm_yday if stamp else step // 288
            if key != day_key:
                day_key = key
                day_losses = 0
                day_trades = 0
            need_fly = (
                ticket is None
                and feat["ready"]
                and feat["regime"] != "CHOP"
                and in_session(stamp)
            )
            fly_trend = fly_fade = flow = "HOLD"
            if need_fly:
                ballots = swarm.vote(feat, args.substeps)
                fly_trend, fly_fade = ballots["trend"], ballots["fade"]
                flow = flow_algo(feat)
            tech, kind = setup.decide(feat, close, step, params)
            decision, tag, n_agree = committee(tech, kind, fly_trend, fly_fade, flow)
            fly = fly_trend

            if stop and ticket is not None:
                close_now("stop-out")
            elif ticket is not None and should_flatten(stamp):
                close_now("session flatten")
            else:
                if step % 15 == 0:
                    snap = account.snapshot(bid, ask)
                    print(
                        f"[{label}] {close:.5f}  tech={tech:4s} t={fly_trend:4s} "
                        f"f={fly_fade:4s} flow={flow:4s} → {decision:4s} {tag}  "
                        f"trend={feat['regime']:<4} k={feat['impulse']:+.2f} "
                        f"RSI={feat['rsi']:.0f}  eq ${snap['equity']:.2f}"
                    )

                if ticket is None and decision in ("BUY", "SELL") and feat["ready"]:
                    room = day_losses < MAX_DAY_LOSSES and day_trades < MAX_DAY_TRADES
                    if room and in_session(stamp):
                        snap = account.snapshot(bid, ask)
                        pip = pip_size((bid + ask) / 2.0)
                        stop_pips = max(params.sl_atr * feat["atr"], MIN_STOP_PIPS * pip) / pip
                        risk_usd_guess = snap["equity"] * (risk_pct / 100.0) if risk_pct > 0 else 0.0
                        scored = scorer.score(
                            decision, feat, fly_trend, snap["equity"], risk_usd_guess, stop_pips
                        )
                        scored["votes"] = f"tech={tech}/{kind} t={fly_trend} f={fly_fade} flow={flow} {tag}"
                        ok, why, size_mult = scorer.gate(scored, tag)
                        if not ok:
                            print(f"[{label}] skip {decision}: {why}  ({scored['note']})")
                        else:
                            lots, size_note = size_lots(
                                account,
                                snap["equity"],
                                snap["free"],
                                (bid + ask) / 2.0,
                                feat["atr"],
                                risk_pct,
                                args.lots,
                                min_lots,
                                max_lots,
                                margin_cap,
                                bayes,
                                best_win_usd,
                                sl_atr=params.sl_atr,
                                size_mult=size_mult,
                                best_win_pips=best_win_pips,
                                lock_pips=params.lock_win_pips,
                            )
                            if lots >= min_lots and costs_ok(
                                account, lots, bid, ask, slip, feat["atr"],
                                params.sl_atr, params.tp_atr,
                            ):
                                fill = account.open(decision, lots, bid, ask, slip, stamp)
                                if not fill.ok:
                                    print(f"[{label}] skip {decision}: {fill.reason}")
                                else:
                                    result = _explain(broker.trade(decision, args.symbol, lots))
                                    print(
                                        f"[{label}] {decision} {args.symbol} {lots:.2f} lots @ {fill.price:.5f}  "
                                        f"margin ${fill.margin:.2f}  comm ${fill.commission:.2f}  "
                                        f"{why}  {scored['votes']}  {scored['note']}  {size_note}  "
                                        f"({feat['regime']}, k {feat['impulse']:+.2f})  → {result}"
                                    )
                                    ticket = int(result.split("|")[1]) if result.startswith("OK|") else fill.ticket
                                    open_side = decision
                                    open_time = label
                                    open_score = scored
                                    open_geom = params.snapshot()
                                    bars_held = 0
                                    day_trades += 1
                                    atr = feat["atr"]
                                    peak = fill.price
                                    sl_atr = open_geom["sl_atr"]
                                    tp_atr = open_geom["tp_atr"]
                                    if decision == "BUY":
                                        sl, tp = fill.price - sl_atr * atr, fill.price + tp_atr * atr
                                    else:
                                        sl, tp = fill.price + sl_atr * atr, fill.price - tp_atr * atr

                elif ticket is not None and account.pos is not None:
                    bars_held += 1
                    pos = account.pos
                    atr = feat["atr"]
                    g = open_geom or params.snapshot()
                    if pos.side == "BUY":
                        peak = max(peak, bid)
                        unreal = bid - pos.entry
                        if bid >= pos.entry + g["be_atr"] * atr:
                            sl = max(sl, pos.entry + 0.15 * atr)
                        if bid >= pos.entry + g["trail_arm_atr"] * atr:
                            sl = max(sl, bid - g["trail_gap_atr"] * atr)
                    else:
                        peak = min(peak, ask) if peak else ask
                        unreal = pos.entry - ask
                        if ask <= pos.entry - g["be_atr"] * atr:
                            sl = min(sl, pos.entry - 0.15 * atr)
                        if ask <= pos.entry - g["trail_arm_atr"] * atr:
                            sl = min(sl, ask + g["trail_gap_atr"] * atr)
                    hit_sl = (pos.side == "BUY" and bid <= sl) or (pos.side == "SELL" and ask >= sl)
                    hit_tp = (pos.side == "BUY" and bid >= tp) or (pos.side == "SELL" and ask <= tp)
                    trend_flip = (pos.side == "BUY" and feat["regime"] == "DOWN") or (
                        pos.side == "SELL" and feat["regime"] == "UP"
                    )
                    chopped = feat["regime"] == "CHOP" and unreal < 0.25 * atr
                    if hit_sl:
                        close_now("ATR stop")
                    elif hit_tp:
                        close_now("ATR target")
                    elif bars_held >= int(round(g["max_hold_bars"])):
                        close_now("time stop")
                    elif bars_held >= int(round(g["min_hold_bars"])) and trend_flip:
                        close_now("trend flip")
                    elif bars_held >= int(round(g["min_hold_bars"])) + 2 and chopped:
                        close_now("chop scratch")

            step += 1
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        print("\nStopping...")
        if ticket is not None:
            close_now("interrupt")

    if ticket is not None:
        close_now("end of data")

    save_adapt_state(state_path, bayes, scorer, params)
    print(
        f"adapt  saved {state_path}  bar {scorer.threshold:.0f}%  "
        f"skips {scorer.skips}  SL {params.sl_atr:.2f}  TP {params.tp_atr:.1f}  "
        f"impulse>={params.min_impulse:.2f}  RSI {params.rsi_buy_arm:.0f}/{params.rsi_sell_arm:.0f}"
    )
    if params.log:
        print("adapt  last param steps:")
        for line in params.log[-8:]:
            print(f"       {line}")
    report = ROOT / "reports" / time.strftime("trades-%Y%m%d-%H%M%S.csv")
    _write_trade_report(
        trades, report, lots, account, bid, ask, tax_rate=tax_rate, tax_model=tax_model
    )


def live_prices(broker, symbol: str):
    last = None
    while True:
        try:
            bid, ask = broker.rates(symbol)
            mid = (bid + ask) / 2.0
            high = max(last, mid) if last else mid
            low = min(last, mid) if last else mid
            last = mid
            yield {
                "time": int(time.time()),
                "open": mid,
                "high": high,
                "low": low,
                "close": mid,
                "bid": bid,
                "ask": ask,
            }
        except zmq.Again:
            print("MT4 timeout — retrying...")
            time.sleep(1)


def replay_prices(bars: list[dict], spread: float):
    for bar in bars:
        close = bar["close"]
        yield {
            "time": bar["time"],
            "open": bar["open"],
            "high": bar["high"],
            "low": bar["low"],
            "close": close,
            "bid": close,
            "ask": close + spread,
        }


def find_hst(symbol: str, period: int) -> Path | None:
    root = Path.home() / "AppData" / "Roaming" / "MetaQuotes" / "Terminal"
    name = f"{symbol}{period}.hst"
    hits = list(root.glob(f"*/history/*/{name}"))
    if not hits:
        return None
    return max(hits, key=lambda p: p.stat().st_mtime)


def load_hst(path: Path, count: int | None = None) -> list[dict]:
    raw = path.read_bytes()
    rec = struct.Struct("<qddddqiq")
    bars: list[dict] = []
    for off in range(148, len(raw) - rec.size + 1, rec.size):
        ts, o, h, l, c, _vol, _spr, _rvol = rec.unpack_from(raw, off)
        if ts <= 0 or c <= 0:
            continue
        bars.append({"time": int(ts), "open": o, "high": h, "low": l, "close": c})
    if count is not None and count > 0:
        bars = bars[-count:]
    return bars


def run(args: argparse.Namespace) -> None:
    if args.ping:
        cmd_ping(args)
        return

    root = ensure_connectome(Path(args.connectome))
    brain = FlyBrain(root, substeps=args.substeps)

    live = None
    hst_path = None
    if args.hst:
        if args.hst.lower() == "auto":
            tf_name = args.timeframe.upper()
            hst_path = find_hst(args.symbol, TIMEFRAMES.get(tf_name, 5))
            if hst_path is None:
                raise SystemExit(f"No {args.symbol} {tf_name} .hst file found under MetaQuotes/Terminal.")
        else:
            hst_path = Path(args.hst)
            if not hst_path.exists():
                raise SystemExit(f"HST file not found: {hst_path}")

    if not args.dry_run and hst_path is None:
        live = MT4Bridge(args.endpoint)
        wait_for_mt4(live)

    want_orders = not (args.paper or args.dry_run or args.replay or hst_path is not None)
    broker = PaperBroker(live) if not want_orders else live
    if args.dry_run:
        mode = "DRY-RUN"
    elif args.replay or hst_path is not None:
        mode = "REPLAY"
    elif args.paper:
        mode = "PAPER"
    else:
        mode = "LIVE"
    print(f"\nFly Forex Trader  {args.symbol}  {args.lots} lots  [{mode}]")

    if args.replay or hst_path is not None:
        tf_name = args.timeframe.upper()
        if tf_name not in TIMEFRAMES:
            raise SystemExit("timeframe must be one of: " + ", ".join(TIMEFRAMES))
        if hst_path is not None:
            want = max(args.bars, WARMUP_BARS + 20)
            print(f"Loading {want} x {tf_name} bars from {hst_path}...")
            bars = load_hst(hst_path, want)
            used_name = tf_name
        else:
            if live is None:
                raise SystemExit("--replay needs FlyTrader attached, or pass --hst auto")
            print(f"Loading {args.bars} x {tf_name} bars from MT4 (works while the market is closed)...")
            used_tf, bars = live.history(args.symbol, TIMEFRAMES[tf_name], min(400, max(args.bars, WARMUP_BARS + 20)))
            tf_names = {v: k for k, v in TIMEFRAMES.items()}
            used_name = tf_names.get(used_tf, str(used_tf))
            if used_tf != TIMEFRAMES[tf_name]:
                print(f"  MT4 had no {tf_name} series loaded; using {used_name} from the open chart instead.")
        if len(bars) < 5:
            raise SystemExit(
                "Not enough history. Switch the FlyTrader chart to M5 (or H1), "
                "click it, press Home a few times to load bars, then retry."
            )
        print(
            f"  {len(bars)} x {used_name}  "
            f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(bars[0]['time']))} -> "
            f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(bars[-1]['time']))}"
        )
        cache = ROOT / "reports" / "last_bars.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(bars), encoding="utf-8")
        trade_loop(brain, replay_prices(bars, args.spread), args, broker, delay=args.interval_ms / 1000.0)
        return

    print("Press Ctrl+C to stop.\n")
    if args.dry_run:
        def synthetic():
            t = time.time()
            step = 0
            price = 1.14800
            while True:
                price += 0.00015 * np.sin(step / 18.0) + 0.00002 * np.sin(step / 5.0)
                high, low = price + 0.00012, price - 0.00012
                yield {
                    "time": int(t + step * 300),
                    "open": price,
                    "high": high,
                    "low": low,
                    "close": price,
                    "bid": price,
                    "ask": price + SPREAD,
                }
                step += 1
        trade_loop(brain, synthetic(), args, broker, delay=args.interval_ms / 1000.0)
        return

    trade_loop(brain, live_prices(broker, args.symbol), args, broker, delay=args.interval_ms / 1000.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fly connectome forex trader (MT4 ZMQ)")
    p.add_argument("--connectome", default=str(CONNECTOME_DIR))
    p.add_argument("--endpoint", default=ZMQ_REP)
    p.add_argument("--symbol", default=SYMBOL)
    p.add_argument("--lots", type=float, default=LOTS, help="fixed lots when --risk is 0")
    p.add_argument("--risk", type=float, default=RISK_PCT, help="percent of equity to risk per trade (0 = fixed --lots)")
    p.add_argument("--min-lots", type=float, default=MIN_LOTS, dest="min_lots")
    p.add_argument("--max-lots", type=float, default=MAX_LOTS, dest="max_lots")
    p.add_argument("--margin-cap", type=float, default=MARGIN_CAP_PCT, dest="margin_cap", help="max percent of free margin per trade")
    p.add_argument("--interval-ms", type=int, default=DECISION_MS)
    p.add_argument("--substeps", type=int, default=SUBSTEPS)
    p.add_argument("--paper", action="store_true", help="read prices, do not send orders")
    p.add_argument("--dry-run", action="store_true", help="no MT4; synthetic prices")
    p.add_argument("--ping", action="store_true", help="only test the MT4 ZMQ bridge")
    p.add_argument("--replay", action="store_true", help="replay MT4 historical bars (works on weekends)")
    p.add_argument("--hst", default="", help="MT4 .hst path, or 'auto' to read the local terminal file")
    p.add_argument("--timeframe", default="M5", help="M1, M5, M15, M30, H1, H4, D1")
    p.add_argument("--bars", type=int, default=400, help="how many historical bars to replay")
    p.add_argument("--spread", type=float, default=SPREAD, help="base bid-ask spread in price")
    p.add_argument("--balance", type=float, default=100000.0, help="simulated starting balance")
    p.add_argument("--leverage", type=float, default=100.0, help="account leverage, e.g. 100 = 1:100")
    p.add_argument("--commission", type=float, default=7.0, help="round-turn commission USD per lot")
    p.add_argument("--slippage-pips", type=float, default=0.2, dest="slippage_pips")
    p.add_argument("--swap-long", type=float, default=-0.72, dest="swap_long", help="USD per lot per night, long")
    p.add_argument("--swap-short", type=float, default=0.12, dest="swap_short", help="USD per lot per night, short")
    p.add_argument("--stop-out", type=float, default=50.0, dest="stop_out", help="stop-out margin level percent")
    p.add_argument("--min-conf", type=float, default=MIN_CONF, dest="min_conf", help="base confidence bar percent; the live bar evolves from calibration")
    p.add_argument("--adapt-state", default="", dest="adapt_state", help="path to persist evolving confidence/params (default reports/adapt_state.json)")
    p.add_argument("--reset-adapt", action="store_true", dest="reset_adapt", help="ignore saved confidence/params and start from priors")
    p.add_argument("--tax-rate", type=float, default=TAX_RATE, dest="tax_rate", help="ordinary/short-term tax rate percent")
    p.add_argument("--tax-model", default="ordinary", dest="tax_model", help="ordinary (spot FX) or 1256 (60/40 futures-style)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
