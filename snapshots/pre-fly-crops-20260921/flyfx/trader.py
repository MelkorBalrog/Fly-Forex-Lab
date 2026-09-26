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
import queue
import struct
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import zmq
from scipy.sparse import csr_matrix

from flyfx.brain.banc_risk import BancRiskBrain, risk_pack
from flyfx.brain.sugar_feed import (
    IDLE as SUGAR_IDLE,
    SugarDoseMachine,
    compute_satiety,
    feeding_from_raw,
    food_nutrition,
    mix_appetite,
    native_lamp_pack,
    parse_sugar_amt,
)
from flyfx.brain.fly_brains import (
    forward_r,
    is_frozen,
    load_brain,
    load_evo,
    list_brain_symbols,
    list_evo_policies,
    save_brain,
    seed_frozen_eurusd,
)
from flyfx.brain.evolve import (
    apply_policy,
    clip_gens,
    clip_pop,
    describe_genome,
    genome_from_args,
    max_dd_frac,
    public_genome,
)
from flyfx.brain.fly_nine import FlyNine, NODE_NAMES, SUBSTEPS_NINE
from flyfx.brain.plasticity import HeadPlasticity
from flyfx.exec.account_sim import AccountSim
from flyfx.exec.bar_io import (
    TIMEFRAMES,
    guess_symbol_from_name,
    load_bars_file,
    load_hst,
    parse_ohlc_text,
    parse_timeframe,
    resample_bars,
)
from flyfx.paths import CONNECTOME_DIR, PROJECT_ROOT as ROOT
from flyfx.risk.bayes_sizer import (
    AdaptiveParams,
    BayesianSizer,
    SetupScorer,
    estimate_tax,
    load_adapt_state,
    save_adapt_state,
)
from flyfx.risk.fly_rails import BookRails
from flyfx.risk.tolerance import overlay_geom, parse_risk_tol, risk_close_reason, risk_profile
from flyfx.risk.pair_book import PairBook, WIDE_PIPS, pair_book_enabled
from flyfx.brain.category_flies import SUBSTEPS_CAT, CategoryFlyPool
from flyfx.brain.fast_dot import ParallelSpmv
from flyfx.sense.category_indicators import CategoryIndicatorEngine
from flyfx.sense.category_kalman import (
    CATEGORY_NAMES,
    CategoryKalmanBank,
    encode_fuse_inds,
    fuse_category_latents,
    parse_fuse_cats,
    parse_fuse_inds,
)
from flyfx.sense.fuse_regime import FuseRegimeMachine, idle_fuse_plan
from flyfx.sense.kalman_signal import IndicatorKalmanFusion
from flyfx.ui.brain_map import CnsAtlas, pack_cns_maps
from flyfx.ui.dash import DashHub, idle_cat_fuse, idle_categories, idle_nodes, pack_category_sensors, pack_head, sensor_nodes
from flyfx.vote.category_tech import run_category_tech, vol_size_mult
from flyfx.vote.category_vote import CONF_INHIBIT_NORM, decide_categories
from flyfx.vote.nested_vote import nested_committee, pack_nodes
from flyfx.vote.bayes_ensemble import StrategyBook, combine_books, pack_ensemble_nodes, run_books
from flyfx.vote.node_fusion import (
    NodeBoard,
    conf_bar,
    fade_channels,
    gate_vote,
    tech_channels,
    trend_channels,
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ─── defaults ─────────────────────────────────────────────────────────────
HF_REPO        = "fernandofernandes/fly-connectome-malecns-166k"
N_NEURONS      = 166_700
SYMBOL         = "EURUSD"
PAIR_SPECS = {
    "EURUSD": {"spread": 0.00010, "quote": "USD", "yahoo": "EURUSD=X", "pip": 0.0001},
    "GBPUSD": {"spread": 0.00016, "quote": "USD", "yahoo": "GBPUSD=X", "pip": 0.0001},
    "AUDUSD": {"spread": 0.00014, "quote": "USD", "yahoo": "AUDUSD=X", "pip": 0.0001},
    "NZDUSD": {"spread": 0.00018, "quote": "USD", "yahoo": "NZDUSD=X", "pip": 0.0001},
    "USDCAD": {"spread": 0.00016, "quote": "CAD", "yahoo": "USDCAD=X", "pip": 0.0001},
    "USDCHF": {"spread": 0.00016, "quote": "CHF", "yahoo": "USDCHF=X", "pip": 0.0001},
    "USDJPY": {"spread": 0.012, "quote": "JPY", "yahoo": "USDJPY=X", "pip": 0.01},
    "EURJPY": {"spread": 0.018, "quote": "JPY", "yahoo": "EURJPY=X", "pip": 0.01},
    "EURGBP": {"spread": 0.00014, "quote": "GBP", "yahoo": "EURGBP=X", "pip": 0.0001},
    "GBPJPY": {"spread": 0.025, "quote": "JPY", "yahoo": "GBPJPY=X", "pip": 0.01},
}
DEFAULT_PAIRS = "EURUSD,GBPUSD,USDJPY,USDCHF,AUDUSD,EURGBP,USDCAD"
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
MAX_DAY_TRADES = 3
LOCK_WIN_PIPS  = 8.0
RISK_PCT       = 1.0
MIN_LOTS       = 0.01
MAX_LOTS       = 20.0
LOT_STEP       = 0.01
MARGIN_CAP_PCT = 25.0
MIN_STOP_PIPS  = 8.0
MIN_CONF       = 45.0
TAX_RATE       = 24.0
SNOWBALL_MAX_ADDS = 2
SNOWBALL_FRACTION = 0.55
SNOWBALL_MIN_PIPS = 5.5
SNOWBALL_ARM_ATR = 1.05
SNOWBALL_PAUSE_BARS = 4
SNOWBALL_CUSHION = 1.50
DASH_PORT      = 8765

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


def pip_size(price: float, symbol: str = "") -> float:
    spec = PAIR_SPECS.get((symbol or "").upper())
    if spec:
        return float(spec["pip"])
    return 0.01 if price > 20 else 0.0001


def pair_spec(symbol: str) -> dict:
    return dict(PAIR_SPECS.get((symbol or "").upper(), {"spread": SPREAD, "quote": "USD", "yahoo": f"{symbol}=X", "pip": 0.0001}))


class Indicators:
    """OHLCV indicators plus per-category Kalman fusion.

    Default mixer ``cat_fuse`` inverse-variance mixes the seven category
    latents into one impulse for the 3-fly 2oo3. ``legacy`` restores the
    EMA+RSI+ATR mixer used by ``--legacy-2oo3``. With ``fuse_dynamic``,
    a regime FSM picks the mix each bar; ``fuse_allow`` is the allowlist.
    """

    def __init__(self, pip: float = 0.0001, mixer: str = "cat_fuse"):
        self.pip = float(pip)
        self.mixer = mixer if mixer in ("legacy", "cat_fuse") else "cat_fuse"
        self.engine = CategoryIndicatorEngine(pip=self.pip)
        self.fusion = IndicatorKalmanFusion()
        self.cats = CategoryKalmanBank()
        self.fuse_dynamic = False
        self.fuse_allow: tuple[str, ...] = tuple(CATEGORY_NAMES)
        self.fuse_machine = FuseRegimeMachine()
        self.fuse_plan: dict = dict(idle_fuse_plan())
        self.fuse_inds: dict[str, tuple[str, ...]] = parse_fuse_inds("")
        self._dyn_was = False
        self.n = 0
        self.ema_fast: float | None = None
        self.ema_slow: float | None = None
        self.rsi = 50.0
        self.atr: float | None = None
        self.macd_sig: float | None = None
        self.bb_mid = 0.0

    def _sync_fuse_mix(self, raw: dict, cats: dict) -> dict:
        """Pick include (manual or FSM), then inverse-variance mix those latents."""
        if self.mixer != "legacy" and self.fuse_dynamic:
            if not self._dyn_was:
                self.fuse_machine = FuseRegimeMachine()
            self._dyn_was = True
            plan = self.fuse_machine.step(raw, cats, allow=self.fuse_allow)
            plan["dynamic"] = True
            self.cats.include = tuple(plan["include"])
            plan["inds"] = encode_fuse_inds(self.fuse_inds)
            self.fuse_plan = plan
        else:
            self._dyn_was = False
            if self.mixer != "legacy":
                self.cats.include = tuple(self.fuse_allow)
            self.fuse_plan = {
                "dynamic": False,
                "state": "OFF",
                "want": "OFF",
                "reason": "legacy mixer" if self.mixer == "legacy" else "manual mix",
                "include": list(self.cats.include),
                "bars_in": 0,
                "squeeze_bars": 0,
                "allow": list(self.fuse_allow),
                "inds": encode_fuse_inds(self.fuse_inds),
            }
        mix = fuse_category_latents(cats, include=self.cats.include)
        self.cats.mix = mix
        return mix

    def update(
        self,
        high: float,
        low: float,
        close: float,
        volume: float | None = None,
        open_: float | None = None,
        extras: dict | None = None,
        stamp: int | None = None,
    ) -> dict:
        raw = self.engine.update(
            high, low, close, volume=volume, open_=open_, extras=extras, stamp=stamp
        )
        self.n = self.engine.n
        self.ema_fast = raw["ema_fast"]
        self.ema_slow = raw["ema_slow"]
        self.rsi = raw["rsi"]
        self.atr = raw["atr"]
        self.macd_sig = raw["macd_sig"]
        self.bb_mid = raw["bb_mid"]
        atr = max(float(raw["atr"]), self.pip)
        self.cats.channel_include = self.fuse_inds
        cats = self.cats.update(raw.get("measurements") or {})
        mix = self._sync_fuse_mix(raw, cats)
        trend_k = cats.get("trend") or {}
        legacy = self.fusion.update(
            close, atr, float(raw["ema_fast"]), float(raw["ema_slow"]), float(raw["rsi"]), float(raw.get("mom") or 0.0)
        )
        if self.mixer == "legacy":
            impulse = float(legacy.get("impulse") or 0.0)
            fused = float(legacy.get("fused") or 0.0)
            k_reg = str(legacy.get("kalman_regime") or "CHOP")
            slope = legacy.get("slope")
            level = legacy.get("level")
            residual = legacy.get("residual")
            parts = legacy.get("parts") or {}
            uncertainty = legacy.get("uncertainty")
            innovation = legacy.get("innovation")
            agreement = legacy.get("agreement")
        else:
            impulse = float(mix.get("impulse") or 0.0)
            fused = float(mix.get("fused") or 0.0)
            k_reg = str(mix.get("regime") or "CHOP")
            slope = mix.get("slope", trend_k.get("slope"))
            level = mix.get("level", trend_k.get("level"))
            residual = mix.get("residual")
            parts = mix.get("parts") or {}
            uncertainty = mix.get("uncertainty")
            innovation = mix.get("innovation")
            agreement = mix.get("agreement")
        regime = str(raw.get("regime") or "CHOP")
        if k_reg == "CHOP":
            regime = "CHOP"
        elif k_reg != regime:
            regime = k_reg if abs(fused) > 0.45 else "CHOP"
        kalman = {
            "impulse": impulse,
            "fused": fused,
            "kalman_regime": k_reg,
            "slope": slope,
            "level": level,
            "residual": residual,
            "parts": parts,
            "uncertainty": uncertainty,
            "innovation": innovation,
            "agreement": agreement,
            "legacy": legacy,
            "cat_fuse": mix,
            "categories": cats,
            "fuse_plan": self.fuse_plan,
        }
        feat = dict(raw)
        feat.update(
            {
                "atr": atr,
                "regime": regime,
                "impulse": impulse,
                "fused": fused,
                "ready": bool(raw.get("ready")) or self.n >= WARMUP_BARS,
                "kalman": kalman,
                "categories": cats,
                "fuse_plan": self.fuse_plan,
                "fuse_inds": encode_fuse_inds(self.fuse_inds),
                "bar_high": high,
                "bar_low": low,
            }
        )
        return feat


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


def load_superclass_table(root: Path) -> tuple[list[str], np.ndarray]:
    labels_path = root / "neurons" / "superclass_labels.json"
    index_path = root / "neurons" / "superclass_index.i32"
    if not labels_path.exists() or not index_path.exists():
        return [], np.array([], dtype=np.int32)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(labels, list):
        return [], np.array([], dtype=np.int32)
    index = np.fromfile(index_path, dtype=np.int32)
    return [str(x) for x in labels], index


def load_superclass(root: Path, name: str) -> np.ndarray:
    labels, index = load_superclass_table(root)
    if not labels or index.size == 0:
        return np.array([], dtype=np.int32)
    try:
        sid = labels.index(name)
    except ValueError:
        return np.array([], dtype=np.int32)
    return np.flatnonzero(index == sid).astype(np.int32)


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
        self.spmv = ParallelSpmv(self.W)
        print(f"  SPMV  {self.spmv.n_threads} threads  row-blocked batched matmul")

        self.retina = load_population(root, "retina")
        self.descending = load_population(root, "descending")
        self.lamina = load_population(root, "lamina")
        self.da = load_dopamine_indices(root)
        sc_labels, sc_index = load_superclass_table(root)
        if sc_index.size != n:
            sc_index = np.array([], dtype=np.int32)
            sc_labels = []

        def _sc(name: str) -> np.ndarray:
            if not sc_labels or sc_index.size == 0:
                return np.array([], dtype=np.int32)
            try:
                sid = sc_labels.index(name)
            except ValueError:
                return np.array([], dtype=np.int32)
            return np.flatnonzero(sc_index == sid).astype(np.int32)

        self.ol_sensory = _sc("ol_sensory")
        self.vnc_sensory = _sc("vnc_sensory")
        self.sugar = load_population(root, "sugar")
        if self.retina.size == 0:
            self.retina = np.arange(200, 1200, dtype=np.int32)
        if self.descending.size == 0:
            self.descending = np.arange(0, 200, dtype=np.int32)
        print(
            f"  retina={self.retina.size}  descending={self.descending.size}  DA={self.da.size}  "
            f"ol_sensory={self.ol_sensory.size}  vnc_sensory={self.vnc_sensory.size}"
        )
        self.cns = CnsAtlas.build(
            n=n,
            retina=self.retina,
            descending=self.descending,
            sugar=self.sugar,
            da=self.da,
            ol_sensory=self.ol_sensory,
            vnc_sensory=self.vnc_sensory,
            lamina=self.lamina,
            sc_index=sc_index,
            sc_labels=sc_labels,
        )
        print(f"  cns map  {self.cns.idx.size} cells  {sum(1 for v in self.cns.counts.values() if v)} regions")

        self.x = np.zeros(n, dtype=np.float32)
        self.reward = np.zeros(n, dtype=np.float32)
        self.rng = np.random.default_rng(7)
        self._jitter = (
            (0.03 * self.rng.random(self.retina.size, dtype=np.float32))
            if self.retina.size
            else np.zeros(0, dtype=np.float32)
        )
        self._x_batch = np.zeros((n, 21), dtype=np.float32)
        self._i_batch = np.zeros((n, 21), dtype=np.float32)
        self._r_batch = np.zeros((n, 21), dtype=np.float32)
        self.readout = np.zeros(self.descending.size, dtype=np.float32)
        self.tau = 10.0
        self.substeps = max(1, substeps)
        self.last_bull = 0.0
        self.last_bear = 0.0
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
        if self.retina.size and self._jitter.size:
            currents[self.retina] = self._jitter
        return currents

    def drive_into(
        self,
        out: np.ndarray,
        bull: float,
        bear: float,
        plastic: HeadPlasticity | None = None,
        floor: float = 0.05,
    ) -> tuple[float, float]:
        """Write a retina drive into ``out`` (no 166k alloc)."""
        out.fill(0.0)
        if self.retina.size and self._jitter.size:
            out[self.retina] = self._jitter
        if plastic is not None:
            bull, bear = plastic.scale_drive(bull, bear)
        bull_f, bear_f = float(bull), float(bear)
        self.last_bull, self.last_bear = bull_f, bear_f
        mid = self.retina.size // 2
        if self.retina.size and bull_f > floor:
            out[self.retina[:mid]] += np.float32(min(bull_f, 1.0))
        if self.retina.size and bear_f > floor:
            out[self.retina[mid:]] += np.float32(min(bear_f, 1.0))
        return bull_f, bear_f

    def encode_trend(self, feat: dict, plastic: HeadPlasticity | None = None) -> np.ndarray:
        """Follow Kalman impulse (motor intent with the trend)."""
        impulse = float(feat.get("impulse", 0.0))
        residual = float(feat.get("kalman", {}).get("residual", 0.0))
        bull = max(0.0, impulse) * 0.75 + max(0.0, -residual) * 0.25
        bear = max(0.0, -impulse) * 0.75 + max(0.0, residual) * 0.25
        out = np.zeros(self.n, dtype=np.float32)
        self.drive_into(out, bull, bear, plastic, floor=0.05)
        return out

    def encode_fade(self, feat: dict, plastic: HeadPlasticity | None = None) -> np.ndarray:
        """Fade Kalman residual (mean-reversion / anti-extension)."""
        impulse = float(feat.get("impulse", 0.0))
        residual = float(feat.get("kalman", {}).get("residual", 0.0))
        bull = max(0.0, -residual) * 0.80 + max(0.0, -impulse) * 0.20
        bear = max(0.0, residual) * 0.80 + max(0.0, impulse) * 0.20
        out = np.zeros(self.n, dtype=np.float32)
        self.drive_into(out, bull, bear, plastic, floor=0.08)
        return out

    def encode_conf(
        self, feat: dict, pack: dict, plastic: HeadPlasticity | None = None
    ) -> np.ndarray:
        """Drive the confidence fly with robustness features, not price."""
        mag = float(
            np.clip(
                0.34 * float(pack.get("p_edge", 0.5))
                + 0.26 * float(pack.get("conf_proxy", 0.5))
                + 0.20 * float(pack.get("p_trend", 0.5))
                + 0.20 * float(pack.get("p_fade", 0.5)),
                0.0,
                1.0,
            )
        )
        side = (
            0.45 * float(pack.get("tech_fused", 0.0))
            + 0.30 * float(pack.get("trend_fused", 0.0))
            + 0.25 * float(feat.get("impulse", 0.0))
        )
        bull = mag * max(0.0, side) + 0.15 * mag
        bear = mag * max(0.0, -side) + 0.15 * mag
        out = np.zeros(self.n, dtype=np.float32)
        self.drive_into(out, bull, bear, plastic, floor=0.06)
        return out

    def encode_risk(self, pack: dict, plastic: HeadPlasticity | None = None) -> np.ndarray:
        """BANC has no retina: map risk onto MaleCNS smell + body-touch hubs.

        The 23 sugar GRNs get a small 0.45×nice lamp (4k-era native behavior).
        That does not pick BUY/SELL. The eat/skip overlay is a separate sugar feed.
        """
        currents = np.zeros(self.n, dtype=np.float32)
        nice = float(
            np.clip(
                0.55 * float(pack.get("p_edge", 0.5)) + 0.45 * float(pack.get("free_frac", 1.0)),
                0.0,
                1.0,
            )
        )
        bad = float(
            np.clip(
                0.45 * float(pack.get("drawdown", 0.0))
                + 0.30 * float(pack.get("day_losses", 0.0)) / 3.0
                + 0.25 * (1.0 - float(pack.get("p_edge", 0.5))),
                0.0,
                1.0,
            )
        )
        touch = float(
            np.clip(
                0.60 * float(pack.get("atr_stress", 0.0))
                + 0.40 * float(pack.get("spread_stress", 0.0)),
                0.0,
                1.0,
            )
        )
        if plastic is not None:
            nice, bad = plastic.scale_drive(nice, bad)
        self.last_bull, self.last_bear = float(nice), float(bad)
        ol = self.ol_sensory
        if ol.size:
            mid = max(ol.size // 2, 1)
            currents[ol[:mid]] = np.float32(nice)
            currents[ol[mid:]] = np.float32(bad)
        vnc = self.vnc_sensory
        if vnc.size:
            currents[vnc] = np.float32(touch)
        if self.sugar.size:
            currents[self.sugar] += np.float32(0.45 * nice)
        return currents

    def encode_sugar(self, nutrition: float) -> np.ndarray:
        """Sweet GRNs only: food on the tongue. Does not touch the retina or risk score."""
        currents = np.zeros(self.n, dtype=np.float32)
        mag = float(min(max(nutrition, 0.0), 1.0))
        if self.sugar.size:
            currents[self.sugar] = np.float32(mag)
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
        syn = self.spmv.dot(x)
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
        plastic: HeadPlasticity | None = None,
    ) -> tuple[str, float, float, int]:
        nstep = max(1, int(substeps) if substeps else self.substeps)
        readout = plastic.mix_readout(self.readout) if plastic is not None else self.readout
        # x=0 ⇒ W@x=0: skip the first 25M-synapse multiply.
        x.fill(0.0)
        np.tanh(currents + reward, out=x)
        x *= np.float32(1.0 - LEAK)
        np.clip(x, -1.0, 1.0, out=x)
        reward[:] = 0
        score = 0.0
        for _ in range(nstep - 1):
            self.step_state(x, currents, reward)
            score = float(np.dot(x[self.descending], readout))
        if nstep == 1:
            score = float(np.dot(x[self.descending], readout))
        nfire = int(np.count_nonzero(np.abs(x[self.descending]) > 0.5))
        if plastic is not None:
            plastic.trace(x[self.descending], self.last_bull, self.last_bear)
        tau = self.tau * max(0.15, float(tau_scale))
        if score > tau:
            return "BUY", score, -score, nfire
        if score < -tau:
            return "SELL", score, -score, nfire
        return "HOLD", score, -score, nfire

    def run_windows_batch(
        self,
        xs: list[np.ndarray],
        currents: list[np.ndarray],
        rewards: list[np.ndarray],
        substeps: int,
        tau_scales: list[float],
        plastics: list,
        bulls: list[float],
        bears: list[float],
    ) -> list[tuple[str, float, float, int]]:
        """Settle many heads with one streamed W per substep (same math as run_window_state)."""
        k = len(xs)
        if k == 0:
            return []
        if k == 1:
            self.last_bull, self.last_bear = float(bulls[0]), float(bears[0])
            return [
                self.run_window_state(
                    xs[0], currents[0], rewards[0], substeps, tau_scales[0], plastics[0]
                )
            ]
        nstep = max(1, int(substeps) if substeps else self.substeps)
        if k > self._x_batch.shape[1]:
            self._x_batch = np.zeros((self.n, k), dtype=np.float32)
            self._i_batch = np.zeros((self.n, k), dtype=np.float32)
            self._r_batch = np.zeros((self.n, k), dtype=np.float32)
        X = self._x_batch[:, :k]
        I = self._i_batch[:, :k]
        R = self._r_batch[:, :k]
        X.fill(0.0)
        for j in range(k):
            I[:, j] = currents[j]
            R[:, j] = rewards[j]
        np.tanh(I + R, out=X)
        X *= np.float32(1.0 - LEAK)
        np.clip(X, -1.0, 1.0, out=X)
        R.fill(0.0)
        one_m = np.float32(1.0 - LEAK)
        leak = np.float32(LEAK)
        for _ in range(nstep - 1):
            syn = self.spmv.dot(X)
            np.tanh(syn + I, out=syn)
            syn *= one_m
            X *= leak
            X += syn
            np.clip(X, -1.0, 1.0, out=X)
        desc = X[self.descending, :]
        n_desc = int(self.descending.size)
        reads = np.empty((n_desc, k), dtype=np.float32)
        for j, plastic in enumerate(plastics):
            if plastic is not None:
                reads[:, j] = plastic.mix_readout(self.readout)
            else:
                reads[:, j] = self.readout
        scores = np.sum(desc * reads, axis=0)
        nfires = np.count_nonzero(np.abs(desc) > 0.5, axis=0)
        out: list[tuple[str, float, float, int]] = []
        for j in range(k):
            xs[j][:] = X[:, j]
            rewards[j][:] = 0
            if plastics[j] is not None:
                plastics[j].trace(desc[:, j], float(bulls[j]), float(bears[j]))
            score = float(scores[j])
            nfire = int(nfires[j])
            tau = self.tau * max(0.15, float(tau_scales[j]))
            if score > tau:
                vote = "BUY"
            elif score < -tau:
                vote = "SELL"
            else:
                vote = "HOLD"
            out.append((vote, score, -score, nfire))
        return out

    def deposit_reward(self, reward: np.ndarray, profit: float) -> None:
        if self.da.size == 0 or profit == 0:
            return
        kick = 0.5 if profit > 0 else -0.3
        reward[self.da] += np.float32(kick)

    def apply_reward(self, profit: float) -> None:
        self.deposit_reward(self.reward, profit)


class FlySwarm:
    """MaleCNS instances sharing W: trend, fade/reversion, confidence, and VNC risk."""

    def __init__(self, core: FlyBrain, plastic: bool = True):
        self.core = core
        n = core.n
        n_desc = int(core.descending.size)
        self.plastic_on = bool(plastic)
        self.trend_x = np.zeros(n, dtype=np.float32)
        self.fade_x = np.zeros(n, dtype=np.float32)
        self.conf_x = np.zeros(n, dtype=np.float32)
        self.risk_x = np.zeros(n, dtype=np.float32)
        self.sugar_x = np.zeros(n, dtype=np.float32)
        self.trend_r = np.zeros(n, dtype=np.float32)
        self.fade_r = np.zeros(n, dtype=np.float32)
        self.conf_r = np.zeros(n, dtype=np.float32)
        self.risk_r = np.zeros(n, dtype=np.float32)
        self.sugar_r = np.zeros(n, dtype=np.float32)
        self.trend_p = HeadPlasticity(n_desc) if self.plastic_on else None
        self.fade_p = HeadPlasticity(n_desc) if self.plastic_on else None
        self.conf_p = HeadPlasticity(n_desc) if self.plastic_on else None
        self.risk_p = HeadPlasticity(n_desc, lr=0.03, decay=0.88) if self.plastic_on else None
        self._risk_rest = 0.0
        self._sugar_rest = (
            abs((1.0 - LEAK) * math.tanh(0.45)) if core.sugar.size else 0.0
        )
        if core.ol_sensory.size or core.vnc_sensory.size:
            self._risk_rest = self._risk_raw(
                {
                    "p_edge": 0.5,
                    "drawdown": 0.0,
                    "atr_stress": 0.55,
                    "spread_stress": 0.50,
                    "day_losses": 0.0,
                    "free_frac": 1.0,
                }
            )

    def vote(self, feat: dict, substeps: int) -> dict:
        t_drive = self.core.encode_trend(feat, self.trend_p)
        t_bull, t_bear = self.core.last_bull, self.core.last_bear
        f_drive = self.core.encode_fade(feat, self.fade_p)
        f_bull, f_bear = self.core.last_bull, self.core.last_bear
        results = self.core.run_windows_batch(
            [self.trend_x, self.fade_x],
            [t_drive, f_drive],
            [self.trend_r, self.fade_r],
            substeps,
            [0.55, 1.0],
            [self.trend_p, self.fade_p],
            [t_bull, f_bull],
            [t_bear, f_bear],
        )
        t_vote, t_score, _, t_n = results[0]
        f_vote, f_score, _, f_n = results[1]
        return {
            "trend": t_vote,
            "trend_score": t_score,
            "fade": f_vote,
            "fade_score": f_score,
            "nfire": t_n + f_n,
            "heads": {},
        }

    def vote_conf(self, feat: dict, pack: dict, substeps: int) -> tuple[str, float, dict]:
        vote, score, _, nfire = self.core.run_window_state(
            self.conf_x,
            self.core.encode_conf(feat, pack, self.conf_p),
            self.conf_r,
            substeps,
            tau_scale=0.70,
            plastic=self.conf_p,
        )
        head = pack_head(
            self.core, self.conf_x, vote, score, nfire, self.core.tau * 0.70, "conf", self.conf_p, viz=False
        )
        return vote, score, head

    def _risk_raw(self, pack: dict, substeps: int = 5) -> float:
        drive = self.core.encode_risk(pack, self.risk_p)
        self.risk_x.fill(0.0)
        nstep = max(3, int(substeps))
        for _ in range(nstep):
            self.core.step_state(self.risk_x, drive, self.risk_r)
        if self.risk_p is not None:
            self.risk_p.trace(self.risk_x[self.core.descending], self.core.last_bull, self.core.last_bear)
        ol = self.core.ol_sensory
        vnc = self.core.vnc_sensory
        nice = bad = touch = 0.0
        if ol.size:
            mid = max(ol.size // 2, 1)
            nice = float(np.mean(np.abs(self.risk_x[ol[:mid]])))
            bad = float(np.mean(np.abs(self.risk_x[ol[mid:]])))
        if vnc.size:
            touch = float(np.mean(np.abs(self.risk_x[vnc])))
        return (nice - bad) - 0.25 * touch

    def _native_sweet(self) -> float:
        """Mean |x| on the 23 GRNs after the risk pass (0.45×nice lamp). Does not change 2oo3."""
        sugar = self.core.sugar
        if not sugar.size:
            return 0.0
        self.sugar_x[sugar] = self.risk_x[sugar]
        return float(np.mean(np.abs(self.risk_x[sugar])))

    def vote_risk(self, pack: dict, substeps: int = 5) -> dict:
        """MaleCNS olfactory + VNC touch as a parallel body-risk voter (no retina)."""
        if not (self.core.ol_sensory.size or self.core.vnc_sensory.size):
            return {"mult": 1.0, "nice": 0.0, "bad": 0.0, "touch": 0.0, "sweet": 0.0, "source": "malecns-missing"}
        raw = self._risk_raw(pack, substeps=substeps)
        score = raw - self._risk_rest
        if self.risk_p is not None:
            score += 0.08 * (self.risk_p.gain_pos - 1.0) - 0.08 * (self.risk_p.gain_neg - 1.0)
        mult = float(np.clip(1.0 + 0.35 * math.tanh(2.2 * score), 0.55, 1.15))
        return {
            "mult": round(mult, 3),
            "score": round(score, 4),
            "sweet": round(self._native_sweet(), 4),
            "source": "malecns-ol+vnc",
        }

    def _sugar_raw(self, nutrition: float, substeps: int = 5) -> float:
        """Stamp the 23 GRNs only. Do not run W — a full SPMV is wasted if we only read sugar."""
        del substeps
        mag = float(min(max(nutrition, 0.0), 1.0))
        sugar = self.core.sugar
        if not sugar.size:
            return mag
        val = np.float32((1.0 - LEAK) * math.tanh(mag))
        self.sugar_x[sugar] = val
        return float(abs(val))

    def vote_sugar(self, nutrition: float, substeps: int = 5) -> dict:
        """Read the 23 sugar GRNs after a sweet drive. Never casts BUY/SELL."""
        if not self.core.sugar.size:
            feed = float(min(max(nutrition, 0.0), 1.0))
            return {"feeding": round(feed, 4), "raw": round(feed, 4), "n": 0, "source": "sugar-missing"}
        raw = self._sugar_raw(nutrition, substeps=substeps)
        feed = feeding_from_raw(raw, self._sugar_rest)
        return {
            "feeding": round(feed, 4),
            "raw": round(raw, 4),
            "n": int(self.core.sugar.size),
            "source": "malecns-sugar",
        }

    def apply_reward(self, profit: float, side: str = "") -> list[str]:
        self.core.deposit_reward(self.trend_r, profit)
        self.core.deposit_reward(self.fade_r, profit)
        self.core.deposit_reward(self.conf_r, profit)
        self.core.deposit_reward(self.risk_r, profit)
        notes: list[str] = []
        if not self.plastic_on or side not in ("BUY", "SELL"):
            return notes
        for name, head in (
            ("trend", self.trend_p),
            ("fade", self.fade_p),
            ("conf", self.conf_p),
            ("risk", self.risk_p),
        ):
            if head is not None:
                notes.append(f"{name} {head.learn(profit, side)}")
        return notes

    def apply_reward_r(self, r_mult: float, side: str = "") -> list[str]:
        """Shadow / weekly train: dopamine from R, not dollar P&L (lot size cannot bias votes)."""
        if r_mult == 0 or side not in ("BUY", "SELL"):
            return []
        proxy = 400.0 * float(np.tanh(r_mult))
        self.core.deposit_reward(self.trend_r, proxy)
        self.core.deposit_reward(self.fade_r, proxy)
        self.core.deposit_reward(self.conf_r, proxy)
        self.core.deposit_reward(self.risk_r, proxy)
        notes: list[str] = []
        if not self.plastic_on:
            return notes
        for name, head in (
            ("trend", self.trend_p),
            ("fade", self.fade_p),
            ("conf", self.conf_p),
            ("risk", self.risk_p),
        ):
            if head is not None:
                notes.append(f"{name} {head.learn_r(r_mult, side)}")
        return notes

    def dump_plastic(self) -> dict:
        out: dict = {}
        for name, head in (
            ("trend", self.trend_p),
            ("fade", self.fade_p),
            ("conf", self.conf_p),
            ("risk", self.risk_p),
        ):
            if head is not None:
                out[name] = head.dump()
        return out

    def load_plastic(self, data: dict) -> None:
        if not isinstance(data, dict):
            return
        for name, head in (
            ("trend", self.trend_p),
            ("fade", self.fade_p),
            ("conf", self.conf_p),
            ("risk", self.risk_p),
        ):
            if head is not None and name in data:
                head.load(data[name])

    def rollback_plastic(self) -> None:
        for head in (self.trend_p, self.fade_p, self.conf_p, self.risk_p):
            if head is not None:
                head.rollback()


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
            rec = {
                "time": int(fields[0]),
                "open": float(fields[1]),
                "high": float(fields[2]),
                "low": float(fields[3]),
                "close": float(fields[4]),
            }
            if len(fields) >= 6:
                try:
                    rec["volume"] = float(fields[5])
                except ValueError:
                    pass
            bars.append(rec)
        return used_tf, bars

    def settings(self) -> str:
        return self.request("SETTINGS")


def parse_ea_token(reply: str | None, key: str) -> str | None:
    text = str(reply or "").strip()
    if not text or text.upper().startswith("UNKNOWN"):
        return None
    parts = [p.strip() for p in text.split("|")]
    needle = str(key or "").strip().upper()
    for i, token in enumerate(parts):
        if token.upper() == needle and i + 1 < len(parts):
            spec = parts[i + 1].strip()
            if spec:
                return spec
    return None


def parse_ea_fuse_spec(reply: str | None) -> str | None:
    """Read fuse mix from ``PONG|…|FUSE|trend,volume`` or ``OK|FUSE|…``."""
    return parse_ea_token(reply, "FUSE")


def parse_ea_inds_spec(reply: str | None) -> str | None:
    """Read indicator mix from ``INDS|ema,macd,rsi`` / ``INDS|all``."""
    return parse_ea_token(reply, "INDS")


def parse_ea_risk_spec(reply: str | None) -> str | None:
    """Read close-side risk tolerance from ``RISK|balanced``."""
    return parse_ea_token(reply, "RISK")


def parse_ea_dynamic(reply: str | None) -> bool | None:
    """Read ``DYNAMIC|0/1`` from SETTINGS/PING. Missing token → None (keep Python flag)."""
    val = parse_ea_token(reply, "DYNAMIC")
    if val is None:
        return None
    tok = val.strip().lower()
    if tok in {"1", "true", "yes", "on"}:
        return True
    if tok in {"0", "false", "no", "off"}:
        return False
    return None


def parse_ea_sugar(reply: str | None) -> bool | None:
    """Read ``SUGAR|0/1`` from SETTINGS/PING. Missing token → None (keep Python flag)."""
    val = parse_ea_token(reply, "SUGAR")
    if val is None:
        return None
    tok = val.strip().lower()
    if tok in {"1", "true", "yes", "on"}:
        return True
    if tok in {"0", "false", "no", "off"}:
        return False
    return None


def parse_ea_sugar_amt(reply: str | None) -> float | None:
    """Read ``SAMT|1.00`` from SETTINGS/PING. Missing token → None (keep Python amt)."""
    val = parse_ea_token(reply, "SAMT")
    if val is None:
        return None
    try:
        float(val)
    except (TypeError, ValueError):
        return None
    return parse_sugar_amt(val)


def parse_ea_sugar_dyn(reply: str | None) -> bool | None:
    """Read ``SDYN|0/1`` from SETTINGS/PING. Missing token → None (keep Python flag)."""
    val = parse_ea_token(reply, "SDYN")
    if val is None:
        return None
    tok = val.strip().lower()
    if tok in {"1", "true", "yes", "on"}:
        return True
    if tok in {"0", "false", "no", "off"}:
        return False
    return None


def mt4_inner(broker) -> MT4Bridge | None:
    if isinstance(broker, MT4Bridge):
        return broker
    inner = getattr(broker, "inner", None)
    return inner if isinstance(inner, MT4Bridge) else None


def fetch_ea_fuse_settings(
    broker,
) -> tuple[str | None, bool | None, str | None, str | None, bool | None, float | None, bool | None]:
    """One SETTINGS/PING: fuse, DYNAMIC, inds, risk-tol, sugar, amt, sugar-dyn."""
    bridge = mt4_inner(broker)
    if bridge is None:
        return None, None, None, None, None, None, None
    for cmd in ("SETTINGS", "PING"):
        try:
            raw = bridge.settings() if cmd == "SETTINGS" else bridge.ping()
        except Exception:
            continue
        spec = parse_ea_fuse_spec(raw)
        dyn = parse_ea_dynamic(raw)
        inds = parse_ea_inds_spec(raw)
        risk = parse_ea_risk_spec(raw)
        sugar = parse_ea_sugar(raw)
        samt = parse_ea_sugar_amt(raw)
        sdyn = parse_ea_sugar_dyn(raw)
        if (
            spec is not None
            or dyn is not None
            or inds is not None
            or risk is not None
            or sugar is not None
            or samt is not None
            or sdyn is not None
        ):
            return spec, dyn, inds, risk, sugar, samt, sdyn
    return None, None, None, None, None, None, None


def fetch_ea_fuse_cats(broker) -> str | None:
    spec, _dyn, _inds, _risk, _sugar, _samt, _sdyn = fetch_ea_fuse_settings(broker)
    return spec


def _fuse_label(names: tuple[str, ...]) -> str:
    if names == tuple(CATEGORY_NAMES):
        return "all live"
    return "{" + ", ".join(names) + "}"


def _inds_label(mapping: dict[str, tuple[str, ...]] | None) -> str:
    spec = encode_fuse_inds(mapping)
    if spec == "all":
        return "all channels"
    if spec == "none":
        return "none"
    n = spec.count(",") + 1 if spec else 0
    return f"{n} channels {{{spec}}}"


def _risk_tol_label(profile: dict | None) -> str:
    prof = profile or risk_profile("balanced")
    name = str(prof.get("name") or "balanced")
    if name == "balanced":
        return "risk-tol  balanced  factory geometry  risk-factor exits off"
    bits = [
        f"risk-tol  {name}  overlay SL {float(prof['sl_atr']):.2f} / "
        f"TP {float(prof['tp_atr']):.1f} ATR"
    ]
    banc_bar = float(prof.get("banc_exit") or 0.0)
    if banc_bar > 0:
        bits.append(f"BANC exit <{banc_bar:.2f}")
    if bool(prof.get("fade_exit")):
        bits.append("fade flatten")
    uncert_bar = float(prof.get("uncert_exit") or 0.0)
    if uncert_bar > 0:
        bits.append(f"uncertainty ≥{uncert_bar:.2f}")
    blow = float(prof.get("atr_blow_exit") or 0.0)
    if blow > 0:
        bits.append(f"vol spike ATR×{blow:.2f}")
    if bool(prof.get("kill_flatten")):
        bits.append("kill flatten")
    return "  ".join(bits)


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

    def settings(self) -> str:
        if self.inner:
            return self.inner.settings()
        return "UNKNOWN"


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
    try:
        raw = None
        try:
            raw = bridge.settings()
        except Exception:
            raw = bridge.ping()
        spec = parse_ea_fuse_spec(raw)
        dyn = parse_ea_dynamic(raw)
        inds = parse_ea_inds_spec(raw)
        risk = parse_ea_risk_spec(raw)
        sugar_flag = parse_ea_sugar(raw)
        sugar_amt = parse_ea_sugar_amt(raw)
        sugar_dyn = parse_ea_sugar_dyn(raw)
        if spec is None:
            try:
                ping = bridge.ping()
            except Exception:
                ping = ""
            spec = parse_ea_fuse_spec(ping)
            if dyn is None:
                dyn = parse_ea_dynamic(ping)
            if inds is None:
                inds = parse_ea_inds_spec(ping)
            if risk is None:
                risk = parse_ea_risk_spec(ping)
            if sugar_flag is None:
                sugar_flag = parse_ea_sugar(ping)
            if sugar_amt is None:
                sugar_amt = parse_ea_sugar_amt(ping)
            if sugar_dyn is None:
                sugar_dyn = parse_ea_sugar_dyn(ping)
        if spec:
            names = parse_fuse_cats(spec)
            print(f"fuse-cats  MT4 Inputs  {_fuse_label(names)}")
        if dyn is not None:
            print(f"fuse-dynamic  MT4 Inputs  {'ON' if dyn else 'OFF'}")
        if inds:
            try:
                print(f"fuse-inds  MT4 Inputs  {_inds_label(parse_fuse_inds(inds))}")
            except ValueError as exc:
                print(f"fuse-inds  MT4 {exc}")
        if risk:
            try:
                print(_risk_tol_label(risk_profile(parse_risk_tol(risk))) + "  (MT4 Inputs)")
            except ValueError as exc:
                print(f"risk-tol  MT4 {exc}")
        if sugar_flag is not None or sugar_amt is not None or sugar_dyn is not None:
            on = "ON" if (True if sugar_flag is None else bool(sugar_flag)) else "OFF"
            amt = f"  amt {float(sugar_amt):.2f}" if sugar_amt is not None else ""
            dyns = f"  dyn {'ON' if sugar_dyn else 'OFF'}" if sugar_dyn is not None else ""
            print(f"sugar  MT4 Inputs  {on}{amt}{dyns}")
    except Exception:
        pass


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


def utc_day(ts: int) -> str:
    if not ts:
        return ""
    return time.strftime("%Y-%m-%d", time.gmtime(ts))


def in_session(ts: int, symbol: str = "") -> bool:
    """London + NY for every pair; Tokyo open for JPY/AUD/NZD. No weekend."""
    g = time.gmtime(ts)
    if g.tm_wday >= 5:
        return False
    h = g.tm_hour
    if 7 <= h < 16:
        return True
    sym = (symbol or "").upper()
    if any(tag in sym for tag in ("JPY", "AUD", "NZD")) and 0 <= h < 4:
        return True
    return False


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
        buy_cont_lo = 100.0 - float(p.rsi_cont_hi)
        buy_cont_hi = 100.0 - float(p.rsi_cont_lo)
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
            elif (
                action == "HOLD"
                and impulse >= p.cont_impulse
                and buy_cont_lo <= rsi <= buy_cont_hi
                and bounced_up
            ):
                action = "BUY"
                kind = "cont"
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
    """Reject only when round-trip cost eats the stop — ATR units, any pair."""
    mid = (bid + ask) / 2.0
    spread = max(ask - bid, 0.0)
    lots = max(float(lots), 0.01)
    cost_usd = abs(account.to_usd(spread + 2.0 * slip, lots, mid)) + account.commission_rt_per_lot * lots
    stop_usd = abs(account.to_usd(sl_atr * atr, lots, mid))
    tp_usd = abs(account.to_usd(tp_atr * atr, lots, mid))
    if stop_usd <= 1e-9:
        return False
    if cost_usd > 0.45 * stop_usd:
        return False
    if tp_usd < 1.8 * cost_usd:
        return False
    return True


def round_lot(lots: float, step: float = LOT_STEP) -> float:
    if lots <= 0 or step <= 0:
        return 0.0
    return math.floor(lots / step + 1e-12) * step


def recovery_scale(
    consec_losses: int,
    equity: float,
    peak_eq: float,
    banc_mult: float,
    enabled: bool,
    tag: str = "",
    last_loss_r: float = 0.0,
    admit_mult: float = 1.0,
) -> tuple[float, str]:
    """Capped recovery martingale, fly-trader rails on top.

    Press winners with snowball. After a *scratch* loss (< 0.45R) on a 2oo3,
    step ×1.20 then ×1.28. Full ATR stops, weak bounces, pair-admit, BANC
    threat, or 2.5% drawdown: no double. Never a third step.
    """
    if not enabled:
        return 1.0, ""
    dd = (max(peak_eq, equity) - equity) / max(peak_eq, 1.0)
    if dd > 0.020 or float(banc_mult) < 0.90:
        return 0.85, f"recovery freeze DD={100.0 * dd:.1f}% BANC×{float(banc_mult):.2f}"
    if float(admit_mult) < 0.99:
        return 1.0, "no martingale until pair admits"
    if consec_losses <= 0:
        return 1.0, ""
    if not (
        str(tag).startswith("2oo3")
        or str(tag).startswith("3oo3")
        or ("oo9" in str(tag) and str(tag)[:1] in "6789")
    ):
        return 1.0, "no martingale on bounce"
    if float(last_loss_r) >= 0.45:
        return 1.0, "no martingale after a full stop"
    if consec_losses == 1:
        return 1.20, "martingale ×1.20 (scratch)"
    return 1.28, "martingale ×1.28 (2 scratches, cap)"


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
    banc_mult: float = 1.0,
    recover_mult: float = 1.0,
    admit_mult: float = 1.0,
    sugar_mult: float = 1.0,
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
    used_risk *= max(0.50, float(banc_mult))
    used_risk *= max(0.50, float(recover_mult))
    used_risk *= max(0.45, float(admit_mult))
    used_risk *= max(0.55, float(sugar_mult))
    used_risk = float(min(used_risk, 1.80))
    risk_usd = max(equity, 0.0) * (used_risk / 100.0)
    if best_win_usd > 0 and best_win_pips >= lock_pips:
        # Only cap 1R after a *meaningful* win so a tiny first profit cannot starve later trades.
        risk_usd = min(risk_usd, 0.90 * best_win_usd)
    stop_usd = max(abs(account.to_usd(stop_px, 1.0, price)), 1e-9)
    raw = risk_usd / stop_usd
    margin_per_lot = account.required_margin(1.0, price)
    cap_margin = (max(free, 0.0) * (margin_cap_pct / 100.0) / margin_per_lot) if margin_per_lot > 0 else 0.0
    lots = round_lot(min(raw, max_lots, cap_margin))
    if lots < min_lots:
        min_risk = stop_usd * min_lots
        if min_risk > 2.0 * risk_usd:
            return 0.0, "size 0 (min lot risks more than 2x the cap)"
        lots = min_lots
    note = (
        f"{lots:.2f} lots  risk ${risk_usd:.0f} ({used_risk:.2f}% eq)  "
        f"SL {stop_pips:.1f} pips  ~${stop_usd * lots:.0f} if stopped"
    )
    if bayes_note:
        note = f"{note}  {bayes_note}"
    if abs(float(banc_mult) - 1.0) > 0.02:
        note = f"{note}  BANC×{float(banc_mult):.2f}"
    if abs(float(recover_mult) - 1.0) > 0.02:
        note = f"{note}  rec×{float(recover_mult):.2f}"
    if abs(float(admit_mult) - 1.0) > 0.02:
        note = f"{note}  admit×{float(admit_mult):.2f}"
    if abs(float(sugar_mult) - 1.0) > 0.02:
        note = f"{note}  sugar×{float(sugar_mult):.2f}"
    return lots, note


def snowball_add_lots(
    account: AccountSim,
    feat: dict,
    fly_trend: str,
    fly_fade: str,
    sl: float,
    bid: float,
    ask: float,
    slip: float,
    bars_held: int,
    adds: int,
    last_add_step: int,
    step: int,
    orig_lots: float,
    origin_entry: float,
    peak: float,
    min_hold: int,
    min_lots: float,
    max_lots: float,
    be_atr: float,
    trail_arm_atr: float,
    banc_mult: float = 1.0,
) -> tuple[float, str]:
    """Add size only if the trend fly is still with us and open profit can pay for the add."""
    pos = account.pos
    if pos is None or orig_lots <= 0:
        return 0.0, "no position"
    if adds >= SNOWBALL_MAX_ADDS:
        return 0.0, "max adds"
    if float(banc_mult) < 0.85:
        return 0.0, f"BANC risk {float(banc_mult):.2f} — no pyramid"
    if bars_held < max(int(min_hold), 5):
        return 0.0, "too early"
    if step - last_add_step < SNOWBALL_PAUSE_BARS:
        return 0.0, "pause"
    if fly_trend != pos.side:
        return 0.0, f"trend fly {fly_trend} not with {pos.side}"
    opp = "SELL" if pos.side == "BUY" else "BUY"
    if fly_fade == opp:
        return 0.0, "fade veto"
    regime = feat.get("regime", "CHOP")
    if pos.side == "BUY" and regime != "UP":
        return 0.0, "regime not UP"
    if pos.side == "SELL" and regime != "DOWN":
        return 0.0, "regime not DOWN"
    impulse = float(feat.get("impulse", 0.0))
    if pos.side == "BUY" and impulse < 0.45:
        return 0.0, "impulse faded"
    if pos.side == "SELL" and impulse > -0.45:
        return 0.0, "impulse faded"
    atr = float(feat.get("atr", 0.0))
    pip = pip_size(origin_entry or pos.entry)
    unreal_px = (bid - origin_entry) if pos.side == "BUY" else (origin_entry - ask)
    unreal_pips = unreal_px / pip
    arm = max(SNOWBALL_MIN_PIPS, SNOWBALL_ARM_ATR * atr / pip)
    if unreal_pips < arm:
        return 0.0, f"need {arm:.1f} pips, have {unreal_pips:.1f}"
    run_atr = unreal_px / max(atr, pip)
    if run_atr < be_atr:
        return 0.0, "not yet enough to cover a snowball"
    if run_atr >= trail_arm_atr:
        return 0.0, "already trailing — too late to add"
    if peak:
        if pos.side == "SELL":
            off = ask - peak
        else:
            off = peak - bid
        if off < 0.30 * atr:
            return 0.0, "still at the extension — wait for a pullback"
        if off > 1.25 * atr:
            return 0.0, "pullback too deep"
    room = max_lots - pos.lots
    want = round_lot(min(orig_lots * SNOWBALL_FRACTION * max(0.55, float(banc_mult)), room))
    if want < min_lots:
        return 0.0, "add smaller than min lot"
    fill = (ask + slip) if pos.side == "BUY" else (bid - slip)
    mid = (bid + ask) / 2.0
    add_risk = abs(account.to_usd(fill - sl, want, mid))
    floating = account.floating(bid, ask)
    if floating < 1.50 * add_risk:
        return 0.0, f"float ${floating:.0f} < 1.50× add risk ${add_risk:.0f}"
    return want, (
        f"trend={fly_trend} k={impulse:+.2f}  +{unreal_pips:.1f} pips  "
        f"float ${floating:.0f} covers add risk ${add_risk:.0f}"
    )


def _sugar_enabled(args: argparse.Namespace) -> bool:
    """Sugar feed is off unless ``--sugar`` or the GUI box is checked. ``--no-sugar`` always wins."""
    if bool(getattr(args, "no_sugar", False)):
        return False
    return bool(getattr(args, "sugar", False))


def trade_loop(brain: FlyBrain, prices, args: argparse.Namespace, broker, delay: float) -> dict:
    quiet = bool(getattr(args, "evolve_quiet", False))
    capture = bool(getattr(args, "evolve_capture", False))
    loop_sym = str(getattr(args, "symbol", SYMBOL))
    force_brain = bool(getattr(args, "force_brain", False))
    use_brain = bool(getattr(args, "use_brain", False))
    reset_brain = bool(getattr(args, "reset_brain", False))
    stored_brain = None
    if use_brain and not reset_brain and (not is_frozen(loop_sym) or force_brain):
        stored_brain = load_brain(loop_sym)
    evo_src = apply_policy(args, live=bool(delay and delay > 0), stored_brain=stored_brain)
    if evo_src and not quiet:
        print(f"evo     loaded  {evo_src}  {describe_genome(genome_from_args(args))}")
    elif bool(getattr(args, "use_evo", False)) and not quiet:
        print(
            f"evo     missing  brains/{loop_sym}.evo.json — Train DA with evolve, "
            "or uncheck use evolved"
        )
    spec = pair_spec(getattr(args, "symbol", SYMBOL))
    indicators = Indicators(pip=float(spec.get("pip", 0.0001)))
    account = AccountSim(
        balance=getattr(args, "balance", 100_000.0),
        leverage=getattr(args, "leverage", 100.0),
        spread=float(getattr(args, "pair_spread", spec.get("spread", SPREAD))),
        commission_rt_per_lot=getattr(args, "commission", 7.0),
        slippage_pips=getattr(args, "slippage_pips", 0.2),
        swap_long_per_lot=getattr(args, "swap_long", -0.72),
        swap_short_per_lot=getattr(args, "swap_short", 0.12),
        stop_out=getattr(args, "stop_out", 50.0) / 100.0,
        quote=str(spec.get("quote", "USD")),
    )
    print(
        f"sim account ${account.start_balance:,.0f}  leverage 1:{account.leverage:.0f}  "
        f"commission ${account.commission_rt_per_lot:.2f}/lot RT  "
        f"slip {account.slippage_pips:.1f} pip  spread {account.spread:.5f}  quote {account.quote}"
    )
    base_risk_pct = float(getattr(args, "risk", RISK_PCT))
    try:
        risk_tol = parse_risk_tol(getattr(args, "risk_tol", None))
    except ValueError as exc:
        print(f"risk-tol  {exc}  — using balanced")
        risk_tol = "balanced"
    profile = risk_profile(risk_tol)
    risk_pct = base_risk_pct * float(profile["risk_scale"]) if base_risk_pct > 0 else base_risk_pct
    min_lots = float(getattr(args, "min_lots", MIN_LOTS))
    max_lots = float(getattr(args, "max_lots", MAX_LOTS))
    margin_cap = float(getattr(args, "margin_cap", MARGIN_CAP_PCT))
    if risk_pct > 0:
        print(
            f"sizing  {risk_pct:.2f}% of equity per trade, scaled by Bayesian half-Kelly  "
            f"lot {min_lots:.2f}-{max_lots:.2f}  margin cap {margin_cap:.0f}% of free"
            + (
                f"  ({risk_tol} ×{float(profile['risk_scale']):.2f})"
                if float(profile["risk_scale"]) != 1.0
                else ""
            )
        )
    else:
        print(f"sizing  fixed {args.lots:.2f} lots")
    print(_risk_tol_label(profile))
    bayes = BayesianSizer()
    params = AdaptiveParams()
    seed_p = getattr(args, "seed_params", None)
    if isinstance(seed_p, dict) and seed_p:
        params.apply(seed_p)
    scorer = SetupScorer(bayes, min_conf=float(getattr(args, "min_conf", MIN_CONF)))
    min_conf = float(getattr(args, "min_conf", MIN_CONF))
    tax_rate = float(getattr(args, "tax_rate", TAX_RATE))
    tax_model = str(getattr(args, "tax_model", "ordinary"))
    state_path = Path(getattr(args, "adapt_state", "") or (ROOT / "reports" / "adapt_state.json"))
    extra: dict = {}
    board = NodeBoard()
    if getattr(args, "reset_adapt", False) and state_path.exists():
        state_path.unlink()
        print(f"adapt  reset {state_path}")
    elif load_adapt_state(state_path, bayes, scorer, params, extra=extra):
        n_hist = bayes.n_win + bayes.n_loss
        if "fusion" in extra:
            board.load(extra["fusion"])
        print(
            f"adapt  loaded {state_path}  n={n_hist}  bar {scorer.threshold:.0f}%  "
            f"SL {params.sl_atr:.2f} ATR  TP {params.tp_atr:.1f} ATR  "
            f"impulse>={params.min_impulse:.2f}  "
            f"fusion tech p={board.tech.p_win:.2f} trend p={board.trend.p_win:.2f} fade p={board.fade.p_win:.2f}"
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
    plastic_on = not bool(getattr(args, "no_plastic", False))
    banc_on = not bool(getattr(args, "no_banc", False))
    sugar_on = _sugar_enabled(args)
    sugar_amt = parse_sugar_amt(getattr(args, "sugar_amt", 1.0))
    sugar_dyn = bool(getattr(args, "sugar_dyn", False)) and sugar_on
    sugar_dose_m = SugarDoseMachine() if sugar_dyn else None
    swarm = FlySwarm(brain, plastic=plastic_on)
    banc = BancRiskBrain() if banc_on else None
    if not force_brain:
        seed_frozen_eurusd()
    loaded_brain = False
    pair_book_on = pair_book_enabled(
        loop_sym,
        frozen=is_frozen(loop_sym),
        no_pair_book=bool(getattr(args, "no_pair_book", False)),
        pair_book=bool(getattr(args, "pair_book", False)),
    )
    pair_book = PairBook(
        loop_sym,
        str(getattr(args, "bar_source", "") or ""),
        enabled=pair_book_on,
    )
    stored = stored_brain
    if use_brain and not reset_brain and (not is_frozen(loop_sym) or force_brain):
        stored = stored or load_brain(loop_sym)
        if stored and stored.get("params"):
            params.load(stored["params"])
            if not quiet:
                print(
                    f"brain   params  SL {params.sl_atr:.2f} ATR  "
                    f"RSI arm {params.rsi_buy_arm:.0f}/{params.rsi_sell_arm:.0f}  "
                    f"impulse>={params.min_impulse:.2f}"
                )
        if stored and plastic_on and stored.get("plastic"):
            swarm.load_plastic(stored["plastic"])
            loaded_brain = True
            n_da = 0
            try:
                n_da = int((stored["plastic"].get("trend") or {}).get("n_updates", 0))
            except Exception:
                n_da = 0
            if not quiet:
                print(f"brain   loaded brains/{loop_sym.upper()}.json  DA n={n_da}")
        if stored and banc_on and banc is not None and stored.get("banc"):
            banc.load(stored["banc"])
    if (not loaded_brain) and plastic_on and getattr(args, "seed_plastic", None):
        swarm.load_plastic(args.seed_plastic)
        loaded_brain = True
    if (not loaded_brain) and plastic_on and extra.get("plastic"):
        swarm.load_plastic(extra["plastic"])
        if not quiet:
            print("plastic  loaded PAM/PPL three-factor traces (W frozen)")
    if banc_on and extra.get("banc") and banc is not None and not loaded_brain:
        banc.load(extra["banc"])
    if banc is not None and getattr(args, "seed_banc", None):
        banc.load(args.seed_banc)
    fusion_on = not bool(getattr(args, "no_fusion", False))
    nest_on = bool(getattr(args, "nest", False)) and not bool(getattr(args, "no_nest", False))
    koo9_on = bool(getattr(args, "koo9", False)) and not nest_on
    ens_on = bool(getattr(args, "ensemble", False)) and not nest_on and not koo9_on
    legacy_2oo3 = bool(getattr(args, "legacy_2oo3", False)) or bool(getattr(args, "no_categories", False))
    cat_on = (
        bool(getattr(args, "categories", False))
        and (not nest_on)
        and (not koo9_on)
        and (not ens_on)
        and (not legacy_2oo3)
    )
    indicators.mixer = "legacy" if legacy_2oo3 else "cat_fuse"
    try:
        fuse_names = parse_fuse_cats(getattr(args, "fuse_cats", None))
    except ValueError as exc:
        print(f"fuse-cats  {exc}  — using all")
        fuse_names = tuple(CATEGORY_NAMES)
    try:
        fuse_inds = parse_fuse_inds(getattr(args, "fuse_inds", None))
    except ValueError as exc:
        print(f"fuse-inds  {exc}  — using all channels")
        fuse_inds = parse_fuse_inds("")
    fuse_dynamic = (
        bool(getattr(args, "fuse_dynamic", False))
        and not legacy_2oo3
        and not cat_on
        and not nest_on
        and not koo9_on
        and not ens_on
    )
    if delay and delay > 0:
        ea_spec, ea_dyn, ea_inds, ea_risk, ea_sugar, ea_sugar_amt, ea_sugar_dyn = fetch_ea_fuse_settings(broker)
        if ea_spec is not None:
            try:
                fuse_names = parse_fuse_cats(ea_spec)
                print(f"fuse-cats  from MT4 Inputs  {_fuse_label(fuse_names)}")
            except ValueError as exc:
                print(f"fuse-cats  MT4 {exc}  — keeping {_fuse_label(fuse_names)}")
        if ea_dyn is not None:
            fuse_dynamic = bool(ea_dyn) and not legacy_2oo3 and not cat_on and not nest_on and not koo9_on and not ens_on
            print(f"fuse-dynamic  from MT4 Inputs  {'ON' if fuse_dynamic else 'OFF'}")
        if ea_inds is not None:
            try:
                fuse_inds = parse_fuse_inds(ea_inds)
                print(f"fuse-inds  from MT4 Inputs  {_inds_label(fuse_inds)}")
            except ValueError as exc:
                print(f"fuse-inds  MT4 {exc}  — keeping {_inds_label(fuse_inds)}")
        if ea_risk is not None:
            try:
                nxt_tol = parse_risk_tol(ea_risk)
                if nxt_tol != risk_tol:
                    risk_tol = nxt_tol
                    profile = risk_profile(risk_tol)
                    risk_pct = (
                        base_risk_pct * float(profile["risk_scale"])
                        if base_risk_pct > 0
                        else base_risk_pct
                    )
                print(_risk_tol_label(profile) + "  (from MT4 Inputs)")
            except ValueError as exc:
                print(f"risk-tol  MT4 {exc}  — keeping {risk_tol}")
        if ea_sugar is not None:
            sugar_on = bool(ea_sugar)
        if ea_sugar_amt is not None:
            sugar_amt = float(ea_sugar_amt)
        if ea_sugar_dyn is not None:
            sugar_dyn = bool(ea_sugar_dyn) and sugar_on
            if sugar_dyn and sugar_dose_m is None:
                sugar_dose_m = SugarDoseMachine()
            if not sugar_dyn:
                sugar_dose_m = None
        if ea_sugar is not None or ea_sugar_amt is not None or ea_sugar_dyn is not None:
            print(
                f"sugar   from MT4 Inputs  {'ON' if sugar_on else 'OFF'}  amt {sugar_amt:.2f}"
                f"  dyn {'ON' if sugar_dyn else 'OFF'}"
            )
    indicators.fuse_allow = fuse_names
    indicators.fuse_dynamic = fuse_dynamic
    indicators.fuse_inds = fuse_inds
    indicators.cats.include = fuse_names
    indicators.cats.channel_include = fuse_inds
    if fuse_dynamic:
        indicators.fuse_machine = FuseRegimeMachine()
    if encode_fuse_inds(fuse_inds) != "all":
        print(f"fuse-inds  {_inds_label(fuse_inds)}")
    cat_inhibit = float(getattr(args, "cat_inhibit", CONF_INHIBIT_NORM) or CONF_INHIBIT_NORM)
    nine = FlyNine(brain, plastic=plastic_on) if koo9_on else None
    cat_pool = CategoryFlyPool(brain, plastic=plastic_on) if cat_on else None
    if koo9_on and nine is not None and plastic_on:
        nine_blob = (stored or {}).get("nine") if stored else None
        if not nine_blob:
            nine_blob = extra.get("nine")
        if nine_blob:
            nine.load_plastic(nine_blob)
    if cat_on and cat_pool is not None and plastic_on:
        cat_blob = (stored or {}).get("cat_flies") if stored else None
        if not cat_blob:
            cat_blob = extra.get("cat_flies")
        if cat_blob:
            cat_pool.load_plastic(cat_blob)
            loaded_brain = True
            src = (
                f"brains/{loop_sym.upper()}.json"
                if stored and stored.get("cat_flies")
                else "adapt_state"
            )
            print(f"brain   cat_flies  {src}  DA n={cat_pool.da_updates()}")
    ens_book = StrategyBook() if ens_on else None
    if ens_on and extra.get("ensemble"):
        ens_book.load(extra["ensemble"])
    snowball_on = not bool(getattr(args, "no_snowball", False))
    martingale_on = not bool(getattr(args, "no_martingale", False))
    rails = BookRails(
        kill_dd=float(profile["kill_dd"]),
        protect_dd=float(profile["protect_dd"]),
    )
    if extra.get("rails"):
        rails.load(extra["rails"])
    dash: DashHub | None = getattr(args, "dash", None)
    print(
        "committee  "
        + (
            "9-fly k-out-of-9  {trend, fade, boll, vol, rsi, macd, flow, mix, struct}  "
            "5oo9–9oo9 scales lots   "
            if koo9_on
            else (
                "9-node nest  3×3oo3 {structure, momentum, fade} then 2oo3 of families   "
                if nest_on
                else (
                    "Bayesian ensemble  5 books {classic 2oo3, structure 2oo3, momentum 2oo3, fade 2oo3, breakout 3oo3}   "
                    if ens_on
                    else (
                        "legacy 2oo3  {tech, fly-trend, fly-fade}   "
                        if legacy_2oo3
                        else (
                            "7-category  Kalman + tech + 3 flies  2oo3/3oo3 then confidence pick   "
                            if cat_on
                            else (
                                "cat-fuse  "
                                + (
                                    "DYNAMIC TREND/RANGE/BREAK/QUIET  allow "
                                    if fuse_dynamic
                                    else ""
                                )
                                + _fuse_label(fuse_names)
                                + " inverse-var mix → {tech, fly-trend, fly-fade} 2oo3   "
                            )
                        )
                    )
                )
            )
        )
        + "both BUY and SELL   parallel: flow-algo   "
        + (
            "fusion: Kalman+Bayes gate + fly-conf (native votes, abstain only)"
            if fusion_on and not koo9_on and not cat_on
            else (
                f"category inhibit |conf|/τ < {cat_inhibit:.2f}"
                if cat_on
                else ("koo9 encodings are the diversity" if koo9_on else "fusion: OFF (--no-fusion, snapshot 2oo3 path)")
            )
        )
    )
    if pair_book_on:
        src = "yahoo ×0.50 until 2 wins" if pair_book.yahoo else "HST full size"
        print(
            f"pair-book  on  {loop_sym}  same 2oo3  wide 3oo3 or 2oo3+flow  {src}  "
            f"source {pair_book.source or 'n/a'}"
        )
    elif is_frozen(loop_sym):
        print(f"pair-book  off  {loop_sym} frozen factory 2oo3")
    elif bool(getattr(args, "no_pair_book", False)):
        print("pair-book  OFF (--no-pair-book)")
    else:
        print("pair-book  OFF (opt-in --pair-book)")
    print(
        "snowball  "
        + (
            f"on  add up to {SNOWBALL_MAX_ADDS}×{SNOWBALL_FRACTION:.0%} of first lots  "
            f"when fly-trend agrees, profit is past BE but not yet trailing, pullback from the extreme"
            if snowball_on
            else "OFF (--no-snowball)"
        )
    )
    print(
        "martingale  "
        + (
            "on  ×1.20 after a scratch, ×1.28 cap; never after a full stop; freeze if DD/BANC/pair-admit"
            if martingale_on
            else "OFF (--no-martingale)"
        )
    )
    print(
        "plastic  "
        + (
            "on  PAM/PPL dopamine × eligibility on readout + sensory gains (connectome W frozen)"
            if plastic_on
            else "OFF (--no-plastic)"
        )
    )
    if banc is not None:
        print(
            f"BANC    female cluster net n={banc.n}  {banc.last.get('source')}  "
            f"risk voter mixed with MaleCNS ol_sensory+vnc_sensory  "
            f"(full 188k edgelist needs CAVE_TOKEN)"
        )
    else:
        print("BANC    OFF (--no-banc)")
    if sugar_on:
        sugar_line = (
            f"on  amt {sugar_amt:.2f}  dyn {'FEAST/FORAGE/NIBBLE/FAST' if sugar_dyn else 'OFF'}  "
            f"{int(swarm.core.sugar.size)} GRNs  food=2oo3/clean  "
            "satiety=equity/float/day  appetite sizes / skips / snowball / spit-out winners  "
            "(never votes a side)"
        )
    else:
        sugar_line = (
            f"native GRN lamp  {int(swarm.core.sugar.size)} cells  0.45×nice on the risk pass  "
            "(4k-era: eat every 2oo3, no skip/flatten; check sugar feed for the overlay)"
        )
    print("sugar   " + sugar_line)
    print(
        "rails   fly-trader kill "
        f"{100.0 * rails.kill_dd:.1f}% DD  ({100.0 * rails.protect_dd:.1f}% after snowball)  "
        f"circuit {rails.circuit_n} losses  "
        f"pair-admit {rails.admit_n} trades / PF≥{rails.admit_pf:.2f} then full size"
    )
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
    consec_losses = 0
    peak_eq = float(account.start_balance)
    day_start_eq = float(account.start_balance)
    last_recover = ""
    best_win_usd = 0.0
    best_win_pips = 0.0
    open_score: dict | None = None
    open_geom: dict | None = None
    snowball_adds = 0
    last_add_step = -99
    orig_lots = 0.0
    origin_entry = 0.0
    last_heads: dict = {}
    last_ballots: dict | None = None
    last_feat_key = None
    conf_head: dict = pack_head(
        brain, brain.x, "HOLD", 0.0, 0, brain.tau * 0.70, "conf", swarm.conf_p
    )
    snow_note = ""
    last_banc: dict = {"mult": 1.0, "threat": 0.0, "walk": 0.0, "vote": "OK", "malecns_mult": 1.0}
    last_sugar: dict = dict(SUGAR_IDLE)
    last_sugar_feed = 0.50
    last_m_risk: dict = {"mult": 1.0, "sweet": 0.0}
    last_nest: dict = {"families": {}, "nodes": {}}
    last_koo9: dict = {"votes": {}, "buy": 0, "sell": 0, "k": 0, "mult": 0.0}
    last_ens: dict = {"books": {}, "on_side": [], "ens_mult": 0.0}
    last_cat: dict = {"winner": "", "categories": {}, "inhibited": [], "losers": []}
    last_cat_tech: dict = {}
    last_cat_flies: dict = {}
    banc_mults: list[float] = []
    last_shadow_step = -99

    def persist_state(*, force: bool = False) -> None:
        if delay == 0 and not force:
            return
        blob: dict = {"fusion": board.dump()}
        if plastic_on:
            blob["plastic"] = swarm.dump_plastic()
        if banc is not None:
            blob["banc"] = banc.dump()
        blob["rails"] = rails.dump()
        if nine is not None:
            blob["nine"] = nine.dump_plastic()
        if cat_pool is not None:
            blob["cat_flies"] = cat_pool.dump_plastic()
        if ens_book is not None:
            blob["ensemble"] = ens_book.dump()
        save_adapt_state(state_path, bayes, scorer, params, extra=blob)
        sym = str(getattr(args, "symbol", SYMBOL))
        extra_brain = {
            "cat_flies": blob.get("cat_flies") or {},
            "nine": blob.get("nine") or {},
        }
        gene = genome_from_args(args)
        gene["params"] = params.dump()
        extra_brain["genome"] = public_genome(gene)
        meta = {
            "n_trades": len(trades),
            "da_n": int(swarm.trend_p.n_updates) if swarm.trend_p is not None else 0,
            "cat_da_n": cat_pool.da_updates() if cat_pool is not None else 0,
        }
        if delay > 0:
            save_brain(
                sym,
                swarm.dump_plastic() if plastic_on else {},
                banc.dump() if banc is not None else None,
                meta,
                live=True,
                params=params.dump(),
                extra=extra_brain,
            )
        if bool(getattr(args, "save_brain", False)):
            save_brain(
                sym,
                swarm.dump_plastic() if plastic_on else {},
                banc.dump() if banc is not None else None,
                meta,
                live=False,
                force=bool(getattr(args, "force_brain", False)),
                params=params.dump(),
                extra=extra_brain,
            )

    def maybe_shadow(decision: str, tag: str, stop_pips: float, pip: float) -> None:
        nonlocal last_shadow_step
        if not bool(getattr(args, "shadow_da", False)):
            return
        if decision not in ("BUY", "SELL"):
            return
        if step - last_shadow_step < 12:
            return
        bars = getattr(args, "replay_bars", None) or []
        r = forward_r(bars, step, decision, stop_pips, pip)
        if r == 0.0:
            return
        last_shadow_step = step
        notes = swarm.apply_reward_r(r, decision)
        if nine is not None:
            notes.extend(nine.apply_reward_r(r, decision))
        if cat_pool is not None:
            notes.extend(cat_pool.apply_reward_r(r, decision, winner=str((last_cat or {}).get("winner") or "")))
        if banc is not None:
            notes.append("banc " + banc.apply_reward_r(r, decision))
        print(
            f"[{label}] DA-shadow {decision}  R={r:+.2f}  {tag}  "
            + " | ".join(notes[:2])
        )

    def close_now(reason: str) -> None:
        nonlocal ticket, open_side, open_time, bars_held, sl, tp, peak, day_losses, best_win_usd, best_win_pips, open_score, open_geom, snowball_adds, last_add_step, orig_lots, origin_entry, snow_note, consec_losses, peak_eq
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
            "snowball_adds": snowball_adds,
        }
        trades.append(row)
        mark = "WIN" if row["usd"] > 0 else "LOSS"
        print(
            f"[{row['close_time']}] CLOSE {open_side} {mark}  {row['lots']:.2f} lots  "
            f"conf {row['conf']:.0f}%  bar {row['threshold']:.0f}%  "
            f"{row['pips']:+.1f} pips  ${row['usd']:+.2f}  "
            f"comm ${row['commission']:.2f} slip ${row['slippage']:.2f} "
            f"swap ${row['swap']:.2f}  snowball {snowball_adds}  ({reason})"
        )
        if ticket is not None:
            print(f"       → {_explain(broker.close(ticket))}")
        brain.apply_reward(res.net)
        da_notes = swarm.apply_reward(res.net, str(open_side or ""))
        if nine is not None:
            da_notes.extend(nine.apply_reward(res.net, str(open_side or "")))
        if cat_pool is not None:
            da_notes.extend(
                cat_pool.apply_reward(
                    res.net, str(open_side or ""), winner=str((open_score or {}).get("cat_winner") or "")
                )
            )
        if banc is not None:
            banc_note = banc.apply_reward(res.net, str(open_side or ""))
            da_notes.append(f"banc {banc_note}")
        if da_notes:
            print("       DA  " + " | ".join(da_notes[:3]))
        if row["usd"] <= 0:
            day_losses += 1
            consec_losses += 1
        else:
            consec_losses = 0
        peak_eq = max(peak_eq, float(snap["equity"]))
        if row["usd"] > best_win_usd:
            best_win_usd = row["usd"]
        if row["usd"] > 0 and row["pips"] > best_win_pips:
            best_win_pips = row["pips"]
        stop_p = float((open_score or {}).get("stop_pips", 8.0) or 8.0)
        r_mult = abs(float(row["pips"])) / max(stop_p, 0.1)
        rails.mark(float(snap["equity"]))
        rail_note = rails.on_close(
            float(row["usd"]),
            r_mult,
            step,
            snowball_n=int(snowball_adds),
            day=utc_day(stamp),
        )
        if rail_note:
            print(f"       rails  {rail_note}")
        pb_note = pair_book.on_close(reason, float(row["usd"]), open_side, step)
        if pb_note:
            print(f"       pair-book  {pb_note}")
        if plastic_on and rails.should_rollback_plastic():
            swarm.rollback_plastic()
            if nine is not None:
                nine.rollback_plastic()
            if cat_pool is not None:
                cat_pool.rollback_plastic()
            print("       plastic  shadow rollback (last 4 trades PF<0.80, W still frozen)")
        scorer.remember(row)
        param_note = params.update(row) if plastic_on else "params frozen (--no-plastic)"
        board.remember_votes(
            {
                "tech": str((open_score or {}).get("cast_tech", "HOLD")),
                "trend": str((open_score or {}).get("cast_trend", "HOLD")),
                "fade": str((open_score or {}).get("cast_fade", "HOLD")),
            },
            row["usd"] > 0,
        )
        if ens_book is not None:
            ens_book.remember(list((open_score or {}).get("ens_on_side") or []), row["usd"] > 0)
        persist_state()
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
        if dash is not None:
            dash.publish(
                {},
                log_line={
                    "t": row["close_time"],
                    "side": row["side"],
                    "what": f"CLOSE {mark}",
                    "detail": f"{row['pips']:+.1f} pips  ${row['usd']:+.2f}  {reason}  snow {snowball_adds}",
                },
            )
        snowball_adds = 0
        last_add_step = -99
        orig_lots = 0.0
        origin_entry = 0.0
        snow_note = ""

    try:
        for bar in prices:
            if dash is not None and dash.cancel.is_set():
                if ticket is not None:
                    close_now("gui stop")
                print("dashboard  stop")
                break
            if delay > 0 and step % 16 == 0:
                ea_spec, ea_dyn, ea_inds, ea_risk, ea_sugar, ea_sugar_amt, ea_sugar_dyn = fetch_ea_fuse_settings(broker)
                changed = False
                if ea_spec is not None:
                    try:
                        nxt = parse_fuse_cats(ea_spec)
                    except ValueError:
                        nxt = fuse_names
                    if nxt != fuse_names:
                        fuse_names = nxt
                        changed = True
                if ea_dyn is not None:
                    nxt_dyn = (
                        bool(ea_dyn)
                        and not legacy_2oo3
                        and not cat_on
                        and not nest_on
                        and not koo9_on
                        and not ens_on
                    )
                    if nxt_dyn != fuse_dynamic:
                        fuse_dynamic = nxt_dyn
                        if fuse_dynamic:
                            indicators.fuse_machine = FuseRegimeMachine()
                        changed = True
                if ea_inds is not None:
                    try:
                        nxt_inds = parse_fuse_inds(ea_inds)
                    except ValueError:
                        nxt_inds = fuse_inds
                    if nxt_inds != fuse_inds:
                        fuse_inds = nxt_inds
                        changed = True
                if ea_risk is not None:
                    try:
                        nxt_tol = parse_risk_tol(ea_risk)
                    except ValueError:
                        nxt_tol = risk_tol
                    if nxt_tol != risk_tol:
                        risk_tol = nxt_tol
                        profile = risk_profile(risk_tol)
                        risk_pct = (
                            base_risk_pct * float(profile["risk_scale"])
                            if base_risk_pct > 0
                            else base_risk_pct
                        )
                        rails.kill_dd = float(profile["kill_dd"])
                        rails.protect_dd = float(profile["protect_dd"])
                        changed = True
                        print(_risk_tol_label(profile) + "  (from MT4 Inputs)")
                amt_changed = False
                if ea_sugar_amt is not None and abs(float(ea_sugar_amt) - sugar_amt) > 1e-6:
                    sugar_amt = float(ea_sugar_amt)
                    amt_changed = True
                    changed = True
                if ea_sugar is not None and bool(ea_sugar) != sugar_on:
                    sugar_on = bool(ea_sugar)
                    changed = True
                    print(
                        f"sugar   from MT4 Inputs  {'ON' if sugar_on else 'OFF'}  amt {sugar_amt:.2f}"
                        f"  dyn {'ON' if sugar_dyn else 'OFF'}"
                    )
                elif amt_changed:
                    print(f"sugar   amt {sugar_amt:.2f}  (from MT4 Inputs)")
                if ea_sugar_dyn is not None:
                    nxt_sdyn = bool(ea_sugar_dyn) and sugar_on
                    if nxt_sdyn != sugar_dyn:
                        sugar_dyn = nxt_sdyn
                        sugar_dose_m = SugarDoseMachine() if sugar_dyn else None
                        changed = True
                        print(f"sugar   dyn {'ON' if sugar_dyn else 'OFF'}  (from MT4 Inputs)")
                if changed:
                    indicators.fuse_allow = fuse_names
                    indicators.fuse_dynamic = fuse_dynamic
                    indicators.fuse_inds = fuse_inds
                    indicators.cats.channel_include = fuse_inds
                    if not fuse_dynamic:
                        indicators.cats.include = fuse_names
                    print(
                        f"fuse-cats  MT4 Inputs  {_fuse_label(fuse_names)}"
                        f"  dynamic {'ON' if fuse_dynamic else 'OFF'}"
                        f"  inds {_inds_label(fuse_inds)}"
                        f"  risk {risk_tol}"
                        f"  sugar {'ON' if sugar_on else 'OFF'}×{sugar_amt:.2f}"
                        f"{' dyn' if sugar_dyn else ''}"
                    )
            high, low, close = bar["high"], bar["low"], bar["close"]
            stamp = bar["time"]
            label = time.strftime("%Y-%m-%d %H:%M", time.gmtime(stamp)) if stamp else str(step)
            feat = indicators.update(
                high,
                low,
                close,
                volume=bar.get("volume"),
                open_=bar.get("open"),
                extras=bar if isinstance(bar, dict) else None,
                stamp=stamp,
            )
            bid, ask, slip = account.quotes(close, stamp, high, low, feat["atr"])
            stop = account.mark(bid, ask, stamp)
            key = time.gmtime(stamp).tm_yday if stamp else step // 288
            if key != day_key:
                day_key = key
                day_losses = 0
                day_trades = 0
                day_start_eq = float(account.snapshot(bid, ask)["equity"])
            need_fly = feat["ready"] and in_session(stamp, str(getattr(args, "symbol", ""))) and (
                ticket is not None
                or feat["regime"] != "CHOP"
                or cat_on
            )
            raw_tech, kind = setup.decide(feat, close, step, params)
            fly_trend = fly_fade = flow = "HOLD"
            t_score = f_score = 0.0
            conf_fly = "HOLD"
            conf_score = 0.0
            mom_f = float(((feat.get("categories") or {}).get("momentum") or {}).get("fused") or 0.0)
            trend_f = float(((feat.get("categories") or {}).get("trend") or {}).get("fused") or 0.0)
            if delay == 0:
                feat_key = (
                    feat.get("regime"),
                    round(float(feat.get("impulse", 0.0)), 1),
                    int(feat.get("rsi", 50) // 5),
                    round(mom_f, 1),
                    round(trend_f, 1),
                )
            else:
                feat_key = (
                    feat.get("regime"),
                    round(float(feat.get("impulse", 0.0)), 2),
                    int(feat.get("rsi", 50) // 3),
                    round(mom_f, 2),
                )
            tech_hit = False
            if cat_on and need_fly:
                last_cat_tech = run_category_tech(feat)
                tech_hit = any(
                    str((last_cat_tech.get(n) or {}).get("vote") or "") in ("BUY", "SELL")
                    for n in last_cat_tech
                )
            run_fly = need_fly
            if delay == 0:
                run_fly = need_fly and (
                    last_ballots is None
                    or raw_tech in ("BUY", "SELL")
                    or tech_hit
                    or last_feat_key != feat_key
                    or (ticket is not None and step % 8 == 0)
                )
            if koo9_on and nine is not None:
                if run_fly:
                    last_koo9 = nine.vote(feat, SUBSTEPS_NINE)
                    last_feat_key = feat_key
                    last_ballots = {
                        "trend": last_koo9["votes"].get("trend", "HOLD"),
                        "fade": last_koo9["votes"].get("fade", "HOLD"),
                        "nfire": int(last_koo9.get("nfire") or 0),
                    }
                nv = last_koo9.get("votes") or {}
                fly_trend = str(nv.get("trend") or "HOLD")
                fly_fade = str(nv.get("fade") or "HOLD")
                t_score = float((last_koo9.get("scores") or {}).get("trend") or 0.0)
                f_score = float((last_koo9.get("scores") or {}).get("fade") or 0.0)
                flow = str(nv.get("flow") or flow_algo(feat))
                ballots = last_ballots or {}
            elif cat_on and cat_pool is not None:
                if run_fly:
                    last_cat_flies = cat_pool.vote(
                        feat,
                        feat.get("categories") or {},
                        last_cat_tech,
                        SUBSTEPS_CAT,
                    )
                    nfire_c = sum(int((last_cat_flies.get(n) or {}).get("nfire") or 0) for n in last_cat_flies)
                    last_ballots = {
                        "trend": (last_cat_flies.get("trend") or {}).get("trend", "HOLD"),
                        "fade": (last_cat_flies.get("trend") or {}).get("fade", "HOLD"),
                        "nfire": nfire_c,
                    }
                    last_feat_key = feat_key
                trb = last_cat_flies.get("trend") or {}
                fly_trend = str(trb.get("trend") or "HOLD")
                fly_fade = str(trb.get("fade") or "HOLD")
                t_score = float(trb.get("trend_score") or 0.0)
                f_score = float(trb.get("fade_score") or 0.0)
                flow = flow_algo(feat)
                ballots = last_ballots or {}
            elif run_fly:
                ballots = swarm.vote(feat, args.substeps)
                last_ballots = ballots
                last_feat_key = feat_key
            else:
                ballots = last_ballots or {}
            if need_fly and not koo9_on and not cat_on:
                fly_trend, fly_fade = ballots.get("trend", "HOLD"), ballots.get("fade", "HOLD")
                t_score, f_score = ballots.get("trend_score", 0.0), ballots.get("fade_score", 0.0)
                flow = flow_algo(feat)
                last_heads.update(ballots.get("heads") or {})
            tau = max(brain.tau, 1e-6)
            tech_obs = board.tech.observe(tech_channels(feat, raw_tech, kind))
            trend_obs = board.trend.observe(
                trend_channels(feat, fly_trend, t_score, tau, flow)
            )
            fade_obs = board.fade.observe(fade_channels(feat, fly_fade, f_score, tau))
            post = bayes.last or {}
            hist = scorer.history
            conf_proxy = float(hist[-1]["conf"]) / 100.0 if hist else float(bayes.p_win)
            pack = {
                "p_edge": float(post.get("p_edge", 0.50)),
                "conf_proxy": conf_proxy,
                "p_trend": float(trend_obs["p_cast"]),
                "p_fade": float(fade_obs["p_cast"]),
                "tech_fused": float(tech_obs["fused"]),
                "trend_fused": float(trend_obs["fused"]),
            }
            if fusion_on and need_fly and run_fly and not koo9_on and not cat_on:
                conf_fly, conf_score, conf_head = swarm.vote_conf(feat, pack, args.substeps)
                last_heads["conf"] = conf_head
            if need_fly and banc is not None and run_fly:
                snap_r = account.snapshot(bid, ask)
                rp = risk_pack(
                    feat,
                    snap_r,
                    account.start_balance,
                    day_losses,
                    float(pack.get("p_edge", 0.50)),
                    account.spread,
                    pip_size(close),
                )
                # MaleCNS risk is another full 166k pass; reuse it on full-speed bars.
                do_malecns = (
                    delay > 0
                    or not banc_mults
                    or (ticket is not None and step % 8 == 0)
                    or step % 16 == 0
                )
                if do_malecns:
                    last_m_risk = swarm.vote_risk(rp)
                last_banc = banc.evaluate(
                    rp, malecns_mult=float(last_m_risk.get("mult", 1.0))
                )
                banc_mults.append(float(last_banc.get("mult", 1.0)))
            robust_bar = conf_bar(conf_fly, raw_tech)
            koo9_mult = 0.0
            ens_mult = 1.0
            decision, tag, n_agree = "HOLD", "init", 0
            if koo9_on:
                if need_fly:
                    decision = last_koo9.get("side") or "HOLD"
                    tag = str(last_koo9.get("tag") or "hold 0B/0S")
                    n_agree = int(last_koo9.get("k") or 0)
                    koo9_mult = float(last_koo9.get("mult") or 0.0)
                    if decision not in ("BUY", "SELL") or koo9_mult <= 0:
                        decision = "HOLD"
                else:
                    decision, tag, n_agree, koo9_mult = "HOLD", "chop-or-session", 0, 0.0
                tech = raw_tech
                fly_trend_f = fly_trend
                fly_fade_f = fly_fade
            elif cat_on:
                tech = raw_tech
                fly_trend_f = fly_trend
                fly_fade_f = fly_fade
                inhibit_bar = cat_inhibit if fusion_on else 0.0
                if need_fly:
                    decision, tag, n_agree, last_cat = decide_categories(
                        last_cat_flies,
                        last_cat_tech,
                        feat.get("categories") or {},
                        flow,
                        thresh=inhibit_bar,
                    )
                    winner = str(last_cat.get("winner") or "")
                    wrow = (last_cat.get("categories") or {}).get(winner) or {}
                    wfly = last_cat_flies.get(winner) or {}
                    if winner:
                        tech = str(wrow.get("tech") or tech)
                        fly_trend_f = str(wfly.get("trend") or fly_trend_f)
                        fly_fade_f = str(wfly.get("fade") or fly_fade_f)
                        conf_fly = str(wfly.get("conf") or conf_fly)
                        conf_score = float(wfly.get("conf_score") or conf_score)
                        t_score = float(wfly.get("trend_score") or t_score)
                        f_score = float(wfly.get("fade_score") or f_score)
                    ens_mult = vol_size_mult(feat)
                    heads_w = (wfly.get("heads") or {}) if wfly else {}
                    if heads_w:
                        last_heads.update(heads_w)
                else:
                    decision, tag, n_agree = "HOLD", "chop-or-session", 0
            elif fusion_on:
                # Native votes go to 2oo3. Fusion + fly-conf may only abstain.
                tech = gate_vote(raw_tech, tech_obs, conf_fly, robust_bar, oppose_drop=-0.10)
                fly_trend_f = gate_vote(fly_trend, trend_obs, conf_fly, robust_bar, oppose_drop=-0.12)
                fly_fade_f = gate_vote(
                    fly_fade,
                    fade_obs,
                    conf_fly,
                    robust_bar,
                    oppose_drop=-0.28,
                    use_conf=False,
                )
            else:
                tech = raw_tech
                fly_trend_f = fly_trend
                fly_fade_f = fly_fade
            if ens_on and ens_book is not None:
                nodes9 = pack_ensemble_nodes(feat, close, tech, fly_trend_f, fly_fade_f, flow, params)
                if need_fly:
                    books = run_books(nodes9)
                    decision, tag, n_agree, last_ens = combine_books(books, ens_book, flow)
                    ens_mult = float(last_ens.get("ens_mult") or 1.0)
                    if decision not in ("BUY", "SELL"):
                        decision = "HOLD"
                        ens_mult = 1.0
                else:
                    decision, tag, n_agree, ens_mult = "HOLD", "chop-or-session", 0, 1.0
                    last_ens = {"books": {}, "on_side": [], "ens_mult": 0.0, "nodes": nodes9}
                last_ens["nodes"] = nodes9
            elif nest_on:
                nodes9 = pack_nodes(feat, close, tech, fly_trend_f, fly_fade_f, flow, params)
                decision, tag, n_agree, last_nest = nested_committee(nodes9, kind, flow)
            elif not koo9_on and not cat_on:
                nodes9 = pack_nodes(feat, close, tech, fly_trend_f, fly_fade_f, flow, params)
                decision, tag, n_agree = committee(tech, kind, fly_trend_f, fly_fade_f, flow)
                last_nest = {"families": {}, "nodes": nodes9}
            fly = fly_trend_f
            pip_now = pip_size(close)
            spread_pips = float(getattr(args, "pair_spread", account.spread)) / max(pip_now, 1e-9)
            if (
                not is_frozen(str(getattr(args, "symbol", SYMBOL)))
                and spread_pips >= WIDE_PIPS
                and decision in ("BUY", "SELL")
            ):
                if nest_on:
                    ok_wide = str(tag).startswith("3oo3") or str(tag).startswith("2oo3")
                elif koo9_on:
                    ok_wide = n_agree >= 6 and "oo9" in str(tag) and str(tag)[:1].isdigit()
                elif ens_on:
                    ok_wide = str(tag).startswith("3oo3") or n_agree >= 3
                elif pair_book_on:
                    ok_wide = pair_book.wide_ok(tag)
                else:
                    ok_wide = str(tag).startswith("3oo3")
                if not ok_wide:
                    decision, tag, n_agree = "HOLD", "wide-spread-need-3oo3", n_agree
            if (
                banc is not None
                and tag == "1oo3-bounce"
                and float(last_banc.get("mult", 1.0)) < 0.70
            ):
                decision, tag, n_agree = "HOLD", "banc-threat-bounce", 0

            if sugar_on:
                kal = feat.get("kalman") or {}
                snap_s = account.snapshot(bid, ask)
                trend_with = bool(
                    ticket is not None
                    and account.pos is not None
                    and str(fly_trend_f or fly_trend or "") == str(account.pos.side)
                )
                nutrition = food_nutrition(
                    tag,
                    banc_mult=float(last_banc.get("mult", 1.0)),
                    uncertainty=float(kal.get("uncertainty") or 0.5),
                    agreement=float(kal.get("agreement") or 0.5),
                    n_agree=int(n_agree or 0),
                    in_trade=ticket is not None,
                    trend_with=trend_with,
                )
                sat = compute_satiety(
                    equity=float(snap_s["equity"]),
                    peak_eq=float(rails.peak or peak_eq),
                    start=float(account.start_balance),
                    day_start_eq=float(day_start_eq),
                    floating=float(snap_s.get("floating") or 0.0),
                    day_trades=int(day_trades),
                    day_trade_cap=int(profile.get("day_trades") or MAX_DAY_TRADES),
                )
                live_amt = sugar_amt
                dyn_pack = None
                if sugar_dyn:
                    if sugar_dose_m is None:
                        sugar_dose_m = SugarDoseMachine()
                    dyn_pack = sugar_dose_m.step(
                        {
                            "tag": tag,
                            "n_agree": int(n_agree or 0),
                            "nutrition": nutrition,
                            "satiety": sat,
                            "banc_mult": float(last_banc.get("mult", 1.0)),
                            "uncertainty": float(kal.get("uncertainty") or 0.5),
                            "agreement": float(kal.get("agreement") or 0.5),
                            "regime": feat.get("regime") or "CHOP",
                            "atr_ratio": float(feat.get("atr_ratio") or 1.0),
                            "in_trade": ticket is not None,
                            "rails_kill": bool(getattr(rails, "kill", False)),
                        },
                        scale=sugar_amt,
                    )
                    live_amt = float(dyn_pack.get("dose") or sugar_amt)
                drive = float(min(max(nutrition * live_amt, 0.0), 1.0))
                last_sugar_feed = float(swarm.vote_sugar(drive).get("feeding") or drive)
                g_hold = overlay_geom(open_geom or params.snapshot(), profile)
                unreal_now = 0.0
                if account.pos is not None:
                    anchor = origin_entry or account.pos.entry
                    unreal_now = (bid - anchor) if account.pos.side == "BUY" else (anchor - ask)
                last_sugar = mix_appetite(
                    last_sugar_feed,
                    sat,
                    tag=tag,
                    bars_held=int(bars_held),
                    min_hold=int(round(float(g_hold.get("min_hold_bars") or 4.0))),
                    unreal=unreal_now,
                    atr=float(feat.get("atr") or 0.0),
                    dose=live_amt,
                )
                last_sugar["nutrition"] = round(float(nutrition), 4)
                last_sugar["dose"] = round(float(live_amt), 3)
                last_sugar["dose_scale"] = round(float(sugar_amt), 3)
                if dyn_pack:
                    last_sugar["dyn"] = True
                    last_sugar["dyn_state"] = str(dyn_pack.get("state") or "")
                    last_sugar["dyn_reason"] = str(dyn_pack.get("reason") or "")
                else:
                    last_sugar["dyn"] = False
                    last_sugar["dyn_state"] = "OFF"
                    last_sugar["dyn_reason"] = "manual"
                if (
                    last_sugar.get("skip")
                    and ticket is None
                    and decision in ("BUY", "SELL")
                ):
                    decision, tag, n_agree = "HOLD", f"sugar-{last_sugar.get('reason') or 'full'}", n_agree
            else:
                last_sugar = native_lamp_pack(
                    float(last_m_risk.get("sweet") or 0.0),
                    swarm._sugar_rest,
                )

            if stop and ticket is not None:
                close_now("stop-out")
            elif ticket is not None and should_flatten(stamp):
                close_now("session flatten")
            else:
                if step % (15 if delay > 0 else 80) == 0:
                    snap = account.snapshot(bid, ask)
                    koo_txt = (
                        f"koo9 {int(last_koo9.get('buy') or 0)}B/{int(last_koo9.get('sell') or 0)}S "
                        if koo9_on
                        else (
                            f"ens {int(last_ens.get('buy') or 0)}B/{int(last_ens.get('sell') or 0)}S "
                            if ens_on
                            else (
                                f"cat {last_cat.get('winner') or '-'} "
                                if cat_on
                                else ""
                            )
                        )
                    )
                    print(
                        f"[{label}] {close:.5f}  raw t={fly_trend:4s} f={fly_fade:4s} tech={raw_tech:4s}  "
                        f"fused T={tech:4s} t={fly_trend_f:4s} f={fly_fade_f:4s} cf={conf_fly:4s}  "
                        f"{koo_txt}→ {decision:4s} {tag}  k={feat['impulse']:+.2f} RSI={feat['rsi']:.0f}  "
                        f"BANC×{float(last_banc.get('mult', 1.0)):.2f}  "
                        + (
                            f"sugar amt {float(last_sugar.get('dose') or sugar_amt):.2f}"
                            f"{(' ' + str(last_sugar.get('dyn_state'))) if last_sugar.get('dyn') else ''} "
                            f"feed {float(last_sugar.get('feeding') or 0):.2f} "
                            f"sat {float(last_sugar.get('satiety') or 0):.2f}  "
                            if sugar_on
                            else f"sugar lamp feed {float(last_sugar.get('feeding') or 0):.2f}  "
                        )
                        + f"eq ${snap['equity']:.2f}"
                    )

                if ticket is None and decision in ("BUY", "SELL") and feat["ready"]:
                    trade_geom = overlay_geom(params.snapshot(), profile)
                    pip = pip_size((bid + ask) / 2.0 if bid else close)
                    stop_pips = max(trade_geom["sl_atr"] * feat["atr"], MIN_STOP_PIPS * pip) / pip
                    day_loss_cap = int(profile["day_losses"])
                    day_trade_cap = int(profile["day_trades"])
                    room = day_losses < day_loss_cap and day_trades < day_trade_cap
                    opened_here = False
                    if room and in_session(stamp, str(getattr(args, "symbol", ""))):
                        snap = account.snapshot(bid, ask)
                        pip = pip_size((bid + ask) / 2.0)
                        stop_pips = max(trade_geom["sl_atr"] * feat["atr"], MIN_STOP_PIPS * pip) / pip
                        risk_usd_guess = snap["equity"] * (risk_pct / 100.0) if risk_pct > 0 else 0.0
                        scored = scorer.score(
                            decision, feat, fly_trend, snap["equity"], risk_usd_guess, stop_pips
                        )
                        scored["koo9_mult"] = koo9_mult
                        scored["ens_mult"] = ens_mult
                        scored["ens_on_side"] = list(last_ens.get("on_side") or [])
                        scored["cat_winner"] = str(last_cat.get("winner") or "")
                        scored["votes"] = (
                            f"raw tech={raw_tech}/{kind} t={fly_trend} f={fly_fade}  "
                            f"fused T={tech} t={fly_trend_f} f={fly_fade_f} cf={conf_fly}@{robust_bar:.2f} "
                            f"koo9 {last_koo9.get('buy', 0)}B/{last_koo9.get('sell', 0)}S {tag}"
                            if koo9_on
                            else (
                                f"raw tech={raw_tech}/{kind} t={fly_trend} f={fly_fade}  "
                                f"fused T={tech} t={fly_trend_f} f={fly_fade_f} cf={conf_fly}@{robust_bar:.2f} "
                                f"ens {last_ens.get('books')} {tag}"
                                if ens_on
                                else (
                                    f"cat {last_cat.get('winner') or '-'} {tag} conf={float(last_cat.get('conf') or 0):.2f}  "
                                    f"inhibit {last_cat.get('inhibited')}  lose {last_cat.get('losers')}  "
                                    f"{last_cat.get('conflict') or ''}"
                                    if cat_on
                                    else (
                                        f"raw tech={raw_tech}/{kind} t={fly_trend} f={fly_fade}  "
                                        f"fused T={tech} t={fly_trend_f} f={fly_fade_f} cf={conf_fly}@{robust_bar:.2f} "
                                        f"fam {last_nest.get('families')} {tag}"
                                    )
                                )
                            )
                        )
                        scored["cast_tech"] = tech
                        scored["cast_trend"] = fly_trend_f
                        scored["cast_fade"] = fly_fade_f
                        scored["conf_fly"] = conf_fly
                        scored["banc"] = float(last_banc.get("mult", 1.0))
                        scored["stop_pips"] = stop_pips
                        ok, why, size_mult = scorer.gate(scored, tag)
                        rails.mark(float(snap["equity"]))
                        block = rails.blocked(step, utc_day(stamp))
                        if not ok:
                            print(f"[{label}] skip {decision}: {why}  ({scored['note']})")
                        elif block:
                            print(f"[{label}] skip {decision}: {block}")
                        else:
                            pb_skip = pair_book.skip_open(decision, step)
                            if pb_skip:
                                print(f"[{label}] skip {decision}: {pb_skip}")
                            else:
                                peak_eq = max(peak_eq, float(snap["equity"]))
                                admit, admit_note = rails.admit()
                                rec, rec_note = recovery_scale(
                                    consec_losses,
                                    float(snap["equity"]),
                                    peak_eq,
                                    float(last_banc.get("mult", 1.0)),
                                    martingale_on,
                                    tag,
                                    last_loss_r=float(rails.last_loss_r),
                                    admit_mult=admit,
                                )
                                last_recover = "  ".join(x for x in (admit_note, rec_note) if x)
                                book_mult = float(pair_book.size_mult())
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
                                    sl_atr=trade_geom["sl_atr"],
                                    size_mult=size_mult * book_mult,
                                    best_win_pips=best_win_pips,
                                    lock_pips=params.lock_win_pips,
                                    banc_mult=float(last_banc.get("mult", 1.0)),
                                    recover_mult=rec,
                                    admit_mult=admit,
                                    sugar_mult=float(last_sugar.get("size_mult") or 1.0) if sugar_on else 1.0,
                                )
                                if book_mult < 0.99:
                                    size_note = f"{size_note}  pair-book×{book_mult:.2f}"
                                if lots >= min_lots and costs_ok(
                                    account, lots, bid, ask, slip, feat["atr"],
                                    trade_geom["sl_atr"], trade_geom["tp_atr"],
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
                                        sl_atr = trade_geom["sl_atr"]
                                        tp_atr = trade_geom["tp_atr"]
                                        if decision == "BUY":
                                            sl, tp = fill.price - sl_atr * atr, fill.price + tp_atr * atr
                                        else:
                                            sl, tp = fill.price + sl_atr * atr, fill.price - tp_atr * atr
                                        snowball_adds = 0
                                        last_add_step = -99
                                        orig_lots = lots
                                        origin_entry = fill.price
                                        snow_note = ""
                                        opened_here = True
                                        if dash is not None:
                                            dash.publish(
                                                {},
                                                log_line={
                                                    "t": label,
                                                    "side": decision,
                                                    "what": "OPEN",
                                                    "detail": f"{lots:.2f} lots @ {fill.price:.5f}  {tag}",
                                                },
                                            )
                    if not opened_here:
                        maybe_shadow(decision, tag, stop_pips, pip)

                elif ticket is not None and account.pos is not None:
                    bars_held += 1
                    pos = account.pos
                    atr = feat["atr"]
                    g = overlay_geom(open_geom or params.snapshot(), profile)
                    anchor = origin_entry or pos.entry
                    if pos.side == "BUY":
                        peak = max(peak, bid)
                        unreal = bid - anchor
                        if bid >= anchor + g["be_atr"] * atr:
                            sl = max(sl, anchor + 0.15 * atr)
                        if bid >= anchor + g["trail_arm_atr"] * atr:
                            sl = max(sl, bid - g["trail_gap_atr"] * atr)
                    else:
                        peak = min(peak, ask) if peak else ask
                        unreal = anchor - ask
                        if ask <= anchor - g["be_atr"] * atr:
                            sl = min(sl, anchor - 0.15 * atr)
                        if ask <= anchor - g["trail_arm_atr"] * atr:
                            sl = min(sl, ask + g["trail_gap_atr"] * atr)
                    if (
                        snowball_on
                        and in_session(stamp, str(getattr(args, "symbol", "")))
                        and not should_flatten(stamp)
                        and not rails.kill
                        and (not sugar_on or bool(last_sugar.get("snowball")))
                    ):
                        snow_trend = fly_trend_f
                        snow_fade = fly_fade_f
                        if koo9_on:
                            snow_trend = last_koo9.get("side") if int(last_koo9.get("k") or 0) >= 6 else "HOLD"
                            snow_fade = "HOLD"
                        add_lots, why_add = snowball_add_lots(
                            account,
                            feat,
                            snow_trend,
                            snow_fade,
                            sl,
                            bid,
                            ask,
                            slip,
                            bars_held,
                            snowball_adds,
                            last_add_step,
                            step,
                            orig_lots,
                            origin_entry,
                            peak,
                            int(round(g["min_hold_bars"])),
                            min_lots,
                            max_lots,
                            g["be_atr"],
                            g["trail_arm_atr"],
                            banc_mult=float(last_banc.get("mult", 1.0)),
                        )
                        if add_lots >= min_lots:
                            added = account.add(add_lots, bid, ask, slip, stamp)
                            if added.ok:
                                snowball_adds += 1
                                last_add_step = step
                                lots = account.pos.lots
                                snow_note = why_add
                                result = _explain(broker.trade(pos.side, args.symbol, add_lots))
                                print(
                                    f"[{label}] SNOWBALL {pos.side} +{add_lots:.2f} lots @ {added.price:.5f}  "
                                    f"total {lots:.2f}  #{snowball_adds}  {why_add}  → {result}"
                                )
                                if dash is not None:
                                    dash.publish(
                                        {},
                                        log_line={
                                            "t": label,
                                            "side": pos.side,
                                            "what": f"SNOWBALL #{snowball_adds}",
                                            "detail": f"+{add_lots:.2f} → {lots:.2f} @ {added.price:.5f}",
                                        },
                                    )
                            elif "margin" in added.reason.lower():
                                print(f"[{label}] snowball skip: {added.reason}")
                    hit_sl = (pos.side == "BUY" and bid <= sl) or (pos.side == "SELL" and ask >= sl)
                    hit_tp = (pos.side == "BUY" and bid >= tp) or (pos.side == "SELL" and ask <= tp)
                    trend_flip = (pos.side == "BUY" and feat["regime"] == "DOWN") or (
                        pos.side == "SELL" and feat["regime"] == "UP"
                    )
                    chopped = feat["regime"] == "CHOP" and unreal < float(profile["chop_unreal_atr"]) * atr
                    eq_now = float(account.snapshot(bid, ask)["equity"])
                    risk_why = risk_close_reason(
                        profile,
                        side=pos.side,
                        bars_held=bars_held,
                        feat=feat,
                        banc_mult=float(last_banc.get("mult", 1.0)),
                        fade=str(fly_fade_f or fly_fade or ""),
                        rails_kill=bool(rails.kill),
                        unreal=unreal,
                        atr=atr,
                        equity=eq_now,
                        peak_eq=float(rails.peak or peak_eq),
                    )
                    if hit_sl:
                        close_now("ATR stop")
                    elif hit_tp:
                        close_now("ATR target")
                    elif risk_why:
                        close_now(risk_why)
                    elif sugar_on and last_sugar.get("flatten"):
                        close_now("sugar full")
                    elif bars_held >= int(round(g["max_hold_bars"])):
                        close_now("time stop")
                    elif bars_held >= int(round(g["min_hold_bars"])) and trend_flip:
                        close_now("trend flip")
                    elif bars_held >= int(round(g["min_hold_bars"])) + 2 and chopped:
                        close_now("chop scratch")

            if dash is not None and (not quiet) and dash.want_publish(
                force=bool(ticket) or decision in ("BUY", "SELL"),
                every_ms=400.0 if delay == 0 else 80.0,
            ):
                snap = account.snapshot(bid, ask)
                pos_now = account.pos
                nfire = int((last_ballots or {}).get("nfire") or 0)
                nv9 = last_koo9.get("votes") or {}
                sc9 = last_koo9.get("scores") or {}
                node_board: dict = {}
                if koo9_on and nine is not None:
                    heads = {
                        "trend": pack_head(
                            brain, nine.x["trend"], nv9.get("trend", "HOLD"),
                            float(sc9.get("trend") or 0.0), nfire,
                            brain.tau * 0.55, "trend", nine.p["trend"], viz=True,
                        ),
                        "fade": pack_head(
                            brain, nine.x["fade"], nv9.get("fade", "HOLD"),
                            float(sc9.get("fade") or 0.0), nfire,
                            brain.tau * 0.85, "fade", nine.p["fade"], viz=True,
                        ),
                        "conf": pack_head(
                            brain, nine.x["mix"], nv9.get("mix", "HOLD"),
                            float(sc9.get("mix") or 0.0), nfire,
                            brain.tau * 0.50, "mix", nine.p["mix"], viz=True,
                        ),
                    }
                elif cat_on and cat_pool is not None:
                    viz_name = str(last_cat.get("winner") or "trend")
                    if viz_name not in last_cat_flies:
                        viz_name = "trend"
                    cfly = last_cat_flies.get(viz_name) or {}
                    heads = cfly.get("heads") or {}
                    if len(heads) < 3:
                        heads = {
                            "trend": pack_head(
                                brain, cat_pool.x[(viz_name, "trend")], cfly.get("trend", "HOLD"),
                                float(cfly.get("trend_score") or 0.0), nfire,
                                brain.tau * 0.55, f"{viz_name}-trend", cat_pool.p[(viz_name, "trend")], viz=True,
                            ),
                            "fade": pack_head(
                                brain, cat_pool.x[(viz_name, "fade")], cfly.get("fade", "HOLD"),
                                float(cfly.get("fade_score") or 0.0), nfire,
                                brain.tau * 1.0, f"{viz_name}-fade", cat_pool.p[(viz_name, "fade")], viz=True,
                            ),
                            "conf": pack_head(
                                brain, cat_pool.x[(viz_name, "conf")], cfly.get("conf", "HOLD"),
                                float(cfly.get("conf_score") or 0.0), nfire,
                                brain.tau * 0.70, f"{viz_name}-conf", cat_pool.p[(viz_name, "conf")], viz=True,
                            ),
                        }
                    node_board = cat_pool.snapshot_nodes(last_cat_flies, last_cat_tech, last_cat)
                else:
                    heads = {
                        "trend": pack_head(
                            brain, swarm.trend_x, fly_trend, t_score, nfire,
                            brain.tau * 0.55, "trend", swarm.trend_p, viz=True,
                        ),
                        "fade": pack_head(
                            brain, swarm.fade_x, fly_fade, f_score, nfire,
                            brain.tau * 1.0, "fade", swarm.fade_p, viz=True,
                        ),
                        "conf": pack_head(
                            brain, swarm.conf_x, conf_fly, conf_score, nfire,
                            brain.tau * 0.70, "conf", swarm.conf_p, viz=True,
                        ),
                    }
                cns_xs = {
                    "trend": swarm.trend_x,
                    "fade": swarm.fade_x,
                    "conf": swarm.conf_x,
                    "risk": swarm.risk_x,
                    "sugar": swarm.sugar_x,
                }
                if koo9_on and nine is not None:
                    cns_xs["trend"] = nine.x.get("trend", swarm.trend_x)
                    cns_xs["fade"] = nine.x.get("fade", swarm.fade_x)
                    cns_xs["conf"] = nine.x.get("mix", swarm.conf_x)
                elif cat_on and cat_pool is not None:
                    viz_key = str(last_cat.get("winner") or "trend")
                    pool_x = getattr(cat_pool, "x", {}) or {}
                    if (viz_key, "trend") not in pool_x:
                        viz_key = "trend"
                    cns_xs["trend"] = pool_x.get((viz_key, "trend"), swarm.trend_x)
                    cns_xs["fade"] = pool_x.get((viz_key, "fade"), swarm.fade_x)
                    cns_xs["conf"] = pool_x.get((viz_key, "conf"), swarm.conf_x)
                dash.publish(
                    {
                        "label": label,
                        "symbol": args.symbol,
                        "close": close,
                        "regime": feat.get("regime"),
                        "decision": decision,
                        "tag": tag,
                        "note": snow_note or tag,
                        "mode": "replay" if delay == 0 else "live",
                        "running": True,
                        "heads": heads,
                        "cns": pack_cns_maps(brain, cns_xs),
                        "votes": {
                            "tech": tech,
                            "trend": fly_trend_f,
                            "fade": fly_fade_f,
                            "conf": conf_fly,
                            "flow": flow,
                            **(nv9 if koo9_on else ((last_ens.get("nodes") or {}) if ens_on else (last_nest.get("nodes") or {}))),
                        },
                        "koo9": {
                            "buy": last_koo9.get("buy", 0),
                            "sell": last_koo9.get("sell", 0),
                            "hold": last_koo9.get("hold", 0),
                            "k": last_koo9.get("k", 0),
                            "mult": last_koo9.get("mult", 0.0),
                            "votes": nv9,
                        } if koo9_on else {},
                        "ensemble": last_ens if ens_on else {},
                        "families": last_nest.get("families") or {},
                        "committee_mode": (
                            "koo9" if koo9_on else
                            "nest" if nest_on else
                            "ensemble" if ens_on else
                            "legacy" if legacy_2oo3 else
                            "categories" if cat_on else
                            "fuse"
                        ),
                        "categories": last_cat if cat_on else pack_category_sensors(
                            feat.get("categories") or {},
                            (feat.get("kalman") or {}).get("cat_fuse"),
                        ),
                        "cat_fuse": (feat.get("kalman") or {}).get("cat_fuse") or idle_cat_fuse(),
                        "fuse_plan": feat.get("fuse_plan") or indicators.fuse_plan,
                        "fuse_inds": feat.get("fuse_inds") or encode_fuse_inds(fuse_inds),
                        "risk_tol": risk_tol,
                        "nodes": node_board if cat_on else sensor_nodes(
                            feat.get("categories") or {},
                            include=(feat.get("fuse_plan") or {}).get("include") or fuse_names,
                        ),
                        "fusion": {
                            "tech": float(tech_obs.get("p_cast", 0.0)),
                            "trend": float(trend_obs.get("p_cast", 0.0)),
                            "fade": float(fade_obs.get("p_cast", 0.0)),
                        },
                        "account": {
                            "equity": snap["equity"],
                            "balance": snap["balance"],
                            "start": account.start_balance,
                            "floating": snap["floating"],
                        },
                        "position": None
                        if pos_now is None
                        else {
                            "side": pos_now.side,
                            "lots": pos_now.lots,
                            "entry": pos_now.entry,
                            "floating": snap["floating"],
                        },
                        "snowball": {
                            "adds": snowball_adds,
                            "max": SNOWBALL_MAX_ADDS,
                            "last": snow_note,
                        },
                        "martingale": {
                            "streak": consec_losses,
                            "note": last_recover,
                        },
                        "rails": {
                            "kill": rails.kill,
                            "admit": rails.admit()[0],
                            "pf": round(rails.profit_factor(), 2),
                            "n": len(rails.pnls),
                            "lock": rails.day_lock,
                            "protect": rails.armed_protect,
                            "pair_book": pair_book.halt_reason or ("on" if pair_book_on else "off"),
                        },
                        "banc": last_banc,
                        "sugar": last_sugar,
                    }
                )

            step += 1
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        print("\nStopping...")
        if ticket is not None:
            close_now("interrupt")

    if ticket is not None:
        close_now("end of data")

    persist_state(force=True)
    if dash is not None and not quiet:
        dash.publish({"running": False})
    if bool(getattr(args, "save_brain", False)) and not quiet:
        tag = str(getattr(args, "symbol", SYMBOL)).upper()
        if (not is_frozen(tag)) or bool(getattr(args, "force_brain", False)):
            print(f"brain   trained  brains/{tag}.json")
    if not quiet:
        print(
            f"adapt  saved {state_path}  bar {scorer.threshold:.0f}%  "
            f"skips {scorer.skips}  SL {params.sl_atr:.2f}  TP {params.tp_atr:.1f}  "
            f"impulse>={params.min_impulse:.2f}  RSI {params.rsi_buy_arm:.0f}/{params.rsi_sell_arm:.0f}"
        )
        if params.log:
            print("adapt  last param steps:")
            for line in params.log[-8:]:
                print(f"       {line}")
    report = ROOT / "reports" / time.strftime(f"trades-{getattr(args, 'symbol', SYMBOL)}-%Y%m%d-%H%M%S.csv")
    if not quiet:
        _write_trade_report(
            trades, report, lots, account, bid, ask, tax_rate=tax_rate, tax_model=tax_model
        )
    wins = [t for t in trades if t["usd"] > 0]
    losses = [t for t in trades if t["usd"] <= 0]
    snap = account.snapshot(bid, ask)
    da_n = 0
    if cat_pool is not None:
        da_n = cat_pool.da_updates()
    elif swarm.trend_p is not None:
        da_n = int(swarm.trend_p.n_updates)
    start_eq = float(account.start_balance)
    out = {
        "symbol": str(getattr(args, "symbol", SYMBOL)),
        "source": str(getattr(args, "bar_source", "")),
        "n": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "net_usd": round(sum(t["usd"] for t in trades), 2),
        "net_pips": round(sum(t["pips"] for t in trades), 1),
        "equity": round(snap["equity"], 2),
        "best": round(max((t["usd"] for t in trades), default=0.0), 2),
        "worst": round(min((t["usd"] for t in trades), default=0.0), 2),
        "report": str(report) if not quiet else "",
        "da_n": da_n,
        "banc_mean": round(float(np.mean(banc_mults)), 3) if banc_mults else 1.0,
        "buys": sum(1 for t in trades if t.get("side") == "BUY"),
        "sells": sum(1 for t in trades if t.get("side") == "SELL"),
        "snowball": sum(int(t.get("snowball_adds") or 0) for t in trades),
        "max_dd": round(max_dd_frac(trades, start_eq), 4),
    }
    if capture:
        out["plastic"] = swarm.dump_plastic() if plastic_on else {}
        out["params"] = params.dump()
        out["banc"] = banc.dump() if banc is not None else {}
    return out


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
        out = {
            "time": bar["time"],
            "open": bar["open"],
            "high": bar["high"],
            "low": bar["low"],
            "close": close,
            "bid": close,
            "ask": close + spread,
        }
        if bar.get("volume") is not None:
            out["volume"] = bar.get("volume")
        for key in ("vix", "put_call", "cot", "cot_net", "advance_decline", "mcclellan", "trin"):
            if key in bar:
                out[key] = bar[key]
        yield out


def find_hst(symbol: str, period: int) -> Path | None:
    root = Path.home() / "AppData" / "Roaming" / "MetaQuotes" / "Terminal"
    name = f"{symbol}{period}.hst"
    hits = list(root.glob(f"*/history/*/{name}"))
    if not hits:
        return None
    return max(hits, key=lambda p: p.stat().st_mtime)


def clamp_balance(value: object, default: float = 100_000.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = float(default)
    if v != v:  # NaN
        v = float(default)
    return float(max(100.0, min(v, 10_000_000.0)))


def apply_user_bars(bars: list[dict], timeframe: str, count: int | None) -> tuple[list[dict], str, str]:
    tf_name, mins = parse_timeframe(timeframe or "NATIVE")
    note = tf_name
    if tf_name != "NATIVE" and mins > 0:
        before = len(bars)
        bars = resample_bars(bars, mins)
        if len(bars) != before:
            note = f"resampled {before} → {len(bars)} x {tf_name}"
    if count is not None and count > 0 and len(bars) > count:
        bars = bars[-count:]
    return bars, tf_name, note


def load_yahoo_m5(symbol: str, span: str = "1mo") -> list[dict]:
    """Intraday bars from Yahoo (no API key). 5m typically caps around 60 days."""
    spec = pair_spec(symbol)
    ticker = spec.get("yahoo") or f"{symbol}=X"
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{ticker}?interval=5m&range={span}&includePrePost=false"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "FlyFOREXTrader/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"  yahoo 5m {symbol}: {exc}")
        return []
    try:
        result = payload["chart"]["result"][0]
        ts = result["timestamp"]
        q = result["indicators"]["quote"][0]
    except (KeyError, IndexError, TypeError):
        print(f"  yahoo 5m {symbol}: unexpected payload")
        return []
    bars: list[dict] = []
    for i, stamp in enumerate(ts):
        c = q["close"][i]
        o = q["open"][i]
        h = q["high"][i]
        l = q["low"][i]
        if c is None or o is None or h is None or l is None or c <= 0:
            continue
        rec = {
            "time": int(stamp),
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(c),
        }
        vols = q.get("volume") or []
        if i < len(vols) and vols[i] is not None:
            try:
                v = float(vols[i])
            except (TypeError, ValueError):
                v = 0.0
            if v > 0:
                rec["volume"] = v
        bars.append(rec)
    return bars


def clip_bars(bars: list[dict], t0: int | None, t1: int | None, strict: bool = False) -> list[dict]:
    if not bars or t0 is None or t1 is None:
        return bars
    out = [b for b in bars if t0 <= int(b["time"]) <= t1]
    if strict:
        return out
    return out if len(out) >= WARMUP_BARS + 20 else bars


def load_pair_bars(symbol: str, tf_name: str, count: int, t0: int | None, t1: int | None) -> tuple[list[dict], str]:
    """Local M5 HST if present, else Yahoo 5m, else local H4."""
    period = TIMEFRAMES.get(tf_name.upper(), 5) or 5
    hst = find_hst(symbol, period)
    bars: list[dict] = []
    source = ""
    if hst is not None:
        bars = load_hst(hst, None)
        source = f"hst {hst.name}"
    if (not bars or (tf_name.upper() == "M5" and hst is None)) and tf_name.upper() in ("M1", "M5"):
        yahoo = load_yahoo_m5(symbol, span="60d") or load_yahoo_m5(symbol, span="1mo")
        if yahoo:
            bars = yahoo
            source = f"yahoo 5m {pair_spec(symbol).get('yahoo')}"
    if not bars:
        h4 = find_hst(symbol, 240)
        if h4 is not None:
            bars = load_hst(h4, None)
            source = f"hst {h4.name} (H4 fallback — coarser than M5)"
    bars = clip_bars(bars, t0, t1, strict=(t0 is not None and t1 is not None))
    if count and count > 0 and (t0 is None or t1 is None):
        bars = bars[-count:]
    elif count and count > 0 and len(bars) > max(count, WARMUP_BARS + 20):
        bars = bars[-max(count, WARMUP_BARS + 20):]
    return bars, source


def write_pair_sweep(rows: list[dict], path: Path, title: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "symbol", "source", "n", "wins", "losses", "buys", "sells", "snowball",
        "net_pips", "net_usd", "best", "worst", "equity", "da_n", "banc_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})
    head = title or "pair sweep (same window / same rules — overfit check)"
    print(f"\n──────── {head} ────────")
    print(f"{'pair':<8} {'n':>3} {'B/S':>5} {'W/L':>6} {'pips':>8} {'USD':>11} {'best':>10} {'worst':>10} {'BANC':>6}  source")
    for row in rows:
        print(
            f"{row['symbol']:<8} {row['n']:3d} {int(row.get('buys') or 0):2d}/{int(row.get('sells') or 0):<2d} "
            f"{row['wins']:2d}/{row['losses']:<2d} "
            f"{row['net_pips']:+8.1f} {row['net_usd']:+11.2f} "
            f"{row['best']:+10.2f} {row['worst']:+10.2f}  "
            f"{float(row.get('banc_mean', 1.0)):.2f}  {row['source']}"
        )
    usd = [float(r["net_usd"]) for r in rows]
    pos = sum(1 for x in usd if x > 0)
    all_green = bool(rows) and all(x > 0 for x in usd)
    print("──────────────────────────────────────────────")
    print(
        f"{len(rows)} pairs  profitable {pos}/{len(rows)}  "
        f"mean ${sum(usd)/len(usd):+.2f}  median ${sorted(usd)[len(usd)//2]:+.2f}"
        if rows else "no pairs"
    )
    print(f"all-green {'yes' if all_green else 'no'}  (promote default only if eval all-green AND holdout 5/7 + mean>0)")
    print(f"saved {path}")
    print("If only EURUSD prints green, the week was likely overfit. Demo/replay only.")


def replay_symbol(
    brain: FlyBrain,
    args: argparse.Namespace,
    broker,
    symbol: str,
    t0: int | None = None,
    t1: int | None = None,
    delay: float = 0.0,
    hst_path: Path | None = None,
    bars: list[dict] | None = None,
    source: str = "",
    mode: str = "replay",
) -> dict | None:
    """Load bars (file / HST / Yahoo) for one pair and run the paper book. delay=0 is full speed."""
    args.symbol = str(symbol).upper()
    spec = pair_spec(args.symbol)
    args.pair_spread = float(spec.get("spread", getattr(args, "spread", SPREAD)))
    tf_name = str(getattr(args, "timeframe", "M5")).upper()
    if tf_name == "NATIVE":
        tf_name = "M5"
    dash = getattr(args, "dash", None)
    if dash is not None:
        dash.cancel.clear()
        dash.reset_run(args.symbol, mode)
        dash.publish(
            {
                "running": True,
                "mode": mode,
                "symbol": args.symbol,
                "live": True,
                "label": f"{mode} {args.symbol}",
            }
        )
    count = int(getattr(args, "bars", 0) or 0)
    if bars:
        bars, tf_name, note = apply_user_bars(bars, getattr(args, "timeframe", "NATIVE"), count if count > 0 else None)
        source = source or "uploaded file"
        if note and note not in source:
            source = f"{source} ({note})"
        args.timeframe = tf_name
    elif hst_path is not None:
        want = count if count > 0 else None
        if want is None:
            print(f"Loading {tf_name} bars from {hst_path}...")
            bars = load_hst(hst_path, None)
        else:
            want = max(want, WARMUP_BARS + 20)
            print(f"Loading {want} x {tf_name} bars from {hst_path}...")
            bars = load_hst(hst_path, want)
        source = str(hst_path)
    else:
        bars, source = load_pair_bars(args.symbol, tf_name, max(count, 0), t0, t1)
    need = WARMUP_BARS + 20 if (t0 is not None and t1 is not None) else 5
    if not bars or len(bars) < need:
        print(f"  skip {args.symbol}: not enough history ({source or 'none'})")
        if dash is not None:
            dash.publish({"running": False, "label": f"no history for {args.symbol}"})
        return None
    args.bar_source = source
    args.replay_bars = bars
    args.adapt_state = str(ROOT / "reports" / f"adapt_{args.symbol}.json")
    print(
        f"  {len(bars)} x {tf_name}  {source}  "
        f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(bars[0]['time']))} -> "
        f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(bars[-1]['time']))}"
    )
    if getattr(args, "reset_adapt", False):
        state_path = Path(args.adapt_state)
        if state_path.exists():
            state_path.unlink()
    if str(mode) == "train" and bool(getattr(args, "evolve", False)):
        from flyfx.brain.evolve import evolve_pair

        return evolve_pair(brain, bars, args, broker)
    return trade_loop(brain, replay_prices(bars, args.pair_spread), args, broker, delay=delay)


def _cmd_truthy(val) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _gui_apply_common(cmd: dict, args: argparse.Namespace) -> None:
    try:
        args.bars = int(cmd.get("bars") or 0)
    except (TypeError, ValueError):
        args.bars = 0
    args.balance = clamp_balance(cmd.get("balance"), getattr(args, "balance", 100_000.0))
    try:
        tf_name, _mins = parse_timeframe(str(cmd.get("timeframe") or args.timeframe or "M5"))
    except ValueError:
        tf_name = "M5"
    args.timeframe = tf_name
    if "fuse_cats" in cmd:
        args.fuse_cats = str(cmd.get("fuse_cats") or "")
    if "fuse_dynamic" in cmd:
        args.fuse_dynamic = _cmd_truthy(cmd.get("fuse_dynamic"))
    if "fuse_inds" in cmd:
        args.fuse_inds = str(cmd.get("fuse_inds") or "")
    if "risk_tol" in cmd:
        try:
            args.risk_tol = parse_risk_tol(str(cmd.get("risk_tol") or "balanced"))
        except ValueError:
            args.risk_tol = "balanced"
    if "legacy_2oo3" in cmd:
        args.legacy_2oo3 = _cmd_truthy(cmd.get("legacy_2oo3"))
    if "use_sugar" in cmd:
        on = _cmd_truthy(cmd.get("use_sugar"))
        args.sugar = on
        args.no_sugar = not on
    if "sugar_amt" in cmd:
        args.sugar_amt = parse_sugar_amt(cmd.get("sugar_amt"))
    if "sugar_dyn" in cmd:
        args.sugar_dyn = _cmd_truthy(cmd.get("sugar_dyn"))
    if "evolve" in cmd:
        args.evolve = _cmd_truthy(cmd.get("evolve"))
    if "evo_pop" in cmd:
        args.evo_pop = clip_pop(cmd.get("evo_pop"))
    if "evo_gens" in cmd:
        args.evo_gens = clip_gens(cmd.get("evo_gens"))


def _gui_read_user_bars(cmd: dict, default_sym: str) -> tuple[str, list | None, str]:
    sym = str(cmd.get("symbol") or default_sym or SYMBOL).upper()
    user_bars = None
    source = ""
    data_csv = cmd.get("data_csv")
    data_path = str(cmd.get("data_path") or "").strip()
    data_name = str(cmd.get("data_name") or "upload.csv")
    if data_csv:
        user_bars = parse_ohlc_text(str(data_csv), data_name)
        source = f"upload {data_name}"
        guessed = guess_symbol_from_name(data_name, sym)
        if guessed:
            sym = guessed
    elif data_path:
        user_bars, source = load_bars_file(data_path, timeframe="NATIVE", count=None)
        guessed = guess_symbol_from_name(data_path, sym)
        if guessed:
            sym = guessed
    return sym, user_bars, source


def _gui_flag_snapshot(args: argparse.Namespace) -> dict:
    keys = (
        "save_brain",
        "shadow_da",
        "use_brain",
        "reset_brain",
        "force_brain",
        "use_evo",
        "no_pair_book",
        "reset_adapt",
        "replay",
        "no_sugar",
        "sugar",
        "sugar_dyn",
        "legacy_2oo3",
        "evolve",
    )
    snap = {k: bool(getattr(args, k, False)) for k in keys}
    snap["fuse_cats"] = str(getattr(args, "fuse_cats", "") or "")
    snap["fuse_dynamic"] = bool(getattr(args, "fuse_dynamic", False))
    snap["fuse_inds"] = str(getattr(args, "fuse_inds", "") or "")
    snap["risk_tol"] = str(getattr(args, "risk_tol", "balanced") or "balanced")
    snap["sugar_amt"] = parse_sugar_amt(getattr(args, "sugar_amt", 1.0))
    snap["evo_pop"] = clip_pop(getattr(args, "evo_pop", 6))
    snap["evo_gens"] = clip_gens(getattr(args, "evo_gens", 4))
    return snap


def _gui_flag_restore(args: argparse.Namespace, snap: dict) -> None:
    for key, val in snap.items():
        setattr(args, key, val)


def gui_command_loop(brain: FlyBrain, args: argparse.Namespace, broker) -> None:
    dash = args.dash
    if dash is None:
        return
    print("dashboard  idle — select a pair, then Replay sim or Train DA. Live needs MT4.")
    dash.publish(
        {
            "mode": "idle",
            "running": False,
            "live": True,
            "symbol": getattr(args, "symbol", SYMBOL),
            "label": "idle — pick a pair, then Replay sim or Train DA",
            "committee_mode": "fuse",
            "categories": idle_categories(),
            "cat_fuse": idle_cat_fuse(),
            "fuse_plan": idle_fuse_plan(),
            "nodes": idle_nodes(),
            "balance": float(getattr(args, "balance", 100_000.0)),
            "brains": list_brain_symbols(),
            "evo": list_evo_policies(),
        }
    )
    try:
        while True:
            try:
                cmd = dash.commands.get(timeout=0.25)
            except queue.Empty:
                continue
            action = str(cmd.get("action") or "")
            if action in ("quit", "exit"):
                break
            if action == "stop":
                dash.cancel.set()
                dash.publish({"running": False, "label": "stopped"})
                continue
            if action in ("replay", "train"):
                prev = _gui_flag_snapshot(args)
                sym = str(getattr(args, "symbol", SYMBOL))
                try:
                    _gui_apply_common(cmd, args)
                    sym, user_bars, source = _gui_read_user_bars(cmd, getattr(args, "symbol", SYMBOL))
                    continue_da = _cmd_truthy(cmd.get("continue_da"))
                    use_box = _cmd_truthy(cmd.get("use_brain"))
                    use_evo_box = _cmd_truthy(cmd.get("use_evo"))
                    train = action == "train"
                    args.reset_adapt = True
                    args.replay = True
                    args.legacy_2oo3 = False
                    if train:
                        args.save_brain = True
                        args.shadow_da = True
                        args.no_pair_book = True
                        args.force_brain = True
                        args.use_brain = continue_da or use_box
                        args.reset_brain = not args.use_brain
                        args.use_evo = use_evo_box
                        args.evolve = _cmd_truthy(cmd.get("evolve"))
                        args.evo_pop = clip_pop(cmd.get("evo_pop"))
                        args.evo_gens = clip_gens(cmd.get("evo_gens"))
                    else:
                        args.save_brain = continue_da
                        args.shadow_da = continue_da
                        args.use_brain = use_box or continue_da
                        args.reset_brain = False
                        args.force_brain = bool(use_box or continue_da or use_evo_box)
                        args.use_evo = use_evo_box
                    replay_symbol(
                        brain,
                        args,
                        PaperBroker(None),
                        sym,
                        None,
                        None,
                        delay=0.0,
                        hst_path=None,
                        bars=user_bars,
                        source=source,
                        mode="train" if train else "replay",
                    )
                except Exception as exc:
                    print(f"  fail {sym}: {exc}")
                    dash.publish({"running": False, "label": f"fail {exc}"})
                else:
                    brains = list_brain_symbols()
                    evo = list_evo_policies()
                    if train:
                        if bool(getattr(args, "evolve", False)):
                            label = f"evolved brains/{sym}.evo.json — check use evolved, then Replay"
                        else:
                            label = f"trained brains/{sym}.json — check use DA brain, then Replay"
                    else:
                        label = "idle — pick a pair, then Replay sim or Train DA"
                    dash.publish(
                        {
                            "running": False,
                            "mode": "idle",
                            "label": label,
                            "brains": brains,
                            "evo": evo,
                        }
                    )
                finally:
                    _gui_flag_restore(args, prev)
                continue
            if action == "live":
                prev = _gui_flag_snapshot(args)
                try:
                    _gui_apply_common(cmd, args)
                    args.legacy_2oo3 = False
                    sym = str(cmd.get("symbol") or args.symbol or SYMBOL).upper()
                    args.symbol = sym
                    spec = pair_spec(sym)
                    args.pair_spread = float(spec.get("spread", getattr(args, "spread", SPREAD)))
                    try:
                        bridge = MT4Bridge(args.endpoint)
                        wait_for_mt4(bridge, retries=6)
                    except (SystemExit, Exception) as exc:
                        print(f"live  MT4 not connected: {exc}")
                        dash.publish({"running": False, "label": "MT4 not connected — Replay sim still works"})
                        continue
                    live_broker = PaperBroker(bridge) if args.paper else bridge
                    dash.cancel.clear()
                    dash.reset_run(sym, "live")
                    dash.publish({"mode": "live", "running": True, "symbol": sym, "label": f"live {sym} — EA Inputs set the fuse mix / dynamic flag"})
                    interval = float(args.interval_ms or 0) / 1000.0
                    if interval <= 0:
                        interval = DECISION_MS / 1000.0
                    try:
                        trade_loop(brain, live_prices(bridge, sym), args, live_broker, delay=interval)
                    except Exception as exc:
                        print(f"live fail: {exc}")
                        dash.publish({"running": False, "label": f"live fail {exc}"})
                finally:
                    _gui_flag_restore(args, prev)
                continue
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
        dash.cancel.set()


def run(args: argparse.Namespace) -> None:
    if args.ping:
        cmd_ping(args)
        return

    if getattr(args, "gui", False):
        args.dash = DashHub(int(getattr(args, "gui_port", DASH_PORT)), pairs=list(PAIR_SPECS.keys()))
        args.dash.start(open_browser=True)
        args.dash.publish(
            {
                "label": "loading MaleCNS connectome…",
                "live": True,
                "running": False,
                "mode": "boot",
                "symbol": getattr(args, "symbol", SYMBOL),
                "committee_mode": "fuse",
                "categories": idle_categories(),
                "cat_fuse": idle_cat_fuse(),
                "fuse_plan": idle_fuse_plan(),
                "nodes": idle_nodes(),
                "balance": float(getattr(args, "balance", 100_000.0)),
            }
        )
        print("  Replay sim is full-speed. Train DA writes brains/{PAIR}.json from the DATA file. Live (MT4) uses the pair you pick.")
    else:
        args.dash = None

    root = ensure_connectome(Path(args.connectome))
    brain = FlyBrain(root, substeps=args.substeps)
    if args.dash is not None:
        args.dash.set_cns_layout(brain.cns.layout_blob())
        args.dash.publish(
            {
                "label": "idle — pick a pair, then Replay sim or Train DA",
                "mode": "idle",
                "running": False,
                "committee_mode": "fuse",
                "categories": idle_categories(),
                "cat_fuse": idle_cat_fuse(),
                "fuse_plan": idle_fuse_plan(),
                "nodes": idle_nodes(),
                "balance": float(getattr(args, "balance", 100_000.0)),
            }
        )

    live = None
    hst_path = None
    pair_list = [p.strip().upper() for p in str(getattr(args, "pairs", "") or "").split(",") if p.strip()]
    if (getattr(args, "overfit", False) or getattr(args, "holdout", False)) and not pair_list:
        pair_list = [p.strip().upper() for p in DEFAULT_PAIRS.split(",") if p.strip()]
    if pair_list:
        args.replay = True

    try:
        tf_name, _tf_mins = parse_timeframe(getattr(args, "timeframe", "M5"))
    except ValueError as exc:
        raise SystemExit(str(exc))
    data_file = str(getattr(args, "data", "") or "").strip()
    if tf_name == "NATIVE" and not data_file:
        tf_name = "M5"
    args.timeframe = tf_name
    args.balance = clamp_balance(getattr(args, "balance", 100_000.0))
    if data_file:
        args.replay = True
        if not Path(data_file).exists():
            raise SystemExit(f"data file not found: {data_file}")
        if int(getattr(args, "bars", 0) or 0) == 400 and not any(
            a == "--bars" or str(a).startswith("--bars=") for a in sys.argv[1:]
        ):
            args.bars = 0

    if args.hst and not pair_list:
        if args.hst.lower() == "auto":
            tf_name = args.timeframe.upper()
            hst_path = find_hst(args.symbol, TIMEFRAMES.get(tf_name, 5) or 5)
            if hst_path is None:
                if getattr(args, "gui", False):
                    print(f"No {args.symbol} {tf_name} .hst — dashboard idle, pick a pair and Replay sim.")
                else:
                    raise SystemExit(f"No {args.symbol} {tf_name} .hst file found under MetaQuotes/Terminal.")
        else:
            hst_path = Path(args.hst)
            if not hst_path.exists():
                raise SystemExit(f"HST file not found: {hst_path}")
    elif args.hst and args.hst.lower() != "auto":
        hst_path = Path(args.hst)
        if not hst_path.exists():
            raise SystemExit(f"HST file not found: {hst_path}")

    idle_gui = bool(getattr(args, "gui", False)) and not args.dry_run and hst_path is None and not pair_list and not args.replay
    if not args.dry_run and hst_path is None and not pair_list and not args.replay and not idle_gui:
        live = MT4Bridge(args.endpoint)
        wait_for_mt4(live)

    want_orders = not (args.paper or args.dry_run or args.replay or hst_path is not None or pair_list or idle_gui)
    broker = PaperBroker(live) if not want_orders else live
    if idle_gui:
        mode = "LAB"
    elif args.dry_run:
        mode = "DRY-RUN"
    elif args.replay or hst_path is not None:
        mode = "REPLAY"
    elif args.paper:
        mode = "PAPER"
    else:
        mode = "LIVE"
    print(f"\nFly Forex Trader  {args.symbol}  ${args.balance:,.0f}  {args.lots} lots  [{mode}]")
    # dashboard already started at the top of run() so the browser is not stuck on connectome/MT4

    if args.replay or hst_path is not None or pair_list:
        tf_name = str(args.timeframe).upper()
        if tf_name != "NATIVE" and tf_name not in TIMEFRAMES:
            raise SystemExit("timeframe must be one of: " + ", ".join(k for k in TIMEFRAMES if k != "NATIVE"))
        symbols = pair_list or [args.symbol]
        t0 = t1 = None
        holdout_on = bool(getattr(args, "holdout", False))
        clip_window = bool(getattr(args, "overfit", False) or holdout_on or len(symbols) > 1)
        sweep_title = "pair sweep (same window / same rules — overfit check)"
        user_bars = None
        user_source = ""
        if data_file and len(symbols) == 1:
            try:
                user_bars, user_source = load_bars_file(data_file, timeframe="NATIVE", count=None)
            except Exception as exc:
                raise SystemExit(f"could not read --data {data_file}: {exc}")
            guessed = guess_symbol_from_name(data_file, symbols[0])
            symbols = [guessed]
            args.symbol = guessed
            clip_window = False
        if clip_window:
            eurusd_hst = find_hst("EURUSD", TIMEFRAMES.get(tf_name, 5))
            if eurusd_hst is not None:
                ref = load_hst(eurusd_hst, max(args.bars, WARMUP_BARS + 20) if args.bars else None)
                if ref:
                    t0, t1 = int(ref[0]["time"]), int(ref[-1]["time"])
                    if holdout_on:
                        hold_t1 = t0 - 60
                        hold_t0 = t0 - 7 * 86400
                        t0, t1 = hold_t0, hold_t1
                        sweep_title = "pair sweep (holdout week before EURUSD eval)"
                        print(
                            f"holdout window  {time.strftime('%Y-%m-%d %H:%M', time.gmtime(t0))} -> "
                            f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(t1))}  UTC  "
                            f"(7d before local EURUSD {tf_name} eval start)"
                        )
                    else:
                        print(
                            f"overfit window  {time.strftime('%Y-%m-%d %H:%M', time.gmtime(t0))} -> "
                            f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(t1))}  UTC  "
                            f"(from local EURUSD {tf_name})"
                        )
        summaries: list[dict] = []
        delay = float(args.interval_ms or 0) / 1000.0
        for i, sym in enumerate(symbols):
            args.symbol = sym
            spec = pair_spec(sym)
            args.pair_spread = float(spec.get("spread", args.spread))
            if user_bars is not None and len(symbols) == 1:
                bars, used_name, note = apply_user_bars(
                    user_bars,
                    tf_name,
                    max(args.bars, 0) if args.bars else None,
                )
                source = user_source if not note or note == used_name else f"{user_source} ({note})"
                args.timeframe = used_name
            elif hst_path is not None and len(symbols) == 1:
                want = max(args.bars, WARMUP_BARS + 20)
                print(f"Loading {want} x {tf_name} bars from {hst_path}...")
                bars = load_hst(hst_path, want)
                source = str(hst_path)
                used_name = tf_name
            else:
                print(f"\n── {i+1}/{len(symbols)}  {sym}  spread {args.pair_spread}  quote {spec.get('quote')} ──")
                bars, source = load_pair_bars(sym, tf_name, max(args.bars, 0), t0, t1)
                used_name = tf_name
            if not bars or len(bars) < (WARMUP_BARS + 20 if clip_window else 5):
                print(f"  skip {sym}: not enough history ({source or 'none'}  n={len(bars) if bars else 0})")
                continue
            args.bar_source = source
            args.replay_bars = bars
            print(
                f"  {len(bars)} x {used_name}  {source}  "
                f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(bars[0]['time']))} -> "
                f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(bars[-1]['time']))}"
            )
            if getattr(args, "reset_adapt", False) or len(symbols) > 1:
                state_path = Path(getattr(args, "adapt_state", "") or (ROOT / "reports" / "adapt_state.json"))
                if len(symbols) > 1:
                    state_path = ROOT / "reports" / f"adapt_{sym}.json"
                    args.adapt_state = str(state_path)
                    args.reset_adapt = True
                if state_path.exists():
                    state_path.unlink()
            if getattr(args, "dash", None) is not None:
                args.dash.cancel.clear()
                args.dash.reset_run(sym, "replay")
            try:
                summary = trade_loop(brain, replay_prices(bars, args.pair_spread), args, broker, delay=delay)
                if summary:
                    summaries.append(summary)
            except Exception as exc:
                print(f"  fail {sym}: {exc}")
        if len(summaries) > 1:
            sweep_path = ROOT / "reports" / time.strftime("pair_sweep-%Y%m%d-%H%M%S.csv")
            write_pair_sweep(summaries, sweep_path, title=sweep_title)
        if getattr(args, "gui", False):
            gui_command_loop(brain, args, broker)
        return

    if idle_gui:
        print("Press Ctrl+C to stop. Dashboard does not need MT4 for Replay sim.\n")
        gui_command_loop(brain, args, PaperBroker(None))
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
        trade_loop(brain, synthetic(), args, broker, delay=float(args.interval_ms or 0) / 1000.0)
        if getattr(args, "gui", False):
            gui_command_loop(brain, args, broker)
        return

    trade_loop(brain, live_prices(broker, args.symbol), args, broker, delay=float(args.interval_ms or DECISION_MS) / 1000.0)
    if getattr(args, "gui", False):
        gui_command_loop(brain, args, broker)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fly connectome forex trader (MT4 ZMQ)")
    p.add_argument("--connectome", default=str(CONNECTOME_DIR))
    p.add_argument("--endpoint", default=ZMQ_REP)
    p.add_argument("--symbol", default=SYMBOL, help="CLI default pair; the dashboard PAIR menu overrides this for Replay/Live")
    p.add_argument("--lots", type=float, default=LOTS, help="fixed lots when --risk is 0")
    p.add_argument("--risk", type=float, default=RISK_PCT, help="percent of equity to risk per trade (0 = fixed --lots)")
    p.add_argument(
        "--risk-tol",
        "--risk-tolerance",
        default="balanced",
        dest="risk_tol",
        help="how open trades get closed: conservative (tighter SL/TP, BANC/fade/uncertainty/vol/kill flatten), "
        "balanced (factory geometry, no extra risk-factor exits — freeze-identical), "
        "aggressive (wider SL/TP, hold longer). Live: MT4 InpRiskTol wins.",
    )
    p.add_argument("--min-lots", type=float, default=MIN_LOTS, dest="min_lots")
    p.add_argument("--max-lots", type=float, default=MAX_LOTS, dest="max_lots")
    p.add_argument("--margin-cap", type=float, default=MARGIN_CAP_PCT, dest="margin_cap", help="max percent of free margin per trade")
    p.add_argument(
        "--interval-ms",
        type=int,
        default=None,
        dest="interval_ms",
        help="sleep between bars in ms. Default 0 (full speed) for replay/GUI sim, 500 for live MT4",
    )
    p.add_argument("--substeps", type=int, default=SUBSTEPS)
    p.add_argument("--paper", action="store_true", help="read prices, do not send orders")
    p.add_argument("--dry-run", action="store_true", help="no MT4; synthetic prices")
    p.add_argument("--ping", action="store_true", help="only test the MT4 ZMQ bridge")
    p.add_argument("--replay", action="store_true", help="replay MT4 historical bars (works on weekends)")
    p.add_argument("--hst", default="", help="MT4 .hst path, or 'auto' to read the local terminal file")
    p.add_argument(
        "--pairs",
        default="",
        help="comma list for an overfit sweep on the same window, e.g. EURUSD,GBPUSD,USDJPY "
        f"(default majors: {DEFAULT_PAIRS}). Uses local M5 HST or Yahoo 5m.",
    )
    p.add_argument(
        "--overfit",
        action="store_true",
        help=f"replay {DEFAULT_PAIRS} on the EURUSD window (same rules — overfit check)",
    )
    p.add_argument(
        "--holdout",
        action="store_true",
        help="replay DEFAULT_PAIRS on the 7 days immediately before the EURUSD eval window",
    )
    p.add_argument(
        "--pair-book",
        action="store_true",
        dest="pair_book",
        help="opt-in PairBook on non-EURUSD (wide 2oo3+flow, Yahoo size cut, pair circuit); "
        "default off; frozen EURUSD never uses it",
    )
    p.add_argument(
        "--no-pair-book",
        action="store_true",
        dest="no_pair_book",
        help="force PairBook off (default is already off; wins if passed with --pair-book)",
    )
    p.add_argument("--timeframe", default="M5", help="M1, M5, M10, M15, M30, H1, H4, D1, or NATIVE (keep the file's bar size)")
    p.add_argument("--bars", type=int, default=400, help="how many historical bars to replay (0 = all bars in --data)")
    p.add_argument(
        "--data",
        "--csv",
        default="",
        dest="data",
        help="OHLC file to replay: CSV, JSON, or MT4 .hst (any bar size: 5m, 10m, 1h, …)",
    )
    p.add_argument("--spread", type=float, default=SPREAD, help="base bid-ask spread in price")
    p.add_argument(
        "--balance",
        "--capital",
        type=float,
        default=100000.0,
        dest="balance",
        help="simulated starting money (account currency)",
    )
    p.add_argument("--leverage", type=float, default=100.0, help="account leverage, e.g. 100 = 1:100")
    p.add_argument("--commission", type=float, default=7.0, help="round-turn commission USD per lot")
    p.add_argument("--slippage-pips", type=float, default=0.2, dest="slippage_pips")
    p.add_argument("--swap-long", type=float, default=-0.72, dest="swap_long", help="USD per lot per night, long")
    p.add_argument("--swap-short", type=float, default=0.12, dest="swap_short", help="USD per lot per night, short")
    p.add_argument("--stop-out", type=float, default=50.0, dest="stop_out", help="stop-out margin level percent")
    p.add_argument("--min-conf", type=float, default=MIN_CONF, dest="min_conf", help="base confidence bar percent; the live bar evolves from calibration")
    p.add_argument("--adapt-state", default="", dest="adapt_state", help="path to persist evolving confidence/params (default reports/adapt_state.json)")
    p.add_argument("--reset-adapt", action="store_true", dest="reset_adapt", help="ignore saved confidence/params and start from priors")
    p.add_argument(
        "--no-fusion",
        action="store_true",
        dest="no_fusion",
        help="skip per-node Kalman/Bayes gate and fly-conf; raw 2oo3 (snapshots/2oo3-swarm-20260919)",
    )
    p.add_argument(
        "--fuse-cats",
        "--fuse",
        default="",
        dest="fuse_cats",
        help="which category Kalmans to mix into the 3-voter impulse: "
        "trend  or  trend,volume  or  trend+structure  (default all live). "
        "Empty overlays still stay out. With --fuse-dynamic this is an allowlist. "
        "Ignored by --legacy-2oo3 / --categories",
    )
    p.add_argument(
        "--fuse-inds",
        "--fuse-indicators",
        default="",
        dest="fuse_inds",
        help="which indicator channels enter each category Kalman: "
        "ema,macd,rsi  or  trend:ema+macd;momentum:rsi  (default all). "
        "Unlisted names in a flat list stay out. Empty overlays still stay out. "
        "Ignored by --legacy-2oo3",
    )
    p.add_argument(
        "--fuse-dynamic",
        "--dynamic-fuse",
        action="store_true",
        dest="fuse_dynamic",
        help="opt-in regime FSM (TREND/RANGE/BREAK/QUIET) picks which Kalmans to mix each bar; "
        "--fuse-cats / FUSE boxes / MT4 Inputs stay an allowlist. Off by default. "
        "Ignored by --legacy-2oo3 / --categories",
    )
    p.add_argument(
        "--categories",
        action="store_true",
        dest="categories",
        help="opt-in 7-category 2oo3/3oo3 committee then confidence pick (default fuses Kalmans into one 3-fly 2oo3)",
    )
    p.add_argument(
        "--legacy-2oo3",
        action="store_true",
        dest="legacy_2oo3",
        help="EURUSD freeze mixer: EMA/RSI/ATR Kalman + 3-voter 2oo3 {tech, fly-trend, fly-fade}",
    )
    p.add_argument(
        "--no-categories",
        action="store_true",
        dest="no_categories",
        help="alias of --legacy-2oo3",
    )
    p.add_argument(
        "--cat-inhibit",
        type=float,
        default=CONF_INHIBIT_NORM,
        dest="cat_inhibit",
        help="category fly-confidence |score|/tau below this inhibits that category (default 0.50)",
    )
    p.add_argument(
        "--nest",
        action="store_true",
        help="9-node 3×3oo3 families then 2oo3 (opt-in: worse than EURUSD freeze on 9–18 Sep)",
    )
    p.add_argument(
        "--koo9",
        action="store_true",
        help="9 diverse MaleCNS flies vote 5oo9–9oo9; more agreement → more lots (opt-in; default stays EURUSD 2oo3 freeze)",
    )
    p.add_argument(
        "--ensemble",
        action="store_true",
        help="5 strategy books (2oo3/3oo3) mixed by Bayesian averaging (opt-in until it beats the 7-pair freeze mean)",
    )
    p.add_argument(
        "--no-nest",
        action="store_true",
        dest="no_nest",
        help="old 3-voter 2oo3 on every pair (EURUSD snapshot path)",
    )
    p.add_argument(
        "--snowball",
        action="store_true",
        help="pyramid into winners (on by default; use --no-snowball to disable)",
    )
    p.add_argument(
        "--no-snowball",
        action="store_true",
        dest="no_snowball",
        help="disable snowball pyramiding",
    )
    p.add_argument(
        "--martingale",
        action="store_true",
        help="capped scratch recovery (on by default; --no-martingale disables)",
    )
    p.add_argument(
        "--no-martingale",
        action="store_true",
        dest="no_martingale",
        help="disable recovery sizing",
    )
    p.add_argument(
        "--gui",
        action="store_true",
        help="open the fly-brain lab in a browser (Replay sim does not need MT4; pick the pair in the PAIR menu)",
    )
    p.add_argument("--gui-port", type=int, default=DASH_PORT, dest="gui_port")
    p.add_argument(
        "--no-plastic",
        action="store_true",
        dest="no_plastic",
        help="freeze PAM/PPL three-factor learning (W is always frozen; this also freezes readout/gains)",
    )
    p.add_argument(
        "--no-banc",
        action="store_true",
        dest="no_banc",
        help="disable female BANC + MaleCNS VNC/olfactory risk voter",
    )
    p.add_argument(
        "--sugar",
        action="store_true",
        dest="sugar",
        help="enable the sugar-feed overlay (satiety skip/size/snowball/spit-out). "
        "Off by default: 23 GRNs still get the 4k-era 0.45×nice lamp on the risk pass "
        "and eat every 2oo3. Never votes BUY/SELL. Live: MT4 InpSugar wins.",
    )
    p.add_argument(
        "--no-sugar",
        action="store_true",
        dest="no_sugar",
        help="disable the sugar-feed overlay even if --sugar is set. "
        "Native GRN lamp on the risk pass stays (4k-era). Does not change 2oo3.",
    )
    p.add_argument(
        "--sugar-amt",
        type=float,
        default=1.0,
        dest="sugar_amt",
        help="how much sugar to add when sugar feed is on: 0=none (eat like --no-sugar), 1=full (default), "
        "2=extra. 10–200 is percent (100=full). Scale when --sugar-dyn is on. "
        "Ignored with --no-sugar. Live: MT4 InpSugarAmt wins.",
    )
    p.add_argument(
        "--sugar-dyn",
        "--dynamic-sugar",
        action="store_true",
        dest="sugar_dyn",
        help="opt-in FEAST/FORAGE/NIBBLE/FAST machine picks sugar amt each bar from 2oo3, BANC, "
        "Kalman, satiety, chop, vol. --sugar-amt is a scale (1=as designed). Off by default. "
        "Ignored with --no-sugar. Live: MT4 InpSugarDyn wins.",
    )
    p.add_argument(
        "--evolve",
        action="store_true",
        dest="evolve",
        help="Train DA as a genetic search: generations of params + dynamic flags + PAM/PPL, "
        "elite survives into brains/{PAIR}.json. GUI Train DA defaults this on. "
        "Does not move W. GUI Train DA writes EURUSD like any pair; CLI still uses --force-brain / --include-eurusd.",
    )
    p.add_argument(
        "--evo-pop",
        type=int,
        default=6,
        dest="evo_pop",
        help="evolve population size (3–16, default 6)",
    )
    p.add_argument(
        "--evo-gens",
        type=int,
        default=4,
        dest="evo_gens",
        help="evolve generations (2–12, default 4)",
    )
    p.add_argument(
        "--evo-seed",
        type=int,
        default=0,
        dest="evo_seed",
        help="evolve RNG seed (0 = from clock + bar count)",
    )
    p.add_argument(
        "--use-brain",
        action="store_true",
        dest="use_brain",
        help="load brains/{SYMBOL}.json PAM/PPL traces (EURUSD never loads unless --force-brain). "
        "Also applies a nested genome if --use-evo is off and no .evo.json is selected.",
    )
    p.add_argument(
        "--use-evo",
        action="store_true",
        dest="use_evo",
        help="load brains/{SYMBOL}.evo.json — evolved AdaptiveParams + DYNAMIC/FUSE/RISK/sugar flags. "
        "Independent of DA traces. Replay only (live mix/sugar/risk stay with the EA).",
    )
    p.add_argument(
        "--reset-brain",
        action="store_true",
        dest="reset_brain",
        help="ignore brains/{SYMBOL}.json (EURUSD is always factory-frozen)",
    )
    p.add_argument(
        "--save-brain",
        action="store_true",
        dest="save_brain",
        help="write brains/{SYMBOL}.json (used by fly_train.py / GUI Train DA; EURUSD refused unless --force-brain)",
    )
    p.add_argument(
        "--force-brain",
        action="store_true",
        dest="force_brain",
        help="allow save/load of the frozen EURUSD brain (DANGER: unfreezes factory DA)",
    )
    p.add_argument(
        "--shadow-da",
        action="store_true",
        dest="shadow_da",
        help="dopamine from forward R on replay bars (fly_train.py / GUI Train DA)",
    )
    p.add_argument("--tax-rate", type=float, default=TAX_RATE, dest="tax_rate", help="ordinary/short-term tax rate percent")
    p.add_argument("--tax-model", default="ordinary", dest="tax_model", help="ordinary (spot FX) or 1256 (60/40 futures-style)")
    args = p.parse_args()
    if args.interval_ms is None:
        sim = bool(args.replay or args.hst or args.dry_run or args.overfit or args.holdout or args.gui)
        args.interval_ms = 0 if sim else DECISION_MS
    return args


if __name__ == "__main__":
    run(parse_args())
