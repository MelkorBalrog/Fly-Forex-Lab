"""Condition-aware Bayesian risk from sealed trade history.

Each closed trade updates Beta posteriors keyed by discretized market + setup
bins (side, regime, vol, session, impulse, BB state, heat, committee tag).
Before the next entry — and while a trade is open — we infer P(win), expectancy,
and a size/manage plan:

- inhibit: skip the entry when posterior risk is high enough (with evidence)
- size_mult: scale volume from half-Kelly / payoff posterior
- trim_frac: close a fraction of an open winner when conditions turned adverse
- add_ok / add_mult: allow or boost pyramiding when similar setups paid
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
import json


def _clip(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


# Weak global prior ≈ 42% win rate (matches BayesianSizer).
PRIOR_A = 3.5
PRIOR_B = 4.8
# Minimum evidence before hard inhibit / aggressive size cuts.
MIN_N_INHIBIT = 5
MIN_N_SIZE = 3
# Soft floors — inhibit when posterior is clearly bad.
P_WIN_FLOOR = 0.36
EXPECT_FLOOR = -0.15  # in R-multiples
# Trim open winners when holding posterior collapses vs entry.
TRIM_P_DROP = 0.12
TRIM_FRAC = 0.50
# Press when posterior is strong and evidence supports it.
ADD_P_WIN = 0.58
ADD_EXPECT = 0.25


def _session_bin(stamp: int) -> str:
    if not stamp:
        return "unk"
    g = time.gmtime(int(stamp))
    if g.tm_wday >= 5:
        return "wknd"
    h = g.tm_hour
    if 12 <= h < 16:
        return "overlap"
    if 8 <= h < 12:
        return "london"
    if 13 <= h < 21:
        return "ny"
    if 0 <= h < 8:
        return "asia"
    return "off"


def _vol_bin(atr_ratio: float) -> str:
    r = float(atr_ratio or 1.0)
    if r < 0.75:
        return "dead"
    if r < 1.25:
        return "mid"
    if r < 1.60:
        return "hot"
    return "spike"


def _impulse_bin(impulse: float) -> str:
    a = abs(float(impulse or 0.0))
    if a < 0.45:
        return "weak"
    if a < 0.75:
        return "mid"
    return "strong"


def _bb_bin(feat: dict) -> str:
    bw = float(feat.get("bb_bw") or 0.0)
    bw_ma = float(feat.get("bb_bw_ma") or 0.0)
    if bw_ma <= 1e-12:
        return "unk"
    ratio = bw / bw_ma
    if ratio <= 0.90:
        return "squeeze"
    if ratio >= 1.15:
        return "wide"
    return "normal"


def _heat_bin(consec_losses: int) -> str:
    n = int(consec_losses or 0)
    if n <= 0:
        return "cold"
    if n == 1:
        return "warm"
    return "hot"


def _tag_bin(tag: str) -> str:
    t = str(tag or "").lower()
    if "3oo3" in t or "3oo4" in t:
        return "full"
    if "2oo3" in t or "soft" in t:
        return "maj"
    if "bias" in t:
        return "bias"
    if "1oo3" in t:
        return "bounce"
    return "other"


def fingerprint(
    *,
    side: str,
    feat: dict | None = None,
    stamp: int = 0,
    consec_losses: int = 0,
    tag: str = "",
) -> dict[str, str]:
    """Discretize trade + market conditions into bins for the posterior table."""
    feat = feat or {}
    regime = str(feat.get("regime") or "CHOP").upper()
    if regime not in ("UP", "DOWN", "CHOP"):
        regime = "CHOP"
    return {
        "side": "BUY" if str(side).upper() == "BUY" else "SELL",
        "regime": regime,
        "vol": _vol_bin(float(feat.get("atr_ratio") or 1.0)),
        "session": _session_bin(int(stamp or 0)),
        "impulse": _impulse_bin(float(feat.get("impulse") or 0.0)),
        "bb": _bb_bin(feat),
        "heat": _heat_bin(consec_losses),
        "tag": _tag_bin(tag),
    }


def fp_key(fp: dict[str, str]) -> str:
    return "|".join(
        f"{k}={fp.get(k, '')}"
        for k in ("side", "regime", "vol", "session", "impulse", "bb", "heat", "tag")
    )


def continuous_vec(
    *,
    side: str,
    feat: dict | None = None,
    stamp: int = 0,
    consec_losses: int = 0,
) -> list[float]:
    """Compact continuous features for a similarity kernel."""
    feat = feat or {}
    side_s = 1.0 if str(side).upper() == "BUY" else -1.0
    regime = str(feat.get("regime") or "CHOP").upper()
    reg = 1.0 if regime == "UP" else (-1.0 if regime == "DOWN" else 0.0)
    atr_r = float(feat.get("atr_ratio") or 1.0)
    imp = float(feat.get("impulse") or 0.0)
    rsi = float(feat.get("rsi") or 50.0) / 100.0
    bw = float(feat.get("bb_bw") or 0.0)
    bw_ma = float(feat.get("bb_bw_ma") or 0.0)
    bb_r = (bw / bw_ma) if bw_ma > 1e-12 else 1.0
    uncert = float((feat.get("kalman") or {}).get("uncertainty") or 0.0)
    sess = _session_bin(int(stamp or 0))
    sess_v = {
        "overlap": 1.0,
        "london": 0.7,
        "ny": 0.6,
        "asia": 0.3,
        "off": 0.4,
        "wknd": 0.1,
        "unk": 0.5,
    }.get(sess, 0.5)
    heat = min(float(consec_losses or 0) / 3.0, 1.0)
    return [side_s, reg, atr_r, imp, rsi, bb_r, uncert, sess_v, heat]


@dataclass
class BinPost:
    a: float = PRIOR_A
    b: float = PRIOR_B
    w_sum: float = 0.0  # sum of winning R
    w_n: float = 0.0
    l_sum: float = 0.0  # sum of |losing| R
    l_n: float = 0.0

    def update(self, won: bool, r_mult: float) -> None:
        r = abs(float(r_mult))
        if won:
            self.a += 1.0
            self.w_sum += r
            self.w_n += 1.0
        else:
            self.b += 1.0
            self.l_sum += r
            self.l_n += 1.0

    @property
    def n(self) -> float:
        return max(0.0, self.a + self.b - PRIOR_A - PRIOR_B)

    @property
    def p_win(self) -> float:
        return float(self.a / max(self.a + self.b, 1e-9))

    @property
    def e_win(self) -> float:
        if self.w_n <= 0:
            return 1.0
        return float(self.w_sum / self.w_n)

    @property
    def e_loss(self) -> float:
        if self.l_n <= 0:
            return 1.0
        return float(self.l_sum / self.l_n)

    @property
    def expectancy(self) -> float:
        p = self.p_win
        return p * self.e_win - (1.0 - p) * self.e_loss

    @property
    def payoff(self) -> float:
        return float(max(self.e_win, 0.2) / max(self.e_loss, 0.2))

    def kelly(self) -> float:
        b = self.payoff
        p = self.p_win
        return float(_clip(p - (1.0 - p) / b, 0.0, 0.40))

    def dump(self) -> dict:
        return {
            "a": self.a,
            "b": self.b,
            "w_sum": self.w_sum,
            "w_n": self.w_n,
            "l_sum": self.l_sum,
            "l_n": self.l_n,
        }

    @classmethod
    def load(cls, data: dict) -> "BinPost":
        return cls(
            a=float(data.get("a", PRIOR_A)),
            b=float(data.get("b", PRIOR_B)),
            w_sum=float(data.get("w_sum", 0.0)),
            w_n=float(data.get("w_n", 0.0)),
            l_sum=float(data.get("l_sum", 0.0)),
            l_n=float(data.get("l_n", 0.0)),
        )


@dataclass(frozen=True)
class TradeBayesPlan:
    inhibit: bool
    size_mult: float
    trim_frac: float
    add_ok: bool
    add_mult: float
    p_win: float
    expectancy: float
    n_eff: float
    risk: float  # 0 = safe, 1 = max risk
    note: str
    fp: dict[str, str] = field(default_factory=dict)

    def pack(self) -> dict:
        return {
            "inhibit": bool(self.inhibit),
            "size_mult": round(float(self.size_mult), 3),
            "trim_frac": round(float(self.trim_frac), 3),
            "add_ok": bool(self.add_ok),
            "add_mult": round(float(self.add_mult), 3),
            "p_win": round(float(self.p_win), 3),
            "expectancy": round(float(self.expectancy), 3),
            "n_eff": round(float(self.n_eff), 2),
            "risk": round(float(self.risk), 3),
            "note": str(self.note),
            "fp": dict(self.fp),
        }


@dataclass
class _MemRow:
    fp: dict[str, str]
    vec: list[float]
    won: bool
    r_mult: float
    usd: float


class TradeBayesRisk:
    """Bayesian condition table + similarity kernel over sealed trades."""

    def __init__(self, *, enabled: bool = True, history_cap: int = 120) -> None:
        self.enabled = bool(enabled)
        self.history_cap = int(history_cap)
        self.global_post = BinPost()
        self.bins: dict[str, BinPost] = {}
        self.memory: list[_MemRow] = []
        self.last = TradeBayesPlan(
            inhibit=False,
            size_mult=1.0,
            trim_frac=0.0,
            add_ok=True,
            add_mult=1.0,
            p_win=PRIOR_A / (PRIOR_A + PRIOR_B),
            expectancy=0.0,
            n_eff=0.0,
            risk=0.5,
            note="trade-bayes off" if not enabled else "idle",
        )
        self._entry_p_win: float | None = None
        self._trimmed = False

    def clear_open(self) -> None:
        self._entry_p_win = None
        self._trimmed = False

    def mark_entry(self, plan: TradeBayesPlan | None = None) -> None:
        p = plan or self.last
        self._entry_p_win = float(p.p_win)
        self._trimmed = False

    def mark_trimmed(self) -> None:
        self._trimmed = True

    @property
    def already_trimmed(self) -> bool:
        return bool(self._trimmed)

    def remember(self, row: dict, *, feat: dict | None = None, stamp: int = 0) -> str:
        """Update posteriors from a sealed trade. Prefer row['mkt'] fingerprint."""
        if not self.enabled:
            return ""
        side = str(row.get("side") or "BUY")
        mkt = row.get("mkt")
        if isinstance(mkt, dict) and mkt:
            fp = {k: str(mkt.get(k, "")) for k in ("side", "regime", "vol", "session", "impulse", "bb", "heat", "tag")}
            fp["side"] = side if side in ("BUY", "SELL") else fp.get("side", "BUY")
        else:
            feat = feat or {}
            # Reconstruct from scored hx / row fields when mkt missing.
            synth = dict(feat)
            if "atr_ratio" not in synth and isinstance(row.get("hx"), dict):
                synth["atr_ratio"] = float((row["hx"] or {}).get("atr_ratio") or 1.0)
            if "impulse" not in synth:
                synth["impulse"] = float(row.get("impulse") or 0.0)
            if "rsi" not in synth:
                synth["rsi"] = float(row.get("rsi") or 50.0)
            if "regime" not in synth:
                synth["regime"] = "UP" if side == "BUY" else "DOWN"
            tag = ""
            if isinstance(row.get("hx"), dict):
                tag = str((row["hx"] or {}).get("tag") or "")
            heat = 0
            if isinstance(row.get("hx"), dict):
                heat = int((row["hx"] or {}).get("streak") or 0)
            fp = fingerprint(side=side, feat=synth, stamp=int(stamp or 0), consec_losses=heat, tag=tag)

        won = float(row.get("usd", 0.0)) > 0.0
        r_mult = float(row.get("r_mult") or (row.get("hx") or {}).get("r_mult") or 0.0)
        if r_mult <= 0:
            # Fallback: pips / stop_pips if present.
            pips = abs(float(row.get("pips") or 0.0))
            stop = float(row.get("stop_pips") or (row.get("hx") or {}).get("stop_pips") or 8.0)
            r_mult = pips / max(stop, 0.1)

        key = fp_key(fp)
        post = self.bins.get(key)
        if post is None:
            post = BinPost()
            self.bins[key] = post
        post.update(won, r_mult)
        self.global_post.update(won, r_mult)

        vec = continuous_vec(
            side=side,
            feat=feat
            or {
                "regime": fp.get("regime"),
                "atr_ratio": {"dead": 0.6, "mid": 1.0, "hot": 1.4, "spike": 1.8}.get(fp.get("vol", "mid"), 1.0),
                "impulse": {"weak": 0.3, "mid": 0.6, "strong": 0.9}.get(fp.get("impulse", "mid"), 0.5),
                "rsi": float(row.get("rsi") or 50.0),
                "bb_bw": 0.01,
                "bb_bw_ma": 0.01
                if fp.get("bb") == "normal"
                else (0.008 if fp.get("bb") == "squeeze" else 0.013),
                "kalman": {},
            },
            stamp=int(stamp or 0),
            consec_losses={"cold": 0, "warm": 1, "hot": 2}.get(fp.get("heat", "cold"), 0),
        )
        self.memory.append(
            _MemRow(fp=fp, vec=vec, won=won, r_mult=float(r_mult), usd=float(row.get("usd") or 0.0))
        )
        if len(self.memory) > self.history_cap:
            self.memory = self.memory[-self.history_cap :]
        self.clear_open()
        return (
            f"trade-bayes  remember  {'WIN' if won else 'LOSS'}  "
            f"R={r_mult:.2f}  bin n={post.n:.0f}  p={post.p_win:.2f}  "
            f"global n={self.global_post.n:.0f}"
        )

    def _kernel_posterior(self, vec: list[float], side: str) -> tuple[float, float, float]:
        """Recency-weighted similar-setup win rate → soft Beta counts."""
        if not self.memory:
            return PRIOR_A, PRIOR_B, 0.0
        a = PRIOR_A * 0.35
        b = PRIOR_B * 0.35
        n_eff = 0.0
        n = len(self.memory)
        for i, row in enumerate(self.memory):
            if row.fp.get("side") != side:
                # Opposite side still informs weakly.
                side_w = 0.35
            else:
                side_w = 1.0
            recency = 0.94 ** (n - 1 - i)
            # Squared Euclidean on continuous vec (scaled).
            dist = 0.0
            for x, y in zip(vec, row.vec):
                d = float(x) - float(y)
                dist += d * d
            w = side_w * recency * math.exp(-0.55 * dist)
            if w < 0.03:
                continue
            if row.won:
                a += w
            else:
                b += w
            n_eff += w
        return float(a), float(b), float(n_eff)

    def _blend(
        self, bin_post: BinPost, ka: float, kb: float, kn: float
    ) -> tuple[float, float, float, float]:
        """Blend exact-bin and kernel into p_win, expectancy, n_eff, kelly."""
        bn = float(bin_post.n)
        # More weight on exact bin as evidence grows.
        bin_w = bn / (bn + 4.0) if bn > 0 else 0.0
        ker_w = kn / (kn + 3.0) if kn > 0 else 0.0
        # Softmax-ish normalize with global residual.
        raw = bin_w + ker_w
        if raw < 1e-9:
            g = self.global_post
            return g.p_win, g.expectancy, float(g.n), g.kelly()
        bin_w /= raw
        ker_w /= raw
        # Shrink toward global when total evidence is thin.
        n_eff = bn * bin_w + kn * ker_w
        shrink = n_eff / (n_eff + 6.0)
        p_bin = bin_post.p_win
        p_ker = ka / max(ka + kb, 1e-9)
        p_glob = self.global_post.p_win
        p = (1.0 - shrink) * p_glob + shrink * (bin_w * p_bin + ker_w * p_ker)

        e_bin = bin_post.expectancy
        # Kernel expectancy from memory R.
        e_ker = 0.0
        if kn > 0.1 and self.memory:
            num = 0.0
            den = 0.0
            # Approximate: use last similar wins/losses weighted by kn already.
            for row in self.memory[-40:]:
                signed = row.r_mult if row.won else -row.r_mult
                num += signed
                den += 1.0
            e_ker = num / max(den, 1.0)
        e = (1.0 - shrink) * self.global_post.expectancy + shrink * (bin_w * e_bin + ker_w * e_ker)

        pay_bin = bin_post.payoff
        b = max(pay_bin, 0.4)
        kelly = _clip(p - (1.0 - p) / b, 0.0, 0.40)
        return float(p), float(e), float(n_eff), float(kelly)

    def infer(
        self,
        *,
        side: str,
        feat: dict | None = None,
        stamp: int = 0,
        consec_losses: int = 0,
        tag: str = "",
        unreal_r: float = 0.0,
        open_position: bool = False,
    ) -> TradeBayesPlan:
        if not self.enabled:
            self.last = TradeBayesPlan(
                inhibit=False,
                size_mult=1.0,
                trim_frac=0.0,
                add_ok=True,
                add_mult=1.0,
                p_win=0.5,
                expectancy=0.0,
                n_eff=0.0,
                risk=0.5,
                note="trade-bayes off",
            )
            return self.last

        feat = feat or {}
        fp = fingerprint(
            side=side,
            feat=feat,
            stamp=int(stamp or 0),
            consec_losses=int(consec_losses),
            tag=str(tag or ""),
        )
        key = fp_key(fp)
        bin_post = self.bins.get(key) or BinPost()
        vec = continuous_vec(
            side=side, feat=feat, stamp=int(stamp or 0), consec_losses=int(consec_losses)
        )
        ka, kb, kn = self._kernel_posterior(vec, "BUY" if str(side).upper() == "BUY" else "SELL")
        p_win, expect, n_eff, kelly = self._blend(bin_post, ka, kb, kn)

        # Risk score: high when low p_win / negative expectancy.
        risk = _clip(0.55 * (1.0 - p_win) / 0.55 + 0.45 * _clip(-expect / 1.2, 0.0, 1.0), 0.0, 1.0)

        inhibit = False
        size_mult = 1.0
        trim_frac = 0.0
        add_ok = True
        add_mult = 1.0

        if n_eff >= MIN_N_INHIBIT and (p_win < P_WIN_FLOOR or expect < EXPECT_FLOOR):
            inhibit = True
            size_mult = 0.0
        elif n_eff >= MIN_N_SIZE:
            # Half-Kelly style volume scale, clipped.
            half = 0.5 * kelly
            # Map half-Kelly ~0.05→0.55, ~0.12→1.0, ~0.20→1.25
            size_mult = _clip(0.45 + half / 0.14, 0.40, 1.35)
            if p_win < 0.42:
                size_mult = min(size_mult, 0.70)
            if expect < 0.0:
                size_mult = min(size_mult, 0.55)

        if open_position and not self._trimmed:
            entry_p = self._entry_p_win if self._entry_p_win is not None else p_win
            # Adverse turn while in profit → trim half.
            if unreal_r > 0.25 and (entry_p - p_win) >= TRIM_P_DROP and (p_win < 0.48 or expect < 0.05):
                trim_frac = TRIM_FRAC
            # Strong posterior while winning → press.
            if unreal_r > 0.40 and p_win >= ADD_P_WIN and expect >= ADD_EXPECT and n_eff >= MIN_N_SIZE:
                add_ok = True
                add_mult = _clip(1.0 + 0.35 * (p_win - 0.50), 1.0, 1.40)
            elif n_eff >= MIN_N_SIZE and (p_win < 0.42 or expect < 0.0):
                # A cold prior is E[R] slightly negative. That is not evidence
                # the open winner should skip the pyramid.
                add_ok = False
                add_mult = 0.0

        if inhibit:
            note = (
                f"trade-bayes INHIBIT  p={p_win:.2f}  E[R]={expect:+.2f}  "
                f"n={n_eff:.1f}  risk={risk:.2f}  {fp.get('regime')}/{fp.get('vol')}/{fp.get('bb')}"
            )
        elif open_position and trim_frac > 0:
            note = (
                f"trade-bayes TRIM {trim_frac:.0%}  p={p_win:.2f}  "
                f"Δp={(self._entry_p_win or p_win) - p_win:+.2f}  E[R]={expect:+.2f}"
            )
        elif open_position and add_ok and add_mult > 1.02:
            note = (
                f"trade-bayes ADD×{add_mult:.2f}  p={p_win:.2f}  E[R]={expect:+.2f}  "
                f"size×{size_mult:.2f}"
            )
        else:
            note = (
                f"trade-bayes  p={p_win:.2f}  E[R]={expect:+.2f}  "
                f"size×{size_mult:.2f}  n={n_eff:.1f}  risk={risk:.2f}  "
                f"{fp.get('regime')}/{fp.get('vol')}/{fp.get('session')}/{fp.get('bb')}"
            )

        self.last = TradeBayesPlan(
            inhibit=bool(inhibit),
            size_mult=float(size_mult),
            trim_frac=float(trim_frac),
            add_ok=bool(add_ok),
            add_mult=float(add_mult),
            p_win=float(p_win),
            expectancy=float(expect),
            n_eff=float(n_eff),
            risk=float(risk),
            note=note,
            fp=fp,
        )
        return self.last

    def pack(self) -> dict:
        p = self.last
        return {
            "enabled": bool(self.enabled),
            "n_trades": int(self.global_post.n),
            "n_bins": len(self.bins),
            **p.pack(),
        }

    def dump(self) -> dict:
        return {
            "enabled": self.enabled,
            "global": self.global_post.dump(),
            "bins": {k: v.dump() for k, v in self.bins.items()},
            "memory": [
                {
                    "fp": r.fp,
                    "vec": r.vec,
                    "won": r.won,
                    "r_mult": r.r_mult,
                    "usd": r.usd,
                }
                for r in self.memory[-self.history_cap :]
            ],
        }

    def load(self, data: dict | None) -> None:
        if not isinstance(data, dict):
            return
        g = data.get("global")
        if isinstance(g, dict):
            self.global_post = BinPost.load(g)
        bins = data.get("bins")
        if isinstance(bins, dict):
            self.bins = {str(k): BinPost.load(v) for k, v in bins.items() if isinstance(v, dict)}
        mem = data.get("memory")
        if isinstance(mem, list):
            rows: list[_MemRow] = []
            for item in mem[-self.history_cap :]:
                if not isinstance(item, dict):
                    continue
                rows.append(
                    _MemRow(
                        fp=dict(item.get("fp") or {}),
                        vec=list(item.get("vec") or []),
                        won=bool(item.get("won")),
                        r_mult=float(item.get("r_mult") or 0.0),
                        usd=float(item.get("usd") or 0.0),
                    )
                )
            self.memory = rows

    def save_file(self, path: Path | str) -> Path:
        """Write posteriors + memory so the next session can continue."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.dump()
        payload["path"] = str(path)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def load_file(self, path: Path | str) -> bool:
        """Load a previously saved trade-bayes file. Returns True if loaded."""
        path = Path(path)
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(data, dict):
            return False
        self.load(data)
        return True


def trade_bayes_path(root: Path | str, symbol: str) -> Path:
    """Durable per-pair path: ``reports/trade_bayes_{SYMBOL}.json``."""
    tag = str(symbol or "EURUSD").upper().replace("/", "")
    return Path(root) / "reports" / f"trade_bayes_{tag}.json"
