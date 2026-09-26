"""Diverse 2oo3 / 3oo3 strategy books, combined by Bayesian averaging.

Each book is three voters that must reach k-of-3 before it speaks:

  classic   2oo3  {tech, fly-trend, fly-fade}  opposite-fly veto (EURUSD freeze)
  structure 2oo3  {tech, EMA hold, rejection candle}  regime-signed
  momentum  2oo3  {fly-trend, flow, impulse}  trend-follow
  fade      2oo3  {fly-fade, RSI stretch, residual}  mean-revert both ways
  breakout  3oo3  {Bollinger, range/ATR volume, MACD}  stricter, both ways

Books that speak BUY or SELL are weighted by their Beta P(win). Classic
starts with a stronger prior so a lone 2oo3 freeze trade still fires.
Opposite books pull the posterior; they only override classic if their
mass is clearly larger.
"""

from __future__ import annotations

from flyfx.vote.nested_vote import pack_nodes

STRATEGY_NAMES = ("classic", "structure", "momentum", "fade", "breakout")

# Stronger prior on the EURUSD-winning book so the ensemble does not
# throw away 2oo3 {tech, trend, fade} until the others have a record.
PRIORS = {
    "classic": (8.0, 4.0),
    "structure": (3.5, 4.8),
    "momentum": (3.5, 4.8),
    "fade": (3.5, 4.8),
    "breakout": (3.0, 5.0),
}


def vote_bb(feat: dict) -> str:
    pct = float(feat.get("bb_pct", 0.5))
    if pct >= 0.78:
        return "BUY"
    if pct <= 0.22:
        return "SELL"
    return "HOLD"


def vote_vol(feat: dict) -> str:
    signed = float(feat.get("signed_vol", 0.0))
    if signed >= 0.55:
        return "BUY"
    if signed <= -0.55:
        return "SELL"
    return "HOLD"


def vote_macd(feat: dict) -> str:
    hist = float(feat.get("macd", 0.0)) - float(feat.get("macd_sig", 0.0))
    if hist >= 0.12:
        return "BUY"
    if hist <= -0.12:
        return "SELL"
    return "HOLD"


def pack_ensemble_nodes(
    feat: dict,
    close: float,
    tech: str,
    fly_trend: str,
    fly_fade: str,
    flow: str,
    params=None,
) -> dict[str, str]:
    nodes = pack_nodes(feat, close, tech, fly_trend, fly_fade, flow, params)
    nodes["bb"] = vote_bb(feat)
    nodes["vol"] = vote_vol(feat)
    nodes["macd"] = vote_macd(feat)
    return nodes


def _k_of_3(vals: list[str], k: int) -> tuple[str, int]:
    buy = sum(1 for v in vals if v == "BUY")
    sell = sum(1 for v in vals if v == "SELL")
    if buy >= k and buy > sell:
        return "BUY", buy
    if sell >= k and sell > buy:
        return "SELL", sell
    return "HOLD", max(buy, sell)


def vote_classic(nodes: dict) -> tuple[str, int]:
    tech = nodes.get("tech", "HOLD")
    trend = nodes.get("trend", "HOLD")
    fade = nodes.get("fade", "HOLD")
    if tech not in ("BUY", "SELL"):
        return "HOLD", 0
    opp = "SELL" if tech == "BUY" else "BUY"
    if trend == opp or fade == opp:
        return "HOLD", 0
    return _k_of_3([tech, trend, fade], 2)


def vote_structure(nodes: dict) -> tuple[str, int]:
    return _k_of_3(
        [nodes.get("tech", "HOLD"), nodes.get("ema", "HOLD"), nodes.get("candle", "HOLD")],
        2,
    )


def vote_momentum(nodes: dict) -> tuple[str, int]:
    return _k_of_3(
        [nodes.get("trend", "HOLD"), nodes.get("flow", "HOLD"), nodes.get("impulse", "HOLD")],
        2,
    )


def vote_fade(nodes: dict) -> tuple[str, int]:
    return _k_of_3(
        [nodes.get("fade", "HOLD"), nodes.get("rsi", "HOLD"), nodes.get("residual", "HOLD")],
        2,
    )


def vote_breakout(nodes: dict) -> tuple[str, int]:
    return _k_of_3(
        [nodes.get("bb", "HOLD"), nodes.get("vol", "HOLD"), nodes.get("macd", "HOLD")],
        3,
    )


BOOK_FNS = {
    "classic": vote_classic,
    "structure": vote_structure,
    "momentum": vote_momentum,
    "fade": vote_fade,
    "breakout": vote_breakout,
}


class StrategyBook:
    """Per-strategy Beta track record. Updated only for books that voted the fill side."""

    def __init__(self) -> None:
        self.stats: dict[str, dict] = {}
        for name in STRATEGY_NAMES:
            a, b = PRIORS[name]
            self.stats[name] = {"a": float(a), "b": float(b), "n": 0}

    def p(self, name: str) -> float:
        s = self.stats[name]
        return float(s["a"] / max(s["a"] + s["b"], 1e-9))

    def weight(self, name: str, k: int) -> float:
        w = self.p(name)
        if k >= 3:
            w *= 1.25
        return w

    def remember(self, names: list[str], won: bool) -> None:
        for name in names:
            if name not in self.stats:
                continue
            if won:
                self.stats[name]["a"] += 1.0
            else:
                self.stats[name]["b"] += 1.0
            self.stats[name]["n"] += 1

    def dump(self) -> dict:
        return {k: dict(v) for k, v in self.stats.items()}

    def load(self, data: dict) -> None:
        if not isinstance(data, dict):
            return
        for name, blob in data.items():
            if name not in self.stats or not isinstance(blob, dict):
                continue
            self.stats[name]["a"] = float(blob.get("a", self.stats[name]["a"]))
            self.stats[name]["b"] = float(blob.get("b", self.stats[name]["b"]))
            self.stats[name]["n"] = int(blob.get("n", 0))


def run_books(nodes: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name, fn in BOOK_FNS.items():
        side, k = fn(nodes)
        tag = "HOLD"
        if side in ("BUY", "SELL"):
            need = 3 if name == "breakout" else 2
            tag = "3oo3" if k >= 3 else ("2oo3" if k >= need else "HOLD")
            if tag == "HOLD":
                side, k = "HOLD", k
        out[name] = {"side": side, "k": int(k), "tag": tag}
    return out


def combine_books(
    books: dict[str, dict],
    track: StrategyBook,
    flow: str,
) -> tuple[str, str, int, dict]:
    buy_mass = sell_mass = 0.0
    buy_n = sell_n = 0
    buy_3 = sell_3 = False
    on_buy: list[str] = []
    on_sell: list[str] = []
    for name, rec in books.items():
        side = rec.get("side")
        k = int(rec.get("k") or 0)
        if side not in ("BUY", "SELL"):
            continue
        w = track.weight(name, k)
        if side == "BUY":
            buy_mass += w
            buy_n += 1
            on_buy.append(name)
            if k >= 3 or rec.get("tag") == "3oo3":
                buy_3 = True
        else:
            sell_mass += w
            sell_n += 1
            on_sell.append(name)
            if k >= 3 or rec.get("tag") == "3oo3":
                sell_3 = True

    classic = books.get("classic") or {"side": "HOLD", "k": 0, "tag": "HOLD"}
    pack = {
        "books": {n: dict(rec) for n, rec in books.items()},
        "buy": buy_n,
        "sell": sell_n,
        "buy_mass": round(buy_mass, 3),
        "sell_mass": round(sell_mass, 3),
        "p": {n: round(track.p(n), 3) for n in STRATEGY_NAMES},
        "on_side": [],
        "ens_mult": 0.0,
    }

    if buy_mass <= 0.0 and sell_mass <= 0.0:
        return "HOLD", "ens-silent", 0, pack

    if buy_mass > sell_mass:
        side, mass, opp, n_agree, has3, names = "BUY", buy_mass, sell_mass, buy_n, buy_3, on_buy
    elif sell_mass > buy_mass:
        side, mass, opp, n_agree, has3, names = "SELL", sell_mass, buy_mass, sell_n, sell_3, on_sell
    else:
        return "HOLD", "ens-tie", 0, pack

    classic_side = classic.get("side")
    classic_mass = track.weight("classic", int(classic.get("k") or 0)) if classic_side in ("BUY", "SELL") else 0.0
    if classic_side in ("BUY", "SELL") and classic_side != side:
        # Mild disagreement: keep the freeze book. Only flip if the rest
        # clearly outweigh it (about three other 2oo3 books).
        if mass < 1.80 * max(classic_mass, 1e-6):
            side = classic_side
            names = on_buy if side == "BUY" else on_sell
            n_agree = buy_n if side == "BUY" else sell_n
            has3 = buy_3 if side == "BUY" else sell_3
            mass, opp = (buy_mass, sell_mass) if side == "BUY" else (sell_mass, buy_mass)

    if classic_side not in ("BUY", "SELL"):
        if n_agree < 2 and not has3:
            return "HOLD", "ens-thin", 0, pack
        total = mass + opp
        if total > 0 and mass / total < 0.56:
            return "HOLD", "ens-split", 0, pack

    total = mass + opp
    strength = mass / total if total > 0 else 0.0
    extra = max(0, n_agree - 1)
    tag = "2oo3"
    if n_agree >= 3 or (has3 and n_agree >= 2) or int(classic.get("k") or 0) >= 3:
        tag = "3oo3"
    elif n_agree >= 2 or (classic_side == side and int(classic.get("k") or 0) >= 2):
        tag = "2oo3"
    else:
        return "HOLD", "ens-thin", 0, pack
    if flow == side:
        tag = tag + "+flow"
    ens_mult = float(max(0.85, min(1.32, 0.92 + 0.10 * extra + 0.20 * (strength - 0.50))))
    pack["on_side"] = list(names)
    pack["ens_mult"] = round(ens_mult, 3)
    pack["strength"] = round(strength, 3)
    pack["side"] = side
    pack["tag"] = tag
    pack["k"] = n_agree
    return side, tag, n_agree, pack
