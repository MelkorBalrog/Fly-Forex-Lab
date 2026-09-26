"""Per-category 2oo3/3oo3, confidence inhibit, and plausibility trade pick.

Voters per category: fly-trend, fly-fade, tech. Confidence fly does not vote
a side into the majority — it only inhibits. Cross-category selection is
winner-take-all on fly-confidence (3oo3 preferred when close), never an
average of BUY and SELL.
"""

from __future__ import annotations

from flyfx.sense.category_kalman import CATEGORY_NAMES, SPARSE_CATEGORIES

# |conf_score|/tau below this is LOW → category decision becomes HOLD.
CONF_INHIBIT_NORM = 0.50
# Prefer 3oo3 by this much when ranking (confidence units).
UNANIMOUS_BONUS = 0.08
# Conflict: HOLD when the two sides are this close, or both are weak.
TIE_EPS = 0.08
WEAK_CONF = 0.45
# Volume may veto a winner only when its own confidence is high AND it disagrees.
VOL_VETO_CONF = 0.70


def majority_2oo3(trend: str, fade: str, tech: str) -> tuple[str, str, int]:
    """2-of-3 (or 3-of-3) on {trend, fade, tech}. No majority → HOLD."""
    votes = [trend if trend in ("BUY", "SELL") else "HOLD",
             fade if fade in ("BUY", "SELL") else "HOLD",
             tech if tech in ("BUY", "SELL") else "HOLD"]
    buy_n = sum(1 for v in votes if v == "BUY")
    sell_n = sum(1 for v in votes if v == "SELL")
    if buy_n >= 2 and buy_n > sell_n:
        tag = "3oo3" if buy_n == 3 else "2oo3"
        return "BUY", tag, buy_n
    if sell_n >= 2 and sell_n > buy_n:
        tag = "3oo3" if sell_n == 3 else "2oo3"
        return "SELL", tag, sell_n
    return "HOLD", "HOLD", max(buy_n, sell_n)


def conf_is_low(conf_norm: float, available: bool, thresh: float = CONF_INHIBIT_NORM) -> bool:
    if not available:
        return True
    return float(conf_norm) < float(thresh)


def inhibit_category(
    decision: str,
    vote_type: str,
    n_agree: int,
    conf_norm: float,
    conf_vote: str,
    available: bool,
    thresh: float = CONF_INHIBIT_NORM,
) -> dict:
    """Drop a category trade vote when fly-confidence is LOW or opposes it."""
    inhibited = False
    why = ""
    if decision not in ("BUY", "SELL"):
        inhibited = conf_is_low(conf_norm, available, thresh)
        return {
            "decision": "HOLD",
            "vote_type": "HOLD",
            "n_agree": int(n_agree),
            "inhibited": bool(inhibited or not available),
            "inhibit_reason": "unavailable" if not available else ("low-conf" if inhibited else ""),
            "conf_norm": round(float(conf_norm), 4),
        }
    if not available or conf_is_low(conf_norm, available, thresh):
        inhibited = True
        why = "unavailable" if not available else "low-conf"
        decision, vote_type = "HOLD", "HOLD"
    elif conf_vote in ("BUY", "SELL") and conf_vote != decision:
        inhibited = True
        why = f"conf-oppose:{conf_vote}"
        decision, vote_type = "HOLD", "HOLD"
    return {
        "decision": decision,
        "vote_type": vote_type,
        "n_agree": int(n_agree),
        "inhibited": inhibited,
        "inhibit_reason": why,
        "conf_norm": round(float(conf_norm), 4),
    }


def _rank(row: dict) -> float:
    s = float(row.get("conf_norm") or 0.0)
    if row.get("vote_type") == "3oo3":
        s += UNANIMOUS_BONUS
    return s


def pick_plausible(
    rows: dict[str, dict],
    flow: str = "HOLD",
    *,
    tie_eps: float = TIE_EPS,
    vol_veto_conf: float = VOL_VETO_CONF,
) -> tuple[str, str, int, dict]:
    """Choose which surviving category actually trades.

    Prefer 3oo3 when confidence is comparable, then highest fly-confidence.
    Conflicting BUY vs SELL → higher rank, or HOLD if close / both weak.
    Volume vetoes only when volume confidence is high AND it disagrees.
    Neutral volume/breadth/sentiment never block a directional winner.
    """
    survivors: list[dict] = []
    inhibited: list[str] = []
    silent: list[str] = []
    for name in CATEGORY_NAMES:
        row = dict(rows.get(name) or {})
        row["name"] = name
        if row.get("inhibited"):
            inhibited.append(name)
        if row.get("decision") in ("BUY", "SELL"):
            survivors.append(row)
        else:
            silent.append(name)

    pack = {
        "winner": "",
        "vote_type": "HOLD",
        "conf": 0.0,
        "inhibited": inhibited,
        "silent": silent,
        "losers": [],
        "conflict": "",
        "categories": {n: dict(rows.get(n) or {}) for n in CATEGORY_NAMES},
        "vol_veto": False,
        "flow": flow,
    }

    if not survivors:
        return "HOLD", "cat-silent", 0, pack

    buys = [r for r in survivors if r["decision"] == "BUY"]
    sells = [r for r in survivors if r["decision"] == "SELL"]
    best_buy = max(buys, key=_rank) if buys else None
    best_sell = max(sells, key=_rank) if sells else None

    winner: dict | None = None
    if best_buy and best_sell:
        rb, rs = _rank(best_buy), _rank(best_sell)
        cb = float(best_buy.get("conf_norm") or 0.0)
        cs = float(best_sell.get("conf_norm") or 0.0)
        if abs(rb - rs) < tie_eps or (cb < WEAK_CONF and cs < WEAK_CONF):
            pack["conflict"] = (
                f"tie {best_buy['name']} {best_buy['decision']}@{cb:.2f} vs "
                f"{best_sell['name']} {best_sell['decision']}@{cs:.2f}"
            )
            pack["losers"] = [best_buy["name"], best_sell["name"]]
            return "HOLD", "cat-conflict", 0, pack
        winner = best_buy if rb > rs else best_sell
        loser = best_sell if winner is best_buy else best_buy
        pack["conflict"] = f"picked {winner['name']} over {loser['name']}"
        pack["losers"] = [r["name"] for r in survivors if r["name"] != winner["name"]]
    else:
        winner = best_buy or best_sell
        pack["losers"] = [r["name"] for r in survivors if r["name"] != winner["name"]]

    assert winner is not None
    vol = rows.get("volume") or {}
    vol_side = vol.get("decision")
    if (
        vol_side in ("BUY", "SELL")
        and vol_side != winner["decision"]
        and not vol.get("inhibited")
        and float(vol.get("conf_norm") or 0.0) >= vol_veto_conf
        and winner["name"] != "volume"
    ):
        pack["vol_veto"] = True
        pack["conflict"] = f"volume-veto {vol_side} vs {winner['name']} {winner['decision']}"
        pack["losers"] = [winner["name"]]
        return "HOLD", "volume-veto", 0, pack

    side = winner["decision"]
    tag = str(winner.get("vote_type") or "2oo3")
    if tag not in ("2oo3", "3oo3"):
        tag = "2oo3" if int(winner.get("n_agree") or 0) >= 2 else "HOLD"
        if tag == "HOLD":
            return "HOLD", "cat-silent", 0, pack
    n_agree = int(winner.get("n_agree") or (3 if tag == "3oo3" else 2))
    if flow == side:
        tag = tag + "+flow"
    pack["winner"] = winner["name"]
    pack["vote_type"] = tag
    pack["conf"] = round(float(winner.get("conf_norm") or 0.0), 4)
    pack["side"] = side
    pack["n_agree"] = n_agree
    return side, tag, n_agree, pack


def decide_categories(
    flies: dict[str, dict],
    tech: dict[str, dict],
    cat_states: dict[str, dict],
    flow: str = "HOLD",
    thresh: float = CONF_INHIBIT_NORM,
) -> tuple[str, str, int, dict]:
    """Run 2oo3/3oo3 + inhibit on every category, then pick the trade."""
    rows: dict[str, dict] = {}
    for name in CATEGORY_NAMES:
        f = flies.get(name) or {}
        t = tech.get(name) or {}
        st = cat_states.get(name) or {}
        available = bool(st.get("available"))
        if name in SPARSE_CATEGORIES and not available:
            available = False
        trend_v = str(f.get("trend") or "HOLD")
        fade_v = str(f.get("fade") or "HOLD")
        tech_v = str(t.get("vote") or "HOLD")
        side, vtype, n_ag = majority_2oo3(trend_v, fade_v, tech_v)
        conf_norm = float(f.get("conf_norm") or 0.0)
        conf_vote = str(f.get("conf") or "HOLD")
        gated = inhibit_category(side, vtype, n_ag, conf_norm, conf_vote, available, thresh)
        gated.update(
            {
                "name": name,
                "trend": trend_v,
                "fade": fade_v,
                "tech": tech_v,
                "tech_reason": t.get("reason", ""),
                "tech_strength": float(t.get("strength") or 0.0),
                "available": available,
                "impulse": float(st.get("impulse") or 0.0),
                "uncertainty": float(st.get("uncertainty") or 1.0),
                "agreement": float(st.get("agreement") or 0.0),
                "n_live": int(st.get("n_live") or 0),
                "n_expected": int(st.get("n_expected") or 0),
                "regime": str(st.get("regime") or ""),
                "raw_side": side,
                "raw_vote": vtype,
            }
        )
        rows[name] = gated
    return pick_plausible(rows, flow, tie_eps=TIE_EPS, vol_veto_conf=VOL_VETO_CONF)
