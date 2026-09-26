"""Per-category Kalman, 2oo3/3oo3, inhibit, and plausibility selection."""

from __future__ import annotations

import math

from flyfx.sense.category_indicators import CategoryIndicatorEngine
from flyfx.sense.category_kalman import CategoryKalman, CategoryKalmanBank
from flyfx.sense.kalman_signal import Kalman1D
from flyfx.vote.category_tech import run_category_tech, tech_breadth, tech_sentiment, tech_volume
from flyfx.vote.category_vote import (
    decide_categories,
    inhibit_category,
    majority_2oo3,
    pick_plausible,
)


def _walk(n: int = 80, drift: float = 0.0004) -> list[dict]:
    price = 1.08000
    bars = []
    for i in range(n):
        price += drift
        high = price + 0.00030
        low = price - 0.00025
        bars.append({"high": high, "low": low, "close": price, "open": price - 0.00010, "volume": 800.0 + 10 * i})
    return bars


def test_kalman1d_missing_measurement_grows_p():
    kf = Kalman1D(q=0.01, r=0.05, x=0.0, p=0.2)
    x0, p0 = kf.step(0.5)
    x1, p1 = kf.step(None)
    x2, p2 = kf.step(float("nan"))
    assert p1 > p0
    assert p2 > p1
    assert x1 == x0
    assert math.isfinite(x2)


def test_category_kalman_skips_nan_and_fuses_live():
    kf = CategoryKalman("momentum", ("rsi", "stoch", "cci"))
    a = kf.update({"rsi": 1.2, "stoch": 1.1, "cci": None})
    assert a["available"] is True
    assert a["n_live"] == 2
    assert "cci" not in a["parts"]
    b = kf.update({"rsi": None, "stoch": None, "cci": None})
    assert b["available"] is False
    assert b["n_live"] == 0
    assert b["uncertainty"] >= a["uncertainty"]
    c = kf.update({"rsi": 1.0, "stoch": None, "cci": None})
    assert c["available"] is True
    assert c["n_live"] == 1


def test_indicators_mark_breadth_sentiment_unavailable():
    eng = CategoryIndicatorEngine(pip=0.0001)
    last = None
    for bar in _walk(60):
        last = eng.update(bar["high"], bar["low"], bar["close"], volume=bar["volume"], open_=bar["open"])
    assert last is not None
    m = last["measurements"]
    assert last["volume_live"] is True
    assert m["volume"]["obv"] is not None
    assert m["volume"]["profile"] is None
    assert m["breadth"]["ad_line"] is None
    assert m["sentiment"]["vix"] is None
    bank = CategoryKalmanBank()
    cats = bank.update(m)
    assert cats["trend"]["available"] is True
    assert cats["momentum"]["available"] is True
    assert cats["breadth"]["available"] is False
    assert cats["sentiment"]["available"] is False


def test_volume_unavailable_without_tick_volume():
    eng = CategoryIndicatorEngine(pip=0.0001)
    last = None
    for bar in _walk(40):
        last = eng.update(bar["high"], bar["low"], bar["close"], volume=None, open_=bar["open"])
    m = last["measurements"]
    assert all(v is None for v in m["volume"].values())
    cats = CategoryKalmanBank().update(m)
    feat = {**last, "categories": cats}
    vol = tech_volume(feat)
    assert vol["vote"] == "HOLD"
    assert vol["strength"] == 0.0
    assert "unavailable" in vol["reason"]


def test_majority_2oo3_and_3oo3():
    side, tag, n = majority_2oo3("BUY", "BUY", "SELL")
    assert (side, tag, n) == ("BUY", "2oo3", 2)
    side, tag, n = majority_2oo3("SELL", "SELL", "SELL")
    assert (side, tag, n) == ("SELL", "3oo3", 3)
    side, tag, n = majority_2oo3("BUY", "SELL", "HOLD")
    assert side == "HOLD"
    side, tag, n = majority_2oo3("HOLD", "HOLD", "BUY")
    assert side == "HOLD"


def test_confidence_inhibit_drops_majority():
    gated = inhibit_category("BUY", "2oo3", 2, conf_norm=0.20, conf_vote="BUY", available=True, thresh=0.50)
    assert gated["inhibited"] is True
    assert gated["decision"] == "HOLD"
    ok = inhibit_category("BUY", "3oo3", 3, conf_norm=0.80, conf_vote="BUY", available=True)
    assert ok["inhibited"] is False
    assert ok["decision"] == "BUY"
    oppose = inhibit_category("BUY", "2oo3", 2, conf_norm=0.90, conf_vote="SELL", available=True)
    assert oppose["inhibited"] is True
    missing = inhibit_category("BUY", "2oo3", 2, conf_norm=0.90, conf_vote="BUY", available=False)
    assert missing["decision"] == "HOLD"


def test_conflict_picks_higher_confidence_or_holds_on_tie():
    rows = {
        "trend": {"decision": "BUY", "vote_type": "2oo3", "n_agree": 2, "inhibited": False, "conf_norm": 0.82},
        "momentum": {"decision": "SELL", "vote_type": "2oo3", "n_agree": 2, "inhibited": False, "conf_norm": 0.40},
        "volatility": {"decision": "HOLD", "vote_type": "HOLD", "n_agree": 0, "inhibited": True, "conf_norm": 0.10},
        "volume": {"decision": "HOLD", "vote_type": "HOLD", "n_agree": 0, "inhibited": True, "conf_norm": 0.0},
        "breadth": {"decision": "HOLD", "vote_type": "HOLD", "n_agree": 0, "inhibited": True, "conf_norm": 0.0},
        "structure": {"decision": "HOLD", "vote_type": "HOLD", "n_agree": 0, "inhibited": True, "conf_norm": 0.20},
        "sentiment": {"decision": "HOLD", "vote_type": "HOLD", "n_agree": 0, "inhibited": True, "conf_norm": 0.0},
    }
    side, tag, n, pack = pick_plausible(rows)
    assert side == "BUY"
    assert pack["winner"] == "trend"
    assert "2oo3" in tag
    assert n == 2
    rows["momentum"]["conf_norm"] = 0.90
    rows["momentum"]["vote_type"] = "3oo3"
    rows["momentum"]["n_agree"] = 3
    side, tag, n, pack = pick_plausible(rows)
    # 3oo3 bonus 0.08: momentum 0.90+0.08=0.98 vs trend 0.82
    assert side == "SELL"
    assert pack["winner"] == "momentum"
    assert tag.startswith("3oo3")
    rows["trend"]["conf_norm"] = 0.81
    rows["momentum"]["conf_norm"] = 0.81
    rows["momentum"]["vote_type"] = "2oo3"
    side, tag, n, pack = pick_plausible(rows)
    assert side == "HOLD"
    assert tag == "cat-conflict"


def test_unavailable_breadth_sentiment_stay_neutral_in_pipeline():
    flies = {
        "trend": {"trend": "BUY", "fade": "BUY", "conf": "BUY", "conf_norm": 0.85},
        "momentum": {"trend": "HOLD", "fade": "HOLD", "conf": "HOLD", "conf_norm": 0.20},
        "volatility": {"trend": "HOLD", "fade": "HOLD", "conf": "HOLD", "conf_norm": 0.20},
        "volume": {"trend": "HOLD", "fade": "HOLD", "conf": "HOLD", "conf_norm": 0.0},
        "breadth": {"trend": "BUY", "fade": "BUY", "conf": "BUY", "conf_norm": 0.99},
        "structure": {"trend": "HOLD", "fade": "HOLD", "conf": "HOLD", "conf_norm": 0.20},
        "sentiment": {"trend": "SELL", "fade": "SELL", "conf": "SELL", "conf_norm": 0.99},
    }
    tech = {k: {"vote": "HOLD", "reason": "", "strength": 0.0} for k in flies}
    tech["trend"] = {"vote": "BUY", "reason": "trend-follow", "strength": 0.7}
    tech["breadth"] = {"vote": "BUY", "reason": "fake", "strength": 0.9}
    tech["sentiment"] = {"vote": "SELL", "reason": "fake", "strength": 0.9}
    states = {k: {"available": False, "impulse": 0.0, "uncertainty": 1.0, "agreement": 0.0} for k in flies}
    states["trend"] = {"available": True, "impulse": 0.6, "uncertainty": 0.2, "agreement": 0.8}
    side, tag, n, pack = decide_categories(flies, tech, states)
    assert pack["categories"]["breadth"]["decision"] == "HOLD"
    assert pack["categories"]["sentiment"]["decision"] == "HOLD"
    assert pack["categories"]["breadth"]["inhibited"] is True
    assert pack["categories"]["trend"]["tech"] == "BUY"
    assert pack["categories"]["trend"]["trend"] == "BUY"
    assert pack["categories"]["trend"]["fade"] == "BUY"
    assert side == "BUY"
    assert pack["winner"] == "trend"


def test_idle_categories_has_seven_rows():
    from flyfx.ui.dash import idle_categories

    pack = idle_categories()
    assert set(pack["categories"]) == {
        "trend",
        "momentum",
        "volatility",
        "volume",
        "breadth",
        "structure",
        "sentiment",
    }
    assert pack["winner"] == ""
    assert pack["categories"]["trend"]["available"] is False


def test_pack_head_compact_bins():
    import numpy as np

    from flyfx.ui.dash import pack_head

    class _B:
        retina = np.arange(40, dtype=np.int32)
        descending = np.arange(10, 30, dtype=np.int32)

    x = np.linspace(0.0, 1.0, 80, dtype=np.float32)
    full = pack_head(_B(), x, "BUY", 2.0, 3, 1.0, "trend-trend", viz=True, compact=False)
    lite = pack_head(_B(), x, "BUY", 2.0, 3, 1.0, "trend-trend", viz=True, compact=True)
    assert len(full["eye_bull"]) == 64
    assert len(lite["eye_bull"]) == 16
    assert len(lite["motor"]) == 24
    assert lite["compact"] is True


def test_idle_nodes_has_seven_by_four():
    from flyfx.ui.dash import idle_nodes

    board = idle_nodes()
    assert set(board) == {
        "trend",
        "momentum",
        "volatility",
        "volume",
        "breadth",
        "structure",
        "sentiment",
    }
    row = board["momentum"]
    assert set(row["heads"]) == {"trend", "fade", "conf"}
    assert row["tech"]["vote"] == "HOLD"


def test_volume_veto_only_when_confident_and_disagrees():
    rows = {
        "trend": {"decision": "BUY", "vote_type": "3oo3", "n_agree": 3, "inhibited": False, "conf_norm": 0.92},
        "momentum": {"decision": "HOLD", "vote_type": "HOLD", "n_agree": 0, "inhibited": True, "conf_norm": 0.1},
        "volatility": {"decision": "HOLD", "vote_type": "HOLD", "n_agree": 0, "inhibited": True, "conf_norm": 0.1},
        "volume": {"decision": "SELL", "vote_type": "2oo3", "n_agree": 2, "inhibited": False, "conf_norm": 0.80},
        "breadth": {"decision": "HOLD", "vote_type": "HOLD", "n_agree": 0, "inhibited": True, "conf_norm": 0.0},
        "structure": {"decision": "HOLD", "vote_type": "HOLD", "n_agree": 0, "inhibited": True, "conf_norm": 0.1},
        "sentiment": {"decision": "HOLD", "vote_type": "HOLD", "n_agree": 0, "inhibited": True, "conf_norm": 0.0},
    }
    side, tag, n, pack = pick_plausible(rows)
    assert tag == "volume-veto"
    assert side == "HOLD"
    rows["volume"]["conf_norm"] = 0.40
    side, tag, n, pack = pick_plausible(rows)
    assert side == "BUY"
    assert pack["winner"] == "trend"


def test_breadth_sentiment_tech_neutral_when_empty():
    feat = {"categories": {"breadth": {"available": False}, "sentiment": {"available": False}}}
    assert tech_breadth(feat)["vote"] == "HOLD"
    assert tech_sentiment(feat)["vote"] == "HOLD"
    assert tech_breadth(feat)["strength"] == 0.0


def test_parse_ea_fuse_spec_from_ping_and_settings():
    from flyfx.sense.category_kalman import parse_fuse_cats
    from flyfx.trader import parse_ea_fuse_spec

    assert parse_ea_fuse_spec("PONG|EURUSD|123") is None
    assert parse_ea_fuse_spec("UNKNOWN") is None
    assert parse_ea_fuse_spec("PONG|EURUSD|123|FUSE|trend,volume") == "trend,volume"
    assert parse_ea_fuse_spec("OK|FUSE|all|SYMBOL|EURUSD|MAGIC|20260919") == "all"
    assert parse_ea_fuse_spec("OK|FUSE|trend,volume|DYNAMIC|1|INDS|ema,rsi|SYMBOL|EURUSD") == "trend,volume"
    assert parse_fuse_cats(parse_ea_fuse_spec("OK|FUSE|trend,volume|SYMBOL|EURUSD")) == ("trend", "volume")


def test_parse_fuse_cats_and_include_filter():
    from flyfx.sense.category_kalman import CATEGORY_NAMES, fuse_category_latents, parse_fuse_cats

    assert parse_fuse_cats("") == CATEGORY_NAMES
    assert parse_fuse_cats("all") == CATEGORY_NAMES
    assert parse_fuse_cats("trend") == ("trend",)
    assert parse_fuse_cats("trend,volume") == ("trend", "volume")
    assert parse_fuse_cats("trend+mom") == ("trend", "momentum")
    assert parse_fuse_cats("vol") == ("volume",)
    try:
        parse_fuse_cats("trend,foo")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    blob = {n: {"available": False, "fused": 9.0, "P": 0.01, "residual": 0.0} for n in CATEGORY_NAMES}
    blob["trend"] = {"available": True, "fused": 1.0, "P": 0.01, "residual": 0.1}
    blob["momentum"] = {"available": True, "fused": -1.0, "P": 0.01, "residual": -0.1}
    only_trend = fuse_category_latents(blob, include=("trend",))
    assert only_trend["names"] == ["trend"]
    assert only_trend["selected"] == ["trend"]
    assert only_trend["fused"] > 0.9
    both = fuse_category_latents(blob, include=("trend", "momentum"))
    assert set(both["names"]) == {"trend", "momentum"}
    assert abs(both["fused"]) < 0.2
    vol_only = fuse_category_latents(blob, include=("volume",))
    assert vol_only["available"] is False
    assert vol_only["selected"] == ["volume"]


def test_fuse_skips_unavailable_and_weights_by_inverse_p():
    from flyfx.sense.category_kalman import CATEGORY_NAMES, fuse_category_latents

    empty = {n: {"available": False, "fused": 9.0, "P": 0.01, "residual": 1.0} for n in CATEGORY_NAMES}
    mix = fuse_category_latents(empty)
    assert mix["available"] is False
    assert mix["impulse"] == 0.0
    assert mix["n_live"] == 0

    empty["trend"] = {"available": True, "fused": 1.0, "P": 0.01, "residual": 0.1}
    empty["momentum"] = {"available": True, "fused": -1.0, "P": 1.0, "residual": -0.1}
    empty["volume"] = {"available": False, "fused": 50.0, "P": 1e-9, "residual": 9.0}
    mix = fuse_category_latents(empty)
    assert mix["n_live"] == 2
    assert "volume" not in mix["parts"]
    assert mix["fused"] > 0.8
    assert mix["available"] is True


def test_cli_default_is_fuse_not_legacy_or_committee(monkeypatch):
    import sys

    from flyfx.trader import parse_args

    monkeypatch.setattr(sys, "argv", ["fly_forex.py", "--replay", "--hst", "auto"])
    args = parse_args()
    assert args.legacy_2oo3 is False
    assert args.no_categories is False
    assert args.categories is False
    assert args.fuse_cats == ""
    assert args.fuse_dynamic is False
    assert args.fuse_inds == ""
    assert args.risk_tol == "scalp"
    assert args.no_sugar is False
    assert args.sugar is False
    assert args.sugar_amt == 1.0
    assert args.sugar_dyn is False
    assert args.cat_inhibit == 0.50
    monkeypatch.setattr(sys, "argv", ["fly_forex.py", "--replay", "--legacy-2oo3"])
    flagged = parse_args()
    assert flagged.legacy_2oo3 is True
    monkeypatch.setattr(sys, "argv", ["fly_forex.py", "--replay", "--categories"])
    committee = parse_args()
    assert committee.categories is True
    assert committee.legacy_2oo3 is False
    monkeypatch.setattr(sys, "argv", ["fly_forex.py", "--replay", "--fuse-cats", "trend,volume"])
    mixed = parse_args()
    assert mixed.fuse_cats == "trend,volume"
    monkeypatch.setattr(sys, "argv", ["fly_forex.py", "--replay", "--fuse-dynamic"])
    dyn = parse_args()
    assert dyn.fuse_dynamic is True
    monkeypatch.setattr(sys, "argv", ["fly_forex.py", "--replay", "--fuse-inds", "ema,rsi"])
    inds = parse_args()
    assert inds.fuse_inds == "ema,rsi"


def test_indicators_wrapper_exposes_categories():
    from flyfx.trader import Indicators

    ind = Indicators(pip=0.0001)
    feat = None
    px = 1.08
    for i in range(60):
        px += 0.0002
        feat = ind.update(px + 0.0002, px - 0.0001, px, volume=500.0, open_=px, stamp=1_700_000_000 + i * 300)
    assert feat is not None
    assert "categories" in feat
    assert feat["categories"]["trend"]["available"] is True
    assert feat["categories"]["breadth"]["available"] is False
    assert "impulse" in feat
    assert feat["ready"] is True
    assert feat["impulse"] == feat["kalman"]["cat_fuse"]["impulse"]
    assert feat["kalman"]["cat_fuse"]["available"] is True


def test_indicators_fuse_cats_trend_only_ignores_momentum():
    from flyfx.trader import Indicators

    all_mix = Indicators(pip=0.0001)
    trend_only = Indicators(pip=0.0001)
    trend_only.cats.include = ("trend",)
    trend_only.fuse_allow = ("trend",)
    px = 1.08
    feat_a = feat_t = None
    for i in range(60):
        px += 0.0002
        args = (px + 0.0002, px - 0.0001, px)
        kw = {"volume": 500.0, "open_": px, "stamp": 1_700_000_000 + i * 300}
        feat_a = all_mix.update(*args, **kw)
        feat_t = trend_only.update(*args, **kw)
    assert feat_t is not None and feat_a is not None
    assert feat_t["kalman"]["cat_fuse"]["selected"] == ["trend"]
    assert feat_t["kalman"]["cat_fuse"]["names"] == ["trend"]
    assert "momentum" not in feat_t["kalman"]["cat_fuse"]["parts"]
    # Single-family fuse impulse is the mix map of that family's fused latent
    # (tanh scale), not the raw per-category impulse field.
    from flyfx.sense.category_kalman import fuse_category_latents

    only = fuse_category_latents(feat_t["categories"], include=("trend",))
    assert abs(feat_t["impulse"] - float(only["impulse"])) < 1e-9
    # And it must differ from the all-family mix on a trending walk.
    assert abs(feat_t["impulse"] - feat_a["impulse"]) > 1e-6


def test_indicators_legacy_mixer_uses_ema_rsi_not_cat_fuse():
    from flyfx.trader import Indicators

    fuse = Indicators(pip=0.0001, mixer="cat_fuse")
    legacy = Indicators(pip=0.0001, mixer="legacy")
    px = 1.08
    feat_f = feat_l = None
    for i in range(60):
        px += 0.0002
        args = (px + 0.0002, px - 0.0001, px)
        kw = {"volume": 500.0, "open_": px, "stamp": 1_700_000_000 + i * 300}
        feat_f = fuse.update(*args, **kw)
        feat_l = legacy.update(*args, **kw)
    assert feat_f is not None and feat_l is not None
    assert feat_l["impulse"] == feat_l["kalman"]["legacy"]["impulse"]
    assert feat_f["impulse"] == feat_f["kalman"]["cat_fuse"]["impulse"]
    assert "legacy" in feat_f["kalman"]


def test_run_category_tech_keys():
    feat = {
        "categories": {n: {"available": False, "impulse": 0.0, "fused": 0.0, "residual": 0.0} for n in
                       ("trend", "momentum", "volatility", "volume", "breadth", "structure", "sentiment")},
        "adx": 12.0,
        "rsi": 50.0,
        "stoch": 0.5,
        "atr": 0.001,
        "close": 1.08,
        "bar_pos": 0.5,
        "atr_ratio": 1.0,
        "bb_pct": 0.5,
        "bb_bw": 0.03,
        "bb_bw_ma": 0.04,
        "donchian_pct": 0.5,
        "pivot": {},
        "fib": {},
    }
    out = run_category_tech(feat)
    assert set(out) == {
        "trend",
        "momentum",
        "volatility",
        "volume",
        "breadth",
        "structure",
        "sentiment",
    }
    assert out["breadth"]["vote"] == "HOLD"
    assert out["sentiment"]["vote"] == "HOLD"
    assert out["volume"]["vote"] == "HOLD"


def test_parse_ea_dynamic_flag():
    from flyfx.trader import parse_ea_dynamic

    assert parse_ea_dynamic("PONG|EURUSD|123") is None
    assert parse_ea_dynamic("OK|FUSE|all|SYMBOL|EURUSD") is None
    assert parse_ea_dynamic("OK|FUSE|all|DYNAMIC|1|SYMBOL|EURUSD") is True
    assert parse_ea_dynamic("PONG|EURUSD|123|FUSE|trend|DYNAMIC|0") is False
    assert parse_ea_dynamic("OK|FUSE|all|DYNAMIC|on|SYMBOL|EURUSD") is True


def test_parse_ea_inds_spec():
    from flyfx.sense.category_kalman import parse_fuse_inds
    from flyfx.trader import parse_ea_inds_spec

    assert parse_ea_inds_spec("PONG|EURUSD|123") is None
    assert parse_ea_inds_spec("OK|FUSE|all|DYNAMIC|0|INDS|all|SYMBOL|EURUSD") == "all"
    assert parse_ea_inds_spec("PONG|EURUSD|1|FUSE|all|DYNAMIC|0|INDS|ema,macd,rsi") == "ema,macd,rsi"
    blob = parse_fuse_inds(parse_ea_inds_spec("OK|INDS|ema,rsi"))
    assert blob["trend"] == ("ema",)
    assert blob["momentum"] == ("rsi",)
    assert blob["volatility"] == ()


def _blob(**fused):
    from flyfx.sense.category_kalman import CATEGORY_NAMES

    out = {n: {"available": False, "fused": 0.0} for n in CATEGORY_NAMES}
    for name, val in fused.items():
        out[name] = {"available": True, "fused": float(val)}
    return out


def test_fuse_regime_states_and_allowlist():
    from flyfx.sense.fuse_regime import FuseRegimeMachine

    m = FuseRegimeMachine(hold_bars=1)
    trend_raw = {
        "adx": 30.0,
        "rsi": 50.0,
        "atr_ratio": 1.0,
        "bb_bw": 0.02,
        "bb_bw_ma": 0.02,
        "regime": "UP",
    }
    cats = _blob(trend=0.80, structure=0.30, momentum=0.10)
    plan = m.step(trend_raw, cats)
    assert plan["state"] == "TREND"
    assert plan["include"] == ["trend", "structure"]
    assert "volume" not in plan["include"]

    cats["volume"] = {"available": False, "fused": 9.0}
    plan = m.step(trend_raw, cats)
    assert "volume" not in plan["include"]

    cats["volume"] = {"available": True, "fused": 0.40}
    plan = m.step(trend_raw, cats)
    assert plan["include"] == ["trend", "structure", "volume"]

    clipped = m.step(trend_raw, cats, allow=("trend",))
    assert clipped["include"] == ["trend"]

    m2 = FuseRegimeMachine(hold_bars=1)
    range_raw = {
        "adx": 12.0,
        "rsi": 24.0,
        "atr_ratio": 1.0,
        "bb_bw": 0.02,
        "bb_bw_ma": 0.02,
        "regime": "CHOP",
    }
    plan = m2.step(range_raw, _blob(trend=0.05, momentum=0.40, structure=0.20))
    assert plan["state"] == "RANGE"
    assert plan["include"] == ["momentum", "structure"]

    m3 = FuseRegimeMachine(hold_bars=1)
    squeeze = {
        "adx": 20.0,
        "rsi": 50.0,
        "atr_ratio": 0.90,
        "bb_bw": 0.010,
        "bb_bw_ma": 0.020,
        "regime": "CHOP",
    }
    quiet = _blob(trend=0.05, volatility=0.10)
    for _ in range(3):
        last = m3.step(squeeze, quiet)
    assert last["state"] == "QUIET"
    expand = dict(squeeze)
    expand["atr_ratio"] = 1.30
    expand["bb_bw"] = 0.021
    plan = m3.step(expand, _blob(trend=0.10, volatility=0.40))
    assert plan["state"] == "BREAK"
    assert plan["include"] == ["volatility", "trend"]


def test_fuse_regime_hysteresis_holds_then_switches():
    from flyfx.sense.fuse_regime import FuseRegimeMachine

    m = FuseRegimeMachine(hold_bars=4)
    cats = _blob(trend=0.80, momentum=0.50, structure=0.30)
    trend_raw = {
        "adx": 30.0,
        "rsi": 50.0,
        "atr_ratio": 1.0,
        "bb_bw": 0.02,
        "bb_bw_ma": 0.02,
        "regime": "UP",
    }
    range_raw = {
        "adx": 10.0,
        "rsi": 22.0,
        "atr_ratio": 1.0,
        "bb_bw": 0.02,
        "bb_bw_ma": 0.02,
        "regime": "CHOP",
    }
    assert m.step(trend_raw, cats)["state"] == "TREND"
    held = m.step(range_raw, cats)
    assert held["state"] == "TREND"
    assert "hold" in held["reason"]
    m.step(range_raw, cats)
    m.step(range_raw, cats)
    switched = m.step(range_raw, cats)
    assert switched["state"] == "RANGE"


def test_indicators_fuse_dynamic_picks_and_legacy_ignores():
    from flyfx.trader import Indicators

    dyn = Indicators(pip=0.0001)
    dyn.fuse_dynamic = True
    dyn.fuse_allow = ("trend", "structure", "momentum", "volatility")
    px = 1.08
    feat = None
    for i in range(80):
        px += 0.00025
        feat = dyn.update(
            px + 0.0003,
            px - 0.00005,
            px,
            volume=500.0,
            open_=px - 0.0001,
            stamp=1_700_000_000 + i * 300,
        )
    assert feat is not None
    assert feat["fuse_plan"]["dynamic"] is True
    assert feat["fuse_plan"]["state"] in ("TREND", "RANGE", "BREAK", "QUIET")
    assert feat["impulse"] == feat["kalman"]["cat_fuse"]["impulse"]
    assert set(feat["kalman"]["cat_fuse"]["selected"]).issubset(
        {"trend", "structure", "momentum", "volatility", "volume"}
    )

    allow = Indicators(pip=0.0001)
    allow.fuse_dynamic = True
    allow.fuse_allow = ("momentum",)
    px = 1.08
    feat_a = None
    for i in range(40):
        px += 0.0002
        feat_a = allow.update(px + 0.0002, px - 0.0001, px, volume=500.0, open_=px, stamp=1_700_000_000 + i * 300)
    assert feat_a["fuse_plan"]["include"] == ["momentum"]
    assert feat_a["kalman"]["cat_fuse"]["selected"] == ["momentum"]

    leg = Indicators(pip=0.0001, mixer="legacy")
    leg.fuse_dynamic = True
    px = 1.08
    feat_l = None
    for i in range(40):
        px += 0.0002
        feat_l = leg.update(px + 0.0002, px - 0.0001, px, volume=500.0, open_=px, stamp=1_700_000_000 + i * 300)
    assert feat_l["fuse_plan"]["dynamic"] is False
    assert feat_l["impulse"] == feat_l["kalman"]["legacy"]["impulse"]


def test_parse_fuse_inds_flat_and_grouped():
    from flyfx.sense.category_kalman import CATEGORY_CHANNELS, encode_fuse_inds, parse_fuse_inds

    full = parse_fuse_inds("")
    assert full["trend"] == CATEGORY_CHANNELS["trend"]
    assert encode_fuse_inds(full) == "all"
    assert parse_fuse_inds("none")["trend"] == ()
    assert encode_fuse_inds(parse_fuse_inds("none")) == "none"

    flat = parse_fuse_inds("ema,macd,rsi")
    assert flat["trend"] == ("ema", "macd")
    assert flat["momentum"] == ("rsi",)
    assert flat["volatility"] == ()
    assert encode_fuse_inds(flat) == "ema,macd,rsi"

    grouped = parse_fuse_inds("trend:ema+macd;momentum:rsi")
    assert grouped["trend"] == ("ema", "macd")
    assert grouped["momentum"] == ("rsi",)
    assert grouped["volatility"] == CATEGORY_CHANNELS["volatility"]

    by_cat = parse_fuse_inds("trend")
    assert by_cat["trend"] == CATEGORY_CHANNELS["trend"]
    assert by_cat["momentum"] == ()


def test_category_kalman_include_skips_unselected_live_channels():
    kf = CategoryKalman("momentum", ("rsi", "stoch", "cci"))
    a = kf.update({"rsi": 1.2, "stoch": 1.1, "cci": 0.9}, include=("rsi",))
    assert a["available"] is True
    assert a["n_live"] == 1
    assert a["channels"] == ["rsi"]
    assert list(a["parts"].keys()) == ["rsi"]
    b = kf.update({"rsi": 1.0, "stoch": 1.1, "cci": 0.9}, include=("rsi", "stoch"))
    assert b["n_live"] == 2
    assert set(b["parts"]) == {"rsi", "stoch"}


def test_indicators_fuse_inds_ema_only_drops_other_channels():
    from flyfx.sense.category_kalman import parse_fuse_inds
    from flyfx.trader import Indicators

    ind = Indicators(pip=0.0001)
    ind.fuse_inds = parse_fuse_inds("ema")
    px = 1.08
    feat = None
    for i in range(60):
        px += 0.0002
        feat = ind.update(px + 0.0002, px - 0.0001, px, volume=500.0, open_=px, stamp=1_700_000_000 + i * 300)
    assert feat is not None
    trend = feat["categories"]["trend"]
    assert trend["channels"] == ["ema"]
    assert list(trend["parts"].keys()) == ["ema"]
    assert feat["categories"]["momentum"]["available"] is False
    assert feat["fuse_inds"] == "ema"
