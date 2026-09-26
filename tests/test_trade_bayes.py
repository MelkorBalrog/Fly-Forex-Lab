"""Condition-aware Bayesian trade risk from sealed history."""

from __future__ import annotations

from flyfx.exec.account_sim import AccountSim
from flyfx.risk.trade_bayes import TradeBayesRisk, fingerprint, trade_bayes_path


def _feat(**kw):
    base = {
        "regime": "UP",
        "atr_ratio": 1.0,
        "impulse": 0.6,
        "rsi": 48.0,
        "bb_bw": 0.010,
        "bb_bw_ma": 0.010,
        "kalman": {},
    }
    base.update(kw)
    return base


def test_fingerprint_bins():
    fp = fingerprint(
        side="BUY",
        feat=_feat(regime="UP", atr_ratio=0.5, impulse=0.9, bb_bw=0.007, bb_bw_ma=0.010),
        stamp=1_700_000_000,  # week day
        consec_losses=2,
        tag="2oo3",
    )
    assert fp["side"] == "BUY"
    assert fp["vol"] == "dead"
    assert fp["impulse"] == "strong"
    assert fp["bb"] == "squeeze"
    assert fp["heat"] == "hot"
    assert fp["tag"] == "maj"


def test_losses_inhibit_similar_setup():
    tb = TradeBayesRisk(enabled=True)
    feat = _feat(regime="CHOP", atr_ratio=1.8, impulse=0.3)
    for _ in range(8):
        tb.remember(
            {
                "side": "BUY",
                "usd": -100.0,
                "r_mult": 1.0,
                "mkt": fingerprint(side="BUY", feat=feat, stamp=0, consec_losses=0, tag="2oo3"),
                "rsi": 50.0,
            },
            feat=feat,
        )
    plan = tb.infer(side="BUY", feat=feat, stamp=0, consec_losses=0, tag="2oo3")
    assert plan.inhibit or plan.size_mult < 0.75
    assert plan.p_win < 0.45


def test_wins_raise_size_and_allow_add():
    tb = TradeBayesRisk(enabled=True)
    feat = _feat(regime="UP", atr_ratio=1.0, impulse=0.65)
    for _ in range(10):
        tb.remember(
            {
                "side": "BUY",
                "usd": 200.0,
                "r_mult": 1.5,
                "mkt": fingerprint(side="BUY", feat=feat, stamp=0, consec_losses=0, tag="3oo3"),
                "rsi": 46.0,
            },
            feat=feat,
        )
    plan = tb.infer(side="BUY", feat=feat, stamp=0, consec_losses=0, tag="3oo3")
    assert not plan.inhibit
    assert plan.size_mult >= 0.9
    assert plan.p_win > 0.55
    tb.mark_entry(plan)
    live = tb.infer(
        side="BUY",
        feat=feat,
        stamp=0,
        consec_losses=0,
        tag="3oo3",
        unreal_r=0.8,
        open_position=True,
    )
    assert live.add_ok
    assert live.add_mult >= 1.0


def test_adverse_turn_trims_open_winner():
    tb = TradeBayesRisk(enabled=True)
    good = _feat(regime="UP", atr_ratio=1.0, impulse=0.7)
    bad = _feat(regime="DOWN", atr_ratio=1.7, impulse=0.2, bb_bw=0.014, bb_bw_ma=0.010)
    for _ in range(6):
        tb.remember(
            {
                "side": "BUY",
                "usd": 150.0,
                "r_mult": 1.2,
                "mkt": fingerprint(side="BUY", feat=good, tag="2oo3"),
            },
            feat=good,
        )
    for _ in range(8):
        tb.remember(
            {
                "side": "BUY",
                "usd": -120.0,
                "r_mult": 1.0,
                "mkt": fingerprint(side="BUY", feat=bad, tag="2oo3"),
            },
            feat=bad,
        )
    entry = tb.infer(side="BUY", feat=good, tag="2oo3")
    tb.mark_entry(entry)
    live = tb.infer(
        side="BUY",
        feat=bad,
        tag="2oo3",
        unreal_r=0.6,
        open_position=True,
    )
    assert live.trim_frac >= 0.4
    assert not live.add_ok


def test_account_reduce_partial():
    acct = AccountSim(balance=100_000.0, leverage=100.0, spread=0.00010)
    bid, ask = 1.10000, 1.10010
    fill = acct.open("BUY", 1.0, bid, ask, 0.0, 1_700_000_000)
    assert fill.ok
    # Move price up so reduce books a win.
    bid2, ask2 = 1.10100, 1.10110
    red = acct.reduce(0.4, bid2, ask2, 0.0, 1_700_000_300, "trim")
    assert red.ok
    assert abs(red.lots - 0.4) < 1e-9
    assert acct.pos is not None
    assert abs(acct.pos.lots - 0.6) < 1e-9
    assert red.net > 0


def test_disabled_is_noop():
    tb = TradeBayesRisk(enabled=False)
    tb.remember({"side": "BUY", "usd": -50, "r_mult": 1.0})
    plan = tb.infer(side="BUY", feat=_feat())
    assert not plan.inhibit
    assert plan.size_mult == 1.0
    assert plan.trim_frac == 0.0


def test_save_and_reload_continues_inference(tmp_path):
    path = tmp_path / "trade_bayes_EURUSD.json"
    tb = TradeBayesRisk(enabled=True)
    feat = _feat(regime="UP")
    for _ in range(5):
        tb.remember(
            {
                "side": "BUY",
                "usd": 100.0,
                "r_mult": 1.0,
                "mkt": fingerprint(side="BUY", feat=feat, tag="2oo3"),
            },
            feat=feat,
        )
    tb.save_file(path)
    assert path.exists()

    tb2 = TradeBayesRisk(enabled=True)
    assert tb2.load_file(path)
    assert int(tb2.global_post.n) == 5
    assert len(tb2.bins) >= 1
    assert len(tb2.memory) == 5
    plan = tb2.infer(side="BUY", feat=feat, tag="2oo3")
    assert plan.p_win > 0.5


def test_trade_bayes_path():
    p = trade_bayes_path("C:/proj", "eurusd")
    assert p.name == "trade_bayes_EURUSD.json"
    assert "reports" in str(p).replace("\\", "/")
