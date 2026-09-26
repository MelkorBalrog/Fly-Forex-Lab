"""Trade-rate appetite governor."""

from __future__ import annotations

from flyfx.risk.appetite import SOFT_AT, TradeAppetite, parse_trade_rate
from flyfx.trader import committee


def test_parse_trade_rate():
    assert parse_trade_rate(None) is None
    assert parse_trade_rate("auto") is None
    assert parse_trade_rate(0) is None
    assert parse_trade_rate(8) == 8.0
    assert parse_trade_rate(100) == 40.0  # clipped


def test_appetite_warms_then_loosens():
    ap = TradeAppetite(target_per_1000=10.0, window=40, enabled=True)
    # Warm-up: fewer than window/2 bars → appetite 1.0, no soft
    for _ in range(15):
        plan = ap.plan(0.50, 0.75)
        ap.end_bar(False)
    assert abs(plan.appetite - 1.0) < 1e-9
    assert not plan.soft_nn_hold
    assert abs(plan.min_impulse - 0.50) < 1e-9

    # Starve the book: many bars, no entries → appetite rises, floors drop, soft on
    for _ in range(30):
        plan = ap.plan(0.50, 0.75)
        ap.end_bar(False)
    assert plan.appetite > SOFT_AT
    assert plan.soft_nn_hold
    assert plan.min_impulse < 0.50
    assert plan.cont_impulse < 0.75


def test_appetite_tightens_when_hot():
    ap = TradeAppetite(target_per_1000=4.0, window=40, enabled=True)
    for i in range(40):
        plan = ap.plan(0.50, 0.75)
        ap.end_bar(i % 5 == 0)  # ~200/1k, well above target 4
    assert plan.appetite < 1.0
    assert not plan.soft_nn_hold
    assert plan.min_impulse >= 0.50


def test_soft_nn_hold_committee():
    # Strict 3oo4: tech+fly, NN HOLD → skip
    side, tag, n = committee(
        "SELL", "cont", "SELL", "HOLD", "HOLD", nn="HOLD", nn_vote=True
    )
    assert side == "HOLD"
    assert "skip" in tag

    # Soft: same votes pass as 2oo3-soft
    side, tag, n = committee(
        "SELL",
        "cont",
        "SELL",
        "HOLD",
        "HOLD",
        nn="HOLD",
        nn_vote=True,
        soft_nn_hold=True,
    )
    assert side == "SELL"
    assert tag.startswith("2oo3-soft")
    assert n == 2

    # Soft does not help when NN opposes (still only 2 agree, nn doesn't count)
    side, tag, n = committee(
        "SELL",
        "cont",
        "SELL",
        "HOLD",
        "HOLD",
        nn="BUY",
        nn_vote=True,
        soft_nn_hold=True,
    )
    assert side == "HOLD"
