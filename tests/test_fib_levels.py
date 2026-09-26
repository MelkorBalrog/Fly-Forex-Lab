"""Trend entries at Fibonacci retracements of the impulse swing."""

from flyfx.risk.bayes_sizer import AdaptiveParams
from flyfx.sense.fib_levels import held_level, impulse_swing, retracement_prices
from flyfx.trader import PullbackSetup, committee
from flyfx.vote.category_tech import tech_trend


def test_up_impulse_is_low_then_high():
    highs = [1.0, 1.02, 1.05, 1.08, 1.10, 1.07, 1.04, 1.05]
    lows = [0.99, 1.00, 1.03, 1.06, 1.08, 1.05, 1.02, 1.03]
    hi, lo, up = impulse_swing(highs, lows)
    assert up
    assert hi == 1.10
    assert lo == 0.99


def test_down_impulse_is_high_then_low():
    highs = [1.02, 1.08, 1.10, 1.06, 1.03, 1.01, 1.00, 1.01]
    lows = [1.00, 1.05, 1.07, 1.02, 0.99, 0.96, 0.94, 0.95]
    hi, lo, up = impulse_swing(highs, lows)
    assert not up
    assert hi == 1.10
    assert lo == 0.94


def test_retracement_prices_from_the_extreme():
    levels = retracement_prices(1.10, 1.00, True)
    assert abs(levels["0.382"] - 1.0618) < 1e-9
    assert abs(levels["0.500"] - 1.0500) < 1e-9
    assert abs(levels["0.618"] - 1.0382) < 1e-9
    assert abs(levels["0.786"] - 1.0214) < 1e-9
    down = retracement_prices(1.10, 1.00, False)
    assert abs(down["0.618"] - 1.0618) < 1e-9


def test_held_618_in_an_uptrend():
    name = held_level(1.042, 1.037, 1.040, 0.002, 1.10, 1.00, True)
    assert name == "0.618"


def test_close_through_the_level_is_not_a_hold():
    name = held_level(1.042, 1.030, 1.031, 0.002, 1.10, 1.00, True)
    assert name == ""


def _feat(**extra):
    base = {
        "ready": True,
        "regime": "UP",
        "rsi": 55.0,
        "impulse": 0.10,
        "fib": {"near": "0.618", "swing_up": True},
    }
    base.update(extra)
    return base


def test_uptrend_buys_a_held_fib_level():
    setup = PullbackSetup()
    side, kind = setup.decide(_feat(), 1.040, 10, AdaptiveParams())
    assert side == "BUY"
    assert kind == "fib-0.618"


def test_fib_does_not_fade_the_trend():
    setup = PullbackSetup()
    side, kind = setup.decide(
        _feat(regime="UP", fib={"near": "0.618", "swing_up": False}),
        1.040,
        10,
        AdaptiveParams(),
    )
    assert side == "HOLD"
    assert kind == "hold"


def test_downtrend_sells_a_held_fib_level():
    setup = PullbackSetup()
    side, kind = setup.decide(
        _feat(regime="DOWN", rsi=40.0, fib={"near": "0.500", "swing_up": False}),
        1.050,
        10,
        AdaptiveParams(),
    )
    assert side == "SELL"
    assert kind == "fib-0.500"


def test_fib_checkbox_off_skips_the_level():
    setup = PullbackSetup()
    side, kind = setup.decide(_feat(fib_trade=False), 1.040, 10, AdaptiveParams())
    assert side == "HOLD"
    assert kind == "hold"
    vote = tech_trend(
        {
            "adx": 28.0,
            "regime": "UP",
            "bar_pos": 0.70,
            "fib_trade": False,
            "fib": {"near": "0.618", "swing_up": True},
            "categories": {"trend": {"impulse": 0.0, "fused": 0.0, "regime": "UP"}},
        }
    )
    assert vote["vote"] == "HOLD"


def test_chop_does_not_trade_fib():
    setup = PullbackSetup()
    side, kind = setup.decide(_feat(regime="CHOP"), 1.040, 10, AdaptiveParams())
    assert side == "HOLD"


def test_fib_enters_on_one_vote_unless_a_fly_vetoes():
    side, tag, n = committee("BUY", "fib-0.618", "HOLD", "HOLD", "HOLD")
    assert side == "BUY"
    assert tag == "1oo3-fib"
    assert n == 1
    blocked, why, _n = committee("BUY", "fib-0.618", "SELL", "HOLD", "HOLD")
    assert blocked == "HOLD"
    assert "veto" in why


def test_trend_node_votes_the_fib_when_follow_is_quiet():
    vote = tech_trend(
        {
            "adx": 28.0,
            "regime": "UP",
            "bar_pos": 0.70,
            "fib": {"near": "0.618", "swing_up": True},
            "categories": {"trend": {"impulse": 0.0, "fused": 0.0, "regime": "UP"}},
        }
    )
    assert vote["vote"] == "BUY"
    assert vote["reason"] == "trend-fib-0.618"
