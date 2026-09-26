"""Day bias and the 2R payoff stop."""

from flyfx.exec.account_sim import AccountSim
from flyfx.risk.payoff import (
    allows_side,
    bias_trade,
    cost_ratio,
    payoff_stop,
    tape_bias,
    usd_to_price,
)


def test_bias_trade_sells_a_down_day_and_refuses_a_fight():
    assert bias_trade("DOWN", "SELL", "HOLD", "SELL", -0.50) == ("SELL", "bias-down")
    assert bias_trade("DOWN", "BUY", "BUY", "SELL", -0.50) == ("HOLD", "")
    assert bias_trade("UP", "BUY", "HOLD", "HOLD", 0.50) == ("BUY", "bias-up")
    assert bias_trade("DOWN", "SELL", "HOLD", "SELL", -0.10) == ("HOLD", "")
    assert bias_trade("FLAT", "SELL", "SELL", "SELL", -0.50) == ("HOLD", "")


def test_bias_needs_a_day_and_then_picks_a_side():
    assert tape_bias(1.10, 1.09, 0.001, bars_seen=100) == "FLAT"
    assert tape_bias(1.10, 1.09, 0.001, bars_seen=288) == "UP"
    assert tape_bias(1.08, 1.10, 0.001, bars_seen=288) == "DOWN"
    assert tape_bias(1.1001, 1.10, 0.001, bars_seen=288) == "FLAT"
    assert allows_side("BUY", "UP")
    assert not allows_side("BUY", "DOWN")
    # FLAT = no day filter yet (freeze / early window still trades).
    assert allows_side("SELL", "FLAT")
    assert allows_side("BUY", "FLAT")
    assert allows_side("SELL", "DOWN")


def test_payoff_stop_waits_for_two_r():
    assert payoff_stop("BUY", 1.10, 1.09, unreal=0.004, risk_px=0.010, extreme=1.104) == 1.09
    assert payoff_stop("BUY", 1.10, 1.09, unreal=0.010, risk_px=0.010, extreme=1.110) == 1.10
    trailed = payoff_stop("BUY", 1.10, 1.09, unreal=0.020, risk_px=0.010, extreme=1.125)
    assert abs(trailed - 1.120) < 1e-9
    kept = payoff_stop("BUY", 1.10, 1.121, unreal=0.020, risk_px=0.010, extreme=1.125)
    assert kept == 1.121
    sell = payoff_stop("SELL", 1.10, 1.11, unreal=0.020, risk_px=0.010, extreme=1.075)
    assert abs(sell - 1.080) < 1e-9


def test_cost_ratio_is_scale_free():
    eur = cost_ratio(0.00010, 0.00002, 0.0010, 0.00007)
    jpy = cost_ratio(0.010, 0.002, 0.100, 0.007)
    assert abs(eur - jpy) < 1e-9
    assert abs(eur - 0.21) < 1e-9
    assert cost_ratio(0.00020, 0.00005, 0.0010, 0.00007) > 0.22
    account = AccountSim()
    px = usd_to_price(account.to_usd, 7.0, 1.16)
    assert abs(px - 7.0 / 100_000.0) < 1e-9
