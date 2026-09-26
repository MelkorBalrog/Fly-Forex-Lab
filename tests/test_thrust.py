"""Cycle thrust: same jump, same size, on every pair."""

from flyfx.exec.account_sim import AccountSim
from flyfx.risk.thrust import CycleTape, roc_flip, thrust_add_frac, thrust_mult
from flyfx.trader import size_lots


def _tape(closes, atr=0.0010) -> CycleTape:
    tape = CycleTape()
    for c in closes:
        tape.push(c, atr)
    return tape


def test_quiet_cycle_does_not_press():
    assert thrust_mult("BUY", 0.05, 0.05) == 1.0
    assert thrust_mult("SELL", -0.05, -0.05) == 1.0
    assert thrust_add_frac("BUY", 0.80, 0.20) == 0.0


def test_acceleration_scales_both_sides_the_same():
    buy = thrust_mult("BUY", 0.90, 0.80)
    sell = thrust_mult("SELL", -0.90, -0.80)
    assert buy == sell
    assert 1.4 < buy < 1.6
    assert thrust_mult("BUY", 0.90, 3.0) == 2.0
    add_buy = thrust_add_frac("BUY", 0.90, 1.20)
    add_sell = thrust_add_frac("SELL", -0.90, -1.20)
    assert add_buy == add_sell
    assert 0.50 < add_buy <= 0.80


def test_flip_banks_a_winner_and_leaves_a_loser():
    assert roc_flip("BUY", -0.40, -0.70, unreal=0.0010, cost_px=0.0002)
    assert roc_flip("SELL", 0.40, 0.70, unreal=0.0010, cost_px=0.0002)
    assert not roc_flip("BUY", -0.40, -0.70, unreal=-0.0004, cost_px=0.0002)
    assert not roc_flip("BUY", 0.40, -0.70, unreal=0.0010, cost_px=0.0002)
    assert not roc_flip("BUY", -0.40, -0.05, unreal=0.0010, cost_px=0.0002)


def test_cycle_jump_is_the_change_in_rate():
    flat = [1.1000 + i * 0.0001 for i in range(8)]
    tape = _tape(flat)
    slow = tape.roc
    for c in (1.1010, 1.1022, 1.1036, 1.1052, 1.1070, 1.1090):
        tape.push(c, 0.0010)
    assert tape.roc > slow
    assert tape.jump > 0.3


def test_size_lots_presses_with_the_cycle_and_stays_capped():
    account = AccountSim()
    common = dict(
        account=account,
        equity=100_000.0,
        free=100_000.0,
        price=1.16,
        atr=0.0010,
        risk_pct=1.0,
        fixed_lots=0.01,
        min_lots=0.01,
        max_lots=20.0,
        margin_cap_pct=25.0,
        sl_atr=2.4,
        cond_mult=1.0,
    )
    base, _note = size_lots(**common, thrust_mult=1.0)
    pressed, _note = size_lots(**common, thrust_mult=1.5)
    capped, _note = size_lots(**common, thrust_mult=4.0)
    assert pressed > base
    assert capped >= pressed
    assert capped < base * 2.05
