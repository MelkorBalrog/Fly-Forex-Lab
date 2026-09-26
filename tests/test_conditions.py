"""Condition risk is the same function for every pair, and it only cuts size."""

import calendar

from flyfx.exec.account_sim import AccountSim
from flyfx.risk.conditions import (
    condition_scale,
    cost_lock_stop,
    dead_scalp,
)
from flyfx.trader import size_lots

OVERLAP = calendar.timegm((2026, 9, 22, 13, 0, 0))
LONDON = calendar.timegm((2026, 9, 22, 9, 0, 0))
TOKYO = calendar.timegm((2026, 9, 22, 2, 0, 0))


def _sweet(**overrides):
    base = dict(
        spread=0.00008,
        slip=0.00002,
        stop_dist=0.0020,
        atr_ratio=1.0,
        uncert=0.2,
        stamp=OVERLAP,
        impulse=0.60,
        consec_losses=0,
    )
    base.update(overrides)
    return condition_scale(**base)


def test_sweet_tape_is_full_size():
    tape = _sweet()
    assert tape.mult == 1.0
    assert tape.cost == 1.0
    assert tape.vol == 1.0
    assert tape.liquidity == 1.0


def test_same_ratios_match_across_quote_scales():
    eur = condition_scale(
        spread=0.00010, slip=0.00002, stop_dist=0.00120,
        atr_ratio=1.4, uncert=0.2, stamp=LONDON, impulse=0.85, consec_losses=1,
    )
    jpy = condition_scale(
        spread=0.010, slip=0.002, stop_dist=0.120,
        atr_ratio=1.4, uncert=0.2, stamp=LONDON, impulse=0.85, consec_losses=1,
    )
    assert eur.mult == jpy.mult
    assert eur.cost == jpy.cost
    assert eur.mult < 1.0


def test_bad_conditions_only_reduce():
    sweet = _sweet().mult
    assert _sweet(spread=0.00070, slip=0.00010).mult < sweet
    assert _sweet(atr_ratio=1.90).mult < sweet
    assert _sweet(atr_ratio=0.50).mult < sweet
    assert _sweet(stamp=LONDON).mult < sweet
    assert _sweet(stamp=TOKYO).mult < _sweet(stamp=LONDON).mult
    assert _sweet(impulse=0.99).mult < sweet
    assert _sweet(consec_losses=2).mult < _sweet(consec_losses=1).mult < sweet
    assert _sweet(atr_ratio=2.2, consec_losses=3, impulse=0.99).mult >= 0.25
    assert _sweet().mult <= 1.0


def test_cost_lock_parks_at_entry_without_loosening():
    locked = cost_lock_stop("BUY", 1.1000, 1.0980, unreal=0.0010, spread=0.00010, slip=0.00002, atr=0.0008)
    assert locked == 1.1000
    kept = cost_lock_stop("BUY", 1.1000, 1.1015, unreal=0.0010, spread=0.00010, slip=0.00002, atr=0.0008)
    assert kept == 1.1015
    early = cost_lock_stop("BUY", 1.1000, 1.0980, unreal=0.00001, spread=0.00010, slip=0.00002, atr=0.0008)
    assert early == 1.0980
    sell = cost_lock_stop("SELL", 1.1000, 1.1040, unreal=0.0010, spread=0.00010, slip=0.00002, atr=0.0008)
    assert sell == 1.1000


def test_dead_scalp_waits_while_the_push_is_alive():
    assert dead_scalp(6, 0.0, 0.0001, 0.0, regime_with=False, impulse_with=True)
    assert not dead_scalp(6, 0.0, 0.0001, 0.0, regime_with=True, impulse_with=True)
    assert dead_scalp(12, 0.0, 0.0001, 0.0, regime_with=True, impulse_with=True)
    assert not dead_scalp(12, 0.0003, 0.0001, 0.0, regime_with=False, impulse_with=False)


def test_size_lots_uses_the_tape_and_will_not_raise_risk():
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
    )
    full, _note = size_lots(**common, cond_mult=1.0)
    half, _note = size_lots(**common, cond_mult=0.5)
    raised, _note = size_lots(**common, cond_mult=4.0)
    assert full > 1.0
    assert half < full
    assert abs(half / full - 0.5) < 0.03
    assert raised == full
