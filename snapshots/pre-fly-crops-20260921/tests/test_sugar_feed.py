"""Sugar GRNs + satiety tank: eat / don't eat, never BUY/SELL."""

from flyfx.brain.sugar_feed import (
    SugarDoseMachine,
    compute_satiety,
    feeding_from_raw,
    food_nutrition,
    mix_appetite,
    native_lamp_pack,
    parse_sugar_amt,
    propose_sugar_state,
)
from flyfx.trader import _sugar_enabled, parse_args, parse_ea_sugar, parse_ea_sugar_amt, parse_ea_sugar_dyn


def test_food_nutrition_3oo3_sweeter_than_hold():
    sweet = food_nutrition("3oo3+flow", banc_mult=1.0, uncertainty=0.2, agreement=0.8, n_agree=3)
    bland = food_nutrition("HOLD", banc_mult=0.5, uncertainty=0.9, agreement=0.1, n_agree=0)
    assert sweet > bland
    assert sweet > 0.7
    assert bland < 0.4


def test_hungry_and_sweet_forages():
    sat = compute_satiety(
        equity=98_000,
        peak_eq=100_000,
        start=100_000,
        day_start_eq=100_000,
        floating=0.0,
        day_trades=0,
        day_trade_cap=3,
    )
    assert sat < 0.35
    out = mix_appetite(0.82, sat, tag="2oo3")
    assert out["skip"] is False
    assert out["snowball"] is True
    assert out["size_mult"] > 1.0
    assert out["flatten"] is False


def test_full_book_skips_weak_food():
    sat = compute_satiety(
        equity=101_600,
        peak_eq=101_600,
        start=100_000,
        day_start_eq=100_000,
        floating=700.0,
        day_trades=3,
        day_trade_cap=3,
    )
    assert sat >= 0.80
    out = mix_appetite(0.40, sat, tag="2oo3")
    assert out["skip"] is True
    assert out["snowball"] is False


def test_full_winner_spit_out():
    out = mix_appetite(
        0.70,
        0.90,
        tag="2oo3",
        bars_held=8,
        min_hold=4,
        unreal=0.001,
        atr=0.0008,
    )
    assert out["flatten"] is True
    assert out["reason"] == "sugar full"


def test_3oo3_feast_not_skipped_for_low_appetite():
    out = mix_appetite(0.20, 0.55, tag="3oo3")
    assert out["skip"] is False


def test_feeding_from_raw_rest_is_half():
    assert abs(feeding_from_raw(0.2, 0.2) - 0.5) < 1e-9
    assert feeding_from_raw(0.5, 0.2) > 0.5
    assert feeding_from_raw(0.05, 0.2) < 0.5


def test_vote_sugar_does_not_run_full_spmv():
    import numpy as np

    from flyfx.trader import FlySwarm

    class _Boom:
        def dot(self, x):
            raise AssertionError("sugar must not SPMV the 166k graph")

    class _Core:
        n = 40
        sugar = np.arange(8, 16, dtype=np.int32)
        ol_sensory = np.array([], dtype=np.int32)
        vnc_sensory = np.array([], dtype=np.int32)
        descending = np.arange(4, dtype=np.int32)
        spmv = _Boom()

        def step_state(self, *args, **kwargs):
            raise AssertionError("sugar must not step W")

    swarm = FlySwarm(_Core(), plastic=False)
    out = swarm.vote_sugar(0.80)
    assert out["n"] == 8
    assert out["feeding"] > 0.5
    assert float(np.mean(np.abs(swarm.sugar_x[8:16]))) > 0.0


def test_parse_ea_sugar():
    assert parse_ea_sugar("PONG|EURUSD|1") is None
    assert parse_ea_sugar("OK|FUSE|all|SUGAR|1|SYMBOL|EURUSD") is True
    assert parse_ea_sugar("PONG|EURUSD|1|FUSE|all|DYNAMIC|0|INDS|all|RISK|balanced|SUGAR|0") is False


def test_cli_sugar_off_by_default(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "argv", ["fly_forex.py", "--replay"])
    args = parse_args()
    assert args.sugar is False
    assert args.no_sugar is False
    assert _sugar_enabled(args) is False
    assert parse_sugar_amt(args.sugar_amt) == 1.0
    monkeypatch.setattr(sys, "argv", ["fly_forex.py", "--replay", "--sugar"])
    on = parse_args()
    assert on.sugar is True
    assert _sugar_enabled(on) is True
    monkeypatch.setattr(sys, "argv", ["fly_forex.py", "--replay", "--no-sugar"])
    off = parse_args()
    assert off.no_sugar is True
    assert _sugar_enabled(off) is False
    monkeypatch.setattr(sys, "argv", ["fly_forex.py", "--replay", "--sugar", "--no-sugar"])
    both = parse_args()
    assert _sugar_enabled(both) is False
    monkeypatch.setattr(sys, "argv", ["fly_forex.py", "--replay", "--sugar-amt", "0.5"])
    half = parse_args()
    assert parse_sugar_amt(half.sugar_amt) == 0.5
    assert half.sugar_dyn is False
    monkeypatch.setattr(sys, "argv", ["fly_forex.py", "--replay", "--sugar-dyn"])
    dyn = parse_args()
    assert dyn.sugar_dyn is True


def test_parse_sugar_amt_scales_and_percent():
    assert parse_sugar_amt(None) == 1.0
    assert parse_sugar_amt(1) == 1.0
    assert parse_sugar_amt(0) == 0.0
    assert parse_sugar_amt(0.5) == 0.5
    assert parse_sugar_amt(2) == 2.0
    assert parse_sugar_amt(2.5) == 2.0
    assert parse_sugar_amt(10) == 0.1
    assert parse_sugar_amt(50) == 0.5
    assert parse_sugar_amt(100) == 1.0
    assert parse_sugar_amt(200) == 2.0
    assert parse_sugar_amt(-1) == 0.0


def test_native_lamp_eats_every_2oo3():
    pack = native_lamp_pack(0.40, 0.20)
    assert pack["reason"] == "native lamp"
    assert pack["skip"] is False
    assert pack["flatten"] is False
    assert pack["snowball"] is True
    assert abs(pack["size_mult"] - 1.0) < 1e-9
    assert abs(pack["appetite"] - 1.0) < 1e-9
    assert 0.0 <= pack["feeding"] <= 1.0
    stuffed = mix_appetite(0.40, 0.95, tag="2oo3", dose=1.0)
    assert stuffed["skip"] is True or stuffed["flatten"] is True or stuffed["size_mult"] < 1.0
    stuffed = mix_appetite(0.40, 0.95, tag="2oo3", dose=0.0)
    assert stuffed["skip"] is False
    assert stuffed["flatten"] is False
    assert stuffed["snowball"] is True
    assert stuffed["size_mult"] == 1.0
    assert stuffed["dose"] == 0.0


def test_dose_one_keeps_full_policy():
    full = mix_appetite(0.40, 0.90, tag="2oo3", dose=1.0)
    assert full["skip"] is True
    assert full["dose"] == 1.0
    none = mix_appetite(0.40, 0.90, tag="2oo3")
    assert none["skip"] is True
    assert none["size_mult"] == full["size_mult"]


def test_dose_half_lerps_size():
    full = mix_appetite(0.82, 0.20, tag="2oo3", dose=1.0)
    half = mix_appetite(0.82, 0.20, tag="2oo3", dose=0.5)
    mid = 1.0 + 0.5 * (full["size_mult"] - 1.0)
    assert abs(half["size_mult"] - mid) < 6e-5
    extra = mix_appetite(0.82, 0.20, tag="2oo3", dose=2.0)
    assert extra["size_mult"] > full["size_mult"]


def test_parse_ea_sugar_amt():
    assert parse_ea_sugar_amt("PONG|EURUSD|1") is None
    assert parse_ea_sugar_amt("OK|FUSE|all|SUGAR|1|SAMT|0.50|SYMBOL|EURUSD") == 0.5
    assert parse_ea_sugar_amt(
        "PONG|EURUSD|1|FUSE|all|DYNAMIC|0|INDS|all|RISK|balanced|SUGAR|1|SAMT|1.00"
    ) == 1.0


def _cond(**kw):
    base = {
        "tag": "2oo3",
        "n_agree": 2,
        "nutrition": 0.70,
        "satiety": 0.50,
        "banc_mult": 1.0,
        "uncertainty": 0.30,
        "agreement": 0.60,
        "regime": "UP",
        "atr_ratio": 1.0,
        "in_trade": False,
        "rails_kill": False,
    }
    base.update(kw)
    return base


def test_propose_feast_hungry_3oo3():
    st, why = propose_sugar_state(_cond(tag="3oo3", n_agree=3, satiety=0.20, uncertainty=0.20))
    assert st == "FEAST"
    assert "3oo3" in why or "hungry" in why


def test_propose_forage_mid_2oo3():
    st, _why = propose_sugar_state(_cond())
    assert st == "FORAGE"


def test_propose_forage_when_stuffed_keeps_policy():
    st, why = propose_sugar_state(_cond(satiety=0.90, nutrition=0.40))
    assert st == "FORAGE"
    assert "policy" in why or "sat" in why
    pack = mix_appetite(0.40, 0.90, tag="2oo3", dose=1.0)
    assert pack["skip"] is True


def test_propose_nibble_chop():
    st, _why = propose_sugar_state(_cond(tag="HOLD", n_agree=0, nutrition=0.20, regime="CHOP"))
    assert st == "NIBBLE"


def test_propose_fast_banc_threat():
    st, why = propose_sugar_state(_cond(banc_mult=0.50))
    assert st == "FAST"
    assert "BANC" in why


def test_sugar_dose_machine_feast_and_scale():
    m = SugarDoseMachine(hold_bars=1)
    out = m.step(_cond(tag="3oo3", n_agree=3, satiety=0.20, uncertainty=0.20), scale=1.0)
    assert out["state"] == "FEAST"
    assert out["dose"] == 1.70
    extra = m.step(_cond(tag="3oo3", n_agree=3, satiety=0.20, uncertainty=0.20), scale=2.0)
    assert extra["dose"] == 2.0


def test_sugar_dose_machine_hysteresis_and_fast_interrupt():
    m = SugarDoseMachine(hold_bars=4)
    first = m.step(_cond(tag="HOLD", n_agree=0, nutrition=0.20, regime="CHOP"))
    assert first["state"] == "NIBBLE"
    held = m.step(_cond())
    assert held["state"] == "NIBBLE"
    assert "hold" in held["reason"]
    threat = m.step(_cond(banc_mult=0.40))
    assert threat["state"] == "FAST"


def test_parse_ea_sugar_dyn():
    assert parse_ea_sugar_dyn("PONG|EURUSD|1") is None
    assert parse_ea_sugar_dyn("OK|FUSE|all|SUGAR|1|SAMT|1.00|SDYN|1|SYMBOL|EURUSD") is True
    assert parse_ea_sugar_dyn(
        "PONG|EURUSD|1|FUSE|all|DYNAMIC|0|INDS|all|RISK|balanced|SUGAR|1|SAMT|1.00|SDYN|0"
    ) is False
