"""Timed forecast: bank the predicted price, cut the adverse move early."""

from flyfx.risk.forecast import forecast_exit, project


def test_a_hot_cycle_lifts_the_target_and_a_quiet_one_uses_the_floor():
    hot = project("BUY", 1.1000, 0.0010, roc=0.80, impulse=0.60, risk_px=0.0016, cost_px=0.00010)
    quiet = project("BUY", 1.1000, 0.0010, roc=0.0, impulse=0.0, risk_px=0.0016, cost_px=0.00010)
    assert hot.target > quiet.target > 1.1000
    assert hot.horizon == 12
    assert quiet.drift_atr >= 0.45
    assert hot.drift_atr <= 1.25
    assert hot.damage_px < 0.0016


def test_sell_is_the_buy_forecast_reflected():
    buy = project("BUY", 1.1000, 0.0010, roc=0.40, impulse=0.50, risk_px=0.0020, cost_px=0.00012)
    sell = project("SELL", 1.1000, 0.0010, roc=-0.40, impulse=-0.50, risk_px=0.0020, cost_px=0.00012)
    assert abs((buy.target - 1.1000) - (1.1000 - sell.target)) < 1e-12
    assert buy.damage_px == sell.damage_px


def test_jpy_distance_matches_the_eur_distance_in_atr():
    eur = project("BUY", 1.1000, 0.0010, roc=0.50, impulse=0.40, risk_px=0.0020, cost_px=0.00010)
    jpy = project("BUY", 150.00, 0.10, roc=0.50, impulse=0.40, risk_px=0.20, cost_px=0.010)
    assert abs(eur.drift_atr - jpy.drift_atr) < 1e-9
    assert abs((eur.target - 1.1000) / 0.0010 - (jpy.target - 150.00) / 0.10) < 1e-6


def test_hit_banks_profit_inside_the_window_only():
    fc = project("BUY", 1.1000, 0.0010, roc=0.20, impulse=0.30, risk_px=0.0020, cost_px=0.00010)
    assert forecast_exit(fc, "BUY", fc.target, unreal=0.00040, bars_held=6, cost_px=0.00010) == "prediction hit"
    assert forecast_exit(fc, "BUY", fc.target, unreal=0.00005, bars_held=6, cost_px=0.00010) is None
    assert forecast_exit(fc, "BUY", fc.target, unreal=0.00040, bars_held=13, cost_px=0.00010) is None
    assert forecast_exit(fc, "BUY", fc.target - 0.0001, unreal=0.00040, bars_held=6, cost_px=0.00010) is None


def test_adverse_move_cuts_before_the_full_stop():
    fc = project("BUY", 1.1000, 0.0010, roc=0.20, impulse=0.30, risk_px=0.0020, cost_px=0.00010)
    assert fc.damage_px <= 0.55 * 0.0020
    assert forecast_exit(fc, "BUY", 1.0990, unreal=-fc.damage_px, bars_held=4, cost_px=0.00010) == "prediction fail"
    assert forecast_exit(fc, "BUY", 1.0996, unreal=-0.00020, bars_held=4, cost_px=0.00010) is None
    assert forecast_exit(fc, "BUY", 1.0990, unreal=-fc.damage_px, bars_held=1, cost_px=0.00010) is None
    sell = project("SELL", 1.1000, 0.0010, roc=-0.20, impulse=-0.30, risk_px=0.0020, cost_px=0.00010)
    assert (
        forecast_exit(sell, "SELL", sell.target, unreal=0.00040, bars_held=5, cost_px=0.00010)
        == "prediction hit"
    )
