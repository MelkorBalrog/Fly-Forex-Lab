"""Latest-win budget lift and the close reflex."""

from flyfx.brain.profit_reflex import (
    CLOSE_SCORE,
    ProfitReflex,
    budget_lift,
    pack_features,
    reflex_close,
)


def test_fresh_size_fleet_does_not_raise_the_budget():
    r = ProfitReflex("ZZREFLEX", enabled=True)
    feat = {"impulse": -0.9, "atr_ratio": 1.0, "rsi": 40.0}
    budget, note = r.lift_budget(1_500.0, feat, "SELL")
    assert abs(budget - 1_500.0) < 50.0
    assert r.lift < 1.15
    assert "size" in note


def test_a_strong_impulse_lifts_the_budget_toward_the_account():
    from flyfx.brain.profit_reflex import worth_budget

    assert worth_budget(200.0, 100_000.0, 0.90) == 200.0
    raised = worth_budget(200.0, 100_000.0, 0.96)
    assert raised > 70_000.0
    reflex = ProfitReflex("ZZREFLEX", enabled=True, transfer=False)
    budget, note = reflex.lift_budget(
        200.0,
        {"impulse": -0.96, "atr_ratio": 1.0, "rsi": 40.0},
        "SELL",
        equity=100_000.0,
    )
    assert budget > 70_000.0
    assert "lift" in note


def test_budget_lift_maps_a_positive_score_above_one():
    assert abs(budget_lift(0.0) - 1.0) < 1e-9
    assert abs(budget_lift(-0.4) - 1.0) < 1e-9
    assert abs(budget_lift(1.0) - 2.0) < 1e-9


def test_a_double_of_a_real_gain_is_a_reflex_close():
    why = reflex_close(0.80, 2_000.0, 5_000.0, top="deep8")
    assert why is not None
    assert why.startswith("reflex close")
    assert reflex_close(0.80, 200.0, 500.0) is None
    assert reflex_close(CLOSE_SCORE - 0.05, 2_000.0, 5_000.0) is None
    assert reflex_close(0.90, 2_000.0, 2_400.0) is None


def test_a_fresh_reflex_does_not_flatten_a_two_k_to_three_k_step():
    r = ProfitReflex("ZZREFLEX", enabled=True)
    why = r.consider(
        2_000.0,
        3_200.0,
        {"impulse": -0.8, "rsi": 35.0, "atr_ratio": 1.1},
        1.4,
        side="SELL",
        balance=100_000.0,
    )
    assert why is None
    assert r.close_score < CLOSE_SCORE


def test_retrain_keeps_the_previous_reflex(tmp_path, monkeypatch):
    import flyfx.brain.profit_reflex as mod

    monkeypatch.setattr(mod, "BRAIN_DIR", tmp_path)
    first = mod.ProfitReflex("ZZREFLEX", enabled=True)
    x = pack_features({"impulse": -0.9, "rsi": 32.0, "atr_ratio": 1.0}, side="SELL")
    first._entry_x = x.copy()
    first._live_x = x.copy()
    note = first.on_close(2_000.0, 2_000.0)
    assert "size teach" in note
    assert (tmp_path / "ZZREFLEX.reflex.npz").exists()
    again = mod.ProfitReflex("ZZREFLEX", enabled=True)
    assert again.note == "reflex transfer"
    assert again._size_teacher is not None
    again._entry_x = x.copy()
    again._live_x = x.copy()
    note2 = again.on_close(1_800.0, 2_200.0)
    assert "size transfer" in note2
    assert "replay" in note2


def test_teaching_a_win_moves_the_size_score_up():
    r = ProfitReflex("ZZREFLEX", enabled=True)
    x = pack_features({"impulse": -0.9, "rsi": 32.0, "atr_ratio": 1.0}, side="SELL")
    before, _top, _w = r.size.bma(x)
    for _ in range(12):
        r.size.teach(x, 0.8, lr=0.15)
    after, _top, _w = r.size.bma(x)
    assert after > before + 0.05
