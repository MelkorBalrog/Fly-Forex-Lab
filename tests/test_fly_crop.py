"""Per-fly sugar crops: attribution, mute, mix, results×risk governor."""

from flyfx.brain.fly_crop import (
    FlyCrop,
    FlyCropPool,
    attributed_r,
    credits_from_votes,
    drive_from_results_risk,
    mix_crop_packs,
    mute_votes,
    scaled_r,
)
from flyfx.trader import FlySwarm, committee


def test_attributed_r_agree_oppose_hold():
    assert attributed_r("trend", "BUY", "BUY", 1.2) > 0
    assert attributed_r("fade", "SELL", "BUY", 1.2) < 0
    assert attributed_r("fade", "HOLD", "BUY", 1.2) == 0.0
    assert attributed_r("risk", "HOLD", "BUY", -0.8) < 0


def test_conf_hold_on_loss_is_correct_refuse():
    assert attributed_r("conf", "HOLD", "BUY", -1.0) > 0
    assert attributed_r("conf", "HOLD", "BUY", 1.0) == 0.0
    assert attributed_r("conf", "BUY", "BUY", 1.0) > 0


def test_mute_full_trend_still_allows_tech_fade_2oo3():
    packs = {
        "trend": {"skip": True, "vote": "BUY"},
        "fade": {"skip": False, "vote": "BUY"},
        "conf": {"skip": False, "vote": "BUY"},
    }
    t, f, c = mute_votes("BUY", "BUY", "BUY", packs)
    assert t == "HOLD"
    assert f == "BUY"
    decision, tag, n = committee("BUY", "pullback", t, f, "HOLD")
    assert decision == "BUY"
    assert tag.startswith("2oo3")
    assert n >= 2


def test_both_direction_flies_full_hold_sugar_crops():
    packs = {
        "trend": {"skip": True, "vote": "BUY", "size_mult": 0.7, "satiety": 0.95, "feeding": 0.4, "nutrition": 0.4, "dose": 1.0, "snowball": False, "reason": "stuffed"},
        "fade": {"skip": True, "vote": "BUY", "size_mult": 0.7, "satiety": 0.93, "feeding": 0.4, "nutrition": 0.4, "dose": 1.0, "snowball": False, "reason": "stuffed"},
        "conf": {"skip": False, "vote": "BUY"},
        "risk": {"skip": False, "size_mult": 1.0, "flatten": False, "satiety": 0.5, "feeding": 0.5, "nutrition": 0.5, "dose": 1.0},
    }
    t, f, _c = mute_votes("BUY", "BUY", "BUY", packs)
    decision, _tag, _n = committee("BUY", "continuation", t, f, "HOLD")
    book = mix_crop_packs(decision if decision in ("BUY", "SELL") else "BUY", packs)
    assert t == "HOLD" and f == "HOLD"
    assert decision == "HOLD" or book["skip"] is True
    if book["skip"]:
        assert "crop" in book["reason"] or book["reason"] == "sugar-crops"


def test_risk_skip_haircuts_size_not_side():
    packs = {
        "trend": {"skip": False, "vote": "BUY", "size_mult": 1.1, "satiety": 0.4, "feeding": 0.7, "nutrition": 0.7, "dose": 1.0, "snowball": True, "reason": "forage"},
        "fade": {"skip": False, "vote": "HOLD", "size_mult": 1.0, "satiety": 0.4, "feeding": 0.5, "nutrition": 0.5, "dose": 1.0, "snowball": True, "reason": "forage"},
        "risk": {"skip": True, "size_mult": 1.0, "flatten": False, "satiety": 0.9, "feeding": 0.3, "nutrition": 0.3, "dose": 0.5},
    }
    book = mix_crop_packs("BUY", packs)
    assert book["skip"] is False
    assert book["size_mult"] < 1.1
    assert book["size_mult"] <= 1.1 * 0.55 + 1e-6


def test_drive_kill_fast_tiny_da():
    out = drive_from_results_risk(
        {"n": 4, "pf": 1.4, "hit_rate": 0.6, "last_r": 0.5, "crop_dd": 0.0},
        {"banc_mult": 0.40, "rails_kill": True, "uncertainty": 0.3, "atr_ratio": 1.0, "risk_tol": "balanced"},
    )
    assert out["state"] == "FAST"
    assert out["da_mult"] <= 0.20


def test_drive_cold_clean_nibble_high_da():
    out = drive_from_results_risk(
        {"n": 4, "pf": 0.50, "hit_rate": 0.25, "last_r": -0.4, "crop_dd": 0.01},
        {"banc_mult": 1.0, "rails_kill": False, "uncertainty": 0.25, "atr_ratio": 1.0, "risk_tol": "balanced"},
    )
    assert out["state"] == "NIBBLE"
    assert out["da_mult"] >= 1.20


def test_drive_hot_clean_feast_moderate_da():
    out = drive_from_results_risk(
        {"n": 4, "pf": 1.80, "hit_rate": 0.75, "last_r": 0.6, "crop_dd": 0.0},
        {"banc_mult": 1.05, "rails_kill": False, "uncertainty": 0.20, "atr_ratio": 1.0, "risk_tol": "balanced"},
    )
    assert out["state"] == "FEAST"
    assert 0.50 <= out["da_mult"] <= 0.85


def test_drive_conservative_tilts_down():
    res = {"n": 4, "pf": 1.80, "hit_rate": 0.75, "last_r": 0.6, "crop_dd": 0.0}
    risk = {"banc_mult": 1.05, "rails_kill": False, "uncertainty": 0.20, "atr_ratio": 1.0}
    bal = drive_from_results_risk(res, {**risk, "risk_tol": "balanced"})
    cons = drive_from_results_risk(res, {**risk, "risk_tol": "conservative"})
    assert cons["dose"] < bal["dose"]
    assert cons["da_mult"] < bal["da_mult"]


def test_fade_cold_trend_hot_different_state():
    risk = {"banc_mult": 1.0, "rails_kill": False, "uncertainty": 0.25, "atr_ratio": 1.0, "risk_tol": "balanced"}
    trend = drive_from_results_risk(
        {"n": 4, "pf": 1.80, "hit_rate": 0.75, "last_r": 0.5, "crop_dd": 0.0}, risk
    )
    fade = drive_from_results_risk(
        {"n": 4, "pf": 0.50, "hit_rate": 0.20, "last_r": -0.5, "crop_dd": 0.02}, risk
    )
    assert trend["state"] != fade["state"]
    assert trend["da_mult"] != fade["da_mult"]


def test_opposite_fade_on_win_does_not_get_positive_da():
    import numpy as np

    class _Core:
        n = 40
        sugar = np.arange(8, 16, dtype=np.int32)
        ol_sensory = np.array([], dtype=np.int32)
        vnc_sensory = np.array([], dtype=np.int32)
        descending = np.arange(4, dtype=np.int32)
        da = np.arange(16, 20, dtype=np.int32)

        def deposit_reward(self, reward, profit):
            if profit == 0:
                return
            kick = 0.5 if profit > 0 else -0.3
            reward[self.da] += np.float32(kick)

    swarm = FlySwarm(_Core(), plastic=True)
    credits = credits_from_votes({"trend": "BUY", "fade": "SELL", "conf": "HOLD", "risk": "HOLD"}, "BUY", 1.0)
    assert credits["trend"] > 0
    assert credits["fade"] < 0
    ones = {n: 1.0 for n in ("trend", "fade", "conf", "risk")}
    swarm.apply_reward(120.0, "BUY", credits=credits, da_scale=ones, hunger=ones)
    assert swarm.trend_p.last_r > 0
    assert swarm.fade_p.last_r < 0
    assert swarm.conf_p.n_updates == 0


def test_shared_da_without_credits_hits_all_heads():
    import numpy as np

    class _Core:
        n = 40
        sugar = np.arange(8, dtype=np.int32)
        ol_sensory = np.array([], dtype=np.int32)
        vnc_sensory = np.array([], dtype=np.int32)
        descending = np.arange(4, dtype=np.int32)
        da = np.arange(10, 14, dtype=np.int32)

        def deposit_reward(self, reward, profit):
            if profit == 0:
                return
            reward[self.da] += np.float32(0.5 if profit > 0 else -0.3)

    swarm = FlySwarm(_Core(), plastic=True)
    swarm.apply_reward(80.0, "BUY")
    assert swarm.trend_p.n_updates == 1
    assert swarm.fade_p.n_updates == 1
    assert swarm.conf_p.n_updates == 1
    assert swarm.risk_p.n_updates == 1


def test_scaled_r_stuffed_and_threat_near_zero():
    hot = scaled_r(1.0, 0.9, 1.25)
    stuffed = scaled_r(1.0, 0.05, 0.15)
    assert hot > stuffed
    assert stuffed < 0.20


def test_crop_on_close_updates_equity():
    crop = FlyCrop("trend")
    start = crop.equity
    ar = crop.on_close(1.0, "BUY", "BUY")
    assert ar > 0
    assert crop.equity > start
    assert crop.last_r > 0


def test_pool_step_and_mix_smoke():
    pool = FlyCropPool()
    packs = pool.step(
        {"trend": "BUY", "fade": "HOLD", "conf": "BUY", "risk": "HOLD"},
        tech="BUY",
        banc_mult=1.0,
        uncertainty=0.3,
        agreement=0.6,
        risk={"banc_mult": 1.0, "rails_kill": False, "uncertainty": 0.3, "atr_ratio": 1.0, "risk_tol": "balanced"},
        amt=1.0,
        sugar_dyn=True,
        tag="2oo3",
    )
    assert "trend" in packs and "fade" in packs
    book = pool.mix("BUY", tag="2oo3")
    assert "size_mult" in book
    assert "crops" in book
