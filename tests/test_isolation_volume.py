"""Isolation and risk-based volume %."""

from __future__ import annotations

from flyfx.risk.isolation import disagree_circuit, isolate_category_states
from flyfx.risk.volume import select_volume_pct
from flyfx.sense.category_kalman import fuse_category_latents, parse_fuse_inds


def test_plausibility_fuse_inds():
    blob = parse_fuse_inds("plausibility")
    assert "ema" in blob["trend"]
    assert "sma" not in blob["trend"]
    assert "rsi" in blob["momentum"]
    assert "roc" not in blob["momentum"]
    assert "bb" in blob["volatility"]
    assert "atr" not in blob["volatility"]
    assert blob["volume"] == ()
    assert blob["breadth"] == ()
    assert blob["sentiment"] == ()


def test_isolate_drops_uncertain_keeps_same_sign_extreme():
    states = {
        "trend": {"available": True, "fused": 0.4, "residual": 0.1, "P": 0.05, "uncertainty": 0.2},
        "momentum": {"available": True, "fused": 0.35, "residual": 0.05, "P": 0.05, "uncertainty": 0.2},
        "volatility": {"available": True, "fused": 0.3, "residual": 0.0, "P": 0.04, "uncertainty": 0.99},
        "structure": {"available": True, "fused": 3.5, "residual": 1.0, "P": 0.01, "uncertainty": 0.1},
    }
    rows, health = isolate_category_states(
        states, ("trend", "momentum", "volatility", "structure")
    )
    names = {n for n, *_ in rows}
    assert "volatility" not in names
    assert "structure" in names  # same-sign extreme reinforces, not dropped
    assert "trend" in names and "momentum" in names
    assert "volatility" in health.dropped

    # Opposing outlier still drops.
    states2 = {
        "trend": {"available": True, "fused": 0.5, "residual": 0.1, "P": 0.05, "uncertainty": 0.2},
        "momentum": {"available": True, "fused": 0.45, "residual": 0.05, "P": 0.05, "uncertainty": 0.2},
        "volatility": {"available": True, "fused": 0.4, "residual": 0.0, "P": 0.05, "uncertainty": 0.2},
        "structure": {"available": True, "fused": -2.5, "residual": 1.0, "P": 0.01, "uncertainty": 0.1},
    }
    rows2, health2 = isolate_category_states(
        states2, ("trend", "momentum", "volatility", "structure")
    )
    assert "structure" not in {n for n, *_ in rows2}
    assert "structure" in health2.dropped


def test_fuse_isolates_without_killing_healthy_impulse():
    states = {
        "trend": {"available": True, "fused": 0.55, "residual": 0.1, "P": 0.08, "uncertainty": 0.25},
        "momentum": {"available": True, "fused": 0.48, "residual": 0.05, "P": 0.09, "uncertainty": 0.28},
        "volatility": {"available": True, "fused": 0.42, "residual": 0.02, "P": 0.10, "uncertainty": 0.30},
    }
    mix = fuse_category_latents(states, include=("trend", "momentum", "volatility"))
    assert mix["available"]
    assert mix["impulse"] > 0.2
    assert mix["regime"] == "UP"


def test_disagree_circuit_detects_split():
    assert disagree_circuit([0.5, -0.5, 0.4], agreement=0.15)
    assert not disagree_circuit([0.5, 0.45, 0.4], agreement=0.8)


def test_volume_pct_follows_risk_profile():
    cons = select_volume_pct(risk_tol="conservative")
    bal = select_volume_pct(risk_tol="balanced")
    agg = select_volume_pct(risk_tol="aggressive")
    assert cons.pct == 50.0
    assert bal.pct == 12.0
    assert agg.pct == 100.0
    assert abs(bal.mult - 0.12) < 1e-9
    assert abs(cons.sized_equity(100_000) - 50_000) < 1e-6


def test_volume_pct_fixed_and_auto():
    fixed = select_volume_pct(volume_mode="fixed", volume_pct=50)
    assert fixed.pct == 50.0
    assert abs(fixed.sized_equity(100_000, free=80_000) - 40_000) < 1e-6
    auto = select_volume_pct(
        risk_tol="balanced",
        volume_mode="auto",
        tape_mult=0.5,
        banc_mult=1.0,
    )
    assert auto.pct == 25.0  # 12 * 0.5 * 1.0 → floor 25
    override = select_volume_pct(risk_tol="conservative", volume_pct=90)
    assert override.pct == 90.0
    # legacy "risk" mode aliases to equity
    legacy = select_volume_pct(volume_mode="risk", volume_pct=50)
    assert legacy.mode == "equity"
    assert legacy.pct == 50.0


def test_size_lots_volume_pct_is_margin_of_available_money():
    from flyfx.exec.account_sim import AccountSim
    from flyfx.trader import size_lots

    account = AccountSim(balance=100_000.0, leverage=100.0)
    price = 1.16
    mpl = account.required_margin(1.0, price)
    half, note = size_lots(
        account,
        100_000.0,
        100_000.0,
        price,
        atr=0.0010,
        risk_pct=1.0,
        fixed_lots=0.01,
        min_lots=0.01,
        max_lots=100.0,
        margin_cap_pct=25.0,
        volume_pct=50.0,
        banc_mult=1.0,
        admit_mult=1.0,
        size_mult=1.0,
        cond_mult=1.0,
        thrust_mult=1.0,
        volume_mult=1.0,
    )
    assert abs(half * mpl - 50_000.0) < mpl  # within one lot of 50% margin
    assert "50%" in note or "VOL 50%" in note
    full, _ = size_lots(
        account,
        100_000.0,
        100_000.0,
        price,
        atr=0.0010,
        risk_pct=1.0,
        fixed_lots=0.01,
        min_lots=0.01,
        max_lots=100.0,
        margin_cap_pct=25.0,
        volume_pct=100.0,
        banc_mult=1.0,
        admit_mult=1.0,
        size_mult=1.0,
        cond_mult=1.0,
        thrust_mult=1.0,
        volume_mult=1.0,
    )
    assert full > half * 1.5
