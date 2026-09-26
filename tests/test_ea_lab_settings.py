"""MT4 FlyTrader lab SETTINGS payload ↔ Python dials."""

from __future__ import annotations

import argparse

from flyfx.trader import parse_ea_lab_blob, parse_ea_volume_pct, _EA_MISSING


SAMPLE = (
    "OK|FUSE|all|DYNAMIC|1|INDS|macd,sar,rsi,bb|RISK|scalp|"
    "SUGAR|1|SAMT|1.00|SDYN|0|"
    "VPCT|auto|VMODE|auto|BANK|0|HOLD|1|HOLDS|1.00|"
    "RECYCLE|1|BAYES|1|APP|1|TRATE|8.0|"
    "NN|1|VISION|1|FRESH|0|ADV|1|SETANN|1|FIBTR|1|EVOLVE|1|"
    "CROPS|0|BRAIN|0|EVO|0|CDA|1|"
    "SYMBOL|EURUSD|MAGIC|20260919"
)


def test_parse_ea_lab_blob_matches_gui_defaults():
    blob = parse_ea_lab_blob(SAMPLE)
    assert blob["fuse_cats"] == "all"
    assert blob["fuse_dynamic"] is True
    assert blob["risk_tol"] == "scalp"
    assert blob["sugar"] is True
    assert blob["no_sugar"] is False
    assert blob["sugar_amt"] == 1.0
    assert blob["sugar_dyn"] is False
    assert blob["volume_pct"] is None
    assert blob["volume_mode"] == "auto"
    assert blob["bank_pct"] == 0.0
    assert blob["hold_risk"] is True
    assert blob["profit_recycle"] is True
    assert blob["trade_bayes"] is True
    assert blob["entry_appetite"] is True
    assert blob["trade_rate"] == 8.0
    assert blob["nn_vote"] is True
    assert blob["nn_vision"] is True
    assert blob["nn_fresh"] is False
    assert blob["nn_adv"] is True
    assert blob["settings_ann"] is True
    assert blob["fib_trade"] is True
    assert blob["evolve"] is True
    assert blob["fly_crops"] is False
    assert blob["use_brain"] is False
    assert blob["use_evo"] is False
    assert blob["continue_da"] is True
    assert blob["apply_latent_live"] is True


def test_parse_ea_volume_pct_auto_vs_missing():
    assert parse_ea_volume_pct("OK|VPCT|auto") is None
    assert parse_ea_volume_pct("OK|VPCT|-1") is None
    assert parse_ea_volume_pct("OK|VPCT|12") == 12.0
    assert parse_ea_volume_pct("OK|FUSE|all") is _EA_MISSING


def test_old_ea_payload_still_parses_core():
    old = "OK|FUSE|trend,volume|DYNAMIC|0|INDS|all|RISK|balanced|SUGAR|0|SAMT|1.00|SDYN|0|SYMBOL|EURUSD"
    blob = parse_ea_lab_blob(old)
    assert blob["fuse_cats"] == "trend,volume"
    assert blob["risk_tol"] == "balanced"
    assert blob["sugar"] is False
    assert "nn_vote" not in blob
    assert "volume_pct" not in blob


def test_apply_blob_keys_onto_namespace():
    args = argparse.Namespace(
        fuse_cats="",
        fuse_dynamic=False,
        risk_tol="balanced",
        sugar=False,
        no_sugar=True,
        volume_pct=50.0,
        volume_mode="equity",
        nn_vote=False,
        nn_vision=False,
        entry_appetite=False,
        trade_bayes=False,
        profit_recycle=False,
        hold_risk=False,
        use_brain=False,
        use_evo=False,
        apply_latent_live=False,
    )
    blob = parse_ea_lab_blob(SAMPLE)
    for key, val in blob.items():
        setattr(args, key, val)
    assert args.risk_tol == "scalp"
    assert args.fuse_dynamic is True
    assert args.volume_pct is None
    assert args.nn_vote is True
    assert args.apply_latent_live is True
