"""GUI lab configuration save/load."""

from __future__ import annotations

import json

from flyfx.ui.dash import (
    DEFAULT_LAB_CONFIG,
    list_saved_configs,
    load_config_file,
    save_config_file,
)


def test_default_lab_config_matches_screenshot_profile():
    cfg = DEFAULT_LAB_CONFIG
    assert cfg["risk_tol"] == "scalp"
    assert cfg["volume_mode"] == "auto"
    assert cfg["nn_vote"] is True
    assert cfg["nn_vision"] is True
    assert cfg["entry_appetite"] is True
    assert cfg["continue_da"] is True
    assert cfg["use_sugar"] is True
    assert cfg["fuse_dynamic"] is True
    assert cfg["settings_ann"] is True
    assert cfg["fib_trade"] is True
    inds = set(cfg["fuse_inds"].split(","))
    assert "macd" in inds and "ema" not in inds
    assert "keltner" in inds and "atr" not in inds
    assert "put_call" in inds


def test_save_and_load_config(tmp_path, monkeypatch):
    import flyfx.ui.dash as dash

    monkeypatch.setattr(dash, "CONFIG_DIR", tmp_path)
    path = save_config_file("my-setup", {"risk_tol": "scalp", "nn_vision": True, "bars": 2000})
    assert path.exists()
    assert path.name == "my-setup.json"
    assert (tmp_path / "_last.json").exists()
    assert "my-setup" in list_saved_configs()
    loaded = load_config_file("my-setup")
    assert loaded["risk_tol"] == "scalp"
    assert loaded["nn_vision"] is True
    assert load_config_file("default")["fuse_dynamic"] is True
    last = load_config_file("_last")
    assert last["name"] == "my-setup"


def test_lab_default_fuse_inds_roundtrip():
    from flyfx.sense.category_kalman import encode_fuse_inds, parse_fuse_inds

    blob = parse_fuse_inds("lab-default")
    assert "macd" in blob["trend"] and "ema" not in blob["trend"]
    assert encode_fuse_inds(blob) == "lab-default"
    assert encode_fuse_inds(parse_fuse_inds("gui-default")) == "lab-default"
