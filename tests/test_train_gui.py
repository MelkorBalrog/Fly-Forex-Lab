"""GUI Train DA + brains/{PAIR}.json category-fly persistence."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from flyfx.paths import DASH_HTML
from flyfx.trader import _cmd_truthy, _gui_apply_common, parse_args
from flyfx.ui.dash import DashHub


def test_save_brain_writes_cat_flies(tmp_path, monkeypatch):
    import flyfx.brain.fly_brains as fb

    monkeypatch.setattr(fb, "BRAIN_DIR", tmp_path)
    path = fb.save_brain(
        "GBPUSD",
        {"trend": {"n_updates": 3}},
        None,
        {"da_n": 3},
        extra={"cat_flies": {"trend": {"trend": {"n_updates": 4}}}},
    )
    assert path is not None
    blob = json.loads(path.read_text(encoding="utf-8"))
    assert blob["symbol"] == "GBPUSD"
    assert blob["cat_flies"]["trend"]["trend"]["n_updates"] == 4
    assert blob["plastic"]["trend"]["n_updates"] == 3
    assert "genome" in blob
    assert "GBPUSD" in fb.list_brain_symbols()
    loaded = fb.load_brain("GBPUSD")
    assert loaded["cat_flies"]["trend"]["trend"]["n_updates"] == 4


def test_save_brain_skips_eurusd_unless_forced(tmp_path, monkeypatch):
    import flyfx.brain.fly_brains as fb

    monkeypatch.setattr(fb, "BRAIN_DIR", tmp_path)
    assert fb.save_brain("EURUSD", {"x": 1}, None, extra={"cat_flies": {"trend": {}}}) is None
    assert list(tmp_path.glob("*.json")) == []
    path = fb.save_brain(
        "EURUSD",
        {"x": 1},
        None,
        extra={"cat_flies": {"trend": {"trend": {"n_updates": 1}}}},
        force=True,
    )
    assert path is not None
    blob = json.loads(path.read_text(encoding="utf-8"))
    assert blob["frozen"] is False
    assert blob["cat_flies"]["trend"]["trend"]["n_updates"] == 1
    assert "EURUSD" in fb.list_brain_symbols()
    again = fb.seed_frozen_eurusd()
    kept = json.loads(again.read_text(encoding="utf-8"))
    assert kept["frozen"] is False
    assert kept["cat_flies"]["trend"]["trend"]["n_updates"] == 1


def test_list_brain_symbols_skips_frozen_empty(tmp_path, monkeypatch):
    import flyfx.brain.fly_brains as fb

    monkeypatch.setattr(fb, "BRAIN_DIR", tmp_path)
    fb.seed_frozen_eurusd()
    assert fb.list_brain_symbols() == []
    (tmp_path / "AUDUSD.json").write_text(json.dumps({"symbol": "AUDUSD", "plastic": {}, "cat_flies": {}}), encoding="utf-8")
    (tmp_path / "NZDUSD.evo.json").write_text(
        json.dumps({"symbol": "NZDUSD", "kind": "evo", "genome": {"risk_tol": "balanced"}}),
        encoding="utf-8",
    )
    assert fb.list_brain_symbols() == []
    assert fb.list_evo_symbols() == ["NZDUSD"]


def test_list_evo_policies_includes_nested_genome(tmp_path, monkeypatch):
    import flyfx.brain.fly_brains as fb

    monkeypatch.setattr(fb, "BRAIN_DIR", tmp_path)
    fb.save_brain(
        "AUDUSD",
        {"trend": {"n_updates": 1}},
        None,
        extra={"genome": {"risk_tol": "conservative", "sugar": True, "params": {"sl_atr": 2.2}}},
    )
    rows = fb.list_evo_policies()
    assert any(r["symbol"] == "AUDUSD" and r["genome"]["risk_tol"] == "conservative" for r in rows)


def test_dash_html_has_train_da():
    html = DASH_HTML.read_text(encoding="utf-8")
    assert "Train DA" in html
    assert "inverse-var fuse" in html
    assert "CAT-FUSE" in html
    assert 'id="fuse-trend"' in html
    assert 'id="fuse-volume"' in html
    assert 'id="fuse-volume" checked' not in html
    assert 'id="fuse-breadth" checked' not in html
    assert 'id="fuse-structure" checked' not in html
    assert 'id="fuse-sentiment" checked' not in html
    assert 'id="fuse-trend" checked' in html
    assert 'const DEFAULT_INDS = ["ema","macd","adx","sar","ichimoku","supertrend","rsi","stoch","cci","willr","bb"]' in html
    assert 'id="fuse-dynamic"' in html
    assert 'id="fuse-inds"' in html
    assert 'id="risk-tol"' in html
    assert 'id="use-sugar"' in html
    assert 'id="use-sugar" checked' not in html
    assert "0.18 + 0.82" in html
    assert "0.42 + 0.58" not in html
    assert 'id="sugar-amt"' in html
    assert 'id="sugar-dyn"' in html
    assert 'id="fly-crops"' in html
    assert 'id="fly-crops" checked' not in html
    assert 'id="evolve"' in html
    assert 'id="evo-pop"' in html
    assert 'id="evo-gens"' in html
    assert 'id="card-cns"' in html
    assert "selectedFuseInds" in html
    assert "selectedFuseCats" in html
    assert 'id="use-brain"' in html
    assert 'id="use-evo"' in html
    assert "applyEvoToForm" in html
    assert "setEvoLocked" in html
    assert "function setEvoLocked" in html
    assert "EVO_LOCK_IDS" in html
    assert "applyUseEvo" in html
    lock_blob = html.split("EVO_LOCK_IDS", 1)[1].split("]", 1)[0]
    for lock_id in ("risk-tol", "use-sugar", "sugar-amt", "sugar-dyn", "fuse-dynamic"):
        assert lock_id in lock_blob
    assert "legacy-2oo3" not in lock_blob
    assert 'id="continue-da"' in html
    assert 'id="force-eurusd"' not in html
    assert 'id="legacy-2oo3"' not in html
    assert 'send("train")' in html


def test_dash_default_committee_mode_is_fuse():
    hub = DashHub(8765, pairs=["EURUSD"])
    assert hub.state["committee_mode"] == "fuse"
    assert "parts" in hub.state["cat_fuse"]


def test_dash_meta_lists_brains(tmp_path, monkeypatch):
    import flyfx.brain.fly_brains as fb

    monkeypatch.setattr(fb, "BRAIN_DIR", tmp_path)
    fb.save_brain("USDJPY", {"trend": {"n_updates": 1}}, None, extra={"cat_flies": {"momentum": {"fade": {"n_updates": 2}}}})
    fb.save_evo("GBPUSD", {"risk_tol": "aggressive", "sugar": False, "params": {"sl_atr": 2.5}})
    hub = DashHub(8765, pairs=["EURUSD", "USDJPY", "GBPUSD"])
    meta = hub.meta()
    assert meta["brains"] == ["USDJPY"]
    assert meta["evo"][0]["symbol"] == "GBPUSD"
    assert "USDJPY" in meta["pairs"]


def test_cli_leverage_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fly_forex.py", "--replay", "--leverage", "30"])
    args = parse_args()
    assert args.leverage == 30.0


def test_cli_force_and_shadow_da_flags(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fly_forex.py", "--gui"])
    args = parse_args()
    assert args.force_brain is False
    assert args.shadow_da is False
    monkeypatch.setattr(sys, "argv", ["fly_forex.py", "--save-brain", "--force-brain", "--shadow-da"])
    flagged = parse_args()
    assert flagged.save_brain is True
    assert flagged.force_brain is True
    assert flagged.shadow_da is True


def test_gui_helpers_parse_command():
    assert _cmd_truthy(True) is True
    assert _cmd_truthy("true") is True
    assert _cmd_truthy("0") is False
    args = SimpleNamespace(bars=2000, balance=100000.0, timeframe="M5")
    _gui_apply_common(
        {
            "bars": 0,
            "balance": 25000,
            "timeframe": "M10",
            "fuse_cats": "trend,volume",
            "fuse_dynamic": True,
            "fuse_inds": "ema,rsi",
            "risk_tol": "conservative",
            "legacy_2oo3": True,
            "use_sugar": False,
            "sugar_amt": 0.5,
            "sugar_dyn": True,
            "fly_crops": True,
            "leverage": 50,
        },
        args,
    )
    assert args.bars == 0
    assert args.balance == 25000.0
    assert args.timeframe == "M10"
    assert args.fuse_cats == "trend,volume"
    assert args.fuse_dynamic is True
    assert args.fuse_inds == "ema,rsi"
    assert args.risk_tol == "conservative"
    assert args.legacy_2oo3 is True
    assert args.no_sugar is True
    assert args.sugar is False
    assert args.sugar_amt == 0.5
    assert args.sugar_dyn is True
    assert args.fly_crops is True
    assert args.leverage == 50.0
    _gui_apply_common({"leverage": 0}, args)
    assert args.leverage == 1.0
    _gui_apply_common({"leverage": "nope"}, args)
    assert args.leverage == 1.0
    fresh = SimpleNamespace(bars=1, balance=100000.0, timeframe="M5")
    _gui_apply_common({"leverage": "nope"}, fresh)
    assert fresh.leverage == 50.0
    _gui_apply_common({"legacy_2oo3": False, "use_sugar": True, "evolve": True, "evo_pop": 8, "evo_gens": 3}, args)
    assert args.legacy_2oo3 is False
    assert args.sugar is True
    assert args.no_sugar is False
    assert args.evolve is True
    assert args.evo_pop == 8
    assert args.evo_gens == 3
