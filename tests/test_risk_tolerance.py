"""Risk-tolerance profiles: overlay geometry + risk-factor closes."""

from flyfx.risk.bayes_sizer import PARAM_DEFAULTS
from flyfx.risk.tolerance import (
    overlay_geom,
    parse_risk_tol,
    risk_close_reason,
    risk_profile,
)
from flyfx.trader import parse_ea_risk_spec, parse_args


def test_parse_risk_tol_aliases():
    assert parse_risk_tol("") == "balanced"
    assert parse_risk_tol("balanced") == "balanced"
    assert parse_risk_tol("cons") == "conservative"
    assert parse_risk_tol("tight") == "conservative"
    assert parse_risk_tol("agg") == "aggressive"
    try:
        parse_risk_tol("yolo")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_scalp_overlay_uses_factory_4k_geom():
    walked = dict(PARAM_DEFAULTS)
    out = overlay_geom(walked, risk_profile("scalp"))
    assert out["sl_atr"] == 2.40
    assert out["tp_atr"] == 12.0
    assert out["min_hold_bars"] == 4.0
    assert out["max_hold_bars"] == 64.0
    assert parse_risk_tol("fast") == "scalp"
    assert parse_risk_tol("scalping") == "scalp"


def test_scalp_companions_restore_4k_book():
    from types import SimpleNamespace
    from flyfx.risk.bayes_sizer import AdaptiveParams
    from flyfx.risk.tolerance import apply_scalp_companions
    from flyfx.risk.volume import profile_volume_pct

    args = SimpleNamespace(
        bank_pct=0.0,
        hold_risk=True,
        hold_risk_sens=1.0,
        entry_appetite=False,
        trade_rate=None,
        no_snowball=False,
    )
    params = AdaptiveParams()
    apply_scalp_companions(args, params)
    assert args.bank_pct == 0.0
    assert args.no_snowball is False
    assert args.entry_appetite is False
    assert args.trade_rate is None
    assert params.tp_atr == 12.0
    assert params.sl_atr == 2.40
    assert params.cooldown_bars == 8.0
    assert params.min_impulse == 0.50
    assert profile_volume_pct("scalp") == 12.0
    # Balanced factory profile must stay untouched on geometry / day rails;
    # VOL 12% matches the 4k cat-fuse margin scale (freeze uses risk-% sizing).
    bal = risk_profile("balanced")
    assert bal["sl_atr"] == 2.40
    assert bal["day_trades"] == 3
    assert profile_volume_pct("balanced") == 12.0



def test_conservative_overlays_tighter_stops():
    walked = dict(PARAM_DEFAULTS)
    walked["sl_atr"] = 2.51
    out = overlay_geom(walked, risk_profile("conservative"))
    assert out["sl_atr"] == 1.90
    assert out["tp_atr"] == 8.0
    assert out["max_hold_bars"] == 40.0
    agg = overlay_geom(walked, risk_profile("aggressive"))
    assert agg["sl_atr"] == 3.10
    assert agg["max_hold_bars"] == 80.0


def test_balanced_risk_exits_never_fire():
    feat = {"kalman": {"uncertainty": 0.99}, "atr_ratio": 3.0}
    why = risk_close_reason(
        risk_profile("balanced"),
        side="BUY",
        bars_held=20,
        feat=feat,
        banc_mult=0.10,
        fade="SELL",
        rails_kill=True,
        unreal=-1.0,
        atr=0.001,
    )
    assert why is None


def test_conservative_banc_fade_kill_uncertainty_vol():
    cons = risk_profile("conservative")
    feat = {"kalman": {"uncertainty": 0.10}, "atr_ratio": 1.0}
    assert risk_close_reason(
        cons, side="BUY", bars_held=2, feat=feat, banc_mult=0.10,
        fade="HOLD", rails_kill=False, unreal=0.0, atr=0.001,
    ) is None
    why = risk_close_reason(
        cons, side="BUY", bars_held=8, feat=feat, banc_mult=0.50,
        fade="HOLD", rails_kill=False, unreal=0.0, atr=0.001,
    )
    assert why and why.startswith("risk BANC")
    assert risk_close_reason(
        cons, side="BUY", bars_held=8, feat=feat, banc_mult=1.0,
        fade="SELL", rails_kill=False, unreal=0.0, atr=0.001,
    ) == "risk fade"
    assert risk_close_reason(
        cons, side="BUY", bars_held=1, feat=feat, banc_mult=1.0,
        fade="HOLD", rails_kill=True, unreal=0.0, atr=0.001,
    ) == "risk kill"
    assert risk_close_reason(
        cons, side="BUY", bars_held=1, feat=feat, banc_mult=1.0,
        fade="HOLD", rails_kill=False, unreal=0.0, atr=0.001,
        equity=98_400.0, peak_eq=100_000.0,
    ) == "risk kill"
    uncert = risk_close_reason(
        cons, side="BUY", bars_held=8,
        feat={"kalman": {"uncertainty": 0.90}, "atr_ratio": 1.0},
        banc_mult=1.0, fade="HOLD", rails_kill=False, unreal=0.0, atr=0.001,
    )
    assert uncert and uncert.startswith("risk uncertainty")
    vol = risk_close_reason(
        cons, side="BUY", bars_held=8,
        feat={"kalman": {"uncertainty": 0.10}, "atr_ratio": 2.0},
        banc_mult=1.0, fade="HOLD", rails_kill=False, unreal=0.0, atr=0.001,
    )
    assert vol and vol.startswith("risk vol spike")


def test_aggressive_skips_risk_factor_exits():
    why = risk_close_reason(
        risk_profile("aggressive"),
        side="BUY",
        bars_held=20,
        feat={"kalman": {"uncertainty": 0.99}, "atr_ratio": 3.0},
        banc_mult=0.10,
        fade="SELL",
        rails_kill=True,
        unreal=-1.0,
        atr=0.001,
    )
    assert why is None


def test_parse_ea_risk_spec():
    assert parse_ea_risk_spec("PONG|EURUSD|123") is None
    assert parse_ea_risk_spec("OK|FUSE|all|DYNAMIC|0|INDS|all|RISK|conservative|SYMBOL|EURUSD") == "conservative"
    assert parse_ea_risk_spec("PONG|EURUSD|1|FUSE|all|DYNAMIC|0|INDS|all|RISK|aggressive") == "aggressive"


def test_cli_default_risk_tol_is_scalp(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "argv", ["fly_forex.py", "--replay", "--hst", "auto"])
    args = parse_args()
    assert args.risk_tol == "scalp"
    monkeypatch.setattr(sys, "argv", ["fly_forex.py", "--replay", "--risk-tol", "conservative"])
    tight = parse_args()
    assert tight.risk_tol == "conservative"
    monkeypatch.setattr(sys, "argv", ["fly_forex.py", "--replay", "--risk-tol", "balanced"])
    bal = parse_args()
    assert bal.risk_tol == "balanced"
