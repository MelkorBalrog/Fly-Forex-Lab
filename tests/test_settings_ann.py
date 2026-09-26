"""20-net Bayesian settings controller."""

from __future__ import annotations

import numpy as np

from flyfx.brain.settings_ann import (
    N_IN,
    N_NETS,
    N_SETTINGS,
    SPECS,
    SettingsFleet,
    normalize_param,
    pack_state,
    posterior_weights,
)
from flyfx.risk.bayes_sizer import PARAM_DEFAULTS, AdaptiveParams
from flyfx.sense.category_kalman import CATEGORY_NAMES


def _feat(rsi_z: float = 0.4, ema_z: float = 0.8) -> dict:
    measurements = {cat: {} for cat in CATEGORY_NAMES}
    measurements["momentum"]["rsi"] = rsi_z
    measurements["trend"]["ema"] = ema_z
    return {
        "impulse": 0.55,
        "fused": 0.40,
        "atr_ratio": 1.3,
        "measurements": measurements,
        "categories": {
            "trend": {"impulse": 0.6, "fused": 0.5, "residual": 0.1, "uncertainty": 0.2, "agreement": 0.8, "slope": 0.01},
            "momentum": {"impulse": -0.2, "fused": -0.1, "residual": 0.0, "uncertainty": 0.3, "agreement": 0.6, "slope": 0.0},
        },
        "kalman": {"residual": 0.1, "uncertainty": 0.25, "agreement": 0.7},
    }


def _ctx(feat: dict | None = None, fuse: str = "rsi") -> dict:
    feat = feat or _feat()
    return {
        "feat": feat,
        "fuse_inds": fuse,
        "tech": {"vote": "BUY", "fused": 0.4, "p_cast": 0.7, "p_win": 0.55},
        "trend": {"vote": "BUY", "fused": 0.3, "p_cast": 0.6, "p_win": 0.52},
        "fade": {"vote": "HOLD", "fused": 0.0, "p_cast": 0.4, "p_win": 0.5},
        "conf_vote": "HOLD",
        "conf_score": 0.1,
        "nn_side": "BUY",
        "nn_conf": 0.62,
        "t_score": 0.4,
        "f_score": -0.1,
    }


def test_twenty_structures_are_distinct():
    assert len(SPECS) == N_NETS == 20
    names = [s["name"] for s in SPECS]
    assert len(set(names)) == 20
    shapes = {(tuple(s["hidden"]), s["act"], bool(s.get("residual")), float(s.get("dropout") or 0)) for s in SPECS}
    assert len(shapes) == 20


def test_pack_keeps_only_selected_indicators():
    params = AdaptiveParams()
    x = pack_state(_feat(), {}, params, "rsi", np.zeros(8))
    assert x.shape == (N_IN,)
    from flyfx.brain.settings_ann import INDICATOR_NAMES, OFF_MASK

    rsi = INDICATOR_NAMES.index("rsi")
    ema = INDICATOR_NAMES.index("ema")
    assert x[OFF_MASK + rsi] == 1.0
    assert x[rsi] == 0.4
    assert x[OFF_MASK + ema] == 0.0
    assert x[ema] == 0.0
    # Current dials are part of the input.
    assert x.shape[0] > N_SETTINGS
    sl = normalize_param("sl_atr", params.sl_atr)
    from flyfx.brain.settings_ann import OFF_SETTINGS

    assert abs(x[OFF_SETTINGS] - sl) < 1e-9


def test_fresh_fleet_holds_the_dials():
    fleet = SettingsFleet("EURUSD", seed=3, parallel=False)
    try:
        params = AdaptiveParams()
        y, w = fleet.propose(_ctx(), params)
        assert y.shape == (N_SETTINGS,)
        assert abs(float(w.sum()) - 1.0) < 1e-9
        assert float(w.min()) > 0.0
        cur = np.array([normalize_param(n, getattr(params, n)) for n in PARAM_DEFAULTS])
        assert np.max(np.abs(y - cur)) < 0.05
    finally:
        fleet.close()


def test_parallel_matches_serial():
    fleet = SettingsFleet("EURUSD", seed=5, parallel=False)
    try:
        params = AdaptiveParams()
        ctx = _ctx(fuse="all")
        a, wa = fleet.propose(ctx, params)
        fleet.parallel = True
        b, wb = fleet.propose(ctx, params)
        np.testing.assert_allclose(a, b, atol=1e-8)
        np.testing.assert_allclose(wa, wb, atol=1e-8)
    finally:
        fleet.close()


def test_linear_head_learns_a_stop_trim():
    fleet = SettingsFleet("EURUSD", seed=1, parallel=False)
    try:
        net = fleet.nets[0]
        assert net.name == "linear"
        x = np.zeros(N_IN)
        settings = np.zeros(N_SETTINGS)
        target = settings.copy()
        target[0] = 0.45
        err0 = net.train_step(x, settings, target, 0.2, fleet.rng)
        err = err0
        for _ in range(12):
            err = net.train_step(x, settings, target, 0.2, fleet.rng)
        assert err < err0 * 0.5
    finally:
        fleet.close()


def test_losses_do_not_widen_the_stop():
    fleet = SettingsFleet("EURUSD", seed=11, parallel=True)
    try:
        params = AdaptiveParams()
        start_sl = float(params.sl_atr)
        start_cool = float(params.cooldown_bars)
        ctx = _ctx(fuse="rsi,ema,macd")
        row = {"usd": -12.0, "pips": -9.0, "reason": "ATR stop"}
        for _ in range(14):
            fleet.arm(ctx, params)
            fleet.on_close(row, params, ctx)
        assert params.sl_atr <= start_sl + 1e-6
        assert params.cooldown_bars >= start_cool - 1e-6
        assert fleet.last_note.startswith("settings-ann")
        w = posterior_weights(np.array([n.nll for n in fleet.nets]))
        assert abs(float(w.sum()) - 1.0) < 1e-9
    finally:
        fleet.close()


def test_a_win_is_kept_and_a_later_loss_steps_back():
    fleet = SettingsFleet("EURUSD", seed=2, parallel=False)
    try:
        params = AdaptiveParams()
        ctx = _ctx()
        fleet.arm(ctx, params)
        fleet.on_close({"usd": 40.0, "pips": 12.0, "reason": "trail stop"}, params, ctx)
        locked = float(params.sl_atr)
        for _ in range(6):
            fleet.arm(ctx, params)
            fleet.on_close({"usd": -15.0, "pips": -8.0, "reason": "ATR stop"}, params, ctx)
        assert params.sl_atr <= locked + 1e-6
        assert fleet.peak_usd > 0
    finally:
        fleet.close()


def test_unlabeled_loss_does_not_widen_sl():
    fleet = SettingsFleet("EURUSD", seed=4, parallel=False)
    try:
        params = AdaptiveParams()
        start = float(params.sl_atr)
        ctx = _ctx()
        row = {"usd": -8.0, "pips": -7.0, "reason": "sugar full"}
        for _ in range(12):
            fleet.arm(ctx, params)
            fleet.on_close(row, params, ctx)
        assert params.sl_atr <= start + 1e-6
    finally:
        fleet.close()


def test_window_train_prints_all_twenty(tmp_path, monkeypatch, capsys):
    import flyfx.brain.settings_ann as mod

    monkeypatch.setattr(mod, "BRAIN_DIR", tmp_path)
    bars = []
    px = 1.10
    for i in range(100):
        px += 0.00015
        bars.append(
            {
                "open": px - 0.00005,
                "high": px + 0.0002,
                "low": px - 0.0002,
                "close": px,
                "volume": 10,
                "time": 1_700_000_000 + i * 300,
            }
        )
    fleet = mod.train_settings_fleet(bars, "ZZSETANN", quiet=False, horizon=8)
    assert fleet is not None
    try:
        assert fleet.n_updates > 0
        assert all(net.n_fit == fleet.n_updates for net in fleet.nets)
        out = capsys.readouterr().out
        for spec in SPECS:
            assert f"settings-ann  {spec['name']}" in out
        assert "ZZSETANN.setann.npz" in out
        assert "transfer" not in out
        assert fleet.cum_usd == 0.0
        assert not fleet._anchored
    finally:
        fleet.close()


def test_second_window_train_transfers(tmp_path, monkeypatch, capsys):
    import flyfx.brain.settings_ann as mod

    monkeypatch.setattr(mod, "BRAIN_DIR", tmp_path)
    bars = []
    px = 1.10
    for i in range(80):
        px += 0.0002
        bars.append(
            {
                "open": px - 0.00005,
                "high": px + 0.0002,
                "low": px - 0.0002,
                "close": px,
                "volume": 10,
                "time": 1_700_000_000 + i * 300,
            }
        )
    first = mod.train_settings_fleet(bars, "ZZSETANN", quiet=True, horizon=8)
    assert first is not None
    first.close()
    capsys.readouterr()
    second = mod.train_settings_fleet(bars, "ZZSETANN", quiet=False, horizon=8)
    assert second is not None
    try:
        out = capsys.readouterr().out
        assert "settings-ann  transfer  warm-start" in out
        assert "replay" in out
        assert (tmp_path / "ZZSETANN.setann_mem.npz").exists()
    finally:
        second.close()


def test_save_load_roundtrip(tmp_path, monkeypatch):
    import flyfx.brain.settings_ann as mod

    monkeypatch.setattr(mod, "BRAIN_DIR", tmp_path)
    fleet = SettingsFleet("GBPUSD", seed=9, parallel=False)
    try:
        params = AdaptiveParams()
        fleet.arm(_ctx(), params)
        fleet.on_close({"usd": -5.0, "pips": -6.0, "reason": "ATR stop"}, params, _ctx())
        path = fleet.save()
        assert path.name == "GBPUSD.setann.npz"
        other = SettingsFleet("GBPUSD", seed=0, parallel=False)
        assert other.load(path)
        assert other.n_updates == fleet.n_updates
        np.testing.assert_allclose(other.nets[3].W[0], fleet.nets[3].W[0])
        np.testing.assert_allclose(other.paid, fleet.paid)
        other.close()
    finally:
        fleet.close()
