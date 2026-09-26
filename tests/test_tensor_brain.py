"""Tensor brain + 3oo4 committee."""

from __future__ import annotations

import numpy as np

from flyfx.brain.tensor_brain import (
    ENSEMBLE_SPECS,
    FEATURE_NAMES,
    N_ENSEMBLE,
    TensorBrain,
    bayes_average,
    build_supervised_set,
    pack_features,
    posterior_weights,
    train_tensor_brain,
)
from flyfx.trader import committee


def test_committee_2oo3_unchanged():
    side, tag, n = committee("SELL", "cont", "SELL", "HOLD", "HOLD")
    assert side == "SELL"
    assert tag.startswith("2oo3")
    assert n == 2


def test_committee_3oo4_needs_nn_agree():
    # Only 2 classic votes — with nn_vote on and NN HOLD, skip.
    side, tag, n = committee("SELL", "cont", "SELL", "HOLD", "HOLD", nn="HOLD", nn_vote=True)
    assert side == "HOLD"
    assert "oo4" in tag
    # NN agrees → 3oo4.
    side, tag, n = committee("SELL", "cont", "SELL", "HOLD", "HOLD", nn="SELL", nn_vote=True)
    assert side == "SELL"
    assert tag.startswith("3oo4")
    assert n == 3


def test_committee_4oo4():
    side, tag, n = committee("BUY", "cont", "BUY", "BUY", "HOLD", nn="BUY", nn_vote=True)
    assert side == "BUY"
    assert tag.startswith("4oo4")
    assert n == 4


def test_pack_chart_shape():
    from flyfx.brain.tensor_brain import CHART_BARS, CHART_H, pack_chart

    n = 40
    o = [1.1 + 0.001 * i for i in range(n)]
    h = [x + 0.0005 for x in o]
    l = [x - 0.0005 for x in o]
    c = [x + 0.0001 for x in o]
    img = pack_chart(o, h, l, c)
    assert img.shape == (1, CHART_H, CHART_BARS)
    assert img.max() <= 1.0 + 1e-6
    assert img.min() >= 0.0


def test_pack_features_shape():
    feat = {
        "impulse": 0.8,
        "rsi": 62.0,
        "atr_ratio": 1.1,
        "kalman": {"residual": -0.2, "uncertainty": 0.3, "agreement": 0.7},
        "close": 1.17,
    }
    x = pack_features(feat, [1.16, 1.165, 1.17])
    assert x.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(x).all()


def test_train_numpy_brain_on_synthetic(tmp_path, monkeypatch):
    import flyfx.brain.tensor_brain as tb

    monkeypatch.setattr(tb, "BRAIN_DIR", tmp_path)
    bars = []
    px = 1.10
    for i in range(220):
        px += 0.0004 if (i // 10) % 2 == 0 else -0.0004
        bars.append(
            {
                "time": 1_700_000_000 + i * 300,
                "open": px,
                "high": px + 0.0003,
                "low": px - 0.0003,
                "close": px,
                "volume": 100.0,
            }
        )
    X, y = build_supervised_set(bars)
    assert len(y) > 20
    brain = train_tensor_brain(bars, "TESTNN", epochs=2, prefer_cuda=False, quiet=True, transfer=False)
    assert brain.n_train > 0
    assert brain.ensemble
    assert len(brain.members) == N_ENSEMBLE
    assert abs(float(brain.weights.sum()) - 1.0) < 1e-6
    assert len({m.name for m in brain.members}) == N_ENSEMBLE
    v = brain.vote({"impulse": 0.9, "rsi": 65, "close": bars[-1]["close"], "ready": True}, [b["close"] for b in bars[-10:]])
    assert v.side in ("BUY", "SELL", "HOLD")
    assert int(v.hold_bars) >= 4
    assert len(v.weights) == N_ENSEMBLE
    assert (tmp_path / "TESTNN.bma.npz").exists() or (tmp_path / "TESTNN.bma.pt").exists()
    assert (tmp_path / "TESTNN.nn_mem.npz").exists()
    loaded = TensorBrain("TESTNN", prefer_cuda=False, ensemble=True)
    assert loaded.load()
    assert loaded.ensemble
    assert len(loaded.members) == N_ENSEMBLE
    v2 = loaded.vote({"impulse": 0.9, "rsi": 65, "close": bars[-1]["close"]}, [b["close"] for b in bars[-10:]])
    assert v2.side == v.side


def test_adversarial_training_runs(tmp_path, monkeypatch):
    import flyfx.brain.tensor_brain as tb

    monkeypatch.setattr(tb, "BRAIN_DIR", tmp_path)
    bars = []
    px = 1.10
    for i in range(220):
        px += 0.0004 if (i // 10) % 2 == 0 else -0.0004
        bars.append(
            {
                "time": 1_700_000_000 + i * 300,
                "open": px,
                "high": px + 0.0003,
                "low": px - 0.0003,
                "close": px,
                "volume": 100.0,
            }
        )
    brain = train_tensor_brain(
        bars, "ADV", epochs=1, prefer_cuda=False, quiet=True, transfer=False, adversarial=True
    )
    assert brain.n_train > 0
    assert "adv" in (brain.note or "").lower()
    # Clean eval still produces a vote.
    v = brain.vote({"impulse": 0.5, "rsi": 50, "close": bars[-1]["close"]}, [b["close"] for b in bars[-8:]])
    assert v.side in ("BUY", "SELL", "HOLD")


def test_retrain_transfers_prior_knowledge(tmp_path, monkeypatch):
    import flyfx.brain.tensor_brain as tb

    monkeypatch.setattr(tb, "BRAIN_DIR", tmp_path)

    def _bars(n=220, drift=0.0004):
        out = []
        px = 1.10
        for i in range(n):
            px += drift if (i // 10) % 2 == 0 else -drift
            out.append(
                {
                    "time": 1_700_000_000 + i * 300,
                    "open": px,
                    "high": px + 0.0003,
                    "low": px - 0.0003,
                    "close": px,
                    "volume": 100.0,
                }
            )
        return out

    first = train_tensor_brain(_bars(), "XFER", epochs=2, prefer_cuda=False, quiet=True, transfer=False)
    assert first.n_train > 0
    w0 = first.weights.copy()
    # Snapshot one member weight for warm-start check.
    m0 = first.members[0].state_dict()
    key = next(iter(m0))
    prior_param = np.asarray(m0[key]).copy()

    second = train_tensor_brain(_bars(drift=0.00035), "XFER", epochs=2, prefer_cuda=False, quiet=True, transfer=True)
    assert "transfer" in (second.note or "").lower() or second.n_train >= first.n_train
    assert (tmp_path / "XFER.nn_mem.npz").exists()
    # Posterior still a valid simplex; not a full wipe away from prior mass.
    assert abs(float(second.weights.sum()) - 1.0) < 1e-6
    # Warm-start should leave parameters correlated with the prior (not random).
    m1 = second.members[0].state_dict()
    now = np.asarray(m1[key])
    corr = np.corrcoef(prior_param.ravel(), now.ravel())[0, 1]
    assert corr > 0.2
    # Blended BMA should not be identical to a pure cold posterior in general,
    # but must stay finite.
    assert np.isfinite(second.weights).all()
    assert np.isfinite(w0).all()


def test_bayes_average_weights_the_better_net():
    probs = np.array(
        [
            [0.1, 0.8, 0.1],
            [0.7, 0.2, 0.1],
        ],
        dtype=np.float64,
    )
    mixed = bayes_average(probs, np.array([0.75, 0.25]))
    assert mixed.shape == (3,)
    assert abs(float(mixed.sum()) - 1.0) < 1e-9
    assert mixed[1] > mixed[0]
    # Lower validation NLL gets more mass even when that net has more weights.
    w = posterior_weights(
        np.array([40.0, 48.0]),
        np.array([5000.0, 100.0]),
        n_train=200,
        n_val=40,
    )
    assert abs(float(w.sum()) - 1.0) < 1e-9
    assert w[0] > w[1]
    assert w[0] < 0.999
    assert len(ENSEMBLE_SPECS) == N_ENSEMBLE


def test_observe_on_single_and_ensemble():
    single = TensorBrain("X", prefer_cuda=False, ensemble=False)
    single.observe({"open": 1.1, "high": 1.2, "low": 1.0, "close": 1.15})
    assert single._closes[-1] == 1.15
    ens = TensorBrain("X", prefer_cuda=False, ensemble=True)
    ens.observe({"open": 1.1, "high": 1.2, "low": 1.0, "close": 1.16})
    assert ens.ensemble
    assert len(ens.members) == N_ENSEMBLE
    assert ens._closes[-1] == 1.16
    # Chart CNN needs torch. Without it the voter is the 10-net average.
    if ens.backend == "numpy":
        vis = TensorBrain("X", prefer_cuda=False, vision=True)
        assert vis.ensemble
        vis.observe({"close": 1.17})
        assert vis._closes[-1] == 1.17


def test_ann_confidence_feeds_confidence_fly():
    from flyfx.vote.node_fusion import confidence_drive

    feat = {"impulse": 0.2}
    pack = {
        "p_edge": 0.5,
        "conf_proxy": 0.5,
        "p_trend": 0.5,
        "p_fade": 0.5,
        "tech_fused": 0.1,
        "trend_fused": 0.1,
    }
    base_bull, base_bear = confidence_drive(feat, pack)
    buy = dict(pack)
    buy["nn_conf"] = 0.9
    buy["nn_sign"] = 1.0
    bull, bear = confidence_drive(feat, buy)
    assert bull > base_bull
    assert bull > bear
    sell = dict(pack)
    sell["nn_conf"] = 0.9
    sell["nn_sign"] = -1.0
    bull_s, bear_s = confidence_drive(feat, sell)
    assert bear_s > bull_s
    assert bear_s > base_bear


def test_tensor_brain_device_string():
    b = TensorBrain("X", prefer_cuda=False)
    assert b.device in ("cpu", "numpy", "cuda")


def test_hidden_size_accepts_3oo4():
    from flyfx.risk.latent import hidden_size

    scored = {"conf": 60.0, "cal_p": 0.55, "n": 10, "similar": 0.6, "similar_w": 0.0}
    ok, note, mult = hidden_size(scored, "3oo4")
    assert ok is True
    assert mult > 0
    ok4, _, mult4 = hidden_size(scored, "4oo4+flow")
    assert ok4 is True
    assert mult4 > 0
    # Classic 2oo3 still works.
    ok2, _, _ = hidden_size(scored, "2oo3")
    assert ok2 is True
