"""Genetic Train DA: genome mutate/crossover/fitness and elite save."""

from __future__ import annotations

import json
from types import SimpleNamespace

from flyfx.brain.evolve import (
    apply_genome,
    apply_policy,
    apply_saved_genome,
    clip_gens,
    clip_pop,
    crossover,
    describe_genome,
    genome_from_args,
    max_dd_frac,
    mutate,
    public_genome,
    score_summary,
)
from flyfx.risk.bayes_sizer import PARAM_BOUNDS
from flyfx.trader import _gui_apply_common, parse_args


def test_clip_pop_gens():
    assert clip_pop(1) == 3
    assert clip_pop(99) == 16
    assert clip_gens(1) == 2
    assert clip_gens(40) == 12
    assert clip_pop("6") == 6


def test_genome_roundtrip_and_bounds():
    args = SimpleNamespace(
        sugar=True,
        no_sugar=False,
        sugar_amt=0.5,
        sugar_dyn=True,
        fuse_dynamic=True,
        fuse_cats="trend,momentum",
        risk_tol="aggressive",
        cat_inhibit=0.4,
        no_snowball=True,
        no_martingale=False,
        seed_params={"sl_atr": 2.9},
    )
    gene = genome_from_args(args)
    assert gene["sugar"] is True
    assert gene["sugar_dyn"] is True
    assert gene["fuse_dynamic"] is True
    assert gene["risk_tol"] == "aggressive"
    assert abs(gene["params"]["sl_atr"] - 2.9) < 1e-9
    ns = SimpleNamespace()
    apply_genome(ns, gene)
    assert ns.sugar is True
    assert ns.no_sugar is False
    assert ns.fuse_cats == "trend,momentum"
    assert ns.fuse_inds
    assert ns.risk_tol == "aggressive"
    rng = __import__("random").Random(0)
    for _ in range(40):
        gene = mutate(gene, rng, rate=0.90)
        for name, (lo, hi) in PARAM_BOUNDS.items():
            val = float(gene["params"][name])
            assert lo - 1e-9 <= val <= hi + 1e-9
        assert 0.0 <= float(gene["sugar_amt"]) <= 2.0
        if gene["sugar_dyn"]:
            assert gene["sugar"] is True
        names = gene["fuse_cats"]
        parsed = __import__("flyfx.sense.category_kalman", fromlist=["parse_fuse_cats"]).parse_fuse_cats(names)
        assert parsed
        assert any(n in parsed for n in ("trend", "momentum", "volatility", "structure"))
    pub = public_genome({**gene, "_fitness": 1.0, "_plastic": {"x": 1}})
    assert "_fitness" not in pub
    assert "_plastic" not in pub
    assert "sugar" in describe_genome(gene)


def test_crossover_and_fitness():
    rng = __import__("random").Random(2)
    a = genome_from_args(SimpleNamespace(sugar=True, fuse_dynamic=False, risk_tol="balanced"))
    b = genome_from_args(SimpleNamespace(sugar=False, no_sugar=True, fuse_dynamic=True, risk_tol="conservative"))
    child = crossover(a, b, rng)
    assert child["risk_tol"] in ("balanced", "conservative")
    assert score_summary({"n": 0, "net_usd": 10, "max_dd": 0}, 100_000) < -900_000
    good = score_summary({"n": 6, "wins": 4, "net_usd": 800, "max_dd": 0.02}, 100_000)
    bad = score_summary({"n": 6, "wins": 1, "net_usd": -800, "max_dd": 0.20}, 100_000)
    assert good > bad
    trades = [{"equity": 100_000}, {"equity": 110_000}, {"equity": 90_000}]
    assert abs(max_dd_frac(trades, 100_000) - (20_000 / 110_000)) < 1e-9


def test_apply_saved_genome_skips_live():
    stored = {"genome": genome_from_args(SimpleNamespace(sugar=True, risk_tol="aggressive"))}
    args = SimpleNamespace(risk_tol="balanced", sugar=False, no_sugar=True)
    assert apply_saved_genome(args, stored, live=True) is False
    assert args.risk_tol == "balanced"
    assert apply_saved_genome(args, stored, live=False) is True
    assert args.risk_tol == "aggressive"
    assert args.sugar is True


def test_evolve_pair_saves_elite(tmp_path, monkeypatch):
    import flyfx.brain.evolve as ev
    import flyfx.brain.fly_brains as fb

    monkeypatch.setattr(fb, "BRAIN_DIR", tmp_path)
    monkeypatch.setattr(ev, "REPORTS_DIR", tmp_path)
    calls = {"n": 0}

    def fake_loop(brain, prices, args, broker, delay):
        calls["n"] += 1
        sl = float((getattr(args, "seed_params", None) or {}).get("sl_atr") or 2.4)
        net = 4000.0 - abs(sl - 3.05) * 1500.0
        return {
            "n": 4,
            "wins": 3,
            "losses": 1,
            "net_usd": net,
            "equity": 100_000.0 + net,
            "da_n": 2,
            "max_dd": 0.03,
            "plastic": {"trend": {"n_updates": 2, "delta": [0.1]}},
            "params": dict(getattr(args, "seed_params", None) or {"sl_atr": sl}),
            "banc": {},
        }

    monkeypatch.setattr("flyfx.trader.trade_loop", fake_loop)
    args = SimpleNamespace(
        symbol="GBPUSD",
        balance=100_000.0,
        pair_spread=0.00012,
        evo_pop=3,
        evo_gens=2,
        evo_seed=7,
        sugar=False,
        no_sugar=True,
        sugar_amt=1.0,
        sugar_dyn=False,
        fuse_dynamic=False,
        fuse_cats="",
        risk_tol="balanced",
        cat_inhibit=0.50,
        no_snowball=False,
        no_martingale=False,
        dash=None,
        force_brain=False,
        use_brain=False,
        reset_brain=True,
        seed_params=None,
    )
    bars = [{"time": i, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0} for i in range(12)]
    out = ev.evolve_pair(None, bars, args, None)
    assert out is not None
    assert calls["n"] == 5
    path = tmp_path / "GBPUSD.json"
    assert path.exists()
    blob = json.loads(path.read_text(encoding="utf-8"))
    assert blob["symbol"] == "GBPUSD"
    assert blob["genome"]["risk_tol"] in ("balanced", "conservative", "aggressive")
    assert "sl_atr" in blob["params"]
    assert blob["plastic"]["trend"]["n_updates"] == 2
    assert blob["meta"]["evolve"]["pop"] == 3
    assert "fitness" in blob["meta"]
    evo = tmp_path / "GBPUSD.evo.json"
    assert evo.exists()
    evo_blob = json.loads(evo.read_text(encoding="utf-8"))
    assert evo_blob["kind"] == "evo"
    assert evo_blob["symbol"] == "GBPUSD"
    assert evo_blob["genome"]["risk_tol"] in ("balanced", "conservative", "aggressive")
    assert "sl_atr" in evo_blob["genome"]["params"]
    assert out.get("evo")
    logs = list(tmp_path.glob("evolve-GBPUSD-*.json"))
    assert logs


def test_cli_evolve_flags(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "argv", ["fly_forex.py", "--gui"])
    args = parse_args()
    assert args.evolve is False
    assert args.evo_pop == 6
    assert args.evo_gens == 4
    assert args.use_evo is False
    monkeypatch.setattr(sys, "argv", ["fly_forex.py", "--replay", "--evolve", "--evo-pop", "8", "--evo-gens", "3"])
    flagged = parse_args()
    assert flagged.evolve is True
    assert flagged.evo_pop == 8
    assert flagged.evo_gens == 3
    monkeypatch.setattr(sys, "argv", ["fly_forex.py", "--replay", "--use-evo", "--use-brain"])
    loaded = parse_args()
    assert loaded.use_evo is True
    assert loaded.use_brain is True


def test_apply_policy_prefers_evo_file(tmp_path, monkeypatch):
    import flyfx.brain.fly_brains as fb
    import flyfx.brain.evolve as ev

    monkeypatch.setattr(fb, "BRAIN_DIR", tmp_path)
    monkeypatch.setattr(ev, "REPORTS_DIR", tmp_path)
    fb.save_brain(
        "GBPUSD",
        {"trend": {"n_updates": 1}},
        None,
        extra={"genome": {"risk_tol": "balanced", "sugar": False, "params": {"sl_atr": 2.4}}},
    )
    fb.save_evo(
        "GBPUSD",
        {
            "risk_tol": "aggressive",
            "sugar": True,
            "sugar_amt": 1.2,
            "sugar_dyn": False,
            "fuse_dynamic": True,
            "fuse_cats": "trend,momentum",
            "fuse_inds": "ema,rsi",
            "params": {"sl_atr": 3.1, "tp_atr": 4.2},
            "no_snowball": False,
            "no_martingale": True,
        },
        fitness=900.0,
        note="test policy",
    )
    args = SimpleNamespace(
        symbol="GBPUSD",
        use_evo=True,
        use_brain=True,
        reset_brain=False,
        risk_tol="balanced",
        sugar=False,
        no_sugar=True,
        fuse_dynamic=False,
        fuse_cats="",
        fuse_inds="",
    )
    src = apply_policy(args, live=False, stored_brain=fb.load_brain("GBPUSD"))
    assert src.endswith("GBPUSD.evo.json")
    assert args.risk_tol == "aggressive"
    assert args.sugar is True
    assert args.fuse_dynamic is True
    assert args.fuse_cats == "trend,momentum"
    assert abs(float(args.seed_params["sl_atr"]) - 3.1) < 1e-9
    live = SimpleNamespace(symbol="GBPUSD", use_evo=True, use_brain=False, reset_brain=False, risk_tol="balanced")
    assert apply_policy(live, live=True) == ""
    assert live.risk_tol == "balanced"


def test_save_evo_eurusd_does_not_touch_factory_da(tmp_path, monkeypatch):
    import flyfx.brain.fly_brains as fb

    monkeypatch.setattr(fb, "BRAIN_DIR", tmp_path)
    assert fb.save_brain("EURUSD", {"x": 1}, None) is None
    path = fb.save_evo("EURUSD", {"risk_tol": "conservative", "sugar": False, "params": {"sl_atr": 2.0}})
    assert path is not None
    assert path.name == "EURUSD.evo.json"
    assert not (tmp_path / "EURUSD.json").exists()
    assert "EURUSD" in fb.list_evo_symbols()
    assert "EURUSD" not in fb.list_brain_symbols()


def test_gui_apply_evolve():
    args = SimpleNamespace(bars=2000, balance=100000.0, timeframe="M5")
    _gui_apply_common({"evolve": True, "evo_pop": 10, "evo_gens": 5}, args)
    assert args.evolve is True
    assert args.evo_pop == 10
    assert args.evo_gens == 5
    _gui_apply_common({"evolve": False, "evo_pop": 1, "evo_gens": 99}, args)
    assert args.evolve is False
    assert args.evo_pop == 3
    assert args.evo_gens == 12
