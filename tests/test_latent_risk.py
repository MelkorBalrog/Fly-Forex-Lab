"""Hidden risk factors: factory identity, and a tape that cuts losing 2oo3 size."""

from flyfx.brain.evolve import genome_from_args, public_genome
from flyfx.risk.bayes_sizer import BayesianSizer, SetupScorer
from flyfx.risk.latent import (
    FACTORY,
    fit_latent,
    hidden_size,
    is_factory,
    score_tape,
    stamp_tape,
)
from flyfx.trader import recovery_scale
from types import SimpleNamespace


def _hx(tag: str, step: int) -> dict:
    return {
        "tag": tag,
        "conf": 60.0,
        "cal": 0.55,
        "n": 0,
        "similar": 0.60,
        "similar_w": 0.0,
        "threshold": 45.0,
        "ens": 1.0,
        "vol_baked": False,
        "koo9": 0.0,
        "book": 1.0,
        "atr_ratio": 1.0,
        "uncert": 0.20,
        "banc": 1.0,
        "sugar": 1.0,
        "risk_pct": 1.0,
        "kelly_raw": 0.80,
        "edge": 0.80,
        "rr_raw": 1.20,
        "streak": 0,
        "step": step,
        "mart_on": False,
        "admit_n_seen": 4,
        "start_eq": 100_000.0,
        "locked": False,
        "lock_ratio": 1.0,
        "r_mult": 0.30,
    }


def _tape() -> list[dict]:
    rows = []
    for i in range(8):
        rows.append({"usd": 220.0, "pips": 12.0, "snowball_adds": 0, "hx": _hx("3oo3", i * 40)})
        rows.append({"usd": -180.0, "pips": -9.0, "snowball_adds": 0, "hx": _hx("2oo3", i * 40 + 20)})
    return stamp_tape(rows, FACTORY)


def test_factory_gate_matches_old_literals():
    scored = {
        "conf": 60.0,
        "cal_p": 0.55,
        "n": 0,
        "similar": 0.6,
        "similar_w": 0.0,
        "ens_mult": 1.0,
    }
    ok, _note, mult = hidden_size(scored, "2oo3", FACTORY, threshold=45.0)
    assert ok
    assert abs(mult - 0.93) < 1e-9
    ok3, _n3, mult3 = hidden_size(scored, "3oo3+flow", FACTORY, threshold=45.0)
    assert ok3
    assert abs(mult3 - (1.18 * 0.93 * 1.06)) < 1e-9
    scorer = SetupScorer(BayesianSizer(), min_conf=45.0)
    ok_s, _note_s, mult_s = scorer.gate(scored, "2oo3")
    assert ok_s and abs(mult_s - mult) < 1e-9


def test_factory_martingale_steps():
    assert recovery_scale(1, 100_000, 100_000, 1.0, True, "2oo3", 0.2, 1.0)[0] == 1.20
    assert recovery_scale(2, 100_000, 100_000, 1.0, True, "2oo3", 0.2, 1.0)[0] == 1.28
    assert recovery_scale(1, 100_000, 100_000, 1.0, True, "2oo3", 0.50, 1.0)[0] == 1.0
    assert is_factory(FACTORY)


def test_short_tape_stays_factory():
    out = fit_latent([{"usd": -10.0, "hx": _hx("2oo3", 1)}])
    assert is_factory(out)
    assert int(out.n_fit or 0) == 0


def test_risk_factors_roundtrip_separate_from_genome(tmp_path, monkeypatch):
    import json

    import flyfx.brain.fly_brains as fb
    from flyfx.risk.latent import dump_latent, load_latent

    monkeypatch.setattr(fb, "BRAIN_DIR", tmp_path)
    fitted = fit_latent(_tape(), current=FACTORY)
    fb.save_brain(
        "GBPUSD",
        {"trend": {"n_updates": 1}},
        None,
        extra={
            "genome": {"risk_tol": "balanced", "params": {}},
            "risk_factors": dump_latent(fitted),
        },
    )
    saved = json.loads((tmp_path / "GBPUSD.json").read_text(encoding="utf-8"))
    assert saved["genome"]["risk_tol"] == "balanced"
    assert "mult_2" not in saved["genome"]
    assert abs(float(saved["risk_factors"]["mult_2"]) - fitted.mult_2) < 1e-9
    fb.save_brain("GBPUSD", {"trend": {"n_updates": 2}}, None, extra={"genome": saved["genome"]})
    kept = json.loads((tmp_path / "GBPUSD.json").read_text(encoding="utf-8"))
    assert abs(float(kept["risk_factors"]["mult_2"]) - fitted.mult_2) < 1e-9
    assert abs(load_latent(kept["risk_factors"]).mult_2 - fitted.mult_2) < 1e-9


def test_losing_2oo3_is_cut_and_rescore_improves():
    tape = _tape()
    raw = score_tape(tape, FACTORY)
    fitted = fit_latent(tape, current=FACTORY)
    assert not is_factory(fitted)
    assert fitted.mult_2 < FACTORY.mult_2
    assert fitted.mult_3 >= FACTORY.mult_3 - 1e-6
    better = score_tape(tape, fitted)
    assert better["net"] > raw["net"] + 25.0
    gene = public_genome(genome_from_args(SimpleNamespace()))
    assert "mult_2" not in gene
    assert "risk_factors" not in gene
    pub = public_genome({"risk_tol": "balanced", "_hx": tape, "params": {}})
    assert "_hx" not in pub
