"""Walk-forward recalibrate helpers."""

from __future__ import annotations

from types import SimpleNamespace

from flyfx.risk.latent import LatentRisk, dump_latent, load_latent
from flyfx.risk.recalibrate import (
    apply_result_to_args,
    candidate_genomes,
    fingerprint_matches,
    load_calibration,
    save_calibration,
    split_walk_forward,
    tape_fingerprint,
)


def _bars(n: int = 100, t0: int = 1_700_000_000) -> list[dict]:
    out = []
    px = 1.10
    for i in range(n):
        px += 0.0001 * ((i % 7) - 3)
        out.append(
            {
                "time": t0 + i * 300,
                "open": px,
                "high": px + 0.0002,
                "low": px - 0.0002,
                "close": px,
            }
        )
    return out


def test_tape_fingerprint_stable():
    bars = _bars(50)
    a = tape_fingerprint("EURUSD", "M5", bars, source="hst EURUSD5.hst")
    b = tape_fingerprint("eurusd", "m5", bars, source="hst EURUSD5.hst")
    assert fingerprint_matches(a, b)
    assert a["n"] == 50
    assert a["source"] == "hst"
    assert a["t0"] == bars[0]["time"]
    assert a["t1"] == bars[-1]["time"]


def test_fingerprint_mismatch_on_length():
    bars = _bars(50)
    a = tape_fingerprint("EURUSD", "M5", bars, source="yahoo 5m EURUSD=X")
    b = tape_fingerprint("EURUSD", "M5", bars[:-5], source="yahoo 5m EURUSD=X")
    assert not fingerprint_matches(a, b)


def test_fingerprint_mismatch_hst_vs_yahoo():
    bars = _bars(50)
    a = tape_fingerprint("EURUSD", "M5", bars, source="hst EURUSD5.hst")
    b = tape_fingerprint("EURUSD", "M5", bars, source="yahoo 5m EURUSD=X")
    assert a["source"] == "hst"
    assert b["source"] == "yahoo"
    assert not fingerprint_matches(a, b)


def test_split_walk_forward_70_30():
    bars = _bars(200)
    cal, hold = split_walk_forward(bars, 0.70)
    assert len(cal) + len(hold) == 200
    assert len(cal) > len(hold)
    assert abs(len(cal) / 200 - 0.70) < 0.05


def test_save_load_calibration_roundtrip(tmp_path, monkeypatch):
    import flyfx.risk.recalibrate as rec

    monkeypatch.setattr(rec, "REPORTS_DIR", tmp_path)
    bars = _bars(80)
    fp = tape_fingerprint("EURUSD", "M5", bars, source="yahoo 5m EURUSD=X")
    lat = LatentRisk(snow_cushion=1.35, n_fit=8, note="unit")
    gene = {"params": {"min_impulse": 0.55}, "risk_tol": "conservative", "no_snowball": True}
    path = save_calibration(
        "EURUSD",
        fingerprint=fp,
        latent=lat,
        genome=gene,
        holdout={"net_usd": 100.0, "n": 4, "score": 50.0},
        note="test",
        fail_safe=False,
    )
    assert path.exists()
    stored = load_calibration("EURUSD")
    assert stored is not None
    assert fingerprint_matches(stored["fingerprint"], fp)
    assert stored["fingerprint"]["source"] == "yahoo"
    assert float(stored["risk_factors"]["snow_cushion"]) == 1.35
    assert stored["genome"]["risk_tol"] == "conservative"
    assert stored["fail_safe"] is False


def test_candidate_genomes_include_fail_safe():
    args = SimpleNamespace(
        risk_tol="balanced",
        sugar=False,
        no_sugar=True,
        sugar_amt=1.0,
        sugar_dyn=False,
        fuse_dynamic=False,
        fuse_cats="",
        fuse_inds="",
        cat_inhibit=0.50,
        no_snowball=False,
        no_martingale=False,
        seed_params=None,
    )
    names = [n for n, _ in candidate_genomes(args)]
    assert "stem" in names
    assert "fail_safe" in names
    assert "plausibility" in names
    assert "loose_impulse" in names


def test_apply_result_fail_safe_sets_args():
    args = SimpleNamespace(
        risk_tol="balanced",
        sugar=False,
        no_sugar=True,
        sugar_amt=1.0,
        sugar_dyn=False,
        fuse_dynamic=False,
        fuse_cats="",
        fuse_inds="",
        cat_inhibit=0.50,
        no_snowball=False,
        no_martingale=False,
        seed_params=None,
    )
    result = {
        "genome": {
            "params": {"min_impulse": 0.55, "cont_impulse": 0.80},
            "risk_tol": "conservative",
            "no_snowball": True,
            "no_martingale": False,
            "fuse_cats": "",
            "fuse_inds": "",
            "fuse_dynamic": False,
            "sugar": False,
            "sugar_amt": 1.0,
            "sugar_dyn": False,
            "cat_inhibit": 0.50,
        },
        "latent": LatentRisk(),
        "name": "fail_safe",
        "fail_safe": True,
        "skipped": False,
        "holdout": {"net_usd": -50.0, "n": 3},
        "note": "fail-safe",
    }
    apply_result_to_args(args, result)
    assert args.risk_tol == "conservative"
    assert args.no_snowball is True
    assert isinstance(args.seed_latent, dict)
    assert args.apply_latent is True
    assert args.recalibrate_meta["fail_safe"] is True


def test_load_latent_from_dump():
    lat = LatentRisk(snow_cushion=1.42, n_fit=10)
    again = load_latent(dump_latent(lat))
    assert abs(again.snow_cushion - 1.42) < 1e-9
    assert again.n_fit == 10


def test_parse_recalibrate_mode():
    from flyfx.risk.recalibrate import parse_recalibrate_mode, resolve_recalibrate_mode
    from types import SimpleNamespace

    assert parse_recalibrate_mode("train") == "train"
    assert parse_recalibrate_mode("calibrate") == "search"
    assert parse_recalibrate_mode("both") == "both"
    assert parse_recalibrate_mode("off") == "off"
    args = SimpleNamespace(recalibrate_mode="train", recalibrate=True, no_recalibrate=False)
    assert resolve_recalibrate_mode(args) == "train"
    args2 = SimpleNamespace(recalibrate_mode=None, recalibrate=True, no_recalibrate=False)
    assert resolve_recalibrate_mode(args2) == "both"
    args3 = SimpleNamespace(recalibrate_mode=None, recalibrate=False, no_recalibrate=False)
    assert resolve_recalibrate_mode(args3) == "off"
