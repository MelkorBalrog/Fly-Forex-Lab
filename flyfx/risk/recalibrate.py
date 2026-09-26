"""Walk-forward recalibration for new tapes.

Auto path (Replay): 70/30 split, small genome grid, fit_latent, holdout pick.
Explicit path: full evolve_pair, then persist risk_factors + fingerprint.

Profit is sought on the holdout slice only — never guaranteed.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from flyfx.brain.evolve import (
    apply_genome,
    genome_from_args,
    public_genome,
    score_summary,
)
from flyfx.paths import REPORTS_DIR
from flyfx.risk.latent import (
    MIN_TRADES,
    LatentRisk,
    dump_latent,
    fit_latent,
    load_latent,
)
from flyfx.risk.bayes_sizer import PARAM_BOUNDS, PARAM_DEFAULTS

CAL_FRAC = 0.70
MIN_BARS = 80

# What auto-recalibrate runs before Replay.
# train  = tensor AI only (fast)
# search = genome/latent walk-forward only (slow)
# both   = train then search (default when recalibrate on)
# off    = skip
RECAL_MODES = ("off", "train", "search", "both")


def parse_recalibrate_mode(raw) -> str:
    s = str(raw or "").strip().lower()
    if s in ("calibrate", "calib", "genome"):
        return "search"
    if s in ("ai", "nn", "train-only", "train_only"):
        return "train"
    if s in ("full", "all", "train+search", "train+calibrate"):
        return "both"
    if s in RECAL_MODES:
        return s
    return "both"


def resolve_recalibrate_mode(args) -> str:
    """Prefer explicit --recalibrate-mode; else map boolean recalibrate flag."""
    raw = getattr(args, "recalibrate_mode", None)
    if raw is not None and str(raw).strip() != "":
        return parse_recalibrate_mode(raw)
    if bool(getattr(args, "no_recalibrate", False)):
        return "off"
    if getattr(args, "recalibrate", None) is False:
        return "off"
    if getattr(args, "recalibrate", None) is True:
        return "both"
    return "off"


def _source_kind(source: str) -> str:
    s = str(source or "").strip().lower()
    if not s:
        return "unknown"
    if "yahoo" in s:
        return "yahoo"
    if s.endswith(".hst") or s.startswith("hst ") or "/history/" in s.replace("\\", "/"):
        return "hst"
    if "upload" in s or s.endswith(".csv") or s.endswith(".json"):
        return "file"
    return s[:48]


def tape_fingerprint(
    symbol: str,
    timeframe: str,
    bars: list[dict] | None,
    source: str = "",
) -> dict:
    """Identify the exact replay window (symbol, TF, span, source, endpoint prices)."""
    rows = bars or []
    if not rows:
        return {
            "symbol": str(symbol or "").upper(),
            "timeframe": str(timeframe or "M5").upper(),
            "n": 0,
            "t0": 0,
            "t1": 0,
            "source": _source_kind(source),
            "c0": 0.0,
            "c1": 0.0,
            "cm": 0.0,
        }
    mid = rows[len(rows) // 2]
    return {
        "symbol": str(symbol or "").upper(),
        "timeframe": str(timeframe or "M5").upper(),
        "n": int(len(rows)),
        "t0": int(rows[0].get("time") or 0),
        "t1": int(rows[-1].get("time") or 0),
        "source": _source_kind(source),
        "c0": round(float(rows[0].get("close") or 0.0), 6),
        "c1": round(float(rows[-1].get("close") or 0.0), 6),
        "cm": round(float(mid.get("close") or 0.0), 6),
    }


def fingerprint_matches(a: dict | None, b: dict | None) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    keys = ("symbol", "timeframe", "n", "t0", "t1", "source", "c0", "c1", "cm")
    try:
        return all(a.get(k) == b.get(k) for k in keys)
    except Exception:
        return False


def describe_window(bars: list[dict] | None, source: str = "") -> str:
    rows = bars or []
    if not rows:
        return f"empty  source={source or '?'}"
    t0 = time.strftime("%Y-%m-%d %H:%M", time.gmtime(int(rows[0]["time"])))
    t1 = time.strftime("%Y-%m-%d %H:%M", time.gmtime(int(rows[-1]["time"])))
    return f"{len(rows)} bars  {_source_kind(source) or source or '?'}  {t0} -> {t1} UTC"


def split_walk_forward(
    bars: list[dict], cal_frac: float = CAL_FRAC
) -> tuple[list[dict], list[dict]]:
    n = len(bars or [])
    if n < MIN_BARS:
        return list(bars or []), []
    cut = int(round(n * float(cal_frac)))
    cut = max(MIN_BARS // 2, min(n - max(20, n // 10), cut))
    return list(bars[:cut]), list(bars[cut:])


def risk_factors_path(symbol: str) -> Path:
    return REPORTS_DIR / f"risk_factors_{str(symbol or '').upper()}.json"


def save_calibration(
    symbol: str,
    *,
    fingerprint: dict,
    latent: LatentRisk | dict | None,
    genome: dict | None = None,
    holdout: dict | None = None,
    note: str = "",
    fail_safe: bool = False,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if isinstance(latent, LatentRisk):
        rf = dump_latent(latent)
    elif isinstance(latent, dict):
        rf = dict(latent)
    else:
        rf = dump_latent(LatentRisk())
    blob = {
        "symbol": str(symbol or "").upper(),
        "fingerprint": fingerprint,
        "risk_factors": rf,
        "genome": public_genome(genome) if genome else {},
        "holdout": holdout or {},
        "note": str(note or ""),
        "fail_safe": bool(fail_safe),
    }
    path = risk_factors_path(symbol)
    path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return path


def load_calibration(symbol: str) -> dict | None:
    path = risk_factors_path(symbol)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _clip_params(params: dict) -> dict:
    out: dict = {}
    for name in PARAM_DEFAULTS:
        if name not in params:
            continue
        lo, hi = PARAM_BOUNDS[name]
        out[name] = float(min(hi, max(lo, float(params[name]))))
    return out


def candidate_genomes(args) -> list[tuple[str, dict]]:
    """Small grid around the operator genome (entry gates + fuse + risk)."""
    base = public_genome(genome_from_args(args))
    base["params"] = _clip_params(dict(base.get("params") or PARAM_DEFAULTS))
    out: list[tuple[str, dict]] = [("stem", copy.deepcopy(base))]

    loose = copy.deepcopy(base)
    loose["params"] = _clip_params(
        {**loose["params"], "min_impulse": 0.45, "cont_impulse": 0.70}
    )
    out.append(("loose_impulse", loose))

    tight = copy.deepcopy(base)
    tight["params"] = _clip_params(
        {**tight["params"], "min_impulse": 0.58, "cont_impulse": 0.85}
    )
    out.append(("tight_impulse", tight))

    plaus = copy.deepcopy(base)
    plaus["fuse_inds"] = "plausibility"
    out.append(("plausibility", plaus))

    soft_snow = copy.deepcopy(base)
    soft_snow["params"] = _clip_params(
        {**soft_snow["params"], "min_impulse": 0.48, "cont_impulse": 0.72}
    )
    out.append(("soft_entry", soft_snow))

    safe = copy.deepcopy(base)
    safe["risk_tol"] = "conservative"
    safe["no_snowball"] = True
    safe["params"] = _clip_params(
        {**safe["params"], "min_impulse": 0.55, "cont_impulse": 0.80}
    )
    out.append(("fail_safe", safe))
    return out


def _hx_rows(summary: dict | None) -> list[dict]:
    rows = []
    for row in (summary or {}).get("hx_trades") or []:
        if not isinstance(row, dict):
            continue
        hx = dict(row.get("hx") or {})
        if float(hx.get("live_used") or 0.0) <= 0.0:
            continue
        packed = dict(row)
        packed["hx"] = hx
        rows.append(packed)
    return rows


def _run_quiet(brain, bars: list[dict], args, broker, gene: dict, seed_latent: dict | None = None) -> dict:
    from flyfx.trader import replay_prices, trade_loop

    handle, tmp = tempfile.mkstemp(prefix="adapt_recal_", suffix=".json")
    os.close(handle)
    ns = SimpleNamespace(**{k: v for k, v in vars(args).items()})
    apply_genome(ns, gene)
    ns.save_brain = False
    ns.shadow_da = False
    ns.reset_brain = True
    ns.use_brain = False
    ns.use_evo = False
    ns.recalibrate = False
    ns.evolve_quiet = True
    ns.evolve_capture = True
    ns.seed_params = dict(gene.get("params") or {})
    ns.seed_latent = dict(seed_latent) if isinstance(seed_latent, dict) else None
    ns.apply_latent = True
    # Keep tensor brain + 3oo4 through quiet candidate evals.
    if getattr(args, "seed_tensor_brain", None) is not None:
        ns.seed_tensor_brain = getattr(args, "seed_tensor_brain")
        ns.nn_vote = True
    ns.adapt_state = tmp
    ns.reset_adapt = True
    ns.settings_ann = False
    ns.replay = True
    ns.interval_ms = 0
    ns.replay_bars = bars
    start = float(getattr(args, "balance", 100_000.0) or 100_000.0)
    summary: dict = {}
    devnull = open(os.devnull, "w", encoding="utf-8")
    try:
        with contextlib.redirect_stdout(devnull):
            summary = (
                trade_loop(
                    brain,
                    replay_prices(bars, float(getattr(ns, "pair_spread", 0.00012) or 0.00012)),
                    ns,
                    broker,
                    delay=0.0,
                )
                or {}
            )
    finally:
        devnull.close()
        try:
            Path(tmp).unlink(missing_ok=True)
        except OSError:
            pass
    summary["_fitness"] = score_summary(summary, start)
    return summary


def _pick_fail_safe(args) -> tuple[str, dict, LatentRisk]:
    for name, gene in candidate_genomes(args):
        if name == "fail_safe":
            return name, gene, LatentRisk()
    gene = public_genome(genome_from_args(args))
    gene["risk_tol"] = "conservative"
    gene["no_snowball"] = True
    return "fail_safe", gene, LatentRisk()


def apply_result_to_args(args, result: dict) -> None:
    """Write chosen genome + latent onto the live Namespace."""
    gene = result.get("genome")
    if isinstance(gene, dict) and gene:
        apply_genome(args, gene)
        args.seed_params = dict(gene.get("params") or {})
    lat = result.get("latent")
    if isinstance(lat, LatentRisk):
        args.seed_latent = dump_latent(lat)
    elif isinstance(lat, dict) and lat:
        args.seed_latent = dict(lat)
    args.apply_latent = True
    if result.get("nn") or getattr(args, "seed_tensor_brain", None) is not None:
        args.nn_vote = True
    args.recalibrate_meta = {
        "name": result.get("name"),
        "holdout": result.get("holdout"),
        "fail_safe": bool(result.get("fail_safe")),
        "skipped": bool(result.get("skipped")),
        "note": result.get("note") or "",
        "nn": result.get("nn") or {},
    }


def auto_recalibrate(brain, bars: list[dict], args, broker) -> dict:
    """Train AI and/or walk-forward genome search on the exact bars about to be traded."""
    symbol = str(getattr(args, "symbol", "") or "").upper()
    tf = str(getattr(args, "timeframe", "M5") or "M5")
    source = str(getattr(args, "bar_source", "") or "")
    fp = tape_fingerprint(symbol, tf, bars, source=source)
    start = float(getattr(args, "balance", 100_000.0) or 100_000.0)
    mode = resolve_recalibrate_mode(args)
    # GUI Replay always sets reset_adapt — force a fresh fit on this window.
    force = bool(getattr(args, "reset_adapt", False)) or bool(
        getattr(args, "force_recalibrate", False)
    )
    print(f"recalibrate  mode={mode}  window  {describe_window(bars, source)}")

    if mode == "off":
        return {
            "applied": False,
            "skipped": True,
            "fail_safe": False,
            "name": "off",
            "genome": public_genome(genome_from_args(args)),
            "latent": LatentRisk(),
            "holdout": {},
            "fingerprint": fp,
            "note": "recalibrate mode off",
        }

    # Evolved / cache skip only for search modes (not train-only).
    if mode == "search" and bool(getattr(args, "use_evo", False)) and not force:
        from flyfx.brain.fly_brains import load_evo

        evo = load_evo(symbol)
        if evo and isinstance(evo.get("genome"), dict) and evo["genome"]:
            meta = evo.get("meta") if isinstance(evo.get("meta"), dict) else {}
            evo_fp = meta.get("fingerprint")
            if fingerprint_matches(evo_fp, fp):
                lat = load_latent(meta.get("risk_factors") if isinstance(meta.get("risk_factors"), dict) else None)
                return {
                    "applied": True,
                    "skipped": True,
                    "fail_safe": False,
                    "name": "evo_fingerprint",
                    "genome": public_genome(evo["genome"]),
                    "latent": lat,
                    "holdout": meta.get("holdout") or {},
                    "fingerprint": fp,
                    "note": "use evolved fingerprint match — skip auto search",
                }

    stored = load_calibration(symbol)
    if (
        mode == "search"
        and stored
        and fingerprint_matches(stored.get("fingerprint"), fp)
        and not force
    ):
        return {
            "applied": True,
            "skipped": True,
            "fail_safe": bool(stored.get("fail_safe")),
            "name": "cached",
            "genome": public_genome(stored.get("genome") or genome_from_args(args)),
            "latent": load_latent(stored.get("risk_factors")),
            "holdout": stored.get("holdout") or {},
            "fingerprint": fp,
            "note": stored.get("note") or "fingerprint match — reuse risk_factors",
        }
    if force and stored and fingerprint_matches(stored.get("fingerprint"), fp) and mode != "train":
        print("recalibrate  force refresh on this window (reset-adapt) — ignoring cache")

    cal_bars, hold_bars = split_walk_forward(bars)

    # Train the tensor AI when mode is train or both.
    nn_brain = None
    want_nn = (
        mode in ("train", "both")
        and bool(getattr(args, "nn_vote", True))
        and not bool(getattr(args, "no_nn_vote", False))
    )
    train_bars = list(bars)
    train_note = source
    if want_nn:
        from flyfx.trader import bars_for_training

        train_bars, train_note = bars_for_training(symbol, tf, bars, source)
        print(f"nn-brain  corpus  {train_note}")
    if want_nn and len(train_bars) >= MIN_BARS // 2:
        from flyfx.brain.tensor_brain import train_tensor_brain

        mixer = "legacy" if bool(getattr(args, "legacy_2oo3", False)) else "cat_fuse"
        print(
            f"nn-brain  training on history + current bars  {describe_window(train_bars, train_note)}…"
        )
        nn_brain = train_tensor_brain(
            train_bars,
            symbol,
            epochs=int(getattr(args, "nn_epochs", 25) or 25),
            prefer_cuda=not bool(getattr(args, "nn_cpu", False)),
            mixer=mixer,
            fuse_cats=str(getattr(args, "fuse_cats", "") or ""),
            fuse_inds=str(getattr(args, "fuse_inds", "") or ""),
            quiet=False,
            vision=bool(getattr(args, "nn_vision", False)),
            transfer=not bool(getattr(args, "nn_fresh", False)),
            adversarial=bool(getattr(args, "nn_adv", True)),
        )
        args.nn_vote = True
        args.seed_tensor_brain = nn_brain
    elif bool(getattr(args, "nn_vote", False)) and not bool(getattr(args, "no_nn_vote", False)):
        args.nn_vote = True

    # The 20 settings controllers are not the 10-voter. Train and print them
    # on this same window. Quiet genome evals keep settings_ann off, so they
    # never enter this block.
    if (
        bool(getattr(args, "settings_ann", False))
        and not bool(getattr(args, "no_plastic", False))
        and not bool(getattr(args, "freeze_learn", False))
    ):
        from flyfx.brain.settings_ann import train_settings_fleet

        settings_fleet = train_settings_fleet(
            train_bars,
            symbol,
            fuse_inds=str(getattr(args, "fuse_inds", "") or ""),
            quiet=False,
            transfer=not bool(getattr(args, "nn_fresh", False)),
        )
        if settings_fleet is not None:
            args.seed_settings_fleet = settings_fleet

    # Train-only: skip the slow genome search; replay with stem knobs + fresh NN.
    if mode == "train":
        gene = public_genome(genome_from_args(args))
        lat = LatentRisk()
        if stored and isinstance(stored.get("risk_factors"), dict):
            lat = load_latent(stored.get("risk_factors"))
        note = (
            f"train-only  nn={'yes' if nn_brain is not None else 'skipped'}  "
            f"— genome search off"
        )
        save_calibration(
            symbol,
            fingerprint=fp,
            latent=lat,
            genome=gene,
            holdout={},
            note=note,
            fail_safe=False,
        )
        return {
            "applied": True,
            "skipped": False,
            "fail_safe": False,
            "name": "train",
            "genome": gene,
            "latent": lat,
            "holdout": {},
            "fingerprint": fp,
            "note": note,
            "nn": (
                {
                    "note": getattr(nn_brain, "note", ""),
                    "device": getattr(nn_brain, "device", ""),
                    "backend": getattr(nn_brain, "backend", ""),
                    "n_train": int(getattr(nn_brain, "n_train", 0) or 0),
                }
                if nn_brain is not None
                else {}
            ),
        }

    if len(cal_bars) < MIN_BARS // 2 or len(hold_bars) < 20:
        name, gene, lat = _pick_fail_safe(args)
        note = "window too short for walk-forward — fail-safe"
        save_calibration(
            symbol,
            fingerprint=fp,
            latent=lat,
            genome=gene,
            holdout={"net_usd": 0.0, "n": 0, "score": 0.0},
            note=note,
            fail_safe=True,
        )
        return {
            "applied": True,
            "skipped": False,
            "fail_safe": True,
            "name": name,
            "genome": gene,
            "latent": lat,
            "holdout": {"net_usd": 0.0, "n": 0, "score": 0.0},
            "fingerprint": fp,
            "note": note,
        }

    best: dict | None = None
    best_score = -1e18
    cands = candidate_genomes(args)
    print(
        f"recalibrate  genome search  {len(cands)} candidates × "
        f"(cal {len(cal_bars)} + hold {len(hold_bars)} bars) — can take several minutes…"
    )
    for i, (name, gene) in enumerate(cands, start=1):
        print(f"recalibrate  candidate {i}/{len(cands)}  {name}  cal pass…", flush=True)
        cal_sum = _run_quiet(brain, cal_bars, args, broker, gene)
        hx = _hx_rows(cal_sum)
        lat = fit_latent(hx, LatentRisk()) if len(hx) >= MIN_TRADES else LatentRisk()
        # Soft snow cushion nudge for soft_entry candidate
        if name == "soft_entry" and int(lat.n_fit or 0) == 0:
            lat = LatentRisk(snow_cushion=1.30)
        print(f"recalibrate  candidate {i}/{len(cands)}  {name}  holdout pass…", flush=True)
        hold_sum = _run_quiet(
            brain, hold_bars, args, broker, gene, seed_latent=dump_latent(lat)
        )
        score = float(hold_sum.get("_fitness") or score_summary(hold_sum, start))
        net = float(hold_sum.get("net_usd") or 0.0)
        n = int(hold_sum.get("n") or 0)
        dd = float(hold_sum.get("max_dd") or 0.0)
        print(
            f"recalibrate  candidate {i}/{len(cands)}  {name}  "
            f"holdout ${net:+.0f}  n={n}  score {score:+.0f}",
            flush=True,
        )
        cand = {
            "name": name,
            "genome": public_genome(gene),
            "latent": lat,
            "holdout": {
                "net_usd": net,
                "n": n,
                "wins": int(hold_sum.get("wins") or 0),
                "losses": int(hold_sum.get("losses") or 0),
                "max_dd": dd,
                "score": score,
            },
            "score": score,
            "net": net,
        }
        # Prefer positive expectancy; among losers keep least-bad.
        better = False
        if best is None:
            better = True
        elif net > 0 and float(best.get("net") or 0.0) <= 0:
            better = True
        elif (net > 0) == (float(best.get("net") or 0.0) > 0) and score > best_score + 1.0:
            better = True
        if better:
            best = cand
            best_score = score

    assert best is not None
    fail_safe = float(best.get("net") or 0.0) <= 0.0
    if fail_safe:
        name, gene, lat = _pick_fail_safe(args)
        # Keep the measured holdout for honesty; mark fail-safe sizing.
        note = (
            f"holdout not profitable (best {best['name']} "
            f"${float(best.get('net') or 0):+.0f}) — fail-safe conservative / no snowball"
        )
        result = {
            "applied": True,
            "skipped": False,
            "fail_safe": True,
            "name": name,
            "genome": gene,
            "latent": lat,
            "holdout": best["holdout"],
            "fingerprint": fp,
            "note": note,
        }
    else:
        note = (
            f"auto {best['name']}  holdout ${float(best['net']):+.2f}  "
            f"n={int(best['holdout'].get('n') or 0)}  score {best_score:+.0f}"
        )
        result = {
            "applied": True,
            "skipped": False,
            "fail_safe": False,
            "name": best["name"],
            "genome": best["genome"],
            "latent": best["latent"],
            "holdout": best["holdout"],
            "fingerprint": fp,
            "note": note,
        }

    result["nn"] = (
        {
            "note": getattr(nn_brain, "note", ""),
            "device": getattr(nn_brain, "device", ""),
            "backend": getattr(nn_brain, "backend", ""),
            "n_train": int(getattr(nn_brain, "n_train", 0) or 0),
        }
        if nn_brain is not None
        else {}
    )
    save_calibration(
        symbol,
        fingerprint=fp,
        latent=result["latent"],
        genome=result["genome"],
        holdout=result["holdout"],
        note=result["note"],
        fail_safe=bool(result["fail_safe"]),
    )
    return result


def explicit_recalibrate(brain, bars: list[dict], args, broker) -> dict | None:
    """Full genetic evolve, then persist risk_factors + fingerprint on the evo meta."""
    from flyfx.brain.evolve import evolve_pair
    from flyfx.brain.fly_brains import load_evo, save_evo

    symbol = str(getattr(args, "symbol", "") or "").upper()
    tf = str(getattr(args, "timeframe", "M5") or "M5")
    source = str(getattr(args, "bar_source", "") or "")
    fp = tape_fingerprint(symbol, tf, bars, source=source)
    from flyfx.trader import bars_for_training

    train_bars, train_note = bars_for_training(symbol, tf, bars, source)
    print(f"recalibrate  explicit  sim {describe_window(bars, source)}")
    print(f"recalibrate  train corpus  {train_note}")
    # Train tensor brain on history + current bars, then evolve on that same corpus.
    if not bool(getattr(args, "no_nn_vote", False)):
        from flyfx.brain.tensor_brain import train_tensor_brain

        mixer = "legacy" if bool(getattr(args, "legacy_2oo3", False)) else "cat_fuse"
        nn_brain = train_tensor_brain(
            train_bars,
            symbol,
            epochs=int(getattr(args, "nn_epochs", 30) or 30),
            prefer_cuda=not bool(getattr(args, "nn_cpu", False)),
            mixer=mixer,
            fuse_cats=str(getattr(args, "fuse_cats", "") or ""),
            fuse_inds=str(getattr(args, "fuse_inds", "") or ""),
            vision=bool(getattr(args, "nn_vision", False)),
            transfer=not bool(getattr(args, "nn_fresh", False)),
            adversarial=bool(getattr(args, "nn_adv", True)),
        )
        args.nn_vote = True
        args.seed_tensor_brain = nn_brain
    if (
        bool(getattr(args, "settings_ann", False))
        and not bool(getattr(args, "no_plastic", False))
        and not bool(getattr(args, "freeze_learn", False))
    ):
        from flyfx.brain.settings_ann import train_settings_fleet

        settings_fleet = train_settings_fleet(
            train_bars,
            symbol,
            fuse_inds=str(getattr(args, "fuse_inds", "") or ""),
            quiet=False,
            transfer=not bool(getattr(args, "nn_fresh", False)),
        )
        if settings_fleet is not None:
            args.seed_settings_fleet = settings_fleet
    args.evolve = True
    args.save_brain = True
    args.force_brain = True
    out = evolve_pair(brain, train_bars, args, broker)
    if not out:
        return None
    gene = out.get("genome") if isinstance(out.get("genome"), dict) else None
    # Quiet pass for hx → latent fit on the surviving genome.
    lat = LatentRisk()
    if gene:
        summary = _run_quiet(brain, bars, args, broker, gene)
        hx = _hx_rows(summary)
        if len(hx) >= MIN_TRADES:
            lat = fit_latent(hx, LatentRisk())
        holdout = {
            "net_usd": float(summary.get("net_usd") or 0.0),
            "n": int(summary.get("n") or 0),
            "wins": int(summary.get("wins") or 0),
            "losses": int(summary.get("losses") or 0),
            "max_dd": float(summary.get("max_dd") or 0.0),
            "score": float(summary.get("_fitness") or 0.0),
        }
    else:
        holdout = {
            "net_usd": float(out.get("net_usd") or 0.0),
            "n": int(out.get("n") or 0),
            "score": float(out.get("fitness") or 0.0),
        }
    note = f"explicit evolve  fit {float(out.get('fitness') or 0):+.0f}"
    save_calibration(
        symbol,
        fingerprint=fp,
        latent=lat,
        genome=gene or genome_from_args(args),
        holdout=holdout,
        note=note,
        fail_safe=float(holdout.get("net_usd") or 0.0) <= 0.0,
    )
    # Refresh evo meta with fingerprint + risk_factors so auto can skip.
    evo = load_evo(symbol)
    if evo and isinstance(evo.get("genome"), dict):
        meta = dict(evo.get("meta") or {})
        meta["fingerprint"] = fp
        meta["risk_factors"] = dump_latent(lat)
        meta["holdout"] = holdout
        save_evo(
            symbol,
            evo["genome"],
            meta=meta,
            fitness=float(evo.get("fitness") or out.get("fitness") or 0.0),
            note=str(evo.get("note") or note),
        )
    out["fingerprint"] = fp
    out["risk_factors"] = dump_latent(lat)
    out["recalibrate"] = note
    return out
