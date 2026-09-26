"""Genetic search around Train DA.

W stays frozen. Each member trains PAM/PPL on the history window, and also
carries a genome of AdaptiveParams plus operator behaviors (risk-tol, sugar,
DYNAMIC mix / sugar, fuse allowlist, snowball / martingale). Elites keep their
traces and params (Lamarckian). The fittest genome is what ``save_brain`` writes.

Empty overlays stay out of the mix — fuse-cat bits never fake volume/breadth/sentiment.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import random
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from flyfx.brain.fly_brains import (
    list_brain_symbols,
    list_evo_policies,
    load_brain,
    load_evo,
    save_brain,
    save_evo,
)
from flyfx.brain.sugar_feed import parse_sugar_amt
from flyfx.paths import REPORTS_DIR
from flyfx.risk.bayes_sizer import PARAM_BOUNDS, PARAM_DEFAULTS
from flyfx.risk.tolerance import NAMES as RISK_NAMES, parse_risk_tol

# Evolve mutates among swing profiles only — scalp is an operator mode package.
EVOLVE_RISK_NAMES = tuple(n for n in RISK_NAMES if n != "scalp")
from flyfx.sense.category_kalman import (
    CATEGORY_NAMES,
    encode_fuse_inds,
    parse_fuse_cats,
    parse_fuse_inds,
)

DENSE_CATS = ("trend", "momentum", "volatility", "structure")
POP_MIN, POP_MAX = 3, 16
GEN_MIN, GEN_MAX = 2, 12
POP_DEFAULT = 6
GEN_DEFAULT = 4


def clip_pop(val, default: int = POP_DEFAULT) -> int:
    return _clip_int(val, POP_MIN, POP_MAX, default)


def clip_gens(val, default: int = GEN_DEFAULT) -> int:
    return _clip_int(val, GEN_MIN, GEN_MAX, default)


def _clip_int(val, lo: int, hi: int, default: int) -> int:
    try:
        n = int(val)
    except (TypeError, ValueError):
        return int(default)
    return max(int(lo), min(int(hi), n))


def _sugar_on(args) -> bool:
    if bool(getattr(args, "no_sugar", False)):
        return False
    return bool(getattr(args, "sugar", False))


def public_genome(gene: dict | None) -> dict:
    """Drop private eval caches before JSON / save_brain."""
    out: dict = {}
    src = gene or {}
    for key, val in src.items():
        if str(key).startswith("_"):
            continue
        out[key] = copy.deepcopy(val)
    return out


def describe_genome(gene: dict | None) -> str:
    g = gene or {}
    params = g.get("params") or {}
    sugar = bool(g.get("sugar"))
    fuse = str(g.get("fuse_cats") or "").strip()
    try:
        fuse_s = "all" if not fuse else ",".join(parse_fuse_cats(fuse))
    except ValueError:
        fuse_s = fuse or "all"
    sl = float(params.get("sl_atr", PARAM_DEFAULTS["sl_atr"]))
    tp = float(params.get("tp_atr", PARAM_DEFAULTS["tp_atr"]))
    inds = str(g.get("fuse_inds") or "all")
    inds_s = "" if inds in ("", "all", "*") else f"  inds={inds}"
    return (
        f"risk={g.get('risk_tol') or 'balanced'}  "
        f"sugar {'ON' if sugar else 'OFF'}×{float(g.get('sugar_amt') or 1):.2f}  "
        f"sdyn {'ON' if (sugar and g.get('sugar_dyn')) else 'OFF'}  "
        f"fuse-dyn {'ON' if g.get('fuse_dynamic') else 'OFF'}  "
        f"fuse={fuse_s}{inds_s}  SL {sl:.2f} TP {tp:.1f}  "
        f"snow {'off' if g.get('no_snowball') else 'on'}  "
        f"mart {'off' if g.get('no_martingale') else 'on'}"
    )


def genome_from_args(args) -> dict:
    params = {name: float(PARAM_DEFAULTS[name]) for name in PARAM_DEFAULTS}
    seed = getattr(args, "seed_params", None)
    if isinstance(seed, dict):
        for name in PARAM_DEFAULTS:
            if name in seed:
                lo, hi = PARAM_BOUNDS[name]
                params[name] = float(min(hi, max(lo, float(seed[name]))))
    try:
        risk = parse_risk_tol(str(getattr(args, "risk_tol", "balanced") or "balanced"))
    except ValueError:
        risk = "balanced"
    sugar = _sugar_on(args)
    fuse = str(getattr(args, "fuse_cats", "") or "")
    try:
        parse_fuse_cats(fuse)
    except ValueError:
        fuse = ""
    raw_inds = getattr(args, "fuse_inds", "") or ""
    try:
        fuse_inds = encode_fuse_inds(parse_fuse_inds(raw_inds))
    except ValueError:
        fuse_inds = str(raw_inds)
    return {
        "params": params,
        "risk_tol": risk,
        "fuse_dynamic": bool(getattr(args, "fuse_dynamic", False)),
        "sugar": sugar,
        "sugar_amt": parse_sugar_amt(getattr(args, "sugar_amt", 1.0)),
        "sugar_dyn": bool(getattr(args, "sugar_dyn", False)) and sugar,
        "fuse_cats": fuse,
        "fuse_inds": fuse_inds,
        "cat_inhibit": float(getattr(args, "cat_inhibit", 0.50) or 0.50),
        "no_snowball": bool(getattr(args, "no_snowball", False)),
        "no_martingale": bool(getattr(args, "no_martingale", False)),
    }


def apply_genome(args, gene: dict | None) -> None:
    """Write genome fields onto a Namespace. Does not touch freeze 2oo3 / W."""
    g = gene or {}
    params = g.get("params")
    if isinstance(params, dict):
        clipped: dict = {}
        for name in PARAM_DEFAULTS:
            if name not in params:
                continue
            lo, hi = PARAM_BOUNDS[name]
            clipped[name] = float(min(hi, max(lo, float(params[name]))))
        args.seed_params = clipped
    try:
        args.risk_tol = parse_risk_tol(str(g.get("risk_tol") or "balanced"))
    except ValueError:
        args.risk_tol = "balanced"
    sugar = bool(g.get("sugar"))
    args.sugar = sugar
    args.no_sugar = not sugar
    args.sugar_amt = parse_sugar_amt(g.get("sugar_amt"))
    args.sugar_dyn = bool(g.get("sugar_dyn")) and sugar
    args.fuse_dynamic = bool(g.get("fuse_dynamic"))
    fuse = str(g.get("fuse_cats") or "")
    try:
        parse_fuse_cats(fuse)
        args.fuse_cats = fuse
    except ValueError:
        args.fuse_cats = ""
    if "fuse_inds" in g:
        raw_inds = str(g.get("fuse_inds") or "")
        try:
            args.fuse_inds = encode_fuse_inds(parse_fuse_inds(raw_inds))
        except ValueError:
            args.fuse_inds = raw_inds
    try:
        inh = float(g.get("cat_inhibit", getattr(args, "cat_inhibit", 0.50)) or 0.50)
    except (TypeError, ValueError):
        inh = 0.50
    args.cat_inhibit = float(min(0.80, max(0.25, inh)))
    args.no_snowball = bool(g.get("no_snowball"))
    args.no_martingale = bool(g.get("no_martingale"))


def apply_saved_genome(args, stored: dict | None, *, live: bool) -> bool:
    """Load surviving genome into args. ``live=True`` skips (EA / chart Inputs own mix)."""
    if live or not stored:
        return False
    gene = stored.get("genome")
    if not isinstance(gene, dict) or not gene:
        return False
    apply_genome(args, gene)
    return True


def apply_policy(args, *, live: bool, stored_brain: dict | None = None) -> str:
    """Load evolved params + flags. Prefers brains/{PAIR}.evo.json.

    When ``live=True``, skips (chart EA Inputs own fuse/risk/sugar).
    GUI Live and ``--use-evo`` / ``--use-brain`` pass ``live=False`` so policy matches Replay.
    ``--use-evo`` / GUI **use evolved** selects the policy file.
    ``--use-brain`` still applies a nested genome when no .evo.json exists.
    """
    if live:
        return ""
    tag = str(getattr(args, "symbol", "") or "").upper()
    use_evo = bool(getattr(args, "use_evo", False))
    use_brain = bool(getattr(args, "use_brain", False)) and not bool(
        getattr(args, "reset_brain", False)
    )
    if not use_evo and not use_brain:
        return ""
    gene = None
    src = ""
    if use_evo:
        blob = load_evo(tag)
        if blob and isinstance(blob.get("genome"), dict) and blob["genome"]:
            gene = blob["genome"]
            src = f"brains/{tag}.evo.json"
    if gene is None and (use_evo or use_brain):
        stored = stored_brain
        if stored is None and use_brain:
            stored = load_brain(tag)
        if stored and isinstance(stored.get("genome"), dict) and stored["genome"]:
            gene = stored["genome"]
            src = f"brains/{tag}.json"
    if not gene:
        return ""
    apply_genome(args, gene)
    return src


def score_summary(summary: dict | None, start: float) -> float:
    """Net USD minus drawdown, with a light overtrade penalty. Zero fills rank last."""
    s = summary or {}
    net = float(s.get("net_usd") or 0.0)
    n = int(s.get("n") or 0)
    dd = float(s.get("max_dd") or 0.0)
    start = float(start or 100_000.0)
    if n < 1:
        return -1_000_000.0 + net
    score = net - (1.25 * dd * start)
    if n > 40:
        score -= 25.0 * (n - 40)
    wins = int(s.get("wins") or 0)
    if n >= 4:
        score += (wins / n - 0.45) * 400.0
    return float(score)


def max_dd_frac(trades: list, start: float) -> float:
    peak = float(start or 0.0)
    eq = float(start or 0.0)
    dd = 0.0
    if peak <= 0:
        return 0.0
    for row in trades or []:
        if isinstance(row, dict) and "equity" in row:
            eq = float(row.get("equity") or eq)
        elif isinstance(row, dict):
            eq += float(row.get("usd") or 0.0)
        peak = max(peak, eq)
        if peak > 0:
            dd = max(dd, (peak - eq) / peak)
    return float(dd)


def mutate(gene: dict, rng: random.Random, rate: float = 0.30) -> dict:
    child = public_genome(gene)
    params = dict(child.get("params") or {})
    for name, (lo, hi) in PARAM_BOUNDS.items():
        cur = float(params.get(name, PARAM_DEFAULTS[name]))
        if rng.random() < rate:
            span = float(hi - lo) or 1.0
            cur = cur + rng.gauss(0.0, 0.12 * span)
        params[name] = float(min(hi, max(lo, cur)))
    child["params"] = params
    if rng.random() < rate:
        child["risk_tol"] = rng.choice(tuple(EVOLVE_RISK_NAMES))
    if rng.random() < rate:
        child["fuse_dynamic"] = not bool(child.get("fuse_dynamic"))
    if rng.random() < rate:
        child["sugar"] = not bool(child.get("sugar"))
    if rng.random() < rate:
        amt = float(child.get("sugar_amt") or 1.0) + rng.gauss(0.0, 0.25)
        child["sugar_amt"] = float(min(2.0, max(0.0, amt)))
    if rng.random() < rate:
        child["sugar_dyn"] = not bool(child.get("sugar_dyn"))
    child["sugar_dyn"] = bool(child.get("sugar_dyn")) and bool(child.get("sugar"))
    if rng.random() < rate:
        inh = float(child.get("cat_inhibit") or 0.50) + rng.gauss(0.0, 0.06)
        child["cat_inhibit"] = float(min(0.80, max(0.25, inh)))
    if rng.random() < rate:
        child["no_snowball"] = not bool(child.get("no_snowball"))
    if rng.random() < rate:
        child["no_martingale"] = not bool(child.get("no_martingale"))
    if rng.random() < rate:
        child["fuse_cats"] = _mutate_fuse_cats(str(child.get("fuse_cats") or ""), rng)
    return child


def _mutate_fuse_cats(spec: str, rng: random.Random) -> str:
    try:
        names = list(parse_fuse_cats(spec))
    except ValueError:
        names = list(CATEGORY_NAMES)
    pick = rng.choice(CATEGORY_NAMES)
    if pick in names:
        if len(names) > 1:
            names.remove(pick)
    else:
        names.append(pick)
    if not any(n in DENSE_CATS for n in names):
        names.append("trend")
    ordered = [n for n in CATEGORY_NAMES if n in names]
    if ordered == list(CATEGORY_NAMES):
        return ""
    return ",".join(ordered)


def crossover(a: dict, b: dict, rng: random.Random) -> dict:
    child = public_genome(a)
    other = public_genome(b)
    params = dict(child.get("params") or {})
    op = other.get("params") or {}
    for name in PARAM_DEFAULTS:
        if rng.random() < 0.50 and name in op:
            params[name] = float(op[name])
    child["params"] = params
    for key in (
        "risk_tol",
        "fuse_dynamic",
        "sugar",
        "sugar_amt",
        "sugar_dyn",
        "fuse_cats",
        "cat_inhibit",
        "no_snowball",
        "no_martingale",
    ):
        if rng.random() < 0.50:
            child[key] = copy.deepcopy(other.get(key))
    child["sugar_dyn"] = bool(child.get("sugar_dyn")) and bool(child.get("sugar"))
    child["sugar_amt"] = parse_sugar_amt(child.get("sugar_amt"))
    try:
        child["risk_tol"] = parse_risk_tol(str(child.get("risk_tol") or "balanced"))
    except ValueError:
        child["risk_tol"] = "balanced"
    return child


def _tournament(scored: list[dict], rng: random.Random, k: int = 3) -> dict:
    pool = scored or []
    if not pool:
        return genome_from_args(SimpleNamespace())
    k = max(1, min(k, len(pool)))
    picks = rng.sample(pool, k)
    return max(picks, key=lambda g: float(g.get("_fitness") or -1e18))


def _stem_from_disk(args, stem: dict) -> dict:
    tag = str(getattr(args, "symbol", "") or "")
    want_evo = bool(getattr(args, "use_evo", False))
    want_brain = bool(getattr(args, "use_brain", False)) and not bool(
        getattr(args, "reset_brain", False)
    )
    if not want_evo and not want_brain:
        return stem
    stored_evo = load_evo(tag) if want_evo else None
    stored = load_brain(tag) if want_brain else None
    if stored_evo and isinstance(stored_evo.get("genome"), dict) and stored_evo["genome"]:
        merged = public_genome(stored_evo["genome"])
        merged["params"] = dict(stem["params"])
        if isinstance(stored_evo.get("genome", {}).get("params"), dict):
            merged["params"] = {**merged["params"], **(stored_evo["genome"].get("params") or {})}
        if stored and isinstance(stored.get("params"), dict) and stored["params"]:
            merged["params"] = {**merged["params"], **stored["params"]}
        stem = merged
    elif stored:
        if isinstance(stored.get("genome"), dict) and stored["genome"]:
            merged = public_genome(stored["genome"])
            merged["params"] = dict(stem["params"])
            if isinstance(stored.get("params"), dict) and stored["params"]:
                merged["params"] = {**merged["params"], **stored["params"]}
            stem = merged
        elif isinstance(stored.get("params"), dict) and stored["params"]:
            stem = public_genome(stem)
            stem["params"] = {**stem["params"], **stored["params"]}
    if stored and stored.get("plastic"):
        stem = public_genome(stem)
        stem["_plastic"] = copy.deepcopy(stored["plastic"])
    if stored and stored.get("banc"):
        stem["_banc"] = copy.deepcopy(stored["banc"])
    return stem


def _eval_member(brain, bars, args, broker, gene: dict) -> dict:
    from flyfx.trader import replay_prices, trade_loop

    handle, tmp = tempfile.mkstemp(prefix="adapt_evo_", suffix=".json")
    os.close(handle)
    ns = SimpleNamespace(**{k: v for k, v in vars(args).items()})
    apply_genome(ns, gene)
    ns.save_brain = False
    ns.shadow_da = True
    ns.reset_brain = True
    ns.use_brain = False
    ns.use_evo = False
    ns.evolve_quiet = True
    ns.evolve_capture = True
    ns.seed_params = dict(gene.get("params") or {})
    ns.seed_plastic = copy.deepcopy(gene.get("_plastic")) if gene.get("_plastic") else None
    ns.seed_banc = copy.deepcopy(gene.get("_banc")) if gene.get("_banc") else None
    ns.adapt_state = tmp
    ns.reset_adapt = True
    # Genome search already walks the dials. The settings fleet learns on the
    # foreground replay, not inside every fitness evaluation.
    ns.settings_ann = False
    ns.replay = True
    ns.interval_ms = 0
    ns.replay_bars = bars
    start = float(getattr(args, "balance", 100_000.0) or 100_000.0)
    inherited = copy.deepcopy(gene.get("_plastic")) if gene.get("_plastic") else None
    gene = public_genome(gene)
    summary: dict = {}
    devnull = open(os.devnull, "w", encoding="utf-8")
    try:
        with contextlib.redirect_stdout(devnull):
            summary = trade_loop(
                brain,
                replay_prices(bars, float(getattr(ns, "pair_spread", 0.00012) or 0.00012)),
                ns,
                broker,
                delay=0.0,
            ) or {}
    finally:
        devnull.close()
        try:
            Path(tmp).unlink(missing_ok=True)
        except OSError:
            pass
    if isinstance(summary.get("params"), dict) and summary["params"]:
        gene["params"] = dict(summary["params"])
    gene["_plastic"] = copy.deepcopy(summary.get("plastic") or inherited or {})
    gene["_banc"] = copy.deepcopy(summary.get("banc") or {})
    gene["_summary"] = {
        "n": int(summary.get("n") or 0),
        "wins": int(summary.get("wins") or 0),
        "losses": int(summary.get("losses") or 0),
        "net_usd": float(summary.get("net_usd") or 0.0),
        "equity": float(summary.get("equity") or start),
        "da_n": int(summary.get("da_n") or 0),
        "max_dd": float(summary.get("max_dd") or 0.0),
    }
    gene["_max_dd"] = float(summary.get("max_dd") or 0.0)
    gene["_da_n"] = int(summary.get("da_n") or 0)
    gene["_fitness"] = score_summary(gene["_summary"], start)
    gene["_hx"] = list(summary.get("hx_trades") or [])
    return gene


def evolve_pair(brain, bars: list, args, broker) -> dict | None:
    """Run pop × gens Train DA evaluations. Save the surviving brain. W never moves."""
    if not bars:
        return None
    pop = clip_pop(getattr(args, "evo_pop", POP_DEFAULT))
    gens = clip_gens(getattr(args, "evo_gens", GEN_DEFAULT))
    elite_n = 1 if pop < 5 else 2
    seed = int(getattr(args, "evo_seed", 0) or 0)
    if seed <= 0:
        seed = (int(time.time()) ^ (len(bars) * 10007)) & 0x7FFFFFFF
    rng = random.Random(seed)
    stem = _stem_from_disk(args, genome_from_args(args))
    population = [public_genome(stem)]
    if stem.get("_plastic"):
        population[0]["_plastic"] = copy.deepcopy(stem["_plastic"])
    while len(population) < pop:
        population.append(mutate(stem, rng, rate=0.55))
    dash = getattr(args, "dash", None)
    history: list[dict] = []
    best: dict | None = None
    print(
        f"evolve  pop {pop}  gens {gens}  elite {elite_n}  seed {seed}  "
        f"bars {len(bars)}  stem  {describe_genome(stem)}"
    )
    cancelled = False
    for g in range(gens):
        scored: list[dict] = []
        for i, gene in enumerate(population):
            if dash is not None and getattr(dash, "cancel", None) is not None and dash.cancel.is_set():
                cancelled = True
                print("evolve  stop")
                break
            if gene.get("_fitness") is not None and gene.get("_plastic") is not None and g > 0 and i < elite_n:
                scored.append(gene)
                continue
            if dash is not None:
                best_s = f"{best['_fitness']:+.0f}" if best is not None else "—"
                dash.publish(
                    {
                        "running": True,
                        "mode": "train",
                        "symbol": str(getattr(args, "symbol", "")),
                        "label": f"evolve gen {g+1}/{gens}  {i+1}/{pop}  best {best_s}",
                    }
                )
            print(f"evolve  gen {g+1}/{gens}  member {i+1}/{pop}  {describe_genome(gene)}")
            gene = _eval_member(brain, bars, args, broker, gene)
            summ = gene.get("_summary") or {}
            print(
                f"         fit {gene['_fitness']:+.1f}  net ${float(summ.get('net_usd') or 0):+.2f}  "
                f"n={int(summ.get('n') or 0)}  dd {100.0 * float(gene.get('_max_dd') or 0):.1f}%  "
                f"DA n={int(gene.get('_da_n') or 0)}"
            )
            scored.append(gene)
            if best is None or float(gene["_fitness"]) > float(best["_fitness"]):
                best = gene
        if cancelled:
            break
        scored.sort(key=lambda x: float(x.get("_fitness") or -1e18), reverse=True)
        if scored and (best is None or float(scored[0]["_fitness"]) >= float(best["_fitness"])):
            best = scored[0]
        history.append(
            {
                "gen": g + 1,
                "best_fitness": float((scored[0] if scored else {}).get("_fitness") or -1e18),
                "best": describe_genome(scored[0] if scored else {}),
                "net_usd": float(((scored[0] if scored else {}).get("_summary") or {}).get("net_usd") or 0.0),
            }
        )
        print(
            f"evolve  gen {g+1}  survivor  fit {float(scored[0]['_fitness']):+.1f}  "
            f"{describe_genome(scored[0])}"
        )
        elites = [copy.deepcopy(s) for s in scored[:elite_n]]
        children: list[dict] = []
        while len(elites) + len(children) < pop:
            if pop >= 6 and len(children) == 0 and rng.random() < 0.35:
                kid = mutate(stem, rng, rate=0.90)
            else:
                p1 = _tournament(scored, rng)
                p2 = _tournament(scored, rng)
                kid = mutate(crossover(p1, p2, rng), rng, rate=0.28)
                parent = p1 if float(p1.get("_fitness") or -1e18) >= float(p2.get("_fitness") or -1e18) else p2
                if parent.get("_plastic"):
                    kid["_plastic"] = copy.deepcopy(parent["_plastic"])
            children.append(kid)
        population = elites + children
    if best is None:
        return None
    apply_genome(args, best)
    args.seed_params = dict(best.get("params") or {})
    meta = {
        "n_trades": int((best.get("_summary") or {}).get("n") or 0),
        "da_n": int(best.get("_da_n") or 0),
        "fitness": float(best.get("_fitness") or 0.0),
        "max_dd": float(best.get("_max_dd") or 0.0),
        "net_usd": float((best.get("_summary") or {}).get("net_usd") or 0.0),
        "evolve": {"pop": pop, "gens": gens, "seed": seed, "history": history},
        "genome_note": describe_genome(best),
    }
    extra = {"genome": public_genome(best)}
    path = save_brain(
        str(getattr(args, "symbol", "")),
        copy.deepcopy(best.get("_plastic") or {}),
        copy.deepcopy(best.get("_banc") or {}),
        meta,
        live=False,
        force=bool(getattr(args, "force_brain", False)),
        params=dict(best.get("params") or {}),
        extra=extra,
    )
    evo_file = save_evo(
        str(getattr(args, "symbol", "")),
        public_genome(best),
        meta=meta,
        fitness=float(best.get("_fitness") or 0.0),
        note=describe_genome(best),
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = REPORTS_DIR / time.strftime(
        f"evolve-{getattr(args, 'symbol', 'PAIR')}-%Y%m%d-%H%M%S.json"
    )
    log_path.write_text(
        json.dumps(
            {
                "symbol": str(getattr(args, "symbol", "")),
                "pop": pop,
                "gens": gens,
                "seed": seed,
                "cancelled": cancelled,
                "best_fitness": float(best.get("_fitness") or 0.0),
                "best": describe_genome(best),
                "summary": best.get("_summary"),
                "genome": public_genome(best),
                "history": history,
                "brain": str(path) if path else None,
                "evo": str(evo_file) if evo_file else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    tag = str(getattr(args, "symbol", "")).upper()
    if evo_file:
        print(f"evolve  survivor  brains/{tag}.evo.json  {describe_genome(best)}")
        print(
            f"         fit {float(best['_fitness']):+.1f}  "
            f"net ${float((best.get('_summary') or {}).get('net_usd') or 0):+.2f}  "
            f"log {log_path.name}"
        )
    if path:
        print(f"         DA     brains/{tag}.json")
    if dash is not None:
        dash.publish(
            {
                "running": False,
                "mode": "idle",
                "label": (
                    f"evolved brains/{tag}.evo.json  fit {float(best['_fitness']):+.0f}  "
                    f"{describe_genome(best)}  — check use evolved, then Replay"
                    if evo_file
                    else f"evolve finished (no save)  {describe_genome(best)}"
                ),
                "brains": list_brain_symbols(),
                "evo": list_evo_policies(),
            }
        )
    out = dict(best.get("_summary") or {})
    out["symbol"] = tag
    out["fitness"] = float(best.get("_fitness") or 0.0)
    out["da_n"] = int(best.get("_da_n") or 0)
    out["genome"] = public_genome(best)
    out["report"] = str(log_path)
    out["evo"] = str(evo_file) if evo_file else None
    return out
