"""Hand-seeded AUDUSD genomes after random evolve sat out / lost.

AUDUSD spread 1.4 pips trips the 3oo3 gate. Tight ATR (~2.4) dies on 8–10 pip
noise, especially bounce BUYs. Candidates push min_impulse to the floor,
widen SL under balanced (so overlay does not replace it), and bias RSI to SELL.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flyfx.trader as fx
from flyfx.brain.evolve import _eval_member, describe_genome, public_genome, save_evo
from flyfx.brain.fly_brains import load_brain, save_brain
from flyfx.risk.bayes_sizer import PARAM_DEFAULTS
from flyfx.train import _eval_window, _load_history, _ns, _split_train

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "snapshots" / "pre-audusd-evolve-20260921"
OUT = ROOT / "reports" / "audusd-candidates-eval-20260921.json"
SYM = "AUDUSD"


def _params(**overrides) -> dict:
    out = {name: float(PARAM_DEFAULTS[name]) for name in PARAM_DEFAULTS}
    out.update({k: float(v) for k, v in overrides.items()})
    return out


def _gene(params: dict, **flags) -> dict:
    g = {
        "params": params,
        "risk_tol": "balanced",
        "fuse_dynamic": False,
        "sugar": False,
        "sugar_amt": 1.0,
        "sugar_dyn": False,
        "fuse_cats": "",
        "fuse_inds": "all",
        "cat_inhibit": 0.50,
        "no_snowball": False,
        "no_martingale": True,
    }
    g.update(flags)
    return public_genome(g)


# Bounce BUY is the leak; 3oo3 needs tech to fire; balanced keeps sl_atr=3.4.
SELL = _params(
    sl_atr=3.4,
    tp_atr=14.0,
    trail_arm_atr=2.2,
    trail_gap_atr=1.2,
    be_atr=1.0,
    rsi_buy_arm=46.0,
    rsi_buy_fire=56.0,
    rsi_sell_arm=46.0,
    rsi_sell_fire=34.0,
    rsi_cont_lo=28.0,
    rsi_cont_hi=46.0,
    min_impulse=0.40,
    cont_impulse=0.65,
    cooldown_bars=5.0,
    lock_win_pips=6.0,
    max_hold_bars=80.0,
    min_hold_bars=3.0,
)
WIDE = _params(sl_atr=3.4, min_impulse=0.40, cont_impulse=0.65, max_hold_bars=80.0)
EASY = _params(
    sl_atr=3.4,
    min_impulse=0.40,
    cont_impulse=0.65,
    rsi_buy_arm=54.0,
    rsi_buy_fire=66.0,
    rsi_sell_arm=46.0,
    rsi_sell_fire=34.0,
    rsi_cont_lo=28.0,
    rsi_cont_hi=46.0,
    trail_arm_atr=2.2,
    trail_gap_atr=1.2,
    cooldown_bars=5.0,
    max_hold_bars=80.0,
)

CANDIDATES: list[tuple[str, dict]] = [
    ("wide_sl_impulse", _gene(WIDE)),
    ("sell_bias", _gene(SELL)),
    ("sell_bias_sugar", _gene(SELL, sugar=True, sugar_dyn=True, sugar_amt=1.0)),
    ("sell_bias_fusedyn", _gene(SELL, fuse_dynamic=True)),
    ("sell_bias_sugar_fuse", _gene(SELL, sugar=True, sugar_dyn=True, fuse_dynamic=True)),
    ("easy_tech", _gene(EASY, fuse_dynamic=True)),
    ("easy_tech_sugar", _gene(EASY, sugar=True, sugar_dyn=True, fuse_dynamic=True)),
    (
        "sell_bias_agg",
        _gene(SELL, risk_tol="aggressive", no_snowball=True),
    ),
]


def _restore_snapshot() -> None:
    src = SNAP / "brains" / "AUDUSD.json"
    dst = ROOT / "brains" / "AUDUSD.json"
    if src.exists():
        shutil.copy2(src, dst)
        print(f"restore {dst} from snapshot")
    evo = ROOT / "brains" / "AUDUSD.evo.json"
    if evo.exists():
        evo.unlink()
        print("removed leftover evo")


def _keep(gene: dict, plastic: dict, banc: dict, meta: dict) -> None:
    stored = load_brain(SYM) or {}
    save_brain(
        SYM,
        plastic or stored.get("plastic") or {},
        banc or stored.get("banc") or {},
        meta,
        live=False,
        params=dict(gene.get("params") or {}),
        extra={
            "genome": public_genome(gene),
            "cat_flies": stored.get("cat_flies") or {},
            "nine": stored.get("nine") or {},
            "crops": stored.get("crops") or {},
        },
    )
    save_evo(
        SYM,
        public_genome(gene),
        meta=meta,
        fitness=float(meta.get("fitness") or 0.0),
        note=describe_genome(gene),
    )
    print(f"KEEP brains/{SYM}.evo.json  {describe_genome(gene)}")


def main() -> None:
    _restore_snapshot()
    stored = load_brain(SYM) or {}
    plastic = stored.get("plastic") or {}
    banc = stored.get("banc") or {}
    t0, t1 = _eval_window()
    bars, source = _load_history(SYM, 8)
    train_bars = _split_train(bars, t0, weekly=False)
    eval_bars = [b for b in bars if t0 and t1 and t0 <= int(b["time"]) <= t1] if bars else []
    if not eval_bars and bars:
        eval_bars = bars[-2000:]
    print(
        f"data    {source}  all={len(bars)}  train={len(train_bars)}  eval={len(eval_bars)}"
    )
    root = fx.ensure_connectome(fx.CONNECTOME_DIR)
    brain = fx.FlyBrain(root, substeps=fx.SUBSTEPS)
    broker = fx.PaperBroker(None)
    ns = _ns(
        None,
        SYM,
        {
            "bar_source": source,
            "replay_bars": eval_bars,
            "use_brain": False,
            "reset_brain": True,
            "no_pair_book": True,
            "no_plastic": True,
            "save_brain": False,
        },
    )
    rows: list[dict] = []
    for name, gene in CANDIDATES:
        g = public_genome(gene)
        if plastic:
            g["_plastic"] = plastic
        if banc:
            g["_banc"] = banc
        print(f"\n======== {name}  {describe_genome(g)} ========")
        t_m = time.perf_counter()
        scored = _eval_member(brain, eval_bars, ns, broker, g)
        summ = scored.get("_summary") or {}
        row = {
            "name": name,
            "elapsed_s": round(time.perf_counter() - t_m, 1),
            "fitness": scored.get("_fitness"),
            "n": summ.get("n"),
            "wins": summ.get("wins"),
            "losses": summ.get("losses"),
            "net_usd": summ.get("net_usd"),
            "equity": summ.get("equity"),
            "max_dd": summ.get("max_dd"),
            "genome": describe_genome(scored),
        }
        print(
            f"RESULT {name}  n={row['n']}  {row['wins']}W/{row['losses']}L  "
            f"net ${float(row['net_usd'] or 0):+.2f}  "
            f"dd {100.0 * float(row['max_dd'] or 0):.2f}%  {row['elapsed_s']}s"
        )
        rows.append(row)
        scored["_name"] = name

    traded = [r for r in rows if int(r.get("n") or 0) > 0]
    ranked = sorted(rows, key=lambda r: float(r.get("net_usd") or -1e18), reverse=True)
    best_row = ranked[0] if ranked else None
    keep = bool(
        best_row
        and float(best_row.get("net_usd") or 0) > 1.0
        and int(best_row.get("n") or 0) > 0
    )
    out = {
        "source": source,
        "eval_n": len(eval_bars),
        "baseline_net": 0.0,
        "rows": rows,
        "keep": keep,
        "best": best_row,
        "traded": len(traded),
    }
    if keep:
        winner = next(g for n, g in CANDIDATES if n == best_row["name"])
        meta = {
            "n_trades": int(best_row.get("n") or 0),
            "net_usd": float(best_row.get("net_usd") or 0.0),
            "fitness": float(best_row.get("fitness") or 0.0),
            "max_dd": float(best_row.get("max_dd") or 0.0),
            "genome_note": describe_genome(winner),
            "window": "EURUSD-eval-week hold-out",
            "note": "hand-seeded after random evolve sat out",
        }
        _keep(winner, plastic, banc, meta)
        if len(train_bars) >= fx.WARMUP_BARS + 20:
            g = public_genome(winner)
            if plastic:
                g["_plastic"] = plastic
            print("\n======== winner train-window check ========")
            train_ns = _ns(None, SYM, {**vars(ns), "replay_bars": train_bars})
            scored = _eval_member(brain, train_bars, train_ns, broker, g)
            summ = scored.get("_summary") or {}
            out["winner_train"] = {
                "n": summ.get("n"),
                "wins": summ.get("wins"),
                "losses": summ.get("losses"),
                "net_usd": summ.get("net_usd"),
                "max_dd": summ.get("max_dd"),
            }
            print(
                f"RESULT winner_train  n={summ.get('n')}  "
                f"net ${float(summ.get('net_usd') or 0):+.2f}"
            )
    else:
        print("no candidate beat $0 hold-out — snapshot brain kept")
        _restore_snapshot()
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
