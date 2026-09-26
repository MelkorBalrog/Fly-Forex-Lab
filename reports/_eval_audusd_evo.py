"""Backup, baseline, evolve, and hold-out eval for AUDUSD brain params.

Train window: M5 before the EURUSD eval week (same split as fly_train.py).
Eval window: that EURUSD week. W stays frozen. Writes brains/AUDUSD.evo.json
only after evolve_pair; restores the snapshot if hold-out is worse than baseline.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flyfx.trader as fx
from flyfx.brain.evolve import describe_genome, evolve_pair
from flyfx.brain.fly_brains import load_brain, load_evo
from flyfx.train import _eval_window, _load_history, _ns, _split_train

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "snapshots" / "pre-audusd-evolve-20260921"
OUT = ROOT / "reports" / "audusd-evolve-eval-20260921.json"
SYM = "AUDUSD"


def _backup() -> None:
    SNAP.mkdir(parents=True, exist_ok=True)
    (SNAP / "brains").mkdir(exist_ok=True)
    src = ROOT / "brains" / "AUDUSD.json"
    if src.exists():
        shutil.copy2(src, SNAP / "brains" / "AUDUSD.json")
    live = ROOT / "brains" / "AUDUSD.live.json"
    if live.exists():
        shutil.copy2(live, SNAP / "brains" / "AUDUSD.live.json")
    (SNAP / "README.txt").write_text(
        "AUDUSD brain before evolve 21 Sep 2026.\n"
        "Restore:\n"
        "  copy snapshots\\pre-audusd-evolve-20260921\\brains\\AUDUSD.json brains\\AUDUSD.json\n",
        encoding="utf-8",
    )
    print(f"backup  {SNAP / 'brains' / 'AUDUSD.json'}")


def _inspect() -> dict:
    stored = load_brain(SYM) or {}
    params = stored.get("params") or {}
    gene = stored.get("genome") or {}
    meta = stored.get("meta") or {}
    plastic = stored.get("plastic") or {}
    da = {}
    for name, blob in plastic.items():
        if isinstance(blob, dict):
            da[name] = {
                "n_updates": blob.get("n_updates"),
                "gain_pos": blob.get("gain_pos"),
                "gain_neg": blob.get("gain_neg"),
            }
    info = {
        "saved_at": stored.get("saved_at"),
        "meta": meta,
        "params": params,
        "genome_flags": {k: v for k, v in gene.items() if k != "params"},
        "genome_params": gene.get("params") or {},
        "da": da,
    }
    print("brain   saved_at", info["saved_at"])
    print("brain   params", json.dumps(params, indent=2)[:800])
    print("brain   genome", describe_genome(gene) if gene else "(none)")
    print("brain   DA", da)
    return info


def _run(label: str, bars: list, source: str, argv: list[str], brain, broker) -> dict:
    sys.argv = ["fly_forex.py", *argv]
    args = fx.parse_args()
    args.replay = True
    args.interval_ms = 0
    args.reset_adapt = True
    args.symbol = SYM
    spec = fx.pair_spec(SYM)
    args.pair_spread = float(spec.get("spread", getattr(args, "spread", fx.SPREAD)))
    args.bar_source = source
    args.replay_bars = bars
    args.adapt_state = str(ROOT / "reports" / f"adapt_eval_audusd_{label}.json")
    state = Path(args.adapt_state)
    if state.exists():
        state.unlink()
    print(f"\n======== {label}  {len(bars)} bars  {source} ========")
    t0 = time.perf_counter()
    summary = fx.trade_loop(
        brain, fx.replay_prices(bars, args.pair_spread), args, broker, delay=0.0
    ) or {}
    summary["label"] = label
    summary["elapsed_s"] = round(time.perf_counter() - t0, 1)
    if bars:
        summary["window"] = {
            "n_bars": len(bars),
            "start": time.strftime("%Y-%m-%d %H:%M", time.gmtime(bars[0]["time"])),
            "end": time.strftime("%Y-%m-%d %H:%M", time.gmtime(bars[-1]["time"])),
            "source": source,
        }
    print(
        f"RESULT {label}  n={summary.get('n')}  "
        f"{summary.get('wins')}W/{summary.get('losses')}L  "
        f"net ${float(summary.get('net_usd') or 0):+.2f}  "
        f"eq ${float(summary.get('equity') or 0):.2f}  "
        f"dd {100.0 * float(summary.get('max_dd') or 0):.2f}%  "
        f"{summary['elapsed_s']}s"
    )
    return summary


def _restore_brain() -> None:
    src = SNAP / "brains" / "AUDUSD.json"
    dst = ROOT / "brains" / "AUDUSD.json"
    if src.exists():
        shutil.copy2(src, dst)
        print("restore brains/AUDUSD.json from snapshot (hold-out did not improve)")
    evo = ROOT / "brains" / "AUDUSD.evo.json"
    if evo.exists():
        evo.unlink()
        print("removed brains/AUDUSD.evo.json")


def main() -> None:
    _backup()
    info = _inspect()
    t0, t1 = _eval_window()
    bars, source = _load_history(SYM, 8)
    train_bars = _split_train(bars, t0, weekly=False)
    eval_bars: list[dict] = []
    eval_source = source
    if t0 and t1 and bars:
        eval_bars = [b for b in bars if t0 <= int(b["time"]) <= t1]
        eval_source = f"{source} eval-clip"
    if not eval_bars and bars:
        eval_bars = bars[-2000:]
        eval_source = f"{source} last2000"
    print(
        f"data    {source}  all={len(bars)}  train={len(train_bars)}  eval={len(eval_bars)}"
    )
    if train_bars:
        print(
            "train   "
            f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(train_bars[0]['time']))} -> "
            f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(train_bars[-1]['time']))}"
        )
    if eval_bars:
        print(
            "eval    "
            f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(eval_bars[0]['time']))} -> "
            f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(eval_bars[-1]['time']))}"
        )

    root = fx.ensure_connectome(fx.CONNECTOME_DIR)
    brain = fx.FlyBrain(root, substeps=fx.SUBSTEPS)
    broker = fx.PaperBroker(None)

    out: dict = {"inspect": info, "source": source}
    if eval_bars:
        out["baseline"] = _run(
            "baseline_use_brain",
            eval_bars,
            eval_source,
            [
                "--replay",
                "--symbol",
                SYM,
                "--use-brain",
                "--no-plastic",
                "--bars",
                "2000",
            ],
            brain,
            broker,
        )
    else:
        out["baseline"] = {"error": "no eval bars"}

    if len(train_bars) < fx.WARMUP_BARS + 20:
        print("skip evolve: not enough train bars")
        OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
        return

    ns = _ns(
        SimpleNamespace(),
        SYM,
        {
            "bar_source": source,
            "replay_bars": train_bars,
            "use_brain": True,
            "reset_brain": False,
            "force_brain": False,
            "no_pair_book": True,
            "evolve": True,
            "evo_pop": 6,
            "evo_gens": 4,
            "evo_seed": 21,
            # AUDUSD leak is bounce BUYs + tight ATR stops; start search from a
            # stabler book. Mutate still explores sugar / fuse / snowball.
            "no_martingale": True,
            "risk_tol": "aggressive",
        },
    )
    print("\n======== evolve train window ========")
    t_evo = time.perf_counter()
    evo_summary = evolve_pair(brain, train_bars, ns, broker) or {}
    evo_summary["elapsed_s"] = round(time.perf_counter() - t_evo, 1)
    out["evolve"] = {
        "fitness": evo_summary.get("fitness"),
        "net_usd": evo_summary.get("net_usd"),
        "n": evo_summary.get("n"),
        "da_n": evo_summary.get("da_n"),
        "elapsed_s": evo_summary["elapsed_s"],
        "genome": describe_genome(load_evo(SYM).get("genome") if load_evo(SYM) else {}),
        "summary": {k: evo_summary.get(k) for k in ("n", "wins", "losses", "net_usd", "equity", "max_dd", "fitness")},
    }
    print(f"evolve  done  {evo_summary['elapsed_s']}s")

    if eval_bars:
        out["evolved_eval"] = _run(
            "evolved_use_evo",
            eval_bars,
            eval_source,
            [
                "--replay",
                "--symbol",
                SYM,
                "--use-brain",
                "--use-evo",
                "--no-plastic",
                "--bars",
                "2000",
            ],
            brain,
            broker,
        )
        base_net = float((out.get("baseline") or {}).get("net_usd") or 0.0)
        evo_net = float((out.get("evolved_eval") or {}).get("net_usd") or 0.0)
        base_n = int((out.get("baseline") or {}).get("n") or 0)
        evo_n = int((out.get("evolved_eval") or {}).get("n") or 0)
        improved = evo_net > base_net + 1.0
        out["keep_evolved"] = improved
        if not improved:
            _restore_brain()
        else:
            print(
                f"KEEP evolved  eval ${evo_net:+.2f} vs baseline ${base_net:+.2f}"
            )
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
