"""One-shot eval: freeze book vs cat-fuse default vs evolved EURUSD on the HST window."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flyfx.trader as fx
from flyfx.brain.fly_brains import load_evo
from flyfx.brain.evolve import describe_genome

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "eurusd-evolve-eval-20260921.json"


def _run(label: str, argv: list[str], brain, broker) -> dict:
    sys.argv = ["fly_forex.py", *argv]
    args = fx.parse_args()
    args.replay = True
    args.interval_ms = 0
    args.reset_adapt = True
    print(f"\n======== {label} ========")
    hst = fx.find_hst("EURUSD", fx.TIMEFRAMES.get("M5", 5))
    bars = fx.load_hst(hst, max(2000, fx.WARMUP_BARS + 20)) if hst else []
    if not bars:
        return {"label": label, "error": "no HST"}
    args.symbol = "EURUSD"
    spec = fx.pair_spec("EURUSD")
    args.pair_spread = float(spec.get("spread", getattr(args, "spread", fx.SPREAD)))
    args.bar_source = f"hst {hst.name}"
    args.replay_bars = bars
    args.adapt_state = str(ROOT / "reports" / f"adapt_eval_{label}.json")
    state = Path(args.adapt_state)
    if state.exists():
        state.unlink()
    t0 = time.perf_counter()
    summary = fx.trade_loop(brain, fx.replay_prices(bars, args.pair_spread), args, broker, delay=0.0) or {}
    summary["label"] = label
    summary["elapsed_s"] = round(time.perf_counter() - t0, 1)
    summary["window"] = {
        "n_bars": len(bars),
        "start": time.strftime("%Y-%m-%d %H:%M", time.gmtime(bars[0]["time"])),
        "end": time.strftime("%Y-%m-%d %H:%M", time.gmtime(bars[-1]["time"])),
        "source": args.bar_source,
    }
    print(
        f"RESULT {label}  n={summary.get('n')}  "
        f"{summary.get('wins')}W/{summary.get('losses')}L  "
        f"net ${float(summary.get('net_usd') or 0):+.2f}  "
        f"eq ${float(summary.get('equity') or 0):.2f}  "
        f"dd {100.0 * float(summary.get('max_dd') or 0):.2f}%"
    )
    return summary


def main() -> None:
    root = fx.ensure_connectome(fx.CONNECTOME_DIR)
    brain = fx.FlyBrain(root, substeps=fx.SUBSTEPS)
    broker = fx.PaperBroker(None)
    evo = load_evo("EURUSD") or {}
    cases = [
        (
            "freeze_2oo3",
            [
                "--replay",
                "--hst",
                "auto",
                "--bars",
                "2000",
                "--legacy-2oo3",
                "--no-sugar",
                "--risk-tol",
                "balanced",
            ],
        ),
        (
            "catfuse_default",
            [
                "--replay",
                "--hst",
                "auto",
                "--bars",
                "2000",
                "--no-sugar",
                "--risk-tol",
                "balanced",
            ],
        ),
        (
            "evolved",
            [
                "--replay",
                "--hst",
                "auto",
                "--bars",
                "2000",
                "--force-brain",
                "--use-evo",
                "--use-brain",
                "--no-plastic",
            ],
        ),
    ]
    rows = []
    for label, argv in cases:
        rows.append(_run(label, argv, brain, broker))
    blob = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "train": {
            "window": "Yahoo M5 2026-08-28 -> 2026-09-09 (held out eval week)",
            "pop": 6,
            "gens": 4,
            "seed": 21,
            "in_sample_net_usd": -263.95,
            "in_sample_fitness": -638.95,
            "da_n": 232,
            "evo_file": "brains/EURUSD.evo.json",
            "brain_file": "brains/EURUSD.json",
            "backup": "snapshots/pre-eurusd-evolve-20260921",
            "genome_note": (evo.get("note") or describe_genome(evo.get("genome"))),
            "genome": evo.get("genome") or {},
        },
        "eval_window": "HST EURUSD M5 2026-09-09 -> 2026-09-18 (freeze book window)",
        "freeze_book_target": {"net_usd": 1906.66, "n": 4, "note": "legacy-2oo3 --no-sugar --risk-tol balanced"},
        "results": rows,
    }
    OUT.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
