"""Walk-forward dopamine trainer for non-EURUSD books.

W (the measured MaleCNS wiring) never moves. Each pair gets its own PAM/PPL
readout + sensory gains, saved under brains/{SYMBOL}.json.

EURUSD is frozen at the factory book (snapshots/eurusd-snowball-lock-20260919).

  python fly_train.py
      Train default majors (except EURUSD) on history *before* the EURUSD
      eval window, save brains/, then you can --overfit to measure.

  python fly_train.py --weekly
      Update existing brains with the last 7 days of M5 (Yahoo or HST).

  python fly_train.py --pairs GBPUSD,USDJPY --weeks 4
      Custom pair list / lookback.

  python fly_train.py --eval
      After training, replay the EURUSD window on all pairs with
      --use-brain --no-plastic (PairBook stays off unless --pair-book).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import flyfx.trader as fx
from flyfx.brain.fly_brains import FROZEN_PAIRS, seed_frozen_eurusd


def _eval_window() -> tuple[int | None, int | None]:
    hst = fx.find_hst("EURUSD", fx.TIMEFRAMES.get("M5", 5))
    if hst is None:
        return None, None
    ref = fx.load_hst(hst, max(2000, fx.WARMUP_BARS + 20))
    if not ref:
        return None, None
    return int(ref[0]["time"]), int(ref[-1]["time"])


def _load_history(symbol: str, weeks: int) -> tuple[list[dict], str]:
    bars, source = fx.load_pair_bars(symbol, "M5", 0, None, None)
    if bars:
        span_days = (int(bars[-1]["time"]) - int(bars[0]["time"])) / 86400.0
    else:
        span_days = 0.0
    need_days = max(14.0, float(weeks) * 5.0) if weeks > 0 else 14.0
    if (not bars) or span_days < need_days:
        yahoo = fx.load_yahoo_m5(symbol, span="60d")
        if not yahoo:
            yahoo = fx.load_yahoo_m5(symbol, span="1mo")
        if yahoo:
            y_span = (int(yahoo[-1]["time"]) - int(yahoo[0]["time"])) / 86400.0
            if y_span > span_days:
                bars = yahoo
                source = f"yahoo 5m {fx.pair_spec(symbol).get('yahoo')} 60d"
    if not bars:
        return [], source
    if weeks > 0:
        cut = bars[-1]["time"] - int(weeks * 7 * 86400)
        trimmed = [b for b in bars if int(b["time"]) >= cut]
        if len(trimmed) >= fx.WARMUP_BARS + 40:
            bars = trimmed
    from flyfx.exec.history_bars import load_history, remember_bars

    remember_bars(symbol, "M5", bars, quiet=True)
    hist = load_history(symbol, "M5")
    if hist:
        bars = fx.merge_bars(hist, bars)
        source = f"history+{source}"
    return bars, source


def _split_train(bars: list[dict], t0: int | None, weekly: bool) -> list[dict]:
    if not bars:
        return []
    if weekly:
        return list(bars)
    if t0 is None:
        last = int(bars[-1]["time"])
        t0 = last - 7 * 86400
    out = [b for b in bars if int(b["time"]) < int(t0)]
    if len(out) < fx.WARMUP_BARS + 40:
        return []
    return out


def _ns(args: argparse.Namespace, symbol: str, extra: dict | None = None) -> SimpleNamespace:
    spec = fx.pair_spec(symbol)
    blob = {
        "symbol": symbol,
        "lots": 0.01,
        "risk": fx.RISK_PCT,
        "min_lots": fx.MIN_LOTS,
        "max_lots": fx.MAX_LOTS,
        "margin_cap": fx.MARGIN_CAP_PCT,
        "interval_ms": 0,
        "substeps": fx.SUBSTEPS,
        "paper": True,
        "dry_run": False,
        "replay": True,
        "pair_spread": float(spec.get("spread", fx.SPREAD)),
        "spread": float(spec.get("spread", fx.SPREAD)),
        "balance": 100_000.0,
        "leverage": 50.0,
        "commission": 7.0,
        "slippage_pips": 0.2,
        "swap_long": -0.72,
        "swap_short": 0.12,
        "stop_out": 50.0,
        "min_conf": fx.MIN_CONF,
        "adapt_state": str(fx.ROOT / "reports" / f"adapt_train_{symbol}.json"),
        "reset_adapt": True,
        "no_fusion": False,
        "no_snowball": False,
        "no_martingale": False,
        "no_plastic": False,
        "no_banc": False,
        "tax_rate": fx.TAX_RATE,
        "tax_model": "ordinary",
        "dash": None,
        "gui": False,
        "shadow_da": True,
        "save_brain": True,
        "reset_brain": False,
        "bar_source": "",
        "no_pair_book": True,
        "pair_book": False,
        "use_brain": False,
        "use_evo": False,
        "force_brain": False,
        "holdout": False,
        "overfit": False,
        "nest": False,
        "koo9": False,
        "ensemble": False,
        "no_nest": False,
        "legacy_2oo3": False,
        "no_categories": False,
        "categories": False,
        "fuse_cats": "",
        "fuse_dynamic": False,
        "fuse_inds": "",
        "risk_tol": "balanced",
        "no_sugar": False,
        "sugar": False,
        "sugar_amt": 1.0,
        "sugar_dyn": False,
        "cat_inhibit": 0.50,
        "evolve": False,
        "evo_pop": 6,
        "evo_gens": 4,
        "evo_seed": 0,
    }
    if extra:
        blob.update(extra)
    return SimpleNamespace(**blob)


def train(args: argparse.Namespace) -> list[dict]:
    seed_frozen_eurusd()
    root = fx.ensure_connectome(Path(args.connectome))
    brain = fx.FlyBrain(root, substeps=fx.SUBSTEPS)
    t0, t1 = _eval_window()
    if t0 and t1:
        print(
            f"eval window  {time.strftime('%Y-%m-%d %H:%M', time.gmtime(t0))} -> "
            f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(t1))}  UTC  (held out)"
        )
    raw = [p.strip().upper() for p in str(args.pairs).split(",") if p.strip()]
    symbols = [s for s in raw if s not in FROZEN_PAIRS]
    if args.include_eurusd:
        symbols = raw
    if not symbols:
        print("no pairs to train (EURUSD is frozen)")
        return []
    broker = fx.PaperBroker(None)
    rows: list[dict] = []
    for i, sym in enumerate(symbols):
        print(f"\n── train {i+1}/{len(symbols)}  {sym} ──")
        bars, source = _load_history(sym, int(args.weeks))
        train_bars = _split_train(bars, t0, weekly=bool(args.weekly))
        if len(train_bars) < fx.WARMUP_BARS + 20:
            print(f"  skip {sym}: not enough history before eval window ({source or 'none'})")
            continue
        print(
            f"  {len(train_bars)} x M5  {source}  "
            f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(train_bars[0]['time']))} -> "
            f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(train_bars[-1]['time']))}"
        )
        ns = _ns(
            args,
            sym,
            {
                "bar_source": source,
                "reset_brain": (not bool(args.weekly)) or bool(args.reset_brain),
                "use_brain": bool(args.weekly) and not bool(args.reset_brain),
                "force_brain": bool(args.include_eurusd) and str(sym).upper() == "EURUSD",
                "replay_bars": train_bars,
                "no_pair_book": True,
                "evolve": bool(getattr(args, "evolve", False)),
                "evo_pop": int(getattr(args, "evo_pop", 6) or 6),
                "evo_gens": int(getattr(args, "evo_gens", 4) or 4),
                "evo_seed": int(getattr(args, "evo_seed", 0) or 0),
            },
        )
        state = Path(ns.adapt_state)
        if state.exists():
            state.unlink()
        try:
            if bool(getattr(args, "evolve", False)):
                from flyfx.brain.evolve import evolve_pair

                summary = evolve_pair(brain, train_bars, ns, broker)
            else:
                summary = fx.trade_loop(
                    brain,
                    fx.replay_prices(train_bars, ns.pair_spread),
                    ns,
                    broker,
                    delay=0.0,
                )
        except Exception as exc:
            print(f"  fail {sym}: {exc}")
            continue
        if summary:
            summary["source"] = source
            rows.append(summary)
            print(
                f"  trained {sym}  DA n={summary.get('da_n', 0)}  "
                f"fills {summary.get('n', 0)}  net ${summary.get('net_usd', 0):+.2f}  "
                + (
                    f"fit {summary.get('fitness'):+.1f}  "
                    if summary.get("fitness") is not None
                    else ""
                )
                + "(train window only, not the eval week)"
            )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Train per-pair fly dopamine from historical M5")
    p.add_argument("--connectome", default=str(fx.CONNECTOME_DIR))
    p.add_argument("--pairs", default=fx.DEFAULT_PAIRS)
    p.add_argument("--weeks", type=int, default=8, help="lookback weeks of M5 (Yahoo caps ~60d)")
    p.add_argument("--weekly", action="store_true", help="update brains from the last 7 days only")
    p.add_argument("--reset-brain", action="store_true", dest="reset_brain", help="start D from 0 (ignored with --weekly)")
    p.add_argument("--include-eurusd", action="store_true", dest="include_eurusd", help="DANGER: train EURUSD (unfreezes)")
    p.add_argument(
        "--evolve",
        action="store_true",
        help="genetic Train DA: evolve params + dynamic behaviors + PAM/PPL; elite survives as brains/{PAIR}.evo.json",
    )
    p.add_argument("--evo-pop", type=int, default=6, dest="evo_pop", help="evolve population (3–16)")
    p.add_argument("--evo-gens", type=int, default=4, dest="evo_gens", help="evolve generations (2–12)")
    p.add_argument("--evo-seed", type=int, default=0, dest="evo_seed", help="evolve RNG seed (0 = clock)")
    p.add_argument("--eval", action="store_true", help="after training, run the 7-pair eval window")
    args = p.parse_args()
    print(
        "Fly dopamine trainer  W frozen  per-pair PAM/PPL  EURUSD locked"
        + ("  EVOLVE" if bool(getattr(args, "evolve", False)) else "")
    )
    rows = train(args)
    if args.eval:
        print("\n── eval window (loads brains/, EURUSD factory) ──")
        sys.argv = [
            "fly_forex.py",
            "--hst", "auto",
            "--bars", "2000",
            "--reset-adapt",
            "--interval-ms", "0",
            "--overfit",
            "--use-brain",
            "--no-plastic",
        ]
        fx.run(fx.parse_args())
    elif rows:
        print("\nSaved brains/ for trained pairs. Evaluate with:")
        print("  python fly_forex.py --hst auto --bars 2000 --reset-adapt --interval-ms 0 --overfit --use-brain --no-plastic")
        print("  python fly_forex.py --hst auto --bars 2000 --reset-adapt --interval-ms 0 --holdout --use-brain --no-plastic")


if __name__ == "__main__":
    main()
