"""AUDUSD walk-forward at fixed size.

Pre-registered books are scored on bars before the EURUSD eval window.
The untouched window is replayed once, for the train winner only.
Snowball and the scratch step-up stay off. Lots are fixed.
"""
from __future__ import annotations

import calendar
import contextlib
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flyfx.trader as fx
from flyfx.brain.evolve import apply_genome, describe_genome, genome_from_args, public_genome
from flyfx.brain.fly_brains import save_evo
from flyfx.risk.bayes_sizer import PARAM_DEFAULTS
from flyfx.train import _eval_window, _ns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "audusd-fixed-wf.json"
SYM = "AUDUSD"
LOTS = 0.10
MIN_TRAIN_N = 3


def _params(**overrides) -> dict:
    out = {name: float(PARAM_DEFAULTS[name]) for name in PARAM_DEFAULTS}
    out.update({k: float(v) for k, v in overrides.items()})
    return out


def _gene(params: dict) -> dict:
    g = genome_from_args(_ns(None, SYM))
    g["params"] = params
    g["risk_tol"] = "balanced"
    g["sugar"] = False
    g["sugar_dyn"] = False
    g["fuse_dynamic"] = False
    g["no_snowball"] = True
    g["no_martingale"] = True
    return public_genome(g)


FACTORY = _gene(_params())
WIDE = _gene(
    _params(sl_atr=3.4, min_impulse=0.40, cont_impulse=0.65, max_hold_bars=80.0)
)
SELL = _gene(
    _params(
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
)
CANDIDATES = [
    ("factory_fixed", FACTORY),
    ("wide_sl", WIDE),
    ("sell_bias", SELL),
]


def _stamp(text: str) -> int:
    return int(calendar.timegm(time.strptime(text, "%Y-%m-%d %H:%M")))


def _rows_after(path: Path, t0: int | None) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if t0 is not None:
                opened = _stamp(row["open_time"])
                if opened < int(t0):
                    continue
            out.append(row)
    return out


def _score_rows(rows: list[dict]) -> dict:
    pips = sum(float(r["pips"]) for r in rows)
    usd = sum(float(r["usd"]) for r in rows)
    wins = sum(1 for r in rows if float(r["usd"]) > 0)
    return {
        "n": len(rows),
        "wins": wins,
        "losses": len(rows) - wins,
        "net_pips": round(pips, 1),
        "net_usd": round(usd, 2),
    }


def _replay(brain, bars, broker, gene: dict, log) -> Path:
    before = {p.name for p in (ROOT / "reports").glob("trades-AUDUSD-*.csv")}
    ns = _ns(
        None,
        SYM,
        {
            "bar_source": "yahoo 5m AUDUSD=X walk-forward",
            "replay_bars": bars,
            "risk": 0.0,
            "lots": LOTS,
            "legacy_2oo3": True,
            "risk_tol": "balanced",
            "no_plastic": True,
            "freeze_learn": True,
            "no_snowball": True,
            "no_martingale": True,
            "no_sugar": True,
            "sugar": False,
            "trade_bayes": False,
            "profit_recycle": False,
            "hold_risk": False,
            "nn_vote": False,
            "save_brain": False,
            "reset_brain": True,
            "use_brain": False,
            "use_evo": False,
            "no_pair_book": True,
            "pair_book": False,
            "volume_mode": "equity",
            "volume_pct": None,
            "evolve_quiet": False,
            "no_recalibrate": True,
        },
    )
    apply_genome(ns, gene)
    ns.risk = 0.0
    ns.lots = LOTS
    ns.legacy_2oo3 = True
    ns.no_plastic = True
    ns.freeze_learn = True
    with contextlib.redirect_stdout(log):
        fx.trade_loop(brain, fx.replay_prices(bars, ns.pair_spread), ns, broker, delay=0.0)
    new = [
        p
        for p in (ROOT / "reports").glob("trades-AUDUSD-*.csv")
        if p.name not in before
    ]
    if not new:
        raise RuntimeError("replay wrote no trade csv")
    return max(new, key=lambda p: p.stat().st_mtime)


def main() -> None:
    t0, t1 = _eval_window()
    if not t0 or not t1:
        raise SystemExit("no EURUSD M5 window to hold out")
    bars = fx.load_yahoo_m5(SYM, span="60d") or fx.load_yahoo_m5(SYM, span="1mo")
    if not bars:
        raise SystemExit("no AUDUSD Yahoo bars")
    train = [b for b in bars if int(b["time"]) < int(t0)]
    if len(train) > 2500:
        train = train[-2500:]
    warm = 80 * 300
    hold = [b for b in bars if int(t0) - warm <= int(b["time"]) <= int(t1)]
    print(
        f"yahoo {len(bars)}  train {len(train)}  hold {len(hold)}  "
        f"cut {time.strftime('%Y-%m-%d %H:%M', time.gmtime(t0))} -> "
        f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(t1))} UTC"
    )
    if len(train) < fx.WARMUP_BARS + 40 or len(hold) < fx.WARMUP_BARS + 20:
        raise SystemExit("window too short")
    root = fx.ensure_connectome(fx.CONNECTOME_DIR)
    brain = fx.FlyBrain(root, substeps=fx.SUBSTEPS)
    broker = fx.PaperBroker(None)
    log_path = ROOT / "reports" / "_audusd_fixed_wf.log"
    scored: list[dict] = []
    with log_path.open("w", encoding="utf-8") as log:
        for name, gene in CANDIDATES:
            t_m = time.perf_counter()
            csv_path = _replay(brain, train, broker, gene, log)
            row = _score_rows(_rows_after(csv_path, None))
            row["name"] = name
            row["csv"] = csv_path.name
            row["elapsed_s"] = round(time.perf_counter() - t_m, 1)
            row["genome"] = describe_genome(gene)
            scored.append(row)
            print(
                f"TRAIN {name}  n={row['n']}  {row['wins']}W/{row['losses']}L  "
                f"{row['net_pips']:+.1f} pips  ${row['net_usd']:+.2f}  {row['elapsed_s']}s",
                flush=True,
            )
        eligible = [r for r in scored if int(r["n"]) >= MIN_TRAIN_N]
        if not eligible:
            blob = {
                "promoted": False,
                "reason": "no pre-registered book traded at least 3 times on the train slice",
                "train": scored,
                "holdout": None,
            }
            OUT.write_text(json.dumps(blob, indent=2), encoding="utf-8")
            print("no book qualified on train; holdout not scored")
            return
        winner = max(eligible, key=lambda r: (float(r["net_pips"]), int(r["n"])))
        gene = dict(CANDIDATES[[n for n, _ in CANDIDATES].index(winner["name"])][1])
        print(f"LOCKED {winner['name']}  train {winner['net_pips']:+.1f} pips", flush=True)
        t_m = time.perf_counter()
        csv_path = _replay(brain, hold, broker, gene, log)
        hold_row = _score_rows(_rows_after(csv_path, t0))
        hold_row["name"] = winner["name"]
        hold_row["csv"] = csv_path.name
        hold_row["elapsed_s"] = round(time.perf_counter() - t_m, 1)
        print(
            f"HOLDOUT {hold_row['name']}  n={hold_row['n']}  "
            f"{hold_row['wins']}W/{hold_row['losses']}L  "
            f"{hold_row['net_pips']:+.1f} pips  ${hold_row['net_usd']:+.2f}",
            flush=True,
        )
    promote = (
        float(winner["net_pips"]) > 0
        and float(hold_row["net_pips"]) > 0
        and int(hold_row["n"]) >= 1
    )
    blob = {
        "promoted": promote,
        "lots": LOTS,
        "rules": "legacy 2oo3, fixed lots, no snowball, no martingale, plastic off",
        "train": scored,
        "locked": winner["name"],
        "holdout": hold_row,
    }
    if promote:
        path = save_evo(
            SYM,
            gene,
            meta={"protocol": "fixed-size walk-forward", "train": winner, "holdout": hold_row},
            fitness=float(hold_row["net_pips"]),
            note=describe_genome(gene),
        )
        blob["evo"] = str(path)
        print(f"saved {path}")
    else:
        print("not promoted: train or holdout pips were not positive")
    OUT.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
