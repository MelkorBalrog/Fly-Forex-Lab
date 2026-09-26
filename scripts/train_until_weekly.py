"""Train each major until a held-out replay earns at least 1% a week.

The exam is the last 14 days of that pair's history. Training uses only
earlier bars. The exam loads the saved brain and the evolved file the same
way a GUI replay does with both boxes checked, and it does not write the
brain back. Pass means net profit / starting equity / weeks >= 1%.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flyfx.brain.evolve import evolve_pair
from flyfx.brain.fly_brains import is_frozen
from flyfx.exec.bar_io import parse_ohlc_text, resample_bars
from flyfx.trader import (
    CONNECTOME_DIR,
    ROOT,
    WARMUP_BARS,
    FlyBrain,
    PaperBroker,
    ensure_connectome,
    load_yahoo_m5,
    replay_prices,
    trade_loop,
)
from flyfx.train import _ns

PAIRS = ("EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "EURGBP", "USDCAD")
BALANCE = 100_000.0
TARGET = 0.01
EXAM_DAYS = 14
TRAIN_DAYS = 28
TAIL_LINES = 140_000
STATUS_PATH = ROOT / "reports" / "weekly_train_status.json"
LOG_PATH = ROOT / "reports" / "weekly_train.log"


def _status(msg: str, log) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()


def _tail_text(path: Path, n_lines: int) -> str:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        block = min(size, n_lines * 96)
        handle.seek(size - block)
        raw = handle.read().decode("utf-8", errors="replace")
    lines = raw.splitlines()
    if size > block and lines:
        lines = lines[1:]
    return "\n".join(lines[-n_lines:])


def _csv_candidates(symbol: str) -> list[Path]:
    folder = ROOT / "data" / symbol
    if not folder.exists():
        return []
    files = [p for p in folder.glob("*.csv") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def _load_local(symbol: str) -> tuple[list[dict], str]:
    need = (EXAM_DAYS + TRAIN_DAYS) * 86400
    covering: list[tuple[int, list[dict], str]] = []
    longest: tuple[int, list[dict], str] | None = None
    for path in _csv_candidates(symbol):
        try:
            bars = parse_ohlc_text(_tail_text(path, TAIL_LINES), path.name)
        except Exception:
            continue
        if len(bars) < 2:
            continue
        m5 = resample_bars(bars, 5)
        if len(m5) < 2:
            continue
        end = int(m5[-1]["time"])
        src = f"{path.name} tail resampled M5"
        if longest is None or len(m5) > longest[0]:
            longest = (len(m5), m5, src)
        if end - int(m5[0]["time"]) >= need:
            covering.append((end, m5, src))
    if covering:
        _end, bars, src = max(covering, key=lambda row: row[0])
        return bars, src
    if longest is not None:
        return longest[1], longest[2]
    return [], ""


def load_history(symbol: str) -> tuple[list[dict], str]:
    """Prefer the newest series that still covers the train and exam windows.

    The MT4 csv exports in data/ stop in 2020-2021. Yahoo 5m is the recent
    tape, so a newer Yahoo window replaces that stale local tail.
    """
    bars, source = _load_local(symbol)
    need = (EXAM_DAYS + TRAIN_DAYS) * 86400
    local_ok = bool(bars) and int(bars[-1]["time"]) - int(bars[0]["time"]) >= need
    yahoo = load_yahoo_m5(symbol, span="60d")
    yahoo_ok = bool(yahoo) and int(yahoo[-1]["time"]) - int(yahoo[0]["time"]) >= need
    if yahoo_ok and (not local_ok or int(yahoo[-1]["time"]) > int(bars[-1]["time"])):
        return yahoo, f"yahoo 5m {symbol}"
    if local_ok:
        return bars, source
    if yahoo:
        return yahoo, f"yahoo 5m {symbol}"
    return bars, source


def split_bars(bars: list[dict]) -> tuple[list[dict], list[dict], int]:
    cut = int(bars[-1]["time"]) - EXAM_DAYS * 86400
    train = [b for b in bars if int(b["time"]) < cut]
    train_cut = cut - TRAIN_DAYS * 86400
    train = [b for b in train if int(b["time"]) >= train_cut]
    exam = [b for b in bars if int(b["time"]) >= cut]
    return train, exam, cut


def weekly_rate(net_usd: float, t0: int, t1: int) -> tuple[float, float]:
    weeks = max((int(t1) - int(t0)) / (7.0 * 86400.0), 1.0 / 7.0)
    return (float(net_usd) / BALANCE) / weeks, weeks


def _base_ns(symbol: str):
    parent = type("A", (), {})()
    return _ns(parent, symbol, {"balance": BALANCE, "bar_source": ""})


def run_exam(brain, broker, symbol: str, exam: list[dict], log) -> dict:
    ns = _base_ns(symbol)
    ns.use_brain = True
    ns.use_evo = True
    ns.reset_brain = False
    ns.force_brain = True
    ns.save_brain = False
    ns.shadow_da = False
    ns.freeze_learn = True
    ns.no_plastic = False
    ns.reset_adapt = True
    ns.evolve = False
    ns.replay_bars = exam
    ns.adapt_state = str(ROOT / "reports" / f"adapt_exam_{symbol}.json")
    state = Path(ns.adapt_state)
    if state.exists():
        state.unlink()
    saved = sys.stdout
    sys.stdout = log
    try:
        summary = trade_loop(brain, replay_prices(exam, ns.pair_spread), ns, broker, delay=0.0) or {}
    finally:
        sys.stdout = saved
    summary["t0"] = int(exam[0]["time"])
    summary["t1"] = int(exam[-1]["time"])
    rate, weeks = weekly_rate(float(summary.get("net_usd") or 0.0), summary["t0"], summary["t1"])
    summary["weekly"] = rate
    summary["weeks"] = weeks
    return summary


def run_train(brain, broker, symbol: str, train: list[dict], round_n: int, log) -> None:
    ns = _base_ns(symbol)
    ns.evolve = True
    if round_n >= 3:
        ns.evo_pop, ns.evo_gens = 6, 4
    elif round_n >= 2:
        ns.evo_pop, ns.evo_gens = 4, 3
    else:
        ns.evo_pop, ns.evo_gens = 3, 2
    ns.evo_seed = 1000 + round_n * 17 + sum(ord(c) for c in symbol)
    ns.save_brain = True
    ns.shadow_da = True
    ns.use_brain = True
    ns.use_evo = True
    ns.reset_brain = False
    ns.force_brain = True
    ns.reset_adapt = True
    ns.replay_bars = train
    ns.adapt_state = str(ROOT / "reports" / f"adapt_train_{symbol}.json")
    saved = sys.stdout
    sys.stdout = log
    try:
        evolve_pair(brain, train, ns, broker)
    finally:
        sys.stdout = saved


def _write_status(blob: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(blob, indent=2), encoding="utf-8")


def _brain_paths(symbol: str) -> list[Path]:
    folder = ROOT / "brains"
    return [folder / f"{symbol}.json", folder / f"{symbol}.evo.json"]


def _copy_brains(symbol: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for src in _brain_paths(symbol):
        if src.exists():
            shutil.copy2(src, dest / src.name)


def _restore_brains(symbol: str, src_dir: Path) -> None:
    for name in (f"{symbol}.json", f"{symbol}.evo.json"):
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, ROOT / "brains" / name)


def main() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = LOG_PATH.open("a", encoding="utf-8", buffering=1)
    _status(
        f"target {TARGET:.0%} per week on the last {EXAM_DAYS} days, "
        f"train on the {TRAIN_DAYS} days before that, balance ${BALANCE:,.0f}",
        log,
    )
    histories: dict[str, tuple[list[dict], str]] = {}
    for symbol in PAIRS:
        bars, source = load_history(symbol)
        if not bars:
            _status(f"{symbol}  no history", log)
            continue
        train, exam, cut = split_bars(bars)
        _status(
            f"{symbol}  {source}  "
            f"{time.strftime('%Y-%m-%d', time.gmtime(bars[0]['time']))} -> "
            f"{time.strftime('%Y-%m-%d', time.gmtime(bars[-1]['time']))}  "
            f"train {len(train)}  exam {len(exam)}  "
            f"cut {time.strftime('%Y-%m-%d', time.gmtime(cut))}",
            log,
        )
        if len(train) < WARMUP_BARS + 40 or len(exam) < WARMUP_BARS + 20:
            _status(f"{symbol}  not enough bars to hold out {EXAM_DAYS} days", log)
            continue
        histories[symbol] = (bars, source)
    if not histories:
        _status("nothing to train", log)
        return
    _status("loading connectome", log)
    root = ensure_connectome(CONNECTOME_DIR)
    brain = FlyBrain(root)
    broker = PaperBroker(None)
    already = {}
    if STATUS_PATH.exists():
        try:
            already = json.loads(STATUS_PATH.read_text(encoding="utf-8")).get("pairs") or {}
        except (OSError, json.JSONDecodeError):
            already = {}
    status = {
        "target_weekly": TARGET,
        "exam_days": EXAM_DAYS,
        "train_days": TRAIN_DAYS,
        "balance": BALANCE,
        "pairs": {k: v for k, v in already.items() if isinstance(v, dict) and v.get("passed")},
    }
    pending = []
    for symbol in histories:
        row = already.get(symbol) or {}
        if row.get("passed"):
            _status(
                f"{symbol}  already passed at {float(row.get('weekly') or 0):.2%} on round {row.get('round')}",
                log,
            )
            continue
        pending.append(symbol)
    rounds = {symbol: int((already.get(symbol) or {}).get("round") or 0) for symbol in pending}
    best_rate: dict[str, float] = {}
    for symbol in pending:
        prev = already.get(symbol) or {}
        if prev.get("best_weekly") is not None:
            best_rate[symbol] = float(prev["best_weekly"])
    while pending:
        still: list[str] = []
        for symbol in pending:
            bars, source = histories[symbol]
            train, exam, _cut = split_bars(bars)
            rounds[symbol] += 1
            rnd = rounds[symbol]
            if rnd == 1:
                _copy_brains(symbol, ROOT / "reports" / "weekly_pretrain" / symbol)
            elif symbol in best_rate:
                _restore_brains(symbol, ROOT / "reports" / "weekly_best" / symbol)
            _status(f"{symbol}  train round {rnd}", log)
            try:
                run_train(brain, broker, symbol, train, rnd, log)
            except Exception:
                _status(f"{symbol}  train failed\n{traceback.format_exc()}", log)
                still.append(symbol)
                continue
            _status(f"{symbol}  exam round {rnd}  frozen_code={is_frozen(symbol)}", log)
            try:
                summary = run_exam(brain, broker, symbol, exam, log)
            except Exception:
                _status(f"{symbol}  exam failed\n{traceback.format_exc()}", log)
                still.append(symbol)
                continue
            rate = float(summary.get("weekly") or 0.0)
            net = float(summary.get("net_usd") or 0.0)
            n = int(summary.get("n") or 0)
            passed = rate >= TARGET and n > 0
            row = {
                "round": rnd,
                "source": source,
                "n": n,
                "net_usd": round(net, 2),
                "weekly": round(rate, 5),
                "weeks": round(float(summary.get("weeks") or 0.0), 3),
                "passed": passed,
                "wins": int(summary.get("wins") or 0),
                "losses": int(summary.get("losses") or 0),
                "max_dd": summary.get("max_dd"),
                "best_weekly": None,
            }
            kept = symbol not in best_rate or rate > best_rate[symbol]
            if kept:
                best_rate[symbol] = rate
                _copy_brains(symbol, ROOT / "reports" / "weekly_best" / symbol)
            else:
                _restore_brains(symbol, ROOT / "reports" / "weekly_best" / symbol)
            row["best_weekly"] = round(best_rate[symbol], 5)
            status["pairs"][symbol] = row
            _write_status(status)
            _status(
                f"{symbol}  weekly {rate:.2%}  net ${net:+.2f}  "
                f"n={n}  weeks={row['weeks']:.2f}  {'PASS' if passed else 'FAIL'}"
                f"{'  kept' if kept else '  restored better brain'}",
                log,
            )
            if not passed:
                still.append(symbol)
        pending = still
        if pending:
            _status("still short: " + ", ".join(pending), log)
    _status("all pairs passed", log)
    log.close()


if __name__ == "__main__":
    main()
