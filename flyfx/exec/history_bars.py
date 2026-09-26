"""Bars from each simulation, kept so later training can see them again.

A new window that overlaps or continues an existing file (the next bar, or a
hole no bigger than three bars) is written back into that file. A real gap
opens a new file. Training loads every file for the pair and timeframe and
then adds the Yahoo or MetaTrader bars selected for this run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from flyfx.paths import PROJECT_ROOT

HISTORY_DIR = PROJECT_ROOT / "data" / "history"
_NAME = re.compile(r"^s(\d+)_e(\d+)\.json$")


def _step_seconds(bars: list[dict], fallback: int = 300) -> int:
    diffs: list[int] = []
    prev = 0
    for bar in bars:
        stamp = int(bar.get("time") or 0)
        if prev and stamp > prev:
            diffs.append(stamp - prev)
        if stamp:
            prev = stamp
    if not diffs:
        return max(int(fallback), 60)
    diffs.sort()
    return max(int(diffs[len(diffs) // 2]), 60)


def _clean(bar: dict) -> dict | None:
    try:
        stamp = int(bar.get("time") or 0)
        close = float(bar.get("close") or 0.0)
    except (TypeError, ValueError):
        return None
    if stamp <= 0 or close <= 0:
        return None
    out = {
        "time": stamp,
        "open": float(bar.get("open") if bar.get("open") is not None else close),
        "high": float(bar.get("high") if bar.get("high") is not None else close),
        "low": float(bar.get("low") if bar.get("low") is not None else close),
        "close": close,
    }
    if bar.get("volume") is not None:
        try:
            vol = float(bar["volume"])
        except (TypeError, ValueError):
            vol = 0.0
        if vol > 0:
            out["volume"] = vol
    return out


def _pair_dir(symbol: str, timeframe: str) -> Path:
    tag = str(symbol or "EURUSD").upper()
    tf = str(timeframe or "M5").upper()
    return HISTORY_DIR / tag / tf


def _spans(folder: Path) -> list[tuple[int, int, Path]]:
    if not folder.exists():
        return []
    out: list[tuple[int, int, Path]] = []
    for path in folder.glob("s*_e*.json"):
        match = _NAME.match(path.name)
        if not match:
            continue
        out.append((int(match.group(1)), int(match.group(2)), path))
    out.sort()
    return out


def _continuous(a0: int, a1: int, b0: int, b1: int, step: int) -> bool:
    if a1 < b0:
        hole = b0 - a1
    elif b1 < a0:
        hole = a0 - b1
    else:
        return True
    return hole <= max(int(step), 60) * 3


def _read(path: Path) -> list[dict]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows = raw.get("bars") if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows:
        if isinstance(row, dict):
            bar = _clean(row)
            if bar is not None:
                out.append(bar)
    out.sort(key=lambda bar: int(bar["time"]))
    return out


def _write(folder: Path, symbol: str, timeframe: str, bars: list[dict]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    start = int(bars[0]["time"])
    end = int(bars[-1]["time"])
    path = folder / f"s{start}_e{end}.json"
    payload = {
        "symbol": str(symbol).upper(),
        "timeframe": str(timeframe).upper(),
        "bars": bars,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _union(chunks: list[list[dict]]) -> list[dict]:
    by_time: dict[int, dict] = {}
    for chunk in chunks:
        for bar in chunk:
            by_time[int(bar["time"])] = bar
    return [by_time[stamp] for stamp in sorted(by_time)]


def remember_bars(symbol: str, timeframe: str, bars: list[dict], *, quiet: bool = False) -> Path | None:
    """Save the bars a simulation just used. Join a continuous file, or start a new one."""
    cleaned = [bar for bar in (_clean(row) for row in (bars or [])) if bar is not None]
    if len(cleaned) < 2:
        return None
    cleaned = _union([cleaned])
    step = _step_seconds(cleaned)
    folder = _pair_dir(symbol, timeframe)
    start = int(cleaned[0]["time"])
    end = int(cleaned[-1]["time"])
    joined: list[list[dict]] = []
    drop: list[Path] = []
    for a0, a1, path in _spans(folder):
        if _continuous(a0, a1, start, end, step):
            joined.append(_read(path))
            drop.append(path)
            start = min(start, a0)
            end = max(end, a1)
    # A file that touched the new window may now touch another file.
    changed = True
    while changed:
        changed = False
        for a0, a1, path in _spans(folder):
            if path in drop:
                continue
            if _continuous(a0, a1, start, end, step):
                joined.append(_read(path))
                drop.append(path)
                start = min(start, a0)
                end = max(end, a1)
                changed = True
    joined.append(cleaned)
    merged = _union(joined)
    for path in drop:
        try:
            path.unlink()
        except OSError:
            pass
    dest = _write(folder, symbol, timeframe, merged)
    if not quiet:
        kind = "merged" if drop else "new file"
        print(
            f"history  {str(symbol).upper()} {str(timeframe).upper()}  {kind}  "
            f"n={len(merged)}  {dest.name}"
        )
    return dest


def load_history(symbol: str, timeframe: str) -> list[dict]:
    """Every saved segment for this pair and timeframe, oldest bar first."""
    return _union([_read(path) for _a0, _a1, path in _spans(_pair_dir(symbol, timeframe))])
