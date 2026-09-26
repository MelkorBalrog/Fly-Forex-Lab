"""Load OHLC bars from CSV, JSON, or MT4 .hst for replay on any timeframe."""

from __future__ import annotations

import csv
import io
import json
import re
import struct
from datetime import datetime, timezone
from pathlib import Path

# Name -> minutes. NATIVE means keep the file's own bar size.
TIMEFRAMES: dict[str, int] = {
    "NATIVE": 0,
    "M1": 1,
    "M2": 2,
    "M3": 3,
    "M4": 4,
    "M5": 5,
    "M6": 6,
    "M10": 10,
    "M12": 12,
    "M15": 15,
    "M20": 20,
    "M30": 30,
    "H1": 60,
    "H2": 120,
    "H3": 180,
    "H4": 240,
    "H6": 360,
    "H8": 480,
    "H12": 720,
    "D1": 1440,
    "W1": 10080,
}

_TF_ALIAS = {
    "1": "M1",
    "1m": "M1",
    "1min": "M1",
    "1minute": "M1",
    "5": "M5",
    "5m": "M5",
    "5min": "M5",
    "5minute": "M5",
    "10": "M10",
    "10m": "M10",
    "10min": "M10",
    "10minute": "M10",
    "15": "M15",
    "15m": "M15",
    "15min": "M15",
    "30": "M30",
    "30m": "M30",
    "30min": "M30",
    "60": "H1",
    "1h": "H1",
    "1hr": "H1",
    "1hour": "H1",
    "hour": "H1",
    "hourly": "H1",
    "4h": "H4",
    "4hour": "H4",
    "1d": "D1",
    "daily": "D1",
    "day": "D1",
    "raw": "NATIVE",
    "as-is": "NATIVE",
    "asis": "NATIVE",
    "native": "NATIVE",
    "file": "NATIVE",
}

_TIME_KEYS = ("time", "timestamp", "datetime", "date", "gmt", "gmttime", "dt")
_OPEN_KEYS = ("open", "o", "<open>")
_HIGH_KEYS = ("high", "h", "<high>")
_LOW_KEYS = ("low", "l", "<low>")
_CLOSE_KEYS = ("close", "c", "<close>", "adjclose", "adj_close")
_VOL_KEYS = ("volume", "vol", "tickvol", "tick_volume", "tickvolume", "<tickvol>", "<vol>")
_DATE_KEYS = ("date", "<date>")
_CLOCK_KEYS = ("clock", "tm", "<time>", "hour")
_EXTRA_KEYS = ("vix", "put_call", "putcall", "cot", "cot_net", "advance_decline", "mcclellan", "trin")


def parse_timeframe(text: str | None) -> tuple[str, int]:
    orig = str(text or "M5").strip()
    compact = orig.replace(" ", "").lower()
    if compact in _TF_ALIAS:
        name = _TF_ALIAS[compact]
        return name, TIMEFRAMES[name]
    raw = compact.upper()
    if raw in TIMEFRAMES:
        return raw, TIMEFRAMES[raw]
    m = re.fullmatch(r"M(\d+)", raw)
    if m:
        mins = int(m.group(1))
        name = f"M{mins}"
        return name, mins
    h = re.fullmatch(r"H(\d+)", raw)
    if h:
        mins = int(h.group(1)) * 60
        name = f"H{h.group(1)}"
        return name, mins
    raise ValueError(f"unknown timeframe {text!r} — try M1, M5, M10, M15, H1, H4, D1, or NATIVE")


def _norm_key(k: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(k).strip().lower())


def _parse_unix(val: object) -> int | None:
    try:
        n = float(str(val).strip())
    except (TypeError, ValueError):
        return None
    if n > 1e12:
        n /= 1000.0
    if n < 1e8 or n > 2e10:
        return None
    return int(n)


def _parse_dt_string(text: str) -> int | None:
    s = text.strip().replace("T", " ")
    s = re.sub(r"Z$", "", s)
    s = re.sub(r"[+-]\d{2}:?\d{2}$", "", s).strip()
    if re.match(r"^\d{4}\.\d{2}\.\d{2}", s):
        parts = s.split(" ", 1)
        parts[0] = parts[0].replace(".", "-")
        s = " ".join(parts)
    fmts = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%d-%m-%Y %H:%M:%S",
        "%d-%m-%Y %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%Y%m%d %H:%M:%S",
        "%Y%m%d %H:%M",
        "%Y%m%d",
    )
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            return int(dt.replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
    if len(s) > 19:
        return _parse_dt_string(s[:19])
    if len(s) > 16 and ":" in s:
        return _parse_dt_string(s[:16])
    return None


def parse_stamp(date_part: object, time_part: object | None = None) -> int | None:
    if time_part in (None, ""):
        unix = _parse_unix(date_part)
        if unix is not None:
            return unix
        return _parse_dt_string(str(date_part or ""))
    clock = str(time_part).strip()
    if " " in clock or "T" in clock:
        return _parse_dt_string(clock) or _parse_dt_string(str(date_part or ""))
    date_s = str(date_part).strip().replace(".", "-")
    if len(clock) == 5:
        clock += ":00"
    return _parse_dt_string(f"{date_s} {clock}")


def _pick(row: dict[str, str], keys: tuple[str, ...]) -> str | None:
    for k in keys:
        if k in row and str(row[k]).strip() != "":
            return row[k]
    return None


def _float(val: object) -> float | None:
    if val is None:
        return None
    s = str(val).strip().replace(" ", "").replace("'", "")
    if not s or s.lower() in ("nan", "null", "none"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _bar_from_parts(
    ts: int | None,
    o: float | None,
    h: float | None,
    l: float | None,
    c: float | None,
    vol: float | None,
    extra: dict | None = None,
) -> dict | None:
    if ts is None or o is None or h is None or l is None or c is None:
        return None
    if c <= 0 or h <= 0 or l <= 0 or o <= 0:
        return None
    rec = {"time": int(ts), "open": float(o), "high": float(h), "low": float(l), "close": float(c)}
    if vol is not None and vol > 0:
        rec["volume"] = float(vol)
    if extra:
        rec.update(extra)
    return rec


def infer_period_minutes(bars: list[dict]) -> int | None:
    if len(bars) < 3:
        return None
    diffs = sorted(int(bars[i]["time"]) - int(bars[i - 1]["time"]) for i in range(1, min(len(bars), 80)))
    diffs = [d for d in diffs if d > 0]
    if not diffs:
        return None
    median = diffs[len(diffs) // 2]
    mins = max(1, int(round(median / 60.0)))
    return mins


def resample_bars(bars: list[dict], period_min: int) -> list[dict]:
    """Bucket bars into period_min minutes. Never upsamples (file already coarser)."""
    if period_min <= 0 or len(bars) < 2:
        return bars
    native = infer_period_minutes(bars) or 0
    if native >= period_min:
        return bars
    bucket = period_min * 60
    out: list[dict] = []
    cur_t = None
    acc: dict | None = None
    extras = ("vix", "put_call", "cot", "cot_net", "advance_decline", "mcclellan", "trin")
    for b in bars:
        t = (int(b["time"]) // bucket) * bucket
        if acc is None or t != cur_t:
            if acc is not None:
                out.append(acc)
            acc = {
                "time": t,
                "open": float(b["open"]),
                "high": float(b["high"]),
                "low": float(b["low"]),
                "close": float(b["close"]),
            }
            if b.get("volume"):
                acc["volume"] = float(b["volume"])
            for k in extras:
                if k in b:
                    acc[k] = b[k]
            cur_t = t
        else:
            acc["high"] = max(acc["high"], float(b["high"]))
            acc["low"] = min(acc["low"], float(b["low"]))
            acc["close"] = float(b["close"])
            if b.get("volume"):
                acc["volume"] = float(acc.get("volume") or 0.0) + float(b["volume"])
            for k in extras:
                if k in b:
                    acc[k] = b[k]
    if acc is not None:
        out.append(acc)
    return out


def _map_header(fieldnames: list[str]) -> dict[str, str]:
    mapping = {}
    for raw in fieldnames:
        mapping[_norm_key(raw)] = raw
    return mapping


def _row_to_bar(raw: dict[str, str], keymap: dict[str, str]) -> dict | None:
    def get(keys: tuple[str, ...]) -> str | None:
        for k in keys:
            src = keymap.get(k)
            if src is not None and str(raw.get(src, "")).strip() != "":
                return raw[src]
        return None

    date_v = get(_DATE_KEYS)
    time_v = get(_TIME_KEYS)
    clock_v = get(_CLOCK_KEYS)
    clockish = None
    for cand in (clock_v, time_v):
        if cand and re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", str(cand).strip()):
            clockish = cand
            break
    if date_v and clockish:
        ts = parse_stamp(date_v, clockish)
    else:
        ts = parse_stamp(time_v or date_v, clockish)
    extra = {}
    for k in _EXTRA_KEYS:
        src = keymap.get(k)
        if src and raw.get(src) not in (None, ""):
            extra[k if k != "putcall" else "put_call"] = _float(raw[src])
    return _bar_from_parts(
        ts,
        _float(get(_OPEN_KEYS)),
        _float(get(_HIGH_KEYS)),
        _float(get(_LOW_KEYS)),
        _float(get(_CLOSE_KEYS)),
        _float(get(_VOL_KEYS)),
        extra or None,
    )


def parse_ohlc_text(text: str, name: str = "upload") -> list[dict]:
    blob = text.lstrip("\ufeff").strip()
    if not blob:
        raise ValueError(f"{name}: empty file")
    if blob[0] in "{[":
        return _parse_json(blob, name)
    sample = blob[:4096]
    dialect_name = ","
    for delim in (",", ";", "\t", "|"):
        if sample.count(delim) >= 3:
            dialect_name = delim
            break
    reader = csv.reader(io.StringIO(blob), delimiter=dialect_name)
    rows = [r for r in reader if any(str(c).strip() for c in r)]
    if not rows:
        raise ValueError(f"{name}: no rows")
    first = [_norm_key(c) for c in rows[0]]
    has_header = any(k in first for k in ("open", "high", "low", "close", "time", "date", "datetime", "<open>"))
    bars: list[dict] = []
    if has_header:
        keys = rows[0]
        keymap = _map_header(keys)
        for row in rows[1:]:
            raw = {keys[i]: row[i] if i < len(row) else "" for i in range(len(keys))}
            rec = _row_to_bar(raw, keymap)
            if rec:
                bars.append(rec)
    else:
        for row in rows:
            if len(row) < 5:
                continue
            # MT4 export without header: date, time, o, h, l, c, vol
            if len(row) >= 6 and ":" in str(row[1]):
                rec = _bar_from_parts(
                    parse_stamp(row[0], row[1]),
                    _float(row[2]),
                    _float(row[3]),
                    _float(row[4]),
                    _float(row[5]),
                    _float(row[6]) if len(row) > 6 else None,
                )
            else:
                rec = _bar_from_parts(
                    parse_stamp(row[0]),
                    _float(row[1]),
                    _float(row[2]),
                    _float(row[3]),
                    _float(row[4]),
                    _float(row[5]) if len(row) > 5 else None,
                )
            if rec:
                bars.append(rec)
    if len(bars) < 5:
        raise ValueError(f"{name}: need at least 5 OHLC rows, got {len(bars)}")
    bars.sort(key=lambda b: b["time"])
    return bars


def _parse_json(blob: str, name: str) -> list[dict]:
    data = json.loads(blob)
    if isinstance(data, dict):
        for key in ("bars", "data", "candles", "ohlc"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"{name}: JSON must be a list of bars")
    bars: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        keymap = {_norm_key(k): k for k in item}
        raw = {k: "" if v is None else str(v) for k, v in item.items()}
        rec = _row_to_bar(raw, keymap)
        if rec:
            bars.append(rec)
    if len(bars) < 5:
        raise ValueError(f"{name}: need at least 5 OHLC rows, got {len(bars)}")
    bars.sort(key=lambda b: b["time"])
    return bars


def load_hst(path: Path, count: int | None = None) -> list[dict]:
    raw = path.read_bytes()
    rec = struct.Struct("<qddddqiq")
    bars: list[dict] = []
    for off in range(148, len(raw) - rec.size + 1, rec.size):
        ts, o, h, l, c, vol, _spr, _rvol = rec.unpack_from(raw, off)
        if ts <= 0 or c <= 0:
            continue
        bar = {"time": int(ts), "open": o, "high": h, "low": l, "close": c}
        tick_vol = float(vol) if vol else 0.0
        if tick_vol > 0:
            bar["volume"] = tick_vol
        bars.append(bar)
    if count is not None and count > 0:
        bars = bars[-count:]
    return bars


def load_bars_file(path: Path | str, *, timeframe: str | None = None, count: int | None = None) -> tuple[list[dict], str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    suffix = p.suffix.lower()
    if suffix == ".hst":
        bars = load_hst(p, None)
        source = f"hst {p.name}"
    else:
        bars = parse_ohlc_text(p.read_text(encoding="utf-8-sig", errors="replace"), p.name)
        source = f"file {p.name}"
    tf_name, mins = parse_timeframe(timeframe or "NATIVE")
    if mins > 0:
        before = len(bars)
        bars = resample_bars(bars, mins)
        if len(bars) != before:
            source = f"{source} → {tf_name}"
    if count is not None and count > 0 and len(bars) > count:
        bars = bars[-count:]
    return bars, source


def clip_count(bars: list[dict], count: int | None) -> list[dict]:
    if count is not None and count > 0 and len(bars) > count:
        return bars[-count:]
    return bars


def guess_symbol_from_name(name: str, fallback: str = "EURUSD") -> str:
    stem = Path(name).stem.upper()
    m = re.search(r"([A-Z]{6,7})", stem)
    return m.group(1) if m else fallback
