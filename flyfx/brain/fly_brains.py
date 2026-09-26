"""Per-pair dopamine brains (W frozen, readout/gains learn).

EURUSD is frozen at factory weights so the +$1,906 snowball-lock replay
cannot be overwritten. Other pairs load/save ``brains/{SYMBOL}.json``.
Weekly training walks Yahoo/HST history and writes those files.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from flyfx.paths import BRAIN_DIR

EURUSD_KEEP_USD = 1906.66
FROZEN_PAIRS = {"EURUSD"}


def brain_path(symbol: str, live: bool = False) -> Path:
    tag = str(symbol or "EURUSD").upper()
    name = f"{tag}.live.json" if live else f"{tag}.json"
    return BRAIN_DIR / name


def evo_path(symbol: str) -> Path:
    tag = str(symbol or "EURUSD").upper()
    return BRAIN_DIR / f"{tag}.evo.json"


def list_brain_symbols() -> list[str]:
    if not BRAIN_DIR.exists():
        return []
    out: list[str] = []
    for path in BRAIN_DIR.glob("*.json"):
        if path.name.endswith(".live.json") or path.name.endswith(".evo.json") or path.name.endswith(".book.json"):
            continue
        tag = path.stem.upper()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        if data.get("frozen"):
            continue
        if not (data.get("cat_flies") or data.get("plastic")):
            continue
        out.append(tag)
    return sorted(set(out))


def is_frozen(symbol: str) -> bool:
    return str(symbol or "").upper() in FROZEN_PAIRS


def load_brain(symbol: str) -> dict | None:
    path = brain_path(symbol)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _evo_tag_from_name(name: str) -> str:
    raw = str(name or "")
    if raw.lower().endswith(".evo.json"):
        return raw[: -len(".evo.json")].upper()
    return Path(raw).stem.upper()


def load_evo(symbol: str) -> dict | None:
    """Load brains/{PAIR}.evo.json (params + dynamic flags). Not the DA traces."""
    path = evo_path(symbol)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def save_evo(
    symbol: str,
    genome: dict | None,
    *,
    meta: dict | None = None,
    fitness: float | None = None,
    note: str = "",
) -> Path | None:
    """Write the selectable evolved policy. Does not touch factory EURUSD DA."""
    tag = str(symbol or "").upper()
    if not tag:
        return None
    gene = genome if isinstance(genome, dict) else {}
    if not gene:
        return None
    BRAIN_DIR.mkdir(parents=True, exist_ok=True)
    blob = {
        "symbol": tag,
        "kind": "evo",
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "fitness": float(fitness or 0.0),
        "note": str(note or ""),
        "genome": gene,
        "meta": meta or {},
    }
    path = evo_path(tag)
    path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return path


def list_evo_policies() -> list[dict]:
    if not BRAIN_DIR.exists():
        return []
    out: list[dict] = []
    seen: set[str] = set()
    for path in sorted(BRAIN_DIR.glob("*.evo.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        gene = data.get("genome")
        if not isinstance(gene, dict) or not gene:
            continue
        tag = str(data.get("symbol") or _evo_tag_from_name(path.name)).upper()
        out.append(
            {
                "symbol": tag,
                "note": str(data.get("note") or ""),
                "fitness": data.get("fitness"),
                "saved_at": data.get("saved_at") or "",
                "genome": gene,
            }
        )
        seen.add(tag)
    for path in sorted(BRAIN_DIR.glob("*.json")):
        if path.name.endswith(".live.json") or path.name.endswith(".evo.json"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        tag = str(data.get("symbol") or path.stem).upper()
        if tag in seen:
            continue
        gene = data.get("genome")
        if not isinstance(gene, dict) or not gene:
            continue
        if not any(k in gene for k in ("params", "risk_tol", "fuse_dynamic", "sugar")):
            continue
        meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
        out.append(
            {
                "symbol": tag,
                "note": str(meta.get("genome_note") or ""),
                "fitness": meta.get("fitness"),
                "saved_at": data.get("saved_at") or "",
                "genome": gene,
            }
        )
        seen.add(tag)
    return out


def list_evo_symbols() -> list[str]:
    return [row["symbol"] for row in list_evo_policies()]


def save_brain(
    symbol: str,
    plastic: dict,
    banc: dict | None,
    meta: dict | None = None,
    *,
    live: bool = False,
    force: bool = False,
    params: dict | None = None,
    extra: dict | None = None,
) -> Path | None:
    tag = str(symbol or "").upper()
    if is_frozen(tag) and not force and not live:
        print(f"brain  skip save {tag} (frozen factory book — see snapshots/eurusd-snowball-lock-20260919)")
        return None
    BRAIN_DIR.mkdir(parents=True, exist_ok=True)
    path = brain_path(tag, live=live)
    extra = extra or {}
    prev_rf = {}
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            old = None
        if isinstance(old, dict) and isinstance(old.get("risk_factors"), dict):
            prev_rf = old["risk_factors"]
    incoming = extra.get("risk_factors")
    if isinstance(incoming, dict) and int(incoming.get("n_fit") or 0) > 0:
        risk_factors = incoming
    else:
        risk_factors = prev_rf
    blob = {
        "symbol": tag,
        "frozen": is_frozen(tag) and not live and not force,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "plastic": plastic or {},
        "cat_flies": extra.get("cat_flies") or {},
        "nine": extra.get("nine") or {},
        "crops": extra.get("crops") or {},
        "banc": banc or {},
        "params": params or {},
        "meta": meta or {},
        "genome": extra.get("genome") or {},
        "risk_factors": risk_factors or {},
    }
    path.write_text(json.dumps(blob), encoding="utf-8")
    return path


def seed_frozen_eurusd() -> Path:
    """Factory EURUSD brain: empty D, freeze flag. Replay stays on the keep path.

    Does not overwrite a trained (unfrozen) EURUSD file written with --force-brain.
    Restore factory from snapshots/pre-eurusd-evolve-20260921/brains/EURUSD.json
    or snapshots/eurusd-snowball-lock-20260919.
    """
    BRAIN_DIR.mkdir(parents=True, exist_ok=True)
    path = brain_path("EURUSD")
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = None
        if isinstance(data, dict) and data.get("frozen") is False:
            return path
    blob = {
        "symbol": "EURUSD",
        "frozen": True,
        "keep_usd": EURUSD_KEEP_USD,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "plastic": {},
        "banc": {},
        "params": {},
        "meta": {
            "note": "factory readout; W frozen; do not train this file",
            "snapshot": "snapshots/eurusd-snowball-lock-20260919",
        },
    }
    path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return path


def forward_r(
    bars: list[dict],
    idx: int,
    side: str,
    stop_pips: float,
    pip: float,
    horizon: int = 36,
) -> float:
    """Label a 2oo3 shot from later bars: -1R on stop, else MFE capped at +1.5R."""
    if side not in ("BUY", "SELL") or not bars or idx >= len(bars) - 2:
        return 0.0
    entry = float(bars[idx]["close"])
    risk = max(float(stop_pips), 8.0) * float(pip)
    if risk <= 1e-12:
        return 0.0
    mfe = 0.0
    last_i = min(idx + horizon, len(bars) - 1)
    for j in range(idx + 1, last_i + 1):
        h = float(bars[j]["high"])
        l = float(bars[j]["low"])
        if side == "BUY":
            if l <= entry - risk:
                return -1.0
            mfe = max(mfe, (h - entry) / risk)
        else:
            if h >= entry + risk:
                return -1.0
            mfe = max(mfe, (entry - l) / risk)
        if mfe >= 1.5:
            return 1.5
    last = float(bars[last_i]["close"])
    sign = 1.0 if side == "BUY" else -1.0
    return float(np.clip(sign * (last - entry) / risk, -1.5, 2.5))
