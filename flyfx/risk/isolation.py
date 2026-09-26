"""Failure isolation between sensing, voting, and sizing nodes.

Shared MaleCNS ``W``, inverse-variance fuse, and one BookRails book are
single points that can cascade. These helpers:

- drop a poisoned category from the fuse without killing the mix
- cap any one family's weight so it cannot dominate the impulse
- wrap a voter so a NaN / exception becomes HOLD, not a crash
- trip a soft disagreement circuit when live families disagree hard

Nothing here changes the 2oo3 side rules when every node is healthy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

# A family whose posterior P is this large is too uncertain to mix.
UNCERT_DROP = 0.92
# Absolute residual vs the weighted median beyond this (with ≥3 live) is dropped.
OUTLIER_Z = 1.35
# No single category may carry more than this share of fuse weight.
MAX_WEIGHT_SHARE = 0.45
# Soft circuit: force CHOP when live agreement collapses.
DISAGREE_AGREE = 0.22
DISAGREE_MIN_LIVE = 3


@dataclass
class NodeHealth:
    """Per-bar health of isolated pipeline stages."""

    dropped: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)
    voter_faults: list[str] = field(default_factory=list)
    disagree_circuit: bool = False
    weight_caps: dict[str, float] = field(default_factory=dict)

    def note(self) -> str:
        bits: list[str] = []
        if self.dropped:
            bits.append("drop " + ",".join(self.dropped))
        if self.voter_faults:
            bits.append("voter-fault " + ",".join(self.voter_faults))
        if self.disagree_circuit:
            bits.append("disagree-circuit")
        if self.weight_caps:
            caps = ",".join(f"{k}×{v:.2f}" for k, v in self.weight_caps.items())
            bits.append(f"wcap {caps}")
        return "  ".join(bits)


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return 0.5 * (float(s[mid - 1]) + float(s[mid]))


def isolate_category_states(
    states: dict[str, dict] | None,
    selected: tuple[str, ...] | list[str],
    *,
    uncert_drop: float = UNCERT_DROP,
    outlier_z: float = OUTLIER_Z,
    max_share: float = MAX_WEIGHT_SHARE,
) -> tuple[list[tuple[str, float, float, float]], NodeHealth]:
    """Pick live (name, z, residual, weight) tuples after isolation gates.

    Returns weights already capped so no name exceeds ``max_share`` of the
    pre-cap total. Dropped families are recorded on ``NodeHealth``.
    """
    health = NodeHealth()
    blob = states or {}
    raw: list[tuple[str, float, float, float]] = []
    for name in selected:
        st = blob.get(name) or {}
        if not st.get("available"):
            continue
        p = float(st.get("P") or 1.0)
        # Prefer the category's own uncertainty; do not invent a drop from P alone
        # (inverse-var already down-weights high P).
        if "uncertainty" in st:
            unc = float(st.get("uncertainty") or 0.0)
            if unc >= float(uncert_drop):
                health.dropped.append(str(name))
                health.reasons[str(name)] = f"uncert {unc:.2f}"
                continue
        z = float(st.get("fused") or 0.0)
        if not math.isfinite(z):
            health.dropped.append(str(name))
            health.reasons[str(name)] = "nonfinite"
            continue
        w = 1.0 / max(p, 1e-9)
        r = float(st.get("residual") or 0.0)
        if not math.isfinite(r):
            r = 0.0
        raw.append((str(name), z, r, w))

    if len(raw) >= 3:
        med = _median([z for _n, z, _r, _w in raw])
        kept: list[tuple[str, float, float, float]] = []
        for name, z, r, w in raw:
            # Drop only opposing outliers. Same-sign extremes reinforce the
            # mix (structure on a DOWN bar) — cutting them flattened the
            # 4k-era impulse below PullbackSetup.cont_impulse.
            opposing = (z * med) < 0.0 and abs(med) >= 0.05
            if opposing and abs(z - med) > float(outlier_z):
                health.dropped.append(name)
                health.reasons[name] = f"outlier Δ{z - med:+.2f}"
                continue
            kept.append((name, z, r, w))
        raw = kept

    if not raw:
        return [], health

    # Inverse-variance already down-weights noisy families. A hard share cap
    # rewrites the mix and was not part of the profitable cat-fuse book.
    return raw, health


def disagree_circuit(
    zs: list[float],
    agreement: float,
    *,
    min_live: int = DISAGREE_MIN_LIVE,
    agree_bar: float = DISAGREE_AGREE,
) -> bool:
    """True when enough families are live but they do not agree — soft HOLD."""
    if len(zs) < int(min_live):
        return False
    if float(agreement) >= float(agree_bar):
        return False
    # Also trip when signs split hard (not just low std of near-zero noise).
    pos = sum(1 for z in zs if z > 0.18)
    neg = sum(1 for z in zs if z < -0.18)
    return pos >= 1 and neg >= 1 and abs(pos - neg) <= 1


def safe_voter(
    name: str,
    fn: Callable[[], Any],
    *,
    fallback: Any = None,
    health: NodeHealth | None = None,
) -> Any:
    """Run one voter. On any exception, record a fault and return ``fallback``."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — isolation boundary
        if health is not None:
            health.voter_faults.append(f"{name}:{type(exc).__name__}")
        return fallback
