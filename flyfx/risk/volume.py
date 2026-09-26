"""Volume as percent of available money (position margin).

``VOL %`` / ``--volume-pct`` is the fraction of **available money** (free margin
if positive, else equity) pledged as **required margin** for the trade:

    base_money = free if free > 0 else equity
    target_margin = base_money × (volume_pct / 100)
    lots          = target_margin / margin_per_lot

BANC / bayes / admit / tape (auto mode) may haircut that target. ``max-lots``
can still clamp the result below the requested money %.

Modes:
- ``equity`` (default): VOL % = % of available money as margin.
- ``fixed``: same when VOL % is set; blank → 100% of money.
- ``auto``: start from profile VOL %, then × tape × BANC (clipped 25–100).
- ``risk``: alias of ``equity`` (kept for older CLI/GUI values).

Profile defaults (%% of available money as margin):
conservative 50 / balanced 12 / aggressive 100 / scalp 12
(balanced+scalp 12% restores the ~4k cat-fuse lot scale under margin sizing;
``--legacy-2oo3 --risk-tol balanced`` uses classic risk-% sizing instead).
"""

from __future__ import annotations

from dataclasses import dataclass

from flyfx.risk.tolerance import parse_risk_tol

# Profile → default %% of available money pledged as margin.
# Scalp / balanced both use 12% so cat-fuse margin sizing matches the ~4k book.
# Freeze (--legacy-2oo3 balanced) bypasses this and uses classic risk-% sizing.
VOLUME_BY_RISK: dict[str, float] = {
    "conservative": 50.0,
    "balanced": 12.0,
    "aggressive": 100.0,
    "scalp": 12.0,
}

MODES = ("equity", "fixed", "auto", "risk")

_MODE_ALIASES = {
    "equity": "equity",
    "money": "equity",
    "cash": "equity",
    "available": "equity",
    "risk": "equity",  # legacy name → equity semantics
    "profile": "equity",
    "tol": "equity",
    "tolerance": "equity",
    "fixed": "fixed",
    "const": "fixed",
    "constant": "fixed",
    "manual": "fixed",
    "auto": "auto",
    "tape": "auto",
    "dynamic": "auto",
}


def _clip(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def parse_volume_mode(spec: str | None) -> str:
    text = str(spec or "equity").strip().lower().replace("_", "-")
    name = _MODE_ALIASES.get(text, "")
    if not name:
        raise ValueError(f"unknown volume mode {spec!r}; use equity, fixed, auto")
    return name


def profile_volume_pct(risk_tol: str | None) -> float:
    try:
        name = parse_risk_tol(risk_tol)
    except ValueError:
        name = "balanced"
    return float(VOLUME_BY_RISK.get(name, 100.0))


def available_money(equity: float, free: float | None = None) -> float:
    """Cash available to size against: free margin if usable, else equity."""
    eq = max(float(equity or 0.0), 0.0)
    try:
        fr = float(free) if free is not None else eq
    except (TypeError, ValueError):
        fr = eq
    if fr > 1e-6:
        return fr
    return eq


@dataclass(frozen=True)
class VolumePlan:
    """``pct`` = percent of available money pledged as position margin."""

    pct: float
    mode: str
    profile_pct: float
    note: str

    @property
    def mult(self) -> float:
        return float(self.pct) / 100.0

    def sized_equity(self, equity: float, free: float | None = None) -> float:
        """Target margin dollars (= available × pct/100)."""
        return available_money(equity, free) * self.mult


def select_volume_pct(
    *,
    risk_tol: str | None = "balanced",
    volume_pct: float | None = None,
    volume_mode: str | None = "equity",
    tape_mult: float = 1.0,
    banc_mult: float = 1.0,
) -> VolumePlan:
    """Pick %% of available money pledged as trade margin."""
    try:
        mode = parse_volume_mode(volume_mode)
    except ValueError:
        mode = "equity"
    profile = profile_volume_pct(risk_tol)
    override = None if volume_pct is None else float(volume_pct)

    if mode == "fixed":
        pct = 100.0 if override is None else override
        pct = _clip(pct, 0.0, 100.0)
        note = f"volume {pct:.0f}% of money as margin (fixed)"
        return VolumePlan(pct=pct, mode=mode, profile_pct=profile, note=note)

    if mode == "auto":
        base = profile if override is None else override
        tape = _clip(float(tape_mult), 0.25, 1.0)
        banc = _clip(float(banc_mult), 0.50, 1.15)
        pct = _clip(base * tape * banc, 25.0, 100.0)
        note = (
            f"volume {pct:.0f}% of money as margin (auto  base {base:.0f}%  "
            f"tape×{tape:.2f}  BANC×{banc:.2f})"
        )
        return VolumePlan(pct=pct, mode="auto", profile_pct=profile, note=note)

    # equity (default): profile or absolute override as %% of available money
    pct = profile if override is None else override
    pct = _clip(pct, 0.0, 100.0)
    try:
        src = "override" if override is not None else f"risk-tol {parse_risk_tol(risk_tol)}"
    except ValueError:
        src = "override" if override is not None else "risk-tol balanced"
    note = f"volume {pct:.0f}% of money as margin ({src})"
    return VolumePlan(pct=pct, mode="equity", profile_pct=profile, note=note)
