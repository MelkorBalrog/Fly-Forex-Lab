"""After each sealed profit, the next trade is sized from that profit alone.

Rule:
- After a sealed profit of ``W`` dollars, the next trade's money base is
  ``1.5 × W``. A later profit replaces that base. It is not added to the
  earlier profits. $2,000 then $800 → next budget $1,200, not $4,200.
- The Bollinger squeeze is logged. It does not skip entries.
- A loss smaller than the last win keeps a cushion (remaining × 1.5).
  A loss that wipes the win returns the book to full account size.
"""

from __future__ import annotations

from dataclasses import dataclass

SQUEEZE_RATIO = 0.90  # bb_bw / bb_bw_ma → bands "closed"
EXPAND_RATIO = 1.08  # must clearly reopen, not a 2% twitch
EXPAND_LIFT = 1.08  # width must rise this far off the squeeze low
PROFIT_BUDGET_MULT = 1.50  # profit + 50% of profit


@dataclass
class RecycleSnap:
    active: bool
    phase: str
    budget: float
    last_win: float
    bw_ratio: float
    note: str


class ProfitRecycleGuard:
    """BB re-entry gate + 1.5× last-win sizing budget."""

    def __init__(self, *, enabled: bool = True, mult: float = PROFIT_BUDGET_MULT) -> None:
        self.enabled = bool(enabled)
        self.mult = float(mult) if float(mult) > 0 else PROFIT_BUDGET_MULT
        self.active = False
        self.phase = "idle"  # idle | squeeze | expand | ready
        self.budget = 0.0
        self.last_win = 0.0
        self.win_n = 0
        self._squeeze_bw = 0.0
        self.last = RecycleSnap(
            active=False, phase="idle", budget=0.0, last_win=0.0, bw_ratio=1.0, note="off"
        )

    def clear(self, reason: str = "cleared") -> None:
        self.active = False
        self.phase = "idle"
        self.budget = 0.0
        self.last_win = 0.0
        self.win_n = 0
        self._squeeze_bw = 0.0
        self.last = RecycleSnap(
            active=False, phase="idle", budget=0.0, last_win=0.0, bw_ratio=1.0, note=reason
        )

    def on_close(self, usd: float) -> str:
        """Arm or clear after a sealed trade. Returns a short log note."""
        if not self.enabled:
            return ""
        pnl = float(usd)
        if pnl > 0:
            # Latest sealed profit only. Do not add it to the previous win.
            self.active = True
            self.phase = "squeeze"
            self.win_n = int(self.win_n) + 1
            self.last_win = pnl
            self.budget = pnl * self.mult
            self._squeeze_bw = 0.0
            note = (
                f"profit-recycle  win #{self.win_n} ${pnl:,.0f} only  "
                f"next budget ${self.budget:,.0f} ({self.mult:.2f}× this profit)"
            )
            self.last = RecycleSnap(
                active=True,
                phase=self.phase,
                budget=self.budget,
                last_win=self.last_win,
                bw_ratio=1.0,
                note=note,
            )
            return note
        if self.active:
            kept = float(self.last_win) + pnl
            if kept > 1.0:
                # A loss spends the cushion. Do not hand the next trade the full account.
                self.last_win = kept
                self.budget = kept * self.mult
                self.phase = "squeeze"
                self._squeeze_bw = 0.0
                self.active = True
                note = (
                    f"profit-recycle  loss ${pnl:,.0f}  cushion ${kept:,.0f}  "
                    f"budget ${self.budget:,.0f}  wait BB squeeze→expand"
                )
                self.last = RecycleSnap(
                    active=True,
                    phase=self.phase,
                    budget=self.budget,
                    last_win=self.last_win,
                    bw_ratio=1.0,
                    note=note,
                )
                return note
            self.clear("profit-recycle  cleared after loss")
            return self.last.note
        return ""

    def observe(self, feat: dict | None) -> RecycleSnap:
        """Update BB phase each bar. Call even when flat."""
        if not self.enabled or not self.active:
            self.last = RecycleSnap(
                active=False,
                phase="idle",
                budget=0.0,
                last_win=0.0,
                bw_ratio=1.0,
                note="profit-recycle off" if not self.enabled else "idle",
            )
            return self.last
        feat = feat or {}
        bw = float(feat.get("bb_bw") or 0.0)
        bw_ma = float(feat.get("bb_bw_ma") or 0.0)
        ratio = (bw / bw_ma) if bw_ma > 1e-12 else 1.0

        if self.phase == "squeeze":
            if ratio <= SQUEEZE_RATIO:
                self.phase = "expand"
                self._squeeze_bw = bw
                note = f"profit-recycle  BB squeezed  bw/ma {ratio:.2f}  wait expand"
            else:
                note = f"profit-recycle  wait BB squeeze  bw/ma {ratio:.2f}>{SQUEEZE_RATIO:.2f}"
        elif self.phase == "expand":
            opened = ratio >= EXPAND_RATIO and (
                self._squeeze_bw <= 0 or bw >= self._squeeze_bw * EXPAND_LIFT
            )
            if opened:
                self.phase = "ready"
                note = (
                    f"profit-recycle  BB opened  bw/ma {ratio:.2f}  "
                    f"READY  budget ${self.budget:,.0f}"
                )
            else:
                note = f"profit-recycle  wait BB expand  bw/ma {ratio:.2f}"
        elif self.phase == "ready":
            note = f"profit-recycle  READY  budget ${self.budget:,.0f} (1.5× win ${self.last_win:,.0f})"
        else:
            note = "profit-recycle idle"

        self.last = RecycleSnap(
            active=True,
            phase=self.phase,
            budget=float(self.budget),
            last_win=float(self.last_win),
            bw_ratio=float(ratio),
            note=note,
        )
        return self.last

    def allow_entry(self) -> tuple[bool, str]:
        """The squeeze is logged. It does not skip the rest of the week."""
        if not self.enabled or not self.active:
            return True, ""
        return True, self.last.note or ""

    def size_budget(self) -> float | None:
        """Money base for the next trade: 1.5× the latest profit, or None = full account."""
        if not self.enabled or not self.active or self.budget <= 0:
            return None
        return float(self.budget)

    def on_entry(self) -> None:
        """After a fill that used the recycle slot — stay armed until that trade closes.

        Budget remains for this open trade's sizing already applied; phase stays
        ready so we don't double-enter before the position is open (caller only
        enters when flat). On the next close, on_close re-arms or clears.
        """
        if self.active and self.phase == "ready":
            # Prevent a second entry same-bar / before close: leave ready but
            # caller has a ticket. Optionally nudge phase so allow stays true
            # only until open — ticket gate handles that.
            pass

    def pack(self) -> dict:
        s = self.last
        return {
            "enabled": bool(self.enabled),
            "active": bool(s.active),
            "phase": str(s.phase),
            "budget": round(float(s.budget), 2),
            "last_win": round(float(s.last_win), 2),
            "bw_ratio": round(float(s.bw_ratio), 3),
            "note": str(s.note),
        }
