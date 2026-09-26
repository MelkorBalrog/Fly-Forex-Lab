"""Per-pair admission book: size haircut, wide-spread gate, chop/ATR halt.

This is NOT a strategy router. Every pair still uses factory 2oo3
{tech, fly-trend, fly-fade}. PairBook only scales lots and refuses some
opens on non-frozen pairs.

Opt-in only (`--pair-book`). EURUSD is never enabled (frozen factory book,
+$1,906.66 lock). Yahoo 5m books start at half size until two wins; local
HST stays full size. Demo / replay only — not a live-profit claim.
"""

from __future__ import annotations

from flyfx.brain.fly_brains import is_frozen

WIDE_PIPS = 1.35
YAHOO_SIZE = 0.50
YAHOO_WINS_FOR_FULL = 2
CHOP_SKIP_BARS = 8


def pair_book_enabled(
    symbol: str,
    frozen: bool = False,
    no_pair_book: bool = False,
    pair_book: bool = False,
) -> bool:
    """True only with --pair-book, not --no-pair-book, and not a frozen pair."""
    if not pair_book or no_pair_book or frozen or is_frozen(symbol):
        return False
    return True


class PairBook:
    def __init__(
        self,
        symbol: str = "",
        source: str = "",
        *,
        enabled: bool = True,
        yahoo: bool | None = None,
    ) -> None:
        self.symbol = str(symbol or "")
        self.source = str(source or "")
        self.enabled = pair_book_enabled(self.symbol, pair_book=bool(enabled))
        if yahoo is None:
            self.yahoo = "yahoo" in self.source.lower()
        else:
            self.yahoo = bool(yahoo)
        self.wins = 0
        self.chop_n = 0
        self.halt_reason = ""
        self.last_chop_side = ""
        self.last_chop_step = -10**9

    def size_mult(self) -> float:
        if not self.enabled or not self.yahoo:
            return 1.0
        if self.wins >= YAHOO_WINS_FOR_FULL:
            return 1.0
        return float(YAHOO_SIZE)

    def wide_ok(self, tag: str) -> bool:
        """Wide spreads: 3oo3/3oo4/4oo4, or 2oo3 that already has flow in the tag.

        Raw 2oo3 without flow is blocked. nest/koo9/ensemble keep their own
        gates in the trader and do not call this.
        """
        if not self.enabled:
            return (
                str(tag or "").startswith("3oo3")
                or str(tag or "").startswith("3oo4")
                or str(tag or "").startswith("4oo4")
            )
        t = str(tag or "")
        if t.startswith("3oo3") or t.startswith("3oo4") or t.startswith("4oo4"):
            return True
        if t.startswith("2oo3") and "flow" in t:
            return True
        return False

    def skip_open(self, decision: str, step: int) -> str:
        if not self.enabled:
            return ""
        if self.halt_reason:
            return self.halt_reason
        if decision not in ("BUY", "SELL"):
            return ""
        if self.last_chop_side != decision:
            return ""
        if int(step) - int(self.last_chop_step) > CHOP_SKIP_BARS:
            return ""
        return f"chop-scratch skip {decision} within {CHOP_SKIP_BARS} bars"

    def on_close(self, reason: str, usd: float, side: str, step: int) -> str:
        if not self.enabled:
            return ""
        notes: list[str] = []
        if float(usd) > 0:
            self.wins += 1
        label = str(reason or "")
        low = label.lower()
        if label == "ATR stop" or low == "atr stop":
            self.halt_reason = "pair circuit after ATR stop"
            notes.append(self.halt_reason)
        if "chop scratch" in low:
            self.chop_n += 1
            self.last_chop_side = str(side or "")
            self.last_chop_step = int(step)
            if self.chop_n >= 2:
                self.halt_reason = "pair circuit after 2 chop scratches"
                notes.append(self.halt_reason)
        return "  ".join(notes)
