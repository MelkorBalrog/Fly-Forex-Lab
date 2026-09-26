"""Risk rails ported from fly-trader-master (kill switch, circuit, pair admission).

fly-trader (bryceweiner/fly-trader) keeps the model away from the bankroll:
a drawdown from peak blocks entries, consecutive failures trip a circuit,
and a book only sizes fully after it has shown an edge out of sample.
We use the same ideas on FX without Postgres — ATR-normalized, per pair.

Martingale is allowed only as a small recovery step after a *scratch*
loss, never after a full stop, and never while the book is halted.
Snowball winners lock the rest of the UTC day so the pyramid is not
given back on the next 2oo3 signal.
"""

from __future__ import annotations


KILL_DD = 0.025
PROTECT_DD = 0.010
CIRCUIT_N = 3
CIRCUIT_BARS = 12
ADMIT_N = 2
ADMIT_PF = 1.00
ADMIT_MULT = 0.60
SHADOW_N = 4
SHADOW_PF = 0.80


class BookRails:
    def __init__(
        self,
        kill_dd: float = KILL_DD,
        circuit_n: int = CIRCUIT_N,
        circuit_bars: int = CIRCUIT_BARS,
        admit_n: int = ADMIT_N,
        admit_pf: float = ADMIT_PF,
        admit_mult: float = ADMIT_MULT,
        protect_dd: float = PROTECT_DD,
    ) -> None:
        self.kill_dd = float(kill_dd)
        self.protect_dd = float(protect_dd)
        self.circuit_n = int(circuit_n)
        self.circuit_bars = int(circuit_bars)
        self.admit_n = int(admit_n)
        self.admit_pf = float(admit_pf)
        self.admit_mult = float(admit_mult)
        self.peak = 0.0
        self.kill = False
        self.kill_reason = ""
        self.fails = 0
        self.pause_until = -1
        self.pnls: list[float] = []
        self.last_loss_r = 0.0
        self.day_lock = ""
        self.armed_protect = False

    def _dd_limit(self) -> float:
        return self.protect_dd if self.armed_protect else self.kill_dd

    def mark(self, equity: float) -> None:
        eq = float(equity)
        if self.peak <= 0:
            self.peak = eq
        if eq > self.peak:
            self.peak = eq
        lim = self._dd_limit()
        if self.peak > 0 and eq <= self.peak * (1.0 - lim):
            self.kill = True
            self.kill_reason = (
                f"kill switch  eq ${eq:.0f} <= {100.0 * (1.0 - lim):.0f}% of peak ${self.peak:.0f}"
            )

    def blocked(self, step: int, day: str = "") -> str:
        if self.kill:
            return self.kill_reason or "kill switch"
        if self.day_lock and day and day == self.day_lock:
            return f"snowball lock  wait next UTC day (kept pyramid {self.day_lock})"
        if int(step) < self.pause_until:
            return f"circuit pause {self.pause_until - int(step)} bars"
        return ""

    def profit_factor(self) -> float:
        g = sum(x for x in self.pnls if x > 0)
        l = -sum(x for x in self.pnls if x <= 0)
        if l <= 1e-9:
            return 99.0 if g > 0 else 0.0
        return float(g / l)

    def admitted(self) -> bool:
        n = len(self.pnls)
        if n < self.admit_n:
            return False
        return self.profit_factor() >= self.admit_pf

    def admit(self) -> tuple[float, str]:
        n = len(self.pnls)
        if n < self.admit_n:
            return self.admit_mult, f"pair-admit {n}/{self.admit_n} ×{self.admit_mult:.2f}"
        pf = self.profit_factor()
        if pf < self.admit_pf:
            return self.admit_mult, f"pair-admit PF={pf:.2f}<{self.admit_pf:.2f} ×{self.admit_mult:.2f}"
        return 1.0, ""

    def on_close(
        self,
        usd: float,
        r_mult: float,
        step: int,
        snowball_n: int = 0,
        day: str = "",
    ) -> str:
        self.pnls.append(float(usd))
        notes: list[str] = []
        if usd <= 0:
            self.fails += 1
            self.last_loss_r = abs(float(r_mult))
            if self.fails >= self.circuit_n:
                self.pause_until = int(step) + self.circuit_bars
                notes.append(f"circuit  {self.fails} losses → pause {self.circuit_bars} bars")
        else:
            self.fails = 0
            self.last_loss_r = 0.0
            if int(snowball_n) > 0:
                self.day_lock = str(day or self.day_lock)
                self.armed_protect = True
                notes.append("snowball lock  rest of UTC day + 1% peak protect")
        return "  ".join(notes)

    def should_rollback_plastic(self) -> bool:
        """fly-trader shadow check: if the last few trades paid worse than break-even, freeze D."""
        r = self.pnls[-SHADOW_N:]
        if len(r) < SHADOW_N:
            return False
        g = sum(x for x in r if x > 0)
        l = -sum(x for x in r if x <= 0)
        if l <= 1e-9:
            return False
        return (g / l) < SHADOW_PF

    def dump(self) -> dict:
        return {
            "peak": self.peak,
            "kill": self.kill,
            "fails": self.fails,
            "pnls": self.pnls[-40:],
            "last_loss_r": self.last_loss_r,
            "day_lock": self.day_lock,
            "armed_protect": self.armed_protect,
        }

    def load(self, data: dict) -> None:
        if not isinstance(data, dict):
            return
        self.peak = float(data.get("peak", self.peak) or 0.0)
        self.kill = bool(data.get("kill", False))
        self.fails = int(data.get("fails", 0))
        raw = data.get("pnls") or []
        if isinstance(raw, list):
            self.pnls = [float(x) for x in raw][-40:]
        self.last_loss_r = float(data.get("last_loss_r", 0.0) or 0.0)
        self.day_lock = str(data.get("day_lock") or "")
        self.armed_protect = bool(data.get("armed_protect", False))
