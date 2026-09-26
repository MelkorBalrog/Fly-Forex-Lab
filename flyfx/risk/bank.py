"""Bank open profit, with hold-risk from rate-of-change over time.

``bank_pct``: close when floating USD ≥ that %% of complete balance.

Hold-risk: if price / float move against the trade quickly, risk of *keeping*
the trade rises. Higher risk lowers the effective bank threshold so the trade
can close before the full BANK % is reached. Extreme risk can flatten a still-
green trade that is giving back a large share of its peak float.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

ROC_WINDOW = 6  # bars — same clock as CycleTape
RISK_HARD = 0.90  # flatten green trade when risk ≥ this and peak giveback is large
GIVEBACK_MIN = 0.30  # need ≥30% of peak float given back for hard hold-risk exit
EARLY_FLOOR = 0.20  # never require less than 20% of full bank target
# Open profit itself is a reason to get out: attachment to a winner is the risk.
SENTIMENT_CLOSE = 0.72
SENTIMENT_R_UNIT = 1.6  # ~1.6R of open profit → keep-risk ~0.76
SENTIMENT_R_MIN = 1.25  # R alone can close once profit is clearly high
SENTIMENT_BAL = 0.030  # ~3% of balance. A $1k scalp pullback is not this.
# Once a high peak exists, give back at most this share before banking.
# 4k → ~3.4k. A $1.3k open gain is still the trail's trade.
PEAK_KEEP = 0.85
PEAK_BANK = 0.030
# One bar that almost doubles an open profit that is already $2k is closed now.
# $2,000 → $5,000 is this. A $1,073 → $2,253 snowball start is not.
SPIKE_MULT = 1.85
SPIKE_FLOOR = 2_000.0


def parse_bank_pct(spec) -> float:
    """Return bank %% of balance (0 = off). Clipped to 0–50."""
    if spec is None:
        return 0.0
    text = str(spec).strip().lower()
    if text in ("", "off", "none", "auto"):
        return 0.0
    try:
        n = float(text)
    except (TypeError, ValueError):
        return 0.0
    if n <= 0:
        return 0.0
    return float(min(50.0, n))


def parse_hold_risk_sens(spec) -> float:
    """Sensitivity 0–2 (1 = nominal). 0 disables hold-risk scaling."""
    if spec is None:
        return 1.0
    try:
        n = float(spec)
    except (TypeError, ValueError):
        return 1.0
    if n <= 0:
        return 0.0
    return float(min(2.0, max(0.0, n)))


def bank_target_usd(balance: float, bank_pct: float) -> float:
    """Dollar profit target from %% of complete balance."""
    pct = parse_bank_pct(bank_pct)
    if pct <= 0:
        return 0.0
    base = max(float(balance or 0.0), 0.0)
    return base * (pct / 100.0)


def _signed(side: str, value: float) -> float:
    return float(value) if str(side) == "BUY" else -float(value)


def _tanh01(x: float) -> float:
    """Smooth 0..1 map for positive magnitudes."""
    import math

    return float(math.tanh(max(float(x), 0.0)))


def sentiment_keep_risk(unreal_r: float, float_usd: float, balance: float) -> float:
    """Risk of keeping a trade whose profit is already high.

    A flat or losing trade scores 0: there is no open gain to get attached to.
    The score rises as open R grows and as open dollars become a large share of
    the balance. High score means close — staying in is the sentiment.
    """
    r = float(unreal_r)
    dollars = float(float_usd)
    if r <= 0.0 or dollars <= 0.0:
        return 0.0
    r_part = _tanh01(r / SENTIMENT_R_UNIT)
    bal = max(float(balance), 1.0)
    usd_part = _tanh01((dollars / bal) / SENTIMENT_BAL)
    return float(max(r_part, usd_part))


def band_still_open(widths: list[float] | tuple[float, ...]) -> bool:
    """True only while bandwidth is still at its recent high.

    A dip off that high means the squeeze expansion has started to close,
    even if the band is still wide versus where it began.
    """
    vals = [float(x) for x in widths if float(x) > 1e-12]
    if len(vals) < 3:
        return False
    peak = max(vals)
    now = vals[-1]
    if peak <= 1e-12:
        return False
    return now >= peak * 0.985


@dataclass(frozen=True)
class HoldRiskPlan:
    risk: float
    early_frac: float
    float_roc: float
    peak_float: float
    giveback: float
    note: str


class BankGuard:
    """Per-trade float tape + hold-risk / early-bank decisions."""

    def __init__(
        self,
        *,
        spike_mult: float = SPIKE_MULT,
        spike_floor: float = SPIKE_FLOOR,
        peak_bank: float = PEAK_BANK,
    ) -> None:
        self.spike_mult = float(spike_mult)
        self.spike_floor = float(spike_floor)
        self.peak_bank = float(peak_bank)
        self._floats: deque[float] = deque(maxlen=ROC_WINDOW + 1)
        self._bb: deque[float] = deque(maxlen=ROC_WINDOW + 1)
        self.peak_float = 0.0
        self.active = False
        self.last: HoldRiskPlan | None = None

    def reset(self) -> None:
        self._floats.clear()
        self._bb.clear()
        self.peak_float = 0.0
        self.active = False
        self.last = None

    def float_step(self) -> tuple[float | None, float | None]:
        """Previous bar's open profit and this bar's, once both exist."""
        if not self._floats:
            return None, None
        now = float(self._floats[-1])
        if len(self._floats) < 2:
            return None, now
        return float(self._floats[-2]), now

    def on_open(self) -> None:
        self.reset()
        self.active = True

    def observe(
        self,
        float_usd: float,
        *,
        side: str,
        roc: float,
        jump: float,
        balance: float,
        bank_pct: float,
        hold_risk: bool = True,
        sens: float = 1.0,
        unreal_r: float | None = None,
        bb_bw: float | None = None,
        bb_pct: float | None = None,
        float_best: float | None = None,
    ) -> str | None:
        """Push this bar's float; return a close reason or None."""
        if not self.active:
            self.active = True
        f = float(float_usd)
        # The close mark can sit under a wick that already doubled the open
        # profit. The spike uses the better of the two; giveback still uses
        # the close so a wick does not become a new peak.
        best = f if float_best is None else max(f, float(float_best))
        prev = float(self._floats[-1]) if self._floats else None
        self._floats.append(f)
        if f > self.peak_float:
            self.peak_float = f
        if (
            prev is not None
            and             prev >= self.spike_floor
            and best >= prev * self.spike_mult
        ):
            return (
                f"profit spike  ${prev:,.0f} → ${best:,.0f}  "
                f"({best / prev:.2f}× this cycle)"
            )
        if bb_bw is not None and float(bb_bw) > 1e-12:
            self._bb.append(float(bb_bw))

        sens_n = parse_hold_risk_sens(sens)
        plan = self._score(side, roc, jump, balance, sens_n if hold_risk else 0.0)
        self.last = plan

        full = bank_target_usd(balance, bank_pct)
        pct = parse_bank_pct(bank_pct)

        # Early bank: high hold-risk → need less of the full BANK % target.
        if full > 0 and f + 1e-9 >= full * plan.early_frac:
            return (
                f"bank {pct:.2f}% (${full * plan.early_frac:,.0f}"
                f"/{full:,.0f} of ${max(float(balance), 0):,.0f}"
                f"  hold-risk {plan.risk:.2f})"
            )

        # Hard hold-risk: still green but giving back fast — close before bank %.
        if (
            hold_risk
            and sens_n > 0
            and plan.risk >= RISK_HARD
            and f > 0.0
            and self.peak_float > 1e-6
            and plan.giveback >= GIVEBACK_MIN
        ):
            return (
                f"hold-risk {plan.risk:.2f}  "
                f"float ${f:,.0f}  peak ${self.peak_float:,.0f}  "
                f"giveback {100.0 * plan.giveback:.0f}%  {plan.note}"
            )

        # Sentiment: an open band may keep running, but only while the
        # dollars are still near the high. A wide band can stay "at the edge"
        # for the whole trip from a $4k peak back to the stop — that hold is
        # what turned the winner into a loser. Bank on the first 15% giveback
        # of a high peak. If the width itself rolls off and profit is already
        # high, bank without waiting for that dip.
        if unreal_r is not None and hold_risk and sens_n > 0 and f > 0.0:
            # Snowball winners on this book peak near $1–2k and then add.
            # Banking those as "already high" cut the EURUSD week down to one
            # trade. Only a peak around 3% of the balance is the giveback case.
            if self.peak_float < max(float(balance), 1.0) * self.peak_bank:
                return None
            running = band_still_open(tuple(self._bb))
            keep = sentiment_keep_risk(float(unreal_r), f, float(balance)) * sens_n
            peak_was_high = (
                sentiment_keep_risk(float(unreal_r), self.peak_float, float(balance)) * sens_n
                >= SENTIMENT_CLOSE
            )
            gave_back = (
                self.peak_float > 1e-6
                and f <= self.peak_float * PEAK_KEEP
                and peak_was_high
            )
            if gave_back:
                bal = max(float(balance), 1.0)
                return (
                    f"sentiment {max(keep, 0.0):.2f}  +{float(unreal_r):.2f}R  "
                    f"float ${f:,.0f}  peak ${self.peak_float:,.0f}  "
                    f"({100.0 * f / bal:.2f}% of balance)  peak giveback"
                )
            if running:
                return None
            sens_floor = max(sens_n, 0.25)
            high_r = float(unreal_r) >= (SENTIMENT_R_MIN / sens_floor)
            high_usd = f >= max(float(balance), 0.0) * (SENTIMENT_BAL / sens_floor)
            if keep >= SENTIMENT_CLOSE and (high_r or high_usd):
                bal = max(float(balance), 1.0)
                return (
                    f"sentiment {keep:.2f}  +{float(unreal_r):.2f}R  "
                    f"float ${f:,.0f}  peak ${self.peak_float:,.0f}  "
                    f"({100.0 * f / bal:.2f}% of balance)  bb closing"
                )
        return None

    def _score(
        self,
        side: str,
        roc: float,
        jump: float,
        balance: float,
        sens: float,
    ) -> HoldRiskPlan:
        n = len(self._floats)
        if n >= 2:
            float_roc = (self._floats[-1] - self._floats[0]) / float(n - 1)
        else:
            float_roc = 0.0

        against = max(0.0, -_signed(side, roc))
        jump_against = max(0.0, -_signed(side, jump))
        base = max(float(balance), 1.0)
        # Negative float_roc (losing USD/bar) raises risk; scale by 0.2% of balance per bar.
        decay_unit = 0.002 * base
        decay = max(0.0, -float_roc) / max(decay_unit, 1e-9)
        giveback = 0.0
        if self.peak_float > 1e-6:
            giveback = max(0.0, (self.peak_float - self._floats[-1]) / self.peak_float)

        raw = (
            0.35 * _tanh01(against)
            + 0.25 * _tanh01(jump_against)
            + 0.25 * _tanh01(decay)
            + 0.15 * min(giveback, 1.0)
        )
        risk = min(1.5, max(0.0, raw * float(sens))) if sens > 0 else 0.0
        # early_frac: risk 0 → 1.0 (full bank %); risk 1 → EARLY_FLOOR
        early = max(EARLY_FLOOR, 1.0 - 0.80 * min(risk, 1.0)) if sens > 0 else 1.0
        note = (
            f"roc {roc:+.2f} jump {jump:+.2f}  "
            f"d$/bar {float_roc:+.0f}  giveback {100.0 * giveback:.0f}%"
        )
        return HoldRiskPlan(
            risk=float(risk),
            early_frac=float(early),
            float_roc=float(float_roc),
            peak_float=float(self.peak_float),
            giveback=float(giveback),
            note=note,
        )


def bank_profit_reason(
    float_usd: float,
    *,
    balance: float,
    bank_pct: float,
) -> str | None:
    """Legacy full-target check (no hold-risk). Prefer ``BankGuard.observe``."""
    target = bank_target_usd(balance, bank_pct)
    if target <= 0:
        return None
    if float(float_usd) + 1e-9 < target:
        return None
    pct = parse_bank_pct(bank_pct)
    return f"bank {pct:.2f}% (${target:,.0f} of ${max(float(balance), 0):,.0f})"
