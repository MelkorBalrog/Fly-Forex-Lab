"""Retail-style FX account: spread, commission, slippage, swap, leverage, stop-out."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


def pip_size(price: float) -> float:
    return 0.01 if price > 20 else 0.0001


def money_pnl(price_diff: float, lots: float, contract: float = 100_000.0) -> float:
    return price_diff * contract * lots


def utc_hour(ts: int) -> int:
    return time.gmtime(ts).tm_hour


def utc_date_ordinal(ts: int) -> int:
    g = time.gmtime(ts)
    return g.tm_year * 1000 + g.tm_yday


def session_spread_mult(ts: int) -> float:
    h = utc_hour(ts)
    if h in (21, 22, 23, 0, 1):
        return 1.8
    if h in (7, 8, 12, 13, 14, 15):
        return 1.0
    return 1.3


@dataclass
class SimPosition:
    ticket: int
    side: str
    lots: float
    entry: float
    open_time: int
    last_swap_day: int
    commission_open: float = 0.0
    slippage_open: float = 0.0
    swap_usd: float = 0.0


@dataclass
class Fill:
    ok: bool
    ticket: int = 0
    price: float = 0.0
    reason: str = ""
    margin: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    spread_cost: float = 0.0


@dataclass
class CloseResult:
    ok: bool
    gross: float = 0.0
    net: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    spread_cost: float = 0.0
    swap: float = 0.0
    exit_price: float = 0.0
    entry_price: float = 0.0
    reason: str = ""
    lots: float = 0.0
    remaining: float = 0.0


class AccountSim:
    def __init__(
        self,
        balance: float = 100_000.0,
        leverage: float = 100.0,
        contract: float = 100_000.0,
        spread: float = 0.00010,
        commission_rt_per_lot: float = 7.0,
        slippage_pips: float = 0.2,
        swap_long_per_lot: float = -0.72,
        swap_short_per_lot: float = 0.12,
        stop_out: float = 0.50,
        margin_call: float = 1.00,
        quote: str = "USD",
    ):
        self.balance = balance
        self.start_balance = balance
        self.leverage = max(leverage, 1.0)
        self.contract = contract
        self.spread = spread
        self.commission_rt_per_lot = commission_rt_per_lot
        self.slippage_pips = slippage_pips
        self.swap_long_per_lot = swap_long_per_lot
        self.swap_short_per_lot = swap_short_per_lot
        self.stop_out = stop_out
        self.margin_call = margin_call
        self.quote = str(quote or "USD").upper()
        self.pos: SimPosition | None = None
        self.next_ticket = 2000
        self.cost_spread = 0.0
        self.cost_commission = 0.0
        self.cost_slippage = 0.0
        self.cost_swap = 0.0
        self.rejected = 0
        self.stopouts = 0

    def quotes(self, mid: float, ts: int, high: float, low: float, atr: float) -> tuple[float, float, float]:
        pip = pip_size(mid)
        wide = self.spread * session_spread_mult(ts)
        rng = max(high - low, pip)
        slip = (self.slippage_pips * pip) * (1.0 + 0.25 * rng / max(atr, pip))
        slip = min(slip, 4.0 * pip)
        half = wide / 2.0
        return mid - half, mid + half, slip

    def to_usd(self, price_diff: float, lots: float, mid: float) -> float:
        """PnL in USD. Quote-USD pairs are 1:1; JPY/CHF/CAD/GBP quotes are converted at mid."""
        raw = money_pnl(price_diff, lots, self.contract)
        if self.quote == "USD":
            return raw
        return raw / max(abs(mid), 1e-9)

    def required_margin(self, lots: float, price: float) -> float:
        return lots * self.contract * price / self.leverage

    def floating(self, bid: float, ask: float) -> float:
        if self.pos is None:
            return 0.0
        px = bid if self.pos.side == "BUY" else ask
        diff = (px - self.pos.entry) if self.pos.side == "BUY" else (self.pos.entry - px)
        mid = (bid + ask) / 2.0
        return self.to_usd(diff, self.pos.lots, mid)

    def equity(self, bid: float, ask: float) -> float:
        return self.balance + self.floating(bid, ask)

    def margin_used(self, price: float) -> float:
        if self.pos is None:
            return 0.0
        return self.required_margin(self.pos.lots, price)

    def snapshot(self, bid: float, ask: float) -> dict:
        mid = (bid + ask) / 2.0
        eq = self.equity(bid, ask)
        mar = self.margin_used(mid)
        free = eq - mar
        level = (eq / mar * 100.0) if mar > 1e-9 else float("inf")
        return {
            "balance": self.balance,
            "equity": eq,
            "margin": mar,
            "free": free,
            "level": level,
            "floating": self.floating(bid, ask),
        }

    def _apply_swap(self, ts: int) -> None:
        if self.pos is None:
            return
        day = utc_date_ordinal(ts)
        if day == self.pos.last_swap_day:
            return
        nights = 3 if time.gmtime(ts).tm_wday == 2 else 1
        rate = self.swap_long_per_lot if self.pos.side == "BUY" else self.swap_short_per_lot
        usd = rate * self.pos.lots * nights
        self.pos.swap_usd += usd
        self.balance += usd
        self.cost_swap += usd
        self.pos.last_swap_day = day

    def mark(self, bid: float, ask: float, ts: int) -> str | None:
        self._apply_swap(ts)
        if self.pos is None:
            return None
        snap = self.snapshot(bid, ask)
        if snap["margin"] > 0 and snap["level"] / 100.0 <= self.stop_out:
            self.stopouts += 1
            return "stop-out"
        return None

    def open(self, side: str, lots: float, bid: float, ask: float, slip: float, ts: int) -> Fill:
        if self.pos is not None:
            return Fill(False, reason="already in a position")
        fill = (ask + slip) if side == "BUY" else (bid - slip)
        margin = self.required_margin(lots, fill)
        eq = self.equity(bid, ask)
        if margin > eq - 1e-9:
            self.rejected += 1
            return Fill(False, reason=f"not enough margin (need ${margin:.2f}, free ${eq:.2f})")
        mid = (bid + ask) / 2.0
        comm = 0.5 * self.commission_rt_per_lot * lots
        spread_cost = abs(self.to_usd((ask - bid) / 2.0, lots, mid))
        slip_cost = abs(self.to_usd(slip, lots, mid))
        self.balance -= comm
        self.cost_commission += comm
        self.cost_spread += spread_cost
        self.cost_slippage += slip_cost
        self.next_ticket += 1
        self.pos = SimPosition(
            ticket=self.next_ticket,
            side=side,
            lots=lots,
            entry=fill,
            open_time=ts,
            last_swap_day=utc_date_ordinal(ts),
            commission_open=comm,
            slippage_open=slip_cost,
        )
        return Fill(
            True,
            ticket=self.pos.ticket,
            price=fill,
            margin=margin,
            commission=comm,
            slippage=slip_cost,
            spread_cost=spread_cost,
        )

    def add(self, lots: float, bid: float, ask: float, slip: float, ts: int) -> Fill:
        """Pyramid into an open position. Volume-weighted average entry."""
        if self.pos is None:
            return Fill(False, reason="no position to add to")
        lots = float(lots)
        if lots <= 0:
            return Fill(False, reason="add lots must be > 0")
        pos = self.pos
        fill = (ask + slip) if pos.side == "BUY" else (bid - slip)
        extra_margin = self.required_margin(lots, fill)
        snap = self.snapshot(bid, ask)
        if extra_margin > snap["free"] - 1e-9:
            self.rejected += 1
            return Fill(
                False,
                reason=f"not enough margin to snowball (need ${extra_margin:.2f}, free ${snap['free']:.2f})",
            )
        mid = (bid + ask) / 2.0
        comm = 0.5 * self.commission_rt_per_lot * lots
        spread_cost = abs(self.to_usd((ask - bid) / 2.0, lots, mid))
        slip_cost = abs(self.to_usd(slip, lots, mid))
        self.balance -= comm
        self.cost_commission += comm
        self.cost_spread += spread_cost
        self.cost_slippage += slip_cost
        new_lots = pos.lots + lots
        pos.entry = (pos.entry * pos.lots + fill * lots) / new_lots
        pos.lots = new_lots
        pos.commission_open += comm
        pos.slippage_open += slip_cost
        margin = self.required_margin(pos.lots, fill)
        return Fill(
            True,
            ticket=pos.ticket,
            price=fill,
            margin=margin,
            commission=comm,
            slippage=slip_cost,
            spread_cost=spread_cost,
        )

    def reduce(
        self, lots: float, bid: float, ask: float, slip: float, ts: int, reason: str
    ) -> CloseResult:
        """Close part of the open position; leave the remainder running."""
        if self.pos is None:
            return CloseResult(False, reason="no position")
        lots = float(lots)
        if lots <= 0:
            return CloseResult(False, reason="reduce lots must be > 0")
        pos = self.pos
        if lots >= pos.lots - 1e-9:
            return self.close(bid, ask, slip, ts, reason)
        self._apply_swap(ts)
        fill = (bid - slip) if pos.side == "BUY" else (ask + slip)
        mid = (bid + ask) / 2.0
        frac = lots / pos.lots
        diff = (fill - pos.entry) if pos.side == "BUY" else (pos.entry - fill)
        gross = self.to_usd(diff, lots, mid)
        comm = 0.5 * self.commission_rt_per_lot * lots
        spread_cost = abs(self.to_usd((ask - bid) / 2.0, lots, mid))
        slip_cost = abs(self.to_usd(slip, lots, mid))
        # Allocate a share of open commission / swap to the closed slice.
        comm_open_part = pos.commission_open * frac
        slip_open_part = pos.slippage_open * frac
        swap_part = pos.swap_usd * frac
        self.balance -= comm
        self.balance += gross
        self.cost_commission += comm
        self.cost_spread += spread_cost
        self.cost_slippage += slip_cost
        net = gross - comm - comm_open_part + swap_part
        pos.lots -= lots
        pos.commission_open -= comm_open_part
        pos.slippage_open -= slip_open_part
        pos.swap_usd -= swap_part
        return CloseResult(
            True,
            gross=gross,
            net=net,
            commission=comm + comm_open_part,
            slippage=slip_cost + slip_open_part,
            spread_cost=spread_cost,
            swap=swap_part,
            exit_price=fill,
            entry_price=pos.entry,
            reason=reason,
            lots=lots,
            remaining=pos.lots,
        )

    def close(self, bid: float, ask: float, slip: float, ts: int, reason: str) -> CloseResult:
        if self.pos is None:
            return CloseResult(False, reason="no position")
        self._apply_swap(ts)
        pos = self.pos
        fill = (bid - slip) if pos.side == "BUY" else (ask + slip)
        mid = (bid + ask) / 2.0
        diff = (fill - pos.entry) if pos.side == "BUY" else (pos.entry - fill)
        gross = self.to_usd(diff, pos.lots, mid)
        comm = 0.5 * self.commission_rt_per_lot * pos.lots
        spread_cost = abs(self.to_usd((ask - bid) / 2.0, pos.lots, mid))
        slip_cost = abs(self.to_usd(slip, pos.lots, mid))
        self.balance -= comm
        self.balance += gross
        self.cost_commission += comm
        self.cost_spread += spread_cost
        self.cost_slippage += slip_cost
        net = gross - comm - pos.commission_open + pos.swap_usd
        entry = pos.entry
        closed_lots = pos.lots
        self.pos = None
        return CloseResult(
            True,
            gross=gross,
            net=net,
            commission=comm + pos.commission_open,
            slippage=slip_cost + pos.slippage_open,
            spread_cost=spread_cost,
            swap=pos.swap_usd,
            exit_price=fill,
            entry_price=entry,
            reason=reason,
            lots=closed_lots,
            remaining=0.0,
        )
