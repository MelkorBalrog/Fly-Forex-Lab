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

    def required_margin(self, lots: float, price: float) -> float:
        return lots * self.contract * price / self.leverage

    def floating(self, bid: float, ask: float) -> float:
        if self.pos is None:
            return 0.0
        px = bid if self.pos.side == "BUY" else ask
        diff = (px - self.pos.entry) if self.pos.side == "BUY" else (self.pos.entry - px)
        return money_pnl(diff, self.pos.lots, self.contract)

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
        comm = 0.5 * self.commission_rt_per_lot * lots
        spread_cost = abs(money_pnl((ask - bid) / 2.0, lots, self.contract))
        slip_cost = abs(money_pnl(slip, lots, self.contract))
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

    def close(self, bid: float, ask: float, slip: float, ts: int, reason: str) -> CloseResult:
        if self.pos is None:
            return CloseResult(False, reason="no position")
        self._apply_swap(ts)
        pos = self.pos
        fill = (bid - slip) if pos.side == "BUY" else (ask + slip)
        diff = (fill - pos.entry) if pos.side == "BUY" else (pos.entry - fill)
        gross = money_pnl(diff, pos.lots, self.contract)
        comm = 0.5 * self.commission_rt_per_lot * pos.lots
        spread_cost = abs(money_pnl((ask - bid) / 2.0, pos.lots, self.contract))
        slip_cost = abs(money_pnl(slip, pos.lots, self.contract))
        self.balance -= comm
        self.balance += gross
        self.cost_commission += comm
        self.cost_spread += spread_cost
        self.cost_slippage += slip_cost
        net = gross - comm - pos.commission_open + pos.swap_usd
        entry = pos.entry
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
        )
