"""Profit recycle: BB squeeze→expand gate + 1.5× win sizing budget."""

from __future__ import annotations

from flyfx.risk.profit_recycle import ProfitRecycleGuard


def test_win_arms_budget_and_blocks_until_bb_opens():
    g = ProfitRecycleGuard(enabled=True, mult=1.5)
    note = g.on_close(1000.0)
    assert "1,000" in note or "1000" in note.replace(",", "")
    assert abs(g.budget - 1500.0) < 1e-6
    assert g.phase == "squeeze"
    ok, why = g.allow_entry()
    assert ok
    assert abs(g.size_budget() - 1500.0) < 1e-6

    # Still wide bands → stay in squeeze, but the book may still enter.
    g.observe({"bb_bw": 0.012, "bb_bw_ma": 0.010})  # ratio 1.2
    assert g.phase == "squeeze"
    assert g.allow_entry()[0]

    # Squeeze
    g.observe({"bb_bw": 0.008, "bb_bw_ma": 0.010})  # 0.80
    assert g.phase == "expand"
    assert g.allow_entry()[0]

    # Expand / open
    g.observe({"bb_bw": 0.011, "bb_bw_ma": 0.010})  # 1.10
    assert g.phase == "ready"
    ok, _ = g.allow_entry()
    assert ok
    assert abs(g.size_budget() - 1500.0) < 1e-6


def test_loss_smaller_than_the_win_keeps_a_cushion():
    g = ProfitRecycleGuard(enabled=True)
    g.on_close(2000.0)
    note = g.on_close(-500.0)
    assert g.active
    assert abs(g.last_win - 1500.0) < 1e-6
    assert abs(g.budget - 2250.0) < 1e-6
    assert g.phase == "squeeze"
    assert g.allow_entry()[0]
    assert abs(g.size_budget() - 2250.0) < 1e-6
    assert "cushion" in note


def test_loss_that_wipes_the_win_clears_recycle():
    g = ProfitRecycleGuard(enabled=True)
    g.on_close(500.0)
    assert g.active
    g.on_close(-600.0)
    assert not g.active
    assert g.size_budget() is None
    assert g.allow_entry()[0]


def test_second_profit_replaces_the_budget():
    g = ProfitRecycleGuard(enabled=True, mult=1.5)
    g.on_close(2_000.0)
    assert abs(g.size_budget() - 3_000.0) < 1e-6
    note = g.on_close(800.0)
    assert g.win_n == 2
    assert abs(g.last_win - 800.0) < 1e-6
    assert abs(g.size_budget() - 1_200.0) < 1e-6
    assert "only" in note
    assert "800" in note.replace(",", "")


def test_disabled_always_allows():
    g = ProfitRecycleGuard(enabled=False)
    g.on_close(2000.0)
    assert g.allow_entry()[0]
    assert g.size_budget() is None
