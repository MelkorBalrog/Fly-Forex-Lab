"""Bank profit + hold-risk from rate of change."""

from __future__ import annotations

from flyfx.risk.bank import (
    BankGuard,
    bank_target_usd,
    parse_bank_pct,
    parse_hold_risk_sens,
    sentiment_keep_risk,
    band_still_open,
)


def test_parse_bank_pct():
    assert parse_bank_pct(None) == 0.0
    assert parse_bank_pct("off") == 0.0
    assert parse_bank_pct(1.5) == 1.5
    assert parse_bank_pct(100) == 50.0
    assert parse_hold_risk_sens(1.5) == 1.5
    assert parse_hold_risk_sens(0) == 0.0


def test_full_bank_without_adverse_roc():
    g = BankGuard()
    g.on_open()
    # Quiet tape, float grows to full 1.5% of 100k
    why = None
    for usd in (500.0, 1000.0, 1500.0):
        why = g.observe(
            usd,
            side="SELL",
            roc=-0.05,
            jump=0.0,
            balance=100_000.0,
            bank_pct=1.5,
            hold_risk=True,
            sens=1.0,
        )
    assert why is not None
    assert "bank" in why
    assert "1.50%" in why


def test_adverse_roc_banks_early():
    g = BankGuard()
    g.on_open()
    # Build some float, then strong adverse cycle + decaying float
    g.observe(1200.0, side="SELL", roc=-0.8, jump=-0.1, balance=100_000.0, bank_pct=1.5)
    g.observe(1100.0, side="SELL", roc=0.6, jump=0.5, balance=100_000.0, bank_pct=1.5)
    g.observe(900.0, side="SELL", roc=0.9, jump=0.6, balance=100_000.0, bank_pct=1.5)
    why = g.observe(
        800.0,
        side="SELL",
        roc=1.2,
        jump=0.8,
        balance=100_000.0,
        bank_pct=1.5,
        hold_risk=True,
        sens=1.5,
    )
    # 800 < 1500 full target, but hold-risk should early-bank or hold-risk exit
    assert why is not None
    assert ("bank" in why) or why.startswith("hold-risk")
    assert g.last is not None
    assert g.last.risk > 0.3
    assert g.last.early_frac < 1.0


def test_small_open_profit_is_not_sentiment():
    g = BankGuard()
    g.on_open()
    why = g.observe(
        400.0,
        side="SELL",
        roc=-0.2,
        jump=0.0,
        balance=100_000.0,
        bank_pct=0.0,
        hold_risk=True,
        sens=1.0,
        unreal_r=0.40,
    )
    assert why is None
    assert sentiment_keep_risk(0.40, 400.0, 100_000.0) < 0.72


def test_high_open_r_closes_without_a_bank_target():
    g = BankGuard()
    g.on_open()
    why = g.observe(
        3_500.0,
        side="BUY",
        roc=0.1,
        jump=0.0,
        balance=100_000.0,
        bank_pct=0.0,
        hold_risk=True,
        sens=1.0,
        unreal_r=3.20,
    )
    assert why is not None
    assert why.startswith("sentiment")


def test_a_small_open_gain_stays_with_the_trail():
    g = BankGuard()
    g.on_open()
    why = g.observe(
        1_366.0,
        side="SELL",
        roc=-0.1,
        jump=0.0,
        balance=100_000.0,
        bank_pct=0.0,
        hold_risk=True,
        sens=1.0,
        unreal_r=0.92,
    )
    assert why is None
    why = g.observe(
        1_053.0,
        side="SELL",
        roc=0.2,
        jump=0.0,
        balance=100_000.0,
        bank_pct=0.0,
        hold_risk=True,
        sens=1.0,
        unreal_r=0.70,
    )
    assert why is None


def test_high_open_dollars_close_even_if_r_is_diluted():
    g = BankGuard()
    g.on_open()
    why = g.observe(
        3_500.0,
        side="SELL",
        roc=-0.1,
        jump=0.0,
        balance=100_000.0,
        bank_pct=0.0,
        hold_risk=True,
        sens=1.0,
        unreal_r=0.55,
    )
    assert why is not None
    assert why.startswith("sentiment")
    assert "3.50% of balance" in why


def test_a_loser_has_no_sentiment_risk():
    assert sentiment_keep_risk(-0.8, -200.0, 100_000.0) == 0.0
    assert sentiment_keep_risk(0.0, 0.0, 100_000.0) == 0.0
    g = BankGuard()
    g.on_open()
    why = g.observe(
        -200.0,
        side="BUY",
        roc=-0.4,
        jump=0.0,
        balance=100_000.0,
        bank_pct=0.0,
        unreal_r=-0.8,
    )
    assert why is None


def test_hold_risk_off_does_not_bank_on_sentiment():
    g = BankGuard()
    g.on_open()
    why = g.observe(
        2_000.0,
        side="SELL",
        roc=-0.2,
        jump=0.0,
        balance=100_000.0,
        bank_pct=0.0,
        hold_risk=False,
        sens=1.0,
        unreal_r=2.0,
    )
    assert why is None


def test_open_bollinger_keeps_a_high_profit_trade():
    g = BankGuard()
    g.on_open()
    why = None
    for bw in (0.010, 0.011, 0.012, 0.013):
        why = g.observe(
            1_500.0,
            side="SELL",
            roc=-0.1,
            jump=0.0,
            balance=100_000.0,
            bank_pct=0.0,
            hold_risk=True,
            sens=1.0,
            unreal_r=1.80,
            bb_bw=bw,
        )
    assert why is None
    assert band_still_open((0.010, 0.011, 0.012, 0.013))


def test_closing_bollinger_banks_a_high_profit_trade():
    g = BankGuard()
    g.on_open()
    why = None
    for bw in (0.014, 0.013, 0.012, 0.011):
        why = g.observe(
            4_000.0,
            side="SELL",
            roc=-0.1,
            jump=0.0,
            balance=100_000.0,
            bank_pct=0.0,
            hold_risk=True,
            sens=1.0,
            unreal_r=2.0,
            bb_bw=bw,
        )
    assert why is not None
    assert why.startswith("sentiment")
    assert "bb closing" in why


def test_leaving_the_band_edge_banks_before_a_deep_giveback():
    g = BankGuard()
    g.on_open()
    why = None
    for bw in (0.010, 0.012, 0.014, 0.015):
        why = g.observe(
            4_000.0,
            side="SELL",
            roc=-0.2,
            jump=0.0,
            balance=100_000.0,
            bank_pct=0.0,
            hold_risk=True,
            sens=1.0,
            unreal_r=2.0,
            bb_bw=bw,
            bb_pct=0.12,
        )
    assert why is None
    why = g.observe(
        3_200.0,
        side="SELL",
        roc=0.1,
        jump=0.0,
        balance=100_000.0,
        bank_pct=0.0,
        hold_risk=True,
        sens=1.0,
        unreal_r=1.6,
        bb_bw=0.015,
        bb_pct=0.55,
    )
    assert why is not None
    assert why.startswith("sentiment")
    assert "peak giveback" in why
    assert "peak $4,000" in why


def test_open_band_at_the_peak_keeps_the_trade():
    """Mid-band is not a reason to flatten while the dollars are still at the high."""
    g = BankGuard()
    g.on_open()
    why = None
    for bw in (0.010, 0.012, 0.014, 0.015):
        why = g.observe(
            4_000.0,
            side="SELL",
            roc=-0.2,
            jump=0.0,
            balance=100_000.0,
            bank_pct=0.0,
            hold_risk=True,
            sens=1.0,
            unreal_r=2.0,
            bb_bw=bw,
            bb_pct=0.55,
        )
    assert why is None


def test_giveback_from_a_high_peak_closes_while_the_band_is_still_open():
    g = BankGuard()
    g.on_open()
    why = None
    for bw, usd in ((0.010, 1_000.0), (0.012, 2_500.0), (0.014, 4_000.0), (0.015, 4_000.0)):
        why = g.observe(
            usd,
            side="SELL",
            roc=-0.2,
            jump=0.0,
            balance=100_000.0,
            bank_pct=0.0,
            hold_risk=True,
            sens=1.0,
            unreal_r=2.0,
            bb_bw=bw,
            bb_pct=0.12,
        )
    assert why is None
    why = g.observe(
        3_400.0,
        side="SELL",
        roc=-0.1,
        jump=0.0,
        balance=100_000.0,
        bank_pct=0.0,
        hold_risk=True,
        sens=1.0,
        unreal_r=1.7,
        bb_bw=0.015,
        bb_pct=0.12,
    )
    assert why is not None
    assert "peak giveback" in why
    assert "float $3,400" in why


def test_hold_risk_off_needs_full_target():
    g = BankGuard()
    g.on_open()
    why = g.observe(
        800.0,
        side="SELL",
        roc=1.2,
        jump=0.8,
        balance=100_000.0,
        bank_pct=1.5,
        hold_risk=False,
        sens=1.0,
    )
    assert why is None
    assert bank_target_usd(100_000, 1.5) == 1500.0
