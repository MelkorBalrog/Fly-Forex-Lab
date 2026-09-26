"""Per-pair book knobs and the pre-sim search."""

from flyfx.brain.book_cfg import (
    MAX_TUNE_EVALS,
    BookCfg,
    BookTuneFleet,
    load_book,
    prescribe_book,
    save_book,
    search_book,
)
from flyfx.risk.fly_rails import BookRails


def test_roundtrip_is_per_symbol(tmp_path, monkeypatch):
    import flyfx.brain.book_cfg as mod

    monkeypatch.setattr(mod, "BRAIN_DIR", tmp_path)
    save_book("EURUSD", BookCfg(spike_floor=2000.0), net_usd=8486.45, balance=100_000.0)
    save_book("GBPUSD", BookCfg(spike_floor=1500.0, target_return=0.05))
    eur = load_book("EURUSD")
    gbp = load_book("GBPUSD")
    assert eur is not None and gbp is not None
    assert eur.spike_floor == 2000.0
    assert gbp.spike_floor == 1500.0
    assert gbp.target_return == 0.05
    assert (tmp_path / "EURUSD.book.json").exists()


def test_search_stops_when_the_saved_book_already_clears_the_target():
    calls = []

    def evaluate(cfg: BookCfg) -> float:
        calls.append(cfg.tape_floor)
        return 8_000.0

    best, net = search_book(BookCfg(), 100_000.0, evaluate)
    assert net == 8_000.0
    assert len(calls) == 1
    assert best.tape_floor == 0.90


def test_search_changes_a_knob_until_the_return_clears_7pct():
    def evaluate(cfg: BookCfg) -> float:
        if cfg.day_lock_usd <= 500.0:
            return 7_500.0
        return 2_000.0

    best, net = search_book(BookCfg(), 100_000.0, evaluate)
    assert net == 7_500.0
    assert best.day_lock_usd == 500.0


def test_search_keeps_adjusting_after_a_round_that_is_still_under_target():
    def evaluate(cfg: BookCfg) -> float:
        if cfg.worth_impulse >= 0.96 and cfg.tape_floor >= 1.0:
            return 8_000.0
        if cfg.worth_impulse >= 0.96:
            return 4_000.0
        return 1_000.0

    best, net = search_book(BookCfg(), 100_000.0, evaluate)
    assert net == 8_000.0
    assert best.worth_impulse >= 0.96
    assert best.tape_floor >= 1.0


def test_search_stops_after_the_budget_when_the_target_stays_out_of_reach():
    calls = []

    def evaluate(cfg: BookCfg) -> float:
        calls.append(cfg.brief())
        return 100.0

    best, net = search_book(BookCfg(), 100_000.0, evaluate)
    assert net == 100.0
    assert len(calls) <= MAX_TUNE_EVALS
    assert len(calls) >= 2
    assert best.target_return == 0.07


def test_a_net_that_improves_the_book_gains_weight():
    fleet = BookTuneFleet()
    before = float(fleet.weights()[2])
    fleet.teach(BookCfg(), 2, True, 1)
    assert float(fleet.weights()[2]) > before


def test_a_target_exit_that_left_the_move_gets_the_trend_hold_book():
    import calendar
    import time

    opened = int(calendar.timegm(time.strptime("2026-09-22 09:00", "%Y-%m-%d %H:%M")))
    later = opened + 3600
    bars = [
        {"time": opened, "high": 0.7110, "low": 0.7108, "close": 0.71082},
        {"time": later, "high": 0.7070, "low": 0.70299, "close": 0.70350},
    ]
    trades = [
        {
            "side": "SELL",
            "open_time": "2026-09-22 09:00",
            "entry": 0.71082,
            "pips": 38.7,
            "usd": 3858.0,
            "lots": 9.97,
            "reason": "ATR target",
        }
    ]
    out = prescribe_book(BookCfg(), trades, bars, 100_000.0, 0.0001)
    assert out.tp_atr == 40.0
    assert out.hold_risk == 0.0
    assert out.max_hold_bars == 400.0
    assert out.flat_hour == 24.0
    assert out.kill_dd == 0.08


def test_a_winner_that_kept_the_move_is_left_alone():
    import calendar
    import time

    opened = int(calendar.timegm(time.strptime("2026-09-22 09:00", "%Y-%m-%d %H:%M")))
    bars = [
        {"time": opened, "high": 0.7110, "low": 0.70440, "close": 0.70446},
    ]
    trades = [
        {
            "side": "SELL",
            "open_time": "2026-09-22 09:00",
            "entry": 0.71082,
            "pips": 63.6,
            "usd": 12723.0,
            "lots": 20.0,
            "reason": "trail stop",
        }
    ]
    out = prescribe_book(BookCfg(tp_atr=40.0, hold_risk=0.0, max_hold_bars=400.0), trades, bars, 100_000.0, 0.0001)
    assert out.tp_atr == 40.0
    assert out.max_hold_bars == 400.0
    assert out.side == 0.0
    assert out.max_trades == 0.0


def test_a_down_week_whose_reentry_lost_the_win_sells_once():
    import calendar
    import time

    t0 = int(calendar.timegm(time.strptime("2026-09-16 21:00", "%Y-%m-%d %H:%M")))
    bars = []
    px = 1.3400
    for i in range(40):
        px -= 0.0004
        bars.append({"time": t0 + i * 300, "high": px + 0.0002, "low": px - 0.0002, "close": px})
    opened = bars[10]["time"]
    trades = [
        {
            "side": "SELL",
            "open_time": "2026-09-16 21:50",
            "entry": 1.3360,
            "pips": 20.0,
            "usd": 4000.0,
            "reason": "trail stop",
        },
        {
            "side": "SELL",
            "open_time": "2026-09-16 23:00",
            "entry": 1.3300,
            "pips": -34.0,
            "usd": -6800.0,
            "reason": "ATR stop",
        },
    ]
    # open_time must match a stamp the MFE walk can parse; the first trade is the winner.
    trades[0]["open_time"] = time.strftime("%Y-%m-%d %H:%M", time.gmtime(opened))
    out = prescribe_book(BookCfg(), trades, bars, 100_000.0, 0.0001)
    assert out.side == -1.0
    assert out.max_trades == 1.0
    assert out.sl_atr >= 8.0


def test_a_win_without_a_pyramid_locks_the_rest_of_the_day():
    rails = BookRails()
    note = rails.on_close(1_638.0, 1.2, 10, snowball_n=0, day="2026-09-14", day_lock_usd=1000.0)
    assert rails.day_lock == "2026-09-14"
    assert rails.armed_protect is False
    assert "day lock" in note
    assert rails.blocked(11, "2026-09-14")
    assert rails.blocked(12, "2026-09-15") == ""


def test_a_small_win_does_not_lock_the_day():
    rails = BookRails()
    rails.on_close(168.0, 0.4, 10, snowball_n=0, day="2026-09-11", day_lock_usd=1000.0)
    assert rails.day_lock == ""
