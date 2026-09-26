"""Per-pair book knobs and the pre-sim search."""

from flyfx.brain.book_cfg import MAX_TUNE_EVALS, BookCfg, BookTuneFleet, load_book, save_book, search_book
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
