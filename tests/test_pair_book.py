"""PairBook gates: same 2oo3, pair-local admission. No connectome."""

from types import SimpleNamespace

from flyfx.brain.fly_brains import is_frozen
from flyfx.risk.pair_book import PairBook, pair_book_enabled
from flyfx.trader import clip_bars, parse_args
from flyfx.train import _split_train


def test_eurusd_never_gets_pair_book():
    assert pair_book_enabled("EURUSD", frozen=True, no_pair_book=False, pair_book=True) is False
    assert pair_book_enabled("EURUSD", frozen=False, no_pair_book=False, pair_book=True) is False
    assert pair_book_enabled("GBPUSD", frozen=False, no_pair_book=False) is False
    assert pair_book_enabled("GBPUSD", frozen=False, no_pair_book=False, pair_book=True) is True
    assert pair_book_enabled("GBPUSD", frozen=False, no_pair_book=True, pair_book=True) is False
    frozen = PairBook("EURUSD", "hst EURUSD5.hst", enabled=True)
    assert frozen.enabled is False
    frozen.on_close("ATR stop", -200.0, "SELL", 40)
    assert frozen.skip_open("BUY", 41) == ""
    assert frozen.size_mult() == 1.0


def test_default_eurusd_path_does_not_enable_pair_book():
    """Trader default (no --pair-book) must not construct an enabled PairBook."""
    args = SimpleNamespace(symbol="EURUSD", pair_book=False, no_pair_book=False)
    loop_sym = str(args.symbol)
    on = pair_book_enabled(
        loop_sym,
        frozen=is_frozen(loop_sym),
        no_pair_book=bool(args.no_pair_book),
        pair_book=bool(args.pair_book),
    )
    book = PairBook(loop_sym, "hst EURUSD5.hst", enabled=on)
    assert on is False
    assert book.enabled is False
    gbp = pair_book_enabled("GBPUSD", frozen=False, no_pair_book=False, pair_book=False)
    assert gbp is False
    assert PairBook("GBPUSD", "yahoo 5m", enabled=gbp).enabled is False


def test_cli_default_does_not_enable_pair_book(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "argv", ["fly_forex.py", "--replay", "--hst", "auto"])
    args = parse_args()
    assert args.pair_book is False
    assert args.no_pair_book is False
    on = pair_book_enabled(
        "EURUSD",
        frozen=is_frozen("EURUSD"),
        no_pair_book=bool(args.no_pair_book),
        pair_book=bool(args.pair_book),
    )
    book = PairBook("EURUSD", "hst EURUSD5.hst", enabled=on)
    assert on is False
    assert book.enabled is False
    monkeypatch.setattr(sys, "argv", ["fly_forex.py", "--replay", "--pair-book"])
    flagged = parse_args()
    assert flagged.pair_book is True
    assert pair_book_enabled("GBPUSD", pair_book=flagged.pair_book) is True
    assert pair_book_enabled("EURUSD", pair_book=flagged.pair_book) is False
    monkeypatch.setattr(sys, "argv", ["fly_forex.py", "--replay", "--pair-book", "--no-pair-book"])
    both = parse_args()
    assert pair_book_enabled("GBPUSD", pair_book=both.pair_book, no_pair_book=both.no_pair_book) is False


def test_wide_ok_needs_flow_on_2oo3():
    book = PairBook("AUDUSD", "yahoo 5m AUDUSD=X", enabled=True)
    assert book.wide_ok("3oo3") is True
    assert book.wide_ok("3oo3+flow") is True
    assert book.wide_ok("2oo3+flow") is True
    assert book.wide_ok("2oo3") is False
    assert book.wide_ok("1oo3-bounce") is False


def test_yahoo_haircut_until_two_wins():
    book = PairBook("GBPUSD", "yahoo 5m GBPUSD=X", enabled=True)
    assert book.size_mult() == 0.50
    book.on_close("ATR target", 100.0, "BUY", 10)
    assert book.size_mult() == 0.50
    book.on_close("ATR target", 80.0, "BUY", 20)
    assert book.size_mult() == 1.0


def test_hst_full_size():
    book = PairBook("USDJPY", "hst USDJPY5.hst", enabled=True)
    assert book.size_mult() == 1.0


def test_atr_stop_halts_pair():
    book = PairBook("GBPUSD", "yahoo 5m", enabled=True)
    note = book.on_close("ATR stop", -200.0, "SELL", 40)
    assert "pair circuit" in note
    assert book.skip_open("BUY", 41)


def test_trail_stop_does_not_halt():
    book = PairBook("USDCHF", "yahoo 5m", enabled=True)
    book.on_close("trail stop", 200.0, "BUY", 10)
    assert book.skip_open("BUY", 11) == ""


def test_chop_scratch_skip_same_side():
    book = PairBook("EURGBP", "yahoo 5m", enabled=True)
    book.on_close("chop scratch", -50.0, "SELL", 100)
    assert "chop-scratch skip SELL" in book.skip_open("SELL", 105)
    assert book.skip_open("BUY", 105) == ""
    assert book.skip_open("SELL", 109) == ""


def test_two_chop_scratches_halt():
    book = PairBook("EURGBP", "yahoo 5m", enabled=True)
    book.on_close("chop scratch", -50.0, "SELL", 10)
    book.on_close("chop scratch", -40.0, "BUY", 30)
    assert "2 chop scratches" in book.skip_open("SELL", 40)


def test_clip_bars_strict_does_not_leak():
    bars = [{"time": 100 + i, "close": 1.0} for i in range(200)]
    out = clip_bars(bars, 1000, 1100, strict=True)
    assert out == []
    leaked = clip_bars(bars, 1000, 1100, strict=False)
    assert leaked == bars


def test_split_train_refuses_eval_leak():
    bars = [{"time": 1_000, "close": 1.0}] * 30
    bars += [{"time": 2_000, "close": 1.0}] * 30
    assert _split_train(bars, t0=1_500, weekly=False) == []
