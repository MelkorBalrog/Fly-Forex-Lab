"""Simulation bars accumulate: one file when the series continues, a new file on a gap."""

from flyfx.exec.history_bars import load_history, remember_bars


def _bar(stamp: int, close: float = 1.1) -> dict:
    return {"time": stamp, "open": close, "high": close, "low": close, "close": close}


def test_continuous_windows_share_one_file(monkeypatch, tmp_path):
    import flyfx.exec.history_bars as hb

    monkeypatch.setattr(hb, "HISTORY_DIR", tmp_path)
    remember_bars("GBPUSD", "M5", [_bar(t) for t in range(1_000, 2_000, 300)], quiet=True)
    remember_bars("GBPUSD", "M5", [_bar(t, 1.2) for t in range(1_600, 3_100, 300)], quiet=True)
    folder = tmp_path / "GBPUSD" / "M5"
    files = list(folder.glob("s*_e*.json"))
    assert len(files) == 1
    loaded = load_history("GBPUSD", "M5")
    assert loaded[0]["time"] == 1_000
    assert loaded[-1]["time"] == 2_800
    same = next(bar for bar in loaded if bar["time"] == 1_600)
    assert same["close"] == 1.2


def test_gap_opens_a_second_file(monkeypatch, tmp_path):
    import flyfx.exec.history_bars as hb

    monkeypatch.setattr(hb, "HISTORY_DIR", tmp_path)
    remember_bars("AUDUSD", "M5", [_bar(t) for t in range(1_000, 2_000, 300)], quiet=True)
    later = 1_000 + 4 * 86400
    remember_bars("AUDUSD", "M5", [_bar(t) for t in range(later, later + 900, 300)], quiet=True)
    folder = tmp_path / "AUDUSD" / "M5"
    assert len(list(folder.glob("s*_e*.json"))) == 2
    loaded = load_history("AUDUSD", "M5")
    assert loaded[0]["time"] == 1_000
    assert loaded[-1]["time"] == later + 600


def test_a_later_window_joins_files_it_connects(monkeypatch, tmp_path):
    import flyfx.exec.history_bars as hb

    monkeypatch.setattr(hb, "HISTORY_DIR", tmp_path)
    remember_bars("EURUSD", "M5", [_bar(t) for t in (60, 360, 660)], quiet=True)
    remember_bars("EURUSD", "M5", [_bar(t) for t in (10_000, 10_300, 10_600)], quiet=True)
    assert len(list((tmp_path / "EURUSD" / "M5").glob("s*_e*.json"))) == 2
    bridge = [_bar(t) for t in range(960, 10_300, 300)]
    remember_bars("EURUSD", "M5", bridge, quiet=True)
    assert len(list((tmp_path / "EURUSD" / "M5").glob("s*_e*.json"))) == 1
    loaded = load_history("EURUSD", "M5")
    assert loaded[0]["time"] == 60
    assert loaded[-1]["time"] == 10_600
