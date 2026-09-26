"""CSV / JSON / HST bar import and timeframe resampling."""

from __future__ import annotations

from pathlib import Path

from flyfx.exec.bar_io import (
    infer_period_minutes,
    load_bars_file,
    parse_ohlc_text,
    parse_timeframe,
    resample_bars,
)
from flyfx.trader import bars_for_training, clamp_balance, merge_bars, parse_args


def test_parse_timeframe_aliases():
    assert parse_timeframe("10min") == ("M10", 10)
    assert parse_timeframe("1h") == ("H1", 60)
    assert parse_timeframe("M5") == ("M5", 5)
    assert parse_timeframe("native") == ("NATIVE", 0)


def test_csv_header_and_mt4_date_time():
    csv_txt = (
        "datetime,open,high,low,close,volume\n"
        "2024-01-02 00:00:00,1.1000,1.1010,1.0990,1.1005,100\n"
        "2024-01-02 00:10:00,1.1005,1.1020,1.1000,1.1015,110\n"
        "2024-01-02 00:20:00,1.1015,1.1030,1.1010,1.1020,120\n"
        "2024-01-02 00:30:00,1.1020,1.1040,1.1015,1.1030,130\n"
        "2024-01-02 00:40:00,1.1030,1.1050,1.1025,1.1040,140\n"
        "2024-01-02 00:50:00,1.1040,1.1060,1.1035,1.1050,150\n"
    )
    bars = parse_ohlc_text(csv_txt, "m10.csv")
    assert len(bars) == 6
    assert bars[0]["open"] == 1.1
    assert infer_period_minutes(bars) == 10

    mt4 = (
        "2016.01.04,00:00,1.08701,1.08713,1.08621,1.08699,10\n"
        "2016.01.04,00:05,1.08699,1.08720,1.08680,1.08710,11\n"
        "2016.01.04,00:10,1.08710,1.08740,1.08700,1.08730,12\n"
        "2016.01.04,00:15,1.08730,1.08750,1.08720,1.08740,13\n"
        "2016.01.04,00:20,1.08740,1.08760,1.08730,1.08755,14\n"
        "2016.01.04,00:25,1.08755,1.08780,1.08750,1.08770,15\n"
    )
    m5 = parse_ohlc_text(mt4, "EURUSD5.csv")
    assert len(m5) == 6
    assert infer_period_minutes(m5) == 5
    h1 = resample_bars(m5, 60)
    assert len(h1) == 1
    assert h1[0]["open"] == 1.08701
    assert h1[0]["close"] == 1.08770
    assert h1[0]["high"] == 1.08780
    assert h1[0]["low"] == 1.08621


def test_json_bars_and_file_roundtrip(tmp_path: Path):
    payload = [
        {"time": 1700000000 + i * 3600, "open": 1.08 + i * 0.0001, "high": 1.081, "low": 1.079, "close": 1.0805}
        for i in range(8)
    ]
    import json

    p = tmp_path / "EURUSD_H1.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    bars, source = load_bars_file(p, timeframe="NATIVE")
    assert "file" in source
    assert len(bars) == 8
    assert infer_period_minutes(bars) == 60


def test_clamp_balance_and_capital_flag(monkeypatch):
    assert clamp_balance(2500) == 2500
    assert clamp_balance(1) == 100
    assert clamp_balance(1e12) == 10_000_000
    monkeypatch.setattr("sys.argv", ["fly_forex.py", "--replay", "--capital", "25000", "--data", "x.csv"])
    args = parse_args()
    assert args.balance == 25000
    assert args.data == "x.csv"
    assert args.timeframe == "M5"


def test_training_corpus_is_history_plus_current(monkeypatch, tmp_path):
    import flyfx.exec.history_bars as hb

    monkeypatch.setattr(hb, "HISTORY_DIR", tmp_path)
    history = [
        {"time": 100, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0},
        {"time": 200, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.1},
        {"time": 300, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.2},
    ]
    current = [
        {"time": 300, "open": 1.5, "high": 1.6, "low": 1.4, "close": 1.55},
        {"time": 400, "open": 1.5, "high": 1.6, "low": 1.4, "close": 1.6},
    ]
    hb.remember_bars("EURUSD", "M5", history, quiet=True)
    merged, note = bars_for_training("EURUSD", "M5", current, "yahoo 5m EURUSD=X")
    assert [b["time"] for b in merged] == [100, 200, 300, 400]
    assert merged[2]["close"] == 1.55
    assert "n=2" in note and "train n=4" in note
    assert merge_bars(current, history)[2]["close"] == 1.2
    monkeypatch.setattr(hb, "HISTORY_DIR", tmp_path / "empty")
    only, _note = bars_for_training("EURUSD", "M5", current, "yahoo 5m EURUSD=X")
    assert [b["time"] for b in only] == [300, 400]
