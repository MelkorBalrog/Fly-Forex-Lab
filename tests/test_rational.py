"""Bull and bear decisions need diverse indicator support."""

from flyfx.vote.rational import assess


def _feat(**over) -> dict:
    base = {
        "close": 1.1050,
        "atr": 0.0010,
        "bars_seen": 80,
        "ema_fast": 1.1040,
        "ema_slow": 1.1020,
        "adx": 28.0,
        "plus_di": 32.0,
        "minus_di": 14.0,
        "sar": 1.1010,
        "st_dir": 1,
        "ichi": {"cloud": 1.0},
        "macd_hist": 0.20,
        "rsi": 62.0,
        "stoch": 0.70,
        "willr": -30.0,
        "cci": 80.0,
        "bb_pct": 0.70,
    }
    base.update(over)
    return base


def _mirror(feat: dict) -> dict:
    """Same tape, reflected through 1.10, so a sell is the bull case upside down."""
    out = dict(feat)
    out["close"] = 1.10 - (feat["close"] - 1.10)
    out["ema_fast"] = 1.10 - (feat["ema_fast"] - 1.10)
    out["ema_slow"] = 1.10 - (feat["ema_slow"] - 1.10)
    out["sar"] = 1.10 - (feat["sar"] - 1.10)
    out["st_dir"] = -int(feat["st_dir"])
    out["ichi"] = {"cloud": -float(feat["ichi"]["cloud"])}
    out["plus_di"] = feat["minus_di"]
    out["minus_di"] = feat["plus_di"]
    out["macd_hist"] = -feat["macd_hist"]
    out["rsi"] = 100.0 - feat["rsi"]
    out["stoch"] = 1.0 - feat["stoch"]
    out["willr"] = -100.0 - feat["willr"]
    out["cci"] = -feat["cci"]
    out["bb_pct"] = 1.0 - feat["bb_pct"]
    return out


def test_a_clean_bull_tape_is_full_size():
    v = assess("BUY", _feat())
    assert v.ok
    assert v.mult == 1.0
    assert v.trend_for >= 3
    assert v.mom_for >= 2
    assert v.against == 0


def test_the_same_tape_reflected_is_a_full_size_bear():
    bull = assess("BUY", _feat())
    bear = assess("SELL", _mirror(_feat()))
    assert bear.ok
    assert bear.mult == bull.mult
    assert bear.trend_for == bull.trend_for
    assert bear.mom_for == bull.mom_for


def test_a_bull_call_against_the_tape_is_refused():
    feat = _feat(
        ema_fast=1.0980,
        ema_slow=1.1020,
        plus_di=12.0,
        minus_di=30.0,
        sar=1.1080,
        st_dir=-1,
        ichi={"cloud": -1.0},
        macd_hist=-0.30,
        rsi=32.0,
        stoch=0.20,
        willr=-80.0,
        cci=-90.0,
        bb_pct=0.25,
    )
    v = assess("BUY", feat)
    assert not v.ok
    assert "implausible" in v.why


def test_a_pinned_extreme_is_irrational_even_when_trend_agrees():
    feat = _feat(rsi=82.0, stoch=0.93, bb_pct=0.96)
    v = assess("BUY", feat)
    assert not v.ok
    assert v.why.startswith("irrational")
    sell = assess("SELL", _mirror(feat))
    assert not sell.ok
    assert sell.why.startswith("irrational")


def test_one_dissent_still_trades_smaller():
    feat = _feat(macd_hist=-0.20)
    v = assess("BUY", feat)
    assert v.ok
    assert v.against == 1
    assert v.mult == 0.70


def test_a_quiet_oscillator_abstains_and_does_not_invent_a_side():
    feat = _feat(stoch=0.50, willr=-50.0, cci=0.0, rsi=50.0)
    v = assess("BUY", feat)
    assert v.ok
    assert v.mom_for == 1
    assert v.mult == 0.85


def test_jpy_gaps_in_atr_match_the_eur_gaps():
    eur = assess("BUY", _feat())
    jpy = assess(
        "BUY",
        _feat(
            close=150.50,
            atr=0.10,
            ema_fast=150.40,
            ema_slow=150.20,
            sar=150.10,
        ),
    )
    assert jpy.ok == eur.ok
    assert jpy.mult == eur.mult
    assert jpy.trend_for == eur.trend_for
