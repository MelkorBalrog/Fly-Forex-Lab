"""Triangle nets: the 7% pairs, a learned trend sign, and the node mix."""

from flyfx.brain.triplet_ann import IMPLIED, TRIPLETS, TripletBook, apply_triplet


def test_seven_percent_pairs_have_a_triangle():
    assert set(TRIPLETS) == {"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD"}
    assert TRIPLETS["EURUSD"] == ("EURUSD", "AUDUSD", "EURAUD")
    assert TRIPLETS["GBPUSD"] == ("GBPUSD", "EURUSD", "EURGBP")
    assert TRIPLETS["AUDUSD"] == ("AUDUSD", "NZDUSD", "AUDNZD")
    assert TRIPLETS["NZDUSD"] == ("NZDUSD", "AUDUSD", "AUDNZD")
    assert TRIPLETS["USDCAD"] == ("USDCAD", "AUDUSD", "AUDCAD")
    assert IMPLIED["EURAUD"] == ("EURUSD", "AUDUSD", "div")
    assert IMPLIED["AUDCAD"] == ("AUDUSD", "USDCAD", "mul")


def _bars(start: float, drift: float, n: int = 180) -> list[dict]:
    px = start
    out = []
    for i in range(n):
        px = px * (1.0 + drift)
        out.append({"time": 1_700_000_000 + i * 300, "close": px, "open": px, "high": px, "low": px})
    return out


def test_rising_triangle_scores_up(monkeypatch, tmp_path):
    import flyfx.brain.triplet_ann as mod

    monkeypatch.setattr(mod, "BRAIN_DIR", tmp_path)
    book = TripletBook("EURUSD")
    book.persist = False
    book.attach("EURUSD", _bars(1.10, 0.0004))
    book.attach("AUDUSD", _bars(0.66, 0.0001))
    book.ensure_cross()
    assert "EURAUD" in book.series
    last = 0.0
    closes = book.series["EURUSD"][1]
    times = book.series["EURUSD"][0]
    for stamp, close in zip(times, closes):
        last = book.observe(int(stamp), float(close))
    assert book.n_updates > 24
    assert last > 0.15


def test_apply_triplet_reaches_every_category_and_the_impulse():
    cats = {
        name: {"available": True, "fused": 0.0, "impulse": 0.0, "P": 1.0, "residual": 0.0, "n_live": 4, "parts": {}}
        for name in ("trend", "momentum", "volatility", "volume", "breadth", "structure", "sentiment")
    }
    feat = {"impulse": 0.0, "fused": 0.0, "categories": cats, "kalman": {"impulse": 0.0, "fused": 0.0}}
    apply_triplet(feat, 0.0)
    assert feat["impulse"] == 0.0
    apply_triplet(feat, 0.8)
    assert all(st["parts"]["triplet"] == 0.8 for st in cats.values())
    assert feat["impulse"] > 0.05
    assert feat["kalman"]["triplet"] == 0.8
