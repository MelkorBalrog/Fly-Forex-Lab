"""Higher-timeframe agreement for tech, flies, and the ANN feature vector."""

from flyfx.brain.tensor_brain import FEATURE_NAMES, N_FEATURES, pack_features
from flyfx.sense.mtf import mtf_allows, mtf_scale


def test_feature_vector_includes_four_timeframes():
    assert N_FEATURES == len(FEATURE_NAMES)
    assert FEATURE_NAMES[-4:] == ("mtf10", "mtf15", "mtf30", "mtf60")
    feat = {
        "impulse": 0.4,
        "rsi": 55,
        "close": 1.1,
        "mtf": {
            10: {"impulse": 0.6, "regime": "UP", "ready": True},
            15: {"impulse": 0.2, "regime": "DOWN", "ready": True},
            30: {"impulse": 0.0, "regime": "CHOP", "ready": False},
            60: {"impulse": 0.5, "regime": "UP", "ready": True},
        },
    }
    x = pack_features(feat, [1.1, 1.11])
    assert x.shape == (N_FEATURES,)
    assert x[-4] > 0
    assert x[-3] < 0
    assert x[-2] == 0


def test_tech_needs_higher_timeframe_majority():
    opposed = {
        "mtf": {
            10: {"regime": "DOWN", "ready": True},
            15: {"regime": "DOWN", "ready": True},
            30: {"regime": "UP", "ready": True},
        }
    }
    assert mtf_allows(opposed, "BUY") is False
    assert mtf_allows(opposed, "SELL") is True
    assert mtf_scale(opposed, "SELL") > mtf_scale(opposed, "BUY")
    assert mtf_allows({"mtf": {}}, "BUY") is True
