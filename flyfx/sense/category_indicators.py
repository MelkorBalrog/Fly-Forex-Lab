"""OHLCV indicator suite grouped by the seven Kalman categories.

Live vs unavailable (single-pair bars; do not invent market-breadth or
options/futures overlays):

  LIVE from OHLC
    Trend: SMA, EMA, WMA, MACD, ADX, Parabolic SAR, Ichimoku (unshifted
      Tenkan/Kijun/cloud), Supertrend
    Momentum: RSI, Stochastic, CCI, Williams %R, ROC
    Volatility: Bollinger, ATR, Keltner, StdDev, Donchian
    Structure: session/daily pivots, Fibonacci swing retracement,
      regression channel

  LIVE only when bar tick/real volume is > 0
    Volume: OBV, Chaikin Money Flow, VWAP, Accumulation/Distribution
    Volume Profile: UNAVAILABLE (needs a tick-volume histogram, not bars)

  UNAVAILABLE unless the bar/extras dict already carries them
    Breadth: Advance-Decline, McClellan Oscillator, Arms Index (TRIN)
    Sentiment: Put/Call, VIX, COT net

Missing values stay None so the category Kalman can skip that channel.
"""

from __future__ import annotations

import math
import time
from collections import deque

import numpy as np

from flyfx.sense.fib_levels import held_level, impulse_swing, retracement_prices
from flyfx.sense.kalman_signal import is_measurement

KEEP = 160
WARM = 55


def _ema(prev: float | None, price: float, period: int) -> float:
    k = 2.0 / (period + 1.0)
    return price if prev is None else prev + k * (price - prev)


def _finite(x: object) -> float | None:
    if not is_measurement(x):
        return None
    return float(x)


def _clip(z: float | None, lo: float = -4.0, hi: float = 4.0) -> float | None:
    if z is None:
        return None
    return float(np.clip(z, lo, hi))


def _sma(xs: list[float], n: int) -> float | None:
    if len(xs) < n or n <= 0:
        return None
    return float(sum(xs[-n:]) / n)


def _wma(xs: list[float], n: int) -> float | None:
    if len(xs) < n or n <= 0:
        return None
    wsum = n * (n + 1) / 2.0
    total = 0.0
    for i, v in enumerate(xs[-n:], start=1):
        total += i * v
    return total / wsum


def _stdev(xs: list[float], n: int) -> float | None:
    if len(xs) < n or n <= 2:
        return None
    window = xs[-n:]
    mean = sum(window) / n
    var = sum((x - mean) ** 2 for x in window) / n
    return math.sqrt(max(var, 0.0))


def _opt(src: dict | None, *keys: str) -> float | None:
    if not src:
        return None
    for key in keys:
        if key in src and is_measurement(src.get(key)):
            v = float(src[key])
            if v == 0.0 and key in ("volume", "tick_volume", "vol"):
                continue
            return v
    return None


class CategoryIndicatorEngine:
    """Rolling OHLCV → raw values + Kalman-ready normalized measurements."""

    def __init__(self, pip: float = 0.0001):
        self.pip = max(float(pip), 1e-12)
        self.opens: list[float] = []
        self.highs: list[float] = []
        self.lows: list[float] = []
        self.closes: list[float] = []
        self.volumes: list[float | None] = []
        self.times: list[int] = []
        self.n = 0

        self.ema_fast: float | None = None
        self.ema_slow: float | None = None
        self.ema_day: float | None = None
        self.ema12: float | None = None
        self.ema26: float | None = None
        self.macd_sig: float | None = None
        self.ema20: float | None = None
        self.atr: float | None = None
        self.atr_slow: float | None = None
        self.rsi = 50.0
        self.avg_gain: float | None = None
        self.avg_loss: float | None = None
        self.prev_close: float | None = None
        self.prev_high: float | None = None
        self.prev_low: float | None = None

        self._tr14 = 0.0
        self._pdm14 = 0.0
        self._mdm14 = 0.0
        self._dx14: float | None = None
        self.adx = 20.0
        self.plus_di = 0.0
        self.minus_di = 0.0
        self._adx_bars = 0

        self.sar: float | None = None
        self._sar_up = True
        self._sar_ep = 0.0
        self._sar_af = 0.02

        self.st_dir = 1
        self.st_upper: float | None = None
        self.st_lower: float | None = None
        self.supertrend: float | None = None

        self.obv = 0.0
        self.ad_line = 0.0
        self._vwap_num = 0.0
        self._vwap_den = 0.0
        self._vwap_day: int | None = None
        self._mfv: deque[float] = deque(maxlen=21)
        self._mvol: deque[float] = deque(maxlen=21)
        self._obv_hist: deque[float] = deque(maxlen=16)
        self._ad_hist: deque[float] = deque(maxlen=16)
        self._bb_bw_hist: deque[float] = deque(maxlen=40)

        self.bb_mid = 0.0
        self.bb_pct = 0.50
        self.bb_bw = 0.0

    def update(
        self,
        high: float,
        low: float,
        close: float,
        volume: float | None = None,
        open_: float | None = None,
        extras: dict | None = None,
        stamp: int | None = None,
    ) -> dict:
        self.n += 1
        o = float(open_) if is_measurement(open_) else (self.closes[-1] if self.closes else float(close))
        h, l, c = float(high), float(low), float(close)
        vol = _finite(volume)
        if vol is not None and vol <= 0.0:
            vol = None
        ts = int(stamp) if stamp else 0

        self.opens.append(o)
        self.highs.append(h)
        self.lows.append(l)
        self.closes.append(c)
        self.volumes.append(vol)
        self.times.append(ts)
        if len(self.closes) > KEEP:
            self.opens.pop(0)
            self.highs.pop(0)
            self.lows.pop(0)
            self.closes.pop(0)
            self.volumes.pop(0)
            self.times.pop(0)

        atr = self._update_atr_rsi(h, l, c)
        self._update_mas(c, atr)
        self._update_adx(h, l)
        self._update_sar(h, l, c)
        self._update_supertrend(h, l, c, atr)
        self._update_volume(h, l, c, vol, ts)
        bb = self._bollinger(c)
        kel = self._keltner(c, atr)
        don = self._donchian(c)
        macd = self._macd(atr)
        stoch, willr = self._stoch_willr(c)
        cci = self._cci()
        roc = self._roc(c, atr)
        mom = 0.0
        if len(self.closes) >= 6:
            mom = (self.closes[-1] - self.closes[-6]) / atr

        sma_f, sma_s = _sma(self.closes, 21), _sma(self.closes, 55)
        wma_f, wma_s = _wma(self.closes, 21), _wma(self.closes, 55)
        sd20 = _stdev(self.closes, 20)
        ichi = self._ichimoku(c, atr)
        piv = self._pivots(c, atr, ts)
        fib = self._fibonacci(h, l, c, atr)
        chan = self._channel(c, atr)
        flow = self._volume_flow(c, atr)
        breadth = self._breadth(extras)
        sentiment = self._sentiment(extras)

        sep = 0.0
        if self.ema_fast is not None and self.ema_slow is not None:
            sep = abs(self.ema_fast - self.ema_slow) / atr
        regime = "CHOP"
        if self.ema_fast is not None and self.ema_slow is not None:
            if self.ema_fast > self.ema_slow and c > self.ema_slow and sep > 0.18:
                regime = "UP"
            elif self.ema_fast < self.ema_slow and c < self.ema_slow and sep > 0.18:
                regime = "DOWN"

        bar_rng = max(h - l, self.pip)
        bar_pos = (c - l) / bar_rng
        prev = self.closes[-2] if len(self.closes) >= 2 else c
        signed_vol = (c - prev) / atr
        vol_rel = 1.0
        if self.atr and self.atr > 0:
            tr = max(h - l, abs(h - (self.prev_close or c)), abs(l - (self.prev_close or c)))
            vol_rel = float(tr / atr)

        atr_ratio = None
        if self.atr_slow and self.atr_slow > 0:
            atr_ratio = _clip(self.atr / self.atr_slow - 1.0, -2.0, 4.0)

        measurements = {
            "trend": {
                "sma": _clip(((sma_f - sma_s) / atr) if sma_f is not None and sma_s is not None else None),
                "ema": _clip(((self.ema_fast - self.ema_slow) / atr) if self.ema_fast is not None and self.ema_slow is not None else None),
                "wma": _clip(((wma_f - wma_s) / atr) if wma_f is not None and wma_s is not None else None),
                "macd": macd["z"],
                "adx": self._adx_z(),
                "sar": _clip(((c - self.sar) / atr) if self.sar is not None else None),
                "ichimoku": ichi["z"],
                "supertrend": _clip(((c - self.supertrend) / atr) * self.st_dir if self.supertrend is not None else None),
            },
            "momentum": {
                "rsi": _clip((self.rsi - 50.0) / 12.5),
                "stoch": _clip((stoch - 0.50) / 0.22) if stoch is not None else None,
                "cci": _clip((cci / 100.0) if cci is not None else None),
                "willr": _clip(((willr + 50.0) / 12.5) if willr is not None else None),
                "roc": roc,
            },
            "volatility": {
                "bb": _clip((bb["pct"] - 0.50) * 4.0) if bb["pct"] is not None else None,
                "atr": atr_ratio,
                "keltner": kel["z"],
                "stdev": _clip((sd20 / atr - 1.0) if sd20 is not None else None, -2.0, 4.0),
                "donchian": don["z"],
            },
            "volume": {
                "obv": flow["obv_z"],
                "cmf": flow["cmf_z"],
                "vwap": flow["vwap_z"],
                "ad": flow["ad_z"],
                "profile": None,  # no tick-volume histogram on bar data
            },
            "breadth": {
                "ad_line": breadth["ad_z"],
                "mcclellan": breadth["mcc_z"],
                "trin": breadth["trin_z"],
            },
            "structure": {
                "pivot": piv["z"],
                "fib": fib["z"],
                "channel": chan["z"],
            },
            "sentiment": {
                "put_call": sentiment["pc_z"],
                "vix": sentiment["vix_z"],
                "cot": sentiment["cot_z"],
            },
        }

        self.prev_close = c
        self.prev_high = h
        self.prev_low = l

        return {
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": vol,
            "ema_fast": self.ema_fast if self.ema_fast is not None else c,
            "ema_slow": self.ema_slow if self.ema_slow is not None else c,
            "ema_day": self.ema_day if self.ema_day is not None else c,
            "bars_seen": int(self.n),
            "rsi": self.rsi,
            "atr": atr,
            "sep": sep,
            "regime": regime,
            "ready": self.n >= WARM,
            "mom": mom,
            "macd": macd["line"],
            "macd_sig": macd["signal"],
            "macd_hist": macd["hist"],
            "stoch": stoch if stoch is not None else 0.50,
            "willr": willr if willr is not None else -50.0,
            "cci": cci if cci is not None else 0.0,
            "roc": roc if roc is not None else 0.0,
            "adx": self.adx,
            "plus_di": self.plus_di,
            "minus_di": self.minus_di,
            "sar": self.sar if self.sar is not None else c,
            "supertrend": self.supertrend if self.supertrend is not None else c,
            "st_dir": int(self.st_dir),
            "bb_pct": bb["pct"] if bb["pct"] is not None else 0.50,
            "bb_bw": bb["bw"] if bb["bw"] is not None else 0.0,
            "bb_bw_ma": float(sum(self._bb_bw_hist) / max(len(self._bb_bw_hist), 1)),
            "bb_mid": self.bb_mid,
            "bb_up": bb["up"],
            "bb_dn": bb["dn"],
            "keltner_pct": kel["pct"] if kel["pct"] is not None else 0.50,
            "donchian_pct": don["pct"] if don["pct"] is not None else 0.50,
            "don_hi": don["hi"],
            "don_lo": don["lo"],
            "atr_ratio": float(self.atr / self.atr_slow) if self.atr and self.atr_slow else 1.0,
            "bar_pos": bar_pos,
            "bar_high": h,
            "bar_low": l,
            "vol_rel": vol_rel,
            "signed_vol": float(np.clip(signed_vol * vol_rel, -4.0, 4.0)),
            "volume_live": vol is not None,
            "obv": self.obv,
            "cmf": flow["cmf"],
            "vwap": flow["vwap"],
            "ad_line": self.ad_line,
            "pivot": piv,
            "fib": fib,
            "channel": chan,
            "ichi": ichi,
            "sma_fast": sma_f,
            "sma_slow": sma_s,
            "wma_fast": wma_f,
            "wma_slow": wma_s,
            "sd20": sd20,
            "breadth_live": any(v is not None for v in breadth.values()),
            "sentiment_live": any(v is not None for v in sentiment.values()),
            "measurements": measurements,
        }

    def _update_atr_rsi(self, h: float, l: float, c: float) -> float:
        if self.prev_close is None:
            tr = max(h - l, self.pip)
        else:
            tr = max(h - l, abs(h - self.prev_close), abs(l - self.prev_close))
            change = c - self.prev_close
            gain, loss = max(change, 0.0), max(-change, 0.0)
            if self.avg_gain is None:
                self.avg_gain, self.avg_loss = gain, loss
            else:
                self.avg_gain = (self.avg_gain * 13.0 + gain) / 14.0
                self.avg_loss = (self.avg_loss * 13.0 + loss) / 14.0
            if self.avg_loss and self.avg_loss > 1e-12:
                rs = self.avg_gain / self.avg_loss
                self.rsi = 100.0 - 100.0 / (1.0 + rs)
            elif self.avg_gain and self.avg_gain > 0:
                self.rsi = 100.0
        if self.atr is None:
            self.atr = max(tr, self.pip)
        else:
            self.atr = (self.atr * 13.0 + tr) / 14.0
        if self.atr_slow is None:
            self.atr_slow = self.atr
        else:
            self.atr_slow = (self.atr_slow * 39.0 + tr) / 40.0
        return max(self.atr, self.pip)

    def _update_mas(self, c: float, atr: float) -> None:
        self.ema_fast = _ema(self.ema_fast, c, 21)
        self.ema_slow = _ema(self.ema_slow, c, 55)
        self.ema_day = _ema(self.ema_day, c, 288)
        self.ema12 = _ema(self.ema12, c, 12)
        self.ema26 = _ema(self.ema26, c, 26)
        self.ema20 = _ema(self.ema20, c, 20)

    def _macd(self, atr: float) -> dict:
        line = 0.0
        if self.ema12 is not None and self.ema26 is not None:
            line = (self.ema12 - self.ema26) / atr
        self.macd_sig = _ema(self.macd_sig, line, 9)
        sig = float(self.macd_sig if self.macd_sig is not None else 0.0)
        hist = line - sig
        return {"line": line, "signal": sig, "hist": hist, "z": _clip(hist)}

    def _update_adx(self, h: float, l: float) -> None:
        if self.prev_high is None or self.prev_low is None or self.prev_close is None:
            return
        up_move = h - self.prev_high
        dn_move = self.prev_low - l
        pdm = up_move if up_move > dn_move and up_move > 0 else 0.0
        mdm = dn_move if dn_move > up_move and dn_move > 0 else 0.0
        tr = max(h - l, abs(h - self.prev_close), abs(l - self.prev_close), self.pip)
        n = 14.0
        if self._adx_bars < 14:
            self._tr14 += tr
            self._pdm14 += pdm
            self._mdm14 += mdm
            self._adx_bars += 1
            if self._adx_bars < 14:
                return
        else:
            self._tr14 = self._tr14 - self._tr14 / n + tr
            self._pdm14 = self._pdm14 - self._pdm14 / n + pdm
            self._mdm14 = self._mdm14 - self._mdm14 / n + mdm
        tr14 = max(self._tr14, self.pip)
        self.plus_di = 100.0 * self._pdm14 / tr14
        self.minus_di = 100.0 * self._mdm14 / tr14
        denom = max(self.plus_di + self.minus_di, 1e-9)
        dx = 100.0 * abs(self.plus_di - self.minus_di) / denom
        if self._dx14 is None:
            self._dx14 = dx
            self.adx = dx
        else:
            self._dx14 = (self._dx14 * 13.0 + dx) / 14.0
            self.adx = self._dx14

    def _adx_z(self) -> float | None:
        if self._adx_bars < 14:
            return None
        strength = float(np.clip((self.adx - 18.0) / 22.0, 0.0, 2.5))
        sign = 1.0 if self.plus_di >= self.minus_di else -1.0
        return _clip(sign * strength)

    def _update_sar(self, h: float, l: float, c: float) -> None:
        if self.sar is None:
            self.sar = l
            self._sar_ep = h
            self._sar_up = True
            self._sar_af = 0.02
            return
        sar = self.sar
        if self._sar_up:
            sar = sar + self._sar_af * (self._sar_ep - sar)
            if self.prev_low is not None:
                sar = min(sar, self.prev_low, l)
            if l < sar:
                self._sar_up = False
                sar = self._sar_ep
                self._sar_ep = l
                self._sar_af = 0.02
            else:
                if h > self._sar_ep:
                    self._sar_ep = h
                    self._sar_af = min(self._sar_af + 0.02, 0.20)
        else:
            sar = sar + self._sar_af * (self._sar_ep - sar)
            if self.prev_high is not None:
                sar = max(sar, self.prev_high, h)
            if h > sar:
                self._sar_up = True
                sar = self._sar_ep
                self._sar_ep = h
                self._sar_af = 0.02
            else:
                if l < self._sar_ep:
                    self._sar_ep = l
                    self._sar_af = min(self._sar_af + 0.02, 0.20)
        self.sar = sar

    def _update_supertrend(self, h: float, l: float, c: float, atr: float) -> None:
        mid = 0.5 * (h + l)
        upper = mid + 3.0 * atr
        lower = mid - 3.0 * atr
        if self.st_upper is None or self.st_lower is None:
            self.st_upper, self.st_lower = upper, lower
            self.supertrend = lower
            self.st_dir = 1
            return
        if lower > self.st_lower:
            self.st_lower = lower
        else:
            self.st_lower = lower if c < self.st_lower else self.st_lower
        if upper < self.st_upper:
            self.st_upper = upper
        else:
            self.st_upper = upper if c > self.st_upper else self.st_upper
        if self.st_dir >= 0:
            if c < self.st_lower:
                self.st_dir = -1
                self.supertrend = self.st_upper
            else:
                self.supertrend = self.st_lower
        else:
            if c > self.st_upper:
                self.st_dir = 1
                self.supertrend = self.st_lower
            else:
                self.supertrend = self.st_upper

    def _ichimoku(self, c: float, atr: float) -> dict:
        def mid(n: int) -> float | None:
            if len(self.highs) < n:
                return None
            return 0.5 * (max(self.highs[-n:]) + min(self.lows[-n:]))

        tenkan, kijun, span_b = mid(9), mid(26), mid(52)
        span_a = None
        if tenkan is not None and kijun is not None:
            span_a = 0.5 * (tenkan + kijun)
        z = None
        cloud = 0.0
        if kijun is not None:
            z = (c - kijun) / atr
            if span_a is not None and span_b is not None:
                top, bot = max(span_a, span_b), min(span_a, span_b)
                if c > top:
                    cloud = 1.0
                elif c < bot:
                    cloud = -1.0
                z = 0.65 * z + 0.35 * cloud
        return {
            "tenkan": tenkan,
            "kijun": kijun,
            "span_a": span_a,
            "span_b": span_b,
            "cloud": cloud,
            "z": _clip(z),
        }

    def _bollinger(self, c: float) -> dict:
        mid = _sma(self.closes, 20)
        sd = _stdev(self.closes, 20)
        if mid is None or sd is None:
            return {"pct": None, "bw": None, "up": None, "dn": None}
        up, dn = mid + 2.0 * sd, mid - 2.0 * sd
        span = max(up - dn, self.pip)
        pct = (c - dn) / span
        bw = span / max(abs(mid), self.pip)
        self.bb_mid, self.bb_pct, self.bb_bw = mid, pct, bw
        self._bb_bw_hist.append(bw)
        return {"pct": pct, "bw": bw, "up": up, "dn": dn}

    def _keltner(self, c: float, atr: float) -> dict:
        mid = self.ema20
        if mid is None:
            return {"pct": None, "z": None}
        up, dn = mid + 1.5 * atr, mid - 1.5 * atr
        span = max(up - dn, self.pip)
        pct = (c - dn) / span
        return {"pct": pct, "z": _clip((pct - 0.50) * 4.0)}

    def _donchian(self, c: float) -> dict:
        if len(self.highs) < 20:
            return {"pct": None, "z": None, "hi": None, "lo": None}
        hi, lo = max(self.highs[-20:]), min(self.lows[-20:])
        span = max(hi - lo, self.pip)
        pct = (c - lo) / span
        return {"pct": pct, "z": _clip((pct - 0.50) * 4.0), "hi": hi, "lo": lo}

    def _stoch_willr(self, c: float) -> tuple[float | None, float | None]:
        if len(self.highs) < 14:
            return None, None
        hh, ll = max(self.highs[-14:]), min(self.lows[-14:])
        span = max(hh - ll, self.pip)
        stoch = (c - ll) / span
        willr = -100.0 * (hh - c) / span
        return stoch, willr

    def _cci(self) -> float | None:
        n = 20
        if len(self.closes) < n:
            return None
        tps = [
            (self.highs[-n + i] + self.lows[-n + i] + self.closes[-n + i]) / 3.0
            for i in range(n)
        ]
        sma = sum(tps) / n
        mad = sum(abs(x - sma) for x in tps) / n
        if mad <= 1e-12:
            return 0.0
        return (tps[-1] - sma) / (0.015 * mad)

    def _roc(self, c: float, atr: float) -> float | None:
        if len(self.closes) < 11:
            return None
        prev = self.closes[-11]
        if abs(prev) <= 1e-12:
            return None
        raw = (c - prev) / prev
        scale = max(atr / max(abs(c), self.pip) * 10.0, 1e-6)
        return _clip(raw / scale)

    def _update_volume(self, h: float, l: float, c: float, vol: float | None, ts: int) -> None:
        if vol is None:
            return
        prev = self.prev_close if self.prev_close is not None else c
        if c > prev:
            self.obv += vol
        elif c < prev:
            self.obv -= vol
        rng = max(h - l, self.pip)
        clv = ((c - l) - (h - c)) / rng
        self.ad_line += clv * vol
        self._obv_hist.append(self.obv)
        self._ad_hist.append(self.ad_line)
        mfv = clv * vol
        self._mfv.append(mfv)
        self._mvol.append(vol)
        day = time.gmtime(ts).tm_yday if ts else None
        if day is not None and day != self._vwap_day:
            self._vwap_num = 0.0
            self._vwap_den = 0.0
            self._vwap_day = day
        typical = (h + l + c) / 3.0
        self._vwap_num += typical * vol
        self._vwap_den += vol

    def _volume_flow(self, c: float, atr: float) -> dict:
        out = {
            "obv_z": None,
            "cmf_z": None,
            "vwap_z": None,
            "ad_z": None,
            "cmf": 0.0,
            "vwap": c,
        }
        if len(self._obv_hist) >= 6:
            slope = (self._obv_hist[-1] - self._obv_hist[-6]) / 5.0
            mean_v = 0.0
            live = [v for v in self.volumes[-20:] if v]
            if live:
                mean_v = sum(live) / len(live)
            out["obv_z"] = _clip(slope / max(mean_v, 1e-9))
        if self._mvol and sum(self._mvol) > 0:
            cmf = sum(self._mfv) / sum(self._mvol)
            out["cmf"] = cmf
            out["cmf_z"] = _clip(cmf / 0.25)
        if self._vwap_den > 0:
            vwap = self._vwap_num / self._vwap_den
            out["vwap"] = vwap
            out["vwap_z"] = _clip((c - vwap) / atr)
        if len(self._ad_hist) >= 6:
            slope = (self._ad_hist[-1] - self._ad_hist[-6]) / 5.0
            live = [v for v in self.volumes[-20:] if v]
            mean_v = sum(live) / len(live) if live else 1.0
            out["ad_z"] = _clip(slope / max(mean_v, 1e-9))
        return out

    def _session_hlc(self, ts: int) -> tuple[float, float, float] | None:
        if len(self.closes) < 20:
            return None
        if ts:
            day = time.gmtime(ts).tm_yday
            year = time.gmtime(ts).tm_year
            prev: list[int] = []
            for i, t in enumerate(self.times[:-1]):
                if not t:
                    continue
                g = time.gmtime(t)
                if g.tm_year < year or (g.tm_year == year and g.tm_yday < day):
                    prev.append(i)
            if prev:
                last_day = time.gmtime(self.times[prev[-1]]).tm_yday
                idxs = [i for i in prev if time.gmtime(self.times[i]).tm_yday == last_day]
                if idxs:
                    return (
                        max(self.highs[i] for i in idxs),
                        min(self.lows[i] for i in idxs),
                        self.closes[idxs[-1]],
                    )
        n = min(288, max(len(self.closes) - 1, 20))
        return max(self.highs[-n - 1 : -1]), min(self.lows[-n - 1 : -1]), self.closes[-2]

    def _pivots(self, c: float, atr: float, ts: int) -> dict:
        hlc = self._session_hlc(ts)
        if hlc is None:
            return {"p": None, "r1": None, "s1": None, "r2": None, "s2": None, "z": None, "near": ""}
        ph, pl, pc = hlc
        p = (ph + pl + pc) / 3.0
        r1 = 2.0 * p - pl
        s1 = 2.0 * p - ph
        r2 = p + (ph - pl)
        s2 = p - (ph - pl)
        levels = {"p": p, "r1": r1, "s1": s1, "r2": r2, "s2": s2}
        nearest, dist = "p", abs(c - p)
        for name, lvl in levels.items():
            d = abs(c - lvl)
            if d < dist:
                nearest, dist = name, d
        # signed: above pivot = bull structure, extra pull toward support
        z = (c - p) / atr
        if nearest.startswith("s"):
            z = z - 0.35 * (dist / atr)
        elif nearest.startswith("r"):
            z = z + 0.35 * (dist / atr)
        levels.update({"z": _clip(z), "near": nearest, "dist_atr": dist / atr})
        return levels

    def _fibonacci(self, h: float, l: float, c: float, atr: float) -> dict:
        n = min(55, len(self.highs))
        if n < 20:
            return {"z": None, "pos": None, "up": True, "near": "", "levels": {}}
        hi, lo = max(self.highs[-n:]), min(self.lows[-n:])
        span = max(hi - lo, self.pip)
        pos = (c - lo) / span
        up = self.closes[-1] >= self.closes[-n]
        # uptrend retracement: buy nearer 0.382–0.618 from high → pos 0.38–0.62
        if up:
            z = (0.50 - pos) * 3.0  # stretched below mid of range → buy
        else:
            z = (pos - 0.50) * 3.0
        out = {"z": _clip(z), "pos": pos, "up": up, "hi": hi, "lo": lo, "near": "", "levels": {}}
        swing = impulse_swing(self.highs[-n:], self.lows[-n:])
        if swing is None:
            return out
        swing_hi, swing_lo, swing_up = swing
        out["swing_hi"] = swing_hi
        out["swing_lo"] = swing_lo
        out["swing_up"] = swing_up
        out["levels"] = retracement_prices(swing_hi, swing_lo, swing_up)
        out["near"] = held_level(h, l, c, atr, swing_hi, swing_lo, swing_up)
        return out

    def _channel(self, c: float, atr: float) -> dict:
        n = min(40, len(self.closes))
        if n < 12:
            return {"z": None, "slope": 0.0, "resid": 0.0}
        ys = np.asarray(self.closes[-n:], dtype=float)
        xs = np.arange(n, dtype=float)
        A = np.vstack([xs, np.ones(n)]).T
        slope, intercept = np.linalg.lstsq(A, ys, rcond=None)[0]
        pred = float(intercept + slope * (n - 1))
        resid = (c - pred) / atr
        slope_z = slope * n / atr
        z = 0.65 * slope_z + 0.35 * (-resid)
        return {"z": _clip(z), "slope": float(slope), "resid": float(resid), "pred": pred}

    def _breadth(self, extras: dict | None) -> dict:
        ad = _opt(extras, "advance_decline", "ad_line", "breadth_ad")
        mcc = _opt(extras, "mcclellan", "mcclellan_oscillator")
        trin = _opt(extras, "trin", "arms_index")
        return {
            "ad_z": _clip(ad / 200.0) if ad is not None else None,
            "mcc_z": _clip(mcc / 50.0) if mcc is not None else None,
            "trin_z": _clip((1.0 - trin) / 0.35) if trin is not None else None,
        }

    def _sentiment(self, extras: dict | None) -> dict:
        # Contrarian: high put/call or VIX → buy fear.
        pc = _opt(extras, "put_call", "put_call_ratio")
        vix = _opt(extras, "vix", "VIX")
        cot = _opt(extras, "cot", "cot_net", "cot_commercial")
        pc_z = _clip((pc - 1.0) / 0.25) if pc is not None else None
        vix_z = _clip((vix - 20.0) / 8.0) if vix is not None else None
        cot_z = _clip(cot / 50_000.0) if cot is not None else None
        return {"pc_z": pc_z, "vix_z": vix_z, "cot_z": cot_z}
