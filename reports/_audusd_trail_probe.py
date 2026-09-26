"""One-off probe of the Sep 22 AUDUSD sell if it is held. Not part of the tool."""
import calendar
import time

from flyfx.trader import load_yahoo_m5

bars = load_yahoo_m5("AUDUSD", "60d")[-2000:]


def utc(text: str) -> int:
    return int(calendar.timegm(time.strptime(text, "%Y-%m-%d %H:%M")))


seq = [b for b in bars if b["time"] >= utc("2026-09-22 09:00")]
entry = 0.71082
spread = 0.00014
atr = 0.000321


def walk(arm_mult: float, gap_mult: float, be_mult: float) -> None:
    arm = arm_mult * atr
    gap = gap_mult * atr
    be = be_mult * atr
    sl = entry + 19.3 * 0.0001
    peak = None
    worst_give = 0.0
    for held, bar in enumerate(seq, start=1):
        ask_hi = bar["high"] + spread / 2
        ask_lo = bar["low"] + spread / 2
        ask = bar["close"] + spread / 2
        if peak is None:
            peak = ask
        if ask_hi >= sl:
            pips = (entry - ask_hi) / 0.0001
            stamp = time.strftime("%m-%d %H:%M", time.gmtime(bar["time"]))
            print(
                f"arm {arm_mult:.1f} gap {gap_mult:.1f} be {be_mult:.1f}  "
                f"EXIT {stamp} bar {held} about {pips:+.1f} pips  giveback {worst_give:.1f}"
            )
            return
        peak = min(peak, ask_lo)
        give = (ask_hi - peak) / 0.0001
        if (entry - peak) / 0.0001 >= 30:
            worst_give = max(worst_give, give)
        if ask_lo <= entry - be:
            sl = min(sl, entry - 0.15 * atr)
        if ask_lo <= entry - arm:
            sl = min(sl, ask_lo + gap)
        g = time.gmtime(bar["time"])
        if g.tm_wday == 4 and g.tm_hour >= 17:
            pips = (entry - ask) / 0.0001
            print(
                f"arm {arm_mult:.1f} gap {gap_mult:.1f} be {be_mult:.1f}  "
                f"FRIDAY {pips:+.1f} pips bar {held} giveback {worst_give:.1f}"
            )
            return
    print("no exit")


for arm_m, gap_m, be_m in (
    (3.2, 1.8, 1.5),
    (3.2, 1.8, 30),
    (8.0, 6.0, 30),
    (12.0, 8.0, 30),
    (20.0, 12.0, 30),
    (40.0, 20.0, 40),
):
    walk(arm_m, gap_m, be_m)
