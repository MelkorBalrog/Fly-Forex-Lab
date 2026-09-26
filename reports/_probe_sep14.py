"""Probe legacy vs cat-fuse tech at 4k / freeze timestamps."""
from __future__ import annotations

import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from flyfx.exec.bar_io import load_hst
from flyfx.risk.bayes_sizer import AdaptiveParams
from flyfx.sense.category_kalman import parse_fuse_inds
from flyfx.trader import Indicators, PullbackSetup, find_hst


def run(mixer: str, inds: str | None) -> None:
    path = find_hst("EURUSD", 5)
    if path is None:
        raise SystemExit("no EURUSD5.hst")
    bars = load_hst(path)[-2000:]
    ind = Indicators(pip=0.0001, mixer=mixer)
    if mixer == "cat_fuse":
        ind.fuse_inds = parse_fuse_inds(inds or "plausibility")
        ind.cats.channel_include = ind.fuse_inds
    setup = PullbackSetup()
    params = AdaptiveParams()
    targets = {
        "2026-09-10 13:40",
        "2026-09-10 13:50",
        "2026-09-11 08:05",
        "2026-09-11 11:25",
        "2026-09-14 08:05",
        "2026-09-15 08:35",
        "2026-09-15 09:15",
    }
    print("====", mixer, inds, "hst", path)
    for step, bar in enumerate(bars):
        stamp = int(bar["time"])
        label = time.strftime("%Y-%m-%d %H:%M", time.gmtime(stamp))
        feat = ind.update(
            bar["high"],
            bar["low"],
            bar["close"],
            volume=bar.get("volume"),
            open_=bar.get("open"),
            stamp=stamp,
        )
        tech, kind = setup.decide(feat, bar["close"], step, params)
        if label in targets:
            k = feat.get("kalman") or {}
            print(
                f"{label} tech={tech}/{kind} reg={feat.get('regime')} "
                f"imp={float(feat.get('impulse') or 0):+.3f} "
                f"rsi={float(feat.get('rsi') or 0):.1f} "
                f"kreg={k.get('kalman_regime')} iso={k.get('isolation')!r} "
                f"disagree={k.get('disagree')}"
            )


if __name__ == "__main__":
    run("legacy", None)
    run("cat_fuse", "plausibility")
    run("cat_fuse", "all")
