"""Held-out replay of the payoff book. Does not write brains."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flyfx.trader import CONNECTOME_DIR, FlyBrain, PaperBroker, ensure_connectome
from flyfx.trader import replay_prices, trade_loop
from scripts.train_until_weekly import (
    BALANCE,
    PAIRS,
    ROOT,
    _base_ns,
    load_history,
    split_bars,
)

WARM_BARS = 400


def main() -> None:
    tag = "-".join(sys.argv[1:]).upper() or "ALL"
    log_path = ROOT / "reports" / f"payoff_book_{tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8", buffering=1)
    print("loading connectome", flush=True)
    brain = FlyBrain(ensure_connectome(CONNECTOME_DIR))
    broker = PaperBroker(None)
    want = [s.upper() for s in sys.argv[1:]] or list(PAIRS)
    rows = []
    for symbol in PAIRS:
        if symbol not in want:
            continue
        bars, source = load_history(symbol)
        if not bars:
            print(f"{symbol}  no history", flush=True)
            continue
        _train, exam, cut = split_bars(bars)
        warm = [b for b in bars if int(b["time"]) < cut][-WARM_BARS:]
        window = warm + exam
        print(
            f"{symbol}  exam {time.strftime('%Y-%m-%d', time.gmtime(exam[0]['time']))}"
            f" -> {time.strftime('%Y-%m-%d', time.gmtime(exam[-1]['time']))}  {source}",
            flush=True,
        )
        ns = _base_ns(symbol)
        ns.use_brain = True
        ns.use_evo = True
        ns.reset_brain = False
        ns.force_brain = True
        ns.save_brain = False
        ns.shadow_da = False
        ns.freeze_learn = True
        ns.reset_adapt = True
        ns.evolve = False
        ns.score_from = cut
        ns.replay_bars = window
        ns.adapt_state = str(ROOT / "reports" / f"adapt_payoff_{symbol}.json")
        saved = sys.stdout
        sys.stdout = log
        try:
            summary = trade_loop(brain, replay_prices(window, ns.pair_spread), ns, broker, delay=0.0) or {}
        finally:
            sys.stdout = saved
        net = float(summary.get("net_usd") or 0.0)
        n = int(summary.get("n") or 0)
        t0, t1 = int(exam[0]["time"]), int(exam[-1]["time"])
        weeks = max((t1 - t0) / (7.0 * 86400.0), 1.0 / 7.0)
        rate = (net / BALANCE) / weeks
        line = (
            f"{symbol}  weekly {rate:.2%}  net ${net:+.2f}  n={n}  "
            f"{int(summary.get('wins') or 0)}W/{int(summary.get('losses') or 0)}L  weeks={weeks:.2f}"
        )
        print(line, flush=True)
        rows.append((symbol, rate, net, n))
    if rows:
        total = sum(r[2] for r in rows)
        print(f"book  net ${total:+.2f}  pairs {len(rows)}", flush=True)
    log.close()


if __name__ == "__main__":
    main()
