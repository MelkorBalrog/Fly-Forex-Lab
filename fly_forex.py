"""FlyFOREXTrader CLI. Same as `python -m flyfx`. See HELP.md."""

from __future__ import annotations

from flyfx.trader import parse_args, run

if __name__ == "__main__":
    run(parse_args())
