"""python -m flyfx  → same as python fly_forex.py"""

from flyfx.trader import parse_args, run

if __name__ == "__main__":
    run(parse_args())
