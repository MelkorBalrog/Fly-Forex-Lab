Snapshot of the 2-out-of-3 FlySwarm (no per-node Kalman/Bayes fusion).

Replay 9-18 Sep 2026 EURUSD M5 (--hst auto --bars 2000 --reset-adapt):
  6 trades  3W/3L  net +$982.86  best +$1,192.83  worst -$915.40
  report: trades-20260919-142041.csv

Restore this baseline:
  copy these .py files over the project root
  fly_forex.py  bayes_sizer.py  kalman_signal.py  account_sim.py

Or keep the fusion code and run:
  python fly_forex.py --hst auto --bars 2000 --reset-adapt --no-fusion

