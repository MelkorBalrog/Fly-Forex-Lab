Frozen copy of the EURUSD-winning book (snowball day-lock + fly-trader rails).

Replay 9-18 Sep 2026 EURUSD M5 (--hst auto --bars 2000 --reset-adapt):
  4 trades  2W/2L  net +$1,906.66  best +$2,909.55  worst -$1,086.06
  snowball locked the rest of 14 Sep after the 20-lot short; 1% peak kill after the next-day stop.
  report: reports/trades-EURUSD-20260919-192336.csv
  sweep:  reports/pair_sweep-20260919-202735.csv  (2/7 green, mean -$184)

Do not overwrite this folder. Restore the EURUSD path:
  copy these .py files over the project root
  fly_forex.py  fly_rails.py  plasticity.py  banc_risk.py
  node_fusion.py  bayes_sizer.py  kalman_signal.py  account_sim.py  fly_dash.py

Then:
  python fly_forex.py --hst auto --bars 2000 --reset-adapt --interval-ms 0

EURUSD starts from factory readout (W frozen, D = 0). Do not load a trained
brain for EURUSD on this replay — later dopamine files live in brains/ and
are for other pairs. Demo/replay only.
