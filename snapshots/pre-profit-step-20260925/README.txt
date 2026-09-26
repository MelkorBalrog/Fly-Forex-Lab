Frozen copy of the recycle and bank logic before the "latest profit only" size step
and the one-bar profit-spike close.

This is the book that replayed EURUSD M5 2000 bars (lab stack: scalp, sugar,
nn vote, settings ann, fib, dynamic fuse, volume auto, appetite) at
+$5,752.59. Report: reports/trades-EURUSD-20260925-223012.csv
Log: reports/eurusd_5k_run7.txt

Do not overwrite this folder. To restore this version, copy these files back:

  snapshots/pre-profit-step-20260925/flyfx/risk/profit_recycle.py
      → flyfx/risk/profit_recycle.py
  snapshots/pre-profit-step-20260925/flyfx/risk/bank.py
      → flyfx/risk/bank.py
  snapshots/pre-profit-step-20260925/flyfx/trader.py
      → flyfx/trader.py
  snapshots/pre-profit-step-20260925/flyfx/ui/web/dash.html
      → flyfx/ui/web/dash.html
  snapshots/pre-profit-step-20260925/mt4/FlyTrader.mq4
      → mt4/FlyTrader.mq4
  snapshots/pre-profit-step-20260925/tests/test_profit_recycle.py
      → tests/test_profit_recycle.py
  snapshots/pre-profit-step-20260925/tests/test_bank.py
      → tests/test_bank.py

Then delete flyfx/brain/profit_reflex.py and tests/test_profit_reflex.py
if they exist. Those files are the later reflex nets.

In this copy, size_budget() returns None (full account after a win) and there
is no one-bar profit-spike flatten. Demo/replay only.
