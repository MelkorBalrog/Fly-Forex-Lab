Backup of the code that replayed EURUSD M5 2000 bars at +$8,486.45 (+8.49%).
Taken before those knobs were stored on each pair brain.

Restore by copying these files back over the live tree:
  flyfx\trader.py
  flyfx\risk\bank.py
  flyfx\risk\fly_rails.py
  flyfx\brain\profit_reflex.py
  flyfx\ui\web\dash.html
  tests\test_bank.py
  tests\test_profit_reflex.py
Then delete flyfx\brain\book_cfg.py, tests\test_book_cfg.py, and brains\*.book.json if present.
