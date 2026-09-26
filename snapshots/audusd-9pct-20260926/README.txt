Backup of the AUDUSD Yahoo M5 replay that finished at +$9,969.80 (+9.97%).
Taken before that exit logic was put into the book tuner.
EURUSD's book in this folder is the +8.49% file and was not part of the AUDUSD edit.

Restore by copying these files back over the live tree:
  flyfx\trader.py
  flyfx\brain\book_cfg.py
  flyfx\brain\profit_reflex.py
  flyfx\risk\bank.py
  flyfx\risk\fly_rails.py
  flyfx\ui\web\dash.html
  brains\AUDUSD.book.json
  brains\EURUSD.book.json
  tests\test_book_cfg.py
  HELP.md
