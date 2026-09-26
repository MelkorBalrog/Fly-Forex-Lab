Backup taken 21 Sep 2026 BEFORE genetic Train DA on EURUSD.

This folder is the restore point for factory EURUSD DA and all pair brains
as they were before `--include-eurusd --evolve`.

Contains:
  brains/                 copy of brains/ (EURUSD.json is factory-frozen)
  reports/adapt_EURUSD.json

Factory freeze book (do not overwrite):
  snapshots/eurusd-snowball-lock-20260919
  Replay: +$1,906.66 / 4 trades  (9–18 Sep 2026 M5, --legacy-2oo3 --no-sugar --risk-tol balanced)

Restore EURUSD factory DA:
  copy snapshots\pre-eurusd-evolve-20260921\brains\EURUSD.json brains\EURUSD.json
  del brains\EURUSD.evo.json   (if present after the evolve run)

Restore every pair brain:
  copy snapshots\pre-eurusd-evolve-20260921\brains\*.json brains\
