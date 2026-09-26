Backup taken 21 Sep 2026 BEFORE per-fly sugar crops + attributed dopamine.

This folder is the restore point for the book-level sugar overlay and shared
PnL PAM/PPL path (one satiety tank, all FlySwarm heads get the same close).

Contains copies of:
  flyfx/brain/sugar_feed.py
  flyfx/brain/plasticity.py
  flyfx/brain/fly_brains.py
  flyfx/trader.py
  flyfx/ui/dash.py
  flyfx/ui/web/dash.html
  tests/test_sugar_feed.py
  docs/ARCHITECTURE.md
  HELP.md

Factory freeze book (do not overwrite):
  snapshots/eurusd-snowball-lock-20260919
  Replay: +$1,906.66 / 4 trades  (9–18 Sep 2026 M5, --legacy-2oo3 --no-sugar --risk-tol balanced)

Restore this sugar/DA path (from the repo root):
  copy snapshots\pre-fly-crops-20260921\flyfx\brain\sugar_feed.py flyfx\brain\sugar_feed.py
  copy snapshots\pre-fly-crops-20260921\flyfx\brain\plasticity.py flyfx\brain\plasticity.py
  copy snapshots\pre-fly-crops-20260921\flyfx\brain\fly_brains.py flyfx\brain\fly_brains.py
  copy snapshots\pre-fly-crops-20260921\flyfx\trader.py flyfx\trader.py
  copy snapshots\pre-fly-crops-20260921\flyfx\ui\dash.py flyfx\ui\dash.py
  copy snapshots\pre-fly-crops-20260921\flyfx\ui\web\dash.html flyfx\ui\web\dash.html
  copy snapshots\pre-fly-crops-20260921\tests\test_sugar_feed.py tests\test_sugar_feed.py
  copy snapshots\pre-fly-crops-20260921\docs\ARCHITECTURE.md docs\ARCHITECTURE.md
  copy snapshots\pre-fly-crops-20260921\HELP.md HELP.md

In-code fallback without restoring files: leave --fly-crops / GUI per-fly crops off
(default). Sugar feed off still uses the native GRN lamp.
