Per-node Kalman + Bayesian fusion on the 2oo3 voters, plus a third fly
head (fly-conf) that reads p_cast / p_edge and sets the abstain bar.

Native votes still go to the committee. Fusion may only abstain; it never
replaces a one-bar pullback with a lagged Kalman mix.

Replay 9-18 Sep 2026 EURUSD M5 (--hst auto --bars 2000 --reset-adapt):
  6 trades  3W/3L  net +$982.86  (same fills as the no-fusion 2oo3 snapshot)
  report: trades-20260919-145549.csv

An earlier mix-as-vote fusion missed the pullbacks (0 trades, then +$34).
Do not restore that path.

Baseline without fusion:
  snapshots/2oo3-swarm-20260919
  or: python fly_forex.py --hst auto --bars 2000 --reset-adapt --no-fusion
