# Weekly dopamine trainer. Run Sunday after the FX week closes.
# Task Scheduler example (use the folder where you placed the project):
#   schtasks /Create /SC WEEKLY /D SUN /ST 18:00 /TN FlyFOREX-WeeklyTrain /TR "powershell -File path\to\FlyFOREXTrader\scripts\weekly_train.ps1"
#
# EURUSD is frozen. Other pairs update brains\{PAIR}.json from Yahoo/HST history.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PYTHONUNBUFFERED = "1"
python fly_train.py --weekly
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python fly_forex.py --hst auto --bars 2000 --reset-adapt --interval-ms 0 --overfit --use-brain --no-plastic
exit $LASTEXITCODE
