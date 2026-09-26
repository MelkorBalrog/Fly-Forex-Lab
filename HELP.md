# FlyFOREXTrader — how to use it

This is the operator guide. Demo / replay only: nothing here guarantees live profit.

**Default path:** seven indicator categories each keep a Kalman latent. Those latents are **inverse-variance fused** into one impulse. Then the old 3-voter 2oo3 runs: `{tech (PullbackSetup), fly-trend, fly-fade}`. Fly-confidence may only abstain. Empty overlays (volume / breadth / sentiment) stay out of the mix — they are not faked.

The EURUSD freeze book (`+$1,906.66` on the 9–18 Sep 2026 M5 window) is **`--legacy-2oo3`** (EMA/RSI/ATR mixer, same 3 voters). The 21-head 7-category committee is **`--categories`**. Optional experiments (`--koo9`, `--nest`, `--ensemble`, `--pair-book`) stay off unless you pass the flag.

Architecture diagrams (how a bar becomes a trade): `docs/ARCHITECTURE.md`. Preview with **Ctrl+Shift+V**.

---

## 1. What you are running

Python owns the strategy (MaleCNS fly connectome + fused 7-Kalman 3-voter + risk). MetaTrader 4 is only I/O: quotes and `OrderSend`.

| You want to… | Use |
|---|---|
| Watch the 7 Kalman sensors fuse into 2oo3 and replay a pair | GUI (`python fly_forex.py --gui`) |
| Mix only some Kalmans (trend, or trend+volume, …) | GUI **FUSE** boxes, `--fuse-cats`, or MT4 **FlyTrader** Inputs (CLI live) |
| Pick which indicators inside a category fuse | GUI rows under FUSE (EMA, RSI, …), `--fuse-inds`, or MT4 **indicator** Inputs (CLI live) |
| Set how trades close (risk tolerance) | GUI **RISK**, `--risk-tol`, or MT4 **InpRiskTol** (`conservative` / `balanced` / `aggressive` / `scalp`) |
| Set %% of available money pledged as trade margin | GUI **VOL %** / **VOL mode**, `--volume-pct` / `--volume-mode`, or MT4 **InpVolumePct** / **InpVolumeMode** |
| Dynamically increase trade count when under-trading | GUI **dynamic entries** + **TRADE/1k**, `--entry-appetite` / `--trade-rate`, or MT4 **InpEntryAppetite** / **InpTradeRate** |
| Bank open profit at %% of complete balance | GUI **BANK %**, `--bank-pct`, or MT4 **InpBankPct** (0 = off) |
| Early bank / giveback from rate-of-change | GUI **hold risk** + **HOLD×**, `--hold-risk` / `--hold-risk-sens`, or MT4 **InpHoldRisk** |
| After a win: BB squeeze→expand then size from 1.5× profit | GUI **profit recycle**, `--profit-recycle`, or MT4 **InpProfitRecycle** |
| Bayesian risk from sealed trade + market conditions | GUI **trade bayes**, `--trade-bayes`, or MT4 **InpTradeBayes** |
| Sugar feed (eat / don't eat) | GUI **sugar feed** (lab default on), **amt** 0–2, `--sugar` / MT4 **InpSugar** / **InpSugarAmt** |
| Let conditions pick sugar amt | GUI **DYNAMIC sugar**, `--sugar-dyn`, or MT4 **InpSugarDyn** (FEAST/FORAGE/NIBBLE/FAST; amt is a scale) |
| Per-fly sugar tanks + attributed DA | GUI **per-fly crops** (needs sugar feed), `--fly-crops`, or MT4 **InpFlyCrops** |
| Let market conditions pick the mix | GUI **DYNAMIC mix**, `--fuse-dynamic`, or MT4 **Dynamic mix** (FUSE boxes become an allowlist) |
| Replay EURUSD from disk, no window | CLI `--hst auto` |
| Old EMA/RSI/ATR 3-voter (CLI only) | `--legacy-2oo3` |
| Restore the 7×3 category committee | add `--categories` |
| Send demo orders through MT4 | GUI **Live** or CLI without `--paper` |
| Train a pair’s dopamine from history files | GUI **Train DA** (evolve on by default), or `python fly_train.py` / `python fly_train.py --evolve` |
| Auto-retune / train AI before Replay | GUI **RECAL** (`train AI` / `calibrate` / `train+calibrate` / `off`) or `--recalibrate-mode` |
| Deep retune (evolve + risk_factors) on this window | GUI **Recalibrate**, or `--recalibrate-explicit` |
| Save / load GUI dials | Lab **CFG** bar: name + Save / Load / Default. Files in `configs/{name}.json`. Last save auto-restores on reopen. |
| CUDA/tensor 4th voter (3oo4) | GUI **AI 3oo4** (default on with recalibrate), `--nn-vote` / `--no-nn-vote`. Trains during recalibrate on the calibrate slice. The 4th vote is a Bayesian average of 10 nets (different depth, width, activation) fit on the same labels. Weights in `brains/{PAIR}.bma.pt` (or `.bma.npz`). Every retrain of every ANN in the project keeps prior knowledge (warm-start, distill, replay) unless GUI **Fresh ANN** / `--nn-fresh`: the 10-net voter (`brains/{PAIR}.nn_mem.npz`), the 20 settings nets (`brains/{PAIR}.setann.npz` and `.setann_mem.npz`), and the profit-reflex size and close groups (`brains/{PAIR}.reflex.npz` and `.reflex_mem.npz`). **Adversarial training** (feature FGSM/PGD) is on by default for the 10-net voter — GUI **adv train** / `--nn-adv` / `--no-nn-adv`. Uses CUDA when PyTorch+GPU is installed, else CPU/numpy. A VLM is **not** used. |
| Dynamic entry rate (appetite) | GUI **dynamic entries** + **TRADE/1k** (default 8 per 1000 bars), or `--entry-appetite --trade-rate 8`. When behind target: lowers `min_impulse` / `cont_impulse` and allows **soft 2oo3** if the NN HOLDs (still vetoes opposite flies; never skips `costs_ok` / rails circuit). |

From the project folder (see [§3](#3-one-time-setup)):

```powershell
python fly_forex.py --gui
```

### How a bar becomes a trade

1. **Indicators** — OHLCV is grouped into seven families (trend, momentum, volatility, volume, breadth, structure, sentiment). Missing overlays stay empty; they are not faked.
2. **Kalman** — each family is fused into one latent (`impulse`, `residual`, uncertainty, agreement).
3. **Fuse** — live latents you selected (GUI **FUSE** boxes / `--fuse-cats`) are inverse-variance mixed into one impulse / residual / regime. Inside each family, only the indicators you checked (`--fuse-inds` / GUI EMA, RSI, …) enter that Kalman. Empty volume / breadth / sentiment do not vote even if checked. The dashboard and the EA start with the plausibility set checked: EMA, MACD, ADX, SAR, Ichimoku, Supertrend, RSI, stochastic, CCI, Williams %R, and Bollinger, in the trend, momentum, and volatility families. SMA, WMA, ROC, ATR, Keltner, StdDev, Donchian, and the volume, breadth, structure, and sentiment families start unchecked. A blank `--fuse-inds` on the CLI still means every channel. **DYNAMIC mix** (off by default) lets a TREND/RANGE/BREAK/QUIET state machine pick the subset each bar; checked FUSE boxes are then an allowlist.
4. **Tech** — one PullbackSetup node on that fused impulse (bounce / continuation; CHOP → HOLD).
5. **Flies** — three MaleCNS heads on one shared wiring diagram `W`: trend, fade, confidence.
6. **2oo3** — `{fly-trend, fly-fade, tech}` must agree 2 or 3 ways. Fly-conf does not vote a side; fusion may only abstain. The side then has to pass an indicator check: at least two of EMA, DI, SAR, Supertrend, and Ichimoku, plus momentum from MACD, RSI, or stochastic / Williams %R / CCI. A pinned RSI + stochastic + Bollinger extreme is refused. A thinner agreement trades smaller.
7. **Size / execute** — existing gate, BANC, ½ Kelly, snowball, trail. On the fill, the next hour has a price forecast from the rate of change and the impulse. The trade closes if that price prints in the hour and it is in profit, and it closes sooner if price runs the other way by about half the initial risk. **RISK** / `--risk-tol` only changes *how* an open trade gets closed. Default **balanced** is the factory close set. Unchecked **sugar feed** keeps the current 2oo3 voters, FUSE indicators, and DYNAMIC mix, plus the **native GRN lamp**: 23 sugar cells get `0.45 × nice` on the risk pass (the behavior from before the sugar-feed checkbox). Checking **sugar feed** / `--sugar` adds the satiety overlay (skip / size / snowball / spit-out). **amt** and **DYNAMIC sugar** only apply to that overlay. Sugar never votes BUY/SELL. Freeze `--legacy-2oo3` is CLI-only.

---

## 2. Folders (where things live)

```
FlyFOREXTrader/
  HELP.md                 ← this file
  ARCHITECTURE.md         ← same diagrams as docs/ARCHITECTURE.md
  fly_forex.py            ← start the trader / GUI
  fly_train.py            ← dopamine trainer
  requirements.txt        ← packages to install
  requirements-nn.txt     ← optional PyTorch voter
  flyfx/                  ← Python package (brains, votes, risk, UI)
    trader.py             ← bar loop and MT4 bridge
    sense/                ← 7-category indicators + Kalman
    vote/                 ← tech nodes, 2oo3, plausibility pick
    brain/category_flies.py
    ui/web/dash.html      ← GUI page
  mt4/FlyTrader.mq4       ← Expert Advisor
  brains/                 ← per-pair dopamine files (EURUSD frozen)
  reports/                ← trade CSVs
  fly-connectome/         ← 166k neuron graph (downloaded on first run)
  snapshots/              ← frozen winning copies — do not edit as live code
  vendor/                 ← BANC + fly-trader source trees
  scripts/weekly_train.ps1
  docs/ARCHITECTURE.md
```

Data (`brains/`, `reports/`, `fly-connectome/`, `snapshots/`) stayed at the repo root so old reports and the EURUSD freeze still resolve.

---

## 3. One-time setup

Put the project in any folder you choose, for example `D:\Tools\FlyFOREXTrader` or `~/src/FlyFOREXTrader`. Every command below is run from that folder: the one that contains `fly_forex.py`.

### 3.1 Install dependencies

Python 3.11 or newer.

```powershell
cd path\to\FlyFOREXTrader
python -m pip install -r requirements.txt
```

On macOS or Linux the same two lines work in a terminal; only the folder path changes.

The 10-net voter uses PyTorch when it is installed, and a numpy network otherwise. GPU support is optional:

```powershell
python -m pip install -r requirements-nn.txt
```

The first GUI open or replay downloads the MaleCNS graph into `fly-connectome/` inside the project folder. That download needs the internet once.

### 3.2 Confirm the folder

```powershell
python -c "import flyfx; print(flyfx.PROJECT_ROOT)"
```

The printed path is the folder you chose. If this import fails, the shell is not in that folder.

---

## 4. Graphical interface (step by step)

Replay sim does **not** need MetaTrader. Live trading does.

### 4.1 Start the lab

```powershell
cd path\to\FlyFOREXTrader
python fly_forex.py --gui
```

The terminal prints `dashboard  http://127.0.0.1:8765/` and should open a browser. If it does not, paste that URL yourself.

Wait until the header is no longer “loading MaleCNS connectome…”. Idle text is: **idle — pick a pair, then Replay sim or Train DA**. The pipeline strip and 7 Kalman sensor rows are already on screen.

If port 8765 is busy, the app tries 8766, 8767, … and prints the port it used. Close any old `python fly_forex.py --gui` first.

### 4.2 What the screen is

**Pipeline strip** (under the buttons): OHLCV → 7 Kalmans → fuse (the **FUSE** boxes) → `{tech, fly-trend, fly-fade}` 2oo3 → BANC / size → trade. Uncheck a family to drop it from the mix. All checked = all live Kalmans.

Top bar: pair, equity, open position, BANC / rails / snowball notes.

Control row:

| Control | Meaning |
|---|---|
| **PAIR** | Symbol to replay or trade (EURUSD, GBPUSD, …). Filename like `EURUSD_M10.csv` also sets this. |
| **TF** | Bar size. **file native** keeps the imported file as-is. M10/H1 resample only if the file is finer (never upsample). |
| **BARS** | How many bars to use from the end of the file. **0 = entire file**. Default 2000 for MT4/Yahoo replay. |
| **CAPITAL $** | Simulated starting money (default $100,000). Lots still scale with `--risk`. |
| **LEV 1:** | Sim account leverage (default 50, so 1:50). Margin is notional divided by this. CLI: `--leverage`. |
| **RISK** | How open trades close. **scalp** (lab default) is the 4k cat-fuse companion. **balanced** is factory SL/TP/trail/hold — freeze-identical. **conservative** tightens stops and flattens on BANC drop, opposite fade, Kalman uncertainty, ATR blow-up, or kill-switch. **aggressive** widens stops and holds longer. GUI Live uses the dashboard. CLI live: MT4 `InpRiskTol`. |
| **sugar feed** | Lab default on. Overlay: skip / size / snowball / spit-out from GRNs + satiety. Unchecked keeps the current voters + FUSE indicators + DYNAMIC mix and the **native GRN lamp** (0.45×nice on the risk pass). Does not vote a side. Check / `--sugar` for the overlay. GUI Live uses the dashboard. CLI live: MT4 `InpSugar`. |
| **amt** | How much sugar to add when feed is on. **1** (default) is full. **0** eats like sugar off. **2** is extra (larger size). With **DYNAMIC sugar**, this is a scale (1 = machine as designed). GUI Live uses the dashboard. CLI live: MT4 `InpSugarAmt`. |
| **DYNAMIC sugar** | Off by default. FEAST / FORAGE / NIBBLE / FAST picks amt each bar from 2oo3, BANC, Kalman, satiety, chop, vol. GUI Live uses the dashboard. CLI live: MT4 `InpSugarDyn`. |
| **per-fly crops** | Off by default. Requires **sugar feed**. Each FlySwarm head (trend / fade / conf / risk) gets its own satiety tank and attributed PAM/PPL. Dose and DA gain follow that head’s results × book risk (`drive_from_results_risk`). GUI Live / Replay / Train DA; CLI live: MT4 **InpFlyCrops**. CLI: `--fly-crops`. |
| **DATA** | Import CSV/JSON from disk. MT4 `.hst` needs the path box (binary). |
| **or path** | Full path to a CSV/JSON/HST on this machine. |
| **Replay sim** | Fast historical sim. No MT4. No live orders. With **recalibrate** on (default), walk-forward retunes before the book. Check **use evolved** to load `brains/{PAIR}.evo.json`. Check **use DA brain** to load PAM/PPL traces. Each sim stores its bars in `data/history/{PAIR}/{TF}/`. A later window that overlaps or continues that series is written into the same file; a gap opens a new file. Train and recalibrate use every saved file plus the Yahoo or MetaTrader bars selected for this run. |
| **Triplet** | On for the 7% pairs. Eight nets read that pair plus the other two legs of its triangle (EURUSD uses EURUSD, AUDUSD, EURAUD; GBPUSD uses GBPUSD, EURUSD, EURGBP; AUDUSD and NZDUSD use AUDUSD, NZDUSD, AUDNZD; USDCAD uses USDCAD, AUDUSD, AUDCAD). Their trend score is mixed into every category node. Weights: `brains/{PAIR}.tri.npz`. |
| **Train DA** | Dopamine train on the selected pair + DATA file (or HST/Yahoo). Writes `brains/{PAIR}.json`. With **evolve** on, also writes `brains/{PAIR}.evo.json` (params + dynamic flags) and stamps a tape fingerprint for auto recalibrate skip. |
| **Recalibrate** | Full evolve + `fit_latent` on the loaded window. Writes `brains/{PAIR}.evo.json` and `reports/risk_factors_{PAIR}.json` (fingerprint + latent + genome). Then check **use evolved** and Replay. CLI: `--recalibrate-explicit`. |
| **RECAL** | Before Replay: **train AI** (tensor brain only, fast), **calibrate** (genome/latent walk-forward, slow), **train+calibrate** (both), or **off**. GUI default is **train AI**. CLI: `--recalibrate-mode train\|search\|both\|off`. |
| **AI 3oo4** | On by default with train / train+calibrate. Trains 10 nets with different structure on the same calibrate-slice labels (forward return), then the **4th voter** is their Bayesian average so entries need **3-of-4** agreement `{tech, fly-trend, fly-fade, nn}`. Opposite NN does not veto. Optional **chart vision** (`--nn-vision` / GUI) is a single Conv2d instead of the average. **Knowledge transfer (default):** retrain warm-starts from `brains/{PAIR}.bma.*`, distills soft policy from frozen prior nets, mixes a replay buffer (`brains/{PAIR}.nn_mem.npz`), and blends the BMA posterior — so RECAL does not wipe past learning. Check **Fresh ANN** / `--nn-fresh` to train from scratch. That same switch is the only scratch start for the 20 settings nets and the profit-reflex size and close groups; every other retrain warm-starts, distills, and replays prior rows. **Adversarial training (default):** each step also fits FGSM/PGD feature-space attacks (`eps≈0.06`, 2 PGD steps, 50% mix) plus light Gaussian noise so the voter stays reliable under noisy indicators — GUI **adv train** / `--nn-adv` / `--no-nn-adv`. Freeze `--legacy-2oo3` stays 2oo3 unless you force `--nn-vote`. CLI: `--nn-vote` / `--no-nn-vote`, `--nn-epochs`, `--nn-cpu`, `--nn-vision`, `--nn-fresh`, `--nn-adv`. Weights: `brains/{PAIR}.bma.pt`. |
| **evolve** | On by default for Train DA. Genetic search: population × generations. Each member trains fly brains (PAM/PPL) *and* mutates AdaptiveParams, RISK, sugar / DYNAMIC sugar, DYNAMIC mix, FUSE allowlist, snowball / martingale. Elites keep traces and params. W never moves. Uncheck for a single DA pass. Risk on the fill follows the live tape (spread versus the stop, ATR regime, London/NY hour, chase, heat). After evolve / Recalibrate, fitted `risk_factors` are applied on sim Replay when the fingerprint matches (live needs `--apply-latent-live`). |
| **pop** / **gens** | Evolve population (3–16, default 6) and generations (2–12, default 4). ~pop×gens full-speed replays. CLI: `--evolve --evo-pop --evo-gens`. |
| **use DA brain** | Replay (and Train if you also want to continue) loads that pair’s saved PAM/PPL traces from `brains/{PAIR}.json`. Nested genome still applies if **use evolved** is off and no `.evo.json` exists. |
| **use evolved** | Replay loads `brains/{PAIR}.evo.json`: evolved SL/TP/trail plus RISK, sugar, DYNAMIC mix/sugar, FUSE cats/indicators, snowball / martingale. Independent of DA traces. Checking it fills and locks those boxes from the file (uncheck to edit again). CLI: `--use-evo`. Live mix/sugar/risk still follow the EA. |
| **continue DA** | Keep learning from the existing brain instead of starting D from 0. Replay + this also saves. |
| **Live (MT4)** | Tick loop through the EA. Needs MT4 + this pair in Market Watch. Uses **dashboard dials** (same as Replay). Chart Inputs apply on CLI live. |
| **Stop** | Cancel the current replay, train, or live loop. The window stays up. |

Three fly cards below the board: retina maps for fly-trend / fly-fade / fly-conf (the 3-voter heads).

**MALECNS** card (top of the board): T-shaped fruit-fly CNS as a cloud of 3D spheres (VNC left, neck, central brain, two optic lobes). Each sphere is a sampled neuron from a real superclass / population. Size and brightness follow live `|x|` of the selected fly. Drag to orbit. Hover a cell for the region name. Positions are a schematic (the 166k package has no soma xyz); activity is not faked.

**ALL NODES** board: one row per Kalman sensor (fused value, agreement, live/empty). Empty overlays (volume/breadth/sentiment without data) stay dim. `--categories` expands each row to tech + three flies.

**COMMITTEE** card — 7 Kalman sensors, then the fused impulse, then `{tech, fly-trend, fly-fade}` 2oo3.

**DECISION TAPE**: opens, snowballs, closes.

`--legacy-2oo3` shows the freeze 3-voter card (EMA/RSI/ATR mixer). `--categories` shows the 7×3 committee table. `--koo9` shows nine flies. `--ensemble` shows five books. `--nest` shows the 3×3oo3 families.

### 4.3 Run a Replay sim (no MT4)

1. Leave PAIR on **EURUSD** the first time.
2. Set **BARS** to `2000` (or leave the default).
3. Click **Replay sim**.
4. Watch the 7 Kalman sensors, the fused impulse, and the 3-voter 2oo3. Replay is full speed (no 500 ms sleep).
5. When it finishes, the header returns to idle. A CSV is written under `reports/trades-EURUSD-….csv`.
6. Click **Stop** if you need to abort mid-run.

This default run is **cat-fuse** with sugar **feed** off: current 2oo3 voters, FUSE indicators, DYNAMIC mix if you check it, and the native 0.45×nice GRN lamp. That is the path from before the sugar-feed overlay.

The old EMA/RSI/ATR mixer is CLI-only (`--legacy-2oo3`), not a dashboard box.

To restore the 21-head 7-category committee:

```powershell
python fly_forex.py --gui --categories
```

You can change PAIR (GBPUSD, USDJPY, …) and Replay again without restarting Python.

### 4.4 Train dopamine on a pair (history files)

1. Pick **PAIR**.
2. Import a history file with **DATA** or **or path** (CSV/JSON, or `.hst` via the path box). Leave the file empty to use MT4 HST / Yahoo like Replay.
3. Set **TF** and **BARS** (0 = whole file). **CAPITAL $** is the sim account.
4. Leave **evolve** checked (default) unless you want a single DA pass. **pop** 6 and **gens** 4 is 24 full-speed replays (elites are not re-scored). Click **Train DA**. Each member trains PAM/PPL *and* mutates SL/TP/trail, RISK, sugar / DYNAMIC sugar, DYNAMIC mix, FUSE cats, snowball / martingale. The fittest **params + dynamic flags** are written to `brains/{PAIR}.evo.json`. PAM/PPL traces go to `brains/{PAIR}.json`. W never moves.
5. When it finishes, the PAIR menu marks that symbol **· evo** (and **· DA** if traces saved). Check **use evolved** (fills and locks RISK / sugar / DYNAMIC / FUSE from the file), then **Replay sim** (the terminal prints `evo loaded brains/{PAIR}.evo.json`). Uncheck **use evolved** to edit those boxes again. Check **use DA brain** as well if you want the traces. Plastic still on unless you started `--no-plastic`.
6. Check **continue DA** if a brain already exists and you want to keep updating it (evolve then seeds the first member from that brain). Check **use evolved** while training to seed from the last `.evo.json`.

EURUSD trains from the GUI like any other pair. `python fly_train.py` still skips EURUSD unless you pass `--include-eurusd`.

CLI equivalent: `python fly_train.py --pairs GBPUSD` (one DA pass) or `python fly_train.py --pairs GBPUSD --evolve --evo-pop 6 --evo-gens 4`. GUI Train DA is one pair at a time and uses the file you picked (any TF), not only the walk-forward M5 window. Uncheck **evolve** for the old single pass.

### 4.5 Attach MT4 for Live (demo)

Do this only on a **demo** account until you have watched many replays.

1. Copy `mt4/FlyTrader.mq4` into your MT4 `MQL4/Experts/` folder (or open it in MetaEditor from that folder).
2. Install the MT4 ZeroMQ library so `#include <Zmq/Zmq.mqh>` compiles (the EA will not load without `libzmq.dll`).
3. Compile in MetaEditor. You want 0 errors.
4. On a chart: **Allow DLL imports** and **Allow live trading**. Attach **FlyTrader**.
5. Attach **FlyTrader v1.31** and the chart shows the same lab as the browser, as clickable buttons. **Lab** has RISK, VOL, BANK, hold risk, profit recycle, trade bayes, dynamic entries, TRADE/1k, sugar, AI 3oo4, chart vision, Fresh ANN, adv train, train 20 ANNs, Fibonacci entries, evolve, use DA brain, use evolved, and continue DA. **Fuse** has DYNAMIC mix and the seven categories. **Indicators** has each channel. Green means on. The EA **Inputs** tab has the same dials and is what the buttons start from; change either one, and the running live loop picks it up. Recompile in MetaEditor and reattach after you copy a new `FlyTrader.mq4`, or the chart stays on the old expert with no buttons.
6. Experts tab should show `FlyTrader: ready` plus fuse / risk / sugar / vol / nn lines.
7. Put every pair you might Live-trade in **Market Watch**.
8. In the GUI, pick that PAIR, then **Live (MT4)**. Replay and Train still use the browser dials. Once the live loop is running, the chart buttons are the dials: fuse, risk, sugar, volume, the 10-net voter, the 20 settings nets, Fibonacci entries, recycle, and the rest. Clicking one updates the next decisions.
9. Default live loop sleeps 500 ms between decisions. **Stop** ends the live loop; the dashboard stays open.

`python fly_forex.py --ping` prints the lab dials the EA is sending. Reattaching the EA restores the buttons from the Inputs tab.

Live **sends orders** unless you started with `--paper`:

```powershell
python fly_forex.py --gui --paper
```

`--paper` still reads MT4 prices but does not `OrderSend`. Use `--no-ea-inputs` on CLI live to keep Python/CLI flags instead of chart Inputs.

If Live says **MT4 not connected**, Replay sim still works. Check: EA attached, DLLs allowed, port `tcp://*:5555` free, Python `--endpoint` still `tcp://127.0.0.1:5555`.

### 4.6 Leave the GUI

Ctrl+C in the terminal, or close that Python window. Then close the browser tab.

---

## 5. Command line (same engine as the GUI)

All commands from the project folder (the directory that contains `fly_forex.py`).

### 5.1 Ping the EA (market can be closed)

```powershell
python fly_forex.py --ping
```

Expect `PONG`, account numbers, a bid/ask, and `fuse-cats  MT4 Inputs` / `fuse-dynamic  MT4 Inputs` from the EA.

### 5.2 Offline EURUSD replay (cat-fuse default / 4k companion)

Uses the local MT4 `.hst` file. No EA required if history is already on disk.
Default **RISK = scalp** is the 4k cat-fuse companion (VOL 12%, factory geom, bank off):

```powershell
python fly_forex.py --replay --hst auto --timeframe M5 --bars 2000 --interval-ms 0 --reset-adapt
```

Same book with flags spelled out:

```powershell
python fly_forex.py --replay --hst auto --timeframe M5 --bars 2000 --interval-ms 0 --reset-adapt --recalibrate-mode off --no-sugar --risk-tol scalp
```

Replay with auto recalibrate on. Pick what runs:

```powershell
# Fast: train tensor AI only, then play the book (recommended while iterating)
python fly_forex.py --replay --hst auto --timeframe M5 --bars 2000 --interval-ms 0 --reset-adapt --recalibrate-mode train

# Slow: genome/latent walk-forward only (uses existing NN weights if AI 3oo4 is on)
python fly_forex.py --replay --hst auto --timeframe M5 --bars 2000 --interval-ms 0 --reset-adapt --recalibrate-mode search

# Full: train AI then genome search
python fly_forex.py --replay --hst auto --timeframe M5 --bars 2000 --interval-ms 0 --reset-adapt --recalibrate-mode both
```

Fingerprint match (`reports/risk_factors_{PAIR}.json`) can skip search when mode is `search`. This seeks positive holdout expectancy on **this** window — it does not guarantee live profit. Freeze regression / known book:

```powershell
python fly_forex.py --replay --hst auto --timeframe M5 --bars 2000 --interval-ms 0 --reset-adapt --recalibrate-mode off --legacy-2oo3 --risk-tol balanced
```

Explicit deep retune (full evolve + risk_factors + evo fingerprint):

```powershell
python fly_forex.py --replay --hst auto --timeframe M5 --bars 2000 --interval-ms 0 --reset-adapt --recalibrate-explicit --evo-pop 6 --evo-gens 4
```

Trend-only, or a subset mix (still 3-fly 2oo3; empty overlays stay out even if named):

```powershell
python fly_forex.py --replay --hst auto --timeframe M5 --bars 2000 --interval-ms 0 --reset-adapt --fuse-cats trend
python fly_forex.py --replay --hst auto --timeframe M5 --bars 2000 --interval-ms 0 --reset-adapt --fuse-cats trend,volume
python fly_forex.py --replay --hst auto --timeframe M5 --bars 2000 --interval-ms 0 --reset-adapt --fuse-cats trend+structure
```

Dynamic mix: the FSM picks TREND/RANGE/BREAK/QUIET each bar. FUSE boxes / `--fuse-cats` stay an allowlist.

```powershell
python fly_forex.py --replay --hst auto --timeframe M5 --bars 2000 --interval-ms 0 --reset-adapt --fuse-dynamic
python fly_forex.py --replay --hst auto --timeframe M5 --bars 2000 --interval-ms 0 --reset-adapt --fuse-dynamic --fuse-cats trend,momentum,structure
```

Inside a family, keep only some indicators (other channels stay warm but do not enter that Kalman mix):

```powershell
python fly_forex.py --replay --hst auto --timeframe M5 --bars 2000 --interval-ms 0 --reset-adapt --fuse-inds ema,macd,rsi
python fly_forex.py --replay --hst auto --timeframe M5 --bars 2000 --interval-ms 0 --reset-adapt --fuse-inds trend:ema+macd;momentum:rsi
```

Risk tolerance (closes). `balanced` is the factory geometry. `conservative` also flattens when BANC, fade, uncertainty, vol, or the kill switch say the trade is unsafe:

```powershell
python fly_forex.py --replay --hst auto --timeframe M5 --bars 2000 --interval-ms 0 --reset-adapt --risk-tol conservative
python fly_forex.py --replay --hst auto --timeframe M5 --bars 2000 --interval-ms 0 --reset-adapt --risk-tol aggressive
python fly_forex.py --replay --hst auto --timeframe M5 --bars 2000 --interval-ms 0 --reset-adapt --legacy-2oo3 --risk-tol conservative
python fly_forex.py --replay --hst auto --timeframe M5 --bars 2000 --interval-ms 0 --reset-adapt --risk-tol scalp
```

**`scalp`** (default) is the 4k cat-fuse companion: factory SL 2.40 / TP 12 ATR, hold 4–64, day≤6, VOL 12%, snowball on, bank off. Freeze book stays `--legacy-2oo3 --risk-tol balanced` (classic risk-% sizing).

Volume % of available money (pledged as **position margin**):

```powershell
python fly_forex.py --replay --hst auto --bars 2000 --interval-ms 0 --reset-adapt --volume-mode equity
python fly_forex.py --replay --hst auto --bars 2000 --interval-ms 0 --reset-adapt --volume-mode fixed --volume-pct 50
python fly_forex.py --replay --hst auto --bars 2000 --interval-ms 0 --reset-adapt --volume-mode auto --risk-tol conservative
```

`equity` mode (default) sizes lots so required margin ≈ available money × VOL%/100 (free margin if usable, else equity). Profile defaults: conservative 50%, balanced/aggressive 100%. `--volume-pct` overrides that number. `auto` also multiplies by live tape × BANC. BANC / bayes / admit may haircut the target; `--max-lots` can still clamp below the requested %. Example: VOL 50 on $100k → target margin $50k → lots = $50k / margin_per_lot.

Dynamic entries (appetite). Off by default so freeze / trained books stay unchanged. When on, tracks sealed entries over a rolling 200-bar window and compares to **TRADE/1k** (default 8 per 1000 bars). Behind target → lower impulse floors and allow **soft 2oo3** when the NN HOLDs (tag `2oo3-soft`). Does not bypass costs, rails circuit, or opposite-fly veto:

```powershell
python fly_forex.py --replay --hst auto --bars 2000 --interval-ms 0 --reset-adapt --entry-appetite --trade-rate 8
python fly_forex.py --replay --hst auto --bars 2000 --interval-ms 0 --reset-adapt --nn-vote --entry-appetite --trade-rate 12
```

Bank open profit at a percent of complete balance (CAPITAL). Off by default (`0`). When floating USD ≥ BANK% × balance, the trade closes immediately (reason `bank …`) so a +$1,500 winner cannot reverse to a large loss. Checked right after the hard stop, before trail/TP give-back:

```powershell
python fly_forex.py --replay --hst auto --bars 2000 --interval-ms 0 --reset-adapt --bank-pct 1.5
python fly_forex.py --replay --hst auto --bars 2000 --interval-ms 0 --reset-adapt --bank-pct 2 --volume-pct 50
python fly_forex.py --replay --hst auto --bars 2000 --interval-ms 0 --reset-adapt --bank-pct 1.5 --hold-risk-sens 1.5
```

**Hold risk** (on by default) scores how dangerous it is to *keep* an open trade from rate-of-change over the last ~6 bars: adverse cycle RoC/jump, falling floating $, and giveback from peak float. Higher risk → lower effective BANK % (early bank). Extreme risk + large giveback can flatten a still-green trade before BANK % (`hold-risk …`). Uncheck **hold risk** / `--no-hold-risk` for pure BANK % only.

**Profit recycle** (on by default). After a sealed win of $W: (1) no new entries until Bollinger bandwidth squeezes (`bw/ma ≤ 0.90`) then expands again (`bw/ma ≥ 1.05`); (2) the next trade’s money base is capped at `1.5 × W` (profit + 50% of that profit — e.g. $1k win → $1.5k budget, not the full account). A loss while armed clears the gate. GUI **profit recycle**; CLI `--profit-recycle` / `--no-profit-recycle` / `--profit-recycle-mult 1.5`.

**Tune book to 7%** (off by default). Each pair has its own book file `brains/{PAIR}.book.json`. Leave it unchecked to reuse that file. Check it to run one diagnostic replay: eight nets read the whole week and those trades, their Bayesian average sets the book, and that book is checked once before the simulation. A pair whose saved return is already at the target is not searched again. At most two quiet replays. A different pair keeps a different file.

**Trade bayes** (on by default). Each sealed close updates Beta posteriors keyed by market + setup bins (side, regime, ATR regime, session, impulse, Bollinger state, loss heat, committee tag), plus a similarity kernel over recent trades. Before the next entry the posterior sets: **inhibit** (skip when P(win)/expectancy is clearly bad with enough evidence), **size×** (half-Kelly style volume scale), and while open: **trim** half a winner if conditions turned adverse, or **allow/boost snowball** when similar setups paid. **Persists** to `reports/trade_bayes_{PAIR}.json` after every sealed close (and into `adapt_{PAIR}.json`) so the next Replay / Live / GUI open **continues** the same posteriors. Cleared only with `--reset-adapt`. GUI **trade bayes**; CLI `--trade-bayes` / `--no-trade-bayes`. Freeze purity: pass `--no-trade-bayes` with `--legacy-2oo3 --risk-tol balanced`.

Sugar feed (off by default). Check **sugar feed** / `--sugar` to turn it on. Uncheck / `--no-sugar` keeps the current 2oo3 + FUSE + DYNAMIC mix and the native GRN lamp. That does **not** switch the mixer. The old EMA/RSI/ATR 3-voter is CLI `--legacy-2oo3`:

```powershell
python fly_forex.py --replay --hst auto --timeframe M5 --bars 2000 --interval-ms 0 --reset-adapt --sugar
python fly_forex.py --replay --hst auto --timeframe M5 --bars 2000 --interval-ms 0 --reset-adapt --sugar --sugar-amt 0.5
python fly_forex.py --replay --hst auto --timeframe M5 --bars 2000 --interval-ms 0 --reset-adapt --sugar --sugar-amt 2
python fly_forex.py --replay --hst auto --timeframe M5 --bars 2000 --interval-ms 0 --reset-adapt --sugar --sugar-dyn
python fly_forex.py --replay --hst auto --timeframe M5 --bars 2000 --interval-ms 0 --reset-adapt --no-sugar
python fly_forex.py --replay --hst auto --timeframe M5 --bars 2000 --interval-ms 0 --reset-adapt --legacy-2oo3 --no-sugar
```

Same engine as the GUI Replay button. Compare against the freeze 3-voter:

```powershell
python fly_forex.py --replay --hst auto --timeframe M5 --bars 2000 --interval-ms 0 --reset-adapt --legacy-2oo3
```

On the 9–18 Sep 2026 EURUSD M5 window, `--legacy-2oo3` is the freeze book: **+$1,906.66**, 4 trades. The 4k+ EURUSD Replay was **cat-fuse** after FUSE indicator picks, DYNAMIC mix, and RISK, with the native GRN lamp and **before** the sugar-feed overlay. Leave **sugar feed** off for that lamp; re-check the same FUSE / DYNAMIC mix / RISK boxes. That is not the freeze mixer. `--categories` is the old 21-head committee (it overtraded that window).

Same thing as:

```powershell
python -m flyfx --replay --hst auto --timeframe M5 --bars 2000 --interval-ms 0 --reset-adapt
```

### 5.2b Your own data files (any bar size) and starting money

CSV (header or MT4 `date,time,open,high,low,close,volume`), JSON list of `{time,open,high,low,close}`, or MT4 `.hst`. Time can be unix seconds or `YYYY-MM-DD HH:MM`. 5-minute, 10-minute, 1-hour, daily — whatever the file is.

```powershell
python fly_forex.py --replay --data C:\data\EURUSD_M10.csv --timeframe NATIVE --capital 25000 --interval-ms 0 --reset-adapt
python fly_forex.py --replay --data C:\data\EURUSD_M1.csv --timeframe M5 --capital 10000
python fly_forex.py --gui --capital 5000
```

`--timeframe NATIVE` (GUI: **file native**) keeps the file’s bar size. `--timeframe M10` / `H1` only **downsamples** if the file is finer. `--bars 0` (and GUI BARS 0) uses the whole file. `--capital` is an alias of `--balance` (default $100,000).

Same thing from the GUI: DATA file picker or path, TF, CAPITAL, Replay sim. Train DA uses the same file pickers and writes `brains/{PAIR}.json`.

### 5.3 Other pairs on the same dates

```powershell
python fly_forex.py --hst auto --bars 2000 --reset-adapt --interval-ms 0 --overfit
```

`--overfit` walks the majors on the EURUSD history window. `--holdout` walks the **7 days before** that window. Both are leakage checks, not live signals.

### 5.4 PairBook (opt-in, not default) and optional committees

Same 2oo3 voters on every pair. EURUSD stays the frozen factory book **only under `--legacy-2oo3`**. **PairBook is opt-in `--pair-book`**, not the default. Without that flag, non-EURUSD keeps the old wide-spread 3oo3-only gate. With `--pair-book` (and not `--no-pair-book`, and not a frozen symbol): wide-spread `2oo3+flow` or `3oo3`, Yahoo size ×0.50 until two wins, halt after one ATR stop or two chop scratches.

Eval did **not** all-green. `pair_sweep-20260920-193015.csv` was 0/7 profitable, mean `-$172.84`, EURUSD `-$219.70` (that sweep used `--no-plastic`, so it is **not** the freeze control). Freeze control without PairBook on EURUSD was still `+$1,906.66` **on the 3-voter**. Do **not** promote PairBook.

Nested 3×3oo3 then 2oo3 **lost** on that EURUSD week (`-$1,158`). Nine diverse flies 5oo9–9oo9 **lost** (`-$1,560` EURUSD; 6-pair mean `-$592`). Bayesian 2oo3/3oo3 ensemble **raised EURUSD to +$3,063** but the 7-pair mean was `-$284` vs freeze `+$308`. Those stay experiments.

```powershell
python fly_forex.py --replay --hst auto --bars 2000 --interval-ms 0 --reset-adapt --overfit
python fly_forex.py --replay --hst auto --bars 2000 --interval-ms 0 --reset-adapt --overfit --pair-book --use-brain --no-plastic
python fly_forex.py --replay --hst auto --bars 2000 --interval-ms 0 --reset-adapt --holdout --pair-book --use-brain --no-plastic
python fly_forex.py --gui --ensemble
python fly_forex.py --replay --hst auto --bars 2000 --interval-ms 0 --reset-adapt --koo9
python fly_forex.py --replay --hst auto --bars 2000 --interval-ms 0 --reset-adapt --nest
```

`--koo9` with `--gui` shows nine committee rows. Do not turn nest/koo9/ensemble on for the EURUSD freeze path.

### 5.5 Useful flags

| Flag | Effect |
|---|---|
| `--gui` | Browser lab, idle until Replay/Live |
| `--paper` | Prices from MT4, no `OrderSend` |
| `--replay --hst auto` | Offline `.hst` bars |
| `--data PATH` | Replay a CSV/JSON/HST file (`--csv` is an alias) |
| `--timeframe` | M1, M5, M10, M15, H1, H4, D1, or NATIVE |
| `--capital` / `--balance` | Simulated starting money (default 100000) |
| `--bars` | Bars from the end of the series; **0 = all** on `--data` |
| `--reset-adapt` | Ignore saved confidence / trail state |
| `--pair-book` | Opt-in PairBook on non-EURUSD (default off; frozen EURUSD never uses it) |
| `--no-pair-book` | Force PairBook off even if `--pair-book` is also passed |
| `--holdout` | 7-pair sweep on the week *before* the EURUSD eval window |
| `--no-snowball` | No pyramid adds |
| `--no-martingale` | No scratch recovery step-up |
| `--no-fusion` | Skip category inhibit (and skip the old fusion gate on `--legacy-2oo3`) |
| `--legacy-2oo3` | Old 3-voter 2oo3 {tech, fly-trend, fly-fade} — EURUSD freeze |
| `--no-categories` | Alias of `--legacy-2oo3` |
| `--cat-inhibit` | Category \|conf\|/τ bar (default 0.50) |
| `--use-brain` | Load `brains/{PAIR}.json` PAM/PPL (EURUSD never loads a trained brain unless `--force-brain`) |
| `--use-evo` | Load `brains/{PAIR}.evo.json` evolved params + dynamic flags (Replay). Independent of DA. |
| `--save-brain` | Write `brains/{PAIR}.json` at the end of the loop (GUI Train DA sets this) |
| `--force-brain` | Allow save/load of frozen EURUSD DA |
| `--shadow-da` | Dopamine from forward R on replay bars (GUI Train DA) |
| `--evolve` | Genetic Train DA (GUI Train DA defaults this on; `fly_train.py` is opt-in) |
| `--evo-pop` / `--evo-gens` / `--evo-seed` | Population (3–16), generations (2–12), RNG seed |
| `--symbol GBPUSD` | CLI default pair (GUI PAIR menu overrides this) |
| `--interval-ms 0` | Full-speed sim (default for replay/GUI) |

```powershell
python fly_forex.py -h
```

prints every flag.

---

## 6. Training other pairs (EURUSD stays frozen)

EURUSD trains from the GUI like any pair. Other pairs can update `brains/{SYMBOL}.json` from **Train DA** or `python fly_train.py`. Evolve also writes `brains/{SYMBOL}.evo.json`. Replay loads traces with **use DA brain** / `--use-brain` and loads the evolved policy with **use evolved** / `--use-evo`. `--reset-brain` ignores DA files, not the `.evo.json`. PairBook does **not** auto-load brains.

Walk-forward train (history *before* the EURUSD eval week; never trains on the eval window):

```powershell
python fly_train.py
python fly_train.py --evolve --evo-pop 6 --evo-gens 4 --pairs GBPUSD
```

`python fly_train.py --eval` trains, then replays the 7-pair eval window with `--overfit --use-brain --no-plastic` (PairBook stays **off** unless you add `--pair-book`; EURUSD stays factory). The 20 Sep eval did not all-green (0/7, mean `-$172.84`); do not promote PairBook to the default.

Last 7 days only:

```powershell
python fly_train.py --weekly
```

Sunday helper (weekly train, then eval with `--overfit --use-brain --no-plastic --reset-adapt`):

```powershell
powershell -File path\to\FlyFOREXTrader\scripts\weekly_train.ps1
```

---

## 7. Reports

After a replay or live session, look in `reports/`:

- `trades-{PAIR}-YYYYMMDD-HHMMSS.csv` — every fill with costs
- `evolve-{PAIR}-….json` — genetic Train DA log (fitness per generation, surviving genome)
- `brains/{PAIR}.evo.json` — selectable evolved params + dynamic flags (**use evolved**)
- `brains/{PAIR}.json` — PAM/PPL traces (**use DA brain**)
- `pair_sweep-….csv` — `--overfit` summary
- `adapt_{PAIR}.json` — evolving bar / trail (use `--reset-adapt` for a clean eval)

The terminal also prints a cost-included P&L table when the loop ends.

---

## 8. Troubleshooting

| Symptom | What to do |
|---|---|
| Browser stuck on “waiting for trader” | Start `python fly_forex.py --gui` first; hard-refresh the tab. |
| `could not bind dashboard port` | Kill the other `--gui` Python process. |
| Replay says no `.hst` | Open that pair in MT4 once (or pick a pair that already has `…/history/…/EURUSD5.hst`). GUI Replay can also use Yahoo 5m when HST is missing. |
| Live: MT4 not connected | EA on chart, DLL imports, `FlyTrader: ready`, pair in Market Watch. Then `--ping`. |
| GUI Replay is slow | Full-speed sim batches all live connectome heads and skips unchanged bars. 2000 M5 bars should be tens of seconds, not minutes. `--koo9` is still heavier (9 flies). |
| Train DA refused on EURUSD | GUI Train DA now writes EURUSD like any pair. `python fly_train.py` still needs `--include-eurusd`. |
| Volume row always empty | Yahoo FX volume is often 0. HST tick volume from MT4 can wake that category. Breadth/sentiment stay empty unless those overlays are in the bar. |
| Train DA is slow | **evolve** runs ~pop×gens replays (default 6×4). Uncheck evolve for one DA pass, or drop pop/gens. |
| Results differ from +$1906 / 4k+ | Default RISK is **scalp** (4k companion, VOL 12%). Freeze book is CLI `--legacy-2oo3 --no-sugar --risk-tol balanced` (~+$1,906 / 4 trades, 9–18 Sep 2026 M5, risk-% sizing). |
| `ModuleNotFoundError: flyfx` | `cd` into the project folder (the one that contains `fly_forex.py`) before `python fly_forex.py`. |

---

## 9. What not to do

- Do not edit files under `snapshots/` as if they were live code. Those copies freeze a past book. Sugar/DA before per-fly crops: `snapshots/pre-fly-crops-20260921/` (README has copy-back steps). `--fly-crops` off is the in-code fallback.
- Do not treat a default (cat-fuse) replay as the EURUSD freeze; that freeze is CLI `--legacy-2oo3`.
- Do not enable `--koo9` or `--nest` as the daily default; they lost money on the EURUSD hold-out week.
- Do not enable `--pair-book` as the daily default; the 20 Sep eval was 0/7 profitable.
- Do not train EURUSD from `fly_train.py --include-eurusd` unless you intend to overwrite the factory walk-forward skip. GUI Train DA already writes EURUSD.
- Do not treat replay P&L as a live promise. Spread, commission, and weekend gaps on demo still differ from a backtest.
