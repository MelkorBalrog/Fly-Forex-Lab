#property strict
#property copyright "FlyFOREXTrader"
#property link      ""
#property version   "1.31"
#property description "ZMQ bridge for the Fly connectome trader. Lab Inputs match the HTML GUI dials. Enable Allow DLL imports and Allow live trading."

#include <Zmq/Zmq.mqh>

input string InpRepEndpoint  = "tcp://*:5555";
input string InpPushEndpoint = "tcp://*:5556";
input int    InpMagic        = 20260919;
input int    InpSlippage     = 30;
input int    InpTimerSec     = 1;

input string InpRiskHdr      = "==== Risk tolerance (closes) ===="; // header
input string InpRiskTol      = "scalp"; // conservative | balanced | aggressive | scalp

input string InpVolHdr       = "==== Volume / bank / appetite ===="; // header
input double InpVolumePct    = -1.0;  // VOL % of available money (-1 = auto / RISK profile)
input string InpVolumeMode   = "auto"; // equity | fixed | auto
input double InpBankPct      = 0.0;   // BANK % of CAPITAL (0=off)
input bool   InpHoldRisk     = true;  // hold-risk RoC early bank / giveback
input double InpHoldRiskSens = 1.0;   // hold-risk sensitivity 0..2
input bool   InpProfitRecycle = true; // 1.5× latest win, spike close, reflex ANNs
input bool   InpTradeBayes   = true;  // Bayesian hx inhibit / size / trim
input bool   InpEntryAppetite = true; // dynamic entries
input double InpTradeRate    = 8.0;   // TRADE/1k target when appetite on

input string InpSugarHdr     = "==== Sugar feed ===="; // header
input bool   InpSugar        = true;  // Sugar feed (eat / don't eat; never votes a side)
input double InpSugarAmt     = 1.0;   // Sugar amount 0..2 (1=full, 2=extra; scale when dynamic)
input bool   InpSugarDyn     = false; // Dynamic sugar amount (FEAST/FORAGE/NIBBLE/FAST)
input bool   InpFlyCrops     = false; // Per-fly sugar tanks (requires sugar)

input string InpNnHdr        = "==== AI / brains ===="; // header
input bool   InpNnVote       = true;  // AI 3oo4 tensor voter
input bool   InpNnVision     = true;  // chart CNN vision features
input bool   InpNnFresh      = false; // Fresh ANN (scratch retrain, no transfer)
input bool   InpNnAdv        = true;  // adversarial train for the 10-net voter
input bool   InpSettingsAnn  = true;  // train 20 ANNs (settings, not buy/sell)
input bool   InpFibTrade     = true;  // Fibonacci trend entries (38.2/50/61.8/78.6)
input bool   InpEvolve       = true;  // evolve during Train DA
input bool   InpUseBrain     = false; // load brains/{PAIR}.json DA traces
input bool   InpUseEvo       = false; // load brains/{PAIR}.evo.json
input bool   InpContinueDa   = true;  // continue DA + apply risk_factors live

input string InpFuseHdr      = "==== Fuse mix (3-voter) ===="; // header
input bool   InpFuseDynamic    = true;  // Dynamic mix (TREND/RANGE/BREAK/QUIET)
input bool   InpFuseTrend      = true;  // Fuse trend (allowlist)
input bool   InpFuseMomentum   = true;  // Fuse momentum (allowlist)
input bool   InpFuseVolatility = true;  // Fuse volatility (allowlist)
input bool   InpFuseVolume     = true;  // Fuse volume (allowlist)
input bool   InpFuseBreadth    = true;  // Fuse breadth (allowlist)
input bool   InpFuseStructure  = true;  // Fuse structure (allowlist)
input bool   InpFuseSentiment  = true;  // Fuse sentiment (allowlist)

input string InpIndTrendHdr    = "==== Trend indicators ===="; // header
input bool   InpIndSma         = false; // SMA
input bool   InpIndEma         = false; // EMA
input bool   InpIndWma         = false; // WMA
input bool   InpIndMacd        = true;  // MACD
input bool   InpIndAdx         = false; // ADX
input bool   InpIndSar         = true;  // Parabolic SAR
input bool   InpIndIchimoku    = true;  // Ichimoku
input bool   InpIndSupertrend  = true;  // Supertrend

input string InpIndMomHdr      = "==== Momentum indicators ===="; // header
input bool   InpIndRsi         = true;  // RSI
input bool   InpIndStoch       = true;  // Stochastic
input bool   InpIndCci         = false; // CCI
input bool   InpIndWillr       = false; // Williams %R
input bool   InpIndRoc         = false; // ROC

input string InpIndVolHdr      = "==== Volatility indicators ===="; // header
input bool   InpIndBb          = true;  // Bollinger
input bool   InpIndAtr         = false; // ATR
input bool   InpIndKeltner     = true;  // Keltner
input bool   InpIndStdev       = false; // StdDev
input bool   InpIndDonchian    = false; // Donchian

input string InpIndFlowHdr     = "==== Volume indicators ===="; // header
input bool   InpIndObv         = true;  // OBV
input bool   InpIndCmf         = false; // Chaikin MF
input bool   InpIndVwap        = true;  // VWAP
input bool   InpIndAd          = false; // A/D
input bool   InpIndProfile     = false; // Volume profile (needs ticks)

input string InpIndBrHdr       = "==== Breadth / structure / sentiment ===="; // header
input bool   InpIndAdLine      = true;  // A/D line
input bool   InpIndMcclellan   = false; // McClellan
input bool   InpIndTrin        = false; // TRIN
input bool   InpIndPivot       = true;  // Pivots
input bool   InpIndFib         = true;  // Fibonacci
input bool   InpIndChannel     = false; // Channel
input bool   InpIndPutCall     = true;  // Put/call
input bool   InpIndVix         = false; // VIX
input bool   InpIndCot         = false; // COT

// Chart buttons edit these. Inputs seed them when the EA attaches.
string gRisk = "scalp";
string gVolMode = "auto";
double gVolPct = -1.0;
double gBank = 0.0;
double gHoldSens = 1.0;
double gTradeRate = 8.0;
double gSugarAmt = 1.0;
bool   gHold = true;
bool   gRecycle = true;
bool   gBayes = true;
bool   gApp = true;
bool   gNn = true;
bool   gVision = true;
bool   gFresh = false;
bool   gAdv = true;
bool   gSetAnn = true;
bool   gFibTrade = true;
bool   gEvolve = true;
bool   gSugar = true;
bool   gSugarDyn = false;
bool   gCrops = false;
bool   gBrain = false;
bool   gEvo = false;
bool   gCda = true;
bool   gFuseDyn = true;
bool   gFuseTrend = true;
bool   gFuseMomentum = true;
bool   gFuseVolatility = true;
bool   gFuseVolume = true;
bool   gFuseBreadth = true;
bool   gFuseStructure = true;
bool   gFuseSentiment = true;
string gIndName[32];
bool   gIndOn[32];
int    gUiPage = 0;

// Do NOT construct Context/Socket at global scope. Their constructors call
// libzmq.dll during EA load and will crash MT4 if the DLL is not ready.
class FlyZmq
  {
public:
   Context ctx;
   Socket *rep;
   Socket *push;

                     FlyZmq(): ctx()
     {
      rep  = new Socket(ctx, ZMQ_REP);
      push = new Socket(ctx, ZMQ_PUSH);
     }
                    ~FlyZmq()
     {
      if(push != NULL) { delete push; push = NULL; }
      if(rep  != NULL) { delete rep;  rep  = NULL; }
     }

   bool              ok() const
     {
      return(rep != NULL && push != NULL && rep.valid() && push.valid());
     }
  };

FlyZmq *g_zmq = NULL;

string HandleCommand(string cmd);
void   DrainRequests();
void   PublishTick();
void   PaintStatus();
void   ReleaseZmq();
string FuseCatsSpec();
string FuseDynamicSpec();
string FuseIndsSpec();
string FuseRiskSpec();
string FuseSugarSpec();
string FuseSugarAmtSpec();
string FuseSugarDynSpec();
string FuseVolumePctSpec();
string FuseVolumeModeSpec();
string FuseBankSpec();
string FuseHoldSpec();
string FuseHoldSensSpec();
string FuseRecycleSpec();
string FuseBayesSpec();
string FuseAppSpec();
string FuseTradeRateSpec();
string FuseNnSpec();
string FuseVisionSpec();
string FuseCropsSpec();
string FuseBrainSpec();
string FuseEvoSpec();
string FuseCdaSpec();
string LabSettingsTail();
void   SeedLab();
void   InitIndNames();
void   BuildLabPanel();
void   UiButton(string id, string text, int x, int y, int w, int h, bool on);
bool   ToggleLab(string id);
double NormPrice(string symbol, double price);
double NormLots(string symbol, double lots);
string OpenTrade(int type, string symbol, double lots, double sl, double tp);
string CloseTicket(int ticket);
string CloseAll();
string ListPositions();
string ListHistory(string symbol, int tf, int count);

int OnInit()
  {
   Print("FlyTrader: OnInit. If MT4 dies after this line, libzmq.dll is the wrong build.");
   Print("FlyTrader: ZMQ ", Zmq::getVersion());

   ReleaseZmq();
   g_zmq = new FlyZmq();
   if(g_zmq == NULL || !g_zmq.ok())
     {
      Print("FlyTrader: socket create failed. Tools > Options > Expert Advisors > Allow DLL imports. ",
            Zmq::errorMessage());
      ReleaseZmq();
      return(INIT_FAILED);
     }

   g_zmq.rep.setLinger(0);
   g_zmq.push.setLinger(0);
   g_zmq.push.setSendHighWaterMark(1);
   g_zmq.push.setConflate(true);

   if(!g_zmq.rep.bind(InpRepEndpoint))
     {
      Print("FlyTrader: REP bind failed on ", InpRepEndpoint, ": ", Zmq::errorMessage());
      ReleaseZmq();
      return(INIT_FAILED);
     }
   if(!g_zmq.push.bind(InpPushEndpoint))
     {
      Print("FlyTrader: PUSH bind failed on ", InpPushEndpoint, ": ", Zmq::errorMessage());
      ReleaseZmq();
      return(INIT_FAILED);
     }

   EventSetTimer(MathMax(1, InpTimerSec));
   SeedLab();
   Print("FlyTrader: ready  REP=", InpRepEndpoint, "  PUSH=", InpPushEndpoint,
         "  fuse=", FuseCatsSpec(), "  dynamic=", FuseDynamicSpec(),
         "  inds=", FuseIndsSpec(), "  risk=", FuseRiskSpec(),
         "  sugar=", FuseSugarSpec(), "  samt=", FuseSugarAmtSpec(),
         "  sdyn=", FuseSugarDynSpec(),
         "  vol=", FuseVolumePctSpec(), "/", FuseVolumeModeSpec(),
         "  nn=", FuseNnSpec(), "  vision=", FuseVisionSpec());
   PaintStatus();
   BuildLabPanel();
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
   ObjectsDeleteAll(0, "FlyUi_");
   Comment("");
   ReleaseZmq();
   Print("FlyTrader: stopped, reason=", reason);
  }

void ReleaseZmq()
  {
   if(g_zmq == NULL)
      return;
   if(g_zmq.rep != NULL)
      g_zmq.rep.unbind(InpRepEndpoint);
   if(g_zmq.push != NULL)
      g_zmq.push.unbind(InpPushEndpoint);
   delete g_zmq;
   g_zmq = NULL;
  }

void OnTick()
  {
   PublishTick();
   DrainRequests();
   PaintStatus();
  }

void OnTimer()
  {
   DrainRequests();
  }

void PaintStatus()
  {
   Comment("FlyTrader  ", Symbol(), "\n",
           "fuse: ", FuseCatsSpec(), "  dyn ", FuseDynamicSpec(), "\n",
           "inds: ", FuseIndsSpec(), "\n",
           "risk: ", FuseRiskSpec(),
           "  vol ", FuseVolumePctSpec(), "/", FuseVolumeModeSpec(), "\n",
           "bank ", FuseBankSpec(), "%  hold ", FuseHoldSpec(), "×", FuseHoldSensSpec(),
           "  recycle ", FuseRecycleSpec(), "  bayes ", FuseBayesSpec(), "\n",
           "appetite ", FuseAppSpec(), "  trate ", FuseTradeRateSpec(),
           "  nn ", FuseNnSpec(), "  vision ", FuseVisionSpec(), "\n",
           "sugar: ", FuseSugarSpec(), "  amt ", FuseSugarAmtSpec(),
           "  dyn ", FuseSugarDynSpec(), "  crops ", FuseCropsSpec(), "\n",
           "brain ", FuseBrainSpec(), "  evo ", FuseEvoSpec(),
           "  continueDA ", FuseCdaSpec(), "\n",
           "setann ", (gSetAnn ? "1" : "0"), "  fib ", (gFibTrade ? "1" : "0"),
           "  fresh ", (gFresh ? "1" : "0"), "  adv ", (gAdv ? "1" : "0"), "\n",
           "Chart buttons = the lab. Click Lab / Fuse / Indicators.");
  }

void InitIndNames()
  {
   gIndName[0]  = "sma";
   gIndName[1]  = "ema";
   gIndName[2]  = "wma";
   gIndName[3]  = "macd";
   gIndName[4]  = "adx";
   gIndName[5]  = "sar";
   gIndName[6]  = "ichimoku";
   gIndName[7]  = "supertrend";
   gIndName[8]  = "rsi";
   gIndName[9]  = "stoch";
   gIndName[10] = "cci";
   gIndName[11] = "willr";
   gIndName[12] = "roc";
   gIndName[13] = "bb";
   gIndName[14] = "atr";
   gIndName[15] = "keltner";
   gIndName[16] = "stdev";
   gIndName[17] = "donchian";
   gIndName[18] = "obv";
   gIndName[19] = "cmf";
   gIndName[20] = "vwap";
   gIndName[21] = "ad";
   gIndName[22] = "profile";
   gIndName[23] = "ad_line";
   gIndName[24] = "mcclellan";
   gIndName[25] = "trin";
   gIndName[26] = "pivot";
   gIndName[27] = "fib";
   gIndName[28] = "channel";
   gIndName[29] = "put_call";
   gIndName[30] = "vix";
   gIndName[31] = "cot";
  }

void SeedLab()
  {
   InitIndNames();
   gRisk = InpRiskTol;
   StringToLower(gRisk);
   StringTrimLeft(gRisk);
   StringTrimRight(gRisk);
   if(gRisk == "conservative" || gRisk == "cons" || gRisk == "tight" || gRisk == "low" || gRisk == "safe")
      gRisk = "conservative";
   else if(gRisk == "aggressive" || gRisk == "agg" || gRisk == "high" || gRisk == "loose")
      gRisk = "aggressive";
   else if(gRisk == "scalp" || gRisk == "scalping" || gRisk == "fast" || gRisk == "m5-scalp" || gRisk == "m5scalp")
      gRisk = "scalp";
   else
      gRisk = "balanced";
   gVolMode = InpVolumeMode;
   StringToLower(gVolMode);
   StringTrimLeft(gVolMode);
   StringTrimRight(gVolMode);
   if(gVolMode != "fixed" && gVolMode != "equity" && gVolMode != "auto")
      gVolMode = "auto";
   gVolPct = InpVolumePct;
   gBank = InpBankPct;
   gHoldSens = InpHoldRiskSens;
   gTradeRate = InpTradeRate;
   gSugarAmt = InpSugarAmt;
   gHold = InpHoldRisk;
   gRecycle = InpProfitRecycle;
   gBayes = InpTradeBayes;
   gApp = InpEntryAppetite;
   gNn = InpNnVote;
   gVision = InpNnVision;
   gFresh = InpNnFresh;
   gAdv = InpNnAdv;
   gSetAnn = InpSettingsAnn;
   gFibTrade = InpFibTrade;
   gEvolve = InpEvolve;
   gSugar = InpSugar;
   gSugarDyn = InpSugarDyn;
   gCrops = InpFlyCrops;
   gBrain = InpUseBrain;
   gEvo = InpUseEvo;
   gCda = InpContinueDa;
   gFuseDyn = InpFuseDynamic;
   gFuseTrend = InpFuseTrend;
   gFuseMomentum = InpFuseMomentum;
   gFuseVolatility = InpFuseVolatility;
   gFuseVolume = InpFuseVolume;
   gFuseBreadth = InpFuseBreadth;
   gFuseStructure = InpFuseStructure;
   gFuseSentiment = InpFuseSentiment;
   gIndOn[0]  = InpIndSma;
   gIndOn[1]  = InpIndEma;
   gIndOn[2]  = InpIndWma;
   gIndOn[3]  = InpIndMacd;
   gIndOn[4]  = InpIndAdx;
   gIndOn[5]  = InpIndSar;
   gIndOn[6]  = InpIndIchimoku;
   gIndOn[7]  = InpIndSupertrend;
   gIndOn[8]  = InpIndRsi;
   gIndOn[9]  = InpIndStoch;
   gIndOn[10] = InpIndCci;
   gIndOn[11] = InpIndWillr;
   gIndOn[12] = InpIndRoc;
   gIndOn[13] = InpIndBb;
   gIndOn[14] = InpIndAtr;
   gIndOn[15] = InpIndKeltner;
   gIndOn[16] = InpIndStdev;
   gIndOn[17] = InpIndDonchian;
   gIndOn[18] = InpIndObv;
   gIndOn[19] = InpIndCmf;
   gIndOn[20] = InpIndVwap;
   gIndOn[21] = InpIndAd;
   gIndOn[22] = InpIndProfile;
   gIndOn[23] = InpIndAdLine;
   gIndOn[24] = InpIndMcclellan;
   gIndOn[25] = InpIndTrin;
   gIndOn[26] = InpIndPivot;
   gIndOn[27] = InpIndFib;
   gIndOn[28] = InpIndChannel;
   gIndOn[29] = InpIndPutCall;
   gIndOn[30] = InpIndVix;
   gIndOn[31] = InpIndCot;
  }

string FuseCatsSpec()
  {
   string names[7];
   int n = 0;
   if(gFuseTrend)      names[n++] = "trend";
   if(gFuseMomentum)   names[n++] = "momentum";
   if(gFuseVolatility) names[n++] = "volatility";
   if(gFuseVolume)     names[n++] = "volume";
   if(gFuseBreadth)    names[n++] = "breadth";
   if(gFuseStructure)  names[n++] = "structure";
   if(gFuseSentiment)  names[n++] = "sentiment";
   if(n <= 0 || n >= 7)
      return("all");
   string out = names[0];
   int i;
   for(i=1; i<n; i++)
      out = out + "," + names[i];
   return(out);
  }

string FuseDynamicSpec()
  {
   return(gFuseDyn ? "1" : "0");
  }

string FuseIndsSpec()
  {
   string names[32];
   int n = 0;
   int i;
   for(i = 0; i < 32; i++)
     {
      if(gIndOn[i])
         names[n++] = gIndName[i];
     }
   if(n <= 0)
      return("none");
   if(n >= 32)
      return("all");
   string out = names[0];
   for(i = 1; i < n; i++)
      out = out + "," + names[i];
   return(out);
  }

string FuseRiskSpec()
  {
   string t = gRisk;
   StringToLower(t);
   StringTrimLeft(t);
   StringTrimRight(t);
   if(t == "conservative" || t == "cons" || t == "tight" || t == "low" || t == "safe")
      return("conservative");
   if(t == "aggressive" || t == "agg" || t == "high" || t == "loose")
      return("aggressive");
   if(t == "scalp" || t == "scalping" || t == "fast" || t == "m5-scalp" || t == "m5scalp")
      return("scalp");
   return("balanced");
  }

string FuseSugarSpec()
  {
   return(gSugar ? "1" : "0");
  }

string FuseSugarAmtSpec()
  {
   double a = gSugarAmt;
   if(a >= 10.0 && a <= 200.0000001)
      a = a / 100.0;
   if(a < 0.0)
      a = 0.0;
   if(a > 2.0)
      a = 2.0;
   return(DoubleToString(a, 2));
  }

string FuseSugarDynSpec()
  {
   return(gSugarDyn ? "1" : "0");
  }

string FuseVolumePctSpec()
  {
   if(gVolPct < 0.0)
      return("auto");
   double a = gVolPct;
   if(a > 100.0)
      a = 100.0;
   return(DoubleToString(a, 2));
  }

string FuseVolumeModeSpec()
  {
   string t = gVolMode;
   StringToLower(t);
   StringTrimLeft(t);
   StringTrimRight(t);
   if(t == "fixed" || t == "equity" || t == "auto")
      return(t);
   return("auto");
  }

string FuseBankSpec()
  {
   double a = gBank;
   if(a < 0.0)
      a = 0.0;
   if(a > 50.0)
      a = 50.0;
   return(DoubleToString(a, 2));
  }

string FuseHoldSpec()
  {
   return(gHold ? "1" : "0");
  }

string FuseHoldSensSpec()
  {
   double a = gHoldSens;
   if(a < 0.0)
      a = 0.0;
   if(a > 2.0)
      a = 2.0;
   return(DoubleToString(a, 2));
  }

string FuseRecycleSpec()
  {
   return(gRecycle ? "1" : "0");
  }

string FuseBayesSpec()
  {
   return(gBayes ? "1" : "0");
  }

string FuseAppSpec()
  {
   return(gApp ? "1" : "0");
  }

string FuseTradeRateSpec()
  {
   double a = gTradeRate;
   if(a < 0.0)
      a = 0.0;
   if(a > 40.0)
      a = 40.0;
   return(DoubleToString(a, 1));
  }

string FuseNnSpec()
  {
   return(gNn ? "1" : "0");
  }

string FuseVisionSpec()
  {
   return(gVision ? "1" : "0");
  }

string FuseFreshSpec()
  {
   return(gFresh ? "1" : "0");
  }

string FuseAdvSpec()
  {
   return(gAdv ? "1" : "0");
  }

string FuseSetAnnSpec()
  {
   return(gSetAnn ? "1" : "0");
  }

string FuseFibTradeSpec()
  {
   return(gFibTrade ? "1" : "0");
  }

string FuseEvolveSpec()
  {
   return(gEvolve ? "1" : "0");
  }

string FuseCropsSpec()
  {
   return(gCrops ? "1" : "0");
  }

string FuseBrainSpec()
  {
   return(gBrain ? "1" : "0");
  }

string FuseEvoSpec()
  {
   return(gEvo ? "1" : "0");
  }

string FuseCdaSpec()
  {
   return(gCda ? "1" : "0");
  }

string LabSettingsTail()
  {
   return("|VPCT|" + FuseVolumePctSpec() +
          "|VMODE|" + FuseVolumeModeSpec() +
          "|BANK|" + FuseBankSpec() +
          "|HOLD|" + FuseHoldSpec() +
          "|HOLDS|" + FuseHoldSensSpec() +
          "|RECYCLE|" + FuseRecycleSpec() +
          "|BAYES|" + FuseBayesSpec() +
          "|APP|" + FuseAppSpec() +
          "|TRATE|" + FuseTradeRateSpec() +
          "|NN|" + FuseNnSpec() +
          "|VISION|" + FuseVisionSpec() +
          "|FRESH|" + FuseFreshSpec() +
          "|ADV|" + FuseAdvSpec() +
          "|SETANN|" + FuseSetAnnSpec() +
          "|FIBTR|" + FuseFibTradeSpec() +
          "|EVOLVE|" + FuseEvolveSpec() +
          "|CROPS|" + FuseCropsSpec() +
          "|BRAIN|" + FuseBrainSpec() +
          "|EVO|" + FuseEvoSpec() +
          "|CDA|" + FuseCdaSpec());
  }

void UiButton(string id, string text, int x, int y, int w, int h, bool on)
  {
   string name = "FlyUi_" + id;
   ObjectCreate(0, name, OBJ_BUTTON, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   ObjectSetInteger(0, name, OBJPROP_XSIZE, w);
   ObjectSetInteger(0, name, OBJPROP_YSIZE, h);
   ObjectSetString(0, name, OBJPROP_TEXT, text);
   ObjectSetInteger(0, name, OBJPROP_BGCOLOR, on ? C'36,110,72' : C'48,52,60');
   ObjectSetInteger(0, name, OBJPROP_COLOR, clrWhite);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, 8);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
   ObjectSetInteger(0, name, OBJPROP_STATE, false);
  }

void UiLabel(string id, string text, int x, int y)
  {
   string name = "FlyUi_" + id;
   ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   ObjectSetString(0, name, OBJPROP_TEXT, text);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clrWhite);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, 8);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
  }

void BuildLabPanel()
  {
   ObjectsDeleteAll(0, "FlyUi_");
   UiButton("page_lab", "Lab", 6, 4, 70, 16, gUiPage == 0);
   UiButton("page_fuse", "Fuse", 80, 4, 70, 16, gUiPage == 1);
   UiButton("page_ind", "Indicators", 154, 4, 90, 16, gUiPage == 2);
   int x0 = 6;
   int y0 = 24;
   int bw = 112;
   int bh = 16;
   int gap = 2;
   if(gUiPage == 0)
     {
      UiButton("risk", "RISK " + gRisk, x0, y0, bw, bh, true);
      UiButton("volmode", "VOL " + gVolMode, x0 + (bw + gap), y0, bw, bh, true);
      string vp = (gVolPct < 0.0) ? "auto" : DoubleToString(gVolPct, 0) + "%";
      UiButton("vpct_m", "VOL% -", x0 + 2 * (bw + gap), y0, 54, bh, false);
      UiLabel("vpct", vp, x0 + 2 * (bw + gap) + 58, y0 + 2);
      UiButton("vpct_p", "VOL% +", x0 + 2 * (bw + gap) + 96, y0, 54, bh, false);
      int y = y0 + bh + gap;
      UiButton("nn", "AI 3oo4", x0, y, bw, bh, gNn);
      UiButton("vision", "chart vision", x0 + (bw + gap), y, bw, bh, gVision);
      UiButton("fresh", "Fresh ANN", x0 + 2 * (bw + gap), y, bw, bh, gFresh);
      UiButton("adv", "adv train", x0 + 3 * (bw + gap), y, bw, bh, gAdv);
      y += bh + gap;
      UiButton("setann", "train 20 ANNs", x0, y, bw, bh, gSetAnn);
      UiButton("fibtrade", "Fibonacci", x0 + (bw + gap), y, bw, bh, gFibTrade);
      UiButton("appetite", "dynamic entries", x0 + 2 * (bw + gap), y, bw, bh, gApp);
      UiButton("hold", "hold risk", x0 + 3 * (bw + gap), y, bw, bh, gHold);
      y += bh + gap;
      UiButton("recycle", "profit recycle", x0, y, bw, bh, gRecycle);
      UiButton("bayes", "trade bayes", x0 + (bw + gap), y, bw, bh, gBayes);
      UiButton("evolve", "evolve", x0 + 2 * (bw + gap), y, bw, bh, gEvolve);
      UiButton("brain", "use DA brain", x0 + 3 * (bw + gap), y, bw, bh, gBrain);
      y += bh + gap;
      UiButton("evo", "use evolved", x0, y, bw, bh, gEvo);
      UiButton("cda", "continue DA", x0 + (bw + gap), y, bw, bh, gCda);
      UiButton("sugar", "sugar feed", x0 + 2 * (bw + gap), y, bw, bh, gSugar);
      UiButton("sugardyn", "DYNAMIC sugar", x0 + 3 * (bw + gap), y, bw, bh, gSugarDyn);
      y += bh + gap;
      UiButton("crops", "per-fly crops", x0, y, bw, bh, gCrops);
      UiButton("trate_m", "rate -", x0 + (bw + gap), y, 54, bh, false);
      UiLabel("trate", "TRADE/1k " + DoubleToString(gTradeRate, 0), x0 + (bw + gap) + 58, y + 2);
      UiButton("trate_p", "rate +", x0 + 2 * (bw + gap), y, 54, bh, false);
      y += bh + gap;
      UiButton("bank_m", "bank -", x0, y, 54, bh, false);
      UiLabel("bank", "BANK " + DoubleToString(gBank, 2) + "%", x0 + 58, y + 2);
      UiButton("bank_p", "bank +", x0 + (bw + gap), y, 54, bh, false);
      UiButton("holds_m", "holdx -", x0 + 2 * (bw + gap), y, 54, bh, false);
      UiLabel("holds", "HOLD× " + DoubleToString(gHoldSens, 1), x0 + 2 * (bw + gap) + 58, y + 2);
      UiButton("holds_p", "holdx +", x0 + 3 * (bw + gap), y, 54, bh, false);
      y += bh + gap;
      UiButton("samt_m", "amt -", x0, y, 54, bh, false);
      UiLabel("samt", "sugar amt " + DoubleToString(gSugarAmt, 1), x0 + 58, y + 2);
      UiButton("samt_p", "amt +", x0 + (bw + gap), y, 54, bh, false);
     }
   else if(gUiPage == 1)
     {
      UiButton("fusedyn", "DYNAMIC mix", x0, y0, bw, bh, gFuseDyn);
      UiButton("ftrend", "trend", x0 + (bw + gap), y0, bw, bh, gFuseTrend);
      UiButton("fmom", "momentum", x0 + 2 * (bw + gap), y0, bw, bh, gFuseMomentum);
      UiButton("fvolat", "volatility", x0 + 3 * (bw + gap), y0, bw, bh, gFuseVolatility);
      int y = y0 + bh + gap;
      UiButton("fvolume", "volume", x0, y, bw, bh, gFuseVolume);
      UiButton("fbreadth", "breadth", x0 + (bw + gap), y, bw, bh, gFuseBreadth);
      UiButton("fstruct", "structure", x0 + 2 * (bw + gap), y, bw, bh, gFuseStructure);
      UiButton("fsent", "sentiment", x0 + 3 * (bw + gap), y, bw, bh, gFuseSentiment);
     }
   else
     {
      int col = 0;
      int row = 0;
      int i;
      for(i = 0; i < 32; i++)
        {
         int x = x0 + col * (bw + gap);
         int y = y0 + row * (bh + gap);
         UiButton("ind_" + gIndName[i], gIndName[i], x, y, bw, bh, gIndOn[i]);
         col++;
         if(col >= 4)
           {
            col = 0;
            row++;
           }
        }
     }
   ChartRedraw(0);
  }

bool ToggleLab(string id)
  {
   if(id == "page_lab")  { gUiPage = 0; return(true); }
   if(id == "page_fuse") { gUiPage = 1; return(true); }
   if(id == "page_ind")  { gUiPage = 2; return(true); }
   if(id == "risk")
     {
      if(gRisk == "scalp") gRisk = "balanced";
      else if(gRisk == "balanced") gRisk = "conservative";
      else if(gRisk == "conservative") gRisk = "aggressive";
      else gRisk = "scalp";
      return(true);
     }
   if(id == "volmode")
     {
      if(gVolMode == "auto") gVolMode = "equity";
      else if(gVolMode == "equity") gVolMode = "fixed";
      else gVolMode = "auto";
      return(true);
     }
   if(id == "vpct_m")
     {
      if(gVolPct < 0.0) gVolPct = 100.0;
      else gVolPct = gVolPct - 5.0;
      if(gVolPct < 0.0) gVolPct = -1.0;
      return(true);
     }
   if(id == "vpct_p")
     {
      if(gVolPct < 0.0) gVolPct = 5.0;
      else gVolPct = gVolPct + 5.0;
      if(gVolPct > 100.0) gVolPct = 100.0;
      return(true);
     }
   if(id == "trate_m") { gTradeRate = MathMax(1.0, gTradeRate - 1.0); return(true); }
   if(id == "trate_p") { gTradeRate = MathMin(40.0, gTradeRate + 1.0); return(true); }
   if(id == "bank_m")  { gBank = MathMax(0.0, gBank - 0.25); return(true); }
   if(id == "bank_p")  { gBank = MathMin(50.0, gBank + 0.25); return(true); }
   if(id == "holds_m") { gHoldSens = MathMax(0.0, gHoldSens - 0.1); return(true); }
   if(id == "holds_p") { gHoldSens = MathMin(2.0, gHoldSens + 0.1); return(true); }
   if(id == "samt_m")  { gSugarAmt = MathMax(0.0, gSugarAmt - 0.1); return(true); }
   if(id == "samt_p")  { gSugarAmt = MathMin(2.0, gSugarAmt + 0.1); return(true); }
   if(id == "nn")       { gNn = !gNn; return(true); }
   if(id == "vision")   { gVision = !gVision; return(true); }
   if(id == "fresh")    { gFresh = !gFresh; return(true); }
   if(id == "adv")      { gAdv = !gAdv; return(true); }
   if(id == "setann")   { gSetAnn = !gSetAnn; return(true); }
   if(id == "fibtrade") { gFibTrade = !gFibTrade; return(true); }
   if(id == "appetite") { gApp = !gApp; return(true); }
   if(id == "hold")     { gHold = !gHold; return(true); }
   if(id == "recycle")  { gRecycle = !gRecycle; return(true); }
   if(id == "bayes")    { gBayes = !gBayes; return(true); }
   if(id == "evolve")   { gEvolve = !gEvolve; return(true); }
   if(id == "brain")    { gBrain = !gBrain; return(true); }
   if(id == "evo")      { gEvo = !gEvo; return(true); }
   if(id == "cda")      { gCda = !gCda; return(true); }
   if(id == "sugar")    { gSugar = !gSugar; return(true); }
   if(id == "sugardyn") { gSugarDyn = !gSugarDyn; return(true); }
   if(id == "crops")    { gCrops = !gCrops; return(true); }
   if(id == "fusedyn")  { gFuseDyn = !gFuseDyn; return(true); }
   if(id == "ftrend")   { gFuseTrend = !gFuseTrend; return(true); }
   if(id == "fmom")     { gFuseMomentum = !gFuseMomentum; return(true); }
   if(id == "fvolat")   { gFuseVolatility = !gFuseVolatility; return(true); }
   if(id == "fvolume")  { gFuseVolume = !gFuseVolume; return(true); }
   if(id == "fbreadth") { gFuseBreadth = !gFuseBreadth; return(true); }
   if(id == "fstruct")  { gFuseStructure = !gFuseStructure; return(true); }
   if(id == "fsent")    { gFuseSentiment = !gFuseSentiment; return(true); }
   if(StringFind(id, "ind_") == 0)
     {
      string key = StringSubstr(id, 4);
      int i;
      for(i = 0; i < 32; i++)
        {
         if(gIndName[i] == key)
           {
            gIndOn[i] = !gIndOn[i];
            return(true);
           }
        }
     }
   return(false);
  }

void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
  {
   if(id != CHARTEVENT_OBJECT_CLICK)
      return;
   if(StringFind(sparam, "FlyUi_") != 0)
      return;
   string key = StringSubstr(sparam, 6);
   if(ToggleLab(key))
     {
      BuildLabPanel();
      PaintStatus();
     }
  }

void DrainRequests()
  {
   if(g_zmq == NULL || g_zmq.rep == NULL)
      return;

   for(int i=0; i<64; i++)
     {
      ZmqMsg request;
      if(!g_zmq.rep.recv(request, true))
         break;

      string cmd = request.getData();
      StringTrimLeft(cmd);
      StringTrimRight(cmd);
      string reply = HandleCommand(cmd);
      if(!g_zmq.rep.send(reply))
         Print("FlyTrader: REP send failed: ", Zmq::errorMessage());
     }
  }

void PublishTick()
  {
   if(g_zmq == NULL || g_zmq.push == NULL)
      return;
   string msg = StringFormat("TICK|%s|%s|%s|%d",
                             Symbol(),
                             DoubleToString(Bid, Digits),
                             DoubleToString(Ask, Digits),
                             (int)TimeCurrent());
   g_zmq.push.send(msg, true);
  }

string HandleCommand(string cmd)
  {
   string parts[];
   int n = StringSplit(cmd, '|', parts);
   if(n < 1)
      return("UNKNOWN");

   string op = parts[0];

   if(op == "PING")
      return("PONG|" + Symbol() + "|" + IntegerToString((int)TimeCurrent()) +
             "|FUSE|" + FuseCatsSpec() + "|DYNAMIC|" + FuseDynamicSpec() +
             "|INDS|" + FuseIndsSpec() + "|RISK|" + FuseRiskSpec() +
             "|SUGAR|" + FuseSugarSpec() + "|SAMT|" + FuseSugarAmtSpec() +
             "|SDYN|" + FuseSugarDynSpec() + LabSettingsTail());

   if(op == "SETTINGS")
      return("OK|FUSE|" + FuseCatsSpec() + "|DYNAMIC|" + FuseDynamicSpec() +
             "|INDS|" + FuseIndsSpec() + "|RISK|" + FuseRiskSpec() +
             "|SUGAR|" + FuseSugarSpec() + "|SAMT|" + FuseSugarAmtSpec() +
             "|SDYN|" + FuseSugarDynSpec() + LabSettingsTail() +
             "|SYMBOL|" + Symbol() + "|MAGIC|" + IntegerToString(InpMagic));

   if(op == "RATES")
     {
      string symbol = (n >= 2 && StringLen(parts[1]) > 0) ? parts[1] : Symbol();
      if(!SymbolSelect(symbol, true))
         return("ERR|UNKNOWN_SYMBOL|" + symbol);
      RefreshRates();
      int digits = (int)MarketInfo(symbol, MODE_DIGITS);
      double bid = MarketInfo(symbol, MODE_BID);
      double ask = MarketInfo(symbol, MODE_ASK);
      if(bid <= 0.0 || ask <= 0.0)
        {
         double last = iClose(symbol, PERIOD_M1, 1);
         if(last <= 0.0)
            last = iClose(symbol, PERIOD_H1, 1);
         if(last <= 0.0)
            return("ERR|NO_QUOTES|" + symbol);
         double point = MarketInfo(symbol, MODE_POINT);
         if(point <= 0.0)
            point = 0.0001;
         bid = last;
         ask = last + 20.0 * point;
        }
      return(symbol + "|" + DoubleToString(bid, digits) + "|" + DoubleToString(ask, digits));
     }

   if(op == "HISTORY")
     {
      string symbol = (n >= 2 && StringLen(parts[1]) > 0) ? parts[1] : Symbol();
      int tf    = (n >= 3) ? (int)StringToInteger(parts[2]) : PERIOD_M5;
      int count = (n >= 4) ? (int)StringToInteger(parts[3]) : 200;
      return(ListHistory(symbol, tf, count));
     }

   if(op == "ACCOUNT")
      return(StringFormat("OK|%.2f|%.2f|%.2f|%d",
                          AccountBalance(), AccountEquity(), AccountMargin(), AccountLeverage()));

   if(op == "POSITIONS")
      return(ListPositions());

   if(op == "TRADE")
     {
      if(n < 7)
         return("ERR|USAGE|TRADE|OPEN|type|symbol|lots|sl|tp");
      int    type   = (int)StringToInteger(parts[2]);
      string symbol = parts[3];
      double lots   = StringToDouble(parts[4]);
      double sl     = StringToDouble(parts[5]);
      double tp     = StringToDouble(parts[6]);
      return(OpenTrade(type, symbol, lots, sl, tp));
     }

   if(op == "CLOSE")
     {
      if(n < 2)
         return("ERR|USAGE|CLOSE|ticket[|lots]");
      if(parts[1] == "ALL")
         return(CloseAll());
      int ticket = (int)StringToInteger(parts[1]);
      if(n >= 3)
         return(CloseTicketLots(ticket, StringToDouble(parts[2])));
      return(CloseTicket(ticket));
     }

   return("UNKNOWN|" + op);
  }

double NormPrice(string symbol, double price)
  {
   if(price == 0.0)
      return(0.0);
   int digits = (int)MarketInfo(symbol, MODE_DIGITS);
   return(NormalizeDouble(price, digits));
  }

double NormLots(string symbol, double lots)
  {
   double minlot = MarketInfo(symbol, MODE_MINLOT);
   double maxlot = MarketInfo(symbol, MODE_MAXLOT);
   double step   = MarketInfo(symbol, MODE_LOTSTEP);
   if(step <= 0.0)
      step = 0.01;
   lots = MathFloor(lots / step + 1.0e-8) * step;
   if(lots < minlot) lots = minlot;
   if(maxlot > 0.0 && lots > maxlot) lots = maxlot;
   int d = 2;
   if(step >= 1.0) d = 0;
   else if(step >= 0.1) d = 1;
   else if(step >= 0.01) d = 2;
   else d = 3;
   return(NormalizeDouble(lots, d));
  }

string OpenTrade(int type, string symbol, double lots, double sl, double tp)
  {
   if(!IsConnected())
      return("ERR|NOT_CONNECTED");
   if(!IsTradeAllowed())
      return("ERR|TRADE_DISABLED");
   if(type != OP_BUY && type != OP_SELL)
      return("ERR|BAD_TYPE");
   if(!SymbolSelect(symbol, true))
      return("ERR|UNKNOWN_SYMBOL|" + symbol);

   RefreshRates();
   lots = NormLots(symbol, lots);
   sl   = NormPrice(symbol, sl);
   tp   = NormPrice(symbol, tp);

   int    cmd   = (type == 0) ? OP_BUY : OP_SELL;
   int    digits= (int)MarketInfo(symbol, MODE_DIGITS);
   double price = (cmd == OP_BUY) ? MarketInfo(symbol, MODE_ASK) : MarketInfo(symbol, MODE_BID);
   price = NormalizeDouble(price, digits);
   color  clr   = (cmd == OP_BUY) ? clrBlue : clrRed;

   int ticket = OrderSend(symbol, cmd, lots, price, InpSlippage, sl, tp,
                          "FlyTrader", InpMagic, 0, clr);
   if(ticket > 0)
      return("OK|" + IntegerToString(ticket) + "|" + DoubleToString(price, digits));

   return("ERR|" + IntegerToString(GetLastError()));
  }

string CloseTicketLots(int ticket, double close_lots)
  {
   if(!OrderSelect(ticket, SELECT_BY_TICKET))
      return("ERR|" + IntegerToString(GetLastError()));
   if(OrderCloseTime() != 0)
      return("ERR|ALREADY_CLOSED");

   int type = OrderType();
   if(type > OP_SELL)
     {
      if(OrderDelete(ticket))
         return("OK|" + IntegerToString(ticket));
      return("ERR|" + IntegerToString(GetLastError()));
     }

   RefreshRates();
   string symbol = OrderSymbol();
   double open_lots = OrderLots();
   double lots = close_lots;
   if(lots <= 0.0 || lots >= open_lots - 1.0e-8)
      lots = open_lots;
   else
      lots = NormLots(symbol, lots);
   if(lots > open_lots)
      lots = open_lots;
   int    digits = (int)MarketInfo(symbol, MODE_DIGITS);
   double price  = (type == OP_BUY) ? MarketInfo(symbol, MODE_BID) : MarketInfo(symbol, MODE_ASK);
   price = NormalizeDouble(price, digits);

   if(OrderClose(ticket, lots, price, InpSlippage, clrGold))
      return("OK|" + IntegerToString(ticket) + "|" + DoubleToString(lots, 2));
   return("ERR|" + IntegerToString(GetLastError()));
  }

string CloseTicket(int ticket)
  {
   return(CloseTicketLots(ticket, 0.0));
  }

string CloseAll()
  {
   int ok = 0;
   int fail = 0;
   for(int i=OrdersTotal()-1; i>=0; i--)
     {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
         continue;
      if(OrderMagicNumber() != InpMagic)
         continue;
      string res = CloseTicket(OrderTicket());
      if(StringFind(res, "OK|") == 0)
         ok++;
      else
         fail++;
     }
   return(StringFormat("OK|closed=%d|failed=%d", ok, fail));
  }

int CountBars(string symbol, int tf)
  {
   ResetLastError();
   iClose(symbol, tf, 0);
   int n = iBars(symbol, tf);
   if(n > 2)
      return(n);
   double rates[][6];
   n = ArrayCopyRates(rates, symbol, tf);
   return(n);
  }

int PickHistoryTf(string symbol, int wanted)
  {
   int choices[8];
   choices[0] = wanted;
   choices[1] = Period();
   choices[2] = PERIOD_M1;
   choices[3] = PERIOD_M5;
   choices[4] = PERIOD_M15;
   choices[5] = PERIOD_H1;
   choices[6] = PERIOD_H4;
   choices[7] = PERIOD_D1;
   for(int i=0; i<8; i++)
     {
      int tf = choices[i];
      if(tf <= 0)
         continue;
      if(CountBars(symbol, tf) >= 3)
         return(tf);
     }
   if(CountBars(Symbol(), Period()) >= 3)
      return(Period());
   return(0);
  }

string ListHistory(string symbol, int tf, int count)
  {
   if(StringLen(symbol) == 0)
      symbol = Symbol();
   SymbolSelect(symbol, true);
   if(count < 2)
      count = 2;
   if(count > 400)
      count = 400;

   int used = PickHistoryTf(symbol, tf);
   if(used <= 0)
     {
      symbol = Symbol();
      used = PickHistoryTf(symbol, Period());
     }
   if(used <= 0)
      return("ERR|NO_HISTORY|" + symbol + "|open EURUSD and press Home to load bars");

   if(used != tf)
      Print("FlyTrader: no bars for TF ", tf, ", using TF ", used);

   int total = CountBars(symbol, used);
   if(total < 3)
     {
      symbol = Symbol();
      used = Period();
      total = CountBars(symbol, used);
     }
   if(total < 3)
      return("ERR|NO_HISTORY|" + symbol);

   if(count > total - 1)
      count = total - 1;

   int digits = (int)MarketInfo(symbol, MODE_DIGITS);
   if(digits <= 0)
      digits = 5;

   string out = "OK|" + IntegerToString(count) + "|" + IntegerToString(used);
   for(int i=count; i>=1; i--)
     {
      datetime t = iTime(symbol, used, i);
      double   o = iOpen(symbol, used, i);
      double   h = iHigh(symbol, used, i);
      double   l = iLow(symbol, used, i);
      double   c = iClose(symbol, used, i);
      if(t <= 0 || c <= 0.0)
         continue;
      out += StringFormat("|%d,%s,%s,%s,%s,%s",
                          (int)t,
                          DoubleToString(o, digits),
                          DoubleToString(h, digits),
                          DoubleToString(l, digits),
                          DoubleToString(c, digits),
                          DoubleToString((double)iVolume(symbol, used, i), 0));
     }
   return(out);
  }

string ListPositions()
  {
   string out = "OK";
   for(int i=0; i<OrdersTotal(); i++)
     {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
         continue;
      if(OrderMagicNumber() != InpMagic && InpMagic != 0)
         continue;
      int digits = (int)MarketInfo(OrderSymbol(), MODE_DIGITS);
      out += StringFormat("|%d,%s,%d,%s,%s,%s,%s",
                          OrderTicket(),
                          OrderSymbol(),
                          OrderType(),
                          DoubleToString(OrderLots(), 2),
                          DoubleToString(OrderOpenPrice(), digits),
                          DoubleToString(OrderStopLoss(), digits),
                          DoubleToString(OrderTakeProfit(), digits));
     }
   return(out);
  }
