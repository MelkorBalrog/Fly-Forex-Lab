#property strict
#property copyright "FlyFOREXTrader"
#property link      ""
#property version   "1.30"
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
input bool   InpProfitRecycle = true; // BB squeeze→expand + 1.5× win budget
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
   Print("FlyTrader: ready  REP=", InpRepEndpoint, "  PUSH=", InpPushEndpoint,
         "  fuse=", FuseCatsSpec(), "  dynamic=", FuseDynamicSpec(),
         "  inds=", FuseIndsSpec(), "  risk=", FuseRiskSpec(),
         "  sugar=", FuseSugarSpec(), "  samt=", FuseSugarAmtSpec(),
         "  sdyn=", FuseSugarDynSpec(),
         "  vol=", FuseVolumePctSpec(), "/", FuseVolumeModeSpec(),
         "  nn=", FuseNnSpec(), "  vision=", FuseVisionSpec());
   PaintStatus();
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
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
           "GUI Live uses dashboard dials; CLI live uses these Inputs.");
  }

string FuseCatsSpec()
  {
   string names[7];
   int n = 0;
   if(InpFuseTrend)      names[n++] = "trend";
   if(InpFuseMomentum)   names[n++] = "momentum";
   if(InpFuseVolatility) names[n++] = "volatility";
   if(InpFuseVolume)     names[n++] = "volume";
   if(InpFuseBreadth)    names[n++] = "breadth";
   if(InpFuseStructure)  names[n++] = "structure";
   if(InpFuseSentiment)  names[n++] = "sentiment";
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
   return(InpFuseDynamic ? "1" : "0");
  }

string FuseIndsSpec()
  {
   string names[32];
   int n = 0;
   if(InpIndSma)        names[n++] = "sma";
   if(InpIndEma)        names[n++] = "ema";
   if(InpIndWma)        names[n++] = "wma";
   if(InpIndMacd)       names[n++] = "macd";
   if(InpIndAdx)        names[n++] = "adx";
   if(InpIndSar)        names[n++] = "sar";
   if(InpIndIchimoku)   names[n++] = "ichimoku";
   if(InpIndSupertrend) names[n++] = "supertrend";
   if(InpIndRsi)        names[n++] = "rsi";
   if(InpIndStoch)      names[n++] = "stoch";
   if(InpIndCci)        names[n++] = "cci";
   if(InpIndWillr)      names[n++] = "willr";
   if(InpIndRoc)        names[n++] = "roc";
   if(InpIndBb)         names[n++] = "bb";
   if(InpIndAtr)        names[n++] = "atr";
   if(InpIndKeltner)    names[n++] = "keltner";
   if(InpIndStdev)      names[n++] = "stdev";
   if(InpIndDonchian)   names[n++] = "donchian";
   if(InpIndObv)        names[n++] = "obv";
   if(InpIndCmf)        names[n++] = "cmf";
   if(InpIndVwap)       names[n++] = "vwap";
   if(InpIndAd)         names[n++] = "ad";
   if(InpIndProfile)    names[n++] = "profile";
   if(InpIndAdLine)     names[n++] = "ad_line";
   if(InpIndMcclellan)  names[n++] = "mcclellan";
   if(InpIndTrin)       names[n++] = "trin";
   if(InpIndPivot)      names[n++] = "pivot";
   if(InpIndFib)        names[n++] = "fib";
   if(InpIndChannel)    names[n++] = "channel";
   if(InpIndPutCall)    names[n++] = "put_call";
   if(InpIndVix)        names[n++] = "vix";
   if(InpIndCot)        names[n++] = "cot";
   if(n <= 0)
      return("none");
   if(n >= 32)
      return("all");
   string out = names[0];
   int i;
   for(i=1; i<n; i++)
      out = out + "," + names[i];
   return(out);
  }

string FuseRiskSpec()
  {
   string t = InpRiskTol;
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
   return(InpSugar ? "1" : "0");
  }

string FuseSugarAmtSpec()
  {
   double a = InpSugarAmt;
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
   return(InpSugarDyn ? "1" : "0");
  }

string FuseVolumePctSpec()
  {
   if(InpVolumePct < 0.0)
      return("auto");
   double a = InpVolumePct;
   if(a > 100.0)
      a = 100.0;
   return(DoubleToString(a, 2));
  }

string FuseVolumeModeSpec()
  {
   string t = InpVolumeMode;
   StringToLower(t);
   StringTrimLeft(t);
   StringTrimRight(t);
   if(t == "fixed" || t == "equity" || t == "auto")
      return(t);
   return("auto");
  }

string FuseBankSpec()
  {
   double a = InpBankPct;
   if(a < 0.0)
      a = 0.0;
   if(a > 50.0)
      a = 50.0;
   return(DoubleToString(a, 2));
  }

string FuseHoldSpec()
  {
   return(InpHoldRisk ? "1" : "0");
  }

string FuseHoldSensSpec()
  {
   double a = InpHoldRiskSens;
   if(a < 0.0)
      a = 0.0;
   if(a > 2.0)
      a = 2.0;
   return(DoubleToString(a, 2));
  }

string FuseRecycleSpec()
  {
   return(InpProfitRecycle ? "1" : "0");
  }

string FuseBayesSpec()
  {
   return(InpTradeBayes ? "1" : "0");
  }

string FuseAppSpec()
  {
   return(InpEntryAppetite ? "1" : "0");
  }

string FuseTradeRateSpec()
  {
   double a = InpTradeRate;
   if(a < 0.0)
      a = 0.0;
   if(a > 40.0)
      a = 40.0;
   return(DoubleToString(a, 1));
  }

string FuseNnSpec()
  {
   return(InpNnVote ? "1" : "0");
  }

string FuseVisionSpec()
  {
   return(InpNnVision ? "1" : "0");
  }

string FuseCropsSpec()
  {
   return(InpFlyCrops ? "1" : "0");
  }

string FuseBrainSpec()
  {
   return(InpUseBrain ? "1" : "0");
  }

string FuseEvoSpec()
  {
   return(InpUseEvo ? "1" : "0");
  }

string FuseCdaSpec()
  {
   return(InpContinueDa ? "1" : "0");
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
          "|CROPS|" + FuseCropsSpec() +
          "|BRAIN|" + FuseBrainSpec() +
          "|EVO|" + FuseEvoSpec() +
          "|CDA|" + FuseCdaSpec());
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
