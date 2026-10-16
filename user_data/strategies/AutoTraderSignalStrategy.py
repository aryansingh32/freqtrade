# pragma pylint: disable=missing-docstring, invalid-name
"""
Trades on the decisions of the AutoTrader bot (TradingAgents repository,
``autotrader`` package), with freqtrade handling execution, stop loss,
trailing stop and take profit.

AutoTrader writes its latest decision per symbol to ``signals.json``:
technical analysis, a history-analog forecast and (optionally) the
TradingAgents multi-agent LLM debate over news and sentiment. This strategy
reads that file on every candle:

* entry: the pair's latest AutoTrader action is BUY, the decision is fresh
  (``signal_max_age_hours``), and price is above its 50 EMA;
* exit: AutoTrader says SELL, or the stop loss / trailing stop / ROI fires.

Point ``AUTOTRADER_SIGNALS`` at the file (default
``../TradingAgents/autotrader_data/signals.json``). In backtesting and
hyperopt there are no historical AutoTrader decisions, so the strategy falls
back to its technical filter alone (EMA trend + RSI), which makes the
backtest a test of the risk settings, not of AutoTrader's signals; use
``autotrader backtest`` for those.

Use freqtrade pair names in the AutoTrader config, e.g. ``"symbol": "BTC/USDT"``
with ``"data_symbol": "BTC-USD"``. Either let this strategy act on the
signals file (AutoTrader on a ``paper`` broker, which only simulates), or give
AutoTrader the ``freqtrade`` broker so it opens trades through the REST API;
with this strategy loaded, freqtrade's stop loss and trailing stop then
protect those trades too.
"""

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import talib.abstract as ta
from pandas import DataFrame

from freqtrade.enums import RunMode
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, informative


DEFAULT_SIGNALS = Path(__file__).resolve().parents[3] / "TradingAgents" / "autotrader_data" / "signals.json"


class AutoTraderSignalStrategy(IStrategy):
    INTERFACE_VERSION = 3
    can_short = False
    timeframe = "1h"
    # 200 daily candles for the daily trend filter, in 1h candles.
    startup_candle_count = 4900

    # Risk: 6% hard stop, trail 3% once 4% in profit, take profit by ROI table.
    stoploss = -0.06
    trailing_stop = True
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True
    minimal_roi = {"0": 0.15, "1440": 0.08, "4320": 0.03}

    use_exit_signal = True
    process_only_new_candles = True

    # Discipline rules: pause after a cluster of stop-outs, cool down after
    # each exit, stop on a deep drawdown, and lock pairs that keep losing.
    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 6},
            {"method": "StoplossGuard", "lookback_period_candles": 24, "trade_limit": 3,
             "stop_duration_candles": 12, "only_per_pair": False},
            {"method": "MaxDrawdown", "lookback_period_candles": 168, "trade_limit": 5,
             "stop_duration_candles": 24, "max_allowed_drawdown": 0.15},
            {"method": "LowProfitPairs", "lookback_period_candles": 336, "trade_limit": 4,
             "stop_duration_candles": 48, "required_profit": 0.0},
        ]

    signal_max_age_hours = 26

    def _signals(self) -> dict:
        path = Path(os.environ.get("AUTOTRADER_SIGNALS", DEFAULT_SIGNALS))
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _signal_for(self, pair: str) -> dict | None:
        signals = self._signals()
        base = pair.split("/")[0]
        sig = signals.get(pair) or signals.get(f"{base}-USD")
        if not sig:
            return None
        age = datetime.now(UTC) - datetime.fromisoformat(sig["at"])
        return sig if age <= timedelta(hours=self.signal_max_age_hours) else None

    def _live(self) -> bool:
        return self.dp.runmode in (RunMode.LIVE, RunMode.DRY_RUN)

    @informative("1d")
    def populate_indicators_1d(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Higher-timeframe trend: only buy while the daily close is above a
        # rising 200-day EMA (columns arrive as ema200_1d, close_1d, ...).
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["ema200_rising"] = (dataframe["ema200"] > dataframe["ema200"].shift(5)).astype(int)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        daily_up = (dataframe["close_1d"] > dataframe["ema200_1d"]) & (dataframe["ema200_rising_1d"] == 1)
        trend = (dataframe["close"] > dataframe["ema50"]) & (dataframe["rsi"] < 75) & daily_up
        if self._live():
            sig = self._signal_for(metadata["pair"])
            if not sig or sig.get("action") != "BUY":
                return dataframe
            dataframe.loc[trend, ["enter_long", "enter_tag"]] = (1, f"autotrader {sig['score']:+.2f}")
        else:
            backtest_rule = trend & (dataframe["ema50"] > dataframe["ema200"]) & (dataframe["rsi"] > 50)
            dataframe.loc[backtest_rule, ["enter_long", "enter_tag"]] = (1, "technical")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self._live():
            dataframe.loc[dataframe["close"] < dataframe["ema200"], "exit_long"] = 1
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        if self._live():
            sig = self._signal_for(pair)
            if sig and sig.get("action") == "SELL":
                return f"autotrader_sell {sig['score']:+.2f}"
        return None
