"""
indicators.py
-------------
Computes technical indicators for ML feature engineering.

Indicators (UPDATED v2):
  - RSI (14)
  - MACD + MACD_signal
  - Bollinger Bands %B
  - EMA 20/50 + EMA ratio
  - Volume change vs 20-day MA
  - ATR (14)
  - Stochastic Oscillator      ← NEW
  - Williams %R                ← NEW
  - Rate of Change (ROC)       ← NEW
  - On-Balance Volume change   ← NEW
  - Lagged returns (1d, 3d, 5d) ← NEW
"""

import pandas as pd
from ta.momentum import (
    RSIIndicator,
    StochasticOscillator,
    WilliamsRIndicator,
    ROCIndicator,
)
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators as new columns.

    Args:
        df: DataFrame with ['Open','High','Low','Close','Volume']

    Returns:
        df with all indicator columns added
    """
    df = df.copy()

    # ───── EXISTING INDICATORS ─────

    # 1. RSI (14)
    df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()

    # 2. MACD
    macd = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()

    # 3. Bollinger Bands %B
    bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["BB_pctB"] = bb.bollinger_pband()

    # 4. EMA 20 & 50
    df["EMA_20"] = EMAIndicator(close=df["Close"], window=20).ema_indicator()
    df["EMA_50"] = EMAIndicator(close=df["Close"], window=50).ema_indicator()
    df["EMA_ratio"] = df["EMA_20"] / df["EMA_50"]

    # 5. Volume change vs 20-day MA
    df["Volume_MA20"] = df["Volume"].rolling(window=20).mean()
    df["Volume_change"] = df["Volume"] / df["Volume_MA20"]

    # 6. ATR (14)
    df["ATR"] = AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    ).average_true_range()

    # ───── NEW INDICATORS ─────

    # 7. Stochastic Oscillator (%K)
    #    Measures where current price is within 14-day high-low range
    stoch = StochasticOscillator(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    )
    df["Stoch_K"] = stoch.stoch()

    # 8. Williams %R
    #    Like stochastic but inverted (0 to -100)
    df["Williams_R"] = WilliamsRIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], lbp=14
    ).williams_r()

    # 9. Rate of Change (10-day)
    #    Pure momentum: % change vs 10 days ago
    df["ROC"] = ROCIndicator(close=df["Close"], window=10).roc()

    # 10. On-Balance Volume — normalized to % change
    obv = OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"]).on_balance_volume()
    df["OBV_change"] = obv.pct_change(periods=5)    # 5-day OBV change

    # 11. Lagged returns — captures short-term momentum
    df["Return_1d"] = df["Close"].pct_change(periods=1)
    df["Return_3d"] = df["Close"].pct_change(periods=3)
    df["Return_5d"] = df["Close"].pct_change(periods=5)

    return df


# ───── TEST BLOCK ─────
if __name__ == "__main__":
    from data_fetcher import get_stock_data

    print("=" * 60)
    print("TESTING indicators.py v2 (UPGRADED)")
    print("=" * 60)

    df = get_stock_data("RELIANCE.NS", period="1y")
    print(f"\nOriginal columns: {list(df.columns)}")

    df = add_indicators(df)
    print(f"\nNew columns: {list(df.columns)}")
    print(f"Total columns now: {len(df.columns)}")

    cols_to_show = ["Close", "RSI", "Stoch_K", "Williams_R", "ROC",
                    "OBV_change", "Return_1d", "Return_3d"]
    print("\n--- Latest values ---")
    print(df[cols_to_show].tail(3).round(2))