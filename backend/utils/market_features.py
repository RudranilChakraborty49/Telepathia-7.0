"""
market_features.py
------------------
Adds broader market context features:
  - Nifty 50 returns & volatility
  - Sector index returns
  - Stock-vs-Nifty relative strength
"""

import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from utils.data_fetcher import get_stock_data
from config import NIFTY_INDEX, SECTOR_INDICES


def _compute_index_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Compute return and volatility features for an index DataFrame."""
    df = df.copy()
    df[f"{prefix}_Return_1d"] = df["Close"].pct_change(periods=1)
    df[f"{prefix}_Return_5d"] = df["Close"].pct_change(periods=5)
    df[f"{prefix}_Volatility_20d"] = (
        df["Close"].pct_change().rolling(window=20).std()
    )
    return df[[
        f"{prefix}_Return_1d",
        f"{prefix}_Return_5d",
        f"{prefix}_Volatility_20d",
    ]]


def add_market_features(stock_df: pd.DataFrame, ticker: str,
                        period: str = "5y") -> pd.DataFrame:
    """Add market regime + sector features to a stock's DataFrame."""
    stock_df = stock_df.copy()

    if not isinstance(stock_df.index, pd.DatetimeIndex):
        if "Date" in stock_df.columns:
            stock_df["Date"] = pd.to_datetime(stock_df["Date"])
            stock_df = stock_df.set_index("Date")
        else:
            raise ValueError("stock_df must have Date as index or column")

    # ───── Nifty features ─────
    nifty_df = get_stock_data(NIFTY_INDEX, period=period)
    nifty_feats = _compute_index_features(nifty_df, "Nifty")

    # ───── Sector features ─────
    sector_ticker = SECTOR_INDICES.get(ticker, NIFTY_INDEX)
    if sector_ticker == NIFTY_INDEX:
        sector_feats = nifty_feats.rename(columns={
            "Nifty_Return_1d":      "Sector_Return_1d",
            "Nifty_Return_5d":      "Sector_Return_5d",
            "Nifty_Volatility_20d": "Sector_Volatility_20d",
        })[["Sector_Return_1d"]]
    else:
        sector_df = get_stock_data(sector_ticker, period=period)
        sector_feats = _compute_index_features(sector_df, "Sector")[
            ["Sector_Return_1d"]
        ]

    # Merge index features into stock DataFrame
    stock_df = stock_df.join(nifty_feats, how="left")
    stock_df = stock_df.join(sector_feats, how="left")

    # ───── Relative Strength vs Nifty ─────
    stock_20d = stock_df["Close"].pct_change(periods=20)
    nifty_20d = nifty_df["Close"].pct_change(periods=20)

    rs_df = pd.DataFrame({"s": stock_20d, "n": nifty_20d}).dropna()
    rs_df["Stock_vs_Nifty_RS"] = (1 + rs_df["s"]) / (1 + rs_df["n"])
    stock_df = stock_df.join(rs_df[["Stock_vs_Nifty_RS"]], how="left")

    return stock_df


# ───── TEST ─────
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING market_features.py")
    print("=" * 60)

    reliance = get_stock_data("RELIANCE.NS", period="1y")
    enriched = add_market_features(reliance, "RELIANCE.NS", period="1y")

    print(f"\nOriginal columns: {len(reliance.columns)}")
    print(f"After market features: {len(enriched.columns)}")
    new_cols = sorted(set(enriched.columns) - set(reliance.columns))
    print(f"\nNew columns added:")
    for c in new_cols:
        print(f"  - {c}")

    cols = [
        "Close",
        "Nifty_Return_1d",
        "Sector_Return_1d",
        "Nifty_Volatility_20d",
        "Stock_vs_Nifty_RS",
    ]
    print("\n--- Last 5 rows ---")
    print(enriched[cols].tail(5).round(4))