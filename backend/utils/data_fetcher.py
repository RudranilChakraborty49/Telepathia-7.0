"""
data_fetcher.py
---------------
Fetches NSE stock data from yfinance with local caching.
"""

import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# ───── CONFIG ─────
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data_cache")
CACHE_EXPIRY_HOURS = 24


def _get_cache_path(ticker: str) -> str:
    """Build the file path where a ticker's data is cached."""
    safe_name = ticker.replace(".", "_")
    return os.path.join(CACHE_DIR, f"{safe_name}.csv")


def _is_cache_fresh(filepath: str) -> bool:
    """Return True if cached file exists and is < 24 hours old."""
    if not os.path.exists(filepath):
        return False
    file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(filepath))
    return file_age < timedelta(hours=CACHE_EXPIRY_HOURS)


def _period_to_days(period: str) -> int:
    """Approximate days for yfinance period strings."""
    mapping = {
        "1mo": 30, "3mo": 90, "6mo": 180,
        "1y": 365, "2y": 730, "5y": 1825, "10y": 3650, "max": 7300,
    }
    return mapping.get(period, 1825)


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where OHLC is NaN — handles yfinance returning
    today's in-progress bar with incomplete data.
    """
    if df is None or df.empty:
        return df
    cols_to_check = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    if cols_to_check:
        df = df.dropna(subset=cols_to_check)
    return df


def get_stock_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV with smart caching that respects period.
    Auto-cleans rows with NaN OHLC values.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = _get_cache_path(ticker)

    # Try cache first
    if _is_cache_fresh(cache_path):
        cached = pd.read_csv(cache_path, index_col="Date", parse_dates=True)

        if not cached.empty:
            cache_days = (cached.index.max() - cached.index.min()).days
            required_days = _period_to_days(period)
            if cache_days >= required_days * 0.9:
                print(f"⚡ Loading {ticker} from cache")
                return _clean_dataframe(cached)
            else:
                print(f"⚠️ Cache for {ticker} too short ({cache_days}d < {required_days}d) — refetching")

    # Fetch fresh
    print(f"📥 Fetching {ticker} from Yahoo Finance...")
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_csv(cache_path)
        print(f"💾 Saved {ticker} → {cache_path}")
        return _clean_dataframe(df)
    except Exception as e:
        print(f"❌ Error fetching {ticker}: {e}")
        return pd.DataFrame()


def get_multiple_stocks(tickers: list, period: str = "5y") -> dict:
    """Fetch data for multiple tickers."""
    result = {}
    for ticker in tickers:
        df = get_stock_data(ticker, period=period)
        if not df.empty:
            result[ticker] = df
    return result


if __name__ == "__main__":
    from pprint import pprint
    print("=" * 50)
    print("TESTING data_fetcher.py")
    print("=" * 50)

    df = get_stock_data("RELIANCE.NS", period="1y")
    print(f"\n✅ RELIANCE shape: {df.shape}")
    print(df.tail(3))
    print(f"\nLast Close: {df['Close'].iloc[-1]}")