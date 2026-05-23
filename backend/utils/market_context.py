"""
utils/market_context.py — global market-context features
----------------------------------------------------------
Adds 4 macro features that are the SAME for every stock on a given date:
  India_VIX            — NSE volatility / fear index (regime gauge)
  India_VIX_change_1d  — 1-day % change in India VIX
  USDINR_change_1d     — 1-day % change in USD/INR
  Crude_change_1d      — 1-day % change in WTI crude

All values are LAGGED 1 day (shift(1)) so a row only ever sees data that
was known BEFORE that day's close — no look-ahead leakage.

The 3 macro series are market-wide, so they are fetched ONCE and cached
in memory, then merged by Date into each stock's frame.
"""

import yfinance as yf
import pandas as pd

# symbol -> short name used in the merged column
_MACRO_SYMBOLS = {
    "^INDIAVIX": "India_VIX",
    "USDINR=X":  "USDINR",
    "CL=F":      "Crude",
}

_macro_cache = None   # built once, reused for all 49 stocks


def _download_one(symbol: str, period: str) -> pd.Series:
    """Download one symbol; return a clean Date-indexed Close series."""
    df = yf.download(symbol, period=period, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):       # newer yfinance quirk
        df.columns = df.columns.get_level_values(0)
    close = df["Close"].dropna()
    close.index = pd.to_datetime(close.index).normalize()
    return close


def _build_macro_frame(period: str) -> pd.DataFrame:
    """Build the cached macro DataFrame: Date + 4 context features."""
    out = pd.DataFrame()
    for symbol, name in _MACRO_SYMBOLS.items():
        close = _download_one(symbol, period)
        out[name] = close

    # 1-day % change for each series
    out["India_VIX_change_1d"] = out["India_VIX"].pct_change()
    out["USDINR_change_1d"]    = out["USDINR"].pct_change()
    out["Crude_change_1d"]     = out["Crude"].pct_change()

    # keep India_VIX level (regime gauge) + the 3 change features
    keep = ["India_VIX", "India_VIX_change_1d",
            "USDINR_change_1d", "Crude_change_1d"]
    out = out[keep]

    # LAG everything by 1 day: a row sees only PRIOR-day macro values
    out = out.shift(1)

    out.index.name = "Date"
    return out.reset_index()


def add_market_context(df: pd.DataFrame, period: str = "5y") -> pd.DataFrame:
    """Merge the 4 macro context features into a single stock's frame.

    Expects `df` to have a 'Date' column (or a DatetimeIndex). Returns the
    same frame with India_VIX, India_VIX_change_1d, USDINR_change_1d,
    Crude_change_1d added. Missing macro days are forward-filled.
    """
    global _macro_cache
    if _macro_cache is None:
        _macro_cache = _build_macro_frame(period)

    df = df.copy()

    # normalise the stock frame's Date for a clean merge
    if "Date" not in df.columns:
        df = df.reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    merged = df.merge(_macro_cache, on="Date", how="left")

    # macro markets occasionally have holidays NSE doesn't (and vice versa)
    # -> forward-fill so a stock day always has the latest known macro value
    macro_cols = ["India_VIX", "India_VIX_change_1d",
                  "USDINR_change_1d", "Crude_change_1d"]
    merged[macro_cols] = merged[macro_cols].ffill()

    return merged
