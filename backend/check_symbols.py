"""
check_symbols.py — quick probe: which market-context symbols pull clean?
-------------------------------------------------------------------------
Run ONCE. Tells us which candidate features have usable Yahoo data
before we design anything around them.
"""

import yfinance as yf
import pandas as pd

# Candidate symbols for new market-context features
CANDIDATES = {
    "India VIX":   "^INDIAVIX",
    "USD/INR":     "USDINR=X",
    "Crude WTI":   "CL=F",
    "Crude Brent": "BZ=F",
}

PERIOD = "5y"

print("=" * 60)
print("SYMBOL AVAILABILITY CHECK")
print("=" * 60)

for name, symbol in CANDIDATES.items():
    try:
        df = yf.download(symbol, period=PERIOD, progress=False, auto_adjust=True)
        # newer yfinance returns multi-level columns -> flatten them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            print(f"  ❌ {name:12s} ({symbol:12s}) — NO DATA")
            continue
        rows = len(df)
        start = df.index[0].date()
        end = df.index[-1].date()
        nan_close = int(df["Close"].isna().sum())
        print(f"  ✅ {name:12s} ({symbol:12s}) — {rows} rows, "
              f"{start} → {end}, NaN closes: {nan_close}")
    except Exception as e:
        print(f"  ❌ {name:12s} ({symbol:12s}) — ERROR: {e}")

print("=" * 60)
print("Paste this output back — we'll pick which features to build.")
print("=" * 60)
