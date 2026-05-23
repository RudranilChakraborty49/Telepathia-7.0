"""
fetch_all_data.py
-----------------
One-time bulk downloader for the full Nifty 50 universe.

Loops over every stock in NIFTY_50_TICKERS plus all sector indices,
calls the existing get_stock_data() (which handles caching + NaN cleaning),
and prints a clear summary. Failures are logged, NOT fatal — one bad
ticker will not abort the run.

Run from the backend/ folder:
    python fetch_all_data.py
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from config import (
    NIFTY_50_TICKERS, SECTOR_INDICES, NIFTY_INDEX, HISTORICAL_PERIOD
)
from utils.data_fetcher import get_stock_data


# Minimum rows we consider "healthy" for ~5y of daily data.
# 5 years ≈ 1250 trading days; we warn below 250 (≈1 year).
MIN_HEALTHY_ROWS = 250


def collect_all_symbols():
    """Build the full de-duplicated list of symbols to fetch."""
    symbols = list(NIFTY_50_TICKERS)

    # Add every unique sector index + the broad Nifty index
    index_set = set(SECTOR_INDICES.values())
    index_set.add(NIFTY_INDEX)

    indices = sorted(index_set)
    return symbols, indices


def fetch_group(label, symbols):
    """Fetch a list of symbols, returning (ok, short, failed) records."""
    ok, short, failed = [], [], []

    print(f"\n{'=' * 70}")
    print(f"  FETCHING {label}  ({len(symbols)} symbols)")
    print(f"{'=' * 70}")

    for i, sym in enumerate(symbols, 1):
        prefix = f"  [{i:>2}/{len(symbols)}] {sym:<16}"
        try:
            df = get_stock_data(sym, period=HISTORICAL_PERIOD)
            if df is None or df.empty:
                print(f"{prefix} ❌ no data returned")
                failed.append(sym)
            elif len(df) < MIN_HEALTHY_ROWS:
                print(f"{prefix} ⚠️  only {len(df)} rows (short history)")
                short.append((sym, len(df)))
            else:
                last_date = df.index.max().strftime("%Y-%m-%d")
                print(f"{prefix} ✅ {len(df):>5} rows  (latest {last_date})")
                ok.append((sym, len(df)))
        except Exception as e:
            print(f"{prefix} ❌ ERROR: {e}")
            failed.append(sym)

        # Gentle pacing — avoids hammering yfinance when cache misses
        time.sleep(0.3)

    return ok, short, failed


def main():
    t0 = time.time()
    print("\n" + "=" * 70)
    print("  TELEPATHIA 7.0 — BULK DATA FETCH (Nifty 50 expansion)")
    print("=" * 70)
    print(f"  Period: {HISTORICAL_PERIOD}")

    stocks, indices = collect_all_symbols()
    print(f"  Stocks: {len(stocks)}   Indices: {len(indices)}")

    s_ok, s_short, s_failed = fetch_group("STOCKS", stocks)
    i_ok, i_short, i_failed = fetch_group("INDICES", indices)

    elapsed = round(time.time() - t0, 1)

    # ───── SUMMARY ─────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Stocks  : {len(s_ok)} ok, {len(s_short)} short, {len(s_failed)} failed")
    print(f"  Indices : {len(i_ok)} ok, {len(i_short)} short, {len(i_failed)} failed")
    print(f"  Time    : {elapsed}s")

    if s_short or i_short:
        print("\n  ⚠️  SHORT HISTORY (usable but less data):")
        for sym, n in s_short + i_short:
            print(f"      {sym:<16} {n} rows")

    if s_failed or i_failed:
        print("\n  ❌ FAILED — these need attention:")
        for sym in s_failed + i_failed:
            print(f"      {sym}")
        print("\n  Likely causes: recent index reshuffle, delisting, or a")
        print("  symbol-format quirk. Swap failed tickers in config.py.")
    else:
        print("\n  🎉 All symbols fetched successfully.")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()