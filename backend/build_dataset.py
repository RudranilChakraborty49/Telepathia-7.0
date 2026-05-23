"""
build_dataset.py v6 — HYBRID FEATURES, 49-STOCK SCALE
------------------------------------------------------
Builds master ML dataset with technical + calendar + market regime features.
Binary labels: UP (1) / DOWN (0), noise days dropped.

v6 changes (Nifty 50 expansion):
  - Per-stock try/except — one bad stock cannot crash or corrupt the build
  - Column-consistency check before concatenation
  - Detailed per-stock report (rows in/out, class balance)
"""

import os
import pandas as pd

from config import NIFTY_50_TICKERS, HISTORICAL_PERIOD
from utils.data_fetcher import get_stock_data
from utils.indicators import add_indicators
from utils.calendar_features import add_calendar_features
from utils.market_features import add_market_features
from utils.market_context import add_market_context

MOVEMENT_THRESHOLD = 0.015      # ±1.5% over the forward window = directional
FORWARD_DAYS = 3                # label horizon


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Binary UP/DOWN label; noise days (small moves) become None and get dropped."""
    df = df.copy()
    df["Forward_Return"] = df["Close"].shift(-FORWARD_DAYS) / df["Close"] - 1

    def classify(ret):
        if pd.isna(ret):
            return None
        if ret > MOVEMENT_THRESHOLD:
            return 1
        elif ret < -MOVEMENT_THRESHOLD:
            return 0
        else:
            return None

    df["Target"] = df["Forward_Return"].apply(classify)
    return df


def process_one_stock(ticker: str, period: str):
    """
    Build the feature+label frame for a single ticker.
    Returns a DataFrame, or None on any failure (caller logs it).
    """
    df = get_stock_data(ticker, period=period)
    if df is None or df.empty:
        raise ValueError("no data returned")

    rows_in = len(df)

    df = add_indicators(df)                              # technical
    df = add_market_features(df, ticker, period=period)  # market regime
    df = add_market_context(df, period=period) 
    df = add_calendar_features(df)                       # calendar
    df = add_labels(df)                                  # binary labels
    df["Ticker"] = ticker

    df = df.dropna()
    rows_out = len(df)

    if rows_out == 0:
        raise ValueError("0 usable rows after feature engineering")

    return df, rows_in, rows_out


def build_master_dataset(tickers: list, period: str = "5y"):
    """Process every ticker with per-stock error isolation."""
    frames = []
    report = []        # (ticker, rows_in, rows_out, up, down) or (ticker, error)
    failed = []

    for i, ticker in enumerate(tickers, 1):
        prefix = f"  [{i:>2}/{len(tickers)}] {ticker:<16}"
        try:
            df, rows_in, rows_out = process_one_stock(ticker, period)
            up = int((df["Target"] == 1).sum())
            down = int((df["Target"] == 0).sum())
            print(f"{prefix} ✅ {rows_out:>4} rows  (UP {up}, DOWN {down})")
            frames.append(df)
            report.append((ticker, rows_in, rows_out, up, down))
        except Exception as e:
            print(f"{prefix} ❌ FAILED: {e}")
            failed.append((ticker, str(e)))

    if not frames:
        raise RuntimeError("No stocks processed successfully — aborting.")

    # ── Column-consistency check before concat ──
    ref_cols = set(frames[0].columns)
    for df in frames[1:]:
        if set(df.columns) != ref_cols:
            extra = set(df.columns) - ref_cols
            missing = ref_cols - set(df.columns)
            raise RuntimeError(
                f"Column mismatch across stocks. extra={extra}, missing={missing}"
            )

    master = pd.concat(frames, axis=0).reset_index(drop=True)
    return master, report, failed


def save_dataset(df: pd.DataFrame, filename: str = "master_dataset.csv"):
    out_dir = os.path.join(os.path.dirname(__file__), "..", "research", "dataset")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    df.to_csv(path, index=False)
    print(f"\n💾 Master dataset saved → {path}")


if __name__ == "__main__":
    print("=" * 70)
    print("BUILDING MASTER DATASET v6 — HYBRID FEATURES (Nifty 50 scale)")
    print("=" * 70)
    print(f"  Stocks:   {len(NIFTY_50_TICKERS)}")
    print(f"  Features: Technical + Calendar + Market Regime")
    print(f"  Labels:   Binary UP/DOWN (±{MOVEMENT_THRESHOLD*100:.1f}% "
          f"over {FORWARD_DAYS} days)")
    print("=" * 70 + "\n")

    master_df, report, failed = build_master_dataset(
        NIFTY_50_TICKERS, period=HISTORICAL_PERIOD
    )

    # ───── SUMMARY ─────
    print("\n" + "=" * 70)
    print("  DATASET SUMMARY")
    print("=" * 70)
    print(f"  Stocks processed OK : {len(report)}")
    print(f"  Stocks failed       : {len(failed)}")
    print(f"  Total directional rows: {master_df.shape[0]}")
    print(f"  Columns               : {master_df.shape[1]}")

    counts = master_df["Target"].value_counts().sort_index()
    total = counts.sum()
    print(f"\n  📊 Binary class distribution:")
    for label, name in [(0, "DOWN"), (1, "UP")]:
        if label in counts.index:
            n = int(counts[label])
            print(f"     {name:5s} ({label}): {n:6d}  ({n/total*100:5.1f}%)")

    # Flag stocks with unusually few rows (possible data quality issue)
    if report:
        avg_rows = sum(r[2] for r in report) / len(report)
        low = [r for r in report if r[2] < avg_rows * 0.5]
        if low:
            print(f"\n  ⚠️  Stocks with <50% of average rows ({avg_rows:.0f}):")
            for tk, _, rout, _, _ in low:
                print(f"      {tk:<16} {rout} rows")

    if failed:
        print(f"\n  ❌ FAILED STOCKS:")
        for tk, err in failed:
            print(f"      {tk:<16} {err}")

    print("=" * 70)

    save_dataset(master_df)