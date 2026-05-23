"""
lstm_data_prep.py
-----------------
Converts the master dataset into LSTM-ready 3D tensors.

Pipeline:
  1. Load master_dataset.csv
  2. For each ticker:
       a. Scale OHLCV to 0-1
       b. Save scaler to disk
       c. Build sliding windows (60 days → next day's close)
  3. Combine all windows into one big training set
  4. Chronological train/test split (80/20)
  5. Save .npy arrays for model training
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# ───── CONFIG ─────
LOOKBACK = 60                  # Use 60 past days
FEATURES = ["Open", "High", "Low", "Close", "Volume"]
TARGET_COL = "Close"           # We're predicting the close price
TRAIN_SPLIT = 0.8              # 80% train, 20% test

# Paths (relative to this file)
HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "..", "..", "research", "dataset", "master_dataset.csv")
SCALERS_DIR = os.path.join(HERE, "..", "saved_models", "scalers")
ARRAYS_DIR = os.path.join(HERE, "..", "saved_models", "arrays")


def create_sequences(data: np.ndarray, lookback: int, target_idx: int):
    """
    Convert a 2D array of features into 3D sliding windows.

    Args:
        data: shape (n_days, n_features), already scaled 0-1
        lookback: how many past days to look at (e.g., 60)
        target_idx: column index of the target (Close = 3)

    Returns:
        X: shape (n_samples, lookback, n_features)
        y: shape (n_samples,)
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback : i])    # Past 60 days
        y.append(data[i, target_idx])        # Day 61's close
    return np.array(X), np.array(y)


def prepare_lstm_data():
    """Main pipeline."""

    # Ensure output folders exist
    os.makedirs(SCALERS_DIR, exist_ok=True)
    os.makedirs(ARRAYS_DIR, exist_ok=True)

    # Load master dataset
    print(f"📥 Loading {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    print(f"   Total rows: {len(df)}")

    # Containers for combined data
    all_X, all_y = [], []

    tickers = df["Ticker"].unique()
    print(f"\n📊 Processing {len(tickers)} tickers...")

    for ticker in tickers:
        # Get just this stock's data, sorted by date
        stock_df = df[df["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)

        # Extract OHLCV
        ohlcv = stock_df[FEATURES].values    # shape: (n_days, 5)

        # Scale each stock independently (0 to 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(ohlcv)

        # Save the scaler — we'll need it to inverse predictions later
        scaler_path = os.path.join(SCALERS_DIR, f"{ticker.replace('.', '_')}.pkl")
        joblib.dump(scaler, scaler_path)

        # Find Close column index (should be 3: Open=0, High=1, Low=2, Close=3, Volume=4)
        close_idx = FEATURES.index(TARGET_COL)

        # Build sliding windows
        X, y = create_sequences(scaled, LOOKBACK, close_idx)
        print(f"  ✅ {ticker:15s} → {X.shape[0]:5d} windows  (scaler saved)")

        all_X.append(X)
        all_y.append(y)

    # Combine all stocks
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"\n📦 Combined shape:")
    print(f"   X: {X.shape}  (samples, timesteps, features)")
    print(f"   y: {y.shape}  (targets)")

    # Chronological split — NO shuffling for time series
    split_idx = int(len(X) * TRAIN_SPLIT)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\n✂️  Train/Test split:")
    print(f"   Train: {X_train.shape[0]:5d} samples ({TRAIN_SPLIT*100:.0f}%)")
    print(f"   Test:  {X_test.shape[0]:5d} samples ({(1-TRAIN_SPLIT)*100:.0f}%)")

    # Save the arrays
    np.save(os.path.join(ARRAYS_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(ARRAYS_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(ARRAYS_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(ARRAYS_DIR, "y_test.npy"), y_test)
    print(f"\n💾 Arrays saved to {ARRAYS_DIR}")

    return X_train, X_test, y_train, y_test


# ───── TEST ─────
if __name__ == "__main__":
    print("=" * 60)
    print("LSTM DATA PREPARATION — TELEPATHIA 7.0")
    print("=" * 60)

    X_train, X_test, y_train, y_test = prepare_lstm_data()

    print("\n" + "=" * 60)
    print("✅ READY FOR MODEL TRAINING")
    print("=" * 60)
    print(f"\nA single training sample looks like:")
    print(f"  X[0] shape: {X_train[0].shape}   → 60 days × 5 features")
    print(f"  y[0]      : {y_train[0]:.4f}    → scaled close price (0-1)")