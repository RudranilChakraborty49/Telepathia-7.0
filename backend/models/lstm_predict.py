"""
lstm_predict.py
---------------
Loads the trained LSTM model and generates BUY/SELL/HOLD signals
for any NSE stock.

Pipeline:
  1. Load trained model + saved scaler
  2. Fetch fresh data for the requested stock
  3. Add indicators (for feature compatibility)
  4. Take last 60 days, scale them
  5. Predict next-day close
  6. Convert prediction → signal + confidence
"""

import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))    # so we can import from utils

from utils.data_fetcher import get_stock_data

# ───── PATHS ─────
MODEL_PATH = os.path.join(HERE, "..", "saved_models", "lstm_model.keras")
SCALERS_DIR = os.path.join(HERE, "..", "saved_models", "scalers")

# ───── CONFIG ─────
LOOKBACK = 60
FEATURES = ["Open", "High", "Low", "Close", "Volume"]
CLOSE_IDX = 3    # Close is the 4th column

# ───── SIGNAL THRESHOLDS ─────
STRONG_BUY_THRESHOLD = 0.015     # +1.5%
BUY_THRESHOLD = 0.005            # +0.5%
SELL_THRESHOLD = -0.005          # -0.5%
STRONG_SELL_THRESHOLD = -0.015   # -1.5%


def load_lstm_artifacts(ticker: str):
    """Load the trained model and the ticker-specific scaler."""
    model = load_model(MODEL_PATH)

    scaler_filename = f"{ticker.replace('.', '_')}.pkl"
    scaler_path = os.path.join(SCALERS_DIR, scaler_filename)

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"No scaler found for {ticker}. "
            f"This ticker wasn't part of training. "
            f"Train on it first via build_dataset.py + lstm_data_prep.py."
        )

    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_next_day(ticker: str) -> dict:
    """
    Generate a Buy/Sell/Hold signal for the next trading day.

    Returns a dict with:
        ticker, current_price, predicted_price, expected_return_pct,
        signal, confidence_pct
    """
    # 1. Load model + scaler
    model, scaler = load_lstm_artifacts(ticker)

    # 2. Get fresh data (last few months is enough for a 60-day window)
    df = get_stock_data(ticker, period="6mo")
    if df.empty or len(df) < LOOKBACK:
        raise ValueError(f"Not enough data for {ticker}")

    # 3. Take the last 60 days of OHLCV
    last_60 = df[FEATURES].tail(LOOKBACK).values    # shape: (60, 5)

    # 4. Scale using THIS ticker's saved scaler
    scaled = scaler.transform(last_60)              # shape: (60, 5)

    # 5. Reshape to (1, 60, 5) — model expects batch dimension
    model_input = scaled.reshape(1, LOOKBACK, len(FEATURES))

    # 6. Predict (returns a scaled value)
    predicted_scaled = model.predict(model_input, verbose=0)[0, 0]

    # 7. Inverse transform — convert back to rupees
    #    The scaler expects all 5 columns; we build a dummy row with prediction in Close slot
    dummy = np.zeros((1, len(FEATURES)))
    dummy[0, CLOSE_IDX] = predicted_scaled
    predicted_price = scaler.inverse_transform(dummy)[0, CLOSE_IDX]

    # 8. Current price = today's close
    current_price = df["Close"].iloc[-1]

    # 9. Expected return
    expected_return = (predicted_price - current_price) / current_price

    # 10. Convert to signal
    signal = classify_signal(expected_return)
    confidence = compute_confidence(expected_return)

    return {
        "ticker": ticker,
        "current_price": round(float(current_price), 2),
        "predicted_price": round(float(predicted_price), 2),
        "expected_return_pct": round(float(expected_return * 100), 2),
        "signal": signal,
        "confidence_pct": round(float(confidence), 1),
    }


def classify_signal(expected_return: float) -> str:
    """Map a return value to a discrete signal."""
    if expected_return > STRONG_BUY_THRESHOLD:
        return "STRONG BUY"
    elif expected_return > BUY_THRESHOLD:
        return "BUY"
    elif expected_return < STRONG_SELL_THRESHOLD:
        return "STRONG SELL"
    elif expected_return < SELL_THRESHOLD:
        return "SELL"
    else:
        return "HOLD"


def compute_confidence(expected_return: float) -> float:
    """
    Confidence increases as the prediction gets further from zero.
    Caps at 100% when the predicted move is >= 3%.
    """
    max_move = 0.03    # 3%
    return min(abs(expected_return) / max_move, 1.0) * 100


# ───── TEST BLOCK ─────
if __name__ == "__main__":
    print("=" * 60)
    print("LSTM PREDICTION — TELEPATHIA 7.0")
    print("=" * 60)

    test_tickers = [
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
    ]

    print(f"\nGenerating signals for {len(test_tickers)} stocks...\n")
    print(f"{'TICKER':<15}{'CURRENT':>12}{'PREDICTED':>12}{'RETURN':>10}{'SIGNAL':>15}{'CONF':>8}")
    print("-" * 75)

    for ticker in test_tickers:
        try:
            result = predict_next_day(ticker)
            print(
                f"{result['ticker']:<15}"
                f"{result['current_price']:>12.2f}"
                f"{result['predicted_price']:>12.2f}"
                f"{result['expected_return_pct']:>9.2f}%"
                f"{result['signal']:>15}"
                f"{result['confidence_pct']:>7.1f}%"
            )
        except Exception as e:
            print(f"{ticker:<15} ❌ Error: {e}")