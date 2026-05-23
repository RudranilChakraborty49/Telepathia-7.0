"""
ensemble.py v5 — DUAL LIVE PATH: LSTM+RF ENSEMBLE  +  XGBOOST
--------------------------------------------------------------
The dashboard now shows TWO predictions per stock:

  1. LSTM + RF Ensemble  — the headline signal (conflict-aware voting)
  2. XGBoost             — an independent second-opinion classifier

WHY THIS CHANGED (v4 -> v5):
  v4 ran XGBoost only, because the LSTM had per-stock scalers for just
  10 of 49 stocks. The scalers were regenerated for all 49 stocks via
  build_dataset.py + lstm_data_prep.py, so the LSTM live path is back.

  The LSTM still needs a per-stock scaler. If one is missing for a given
  ticker, predict() degrades gracefully to an RF-only ensemble for that
  stock instead of crashing — see the fallback in predict().

  get_lstm_signal / get_rf_signal / combine_signals are now LIVE again
  (called by predict()). get_xgb_signal runs alongside them.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from utils.data_fetcher import get_stock_data
from utils.indicators import add_indicators
from utils.calendar_features import add_calendar_features
from utils.market_features import add_market_features
from config import SECTOR_GROUPS

import explainer

# ───── PATHS ─────
LSTM_MODEL_PATH   = os.path.join(HERE, "..", "saved_models", "lstm_model.keras")
RF_MODEL_PATH     = os.path.join(HERE, "..", "saved_models", "rf_model.pkl")
XGB_MODEL_PATH    = os.path.join(HERE, "..", "saved_models", "xgb_model.pkl")
SCALERS_DIR       = os.path.join(HERE, "..", "saved_models", "scalers")
SECTOR_MODELS_DIR = os.path.join(HERE, "..", "saved_models", "sector_models")

# ───── CONFIG ─────
LOOKBACK = 60
LSTM_FEATURES = ["Open", "High", "Low", "Close", "Volume"]
CLOSE_IDX = 3

RF_FEATURES = [
    # Technical (14)
    "RSI", "MACD", "MACD_signal", "BB_pctB", "EMA_ratio",
    "Volume_change", "ATR", "Stoch_K", "Williams_R", "ROC",
    "OBV_change", "Return_1d", "Return_3d", "Return_5d",
    # Calendar (7)
    "Day_Of_Week", "Days_To_Weekly_Expiry", "Days_To_Monthly_Expiry",
    "Is_Expiry_Day", "Is_Monthly_Expiry_Week", "Month_Of_Year", "Quarter",
    # Market regime (5)
    "Nifty_Return_1d", "Nifty_Return_5d", "Nifty_Volatility_20d",
    "Sector_Return_1d", "Stock_vs_Nifty_RS",
]

# ───── THRESHOLDS ─────
LSTM_STRONG_THRESHOLD = 0.015
LSTM_WEAK_THRESHOLD   = 0.005
RF_STRONG_THRESHOLD   = 0.60
RF_WEAK_THRESHOLD     = 0.55
# XGBoost's probabilities cluster tightly around 0.50 (AUC ~0.53), so the
# decision boundary is set narrow to match the model's actual spread.
XGB_STRONG_THRESHOLD  = 0.58   # rare — a clear lean
XGB_WEAK_THRESHOLD    = 0.52   # mild lean


# ═══════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════
_models = {"lstm": None, "rf": None, "scalers": {}, "sector_models": None}
_xgb_cache = {"model": None}


def _load_xgb():
    """Lazy-load the global XGBoost model (covers all 49 stocks)."""
    if _xgb_cache["model"] is None:
        print("Loading XGBoost model...")
        _xgb_cache["model"] = joblib.load(XGB_MODEL_PATH)
    return _xgb_cache["model"]


def _load_models():
    """Load all models needed by the live path: XGBoost, LSTM, and RF."""
    _load_xgb()
    if _models["lstm"] is None:
        print("Loading LSTM model...")
        from tensorflow.keras.models import load_model
        _models["lstm"] = load_model(LSTM_MODEL_PATH)
    if _models["rf"] is None:
        print("Loading Random Forest model...")
        _models["rf"] = joblib.load(RF_MODEL_PATH)


def _load_scaler(ticker):
    """Per-stock LSTM scaler. Raises FileNotFoundError if missing."""
    if ticker not in _models["scalers"]:
        scaler_path = os.path.join(SCALERS_DIR, f"{ticker.replace('.', '_')}.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"No scaler for {ticker}")
        _models["scalers"][ticker] = joblib.load(scaler_path)
    return _models["scalers"][ticker]


# ═══════════════════════════════════════════════════════════════════
# XGBOOST SIGNAL — independent second opinion
# ═══════════════════════════════════════════════════════════════════
def get_xgb_signal(df: pd.DataFrame) -> dict:
    """Generate the directional signal from the global XGBoost model."""
    model = _load_xgb()
    latest = df[RF_FEATURES].iloc[-1].values.reshape(1, -1)
    proba = model.predict_proba(latest)[0]
    up_idx = list(model.classes_).index(1)
    prob_up = float(proba[up_idx])

    if prob_up > XGB_STRONG_THRESHOLD:
        direction, strength, signal = "UP", "STRONG", "STRONG BUY"
    elif prob_up > XGB_WEAK_THRESHOLD:
        direction, strength, signal = "UP", "WEAK", "BUY"
    elif prob_up < (1 - XGB_STRONG_THRESHOLD):
        direction, strength, signal = "DOWN", "STRONG", "STRONG SELL"
    elif prob_up < (1 - XGB_WEAK_THRESHOLD):
        direction, strength, signal = "DOWN", "WEAK", "SELL"
    else:
        direction, strength, signal = "NEUTRAL", "NONE", "HOLD"

    # Honest confidence: distance from a coin-flip, mapped into a 50-75 band.
    edge = abs(prob_up - 0.50)
    if direction == "NEUTRAL":
        confidence = 30
    else:
        confidence = int(round(min(75, 50 + edge * 100)))

    return {
        "direction": direction,
        "strength": strength,
        "prob_up": prob_up,
        "signal": signal,
        "confidence": confidence,
    }


# ═══════════════════════════════════════════════════════════════════
# FULL PIPELINE — LSTM+RF ENSEMBLE  +  XGBOOST  (both shown live)
# ═══════════════════════════════════════════════════════════════════
def predict(ticker: str) -> dict:
    """Run BOTH the LSTM+RF ensemble and XGBoost for a single ticker."""
    # 1. Fetch fresh data
    df = get_stock_data(ticker, period="2y")
    if df.empty or len(df) < LOOKBACK + 50:
        raise ValueError(f"Not enough data for {ticker}")

    # 2. Add ALL features (needed by RF + XGBoost)
    df_enriched = add_indicators(df)
    df_enriched = add_market_features(df_enriched, ticker, period="2y")
    df_enriched = add_calendar_features(df_enriched)
    df_enriched = df_enriched.dropna()
    if len(df_enriched) < 1:
        raise ValueError(f"No usable rows after feature engineering for {ticker}")

    current_price = float(df["Close"].iloc[-1])

    # 3. XGBoost signal (independent model)
    xgb_result = get_xgb_signal(df_enriched)

    # 4. LSTM + RF ensemble. The LSTM needs a per-stock scaler; if one is
    #    missing we degrade gracefully to an RF-only ensemble for this
    #    stock instead of crashing the whole prediction.
    lstm_result = None
    lstm_error = None
    try:
        lstm_result = get_lstm_signal(ticker, df)   # raw OHLCV df, not enriched
    except Exception as e:
        lstm_error = str(e)

    rf_result = get_rf_signal(df_enriched, ticker)

    if lstm_result is not None:
        ens = combine_signals(lstm_result, rf_result)
        lstm_block = {
            "direction":           lstm_result["direction"],
            "strength":            lstm_result["strength"],
            "expected_return_pct": round(lstm_result["expected_return"] * 100, 2),
            "predicted_price":     round(lstm_result["predicted_price"], 2),
            "available":           True,
        }
    else:
        # RF-only fallback for stocks with no LSTM scaler
        rf_dir = rf_result["direction"]
        if rf_dir == "NEUTRAL":
            ens = {"signal": "HOLD", "confidence": 15,
                   "reason": "LSTM unavailable; RF neutral"}
        else:
            base = "BUY" if rf_dir == "UP" else "SELL"
            ens = {"signal": base,
                   "confidence": 50 if rf_result["strength"] == "STRONG" else 35,
                   "reason": f"LSTM unavailable; RF-only signals {rf_dir}"}
        lstm_block = {
            "direction": "N/A", "strength": "NONE",
            "expected_return_pct": None, "predicted_price": None,
            "available": False, "error": lstm_error,
        }

    # 5. SHAP explanation (TreeExplainer — XGBoost)
    try:
        explanation = explainer.explain_prediction(df_enriched, ticker, top_n=3)
    except Exception as e:
        explanation = {
            "model_used": "error",
            "top_bullish": [],
            "top_bearish": [],
            "summary": f"Explanation unavailable: {e}",
        }

    # 6. Assemble result — LSTM+RF ensemble and XGBoost as separate blocks
    return {
        "ticker": ticker,
        "sector": SECTOR_GROUPS.get(ticker, "Unknown"),
        "current_price": round(current_price, 2),
        "predicted_price": lstm_block["predicted_price"],
        "model": "LSTM+RF Ensemble + XGBoost",

        "lstm": lstm_block,
        "rf": {
            "direction":   rf_result["direction"],
            "strength":    rf_result["strength"],
            "prob_up_pct": round(rf_result["prob_up"] * 100, 1),
            "model_used":  rf_result["model_used"],
        },
        # LSTM + RF ensemble — the headline signal
        "ensemble": {
            "signal":     ens["signal"],
            "confidence": ens["confidence"],
            "reason":     ens["reason"],
        },
        # XGBoost — independent second opinion
        "xgb": {
            "direction":   xgb_result["direction"],
            "strength":    xgb_result["strength"],
            "prob_up_pct": round(xgb_result["prob_up"] * 100, 1),
            "signal":      xgb_result["signal"],
            "confidence":  xgb_result["confidence"],
        },
        "explanation": explanation,
    }


# ═══════════════════════════════════════════════════════════════════
# LSTM SIGNAL — live path component
# ═══════════════════════════════════════════════════════════════════
def get_lstm_signal(ticker: str, df: pd.DataFrame) -> dict:
    """Classify LSTM expected return into UP/DOWN/NEUTRAL."""
    _load_models()
    scaler = _load_scaler(ticker)

    last_60 = df[LSTM_FEATURES].tail(LOOKBACK).values
    scaled = scaler.transform(last_60)
    model_input = scaled.reshape(1, LOOKBACK, len(LSTM_FEATURES))

    predicted_scaled = _models["lstm"].predict(model_input, verbose=0)[0, 0]

    dummy = np.zeros((1, len(LSTM_FEATURES)))
    dummy[0, CLOSE_IDX] = predicted_scaled
    predicted_price = scaler.inverse_transform(dummy)[0, CLOSE_IDX]

    current_price = df["Close"].iloc[-1]
    expected_return = (predicted_price - current_price) / current_price

    if expected_return > LSTM_STRONG_THRESHOLD:
        direction, strength = "UP", "STRONG"
    elif expected_return > LSTM_WEAK_THRESHOLD:
        direction, strength = "UP", "WEAK"
    elif expected_return < -LSTM_STRONG_THRESHOLD:
        direction, strength = "DOWN", "STRONG"
    elif expected_return < -LSTM_WEAK_THRESHOLD:
        direction, strength = "DOWN", "WEAK"
    else:
        direction, strength = "NEUTRAL", "NONE"

    return {
        "direction": direction,
        "strength": strength,
        "expected_return": float(expected_return),
        "current_price": float(current_price),
        "predicted_price": float(predicted_price),
    }


# ═══════════════════════════════════════════════════════════════════
# RANDOM FOREST SIGNAL — live path component
# ═══════════════════════════════════════════════════════════════════
def get_rf_signal(df: pd.DataFrame, ticker: str = None) -> dict:
    """Generate RF signal from the global Random Forest model."""
    _load_models()
    model = _models["rf"]

    latest = df[RF_FEATURES].iloc[-1].values.reshape(1, -1)
    proba = model.predict_proba(latest)[0]
    up_idx = list(model.classes_).index(1)
    prob_up = proba[up_idx]

    if prob_up > RF_STRONG_THRESHOLD:
        direction, strength = "UP", "STRONG"
    elif prob_up > RF_WEAK_THRESHOLD:
        direction, strength = "UP", "WEAK"
    elif prob_up < (1 - RF_STRONG_THRESHOLD):
        direction, strength = "DOWN", "STRONG"
    elif prob_up < (1 - RF_WEAK_THRESHOLD):
        direction, strength = "DOWN", "WEAK"
    else:
        direction, strength = "NEUTRAL", "NONE"

    return {
        "direction": direction,
        "strength": strength,
        "prob_up": float(prob_up),
        "model_used": "RF-Global",
    }


# ═══════════════════════════════════════════════════════════════════
# ENSEMBLE VOTING — combines LSTM + RF into the headline signal
# ═══════════════════════════════════════════════════════════════════
def combine_signals(lstm: dict, rf: dict) -> dict:
    """Conflict-aware voting between LSTM and RF."""
    l_dir, r_dir = lstm["direction"], rf["direction"]
    l_str, r_str = lstm["strength"], rf["strength"]

    if l_dir == "NEUTRAL" and r_dir == "NEUTRAL":
        return {"signal": "HOLD", "confidence": 10, "reason": "Both models neutral"}

    if l_dir != "NEUTRAL" and r_dir != "NEUTRAL" and l_dir != r_dir:
        return {"signal": "HOLD", "confidence": 20,
                "reason": f"LSTM says {l_dir}, RF says {r_dir} — conflict"}

    if l_dir == r_dir:
        if l_str == "STRONG" and r_str == "STRONG":
            signal = "STRONG BUY" if l_dir == "UP" else "STRONG SELL"
            confidence = 90
        else:
            signal = "BUY" if l_dir == "UP" else "SELL"
            confidence = 70
        return {"signal": signal, "confidence": confidence,
                "reason": f"Both models agree ({l_dir})"}

    direction = l_dir if r_dir == "NEUTRAL" else r_dir
    strength  = l_str if r_dir == "NEUTRAL" else r_str
    base = "BUY" if direction == "UP" else "SELL"
    signal = f"WEAK {base}"
    confidence = 50 if strength == "STRONG" else 35
    active_model = "LSTM" if r_dir == "NEUTRAL" else "RF"
    return {"signal": signal, "confidence": confidence,
            "reason": f"Only {active_model} signals {direction}"}


# ───── TEST ─────
if __name__ == "__main__":
    print("=" * 80)
    print("ENSEMBLE v5 (LSTM+RF ENSEMBLE + XGBOOST) — TELEPATHIA 7.0")
    print("=" * 80)

    tickers = ["RELIANCE.NS", "ONGC.NS", "TCS.NS", "WIPRO.NS",
               "HDFCBANK.NS", "SBIN.NS", "ITC.NS", "BHARTIARTL.NS"]

    _load_models()

    for ticker in tickers:
        try:
            r = predict(ticker)
            print(f"\n{'=' * 80}")
            print(f"  {ticker}  ({r['sector']})")
            print(f"{'=' * 80}")
            print(f"  Price: Rs.{r['current_price']}")
            print(f"  LSTM:  {r['lstm']['direction']}/{r['lstm']['strength']}  "
                  f"(available={r['lstm']['available']})")
            print(f"  RF:    {r['rf']['direction']}/{r['rf']['strength']}  "
                  f"(P_UP={r['rf']['prob_up_pct']:.1f}%)")
            print(f"  ENSEMBLE SIGNAL: {r['ensemble']['signal']}   "
                  f"({r['ensemble']['confidence']}% confidence)")
            print(f"  {r['ensemble']['reason']}")
            print(f"  XGBoost: {r['xgb']['signal']}  "
                  f"(P_UP={r['xgb']['prob_up_pct']:.1f}%, "
                  f"{r['xgb']['confidence']}% confidence)")
            print(f"  WHY: {r['explanation']['summary']}")
        except Exception as e:
            print(f"FAILED {ticker}: {e}")

    print("\n" + "=" * 80)
    print("Dual live pipeline working")
    print("=" * 80)
