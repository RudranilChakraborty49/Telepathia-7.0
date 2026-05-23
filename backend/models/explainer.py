"""
explainer.py — SHAP-based Explainable AI  (XGBoost edition)
------------------------------------------------------------
Generates human-readable explanations for each signal:
  - Top 3 features pushing toward UP
  - Top 3 features pushing toward DOWN
  - Natural language interpretations

NOTE: switched from Random Forest to the global XGBoost model.
SHAP's TreeExplainer supports XGBoost natively, so the logic is
unchanged — only the model file and routing differ. One global
model is used for all 49 stocks (no sector routing).
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import shap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from config import SECTOR_GROUPS

# ───── PATHS ─────
# Now points at the global XGBoost model.
XGB_MODEL_PATH    = os.path.join(HERE, "..", "saved_models", "xgb_model.pkl")
SECTOR_MODELS_DIR = os.path.join(HERE, "..", "saved_models", "sector_models")

# ───── ROUTING ─────
# XGBoost-only mode: one global model for all 49 stocks, no sector routing.
USE_SECTOR_MODEL = {}

RF_FEATURES = [
    "RSI", "MACD", "MACD_signal", "BB_pctB", "EMA_ratio",
    "Volume_change", "ATR", "Stoch_K", "Williams_R", "ROC",
    "OBV_change", "Return_1d", "Return_3d", "Return_5d",
    "Day_Of_Week", "Days_To_Weekly_Expiry", "Days_To_Monthly_Expiry",
    "Is_Expiry_Day", "Is_Monthly_Expiry_Week", "Month_Of_Year", "Quarter",
    "Nifty_Return_1d", "Nifty_Return_5d", "Nifty_Volatility_20d",
    "Sector_Return_1d", "Stock_vs_Nifty_RS",
]

# ───── HUMAN-READABLE FEATURE INTERPRETATIONS ─────
FEATURE_INTERPRETATIONS = {
    "RSI": lambda v: (
        f"RSI is overbought ({v:.0f})" if v > 70
        else f"RSI is oversold ({v:.0f})" if v < 30
        else f"RSI is bullish-leaning ({v:.0f})" if v > 50
        else f"RSI is bearish-leaning ({v:.0f})"
    ),
    "MACD": lambda v: (
        "MACD positive (bullish momentum)" if v > 0
        else "MACD negative (bearish momentum)"
    ),
    "MACD_signal": lambda v: (
        "MACD signal line above zero" if v > 0
        else "MACD signal line below zero"
    ),
    "BB_pctB": lambda v: (
        "Price near upper Bollinger Band (stretched)" if v > 0.8
        else "Price near lower Bollinger Band (oversold)" if v < 0.2
        else f"Price within Bollinger range ({v:.2f})"
    ),
    "EMA_ratio": lambda v: (
        f"EMA20 above EMA50 ({v:.3f}) — uptrend" if v > 1.0
        else f"EMA20 below EMA50 ({v:.3f}) — downtrend"
    ),
    "Volume_change": lambda v: (
        f"Volume {v:.1f}x above average (strong activity)" if v > 1.5
        else f"Volume {v:.1f}x below average (low interest)" if v < 0.7
        else f"Volume near average ({v:.1f}x)"
    ),
    "ATR": lambda v: f"ATR = {v:.1f} (volatility measure)",
    "Stoch_K": lambda v: (
        f"Stochastic overbought ({v:.0f})" if v > 80
        else f"Stochastic oversold ({v:.0f})" if v < 20
        else f"Stochastic mid-range ({v:.0f})"
    ),
    "Williams_R": lambda v: (
        f"Williams%R bullish reversal zone ({v:.0f})" if v > -20
        else f"Williams%R bearish reversal zone ({v:.0f})" if v < -80
        else f"Williams%R neutral ({v:.0f})"
    ),
    "ROC": lambda v: (
        f"Strong upward momentum ({v:+.2f}%)" if v > 3
        else f"Strong downward momentum ({v:+.2f}%)" if v < -3
        else f"Moderate momentum ({v:+.2f}%)"
    ),
    "OBV_change": lambda v: (
        "Accumulation (volume flowing in)" if v > 0
        else "Distribution (volume flowing out)"
    ),
    "Return_1d": lambda v: f"Yesterday's return: {v*100:+.2f}%",
    "Return_3d": lambda v: f"3-day return: {v*100:+.2f}%",
    "Return_5d": lambda v: f"5-day return: {v*100:+.2f}%",
    "Day_Of_Week": lambda v: ["Monday", "Tuesday", "Wednesday",
                              "Thursday", "Friday"][int(v)],
    "Days_To_Weekly_Expiry": lambda v: (
        "Expiry day today!" if v == 0
        else f"{int(v)} days to weekly F&O expiry"
    ),
    "Days_To_Monthly_Expiry": lambda v: f"{int(v)} days to monthly expiry",
    "Is_Expiry_Day": lambda v: "Today is expiry day" if v else "Not expiry day",
    "Is_Monthly_Expiry_Week": lambda v: (
        "In monthly expiry week" if v else "Not in expiry week"
    ),
    "Month_Of_Year": lambda v: f"Month: {int(v)}",
    "Quarter": lambda v: f"Q{int(v)}",
    "Nifty_Return_1d": lambda v: f"Nifty yesterday: {v*100:+.2f}%",
    "Nifty_Return_5d": lambda v: f"Nifty 5-day: {v*100:+.2f}%",
    "Nifty_Volatility_20d": lambda v: (
        f"High Nifty volatility ({v*100:.2f}%)" if v > 0.012
        else f"Low Nifty volatility ({v*100:.2f}%)" if v < 0.008
        else f"Normal Nifty volatility ({v*100:.2f}%)"
    ),
    "Sector_Return_1d": lambda v: f"Sector yesterday: {v*100:+.2f}%",
    "Stock_vs_Nifty_RS": lambda v: (
        f"Outperforming Nifty (RS = {v:.3f})" if v > 1.02
        else f"Underperforming Nifty (RS = {v:.3f})" if v < 0.98
        else f"Tracking Nifty (RS = {v:.3f})"
    ),
}


# ═══════════════════════════════════════════════════════════════════
# Module-level SHAP explainer cache
# ═══════════════════════════════════════════════════════════════════
_explainers = {}


def _load_model_for_ticker(ticker: str):
    """Load the model for this ticker. XGBoost-only mode -> always global."""
    return joblib.load(XGB_MODEL_PATH), "xgboost_global"


def _get_explainer(model, model_key: str):
    """Return cached SHAP explainer for this model (build if missing)."""
    if model_key not in _explainers:
        # TreeExplainer is fast and exact for tree models (RF and XGBoost).
        _explainers[model_key] = shap.TreeExplainer(model)
    return _explainers[model_key]


def _clean_numeric(series: pd.Series) -> pd.Series:
    """
    Force a feature row to clean float64.

    Some feature-engineering steps can leave a value as a string like
    '[5E-1]' (a number wrapped in brackets) instead of a real number.
    SHAP needs a pure numeric array, so we strip brackets/whitespace,
    coerce to numeric, and fill anything unrecoverable with 0.0.
    """
    cleaned = series.copy()
    for idx, val in cleaned.items():
        if isinstance(val, str):
            # strip brackets, spaces, and stray quotes
            stripped = val.strip().strip("[]").strip().strip("'\"")
            cleaned[idx] = stripped
    # coerce everything to numbers; unparseable -> NaN -> 0.0
    cleaned = pd.to_numeric(cleaned, errors="coerce").fillna(0.0)
    return cleaned.astype("float64")


def explain_prediction(df_enriched: pd.DataFrame, ticker: str,
                        top_n: int = 3) -> dict:
    """Generate SHAP-based explanation for the latest prediction."""
    # 1. Load the model (global XGBoost)
    model, model_key = _load_model_for_ticker(ticker)

    # 2. Latest row as a feature vector — cleaned to pure numeric.
    #    (Guards against string values like '[5E-1]' that break SHAP.)
    latest_row = _clean_numeric(df_enriched[RF_FEATURES].iloc[-1])
    latest = latest_row.to_frame().T          # 1-row DataFrame
    latest.columns = RF_FEATURES
    feature_values = latest_row.to_dict()

    # 3. SHAP explainer
    explainer = _get_explainer(model, model_key)

    # 4. Compute SHAP values — pass a clean float array
    shap_values = explainer.shap_values(latest.values.astype("float64"))

    # XGBoost binary: shap_values is usually a 2D array (samples, features).
    # Some versions/objectives return a list or a 3D array — handle all.
    if isinstance(shap_values, list):
        shap_for_up = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            shap_for_up = arr[0, :, 1]
        else:
            shap_for_up = arr[0]

    # 5. Pair features with their SHAP values
    contributions = []
    for feat, shap_val in zip(RF_FEATURES, shap_for_up):
        contributions.append({
            "feature": feat,
            "value": float(feature_values[feat]),
            "shap": float(shap_val),
        })

    # 6. Sort by SHAP: most bullish and most bearish
    sorted_by_shap = sorted(contributions, key=lambda x: x["shap"], reverse=True)
    top_bullish = sorted_by_shap[:top_n]
    top_bearish = sorted_by_shap[-top_n:][::-1]

    # 7. Human-readable messages
    def _build_message(item):
        feat = item["feature"]
        val = item["value"]
        interpreter = FEATURE_INTERPRETATIONS.get(feat)
        try:
            return interpreter(val) if interpreter else f"{feat} = {val:.3f}"
        except Exception:
            # never let a single interpreter crash the whole explanation
            return f"{feat} = {val:.3f}"

    top_bullish_msgs = [{
        "feature": item["feature"],
        "value": round(item["value"], 4),
        "contribution": round(item["shap"], 4),
        "message": _build_message(item),
    } for item in top_bullish if item["shap"] > 0]

    top_bearish_msgs = [{
        "feature": item["feature"],
        "value": round(item["value"], 4),
        "contribution": round(item["shap"], 4),
        "message": _build_message(item),
    } for item in top_bearish if item["shap"] < 0]

    # 8. Natural language summary
    if not top_bullish_msgs and not top_bearish_msgs:
        summary = "No strong driving factors detected."
    elif len(top_bullish_msgs) > len(top_bearish_msgs):
        summary = (f"Bullish drivers outweigh bearish ones. "
                   f"Main reason: {top_bullish_msgs[0]['message']}.")
    elif len(top_bearish_msgs) > len(top_bullish_msgs):
        summary = (f"Bearish drivers outweigh bullish ones. "
                   f"Main reason: {top_bearish_msgs[0]['message']}.")
    else:
        summary = "Mixed signals — bullish and bearish drivers are balanced."

    return {
        "model_used": model_key,
        "top_bullish": top_bullish_msgs,
        "top_bearish": top_bearish_msgs,
        "summary": summary,
    }


# ───── TEST ─────
if __name__ == "__main__":
    from utils.data_fetcher import get_stock_data
    from utils.indicators import add_indicators
    from utils.calendar_features import add_calendar_features
    from utils.market_features import add_market_features

    print("=" * 70)
    print("EXPLAINER (XGBoost) — TELEPATHIA 7.0")
    print("=" * 70)

    test_tickers = ["TCS.NS", "HDFCBANK.NS", "BHARTIARTL.NS", "ITC.NS"]

    for ticker in test_tickers:
        print(f"\n{'='*70}")
        print(f"  EXPLAINING PREDICTION FOR {ticker}")
        print("=" * 70)

        df = get_stock_data(ticker, period="2y")
        df = add_indicators(df)
        df = add_market_features(df, ticker, period="2y")
        df = add_calendar_features(df)
        df = df.dropna()

        explanation = explain_prediction(df, ticker, top_n=3)

        print(f"\nModel used: {explanation['model_used']}")
        print(f"Summary: {explanation['summary']}")

        print("\nTOP BULLISH FACTORS:")
        for item in explanation["top_bullish"]:
            print(f"  + {item['message']}")
            print(f"      (SHAP contribution: {item['contribution']:+.4f})")

        print("\nTOP BEARISH FACTORS:")
        for item in explanation["top_bearish"]:
            print(f"  - {item['message']}")
            print(f"      (SHAP contribution: {item['contribution']:+.4f})")

    print("\n" + "=" * 70)
    print("Explainer working")
    print("=" * 70)
