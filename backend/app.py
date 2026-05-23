"""
app.py — Telepathia 7.0 REST API (v4 — adds /api/backtest)
------------------------------------------------------------------------------
Flask backend serving ML signals to the React dashboard.

Endpoints:
  GET /api/health            — Server status
  GET /api/tickers           — List of supported NSE stocks
  GET /api/stock/<ticker>    — OHLCV + indicators (for charts)
  GET /api/signal/<ticker>   — Single ticker prediction + explanation
  GET /api/signals/all       — Batch prediction for all supported stocks
  GET /api/sectors           — Sector breakdown
  GET /api/backtest          — Pre-computed backtest results (RF + XGBoost)
"""

import os
import sys
import math
import json
import time
import traceback
from flask import Flask, jsonify, request
from flask_cors import CORS

# ───── PATHS & IMPORTS ─────
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "models"))

from config import NIFTY_50_TICKERS, SECTOR_GROUPS
from utils.data_fetcher import get_stock_data
from utils.indicators import add_indicators
from utils.calendar_features import add_calendar_features
from utils.market_features import add_market_features

from models import ensemble

BACKTEST_RESULTS_PATH = os.path.join(HERE, "saved_models", "backtest_results.json")


# ═══════════════════════════════════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════════════════════════════════
app = Flask(__name__)
CORS(app)


# ═══════════════════════════════════════════════════════════════════
# STARTUP — Pre-load ML models
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("🚀 TELEPATHIA 7.0 API — STARTING UP")
print("=" * 60)
print("Pre-loading ML models (one-time)...")
try:
    ensemble._load_models()
    print("✅ All models loaded and ready")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
print("=" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════
def clean_nans(obj):
    """Recursively replace NaN/Infinity with None (so JSON is valid)."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nans(v) for v in obj]
    return obj


def safe_jsonify(data):
    """jsonify() with NaN-safe encoding."""
    return jsonify(clean_nans(data))


def error_response(message: str, code: int = 400):
    return jsonify({"error": message, "code": code}), code


# ═══════════════════════════════════════════════════════════════════
# ENDPOINT 1: Health Check
# ═══════════════════════════════════════════════════════════════════
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "telepathia-7.0",
        "version": "1.0.0",
        "models_loaded": ensemble._models["lstm"] is not None,
    })


# ═══════════════════════════════════════════════════════════════════
# ENDPOINT 2: List Supported Tickers
# ═══════════════════════════════════════════════════════════════════
@app.route("/api/tickers", methods=["GET"])
def tickers():
    return jsonify({
        "count": len(NIFTY_50_TICKERS),
        "tickers": [
            {"ticker": t, "sector": SECTOR_GROUPS.get(t, "Unknown")}
            for t in NIFTY_50_TICKERS
        ],
    })


# ═══════════════════════════════════════════════════════════════════
# ENDPOINT 3: Stock Data (OHLCV + Indicators for Charts)
# ═══════════════════════════════════════════════════════════════════
@app.route("/api/stock/<ticker>", methods=["GET"])
def stock_data(ticker):
    if ticker not in NIFTY_50_TICKERS:
        return error_response(f"Ticker '{ticker}' not supported", 400)

    period = request.args.get("period", "1y")

    try:
        df = get_stock_data(ticker, period=period)
        if df.empty:
            return error_response(f"No data available for {ticker}", 404)

        df = add_indicators(df)
        df = df.dropna()

        df = df.tail(250).reset_index()
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

        records = df[[
            "Date", "Open", "High", "Low", "Close", "Volume",
            "RSI", "MACD", "MACD_signal", "BB_pctB",
            "EMA_20", "EMA_50", "ATR",
        ]].to_dict(orient="records")

        return safe_jsonify({
            "ticker": ticker,
            "sector": SECTOR_GROUPS.get(ticker, "Unknown"),
            "period": period,
            "rows": len(records),
            "data": records,
        })

    except Exception as e:
        traceback.print_exc()
        return error_response(f"Error fetching {ticker}: {str(e)}", 500)


# ═══════════════════════════════════════════════════════════════════
# ENDPOINT 4: Single Ticker Signal + Explanation
# ═══════════════════════════════════════════════════════════════════
@app.route("/api/signal/<ticker>", methods=["GET"])
def signal(ticker):
    if ticker not in NIFTY_50_TICKERS:
        return error_response(f"Ticker '{ticker}' not supported", 400)

    try:
        t0 = time.time()
        result = ensemble.predict(ticker)
        elapsed = round((time.time() - t0) * 1000)

        result["timing_ms"] = elapsed
        return safe_jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return error_response(f"Prediction failed for {ticker}: {str(e)}", 500)


# ═══════════════════════════════════════════════════════════════════
# ENDPOINT 5: Batch Signals for All Tickers
# ═══════════════════════════════════════════════════════════════════
@app.route("/api/signals/all", methods=["GET"])
def all_signals():
    """
    Returns dual-model predictions for all supported tickers:
      - LSTM + RF ensemble (the headline signal)
      - XGBoost (independent second opinion)
    """
    results = []
    errors = []

    t0 = time.time()
    for ticker in NIFTY_50_TICKERS:
        try:
            r = ensemble.predict(ticker)
            results.append({
                "ticker":           r["ticker"],
                "sector":           r["sector"],
                "current_price":    r["current_price"],
                "predicted_price":  r["predicted_price"],
                # LSTM + RF ensemble (headline)
                "signal":           r["ensemble"]["signal"],
                "confidence":       r["ensemble"]["confidence"],
                "reason":           r["ensemble"]["reason"],
                "lstm_direction":   r["lstm"]["direction"],
                "lstm_available":   r["lstm"]["available"],
                "rf_direction":     r["rf"]["direction"],
                "rf_prob_up":       r["rf"]["prob_up_pct"],
                "rf_model_used":    r["rf"]["model_used"],
                # XGBoost (independent second opinion)
                "xgb_signal":       r["xgb"]["signal"],
                "xgb_confidence":   r["xgb"]["confidence"],
                "xgb_prob_up":      r["xgb"]["prob_up_pct"],
                "xgb_direction":    r["xgb"]["direction"],
            })
        except Exception as e:
            traceback.print_exc()
            errors.append({"ticker": ticker, "error": str(e)})

    elapsed = round((time.time() - t0) * 1000)
    return safe_jsonify({
        "count": len(results),
        "errors": errors,
        "timing_ms": elapsed,
        "signals": results,
    })


# ═══════════════════════════════════════════════════════════════════
# ENDPOINT 6: Sector Breakdown
# ═══════════════════════════════════════════════════════════════════
@app.route("/api/sectors", methods=["GET"])
def sectors():
    sector_map = {}
    for ticker, sector in SECTOR_GROUPS.items():
        sector_map.setdefault(sector, []).append(ticker)

    return jsonify({
        "sectors": [
            {
                "name": sector,
                "tickers": tickers_list,
            }
            for sector, tickers_list in sector_map.items()
        ]
    })


# ═══════════════════════════════════════════════════════════════════
# ENDPOINT 7: Backtest Results (pre-computed by utils/backtester.py)
# ═══════════════════════════════════════════════════════════════════
@app.route("/api/backtest", methods=["GET"])
def backtest():
    """
    Serves the pre-computed backtest results. These are generated offline
    by running `python utils/backtester.py`, which writes
    saved_models/backtest_results.json. The endpoint just reads that file —
    backtesting live would be far too slow for a page load.
    """
    if not os.path.exists(BACKTEST_RESULTS_PATH):
        return error_response(
            "Backtest results not found. Run `python utils/backtester.py` "
            "to generate them.", 404
        )

    try:
        with open(BACKTEST_RESULTS_PATH) as f:
            data = json.load(f)
        return safe_jsonify(data)
    except Exception as e:
        traceback.print_exc()
        return error_response(f"Failed to read backtest results: {str(e)}", 500)


# ═══════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
