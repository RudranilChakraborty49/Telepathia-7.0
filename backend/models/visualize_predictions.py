"""
visualize_predictions.py
------------------------
Generates a beautiful plot comparing LSTM predictions
to actual stock prices on the test set.

This becomes a key figure in your research paper.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

# ───── PATHS ─────
MODEL_PATH = os.path.join(HERE, "..", "saved_models", "lstm_model.keras")
SCALERS_DIR = os.path.join(HERE, "..", "saved_models", "scalers")
ARRAYS_DIR = os.path.join(HERE, "..", "saved_models", "arrays")
PLOTS_DIR = os.path.join(HERE, "..", "..", "research", "paper", "figures")

CLOSE_IDX = 3


def inverse_scale(values, scaler):
    """Convert scaled values (0-1) back to rupee prices."""
    dummy = np.zeros((len(values), 5))
    dummy[:, CLOSE_IDX] = values
    return scaler.inverse_transform(dummy)[:, CLOSE_IDX]


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load model + test arrays
    print("📥 Loading model and test data...")
    model = load_model(MODEL_PATH)
    X_test = np.load(os.path.join(ARRAYS_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(ARRAYS_DIR, "y_test.npy"))

    # Predict on entire test set
    print("🔮 Generating predictions...")
    y_pred_scaled = model.predict(X_test, verbose=1).flatten()

    # Use RELIANCE's scaler for visualization (approximation since data is mixed)
    # For a cleaner viz, you'd predict per-stock — but this shows the overall trend
    sample_scaler = joblib.load(os.path.join(SCALERS_DIR, "RELIANCE_NS.pkl"))

    y_test_prices = inverse_scale(y_test, sample_scaler)
    y_pred_prices = inverse_scale(y_pred_scaled, sample_scaler)

    # ── Plot 1: Predicted vs Actual ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    # Show last 200 samples
    n_show = 200
    axes[0].plot(y_test_prices[-n_show:], label="Actual", color="#0ea5e9", linewidth=1.8)
    axes[0].plot(y_pred_prices[-n_show:], label="Predicted", color="#ef4444",
                 linewidth=1.5, linestyle="--", alpha=0.85)
    axes[0].set_title("LSTM Predictions vs Actual Prices (Test Set)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Price (₹) — Scaled by RELIANCE")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # ── Plot 2: Scatter plot of predicted vs actual ──
    axes[1].scatter(y_test_prices, y_pred_prices, s=8, alpha=0.4, color="#10b981")
    min_v, max_v = y_test_prices.min(), y_test_prices.max()
    axes[1].plot([min_v, max_v], [min_v, max_v], "k--", alpha=0.5, label="Perfect Prediction")
    axes[1].set_title("Predicted vs Actual — Scatter View", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Actual Price (₹)")
    axes[1].set_ylabel("Predicted Price (₹)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "lstm_predictions_vs_actual.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"📊 Plot saved → {save_path}")

    # ── Stats ──
    errors = y_pred_prices - y_test_prices
    print("\n" + "=" * 50)
    print("PREDICTION QUALITY STATS")
    print("=" * 50)
    print(f"Mean Absolute Error: ₹{np.mean(np.abs(errors)):.2f}")
    print(f"Median Error:        ₹{np.median(np.abs(errors)):.2f}")
    print(f"Max Error:           ₹{np.max(np.abs(errors)):.2f}")
    print(f"Correlation:         {np.corrcoef(y_test_prices, y_pred_prices)[0,1]:.4f}")

    plt.show()


if __name__ == "__main__":
    main()