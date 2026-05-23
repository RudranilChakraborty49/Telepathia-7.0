"""
train_rf.py v4 — BINARY CLASSIFIER (49-stock scale, date-based split)
---------------------------------------------------------------------
Trains the global Random Forest for binary UP/DOWN prediction.

v4 changes:
  - Date-BASED train/test split (no calendar date straddles the boundary)
  - Saves the test-cutoff date so sector models use the SAME split
  - Honest reporting: AUC primary, accuracy vs always-UP baseline
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score,
)

# ───── PATHS ─────
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "research", "dataset", "master_dataset.csv")
MODEL_PATH = os.path.join(HERE, "saved_models", "rf_model.pkl")
SPLIT_INFO_PATH = os.path.join(HERE, "saved_models", "split_info.json")
PLOTS_DIR = os.path.join(HERE, "..", "research", "paper", "figures")

# ───── HYBRID FEATURE SET (26) ─────
FEATURE_COLS = [
    "RSI", "MACD", "MACD_signal", "BB_pctB", "EMA_ratio",
    "Volume_change", "ATR", "Stoch_K", "Williams_R", "ROC",
    "OBV_change", "Return_1d", "Return_3d", "Return_5d",
    "Day_Of_Week", "Days_To_Weekly_Expiry", "Days_To_Monthly_Expiry",
    "Is_Expiry_Day", "Is_Monthly_Expiry_Week", "Month_Of_Year", "Quarter",
    "Nifty_Return_1d", "Nifty_Return_5d", "Nifty_Volatility_20d",
    "Sector_Return_1d", "Stock_vs_Nifty_RS",
]
TARGET_COL = "Target"

# ───── HYPERPARAMETERS ─────
N_ESTIMATORS = 300
MAX_DEPTH = 12
MIN_SAMPLES_SPLIT = 10
RANDOM_STATE = 42
TRAIN_SPLIT = 0.8


def load_and_split_data():
    """Load master dataset; split by DATE so no date straddles train/test."""
    print(f"📥 Loading {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Find the cutoff DATE at the 80th percentile of unique dates
    unique_dates = np.sort(df["Date"].unique())
    cutoff_date = unique_dates[int(len(unique_dates) * TRAIN_SPLIT)]

    train_df = df[df["Date"] < cutoff_date]
    test_df  = df[df["Date"] >= cutoff_date]

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].astype(int).values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df[TARGET_COL].astype(int).values

    # Persist the cutoff so sector models use the identical split
    os.makedirs(os.path.dirname(SPLIT_INFO_PATH), exist_ok=True)
    with open(SPLIT_INFO_PATH, "w") as f:
        json.dump({"cutoff_date": str(pd.Timestamp(cutoff_date).date())}, f)

    print(f"   Total samples : {len(df)}")
    print(f"   Cutoff date   : {pd.Timestamp(cutoff_date).date()}")
    print(f"   Train         : {len(X_train)}  (before cutoff)")
    print(f"   Test          : {len(X_test)}  (on/after cutoff)")

    # Honest baseline: accuracy of always predicting the majority class
    up_rate = y_test.mean()
    baseline = max(up_rate, 1 - up_rate)
    print(f"   Test UP rate  : {up_rate*100:.1f}%  "
          f"→ always-guess baseline accuracy = {baseline*100:.1f}%")

    return X_train, X_test, y_train, y_test, baseline


def train_rf(X_train, y_train):
    print("\n🌲 Training global Random Forest...")
    print(f"   n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}")
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    print("   ✅ Training complete")
    return rf


def evaluate(rf, X_test, y_test, baseline):
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, list(rf.classes_).index(1)]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)
    print(f"  🎯 Accuracy : {acc*100:.2f}%   (always-guess baseline: {baseline*100:.1f}%)")
    print(f"  📈 ROC-AUC  : {auc:.4f}   (0.50 = random, 1.0 = perfect)")
    print(f"  → AUC is the honest metric; accuracy near baseline means little.")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["DOWN", "UP"], digits=3))
    return y_pred, y_proba, acc, auc


def plot_confusion_matrix(y_test, y_pred, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"], cbar=False)
    plt.title("Confusion Matrix — Global Random Forest", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"📊 Confusion matrix saved → {save_path}")
    plt.close()


def plot_feature_importance(rf, save_path):
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]
    feats = [FEATURE_COLS[i] for i in order]
    vals = importances[order] * 100

    plt.figure(figsize=(10, 8))
    bars = plt.barh(feats, vals, color="#10b981")
    plt.gca().invert_yaxis()
    plt.title("Feature Importance — Global Random Forest", fontsize=13, fontweight="bold")
    plt.xlabel("Importance (%)")
    for bar, val in zip(bars, vals):
        plt.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"📊 Feature importance saved → {save_path}")
    plt.close()

    print("\n🌟 TOP 10 FEATURES")
    for feat, val in zip(feats[:10], vals[:10]):
        print(f"  {feat:<22} {val:>5.1f}%")


def main():
    print("=" * 60)
    print("GLOBAL RANDOM FOREST v4 — BINARY UP/DOWN — TELEPATHIA 7.0")
    print("=" * 60)
    X_train, X_test, y_train, y_test, baseline = load_and_split_data()
    rf = train_rf(X_train, y_train)
    y_pred, y_proba, acc, auc = evaluate(rf, X_test, y_test, baseline)

    plot_confusion_matrix(y_test, y_pred,
                          os.path.join(PLOTS_DIR, "rf_confusion_matrix.png"))
    plot_feature_importance(rf,
                            os.path.join(PLOTS_DIR, "rf_feature_importance.png"))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(rf, MODEL_PATH)
    print(f"\n💾 Model saved → {MODEL_PATH}")

    # Save the global metrics so the sector script can use them as baseline
    metrics_path = os.path.join(HERE, "saved_models", "rf_global_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": acc, "auc": auc}, f)
    print(f"💾 Global metrics saved → {metrics_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()