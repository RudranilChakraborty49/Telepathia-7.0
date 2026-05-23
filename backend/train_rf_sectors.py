"""
train_rf_sectors.py
-------------------
Trains SECTOR-SPECIFIC Random Forest sub-models.

Methodology (novel for NSE BTech-level papers):
  - Group tickers into sector buckets (Banking, IT, FMCG, Other)
  - Train one RF per sector group with the same features
  - Save each model separately for later routing
  - Generate per-sector performance comparison plot
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from config import SECTOR_GROUPS, SECTOR_LIST


# ───── PATHS ─────
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "research", "dataset", "master_dataset.csv")
SECTOR_MODELS_DIR = os.path.join(HERE, "saved_models", "sector_models")
PLOTS_DIR = os.path.join(HERE, "..", "research", "paper", "figures")

# ───── FEATURE COLUMNS (same as main RF) ─────
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


def load_full_dataset():
    """Load master dataset and tag each row with sector group."""
    print(f"📥 Loading {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Tag each row by sector
    df["Sector"] = df["Ticker"].map(SECTOR_GROUPS)

    # Drop any tickers not in our sector mapping
    df = df[df["Sector"].notna()]

    print(f"   Total samples: {len(df)}")
    print(f"\n📊 Per-sector sample counts:")
    for sector in SECTOR_LIST:
        n = (df["Sector"] == sector).sum()
        print(f"   {sector:10s}: {n:5d} samples")

    return df


def chronological_split(sector_df, train_split=TRAIN_SPLIT):
    """Split a sector's data chronologically (no shuffling)."""
    sector_df = sector_df.sort_values("Date").reset_index(drop=True)
    X = sector_df[FEATURE_COLS].values
    y = sector_df[TARGET_COL].astype(int).values

    split_idx = int(len(sector_df) * train_split)
    return (
        X[:split_idx], X[split_idx:],
        y[:split_idx], y[split_idx:],
    )


def train_sector_model(X_train, y_train, sector_name):
    """Train one RF model for one sector."""
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def evaluate_sector(rf, X_test, y_test, sector_name):
    """Compute test metrics for one sector model."""
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, list(rf.classes_).index(1)]

    acc = accuracy_score(y_test, y_pred)

    # AUC needs both classes present in test set
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = np.nan    # Edge case: only one class in test

    return {
        "sector": sector_name,
        "n_train": len(y_test) * 4,    # approx, since 80/20 split
        "n_test": len(y_test),
        "accuracy": acc,
        "auc": auc,
        "y_pred": y_pred,
        "y_test": y_test,
    }


def plot_sector_comparison(results, save_path):
    """Bar chart comparing AUC across sectors + global baseline."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    sectors = [r["sector"] for r in results]
    aucs = [r["auc"] for r in results]
    accs = [r["accuracy"] * 100 for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # AUC plot
    colors = ["#10b981", "#3b82f6", "#f59e0b", "#8b5cf6"]
    bars1 = ax1.bar(sectors, aucs, color=colors[:len(sectors)])
    ax1.axhline(0.50, color="gray", linestyle="--", alpha=0.7, label="Random baseline")
    ax1.axhline(0.5343, color="red", linestyle=":", alpha=0.7, label="Global model (0.534)")
    ax1.set_title("Per-Sector ROC-AUC", fontsize=13, fontweight="bold")
    ax1.set_ylabel("AUC")
    ax1.set_ylim(0.40, max(0.75, max(aucs) + 0.05))
    ax1.legend(loc="lower right")
    for bar, val in zip(bars1, aucs):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                 f"{val:.3f}", ha="center", fontweight="bold")

    # Accuracy plot
    bars2 = ax2.bar(sectors, accs, color=colors[:len(sectors)])
    ax2.axhline(50, color="gray", linestyle="--", alpha=0.7, label="Random baseline")
    ax2.axhline(50.27, color="red", linestyle=":", alpha=0.7, label="Global model (50.3%)")
    ax2.set_title("Per-Sector Accuracy", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(40, max(70, max(accs) + 5))
    ax2.legend(loc="lower right")
    for bar, val in zip(bars2, accs):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                 f"{val:.1f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"📊 Sector comparison plot saved → {save_path}")
    plt.show()


def main():
    print("=" * 60)
    print("SECTOR-AWARE RF SUB-MODELS — TELEPATHIA 7.0")
    print("=" * 60)

    os.makedirs(SECTOR_MODELS_DIR, exist_ok=True)

    # Load full dataset
    df = load_full_dataset()

    results = []
    print("\n" + "=" * 60)
    print("TRAINING PER-SECTOR MODELS")
    print("=" * 60)

    for sector in SECTOR_LIST:
        print(f"\n🌲 Training {sector} model...")
        sector_df = df[df["Sector"] == sector]

        if len(sector_df) < 100:
            print(f"   ⚠️ Skipping {sector} — too few samples ({len(sector_df)})")
            continue

        X_train, X_test, y_train, y_test = chronological_split(sector_df)
        print(f"   Samples: train={len(X_train)}, test={len(X_test)}")

        rf = train_sector_model(X_train, y_train, sector)
        result = evaluate_sector(rf, X_test, y_test, sector)
        results.append(result)

        print(f"   🎯 Accuracy: {result['accuracy']*100:.2f}%")
        print(f"   📈 AUC:      {result['auc']:.4f}")

        # Save model
        model_path = os.path.join(SECTOR_MODELS_DIR, f"rf_{sector.lower()}.pkl")
        joblib.dump(rf, model_path)
        print(f"   💾 Saved → {model_path}")

    # ── Summary Table ──
    print("\n" + "=" * 60)
    print("SUMMARY — PER-SECTOR PERFORMANCE")
    print("=" * 60)
    print(f"\n{'Sector':<12}{'Test N':>10}{'Accuracy':>12}{'AUC':>10}")
    print("-" * 44)
    for r in results:
        print(f"{r['sector']:<12}{r['n_test']:>10}"
              f"{r['accuracy']*100:>11.2f}%{r['auc']:>10.4f}")

    # ── Comparison plot ──
    plot_sector_comparison(
        results,
        os.path.join(PLOTS_DIR, "sector_model_comparison.png"),
    )

    print("\n✅ All sector sub-models trained and saved")
    print("=" * 60)


if __name__ == "__main__":
    main()