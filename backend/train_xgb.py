"""
train_xgb.py — BINARY CLASSIFIER (49-stock scale, date-based split)
---------------------------------------------------------------------
Trains the global XGBoost classifier for binary UP/DOWN prediction.

This is the head-to-head counterpart of train_rf.py v4. It uses the
IDENTICAL dataset, feature set, and test cutoff date so the AUC numbers
are directly comparable to the Random Forest baseline.

Differences vs train_rf.py (all deliberate — see chat notes):
  - READS the cutoff date from split_info.json (does NOT recompute it),
    so train/test is byte-identical to the RF run.
  - Adds a date-based VALIDATION slice carved from the training data,
    needed for early stopping. No date straddles train/valid/test.
  - class_weight="balanced"  ->  scale_pos_weight (XGBoost's equivalent).
  - Fixed n_estimators        ->  ceiling + early stopping (tree count
    becomes an OUTPUT, printed for honesty).
  - Conservative, regularized hyperparameters (shallow trees, slow LR)
    because the signal is weak and XGBoost overfits faster than RF.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score,
)

# ───── PATHS (mirrors train_rf.py) ─────
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "research", "dataset", "master_dataset.csv")
MODEL_PATH = os.path.join(HERE, "saved_models", "xgb_model.pkl")
SPLIT_INFO_PATH = os.path.join(HERE, "saved_models", "split_info.json")
PLOTS_DIR = os.path.join(HERE, "..", "research", "paper", "figures")

# ───── HYBRID FEATURE SET (26) — IDENTICAL to train_rf.py ─────
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
# Deliberately conservative. Weak signal -> shallow trees + slow learning
# rate + subsampling, and let early stopping decide how many trees.
N_ESTIMATORS_MAX = 1000      # ceiling only; early stopping ends it sooner
LEARNING_RATE = 0.05
MAX_DEPTH = 4                # shallow (RF used 12; XGBoost overfits faster)
SUBSAMPLE = 0.8              # each tree sees 80% of rows
COLSAMPLE_BYTREE = 0.8       # each tree sees 80% of features
EARLY_STOPPING_ROUNDS = 50
RANDOM_STATE = 42
VALID_FRACTION = 0.15        # last 15% of TRAIN dates -> validation slice


def load_and_split_data():
    """Load master dataset; split into train/valid/test, all by DATE.

    test cutoff is READ from split_info.json (written by train_rf.py) so
    the test set matches the RF run exactly. The validation slice is the
    last VALID_FRACTION of the pre-cutoff dates.
    """
    print(f"📥 Loading {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # --- read the cutoff RF already chose; never recompute it ---
    if not os.path.exists(SPLIT_INFO_PATH):
        raise FileNotFoundError(
            f"split_info.json not found at {SPLIT_INFO_PATH}.\n"
            f"Run train_rf.py first so XGBoost uses the IDENTICAL split."
        )
    with open(SPLIT_INFO_PATH) as f:
        cutoff_date = pd.Timestamp(json.load(f)["cutoff_date"])

    pre_cutoff = df[df["Date"] < cutoff_date]
    test_df    = df[df["Date"] >= cutoff_date]

    # --- carve a date-based validation slice out of the pre-cutoff data ---
    pre_dates = np.sort(pre_cutoff["Date"].unique())
    valid_start = pre_dates[int(len(pre_dates) * (1 - VALID_FRACTION))]

    train_df = pre_cutoff[pre_cutoff["Date"] < valid_start]
    valid_df = pre_cutoff[pre_cutoff["Date"] >= valid_start]

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].astype(int).values
    X_valid = valid_df[FEATURE_COLS].values
    y_valid = valid_df[TARGET_COL].astype(int).values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df[TARGET_COL].astype(int).values

    print(f"   Total samples : {len(df)}")
    print(f"   Test cutoff   : {cutoff_date.date()}  (read from split_info.json)")
    print(f"   Valid start   : {pd.Timestamp(valid_start).date()}")
    print(f"   Train         : {len(X_train)}  (earliest dates)")
    print(f"   Valid         : {len(X_valid)}  (last {VALID_FRACTION*100:.0f}% before cutoff)")
    print(f"   Test          : {len(X_test)}  (on/after cutoff — matches RF)")

    # Honest baseline: always predict the majority class on the test set
    up_rate = y_test.mean()
    baseline = max(up_rate, 1 - up_rate)
    print(f"   Test UP rate  : {up_rate*100:.1f}%  "
          f"→ always-guess baseline accuracy = {baseline*100:.1f}%")

    return X_train, y_train, X_valid, y_valid, X_test, y_test, baseline


def train_xgb(X_train, y_train, X_valid, y_valid):
    print("\n🚀 Training global XGBoost...")
    print(f"   max_depth={MAX_DEPTH}, learning_rate={LEARNING_RATE}, "
          f"n_estimators(ceiling)={N_ESTIMATORS_MAX}")

    # scale_pos_weight = (#negatives / #positives) on the TRAIN set.
    # This is XGBoost's analogue of RF's class_weight="balanced".
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / n_pos
    print(f"   scale_pos_weight={scale_pos_weight:.3f}  "
          f"(neg={n_neg}, pos={n_pos})")

    model = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS_MAX,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="auc",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Early stopping watches the validation AUC and halts when it
    # stops improving for EARLY_STOPPING_ROUNDS consecutive rounds.
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    best_trees = model.best_iteration + 1
    best_valid_auc = model.best_score
    print(f"   ✅ Training complete")
    print(f"   Trees used    : {best_trees}  (early stopping picked this)")
    print(f"   Best valid AUC: {best_valid_auc:.4f}")
    return model


def evaluate(model, X_test, y_test, baseline):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, list(model.classes_).index(1)]

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
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"], cbar=False)
    plt.title("Confusion Matrix — Global XGBoost", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"📊 Confusion matrix saved → {save_path}")
    plt.close()


def plot_feature_importance(model, save_path):
    # XGBoost importance: "gain" = average improvement in loss a feature
    # brought when it was used to split. More meaningful than raw counts.
    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")
    # booster keys are f0, f1, ... -> map back to FEATURE_COLS names
    importances = np.array([score.get(f"f{i}", 0.0)
                            for i in range(len(FEATURE_COLS))])
    if importances.sum() > 0:
        importances = importances / importances.sum()  # normalise to fractions

    order = np.argsort(importances)[::-1]
    feats = [FEATURE_COLS[i] for i in order]
    vals = importances[order] * 100

    plt.figure(figsize=(10, 8))
    bars = plt.barh(feats, vals, color="#a855f7")
    plt.gca().invert_yaxis()
    plt.title("Feature Importance (gain) — Global XGBoost",
              fontsize=13, fontweight="bold")
    plt.xlabel("Importance (%)")
    for bar, val in zip(bars, vals):
        plt.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"📊 Feature importance saved → {save_path}")
    plt.close()

    print("\n🌟 TOP 10 FEATURES (by gain)")
    for feat, val in zip(feats[:10], vals[:10]):
        print(f"  {feat:<22} {val:>5.1f}%")


def main():
    print("=" * 60)
    print("GLOBAL XGBOOST — BINARY UP/DOWN — TELEPATHIA 7.0")
    print("=" * 60)
    (X_train, y_train, X_valid, y_valid,
     X_test, y_test, baseline) = load_and_split_data()

    model = train_xgb(X_train, y_train, X_valid, y_valid)
    y_pred, y_proba, acc, auc = evaluate(model, X_test, y_test, baseline)

    plot_confusion_matrix(y_test, y_pred,
                          os.path.join(PLOTS_DIR, "xgb_confusion_matrix.png"))
    plot_feature_importance(model,
                            os.path.join(PLOTS_DIR, "xgb_feature_importance.png"))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n💾 Model saved → {MODEL_PATH}")

    # Save metrics in the SAME shape as rf_global_metrics.json so the two
    # can be compared (and so a sector script could read either).
    metrics_path = os.path.join(HERE, "saved_models", "xgb_global_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "accuracy": acc,
            "auc": auc,
            "best_iteration": int(model.best_iteration),
            "n_trees_used": int(model.best_iteration) + 1,
        }, f, indent=2)
    print(f"💾 Global metrics saved → {metrics_path}")

    # ── Head-to-head printout vs the RF baseline ──
    rf_metrics_path = os.path.join(HERE, "saved_models", "rf_global_metrics.json")
    if os.path.exists(rf_metrics_path):
        with open(rf_metrics_path) as f:
            rf = json.load(f)
        print("\n" + "=" * 60)
        print("HEAD-TO-HEAD  (same data, same test cutoff)")
        print("=" * 60)
        print(f"  Random Forest  AUC : {rf['auc']:.4f}   "
              f"Acc : {rf['accuracy']*100:.2f}%")
        print(f"  XGBoost        AUC : {auc:.4f}   Acc : {acc*100:.2f}%")
        delta = auc - rf["auc"]
        print(f"  Δ AUC (XGB − RF)   : {delta:+.4f}")
        if abs(delta) < 0.01:
            print("  → Effectively a TIE. Model capacity is not the bottleneck.")
        elif delta > 0:
            print("  → XGBoost ahead. Check it holds up before declaring a winner.")
        else:
            print("  → RF ahead. An honest, reportable result.")
    else:
        print("\n(ℹ️  rf_global_metrics.json not found — run train_rf.py "
              "for the head-to-head comparison.)")
    print("=" * 60)


if __name__ == "__main__":
    main()
