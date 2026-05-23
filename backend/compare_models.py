"""
compare_models.py — RF vs XGBoost, leak-free date-based cross-validation
-------------------------------------------------------------------------
Telepathia 7.0 — head-to-head model comparison for the research paper.

WHY THIS SCRIPT EXISTS
  A naive sklearn TimeSeriesSplit on master_dataset.csv LEAKS, because the
  dataset is 49 stocks stacked into one table. TimeSeriesSplit cuts by ROW
  index, so the same calendar date lands in both train and test via
  different stocks (date-straddle leakage — see handoff Section 9).

  This script folds by DATE instead. Unique trading dates are cut into 5
  expanding windows; every row is assigned to train/test by its date, so
  no calendar date can ever straddle the split.

OUTPUT
  Per-fold AUC for both models + mean ± std summary. The std matters as
  much as the mean: it shows fold-to-fold stability.
"""

import os
import json
import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

# ───── PATHS ─────
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "research", "dataset", "master_dataset.csv")
OUT_PATH = os.path.join(HERE, "saved_models", "model_comparison.json")

# ───── FEATURE SET (30) — 26 original + 4 market-context ─────
FEATURE_COLS = [
    "RSI", "MACD", "MACD_signal", "BB_pctB", "EMA_ratio",
    "Volume_change", "ATR", "Stoch_K", "Williams_R", "ROC",
    "OBV_change", "Return_1d", "Return_3d", "Return_5d",
    "Day_Of_Week", "Days_To_Weekly_Expiry", "Days_To_Monthly_Expiry",
    "Is_Expiry_Day", "Is_Monthly_Expiry_Week", "Month_Of_Year", "Quarter",
    "Nifty_Return_1d", "Nifty_Return_5d", "Nifty_Volatility_20d",
    "Sector_Return_1d", "Stock_vs_Nifty_RS",
    # ── new market-context features ──
    "India_VIX", "India_VIX_change_1d", "USDINR_change_1d", "Crude_change_1d",
]
TARGET_COL = "Target"

# ───── CV CONFIG ─────
N_SPLITS = 5
VALID_FRACTION = 0.15   # tail of each fold's TRAIN dates -> XGB early-stop set

# ───── HYPERPARAMETERS (match the standalone training scripts) ─────
RF_PARAMS = dict(
    n_estimators=300, max_depth=12, min_samples_split=10,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
XGB_PARAMS = dict(
    n_estimators=1000, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8,
    objective="binary:logistic", eval_metric="auc",
    early_stopping_rounds=50, random_state=42, n_jobs=-1,
)


def date_based_folds(unique_dates, n_splits):
    """Yield (train_dates, test_dates) as expanding windows over DATES.

    Mirrors sklearn's TimeSeriesSplit logic, but the unit is a date, not
    a row. Train always grows; test is the next contiguous date block.
    """
    n = len(unique_dates)
    fold_size = n // (n_splits + 1)   # +1 so fold 1 has a real training set
    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        test_end = fold_size * (i + 1) if i < n_splits else n
        train_dates = unique_dates[:train_end]
        test_dates = unique_dates[train_end:test_end]
        yield train_dates, test_dates


def split_xy(df, dates):
    """Select all rows whose Date is in `dates`; return X, y arrays."""
    mask = df["Date"].isin(dates)
    sub = df[mask]
    X = sub[FEATURE_COLS].values
    y = sub[TARGET_COL].astype(int).values
    return X, y


def metrics(y_true, y_pred, y_prob):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "auc":       roc_auc_score(y_true, y_prob),
    }


def run_rf(X_tr, y_tr, X_te):
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    y_prob = rf.predict_proba(X_te)[:, list(rf.classes_).index(1)]
    return y_pred, y_prob


def run_xgb(X_tr, y_tr, X_te, train_dates, df):
    """Fit XGBoost with a date-based early-stopping validation slice.

    The validation slice is the LAST VALID_FRACTION of this fold's train
    dates — still leak-free, still date-ordered.
    """
    cut = train_dates[int(len(train_dates) * (1 - VALID_FRACTION))]
    inner_train_dates = train_dates[train_dates < cut]
    valid_dates = train_dates[train_dates >= cut]

    X_itr, y_itr = split_xy(df, inner_train_dates)
    X_val, y_val = split_xy(df, valid_dates)

    # scale_pos_weight = #neg / #pos on the inner-training set
    n_pos = int((y_itr == 1).sum())
    n_neg = int((y_itr == 0).sum())
    params = dict(XGB_PARAMS)
    params["scale_pos_weight"] = n_neg / n_pos

    model = xgb.XGBClassifier(**params)
    model.fit(X_itr, y_itr, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, list(model.classes_).index(1)]
    return y_pred, y_prob, model.best_iteration + 1


def summarise(name, fold_metrics):
    """Print mean ± std for every metric in a list of per-fold dicts."""
    print(f"\n{name}")
    keys = fold_metrics[0].keys()
    summary = {}
    for k in keys:
        vals = np.array([m[k] for m in fold_metrics])
        summary[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
        print(f"  {k.upper():10s}: {vals.mean():.4f}  ± {vals.std():.4f}")
    return summary


def main():
    print("=" * 70)
    print("MODEL COMPARISON — RF vs XGBOOST  (leak-free date-based CV)")
    print("=" * 70)

    print(f"\n📥 Loading {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    unique_dates = np.sort(df["Date"].unique())
    print(f"   Rows          : {len(df)}")
    print(f"   Unique dates  : {len(unique_dates)}")
    print(f"   Features      : {len(FEATURE_COLS)}")
    print(f"   CV folds      : {N_SPLITS}  (expanding window, by DATE)")

    rf_metrics, xgb_metrics = [], []

    for fold, (train_dates, test_dates) in enumerate(
            date_based_folds(unique_dates, N_SPLITS), start=1):

        X_tr, y_tr = split_xy(df, train_dates)
        X_te, y_te = split_xy(df, test_dates)

        # ── leak guard: no date may appear in both train and test ──
        overlap = set(train_dates) & set(test_dates)
        assert not overlap, f"LEAK: {len(overlap)} dates in both sets!"

        print("\n" + "-" * 70)
        print(f"FOLD {fold}")
        print(f"  Train: {pd.Timestamp(train_dates[0]).date()} → "
              f"{pd.Timestamp(train_dates[-1]).date()}  "
              f"({len(train_dates)} dates, {len(X_tr)} rows)")
        print(f"  Test : {pd.Timestamp(test_dates[0]).date()} → "
              f"{pd.Timestamp(test_dates[-1]).date()}  "
              f"({len(test_dates)} dates, {len(X_te)} rows)")

        # Random Forest
        rf_pred, rf_prob = run_rf(X_tr, y_tr, X_te)
        rf_m = metrics(y_te, rf_pred, rf_prob)
        rf_metrics.append(rf_m)

        # XGBoost (with its own early-stopping validation slice)
        xgb_pred, xgb_prob, n_trees = run_xgb(X_tr, y_tr, X_te,
                                              train_dates, df)
        xgb_m = metrics(y_te, xgb_pred, xgb_prob)
        xgb_metrics.append(xgb_m)

        print(f"  RF  AUC : {rf_m['auc']:.4f}")
        print(f"  XGB AUC : {xgb_m['auc']:.4f}   ({n_trees} trees)")

    # ── summary ──
    print("\n" + "=" * 70)
    print("FINAL RESULTS  (mean ± std across folds)")
    print("=" * 70)
    rf_summary = summarise("RANDOM FOREST", rf_metrics)
    xgb_summary = summarise("XGBOOST", xgb_metrics)

    rf_auc = rf_summary["auc"]["mean"]
    xgb_auc = xgb_summary["auc"]["mean"]
    delta = xgb_auc - rf_auc
    print("\n" + "-" * 70)
    print(f"Δ AUC (XGB − RF mean) : {delta:+.4f}")
    if abs(delta) < 0.01:
        print("→ Effectively a TIE — consistent with the date-split result.")
    elif delta > 0:
        print("→ XGBoost ahead. Check the per-fold std before claiming a win.")
    else:
        print("→ RF ahead. An honest, reportable result.")
    print("\nNOTE: if these AUCs are ~0.60+, suspect a leak — they should")
    print("      land near the 0.54 you got from the honest date split.")

    # ── persist for the paper ──
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "cv": "expanding-window, date-based, leak-free",
            "n_splits": N_SPLITS,
            "random_forest": rf_summary,
            "xgboost": xgb_summary,
            "delta_auc_mean": delta,
        }, f, indent=2)
    print(f"\n💾 Comparison saved → {OUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
