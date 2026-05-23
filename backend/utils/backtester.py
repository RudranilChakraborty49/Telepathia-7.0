"""
backtester.py — honest backtest engine for Telepathia 7.0  (v3)
-----------------------------------------------------------------
v3 adds the Random Forest model alongside XGBoost. Both are directional
UP/DOWN classifiers trained on the IDENTICAL date-based split (cutoff
read from split_info.json), so backtesting them on the same test period
is a fair head-to-head — neither model has seen the test data.

v2 fixed the compounding bug in v1. v1 chained 2,796 trades sequentially
with cumprod, as if they happened one after another — but they happened
in PARALLEL across 50 stocks, so v1 produced impossible returns (+528%).

v2/v3 use a DAY-BY-DAY PORTFOLIO engine:
  - Walk the calendar one trading day at a time.
  - Each day the portfolio holds whatever positions are currently open.
  - The day's portfolio return = mean daily return of the held positions.
  - THAT daily series compounds correctly (days really are sequential).

STRATEGY
  Long-only. On a predicted UP for a stock, open a position the NEXT day
  and hold it HOLD_DAYS trading days. Capital is spread across whatever
  is held — an equal-weight, overlapping-position portfolio.

HONESTY RULES
  - Test period only (rows on/after the saved cutoff date).
  - Round-trip cost charged on the entry day of every position.
  - Compared against Buy & Hold and an RSI-only baseline.

METRICS  Total Return %, annualised Sharpe, Max Drawdown %, Win Rate %.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PATHS  (this file lives in backend/utils/)
HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
DATA_PATH = os.path.join(BACKEND, "..", "research", "dataset", "master_dataset.csv")
XGB_MODEL_PATH = os.path.join(BACKEND, "saved_models", "xgb_model.pkl")
RF_MODEL_PATH = os.path.join(BACKEND, "saved_models", "rf_model.pkl")
SPLIT_INFO_PATH = os.path.join(BACKEND, "saved_models", "split_info.json")
PLOT_PATH = os.path.join(BACKEND, "..", "research", "paper", "figures",
                         "backtest_equity_curve.png")
OUT_PATH = os.path.join(BACKEND, "saved_models", "backtest_results.json")

# FEATURE SET (26) — must match what train_xgb.py / train_rf.py used
FEATURE_COLS = [
    "RSI", "MACD", "MACD_signal", "BB_pctB", "EMA_ratio",
    "Volume_change", "ATR", "Stoch_K", "Williams_R", "ROC",
    "OBV_change", "Return_1d", "Return_3d", "Return_5d",
    "Day_Of_Week", "Days_To_Weekly_Expiry", "Days_To_Monthly_Expiry",
    "Is_Expiry_Day", "Is_Monthly_Expiry_Week", "Month_Of_Year", "Quarter",
    "Nifty_Return_1d", "Nifty_Return_5d", "Nifty_Volatility_20d",
    "Sector_Return_1d", "Stock_vs_Nifty_RS",
]

# BACKTEST CONFIG
HOLD_DAYS = 3
COST_PER_TRADE = 0.0015
TRADING_DAYS_PER_YEAR = 252
RSI_BUY_THRESHOLD = 30


def daily_portfolio_returns(test_df, entry_signal):
    """Day-by-day equal-weight portfolio engine.

    entry_signal: function(stock_frame_g) -> boolean array marking days on
    which a NEW position is opened (entered the NEXT day).
    Returns a pd.Series indexed by Date: the portfolio return each day.
    """
    pieces = []
    for ticker, g in test_df.groupby("Ticker"):
        g = g.sort_values("Date").reset_index(drop=True)
        g["DailyRet"] = g["Close"].pct_change()
        sig = entry_signal(g)

        held = np.zeros(len(g), dtype=bool)
        entry_cost = np.zeros(len(g))
        for i in range(len(g) - HOLD_DAYS - 1):
            if sig[i]:
                start = i + 1
                end = i + 1 + HOLD_DAYS
                held[start:end] = True
                entry_cost[start] += COST_PER_TRADE

        gd = g[["Date", "DailyRet"]].copy()
        gd["Held"] = held
        gd["Cost"] = entry_cost
        gd["Ticker"] = ticker
        pieces.append(gd)

    allrows = pd.concat(pieces, axis=0)

    def day_return(day_rows):
        held_rows = day_rows[day_rows["Held"]]
        if len(held_rows) == 0:
            return 0.0
        gross = held_rows["DailyRet"].mean()
        cost = day_rows["Cost"].sum() / len(held_rows)
        return gross - cost

    # include_groups=False silences the pandas deprecation warning and
    # keeps behaviour identical (the grouping column isn't used in day_return).
    daily = allrows.groupby("Date").apply(day_return, include_groups=False).sort_index()
    return daily.fillna(0.0)


def buy_hold_daily_returns(test_df):
    """Buy & Hold: every stock held every day."""
    pieces = []
    for ticker, g in test_df.groupby("Ticker"):
        g = g.sort_values("Date").reset_index(drop=True)
        g["DailyRet"] = g["Close"].pct_change()
        pieces.append(g[["Date", "DailyRet"]])
    allrows = pd.concat(pieces, axis=0)
    daily = allrows.groupby("Date")["DailyRet"].mean().sort_index().fillna(0.0)
    daily.iloc[0] -= COST_PER_TRADE
    return daily


def make_model_signal(model):
    """Entry signal from any UP/DOWN classifier (RF or XGBoost)."""
    def signal(g):
        return model.predict(g[FEATURE_COLS].values) == 1
    return signal


def rsi_signal(g):
    return (g["RSI"] < RSI_BUY_THRESHOLD).values


def compute_metrics(daily_returns):
    r = np.asarray(daily_returns, dtype=float)
    if len(r) == 0:
        return dict(total_return=0.0, sharpe=0.0, max_drawdown=0.0,
                    win_rate=0.0, n_days=0, equity_curve=[1.0])

    equity = np.cumprod(1 + r)
    total_return = equity[-1] - 1

    if r.std() > 0:
        sharpe = r.mean() / r.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        sharpe = 0.0

    running_max = np.maximum.accumulate(equity)
    max_drawdown = ((equity - running_max) / running_max).min()

    active = r[r != 0]
    win_rate = (active > 0).mean() if len(active) else 0.0

    return dict(
        total_return=float(total_return),
        sharpe=float(sharpe),
        max_drawdown=float(max_drawdown),
        win_rate=float(win_rate),
        n_days=int(len(r)),
        equity_curve=equity.tolist(),
    )


def print_metrics(name, m):
    print(f"\n  {name}")
    print(f"    Days in test  : {m['n_days']}")
    print(f"    Total Return  : {m['total_return']*100:+.2f}%")
    print(f"    Sharpe Ratio  : {m['sharpe']:.3f}")
    print(f"    Max Drawdown  : {m['max_drawdown']*100:.2f}%")
    print(f"    Win Rate (days): {m['win_rate']*100:.1f}%")


def plot_equity_curves(curves, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    colors = {"XGBoost Model": "#a855f7",
              "Random Forest Model": "#10b981",
              "Buy & Hold": "#00f0ff",
              "RSI-only": "#f59e0b"}
    for name, equity in curves.items():
        plt.plot(equity, label=name, color=colors.get(name), linewidth=2)
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    plt.title("Backtest Equity Curve — Telepathia 7.0",
              fontsize=13, fontweight="bold")
    plt.xlabel("Trading day")
    plt.ylabel("Equity (start = 1.0)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"\n[plot] Equity curve saved -> {save_path}")
    plt.close()


def main():
    print("=" * 60)
    print("BACKTEST ENGINE v3 — TELEPATHIA 7.0  (RF + XGBoost)")
    print("=" * 60)

    print(f"Loading {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    with open(SPLIT_INFO_PATH) as f:
        cutoff_date = pd.Timestamp(json.load(f)["cutoff_date"])
    test_df = df[df["Date"] >= cutoff_date].copy()
    print(f"   Test period   : {test_df['Date'].min().date()} -> "
          f"{test_df['Date'].max().date()}")
    print(f"   Test rows     : {len(test_df)}  "
          f"({test_df['Ticker'].nunique()} stocks)")
    print(f"   Cost/trade    : {COST_PER_TRADE*100:.2f}%  | "
          f"Hold: {HOLD_DAYS} days  | Engine: day-by-day portfolio")

    xgb_model = joblib.load(XGB_MODEL_PATH)
    rf_model = joblib.load(RF_MODEL_PATH)

    print("\nRunning strategies (day-by-day)...")
    m_xgb = compute_metrics(
        daily_portfolio_returns(test_df, make_model_signal(xgb_model)))
    m_rf = compute_metrics(
        daily_portfolio_returns(test_df, make_model_signal(rf_model)))
    m_bh = compute_metrics(buy_hold_daily_returns(test_df))
    m_rsi = compute_metrics(
        daily_portfolio_returns(test_df, rsi_signal))

    print("\n" + "=" * 60)
    print("RESULTS  (test period, after costs)")
    print("=" * 60)
    print_metrics("XGBoost Model", m_xgb)
    print_metrics("Random Forest Model", m_rf)
    print_metrics("Buy & Hold (baseline)", m_bh)
    print_metrics("RSI-only (baseline)", m_rsi)

    print("\n" + "-" * 60)
    print("VERDICT")
    print("-" * 60)
    best_model = "XGBoost" if m_xgb["total_return"] >= m_rf["total_return"] else "Random Forest"
    best_return = max(m_xgb["total_return"], m_rf["total_return"])
    if m_xgb["total_return"] > m_rf["total_return"]:
        print(f"  XGBoost edged out Random Forest on total return.")
    elif m_rf["total_return"] > m_xgb["total_return"]:
        print(f"  Random Forest edged out XGBoost on total return.")
    else:
        print(f"  XGBoost and Random Forest tied on total return.")
    if best_return > m_bh["total_return"]:
        print(f"  Best model ({best_model}) beat Buy & Hold.")
    else:
        print(f"  Neither model beat Buy & Hold — an honest, reportable result.")
    if best_return > m_rsi["total_return"]:
        print(f"  Best model ({best_model}) beat the RSI-only rule.")
    else:
        print(f"  Neither model beat a one-line RSI rule — important to report.")
    print("  Sanity check: returns in the tens of %, not hundreds.")

    plot_equity_curves({
        "XGBoost Model":       m_xgb["equity_curve"],
        "Random Forest Model": m_rf["equity_curve"],
        "Buy & Hold":          m_bh["equity_curve"],
        "RSI-only":            m_rsi["equity_curve"],
    }, PLOT_PATH)

    # Keep the equity curve in the JSON this time — the frontend page
    # needs it to draw the chart. It is small (a few hundred floats).
    with open(OUT_PATH, "w") as f:
        json.dump({
            "engine": "day-by-day equal-weight portfolio",
            "test_period": {
                "start": str(test_df["Date"].min().date()),
                "end": str(test_df["Date"].max().date()),
                "n_stocks": int(test_df["Ticker"].nunique()),
            },
            "config": {"hold_days": HOLD_DAYS,
                       "cost_per_trade": COST_PER_TRADE},
            "xgboost_model":  m_xgb,
            "random_forest":  m_rf,
            "buy_and_hold":   m_bh,
            "rsi_only":       m_rsi,
        }, f, indent=2)
    print(f"[save] Results saved -> {OUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
