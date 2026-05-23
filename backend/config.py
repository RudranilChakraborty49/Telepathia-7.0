"""
config.py
---------
Central configuration for Telepathia 7.0 backend.
All tickers, time windows, and hyperparameters live here.
Edit this file to change settings — don't hardcode values elsewhere.

SCOPE: Phase 1 expanded — full Nifty 50 universe (~50 stocks).
NOTE:  Nifty 50 membership changes ~twice yearly. This list reflects
       early-2026 composition. If a ticker fails to fetch, it was likely
       added/removed in a recent index reshuffle — swap it then.
"""

# ───── STOCK UNIVERSE (Phase 1 — full Nifty 50) ─────
# Kept the name NIFTY_50_TICKERS for backward compatibility with the rest
# of the codebase (app.py, ensemble.py, etc. all import this name).
NIFTY_50_TICKERS = [
    # Energy & Oil
    "RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS",
    "BPCL.NS",
    # IT
    "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
    # Banking
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    "INDUSINDBK.NS",
    # Financial Services (non-bank)
    "BAJFINANCE.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS", "SBILIFE.NS",
    "SHRIRAMFIN.NS",
    # FMCG
    "ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS",
    # Auto
    "MARUTI.NS",  "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS",
    "HEROMOTOCO.NS",
    # Pharma & Healthcare
    "SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS", "APOLLOHOSP.NS",
    # Metals
    "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS",
    # Infrastructure / Cement / Construction
    "LT.NS", "ULTRACEMCO.NS", "GRASIM.NS", "ADANIPORTS.NS", "ADANIENT.NS",
    # Consumer / Telecom / Misc
    "BHARTIARTL.NS", "ASIANPAINT.NS", "TITAN.NS", "TRENT.NS", "JIOFIN.NS",
]

# ───── DATA SETTINGS ─────
HISTORICAL_PERIOD = "5y"          # How much history to fetch
DATA_INTERVAL = "1d"              # Daily candles
LOOKBACK_WINDOW = 60              # LSTM uses last 60 days

# ───── LSTM HYPERPARAMETERS ─────
# NOTE: LSTM now predicts next-day RETURN (% change), not absolute price.
#       One pooled model trained across all stocks (returns are scale-free).
LSTM_UNITS = 50
LSTM_EPOCHS = 25
LSTM_BATCH_SIZE = 32
LSTM_TRAIN_SPLIT = 0.8

# ───── RANDOM FOREST HYPERPARAMETERS ─────
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 10
RF_RANDOM_STATE = 42

# ───── ENSEMBLE WEIGHTS ─────
LSTM_WEIGHT = 0.5
RF_WEIGHT = 0.5

# ───── BACKTEST SETTINGS ─────
INITIAL_CAPITAL = 100000          # ₹1,00,000
TRANSACTION_COST = 0.001          # 0.1% per trade

# ───── MARKET INDEX ─────
NIFTY_INDEX = "^NSEI"             # Nifty 50 broad market

# ───── SECTOR INDEX MAPPING (for relative-strength features) ─────
# NSE lacks a clean sector index for some sectors (Infra, diversified
# Financials); those fall back to ^NSEI. The Stock_vs_Nifty_RS feature
# still works — it just compares against the broad market.
SECTOR_INDICES = {
    # Energy & Oil  → Nifty Energy
    "RELIANCE.NS": "^CNXENERGY", "ONGC.NS": "^CNXENERGY",
    "NTPC.NS": "^CNXENERGY", "POWERGRID.NS": "^CNXENERGY",
    "COALINDIA.NS": "^CNXENERGY", "BPCL.NS": "^CNXENERGY",
    # IT → Nifty IT
    "TCS.NS": "^CNXIT", "INFY.NS": "^CNXIT", "HCLTECH.NS": "^CNXIT",
    "WIPRO.NS": "^CNXIT", "TECHM.NS": "^CNXIT",
    # Banking → Nifty Bank
    "HDFCBANK.NS": "^NSEBANK", "ICICIBANK.NS": "^NSEBANK",
    "SBIN.NS": "^NSEBANK", "KOTAKBANK.NS": "^NSEBANK",
    "AXISBANK.NS": "^NSEBANK", "INDUSINDBK.NS": "^NSEBANK",
    # Financial Services → routed to Nifty Bank (close proxy, stable symbol)
    "BAJFINANCE.NS": "^NSEBANK", "BAJAJFINSV.NS": "^NSEBANK",
    "HDFCLIFE.NS": "^NSEBANK", "SBILIFE.NS": "^NSEBANK",
    "SHRIRAMFIN.NS": "^NSEBANK", "JIOFIN.NS": "^NSEBANK",
    # FMCG → Nifty FMCG
    "ITC.NS": "^CNXFMCG", "HINDUNILVR.NS": "^CNXFMCG",
    "NESTLEIND.NS": "^CNXFMCG", "BRITANNIA.NS": "^CNXFMCG",
    "TATACONSUM.NS": "^CNXFMCG",
    # Auto → Nifty Auto
    "MARUTI.NS": "^CNXAUTO", 
    "M&M.NS": "^CNXAUTO", "BAJAJ-AUTO.NS": "^CNXAUTO",
    "EICHERMOT.NS": "^CNXAUTO", "HEROMOTOCO.NS": "^CNXAUTO",
    # Pharma → Nifty Pharma
    "SUNPHARMA.NS": "^CNXPHARMA", "CIPLA.NS": "^CNXPHARMA",
    "DRREDDY.NS": "^CNXPHARMA", "DIVISLAB.NS": "^CNXPHARMA",
    "APOLLOHOSP.NS": "^CNXPHARMA",
    # Metals → Nifty Metal
    "TATASTEEL.NS": "^CNXMETAL", "JSWSTEEL.NS": "^CNXMETAL",
    "HINDALCO.NS": "^CNXMETAL",
    # Infrastructure / Cement → fallback to Nifty (no single clean index)
    "LT.NS": "^NSEI", "ULTRACEMCO.NS": "^NSEI", "GRASIM.NS": "^NSEI",
    "ADANIPORTS.NS": "^NSEI", "ADANIENT.NS": "^NSEI",
    # Consumer / Telecom / Misc → fallback to Nifty
    "BHARTIARTL.NS": "^NSEI", "ASIANPAINT.NS": "^NSEI",
    "TITAN.NS": "^NSEI", "TRENT.NS": "^NSEI",
}

# ───── SECTOR GROUPING FOR SUB-MODELS ─────
# Each ticker → its model-training sector group.
# 9 real sectors now (no more "Other" catch-all).
SECTOR_GROUPS = {
    # Energy
    "RELIANCE.NS": "Energy", "ONGC.NS": "Energy", "NTPC.NS": "Energy",
    "POWERGRID.NS": "Energy", "COALINDIA.NS": "Energy", "BPCL.NS": "Energy",
    # IT
    "TCS.NS": "IT", "INFY.NS": "IT", "HCLTECH.NS": "IT",
    "WIPRO.NS": "IT", "TECHM.NS": "IT",
    # Banking
    "HDFCBANK.NS": "Banking", "ICICIBANK.NS": "Banking", "SBIN.NS": "Banking",
    "KOTAKBANK.NS": "Banking", "AXISBANK.NS": "Banking",
    "INDUSINDBK.NS": "Banking",
    # Financial Services (non-bank)
    "BAJFINANCE.NS": "FinServ", "BAJAJFINSV.NS": "FinServ",
    "HDFCLIFE.NS": "FinServ", "SBILIFE.NS": "FinServ",
    "SHRIRAMFIN.NS": "FinServ", "JIOFIN.NS": "FinServ",
    # FMCG
    "ITC.NS": "FMCG", "HINDUNILVR.NS": "FMCG", "NESTLEIND.NS": "FMCG",
    "BRITANNIA.NS": "FMCG", "TATACONSUM.NS": "FMCG",
    # Auto
    "MARUTI.NS": "Auto", "M&M.NS": "Auto",
    "BAJAJ-AUTO.NS": "Auto", "EICHERMOT.NS": "Auto", "HEROMOTOCO.NS": "Auto",
    # Pharma
    "SUNPHARMA.NS": "Pharma", "CIPLA.NS": "Pharma", "DRREDDY.NS": "Pharma",
    "DIVISLAB.NS": "Pharma", "APOLLOHOSP.NS": "Pharma",
    # Metals
    "TATASTEEL.NS": "Metals", "JSWSTEEL.NS": "Metals", "HINDALCO.NS": "Metals",
    # Infrastructure / Cement / Construction
    "LT.NS": "Infra", "ULTRACEMCO.NS": "Infra", "GRASIM.NS": "Infra",
    "ADANIPORTS.NS": "Infra", "ADANIENT.NS": "Infra",
    # Consumer / Telecom / Misc → grouped as Consumer
    "BHARTIARTL.NS": "Consumer", "ASIANPAINT.NS": "Consumer",
    "TITAN.NS": "Consumer", "TRENT.NS": "Consumer",
}

# Unique sector groups (used for iterating in training)
SECTOR_LIST = [
    "Energy", "IT", "Banking", "FinServ", "FMCG",
    "Auto", "Pharma", "Metals", "Infra", "Consumer",
]

# ───── SECTOR ROUTING (to be determined AFTER retraining) ─────
# The old IT/FMCG/Banking routing decisions were made on 2–4 stock samples
# and are now statistically void. After retraining the global RF and the 10
# sector sub-models, measure each sector's test AUC vs the global model and
# fill this table: True  → sector model empirically beats global, route to it
#                   False → global model wins/ties, use global.
# Defaults to all False (use global everywhere) until measured.
USE_SECTOR_MODEL = {sector: False for sector in SECTOR_LIST}