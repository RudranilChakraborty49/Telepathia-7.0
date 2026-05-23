# Telepathia 7.0

An AI-powered quantitative engine that generates explainable BUY / SELL / HOLD
trading signals for NSE (National Stock Exchange of India) large-cap stocks.

Telepathia runs three machine-learning models over technical, calendar, and
market-regime features, and presents their predictions through a real-time
React dashboard with SHAP-based explanations and an honest historical backtest.

> **Disclaimer:** Built for research and education only. This is **not financial
> advice**. Past performance does not predict future results.

---

## What it does

For each of 50 NSE stocks, Telepathia produces:

- **An LSTM + Random Forest ensemble signal** — the headline prediction. An LSTM
  price regressor and a Random Forest directional classifier vote together;
  when they conflict, the ensemble returns HOLD.
- **An independent XGBoost signal** — a separate gradient-boosted classifier
  used as a cross-check / second opinion.
- **A SHAP explanation** — the top features pushing the prediction up or down.
- **A historical backtest** — strategy performance for the RF and XGBoost
  models against Buy & Hold and an RSI baseline.

## Tech stack

**Backend:** Python, Flask, TensorFlow/Keras (LSTM), scikit-learn (Random
Forest), XGBoost, SHAP, yfinance, pandas, NumPy.

**Frontend:** React, Vite, Tailwind CSS, Recharts, Axios.

---

## Project structure

```
telepathia-7.0/
├── backend/
│   ├── app.py                 # Flask REST API
│   ├── build_dataset.py       # builds the master feature dataset
│   ├── train_lstm.py          # trains the LSTM model
│   ├── train_rf.py            # trains the Random Forest
│   ├── train_xgb.py           # trains XGBoost
│   ├── config.py              # tickers, sector groups, settings
│   ├── models/                # ensemble & explainer logic
│   ├── utils/                 # data fetch, indicators, features, backtester
│   └── requirements.txt
├── frontend/                  # React + Vite dashboard
└── research/                  # dataset, notebooks, paper
```

---

## Setup & run

The project has two parts — a Python backend and a React frontend — run them in
two separate terminals.

### 1. Backend

From the `backend/` folder:

```bash
# create and activate a virtual environment (Python 3.11)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# install dependencies
pip install -r requirements.txt
```

**Generate the data and train the models** (run once, in this order):

```bash
python build_dataset.py        # fetch data + build feature dataset
python utils/lstm_data_prep.py # build LSTM arrays + per-stock scalers
python train_lstm.py           # train the LSTM
python train_rf.py             # train the Random Forest
python train_xgb.py            # train XGBoost
python utils/backtester.py     # generate backtest results
```

**Start the API server:**

```bash
python app.py
```

The backend runs at `http://localhost:5000`.

### 2. Frontend

From the `frontend/` folder, in a second terminal:

```bash
npm install
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`).

---

## API endpoints

| Endpoint | Description |
|---|---|
| `GET /api/health` | Server status |
| `GET /api/tickers` | Supported NSE stocks |
| `GET /api/stock/<ticker>` | OHLCV + indicators for charts |
| `GET /api/signal/<ticker>` | Single-stock prediction + explanation |
| `GET /api/signals/all` | Predictions for all stocks |
| `GET /api/sectors` | Sector breakdown |
| `GET /api/backtest` | Pre-computed backtest results |

---

## A note on the results

The backtest reports that the ML models do **not** beat a simple Buy & Hold
baseline over the test period, and their directional AUC sits near 0.53 — a
weak edge. This is reported honestly rather than hidden. Predicting short-term
price direction is genuinely hard; an accurate, modest result is more valuable
than an inflated one.

---

## Author

Built as a Computer Science Engineering project at IEM Kolkata.
