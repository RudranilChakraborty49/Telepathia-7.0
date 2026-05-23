import { useState, useEffect, useMemo } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from 'recharts'
import { ArrowLeft, TrendingUp, AlertCircle } from 'lucide-react'

import { apiGetBacktest } from '../services/api'


// ───── STRATEGY DISPLAY CONFIG ─────
// Maps the JSON keys to display names + colours.
const STRATEGIES = [
  { key: 'xgboost_model', label: 'XGBoost Model',  color: '#a855f7', isModel: true  },
  { key: 'random_forest', label: 'Random Forest',  color: '#10b981', isModel: true  },
  { key: 'buy_and_hold',  label: 'Buy & Hold',     color: '#00f0ff', isModel: false },
  { key: 'rsi_only',      label: 'RSI-only',       color: '#f59e0b', isModel: false },
]

const pct = (v, d = 2) => {
  if (v == null || isNaN(Number(v))) return '—'
  return `${(Number(v) * 100).toFixed(d)}%`
}
const num = (v, d = 3) => {
  if (v == null || isNaN(Number(v))) return '—'
  return Number(v).toFixed(d)
}


export default function Backtest({ onBack }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    apiGetBacktest()
      .then((d) => {
        setData(d)
        setLoading(false)
      })
      .catch((err) => {
        setError(
          err.response?.data?.error ||
          err.message ||
          'Failed to load backtest results'
        )
        setLoading(false)
      })
  }, [])

  // Build the equity-curve chart data: one row per trading day,
  // each strategy as a column.
  const chartData = useMemo(() => {
    if (!data) return []
    const curves = STRATEGIES.map((s) => data[s.key]?.equity_curve || [])
    const maxLen = Math.max(...curves.map((c) => c.length), 0)
    const rows = []
    for (let i = 0; i < maxLen; i++) {
      const row = { day: i }
      STRATEGIES.forEach((s, idx) => {
        row[s.key] = curves[idx][i] != null ? curves[idx][i] : null
      })
      rows.push(row)
    }
    return rows
  }, [data])

  // Honest verdict: did either model beat Buy & Hold?
  const verdict = useMemo(() => {
    if (!data) return null
    const xgb = data.xgboost_model?.total_return ?? 0
    const rf = data.random_forest?.total_return ?? 0
    const bh = data.buy_and_hold?.total_return ?? 0
    const bestModel = Math.max(xgb, rf)
    if (bestModel > bh) {
      return {
        tone: 'good',
        text: 'The best model outperformed Buy & Hold over the test period.',
      }
    }
    return {
      tone: 'honest',
      text: 'Neither model beat a simple Buy & Hold baseline over the test ' +
            'period. This is an honest, reportable result — it shows the ' +
            'directional edge is weak, consistent with the models\u2019 ' +
            'low AUC (~0.53).',
    }
  }, [data])

  return (
    <div className="min-h-screen animate-fade-in">

      {/* Header */}
      <header className="border-b border-terminal-border bg-terminal-bg/95 backdrop-blur">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center gap-4">
          <button
            onClick={onBack}
            className="p-2 rounded-lg hover:bg-terminal-border/50 transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div>
            <h1 className="font-display font-bold text-xl tracking-wide">
              Strategy <span className="neon-text-cyan">Backtest</span>
            </h1>
            <p className="text-xs text-terminal-muted">
              Historical performance · after transaction costs
            </p>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">

        {error && (
          <div className="card border-signal-sell/40 bg-signal-sell/5 flex items-center gap-3 mb-6">
            <AlertCircle className="w-5 h-5 text-signal-sell flex-shrink-0" />
            <div>
              <p className="font-semibold text-signal-sell text-sm">Could not load backtest</p>
              <p className="text-xs text-terminal-muted mt-0.5">{error}</p>
            </div>
          </div>
        )}

        {loading && (
          <div className="flex items-center justify-center py-20">
            <div className="text-terminal-muted text-sm animate-pulse">
              Loading backtest results...
            </div>
          </div>
        )}

        {data && !loading && (
          <>
            {/* Test period summary */}
            <section className="mb-6">
              <p className="text-sm text-terminal-muted">
                Test period:{' '}
                <span className="font-mono text-neon-cyan">
                  {data.test_period?.start} → {data.test_period?.end}
                </span>
                {' · '}
                <span className="font-mono">{data.test_period?.n_stocks} stocks</span>
                {' · '}
                <span className="font-mono">
                  {data.config?.hold_days}-day holds
                </span>
                {' · '}
                <span className="font-mono">
                  {pct(data.config?.cost_per_trade, 2)} cost/trade
                </span>
              </p>
              <p className="text-xs text-terminal-muted/70 mt-1">
                Models were trained only on data before the test period —
                they never saw any of it.
              </p>
            </section>

            {/* Metrics cards */}
            <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
              {STRATEGIES.map((s) => {
                const m = data[s.key] || {}
                return (
                  <div key={s.key} className="card">
                    <div className="flex items-center gap-2 mb-3">
                      <span
                        className="w-2.5 h-2.5 rounded-full"
                        style={{ backgroundColor: s.color }}
                      />
                      <span className="font-display font-bold text-sm">
                        {s.label}
                      </span>
                      {s.isModel ? (
                        <span className="text-[9px] font-mono uppercase tracking-wider
                                         px-1.5 py-0.5 rounded border border-neon-purple/40
                                         text-neon-purple">
                          Model
                        </span>
                      ) : (
                        <span className="text-[9px] font-mono uppercase tracking-wider
                                         px-1.5 py-0.5 rounded border border-terminal-border
                                         text-terminal-muted">
                          Baseline
                        </span>
                      )}
                    </div>
                    <div className="space-y-1.5 text-xs">
                      <MetricRow
                        label="Total Return"
                        value={pct(m.total_return)}
                        positive={m.total_return > 0}
                      />
                      <MetricRow label="Sharpe Ratio" value={num(m.sharpe)} />
                      <MetricRow
                        label="Max Drawdown"
                        value={pct(m.max_drawdown)}
                        negative
                      />
                      <MetricRow label="Win Rate (days)" value={pct(m.win_rate, 1)} />
                    </div>
                  </div>
                )
              })}
            </section>

            {/* Equity curve chart */}
            <section className="mb-8">
              <div className="flex items-center gap-2 mb-3">
                <TrendingUp className="w-4 h-4 text-neon-cyan" />
                <h2 className="text-sm font-mono uppercase tracking-widest text-terminal-muted">
                  Equity Curve
                </h2>
              </div>
              <div className="card">
                <p className="text-xs text-terminal-muted mb-4">
                  Growth of ₹1 invested at the start of the test period,
                  following each strategy day by day.
                </p>
                <ResponsiveContainer width="100%" height={380}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                    <XAxis
                      dataKey="day"
                      stroke="#6b7280"
                      tick={{ fontSize: 11 }}
                      label={{
                        value: 'Trading day', position: 'insideBottom',
                        offset: -4, fill: '#6b7280', fontSize: 11,
                      }}
                    />
                    <YAxis
                      stroke="#6b7280"
                      tick={{ fontSize: 11 }}
                      domain={['auto', 'auto']}
                      tickFormatter={(v) => v.toFixed(2)}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#0f172a',
                        border: '1px solid #1f2937',
                        borderRadius: 8,
                        fontSize: 12,
                      }}
                      formatter={(v) => (v != null ? Number(v).toFixed(4) : '—')}
                      labelFormatter={(d) => `Day ${d}`}
                    />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    {STRATEGIES.map((s) => (
                      <Line
                        key={s.key}
                        type="monotone"
                        dataKey={s.key}
                        name={s.label}
                        stroke={s.color}
                        strokeWidth={2}
                        dot={false}
                        connectNulls
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>

            {/* Honest verdict */}
            {verdict && (
              <section className="mb-8">
                <div className={`card ${
                  verdict.tone === 'good'
                    ? 'border-neon-green/40 bg-neon-green/5'
                    : 'border-neon-cyan/30 bg-neon-cyan/5'
                }`}>
                  <h3 className="font-display font-bold text-sm mb-2">
                    Verdict
                  </h3>
                  <p className="text-xs text-terminal-muted leading-relaxed">
                    {verdict.text}
                  </p>
                </div>
              </section>
            )}

            {/* Methodology note */}
            <section>
              <div className="card">
                <h3 className="font-display font-bold text-sm mb-2">
                  Methodology
                </h3>
                <p className="text-xs text-terminal-muted leading-relaxed">
                  A day-by-day equal-weight portfolio engine. On a predicted UP,
                  a position is opened the next trading day and held for{' '}
                  {data.config?.hold_days} days; capital is spread equally across
                  all open positions. A round-trip transaction cost of{' '}
                  {pct(data.config?.cost_per_trade, 2)} is charged on every entry.
                  Returns compound across sequential trading days. The two ML
                  strategies are compared against Buy & Hold and a one-line RSI
                  rule so the models are judged against honest baselines.
                </p>
              </div>
            </section>
          </>
        )}
      </main>
    </div>
  )
}


function MetricRow({ label, value, positive, negative }) {
  let valueClass = 'font-mono font-medium'
  if (positive) valueClass += ' text-signal-buy'
  if (negative) valueClass += ' text-signal-sell'
  return (
    <div className="flex items-center justify-between">
      <span className="text-terminal-muted">{label}</span>
      <span className={valueClass}>{value}</span>
    </div>
  )
}
