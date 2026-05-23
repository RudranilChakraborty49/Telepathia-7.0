import { useState, useEffect } from 'react'
import { X, Cpu, TrendingUp, TrendingDown, Minus } from 'lucide-react'

import { apiGetSignal, apiGetStock } from '../services/api'
import PriceChart from './PriceChart'
import IndicatorChart from './IndicatorChart'
import ExplanationPanel from './ExplanationPanel'


// Signal styling mapping
const signalColors = {
  'STRONG BUY':  { text: 'text-neon-green',  bg: 'bg-neon-green/10',  border: 'border-neon-green/40',  icon: TrendingUp,   pulse: true  },
  'BUY':         { text: 'text-signal-buy',  bg: 'bg-signal-buy/10',  border: 'border-signal-buy/40',  icon: TrendingUp,   pulse: false },
  'WEAK BUY':    { text: 'text-signal-buy/80',bg:'bg-signal-buy/5',   border: 'border-signal-buy/20',  icon: TrendingUp,   pulse: false },
  'HOLD':        { text: 'text-terminal-muted',bg:'bg-terminal-border/20',border:'border-terminal-border',icon: Minus,    pulse: false },
  'WEAK SELL':   { text: 'text-signal-sell/80',bg:'bg-signal-sell/5', border: 'border-signal-sell/20', icon: TrendingDown, pulse: false },
  'SELL':        { text: 'text-signal-sell', bg: 'bg-signal-sell/10', border: 'border-signal-sell/40', icon: TrendingDown, pulse: false },
  'STRONG SELL': { text: 'text-neon-pink',   bg: 'bg-neon-pink/10',   border: 'border-neon-pink/40',   icon: TrendingDown, pulse: true  },
}

const fmt = (v, d = 2) => {
  if (v == null || isNaN(Number(v))) return '—'
  return Number(v).toFixed(d)
}


export default function DetailPanel({ ticker, onClose }) {
  const [signal, setSignal] = useState(null)
  const [stock, setStock] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!ticker) return
    setLoading(true)
    setError(null)

    // Fetch both signal (with explanation) and stock chart data
    Promise.all([
      apiGetSignal(ticker),
      apiGetStock(ticker, '1y'),
    ])
      .then(([sigData, stockData]) => {
        setSignal(sigData)
        setStock(stockData)
        setLoading(false)
      })
      .catch((err) => {
        setError(err.message || 'Failed to load details')
        setLoading(false)
      })
  }, [ticker])

  if (!ticker) return null

  const config = signal ? (signalColors[signal.ensemble?.signal] || signalColors['HOLD']) : signalColors['HOLD']
  const Icon = config.icon

  // XGBoost probability of UP
  const probUp = signal && signal.xgb ? signal.xgb.prob_up_pct : null
  const probLeansUp = probUp != null && Number(probUp) >= 50

  // RF probability of UP
  const rfProbUp = signal && signal.rf ? signal.rf.prob_up_pct : null

  // Is the real LSTM available for this ticker?
  const lstmAvailable = signal && signal.lstm ? signal.lstm.available : false

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/70 backdrop-blur-sm z-40 animate-fade-in"
        onClick={onClose}
      />

      {/* Slide-in panel */}
      <div className="fixed top-0 right-0 bottom-0 w-full md:w-[700px] z-50 bg-terminal-bg border-l border-terminal-border
                      overflow-y-auto animate-fade-up shadow-2xl">

        {/* Close button (sticky) */}
        <div className="sticky top-0 z-10 flex items-center justify-between px-6 py-4
                        bg-terminal-bg/95 backdrop-blur border-b border-terminal-border">
          <div>
            <h2 className="font-display font-bold text-2xl tracking-wide">
              {ticker.replace('.NS', '')}
            </h2>
            {signal && (
              <span className="text-xs font-mono uppercase tracking-wider text-terminal-muted">
                {signal.sector} · NSE
              </span>
            )}
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-terminal-border/50 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="px-6 py-6 space-y-6">

          {error && (
            <div className="card border-signal-sell/40 bg-signal-sell/5 text-sm text-signal-sell">
              {error}
            </div>
          )}

          {loading && (
            <div className="flex items-center justify-center py-12">
              <div className="text-terminal-muted text-sm animate-pulse">Loading insights...</div>
            </div>
          )}

          {signal && !loading && (
            <>
              {/* Price + signal summary */}
              <section>
                <div className="flex items-baseline gap-3 mb-2">
                  <span className="font-mono text-4xl font-bold">₹{fmt(signal.current_price)}</span>
                  <span className="text-xs font-mono uppercase tracking-wider text-terminal-muted">
                    Live · NSE
                  </span>
                </div>

                <div className={`mt-4 p-4 rounded-lg border ${config.bg} ${config.border} ${config.pulse ? 'animate-pulse-slow' : ''}`}>
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                      <Icon className={`w-5 h-5 ${config.text}`} />
                      <span className={`font-display font-bold text-lg tracking-wide ${config.text}`}>
                        {signal.ensemble?.signal}
                      </span>
                    </div>
                    <span className={`font-mono text-sm font-bold ${config.text}`}>
                      {signal.ensemble?.confidence}% CONFIDENCE
                    </span>
                  </div>
                  <p className="text-xs text-terminal-muted">
                    {signal.ensemble?.reason}
                  </p>
                  <p className="text-[10px] text-terminal-muted/70 mt-1 font-mono uppercase tracking-wider">
                    Headline signal · LSTM + RF Ensemble
                  </p>
                </div>
              </section>

              {/* Price chart */}
              <section>
                <h3 className="text-sm font-mono uppercase tracking-widest text-terminal-muted mb-3">
                  Price & Trend
                </h3>
                <div className="card">
                  <PriceChart data={stock?.data || []} />
                </div>
              </section>

              {/* Indicator charts */}
              <section>
                <h3 className="text-sm font-mono uppercase tracking-widest text-terminal-muted mb-3">
                  Technical Indicators
                </h3>
                <div className="card space-y-4">
                  <IndicatorChart
                    data={stock?.data || []}
                    title="RSI (14)"
                    dataKey="RSI"
                    color="#8b5cf6"
                    refLines={[
                      { value: 70, color: '#ef4444' },
                      { value: 30, color: '#10b981' },
                    ]}
                  />
                  <IndicatorChart
                    data={stock?.data || []}
                    title="MACD"
                    dataKey="MACD"
                    color="#00f0ff"
                    refLines={[{ value: 0, color: '#6b7280' }]}
                  />
                </div>
              </section>

              {/* SHAP explanation */}
              <section>
                <h3 className="text-sm font-mono uppercase tracking-widest text-terminal-muted mb-3">
                  Why This Signal · <span className="text-neon-cyan">Explainable AI</span>
                </h3>
                <div className="card">
                  <ExplanationPanel explanation={signal.explanation} />
                </div>
              </section>

              {/* Model details */}
              <section>
                <h3 className="text-sm font-mono uppercase tracking-widest text-terminal-muted mb-3">
                  Model Detail
                </h3>

                {/* ── Model 1: LSTM + RF Ensemble ── */}
                <div className="card mb-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Cpu className="w-4 h-4 text-neon-purple" />
                    <span className="font-mono text-xs uppercase tracking-widest text-neon-purple">
                      LSTM + RF Ensemble · Headline Signal
                    </span>
                  </div>
                  <div className="space-y-2 text-xs">
                    <Row label="Ensemble Signal" value={signal.ensemble?.signal} />
                    <Row
                      label="Ensemble Confidence"
                      value={signal.ensemble?.confidence != null
                        ? `${signal.ensemble.confidence}%` : '—'}
                    />
                    <Row
                      label="LSTM Direction"
                      value={lstmAvailable ? signal.lstm?.direction : 'Unavailable'}
                    />
                    <Row
                      label="LSTM Predicted Price"
                      value={signal.lstm?.predicted_price != null
                        ? `₹${fmt(signal.lstm.predicted_price)}` : '—'}
                    />
                    <Row
                      label="LSTM Expected Return"
                      value={signal.lstm?.expected_return_pct != null
                        ? `${fmt(signal.lstm.expected_return_pct, 2)}%` : '—'}
                    />
                    <Row label="RF Direction" value={signal.rf?.direction} />
                    <Row label="RF Strength" value={signal.rf?.strength} />
                    <Row
                      label="RF Probability of UP"
                      value={rfProbUp != null ? `${fmt(rfProbUp, 1)}%` : '—'}
                    />
                    <Row label="Decision" value={signal.ensemble?.reason} />
                  </div>
                  <p className="mt-3 text-[10px] leading-relaxed text-terminal-muted">
                    The LSTM (price regressor) and Random Forest (directional
                    classifier) vote together. When they agree, confidence is high;
                    when they conflict, the ensemble returns HOLD. This is the
                    headline signal shown on the dashboard.
                  </p>
                </div>

                {/* ── Model 2: XGBoost (independent) ── */}
                <div className="card">
                  <div className="flex items-center gap-2 mb-3">
                    <Cpu className="w-4 h-4 text-neon-cyan" />
                    <span className="font-mono text-xs uppercase tracking-widest text-neon-cyan">
                      XGBoost · Independent Second Opinion
                    </span>
                  </div>
                  <div className="space-y-2 text-xs">
                    <Row label="Signal" value={signal.xgb?.signal} />
                    <Row label="Direction" value={signal.xgb?.direction} />
                    <Row label="Strength" value={signal.xgb?.strength} />
                    <Row
                      label="Probability of UP"
                      value={probUp != null ? `${fmt(probUp, 1)}%` : '—'}
                    />
                    <Row
                      label="Lean"
                      value={
                        probUp == null
                          ? '—'
                          : probLeansUp ? '▲ Upward' : '▼ Downward'
                      }
                    />
                    <Row
                      label="Confidence"
                      value={signal.xgb?.confidence != null
                        ? `${signal.xgb.confidence}%` : '—'}
                    />
                    <Row label="Model" value="XGBoost-Global (50 stocks)" />
                  </div>
                  <p className="mt-3 text-[10px] leading-relaxed text-terminal-muted">
                    A gradient-boosted classifier trained on 26 technical, calendar
                    and market-regime features. It runs independently of the
                    ensemble as a cross-check. Probabilities near 50% indicate a
                    genuinely weak edge — confidence is reported honestly.
                  </p>
                </div>
              </section>
            </>
          )}
        </div>
      </div>
    </>
  )
}


function Row({ label, value }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <span className="text-terminal-muted shrink-0">{label}</span>
      <span className="font-mono font-medium text-right">{value || '—'}</span>
    </div>
  )
}
