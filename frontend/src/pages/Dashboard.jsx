import { useState, useEffect, useMemo } from 'react'
import { RefreshCw, AlertCircle, BarChart3 } from 'lucide-react'

import { apiHealth, apiGetAllSignals } from '../services/api'
import Header from '../components/Header'
import SignalCard from '../components/SignalCard'
import SignalCardSkeleton from '../components/SignalCardSkeleton'
import SectorFilter from '../components/SectorFilter'
import DetailPanel from '../components/DetailPanel'


export default function Dashboard({ onBack, onShowBacktest }) {
  const [apiOnline, setApiOnline] = useState(false)
  const [signals, setSignals] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [sector, setSector] = useState('ALL')
  const [refreshKey, setRefreshKey] = useState(0)
  const [selectedTicker, setSelectedTicker] = useState(null)

  useEffect(() => {
    apiHealth().then(() => setApiOnline(true)).catch(() => setApiOnline(false))
  }, [refreshKey])

  useEffect(() => {
    setLoading(true)
    setError(null)
    apiGetAllSignals()
      .then((data) => {
        setSignals(Array.isArray(data.signals) ? data.signals : [])
        setLoading(false)
      })
      .catch((err) => {
        setError(err.message || 'Failed to load signals')
        setLoading(false)
      })
  }, [refreshKey])

  const filtered = useMemo(() => {
    if (sector === 'ALL') return signals
    return signals.filter((s) => s.sector === sector)
  }, [signals, sector])

  const stats = useMemo(() => {
    const counts = { buy: 0, sell: 0, hold: 0 }
    signals.forEach((s) => {
      const sig = s.signal || ''
      if (sig.includes('BUY')) counts.buy++
      else if (sig.includes('SELL')) counts.sell++
      else counts.hold++
    })
    return counts
  }, [signals])

  return (
    <div className="min-h-screen animate-fade-in">
      <Header apiOnline={apiOnline} onBack={onBack} />

      <main className="max-w-7xl mx-auto px-6 py-8">

        <section className="mb-8 animate-fade-up">
          <div className="flex items-end justify-between flex-wrap gap-4">
            <div>
              <h2 className="font-display font-bold text-4xl mb-2">
                Live <span className="neon-text-gradient">Signals</span>
              </h2>
              <p className="text-terminal-muted text-sm">
                LSTM + Random Forest ensemble · independent XGBoost · SHAP explanations
              </p>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={onShowBacktest}
                className="flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold uppercase tracking-wider border border-neon-purple/40 bg-neon-purple/5 text-neon-purple hover:bg-neon-purple/10 hover:shadow-neon-purple transition-all duration-200"
              >
                <BarChart3 className="w-3.5 h-3.5" />
                <span>Backtest</span>
              </button>
              <button
                onClick={() => setRefreshKey((k) => k + 1)}
                disabled={loading}
                className="flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold uppercase tracking-wider border border-neon-cyan/40 bg-neon-cyan/5 text-neon-cyan hover:bg-neon-cyan/10 hover:shadow-neon-cyan transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
                <span>Refresh</span>
              </button>
            </div>
          </div>
        </section>

        <section className="mb-6 grid grid-cols-2 md:grid-cols-4 gap-4 animate-fade-up"
                 style={{ animationDelay: '0.1s', animationFillMode: 'backwards' }}>
          <StatCard label="Total Signals" value={signals.length} color="neon-cyan" />
          <StatCard label="Buy"  value={stats.buy}  color="signal-buy" />
          <StatCard label="Sell" value={stats.sell} color="signal-sell" />
          <StatCard label="Hold" value={stats.hold} color="terminal-muted" />
        </section>

        <section className="mb-6 animate-fade-up"
                 style={{ animationDelay: '0.2s', animationFillMode: 'backwards' }}>
          <SectorFilter value={sector} onChange={setSector} signalCount={filtered.length} />
        </section>

        {error && (
          <div className="card border-signal-sell/40 bg-signal-sell/5 flex items-center gap-3 mb-6">
            <AlertCircle className="w-5 h-5 text-signal-sell flex-shrink-0" />
            <div>
              <p className="font-semibold text-signal-sell text-sm">Failed to load signals</p>
              <p className="text-xs text-terminal-muted mt-0.5">{error}</p>
            </div>
          </div>
        )}

        <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 animate-fade-up"
                 style={{ animationDelay: '0.3s', animationFillMode: 'backwards' }}>
          {loading
            ? Array.from({ length: 8 }).map((_, i) => <SignalCardSkeleton key={i} />)
            : filtered.map((signal, i) => (
                <SignalCard
                  key={signal.ticker || i}
                  signal={signal}
                  onClick={(ticker) => setSelectedTicker(ticker)}
                />
              ))}
        </section>

        {!loading && filtered.length === 0 && !error && (
          <div className="text-center py-12 text-terminal-muted">
            <p>No signals matching the selected sector.</p>
          </div>
        )}

      </main>

      {/* Detail panel slides in from the right */}
      {selectedTicker && (
        <DetailPanel
          ticker={selectedTicker}
          onClose={() => setSelectedTicker(null)}
        />
      )}
    </div>
  )
}


function StatCard({ label, value, color }) {
  return (
    <div className="card flex flex-col">
      <span className="text-[10px] uppercase tracking-widest text-terminal-muted mb-1">{label}</span>
      <span className={`font-display font-bold text-3xl text-${color}`}>{value}</span>
    </div>
  )
}
