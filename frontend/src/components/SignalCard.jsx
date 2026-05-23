import { TrendingUp, TrendingDown, Minus, ChevronRight, Cpu } from 'lucide-react'

// ───── HELPERS ─────
const signalConfig = {
  'STRONG BUY':  { color: 'neon-green',  text: 'text-neon-green',   bg: 'bg-neon-green/10',     border: 'border-neon-green/40',  glow: 'shadow-neon-green',  icon: TrendingUp,   pulse: true  },
  'BUY':         { color: 'signal-buy',  text: 'text-signal-buy',   bg: 'bg-signal-buy/10',     border: 'border-signal-buy/40',  glow: 'shadow-neon-green',  icon: TrendingUp,   pulse: false },
  'WEAK BUY':    { color: 'signal-buy',  text: 'text-signal-buy/80',bg: 'bg-signal-buy/5',      border: 'border-signal-buy/20',  glow: '',                   icon: TrendingUp,   pulse: false },
  'HOLD':        { color: 'signal-hold', text: 'text-terminal-muted',bg: 'bg-terminal-border/20',border: 'border-terminal-border',glow: '',                  icon: Minus,        pulse: false },
  'WEAK SELL':   { color: 'signal-sell', text: 'text-signal-sell/80',bg: 'bg-signal-sell/5',    border: 'border-signal-sell/20', glow: '',                   icon: TrendingDown, pulse: false },
  'SELL':        { color: 'signal-sell', text: 'text-signal-sell',  bg: 'bg-signal-sell/10',    border: 'border-signal-sell/40', glow: 'shadow-neon-pink',   icon: TrendingDown, pulse: false },
  'STRONG SELL': { color: 'neon-pink',   text: 'text-neon-pink',    bg: 'bg-neon-pink/10',      border: 'border-neon-pink/40',   glow: 'shadow-neon-pink',   icon: TrendingDown, pulse: true  },
}

const sectorColors = {
  'IT':      'text-neon-cyan border-neon-cyan/40',
  'Banking': 'text-neon-purple border-neon-purple/40',
  'FMCG':    'text-neon-green border-neon-green/40',
  'Other':   'text-terminal-muted border-terminal-border',
}

// Safely formats null / undefined / NaN → '—'
const fmt = (v, decimals = 2) => {
  if (v === null || v === undefined) return '—'
  const n = Number(v)
  if (isNaN(n)) return '—'
  return n.toFixed(decimals)
}

// Safely checks if a value is a real number
const isNum = (v) => v !== null && v !== undefined && !isNaN(Number(v))


// ───── COMPONENT ─────
export default function SignalCard({ signal, onClick }) {
  // Headline badge = the LSTM + RF ensemble signal
  const config = signalConfig[signal.signal] || signalConfig['HOLD']
  const Icon = config.icon
  const sectorClass = sectorColors[signal.sector] || sectorColors['Other']

  const confidence = isNum(signal.confidence) ? signal.confidence : 0

  // XGBoost probability of UP (0–100)
  const xgbProbUp = isNum(signal.xgb_prob_up) ? Number(signal.xgb_prob_up) : null
  const xgbLeansUp = xgbProbUp !== null && xgbProbUp >= 50

  // RF probability of UP (0–100)
  const rfProbUp = isNum(signal.rf_prob_up) ? Number(signal.rf_prob_up) : null

  return (
    <div
      onClick={() => onClick && onClick(signal.ticker)}
      className="relative card cursor-pointer transition-all duration-300 hover:-translate-y-1 hover:border-neon-cyan/40 hover:shadow-neon-cyan"
    >
      {/* Top accent bar */}
      <div className={`absolute top-0 left-0 right-0 h-0.5 rounded-t-xl bg-${config.color}`} />

      {/* HEADER — ticker + sector */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="font-display font-bold text-xl tracking-wide">
            {(signal.ticker || 'N/A').replace('.NS', '')}
          </h3>
          <span className={`inline-block mt-1 px-2 py-0.5 text-[10px] font-mono uppercase tracking-wider border rounded ${sectorClass}`}>
            {signal.sector || 'Other'}
          </span>
        </div>
        <ChevronRight className="w-5 h-5 text-terminal-muted opacity-50" />
      </div>

      {/* PRICE BLOCK */}
      <div className="mb-4">
        <div className="flex items-baseline gap-2 mb-1">
          <span className="font-mono text-2xl font-bold">₹{fmt(signal.current_price)}</span>
        </div>
        <div className="text-xs text-terminal-muted">
          Signal:{' '}
          <span className="font-mono text-neon-cyan">LSTM + RF Ensemble</span>
        </div>
      </div>

      {/* SIGNAL BADGE — LSTM + RF ensemble (headline) */}
      <div className={`
        relative flex items-center justify-between px-4 py-3 rounded-lg border mb-4
        ${config.bg} ${config.border}
        ${config.pulse ? 'animate-pulse-slow' : ''}
      `}>
        <div className="flex items-center gap-2">
          <Icon className={`w-5 h-5 ${config.text}`} />
          <span className={`font-display font-bold text-sm tracking-wider ${config.text}`}>
            {signal.signal || 'HOLD'}
          </span>
        </div>
        <span className={`font-mono text-xs font-bold ${config.text}`}>
          {confidence}%
        </span>
      </div>

      {/* CONFIDENCE BAR — ensemble */}
      <div className="mb-4">
        <div className="flex items-center justify-between text-[10px] text-terminal-muted uppercase tracking-wider mb-1">
          <span>Ensemble Confidence</span>
          <span className="font-mono">{confidence}/100</span>
        </div>
        <div className="h-1.5 bg-terminal-border/40 rounded-full overflow-hidden">
          <div
            className={`h-full bg-${config.color} rounded-full transition-all duration-500`}
            style={{ width: `${confidence}%` }}
          />
        </div>
      </div>

      {/* TWO-MODEL COMPARISON */}
      <div className="space-y-2">
        {/* LSTM + RF ensemble breakdown */}
        <div className="flex items-center gap-2 px-3 py-2 rounded bg-terminal-border/20 text-[11px]">
          <Cpu className="w-3.5 h-3.5 text-neon-purple shrink-0" />
          <div className="flex-1 min-w-0">
            <div className="text-terminal-muted text-[9px] uppercase tracking-wider">
              LSTM + RF Ensemble
            </div>
            <div className="font-mono">
              {signal.signal || 'HOLD'}
              <span className="ml-1.5 text-terminal-muted">
                {isNum(signal.confidence) ? `· ${signal.confidence}% conf` : ''}
              </span>
              {signal.lstm_available === false && (
                <span className="ml-1.5 text-signal-hold">(RF-only)</span>
              )}
            </div>
            {rfProbUp !== null && (
              <div className="text-terminal-muted text-[9px] mt-0.5">
                RF P(up) {fmt(rfProbUp, 1)}% · LSTM {signal.lstm_direction || '—'}
              </div>
            )}
          </div>
        </div>

        {/* XGBoost — independent second opinion */}
        <div className="flex items-center gap-2 px-3 py-2 rounded bg-terminal-border/20 text-[11px]">
          <Cpu className="w-3.5 h-3.5 text-neon-cyan shrink-0" />
          <div className="flex-1 min-w-0">
            <div className="text-terminal-muted text-[9px] uppercase tracking-wider">
              XGBoost — Independent
            </div>
            <div className="font-mono">
              {signal.xgb_signal || 'HOLD'}
              {xgbProbUp !== null && (
                <span className={`ml-1.5 ${xgbLeansUp ? 'text-signal-buy' : 'text-signal-sell'}`}>
                  {xgbLeansUp ? '▲' : '▼'} P(up) {fmt(xgbProbUp, 1)}%
                </span>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
