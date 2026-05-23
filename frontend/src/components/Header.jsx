import { Activity, Brain, ChevronLeft } from 'lucide-react'

export default function Header({ apiOnline, onBack }) {
  return (
    <header className="border-b border-terminal-border bg-terminal-surface/50 backdrop-blur-lg sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">

        <div className="flex items-center gap-3">
          {onBack && (
            <button onClick={onBack}
                    className="p-2 rounded-lg hover:bg-terminal-border/50 transition-colors">
              <ChevronLeft className="w-5 h-5" />
            </button>
          )}
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-neon-cyan to-neon-purple
                          flex items-center justify-center shadow-neon-cyan">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="font-display font-bold text-xl tracking-wider">
              Telepathia <span className="neon-text-cyan">7.0</span>
            </h1>
            <p className="text-xs text-terminal-muted -mt-0.5 uppercase tracking-wider">
              AI Trading Signals · NSE
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border ${
            apiOnline ? 'border-neon-green/40 bg-neon-green/5' : 'border-signal-sell/40 bg-signal-sell/5'
          }`}>
            <span className={`w-2 h-2 rounded-full ${
              apiOnline ? 'bg-neon-green animate-pulse' : 'bg-signal-sell'
            }`} />
            <span className="text-xs font-semibold uppercase tracking-wider">
              {apiOnline ? 'Live' : 'Offline'}
            </span>
          </div>
          <div className="hidden md:flex items-center gap-1.5 text-xs text-terminal-muted">
            <Activity className="w-3.5 h-3.5 text-neon-cyan" />
            <span>Real-time AI</span>
          </div>
        </div>
      </div>
    </header>
  )
}