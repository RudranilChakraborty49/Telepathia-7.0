import { Brain, Sparkles, TrendingUp, Zap, ChevronRight, Activity } from 'lucide-react'

export default function Landing({ onEnter }) {
  const features = [
    { icon: Brain,      label: "LSTM + RF Ensemble", desc: "Combined vote, HOLD on conflict"    },
    { icon: TrendingUp, label: "XGBoost",            desc: "Independent gradient-boosted model" },
    { icon: Sparkles,   label: "Explainable AI",     desc: "Every signal, fully reasoned"       },
    { icon: Zap,        label: "Conflict-Aware",     desc: "HOLD when models disagree"          },
  ]

  return (
    <div className="relative min-h-screen overflow-hidden">

      {/* Background grid */}
      <div className="absolute inset-0 grid-bg opacity-40 pointer-events-none" />

      {/* Scanning line effect */}
      <div className="scan-line" />

      {/* Floating glow orbs */}
      <div className="absolute top-20 left-10 w-72 h-72 bg-neon-purple/20 rounded-full blur-3xl animate-float" />
      <div className="absolute bottom-20 right-10 w-96 h-96 bg-neon-cyan/20 rounded-full blur-3xl animate-float"
           style={{ animationDelay: "2s" }} />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px]
                      bg-radial-glow rounded-full pointer-events-none" />

      <div className="relative z-10 max-w-6xl mx-auto px-6 py-12 min-h-screen flex flex-col">

        {/* Top brand bar */}
        <div className="flex items-center justify-between animate-fade-in">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-neon-cyan to-neon-purple
                            flex items-center justify-center shadow-neon-purple">
              <Brain className="w-7 h-7 text-white" />
            </div>
            <div>
              <h2 className="font-display font-bold text-xl tracking-wider">TELEPATHIA <span className="neon-text-cyan">7.0</span></h2>
              <p className="text-xs text-terminal-muted">v1.0 · NSE INTELLIGENCE</p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs text-terminal-muted">
            <Activity className="w-4 h-4 text-neon-green" />
            <span>SYSTEM ONLINE</span>
          </div>
        </div>

        {/* Hero section */}
        <div className="flex-1 flex flex-col justify-center items-center text-center animate-fade-up">

          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full
                          border border-neon-cyan/30 bg-neon-cyan/5 mb-8">
            <Sparkles className="w-4 h-4 text-neon-cyan" />
            <span className="text-xs font-mono text-neon-cyan uppercase tracking-widest">
              AI-Powered Quantitative Engine
            </span>
          </div>

          <h1 className="font-display font-black text-6xl md:text-8xl tracking-tight mb-6 leading-none">
            <span className="block">Reading the</span>
            <span className="block neon-text-gradient">Market's Mind</span>
          </h1>

          <p className="max-w-2xl text-lg md:text-xl text-terminal-muted leading-relaxed mb-12">
            A multi-model engine — an <span className="text-neon-cyan">LSTM + Random Forest</span> ensemble
            alongside an independent <span className="text-neon-cyan">XGBoost</span> classifier — combining
            technical indicators, market regime signals, and F&O microstructure to generate
            <span className="text-neon-purple"> explainable trading signals </span>
            on NSE large-caps.
          </p>

          {/* CTA button */}
          <button onClick={onEnter} className="neon-btn group inline-flex items-center gap-3 text-lg">
            <span>ENTER TERMINAL</span>
            <ChevronRight className="w-5 h-5 transition-transform group-hover:translate-x-1" />
          </button>

          {/* Stats row */}
          <div className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-6 max-w-3xl w-full">
            {[
              { n: "26",  l: "Features"     },
              { n: "50",  l: "NSE Stocks"   },
              { n: "3",   l: "ML Models"    },
              { n: "5Y",  l: "Backtested"   },
            ].map((s, i) => (
              <div key={i} className="text-center py-4 border-l border-r border-terminal-border/30
                                       first:border-l-0 last:border-r-0">
                <div className="font-display font-bold text-2xl md:text-3xl neon-text-cyan">{s.n}</div>
                <div className="text-xs text-terminal-muted uppercase tracking-wider mt-1">{s.l}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Feature highlights */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8 animate-fade-up"
             style={{ animationDelay: "0.3s", animationFillMode: "backwards" }}>
          {features.map((f, i) => {
            const Icon = f.icon
            return (
              <div key={i} className="card hover:border-neon-cyan/30 hover:shadow-neon-cyan transition-all duration-300 group">
                <Icon className="w-6 h-6 text-neon-cyan mb-3 group-hover:scale-110 transition-transform" />
                <h3 className="font-semibold text-sm mb-1">{f.label}</h3>
                <p className="text-xs text-terminal-muted leading-relaxed">{f.desc}</p>
              </div>
            )
          })}
        </div>

        {/* Footer caveat */}
        <div className="text-center text-xs text-terminal-muted/70 pb-6">
          <p>Built for research & education · Not financial advice · Past performance ≠ future results</p>
        </div>
      </div>
    </div>
  )
}
