import { TrendingUp, TrendingDown, Info } from 'lucide-react'

export default function ExplanationPanel({ explanation }) {
  if (!explanation) {
    return (
      <div className="text-terminal-muted text-sm py-4 text-center">
        Loading explanation...
      </div>
    )
  }

  const { summary, top_bullish = [], top_bearish = [] } = explanation

  return (
    <div className="space-y-4">

      {/* Summary banner */}
      <div className="flex items-start gap-3 p-4 rounded-lg bg-accent-blue/5 border border-accent-blue/20">
        <Info className="w-5 h-5 text-neon-cyan flex-shrink-0 mt-0.5" />
        <p className="text-sm leading-relaxed">{summary}</p>
      </div>

      {/* Bullish drivers */}
      {top_bullish.length > 0 && (
        <div>
          <h4 className="flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-signal-buy mb-3">
            <TrendingUp className="w-3.5 h-3.5" />
            Bullish Drivers
          </h4>
          <div className="space-y-2">
            {top_bullish.map((item, i) => (
              <DriverRow key={`b-${i}`} item={item} type="bullish" />
            ))}
          </div>
        </div>
      )}

      {/* Bearish drivers */}
      {top_bearish.length > 0 && (
        <div>
          <h4 className="flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-signal-sell mb-3">
            <TrendingDown className="w-3.5 h-3.5" />
            Bearish Drivers
          </h4>
          <div className="space-y-2">
            {top_bearish.map((item, i) => (
              <DriverRow key={`s-${i}`} item={item} type="bearish" />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}


function DriverRow({ item, type }) {
  const isBullish = type === 'bullish'
  const color = isBullish ? 'signal-buy' : 'signal-sell'
  const symbol = isBullish ? '✓' : '✗'
  const contribution = item.contribution || 0
  const absContrib = Math.abs(contribution)

  // Normalize to 0-100 for the mini bar (most SHAP values are under 0.1)
  const barWidth = Math.min(absContrib * 1000, 100)

  return (
    <div className={`flex items-center gap-3 p-3 rounded-lg bg-${color}/5 border border-${color}/15`}>
      <span className={`text-${color} text-lg font-bold flex-shrink-0`}>{symbol}</span>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium truncate">{item.message}</p>
        <div className="flex items-center gap-2 mt-1.5">
          <div className="flex-1 h-1 bg-terminal-border/40 rounded-full overflow-hidden">
            <div
              className={`h-full bg-${color} rounded-full`}
              style={{ width: `${barWidth}%` }}
            />
          </div>
          <span className="text-[10px] font-mono text-terminal-muted whitespace-nowrap">
            {contribution >= 0 ? '+' : ''}{contribution.toFixed(4)}
          </span>
        </div>
      </div>
    </div>
  )
}