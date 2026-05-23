const sectors = [
  { id: 'ALL',     label: 'All',     color: 'neon-cyan'    },
  { id: 'IT',      label: 'IT',      color: 'neon-cyan'    },
  { id: 'Banking', label: 'Banking', color: 'neon-purple'  },
  { id: 'FMCG',    label: 'FMCG',    color: 'neon-green'   },
  { id: 'Other',   label: 'Other',   color: 'terminal-muted' },
]

export default function SectorFilter({ value, onChange, signalCount }) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      {sectors.map((s) => {
        const isActive = value === s.id
        return (
          <button
            key={s.id}
            onClick={() => onChange(s.id)}
            className={`
              px-4 py-2 rounded-lg text-xs font-semibold uppercase tracking-wider
              border transition-all duration-200
              ${isActive
                ? `bg-${s.color}/10 border-${s.color}/50 text-${s.color} shadow-neon-cyan`
                : 'bg-terminal-surface border-terminal-border text-terminal-muted hover:border-terminal-muted/40 hover:text-terminal-text'
              }
            `}
          >
            {s.label}
            {isActive && signalCount !== undefined && (
              <span className="ml-2 font-mono">{signalCount}</span>
            )}
          </button>
        )
      })}
    </div>
  )
}