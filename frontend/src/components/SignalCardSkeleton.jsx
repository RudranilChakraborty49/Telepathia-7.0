export default function SignalCardSkeleton() {
  return (
    <div className="card animate-pulse">
      <div className="flex justify-between mb-4">
        <div>
          <div className="h-5 w-24 bg-terminal-border/50 rounded mb-2" />
          <div className="h-3 w-12 bg-terminal-border/30 rounded" />
        </div>
        <div className="h-5 w-5 bg-terminal-border/30 rounded" />
      </div>
      <div className="h-7 w-32 bg-terminal-border/50 rounded mb-2" />
      <div className="h-3 w-24 bg-terminal-border/30 rounded mb-4" />
      <div className="h-12 bg-terminal-border/30 rounded-lg mb-4" />
      <div className="h-1.5 bg-terminal-border/30 rounded-full mb-4" />
      <div className="grid grid-cols-2 gap-2">
        <div className="h-8 bg-terminal-border/30 rounded" />
        <div className="h-8 bg-terminal-border/30 rounded" />
      </div>
    </div>
  )
}