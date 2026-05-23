import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
         ReferenceLine, ResponsiveContainer } from 'recharts'

export default function IndicatorChart({ data, title, dataKey, color, refLines = [] }) {
  if (!data || data.length === 0) return null
  const chartData = data.slice(-120)

  return (
    <div>
      <h4 className="text-xs font-mono uppercase tracking-widest text-terminal-muted mb-2">{title}</h4>
      <ResponsiveContainer width="100%" height={120}>
        <LineChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis
            dataKey="Date"
            stroke="#6b7280"
            tick={{ fontSize: 9 }}
            tickFormatter={(d) => d?.slice(5) || ''}
            minTickGap={30}
          />
          <YAxis stroke="#6b7280" tick={{ fontSize: 9 }} domain={['auto', 'auto']} />
          <Tooltip
            contentStyle={{
              backgroundColor: '#0d1117',
              border: '1px solid #1f2937',
              borderRadius: '8px',
              fontSize: '11px',
            }}
          />
          {refLines.map((rl, i) => (
            <ReferenceLine
              key={i}
              y={rl.value}
              stroke={rl.color}
              strokeDasharray="3 3"
              strokeOpacity={0.5}
            />
          ))}
          <Line
            type="monotone"
            dataKey={dataKey}
            stroke={color}
            strokeWidth={1.5}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}