import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'

export default function PriceChart({ data }) {
  if (!data || data.length === 0) {
    return <div className="text-terminal-muted text-sm py-8 text-center">No price data</div>
  }

  // Show last 120 points
  const chartData = data.slice(-120)

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
        <XAxis
          dataKey="Date"
          stroke="#6b7280"
          tick={{ fontSize: 10 }}
          tickFormatter={(d) => d?.slice(5) || ''}    /* show MM-DD */
          minTickGap={30}
        />
        <YAxis
          stroke="#6b7280"
          tick={{ fontSize: 10 }}
          domain={['auto', 'auto']}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: '#0d1117',
            border: '1px solid #1f2937',
            borderRadius: '8px',
            fontSize: '12px',
          }}
          labelStyle={{ color: '#e5e7eb' }}
        />
        <Legend
          wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }}
          iconType="line"
        />
        <Line
          type="monotone"
          dataKey="Close"
          stroke="#00f0ff"
          strokeWidth={2}
          dot={false}
          name="Close"
        />
        <Line
          type="monotone"
          dataKey="EMA_20"
          stroke="#10b981"
          strokeWidth={1.5}
          dot={false}
          name="EMA 20"
          strokeDasharray="5 3"
        />
        <Line
          type="monotone"
          dataKey="EMA_50"
          stroke="#b400ff"
          strokeWidth={1.5}
          dot={false}
          name="EMA 50"
          strokeDasharray="5 3"
        />
      </LineChart>
    </ResponsiveContainer>
  )
}