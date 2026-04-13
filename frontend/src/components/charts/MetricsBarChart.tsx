import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, LabelList,
} from 'recharts'
import type { ModelMetrics } from '../../types'
import { MODEL_COLORS } from '../../utils/formatters'

interface MetricsBarChartProps {
  data: ModelMetrics[]
  metric?: 'rmse' | 'mae' | 'r2' | 'nse'
  title?: string
}

export default function MetricsBarChart({ data, metric = 'rmse', title }: MetricsBarChartProps) {
  const chartData = data
    .map((d) => ({
      model: d.model,
      value: d[metric],
    }))
    .sort((a, b) => (metric === 'r2' || metric === 'nse' ? b.value - a.value : a.value - b.value))

  const metricLabels: Record<string, string> = {
    rmse: 'RMSE',
    mae: 'MAE',
    r2: 'R²',
    nse: 'NSE',
  }

  return (
    <div className="card p-5">
      <h3 className="text-sm font-semibold text-text-primary dark:text-white mb-4">
        {title || `${metricLabels[metric]} Comparison`}
      </h3>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 60, left: 70, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="text-gray-100 dark:text-gray-800" horizontal={false} />
          <XAxis type="number" tick={{ fontSize: 11 }} stroke="#6B7280" />
          <YAxis
            type="category"
            dataKey="model"
            tick={{ fontSize: 11 }}
            stroke="#6B7280"
            width={70}
          />
          <Tooltip
            contentStyle={{
              fontSize: 12,
              backgroundColor: 'var(--tooltip-bg, #fff)',
              border: '1px solid var(--tooltip-border, #E5E7EB)',
              borderRadius: 8,
            }}
            formatter={(value: number) => [value.toFixed(3), metricLabels[metric]]}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={28}>
            {chartData.map((entry) => (
              <Cell key={entry.model} fill={MODEL_COLORS[entry.model]} fillOpacity={0.85} />
            ))}
            <LabelList
              dataKey="value"
              position="right"
              style={{ fontSize: 10, fill: '#6B7280' }}
              formatter={(v: number) => v.toFixed(3)}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
