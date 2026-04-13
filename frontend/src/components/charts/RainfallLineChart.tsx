import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Legend,
} from 'recharts'
import { formatDate } from '../../utils/formatters'
import type { RainfallPrediction } from '../../types'

interface RainfallLineChartProps {
  data: RainfallPrediction[]
  showActual?: boolean
}

interface TooltipEntry {
  dataKey?: string
  color?: string
  name?: string
  value?: number
}

interface CustomTooltipProps {
  active?: boolean
  payload?: TooltipEntry[]
  label?: string
}

function CustomTooltip({ active, payload, label }: CustomTooltipProps) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-white dark:bg-surface-dark border border-gray-100 dark:border-gray-700 rounded-lg shadow-lg p-3 text-xs">
      <p className="font-semibold text-text-primary dark:text-white mb-1.5">
        {formatDate(label, 'EEEE, MMM dd')}
      </p>
      {payload.map((entry: TooltipEntry) => (
        <div key={entry.dataKey} className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
          <span className="text-text-muted dark:text-text-dark-muted">{entry.name}:</span>
          <span className="font-mono font-medium text-text-primary dark:text-white">
            {entry.value != null ? `${entry.value.toFixed(1)} mm` : '—'}
          </span>
        </div>
      ))}
    </div>
  )
}

export default function RainfallLineChart({ data, showActual = true }: RainfallLineChartProps) {
  return (
    <div className="card p-5">
      <h3 className="text-sm font-semibold text-text-primary dark:text-white mb-4">
        14-Day Rainfall Prediction
      </h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="text-gray-100 dark:text-gray-800" />
          <XAxis
            dataKey="date"
            tickFormatter={(d) => formatDate(d)}
            tick={{ fontSize: 11 }}
            stroke="#6B7280"
          />
          <YAxis
            tick={{ fontSize: 11 }}
            stroke="#6B7280"
            label={{ value: 'mm', angle: -90, position: 'insideLeft', style: { fontSize: 11, fill: '#6B7280' } }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: 12 }}
          />
          <ReferenceLine
            y={10}
            stroke="#BA7517"
            strokeDasharray="6 3"
            label={{ value: 'Irrigation Threshold', position: 'right', style: { fontSize: 10, fill: '#BA7517' } }}
          />
          <Line
            type="monotone"
            dataKey="predicted_mm"
            name="Predicted"
            stroke="#0F6E56"
            strokeWidth={2.5}
            dot={{ r: 4, fill: '#0F6E56', stroke: '#fff', strokeWidth: 2 }}
            activeDot={{ r: 6 }}
          />
          {showActual && (
            <Line
              type="monotone"
              dataKey="actual_mm"
              name="Actual"
              stroke="#378ADD"
              strokeWidth={2}
              strokeDasharray="5 3"
              dot={{ r: 3, fill: '#378ADD', stroke: '#fff', strokeWidth: 2 }}
              connectNulls
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
