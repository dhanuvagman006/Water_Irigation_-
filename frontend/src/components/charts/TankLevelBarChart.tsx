import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'
import { formatDate, TANK_LEVEL_COLORS } from '../../utils/formatters'
import type { TankPrediction } from '../../types'

interface TankLevelBarChartProps {
  data: TankPrediction[]
}

interface TooltipEntry {
  payload: {
    level: string
    percentage: number
  }
}

interface CustomTooltipProps {
  active?: boolean
  payload?: TooltipEntry[]
  label?: string
}

function CustomTooltip({ active, payload, label }: CustomTooltipProps) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="bg-white dark:bg-surface-dark border border-gray-100 dark:border-gray-700 rounded-lg shadow-lg p-3 text-xs">
      <p className="font-semibold text-text-primary dark:text-white mb-1">
        {formatDate(label, 'EEEE, MMM dd')}
      </p>
      <div className="flex items-center gap-2">
        <div
          className="w-2 h-2 rounded-full"
          style={{ backgroundColor: TANK_LEVEL_COLORS[d.level as keyof typeof TANK_LEVEL_COLORS] }}
        />
        <span className="text-text-muted dark:text-text-dark-muted">Level:</span>
        <span className="font-semibold text-text-primary dark:text-white">{d.level}</span>
      </div>
      <div className="flex items-center gap-2 mt-0.5">
        <div className="w-2 h-2 rounded-full bg-transparent" />
        <span className="text-text-muted dark:text-text-dark-muted">Fill:</span>
        <span className="font-mono font-medium text-text-primary dark:text-white">{d.percentage}%</span>
      </div>
    </div>
  )
}

export default function TankLevelBarChart({ data }: TankLevelBarChartProps) {
  return (
    <div className="card p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-text-primary dark:text-white">
          Tank Level — 14 Days
        </h3>
        <div className="flex gap-3">
          {Object.entries(TANK_LEVEL_COLORS).map(([level, color]) => (
            <div key={level} className="flex items-center gap-1.5 text-xs text-text-muted dark:text-text-dark-muted">
              <div className="w-2.5 h-2.5 rounded" style={{ backgroundColor: color }} />
              {level}
            </div>
          ))}
        </div>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="text-gray-100 dark:text-gray-800" />
          <XAxis
            dataKey="date"
            tickFormatter={(d) => formatDate(d)}
            tick={{ fontSize: 11 }}
            stroke="#6B7280"
          />
          <YAxis
            domain={[0, 100]}
            tick={{ fontSize: 11 }}
            stroke="#6B7280"
            label={{ value: '%', angle: -90, position: 'insideLeft', style: { fontSize: 11, fill: '#6B7280' } }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="percentage" radius={[4, 4, 0, 0]} maxBarSize={40}>
            {data.map((entry, i) => (
              <Cell
                key={i}
                fill={TANK_LEVEL_COLORS[entry.level]}
                fillOpacity={0.85}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
