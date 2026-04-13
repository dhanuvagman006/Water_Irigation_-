import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Legend, ResponsiveContainer } from 'recharts'
import type { ModelMetrics } from '../../types'
import { MODEL_COLORS } from '../../utils/formatters'

interface ModelComparisonRadarProps {
  data: ModelMetrics[]
  title?: string
}

export default function ModelComparisonRadar({ data, title = 'Model Comparison' }: ModelComparisonRadarProps) {
  // Normalize metrics for radar chart (all axes should be 0-1 where higher is better)
  // For RMSE and MAE, we invert (1 - normalized) so lower is better on chart
  const maxRmse = Math.max(...data.map((d) => d.rmse))
  const maxMae = Math.max(...data.map((d) => d.mae))

  const radarData = [
    {
      metric: 'R²',
      ...Object.fromEntries(data.map((d) => [d.model, d.r2 ?? 0])),
    },
    {
      metric: 'NSE',
      ...Object.fromEntries(data.map((d) => [d.model, d.nse ?? 0])),
    },
    {
      metric: '1-RMSE',
      ...Object.fromEntries(data.map((d) => [d.model, 1 - (d.rmse ?? maxRmse) / (maxRmse * 1.2)])),
    },
    {
      metric: '1-MAE',
      ...Object.fromEntries(data.map((d) => [d.model, 1 - (d.mae ?? maxMae) / (maxMae * 1.2)])),
    },
  ]

  if (data.some(d => d.accuracy != null)) {
    radarData.push({
      metric: 'Accuracy',
      ...Object.fromEntries(data.map((d) => [d.model, d.accuracy ?? 0])),
    })
  }

  if (data.some(d => d.f1 != null)) {
    radarData.push({
      metric: 'F1',
      ...Object.fromEntries(data.map((d) => [d.model, d.f1 ?? 0])),
    })
  }

  return (
    <div className="card p-5">
      <h3 className="text-sm font-semibold text-text-primary dark:text-white mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height={350}>
        <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="70%">
          <PolarGrid stroke="#E5E7EB" className="dark:stroke-gray-700" />
          <PolarAngleAxis
            dataKey="metric"
            tick={{ fontSize: 11, fill: '#6B7280' }}
          />
          <PolarRadiusAxis
            angle={30}
            domain={[0, 1]}
            tick={{ fontSize: 9, fill: '#9CA3AF' }}
          />
          {data.map((model) => (
            <Radar
              key={model.model}
              name={model.model}
              dataKey={model.model}
              stroke={MODEL_COLORS[model.model]}
              fill={MODEL_COLORS[model.model]}
              fillOpacity={0.1}
              strokeWidth={2}
            />
          ))}
          <Legend
            wrapperStyle={{ fontSize: 11 }}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  )
}
