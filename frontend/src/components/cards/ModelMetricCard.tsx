import { formatMetric } from '../../utils/formatters'
import type { ModelMetrics } from '../../types'

interface ModelMetricCardProps {
  metrics: ModelMetrics
  highlightMetric?: string
}

export default function ModelMetricCard({ metrics, highlightMetric }: ModelMetricCardProps) {
  const metricItems = [
    { key: 'rmse', label: 'RMSE', value: metrics.rmse },
    { key: 'mae', label: 'MAE', value: metrics.mae },
    { key: 'r2', label: 'R²', value: metrics.r2 },
    { key: 'nse', label: 'NSE', value: metrics.nse },
  ]

  return (
    <div className="card p-4">
      <h4 className="font-semibold text-sm text-text-primary dark:text-white mb-3">{metrics.model}</h4>
      <div className="grid grid-cols-2 gap-3">
        {metricItems.map((m) => (
          <div
            key={m.key}
            className={`text-center p-2 rounded-lg ${
              highlightMetric === m.key
                ? 'bg-primary/10 dark:bg-primary/20'
                : 'bg-gray-50 dark:bg-gray-800'
            }`}
          >
            <p className="text-[10px] uppercase tracking-wider text-text-muted dark:text-text-dark-muted font-medium">
              {m.label}
            </p>
            <p className="text-sm font-mono font-semibold text-text-primary dark:text-white mt-0.5">
              {formatMetric(m.value, m.key)}
            </p>
          </div>
        ))}
      </div>
      {(metrics.accuracy !== undefined || metrics.f1 !== undefined) && (
        <div className="flex gap-3 mt-3">
          {metrics.accuracy !== undefined && (
            <div className="flex-1 text-center p-2 rounded-lg bg-gray-50 dark:bg-gray-800">
              <p className="text-[10px] uppercase tracking-wider text-text-muted dark:text-text-dark-muted font-medium">Accuracy</p>
              <p className="text-sm font-mono font-semibold text-text-primary dark:text-white mt-0.5">
                {formatMetric(metrics.accuracy, 'accuracy')}
              </p>
            </div>
          )}
          {metrics.f1 !== undefined && (
            <div className="flex-1 text-center p-2 rounded-lg bg-gray-50 dark:bg-gray-800">
              <p className="text-[10px] uppercase tracking-wider text-text-muted dark:text-text-dark-muted font-medium">F1</p>
              <p className="text-sm font-mono font-semibold text-text-primary dark:text-white mt-0.5">
                {formatMetric(metrics.f1, 'f1')}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
