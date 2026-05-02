import { formatDate } from '../../utils/formatters'
import type { IrrigationPlan, CropType } from '../../types'

interface IrrigationHeatmapProps {
  data: IrrigationPlan[]
}

const crops: CropType[] = ['Arecanut', 'Coconut', 'Pepper']

function formatLiters(liters: number): string {
  if (liters <= 0) return ''
  if (liters >= 100) return liters.toFixed(0)
  return liters.toFixed(1)
}

function getIntensityColor(liters: number, decision: string, maxLiters: number): string {
  if (decision === 'No Irrigate') return '#F3F4F6'
  if (decision === 'Monitor') return '#FDE68A'

  const intensity = maxLiters > 0 ? Math.min(liters / maxLiters, 1) : 0
  const lightness = Math.round(78 - intensity * 36)
  return `hsl(145, 58%, ${lightness}%)`
}

function getTextColor(liters: number, maxLiters: number): string {
  if (maxLiters <= 0 || liters / maxLiters < 0.45) return 'text-gray-700'
  return 'text-white'
}

export default function IrrigationHeatmap({ data }: IrrigationHeatmapProps) {
  const dates = [...new Set(data.map((d) => d.date))].sort()
  const maxLiters = Math.max(0, ...data.map((d) => d.water_liters))
  const gridTemplateColumns = `6rem repeat(${dates.length}, minmax(76px, 1fr))`

  return (
    <div className="card p-5">
      <h3 className="text-sm font-semibold text-text-primary dark:text-white mb-4">
        14-Day Irrigation Heatmap
      </h3>
      <div className="overflow-x-auto custom-scrollbar">
        <div className="min-w-[1180px]">
          <div className="grid gap-2 mb-2" style={{ gridTemplateColumns }}>
            <div />
            {dates.map((date) => (
              <div
                key={date}
                className="text-center text-xs font-medium text-text-muted dark:text-text-dark-muted"
              >
                {formatDate(date)}
              </div>
            ))}
          </div>

          {crops.map((crop) => (
            <div key={crop} className="grid gap-2 mb-2" style={{ gridTemplateColumns }}>
              <div className="flex items-center text-sm font-semibold text-text-primary dark:text-text-dark">
                {crop}
              </div>
              {dates.map((date) => {
                const item = data.find((d) => d.date === date && d.crop === crop)
                if (!item) {
                  return <div key={date} className="h-12 bg-gray-50 dark:bg-gray-800 rounded-md" />
                }

                const displayValue = formatLiters(item.water_liters)

                return (
                  <div
                    key={date}
                    className="h-12 rounded-md cursor-pointer transition-transform hover:scale-[1.03] group relative overflow-visible"
                    style={{
                      backgroundColor: getIntensityColor(item.water_liters, item.decision, maxLiters),
                    }}
                    title={`${item.decision}: ${item.water_liters.toFixed(1)}L - ${item.reason}`}
                  >
                    {item.water_liters > 0 && (
                      <div className={`h-full flex items-center justify-center px-2 text-xs font-mono font-semibold tabular-nums ${getTextColor(item.water_liters, maxLiters)}`}>
                        <span className="truncate">{displayValue}</span>
                      </div>
                    )}
                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block z-10">
                      <div className="bg-white dark:bg-surface-dark border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg p-2 text-xs whitespace-nowrap">
                        <p className="font-semibold">{item.decision}</p>
                        <p className="text-text-muted dark:text-text-dark-muted">
                          {item.water_liters.toFixed(1)}L - {item.reason}
                        </p>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          ))}

          <div className="flex items-center gap-4 mt-4 pt-3 border-t border-gray-100 dark:border-gray-800">
            <span className="text-[10px] text-text-muted dark:text-text-dark-muted font-medium">Intensity:</span>
            <div className="flex items-center gap-1">
              <div className="w-5 h-3 rounded bg-gray-100 dark:bg-gray-800" />
              <span className="text-[10px] text-text-muted dark:text-text-dark-muted">None</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-5 h-3 rounded" style={{ backgroundColor: '#FDE68A' }} />
              <span className="text-[10px] text-text-muted dark:text-text-dark-muted">Monitor</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-5 h-3 rounded" style={{ backgroundColor: 'hsl(145, 58%, 68%)' }} />
              <span className="text-[10px] text-text-muted dark:text-text-dark-muted">Low</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-5 h-3 rounded" style={{ backgroundColor: 'hsl(145, 58%, 42%)' }} />
              <span className="text-[10px] text-text-muted dark:text-text-dark-muted">High</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
