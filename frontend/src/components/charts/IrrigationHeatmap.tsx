import { formatDate } from '../../utils/formatters'
import type { IrrigationPlan, CropType } from '../../types'

interface IrrigationHeatmapProps {
  data: IrrigationPlan[]
}

const crops: CropType[] = ['Arecanut', 'Coconut', 'Pepper']

function getIntensityColor(liters: number, decision: string): string {
  if (decision === 'No Irrigate') return '#F3F4F6'
  if (decision === 'Monitor') return '#FDE68A'
  // Irrigate: gradient from light to dark green based on liters
  const intensity = Math.min(liters / 60, 1)
  const lightness = Math.round(85 - intensity * 45)
  return `hsl(140, 60%, ${lightness}%)`
}


export default function IrrigationHeatmap({ data }: IrrigationHeatmapProps) {
  // Group data by date
  const dates = [...new Set(data.map((d) => d.date))].sort()

  return (
    <div className="card p-5">
      <h3 className="text-sm font-semibold text-text-primary dark:text-white mb-4">
        14-Day Irrigation Heatmap
      </h3>
      <div className="overflow-x-auto custom-scrollbar">
        <div className="min-w-[600px]">
          {/* Header row */}
          <div className="flex gap-1.5 mb-2">
            <div className="w-20 shrink-0" />
            {dates.map((date) => (
              <div
                key={date}
                className="flex-1 text-center text-[10px] font-medium text-text-muted dark:text-text-dark-muted"
              >
                {formatDate(date)}
              </div>
            ))}
          </div>

          {/* Crop rows */}
          {crops.map((crop) => (
            <div key={crop} className="flex gap-1.5 mb-1.5">
              <div className="w-20 shrink-0 flex items-center text-xs font-medium text-text-primary dark:text-text-dark">
                {crop}
              </div>
              {dates.map((date) => {
                const item = data.find((d) => d.date === date && d.crop === crop)
                if (!item) return <div key={date} className="flex-1 h-10 bg-gray-50 dark:bg-gray-800 rounded" />
                return (
                  <div
                    key={date}
                    className="flex-1 h-10 rounded cursor-pointer transition-transform hover:scale-105 group relative"
                    style={{
                      backgroundColor: getIntensityColor(item.water_liters, item.decision),
                    }}
                    title={`${item.decision}: ${item.water_liters}L — ${item.reason}`}
                  >
                    {item.water_liters > 0 && (
                      <div className="h-full flex items-center justify-center text-[10px] font-mono font-medium text-gray-700">
                        {item.water_liters}
                      </div>
                    )}
                    {/* Tooltip on hover */}
                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block z-10">
                      <div className="bg-white dark:bg-surface-dark border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg p-2 text-xs whitespace-nowrap">
                        <p className="font-semibold">{item.decision}</p>
                        <p className="text-text-muted dark:text-text-dark-muted">{item.water_liters}L — {item.reason}</p>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          ))}

          {/* Legend */}
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
              <div className="w-5 h-3 rounded" style={{ backgroundColor: 'hsl(140, 60%, 70%)' }} />
              <span className="text-[10px] text-text-muted dark:text-text-dark-muted">Low</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-5 h-3 rounded" style={{ backgroundColor: 'hsl(140, 60%, 45%)' }} />
              <span className="text-[10px] text-text-muted dark:text-text-dark-muted">High</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
