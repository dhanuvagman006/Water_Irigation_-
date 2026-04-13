import { useMemo, useEffect } from 'react'
import { CloudRain, Database, Sprout, CalendarClock } from 'lucide-react'
import {
  useReactTable,
  getCoreRowModel,
  flexRender,
  createColumnHelper,
} from '@tanstack/react-table'
import { motion } from 'framer-motion'
import StatCard from '../components/cards/StatCard'
import RainfallLineChart from '../components/charts/RainfallLineChart'
import TankLevelBarChart from '../components/charts/TankLevelBarChart'
import ErrorBoundary from '../components/shared/ErrorBoundary'
import { useRainfallPrediction } from '../api/rainfallApi'
import { useIrrigationPrediction } from '../api/irrigationApi'
import { useTankPrediction } from '../api/tankApi'
import { ChartSkeleton } from '../components/shared/LoadingSpinner'
import useAppStore from '../store/useAppStore'
import type { RainfallPrediction, TankPrediction, IrrigationPlan } from '../types'
import { formatDate } from '../utils/formatters'

// Build summary table data
type DaySummary = {
  date: string
  arecanut: string
  coconut: string
  pepper: string
  totalWater: number
}

const columnHelper = createColumnHelper<DaySummary>()

function DecisionBadge({ decision }: { decision: string }) {
  const colorMap: Record<string, string> = {
    Irrigate: 'badge-success',
    'No Irrigate': 'badge-neutral',
    Monitor: 'badge-warning',
  }
  return <span className={colorMap[decision] || 'badge-neutral'}>{decision}</span>
}

export default function Dashboard() {
  const { selectedModel } = useAppStore()
  const { data: rainfallData, isLoading: rainfallLoading } = useRainfallPrediction(selectedModel)
  const tankMutation = useTankPrediction()
  const irrigationMutation = useIrrigationPrediction()

  const rainfall: RainfallPrediction[] = (rainfallData ?? [])
  const tank: TankPrediction[] = (tankMutation.data ?? [])
  const irrigation: IrrigationPlan[] = (irrigationMutation.data ?? [])
  
  // Auto-fetch tank and irrigation predictions on first load
  useEffect(() => {
    // Fetch tank predictions
    if (!tankMutation.data && !tankMutation.isPending && !tankMutation.isError) {
      tankMutation.mutate({
        roof_area: 100, // 100 m²
        tank_capacity: 50000, // 50,000 liters
        current_level: 40000, // 40,000 liters
        daily_consumption: 1000, // 1,000 liters/day
        model: selectedModel,
      })
    }
    
    // Fetch irrigation predictions
    if (!irrigationMutation.data && !irrigationMutation.isPending && !irrigationMutation.isError) {
      irrigationMutation.mutate({
        soil_moisture: 0.35,
        crop_types: ['Arecanut', 'Coconut', 'Pepper'],
        model: selectedModel,
      })
    }
  }, [])
  
  const isLoading = rainfallLoading || tankMutation.isPending || irrigationMutation.isPending

  // Compute stats
  const todayRainfall = rainfall[0]?.predicted_mm ?? 0
  const currentTank = tank[0] || { percentage: 0, level: 'N/A' }
  const activeCrops = 3
  const nextIrrigation = useMemo(() => {
    const next = irrigation.find((i) => i.decision === 'Irrigate')
    if (!next) return 'N/A'
    const daysDiff = Math.ceil(
      (new Date(next.date).getTime() - Date.now()) / (1000 * 60 * 60 * 24)
    )
    return daysDiff <= 0 ? 'Today' : `${daysDiff}`
  }, [irrigation])

  // Irrigation summary table
  const summaryData = useMemo(() => {
    const dates = [...new Set(irrigation.map((i) => i.date))].sort()
    return dates.map((date) => {
      const dayPlans = irrigation.filter((i) => i.date === date)
      return {
        date,
        arecanut: dayPlans.find((p) => p.crop === 'Arecanut')?.decision || '-',
        coconut: dayPlans.find((p) => p.crop === 'Coconut')?.decision || '-',
        pepper: dayPlans.find((p) => p.crop === 'Pepper')?.decision || '-',
        totalWater: dayPlans.reduce((sum, p) => sum + p.water_liters, 0),
      }
    })
  }, [irrigation])

  const columns = useMemo(() => [
    columnHelper.accessor('date', {
      header: 'Date',
      cell: (info) => (
        <span className="text-sm font-medium">{formatDate(info.getValue(), 'MMM dd, EEE')}</span>
      ),
    }),
    columnHelper.accessor('arecanut', {
      header: 'Arecanut',
      cell: (info) => <DecisionBadge decision={info.getValue()} />,
    }),
    columnHelper.accessor('coconut', {
      header: 'Coconut',
      cell: (info) => <DecisionBadge decision={info.getValue()} />,
    }),
    columnHelper.accessor('pepper', {
      header: 'Pepper',
      cell: (info) => <DecisionBadge decision={info.getValue()} />,
    }),
    columnHelper.accessor('totalWater', {
      header: 'Total Water (L)',
      cell: (info) => (
        <span className="font-mono text-sm font-medium">
          {info.getValue() > 0 ? `${info.getValue()}L` : '—'}
        </span>
      ),
    }),
  ], [])

  // eslint-disable-next-line react-hooks/incompatible-library
  const table = useReactTable({
    data: summaryData,
    columns,
    getCoreRowModel: getCoreRowModel(),
  })

  return (
    <ErrorBoundary>
      <div className="space-y-6">
        {/* Stat cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            title="Today's Predicted Rainfall"
            value={todayRainfall}
            suffix="mm"
            icon={CloudRain}
            color="accent"
            loading={isLoading}
            trend={{ value: 12, label: 'vs yesterday' }}
          />
          <StatCard
            title="Tank Status"
            value={currentTank?.percentage ?? 0}
            suffix={`% (${currentTank?.level})`}
            icon={Database}
            color={currentTank?.level === 'Low' ? 'danger' : currentTank?.level === 'Medium' ? 'warning' : 'success'}
            loading={isLoading}
          />
          <StatCard
            title="Active Crops Monitored"
            value={activeCrops}
            suffix="crops"
            icon={Sprout}
            color="primary"
            loading={isLoading}
          />
          <StatCard
            title="Next Irrigation"
            value={nextIrrigation === 'Today' ? 'Today' : nextIrrigation}
            suffix={nextIrrigation !== 'Today' && nextIrrigation !== 'N/A' ? 'days' : ''}
            icon={CalendarClock}
            color="warning"
            loading={isLoading}
          />
        </div>

        {/* Charts row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {isLoading ? (
            <>
              <ChartSkeleton />
              <ChartSkeleton />
            </>
          ) : (
            <>
              <RainfallLineChart data={rainfall} />
              <TankLevelBarChart data={tank} />
            </>
          )}
        </div>

        {/* Irrigation summary table */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="card p-5"
        >
          <h3 className="text-sm font-semibold text-text-primary dark:text-white mb-4">
            Irrigation Schedule Summary
          </h3>
          <div className="overflow-x-auto custom-scrollbar">
            <table className="w-full text-left">
              <thead>
                {table.getHeaderGroups().map((hg) => (
                  <tr key={hg.id} className="border-b border-gray-100 dark:border-gray-800">
                    {hg.headers.map((h) => (
                      <th
                        key={h.id}
                        className="text-xs font-semibold text-text-muted dark:text-text-dark-muted uppercase tracking-wider py-3 px-4"
                      >
                        {flexRender(h.column.columnDef.header, h.getContext())}
                      </th>
                    ))}
                  </tr>
                ))}
              </thead>
              <tbody>
                {table.getRowModel().rows.map((row) => (
                  <tr
                    key={row.id}
                    className="border-b border-gray-50 dark:border-gray-800/50 hover:bg-gray-50/50 dark:hover:bg-gray-800/30 transition-colors"
                  >
                    {row.getVisibleCells().map((cell) => (
                      <td key={cell.id} className="py-3 px-4">
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>
      </div>
    </ErrorBoundary>
  )
}
