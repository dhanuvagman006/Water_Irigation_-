import { useState, useMemo, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Play, Download, Droplet, TrendingUp, AlertCircle } from 'lucide-react'
import {
  useReactTable, getCoreRowModel, getSortedRowModel, flexRender, createColumnHelper,
  type SortingState,
} from '@tanstack/react-table'
import ModelSelector from '../components/shared/ModelSelector'
import CropStatusCard from '../components/cards/CropStatusCard'
import AlertCard from '../components/cards/AlertCard'
import IrrigationHeatmap from '../components/charts/IrrigationHeatmap'
import ErrorBoundary from '../components/shared/ErrorBoundary'
import { ChartSkeleton } from '../components/shared/LoadingSpinner'
import { useIrrigationPrediction } from '../api/irrigationApi'
import { formatDate, exportToCSV } from '../utils/formatters'
import type { ModelName, CropType, IrrigationPlan } from '../types'

const columnHelper = createColumnHelper<IrrigationPlan>()

export default function IrrigationPage() {
  const [model, setModel] = useState<ModelName>('LSTM')
  const [soilMoisture, setSoilMoisture] = useState(0.35)
  const [selectedCrops, setSelectedCrops] = useState<CropType[]>(['Arecanut', 'Coconut', 'Pepper'])
  const [sorting, setSorting] = useState<SortingState>([])

  const mutation = useIrrigationPrediction()
  const plans = (mutation.data ?? []) as IrrigationPlan[]

  // Auto-generate on first load
  useEffect(() => {
    if (!mutation.data && !mutation.isError && !mutation.isPending) {
      mutation.mutate({
        soil_moisture: soilMoisture,
        crop_types: selectedCrops,
        model,
      })
    }
  }, [])

  const handleGenerate = () => {
    mutation.mutate({
      soil_moisture: soilMoisture,
      crop_types: selectedCrops,
      model,
    })
  }

  const toggleCrop = (crop: CropType) => {
    setSelectedCrops((prev) =>
      prev.includes(crop) ? prev.filter((c) => c !== crop) : [...prev, crop]
    )
  }

  // Today's status per crop
  const todayStatus = useMemo(() => {
    const today = plans.length > 0 ? plans[0].date : ''
    return (['Arecanut', 'Coconut', 'Pepper'] as CropType[]).map((crop) => {
      const plan = plans.find((p) => p.date === today && p.crop === crop)
      return {
        crop,
        decision: plan?.decision ?? 'No Irrigate' as const,
        waterLiters: plan?.water_liters ?? 0,
      }
    })
  }, [plans])

  // Calculate total water for today
  const todayTotalWater = useMemo(() => {
    return todayStatus.reduce((sum, s) => sum + s.waterLiters, 0)
  }, [todayStatus])

  // Count irrigate decisions in 14-day plan
  const irrigateDaysCount = useMemo(() => {
    return plans.filter((p) => p.decision === 'Irrigate').length
  }, [plans])

  const columns = useMemo(() => [
    columnHelper.accessor('date', {
      header: 'Date',
      cell: (info) => <span className="text-sm font-medium">{formatDate(info.getValue(), 'MMM dd')}</span>,
    }),
    columnHelper.accessor('crop', {
      header: 'Crop',
      cell: (info) => <span className="text-sm">{info.getValue()}</span>,
    }),
    columnHelper.accessor('decision', {
      header: 'Decision',
      cell: (info) => {
        const d = info.getValue()
        const colorClass = d === 'Irrigate' ? 'badge-success' : d === 'Monitor' ? 'badge-warning' : 'badge-neutral'
        return <span className={colorClass}>{d}</span>
      },
    }),
    columnHelper.accessor('water_liters', {
      header: 'Water (L)',
      cell: (info) => (
        <span className="font-mono text-sm">{info.getValue() > 0 ? `${info.getValue()}L` : '—'}</span>
      ),
    }),
    columnHelper.accessor('soil_moisture', {
      header: 'Soil Moisture',
      cell: (info) => {
        const val = info.getValue()
        if (val == null) return <span className="text-text-muted text-sm">—</span>
        return <span className="font-mono text-sm">{(val * 100).toFixed(0)}%</span>
      },
    }),
    columnHelper.accessor('reason', {
      header: 'Reason',
      cell: (info) => (
        <span className="text-xs text-text-muted dark:text-text-dark-muted max-w-[200px] truncate block">
          {info.getValue()}
        </span>
      ),
    }),
  ], [])

  // eslint-disable-next-line react-hooks/incompatible-library
  const table = useReactTable({
    data: plans,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  })

  return (
    <ErrorBoundary>
      <div className="space-y-6">
        {/* Page header */}
        <motion.div initial={{ opacity: 0, y: -12 }} animate={{ opacity: 1, y: 0 }}>
          <h1 className="text-2xl font-bold text-text-primary dark:text-white mb-1">Smart Irrigation Planner</h1>
          <p className="text-sm text-text-muted dark:text-text-dark-muted">AI-powered 14-day irrigation scheduling for optimal crop yield</p>
        </motion.div>

        {/* Controls bar */}
        <motion.div
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          className="card p-4 flex flex-wrap items-end gap-4"
        >
          {/* Soil moisture slider */}
          <div className="min-w-[200px]">
            <div className="flex justify-between text-xs mb-1">
              <span className="text-text-muted dark:text-text-dark-muted font-medium">Soil Moisture</span>
              <span className="font-mono font-semibold text-text-primary dark:text-white">{soilMoisture != null ? soilMoisture.toFixed(2) : '—'}</span>
            </div>
            <input
              type="range" min="0" max="1" step="0.01" value={soilMoisture}
              onChange={(e) => setSoilMoisture(Number(e.target.value))}
              className="w-full accent-primary h-1.5"
            />
          </div>

          {/* Crop selection */}
          <div>
            <span className="text-xs text-text-muted dark:text-text-dark-muted font-medium block mb-1.5">Crops</span>
            <div className="flex gap-2">
              {(['Arecanut', 'Coconut', 'Pepper'] as CropType[]).map((crop) => (
                <label
                  key={crop}
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium cursor-pointer transition-all
                    ${selectedCrops.includes(crop)
                      ? 'bg-primary/10 text-primary border border-primary/30'
                      : 'bg-gray-50 dark:bg-gray-800 text-text-muted dark:text-text-dark-muted border border-transparent'}`}
                >
                  <input
                    type="checkbox"
                    checked={selectedCrops.includes(crop)}
                    onChange={() => toggleCrop(crop)}
                    className="sr-only"
                  />
                  {crop}
                </label>
              ))}
            </div>
          </div>

          <div>
            <label className="text-xs text-text-muted dark:text-text-dark-muted font-medium block mb-1.5">Model</label>
            <ModelSelector value={model} onChange={setModel} />
          </div>

          <button onClick={handleGenerate} className="btn-primary" disabled={mutation.isPending}>
            <Play className="w-4 h-4" />
            Generate 14-Day Plan
          </button>
        </motion.div>

        {mutation.isError && (
          <AlertCard 
            type="danger" 
            title="Generation Failed" 
            message={mutation.error.message || "Failed to generate irrigation plan. Check NASA data."} 
          />
        )}

        {/* Insights cards */}
        {plans.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="grid grid-cols-1 sm:grid-cols-2 gap-4"
          >
            {/* Today's water need */}
            <div className="card p-5 border-l-4 border-l-blue-500 dark:border-l-blue-400">
              <div className="flex items-start justify-between mb-2">
                <div>
                  <p className="text-xs font-medium text-text-muted dark:text-text-dark-muted uppercase tracking-wide mb-1">
                    Today's Water Need
                  </p>
                  <p className="text-3xl font-bold text-text-primary dark:text-white">{todayTotalWater.toFixed(0)}<span className="text-lg ml-1">L</span></p>
                </div>
                <Droplet className="w-8 h-8 text-blue-500 dark:text-blue-400 opacity-20" />
              </div>
              <p className="text-xs text-text-muted dark:text-text-dark-muted">
                for {selectedCrops.length} crop{selectedCrops.length !== 1 ? 's' : ''}
              </p>
            </div>

            {/* 14-day forecast */}
            <div className="card p-5 border-l-4 border-l-green-500 dark:border-l-green-400">
              <div className="flex items-start justify-between mb-2">
                <div>
                  <p className="text-xs font-medium text-text-muted dark:text-text-dark-muted uppercase tracking-wide mb-1">
                    Irrigation Days
                  </p>
                  <p className="text-3xl font-bold text-text-primary dark:text-white">{irrigateDaysCount}<span className="text-lg ml-1">/42</span></p>
                </div>
                <TrendingUp className="w-8 h-8 text-green-500 dark:text-green-400 opacity-20" />
              </div>
              <p className="text-xs text-text-muted dark:text-text-dark-muted">
                {((irrigateDaysCount / 42) * 100).toFixed(0)}% of 14-day plan
              </p>
            </div>
          </motion.div>
        )}

        {/* Crop status cards */}
        {mutation.isPending && plans.length === 0 ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="grid grid-cols-1 md:grid-cols-3 gap-4"
          >
            {[1, 2, 3].map((i) => (
              <div key={i} className="card p-5 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-700 animate-pulse" />
            ))}
          </motion.div>
        ) : plans.length === 0 ? (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="card p-12 text-center bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border border-blue-200 dark:border-blue-800"
          >
            <div className="mb-4 flex justify-center">
              <div className="p-3 rounded-full bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-300">
                <AlertCircle className="w-6 h-6" />
              </div>
            </div>
            <h3 className="text-lg font-semibold text-text-primary dark:text-white mb-2">No Plan Generated Yet</h3>
            <p className="text-sm text-text-muted dark:text-text-dark-muted mb-4">
              Adjust the parameters above and click "Generate 14-Day Plan" to create an irrigation schedule.
            </p>
            <button
              onClick={handleGenerate}
              className="btn-primary inline-flex gap-2"
              disabled={mutation.isPending}
            >
              <Play className="w-4 h-4" />
              Generate Plan Now
            </button>
          </motion.div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {todayStatus.map((s) => (
              <CropStatusCard key={s.crop} {...s} />
            ))}
          </div>
        )}

        {/* Heatmap */}
        {plans.length > 0 && (
          <>
            {mutation.isPending ? <ChartSkeleton /> : <IrrigationHeatmap data={plans} />}

            {/* Schedule table */}
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="card p-5"
            >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-text-primary dark:text-white">14-Day Schedule</h3>
            <button
              onClick={() => exportToCSV(plans as unknown as Record<string, string | number>[], 'irrigation_schedule')}
              className="btn-secondary text-xs"
            >
              <Download className="w-3.5 h-3.5" />
              Export CSV
            </button>
          </div>
          <div className="overflow-x-auto custom-scrollbar">
            <table className="w-full text-left">
              <thead>
                {table.getHeaderGroups().map((hg) => (
                  <tr key={hg.id} className="border-b border-gray-100 dark:border-gray-800">
                    {hg.headers.map((h) => (
                      <th
                        key={h.id}
                        className="text-xs font-semibold text-text-muted dark:text-text-dark-muted uppercase tracking-wider py-3 px-3 cursor-pointer hover:text-text-primary dark:hover:text-white"
                        onClick={h.column.getToggleSortingHandler()}
                      >
                        <div className="flex items-center gap-1">
                          {flexRender(h.column.columnDef.header, h.getContext())}
                          {{ asc: ' ↑', desc: ' ↓' }[h.column.getIsSorted() as string] ?? ''}
                        </div>
                      </th>
                    ))}
                  </tr>
                ))}
              </thead>
              <tbody>
                {table.getRowModel().rows.map((row) => (
                  <tr key={row.id} className="border-b border-gray-50 dark:border-gray-800/50 hover:bg-gray-50/50 dark:hover:bg-gray-800/30 transition-colors">
                    {row.getVisibleCells().map((cell) => (
                      <td key={cell.id} className="py-2.5 px-3">
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
            </motion.div>
          </>
        )}
      </div>
    </ErrorBoundary>
  )
}
