import { useState, useMemo } from 'react'
import { motion } from 'framer-motion'
import { Trophy, Award } from 'lucide-react'
import {
  useReactTable, getCoreRowModel, getSortedRowModel, flexRender, createColumnHelper,
  type SortingState,
} from '@tanstack/react-table'
import MetricsBarChart from '../components/charts/MetricsBarChart'
import ModelComparisonRadar from '../components/charts/ModelComparisonRadar'
import ErrorBoundary from '../components/shared/ErrorBoundary'
import { useRainfallMetrics } from '../api/rainfallApi'
import { useIrrigationMetrics } from '../api/irrigationApi'
import { useTankMetrics } from '../api/tankApi'
import { MODEL_COLORS } from '../utils/formatters'
import type { ModelMetrics, ModelName } from '../types'

const tabs = ['Rainfall', 'Tank', 'Irrigation'] as const
type TabType = (typeof tabs)[number]

const columnHelper = createColumnHelper<ModelMetrics>()

function getBest(data: ModelMetrics[], metric: keyof ModelMetrics, lower: boolean = true): ModelName {
  if (!data || data.length === 0) return 'LSTM' as ModelName
  
  // Need to filter out nulls/undefined based on if the model recorded that metric type
  const metricData = data.filter(d => d[metric] != null)
  if (metricData.length === 0) {
    // Fallbacks if metric doesn't exist for this classification
    if (metric === 'rmse') return getBest(data, 'accuracy', false)
    return data[0].model
  }
  
  const sorted = [...metricData].sort((a, b) => {
    const av = a[metric] as number
    const bv = b[metric] as number
    return lower ? av - bv : bv - av
  })
  return sorted[0].model
}

function getConfidence(data: ModelMetrics[]): number {
  if (!data || data.length === 0) return 0
  const r2s = data.map((d) => d.r2).filter(v => v != null) as number[]
  if (r2s.length === 0) return 0 // Provide fallback if no r2 metrics (e.g. classification tasks like tank/irrigation)
  const max = Math.max(...r2s)
  const min = Math.min(...r2s)
  return Math.round((1 - (max - min)) * 100)
}

export default function ModelComparisonPage() {
  const [activeTab, setActiveTab] = useState<TabType>('Rainfall')
  const [sorting, setSorting] = useState<SortingState>([])

  const { data: rainfallMetrics } = useRainfallMetrics()
  const { data: tankMetrics } = useTankMetrics()
  const { data: irrigationMetrics } = useIrrigationMetrics()

  const metricsMap: Record<TabType, ModelMetrics[] | undefined> = useMemo(() => ({
    Rainfall: rainfallMetrics,
    Tank: tankMetrics,
    Irrigation: irrigationMetrics,
  }), [rainfallMetrics, tankMetrics, irrigationMetrics])

  const currentMetrics = useMemo(() => {
    const metrics = metricsMap[activeTab] ?? []
    const validModels: ModelName[] = ['LSTM', 'GRU', 'BiLSTM', 'CNN-LSTM', 'Transformer', 'StackedLSTM']
    
    // Deduplicate and filter, keeping the most recent record
    const latestMetricsMap = new Map<ModelName, ModelMetrics>()
    metrics.forEach((m) => {
      if (validModels.includes(m.model)) {
        latestMetricsMap.set(m.model, m)
      }
    })

    return Array.from(latestMetricsMap.values())
  }, [metricsMap, activeTab])

  const bestModel = useMemo(() => {
    if (!currentMetrics.length) return null
    return getBest(currentMetrics, 'rmse', true)
  }, [currentMetrics])

  const columns = useMemo(() => [
    columnHelper.accessor('model', {
      header: 'Model',
      cell: (info) => {
        const isBest = info.getValue() === bestModel
        return (
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: MODEL_COLORS[info.getValue()] }} />
            <span className={`text-sm font-medium ${isBest ? 'text-primary font-bold' : ''}`}>
              {info.getValue()}
            </span>
            {isBest && <Trophy className="w-3.5 h-3.5 text-warning" />}
          </div>
        )
      },
    }),
    columnHelper.accessor('rmse', {
      header: 'RMSE ↓',
      cell: (info) => {
        const val = info.getValue()
        if (val == null) return <span className="text-text-muted text-sm">—</span>
        const validValues = currentMetrics.map(m => m.rmse).filter(m => m != null) as number[]
        const best = validValues.length ? Math.min(...validValues) : -1
        return (
          <span className={`font-mono text-sm ${val === best ? 'text-primary font-bold' : ''}`}>
            {val.toFixed(3)}
          </span>
        )
      },
    }),
    columnHelper.accessor('mae', {
      header: 'MAE ↓',
      cell: (info) => {
        const val = info.getValue()
        if (val == null) return <span className="text-text-muted text-sm">—</span>
        const validValues = currentMetrics.map(m => m.mae).filter(m => m != null) as number[]
        const best = validValues.length ? Math.min(...validValues) : -1
        return (
          <span className={`font-mono text-sm ${val === best ? 'text-primary font-bold' : ''}`}>
            {val.toFixed(3)}
          </span>
        )
      },
    }),
    columnHelper.accessor('r2', {
      header: 'R² ↑',
      cell: (info) => {
        const val = info.getValue() ?? 0
        const validValues = currentMetrics.map(m => m.r2).filter(m => m != null) as number[]
        const best = validValues.length ? Math.max(...validValues) : -1
        return (
          <span className={`font-mono text-sm ${val === best && val !== 0 ? 'text-primary font-bold' : ''}`}>
            {val.toFixed(3)}
          </span>
        )
      },
    }),
    columnHelper.accessor('nse', {
      header: 'NSE ↑',
      cell: (info) => {
        const val = info.getValue() ?? 0
        const validValues = currentMetrics.map(m => m.nse).filter(m => m != null) as number[]
        const best = validValues.length ? Math.max(...validValues) : -1
        return (
          <span className={`font-mono text-sm ${val === best && val !== 0 ? 'text-primary font-bold' : ''}`}>
            {val.toFixed(3)}
          </span>
        )
      },
    }),
    columnHelper.accessor('accuracy', {
      header: 'Accuracy',
      cell: (info) => {
        const val = info.getValue() ?? 0
        const validValues = currentMetrics.map(m => m.accuracy).filter(m => m != null) as number[]
        const best = validValues.length ? Math.max(...validValues) : -1
        return (
          <span className={`font-mono text-sm ${val === best && val !== 0 ? 'text-primary font-bold' : ''}`}>
            {(val * 100).toFixed(1)}%
          </span>
        )
      },
    }),
    columnHelper.accessor('f1', {
      header: 'F1',
      cell: (info) => {
        const val = info.getValue() ?? 0
        const validValues = currentMetrics.map(m => m.f1).filter(m => m != null) as number[]
        const best = validValues.length ? Math.max(...validValues) : -1
        return (
          <span className={`font-mono text-sm ${val === best && val !== 0 ? 'text-primary font-bold' : ''}`}>
            {(val * 100).toFixed(1)}%
          </span>
        )
      },
    }),
  ], [currentMetrics, bestModel])

  // eslint-disable-next-line react-hooks/incompatible-library
  const table = useReactTable({
    data: currentMetrics,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  })

  // Summary section — recommended model per module
  const recommendations = useMemo(() => {
    return tabs.map((tab) => {
      const data = metricsMap[tab]
      if (!data?.length) return { tab, model: 'N/A', confidence: 0 }
      return {
        tab,
        model: getBest(data, 'rmse', true),
        confidence: getConfidence(data),
      }
    })
  }, [metricsMap])

  return (
    <ErrorBoundary>
      <div className="space-y-6">
        {/* Tab bar */}
        <div className="flex gap-1 p-1 bg-gray-100 dark:bg-gray-800 rounded-xl w-fit">
          {tabs.map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-5 py-2 rounded-lg text-sm font-medium transition-all duration-200
                ${activeTab === tab
                  ? 'bg-white dark:bg-surface-dark text-primary shadow-sm'
                  : 'text-text-muted dark:text-text-dark-muted hover:text-text-primary dark:hover:text-white'}`}
            >
              {tab}
            </button>
          ))}
        </div>

        {/* Best model badge */}
        {bestModel && (
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="inline-flex items-center gap-2 px-4 py-2 bg-primary/10 dark:bg-primary/20 rounded-xl"
          >
            <Award className="w-5 h-5 text-primary" />
            <span className="text-sm font-semibold text-primary">
              Best {activeTab} Model: {bestModel}
            </span>
            <span className="text-xs text-text-muted dark:text-text-dark-muted">(lowest RMSE)</span>
          </motion.div>
        )}

        {/* Metrics table */}
        <motion.div
          key={`table-${activeTab}`}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="card p-5"
        >
          <h3 className="text-sm font-semibold text-text-primary dark:text-white mb-4">
            {activeTab} Model Metrics
          </h3>
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

        {/* Charts row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <MetricsBarChart data={currentMetrics} metric="rmse" title={`${activeTab} — RMSE Comparison`} />
          <ModelComparisonRadar data={currentMetrics} title={`${activeTab} — Multi-Metric Radar`} />
        </div>

        {/* Summary — Recommended models */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <h3 className="text-sm font-semibold text-text-primary dark:text-white mb-4">
            Recommended Models Summary
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {recommendations.map((rec) => (
              <div key={rec.tab} className="card p-5 border-l-4 border-primary">
                <p className="text-xs text-text-muted dark:text-text-dark-muted font-medium uppercase tracking-wider">
                  {rec.tab}
                </p>
                <div className="flex items-center gap-2 mt-2">
                  <Trophy className="w-5 h-5 text-warning" />
                  <span className="text-lg font-bold text-text-primary dark:text-white">{rec.model}</span>
                </div>
                <div className="mt-3">
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-text-muted dark:text-text-dark-muted">Confidence</span>
                    <span className="font-mono font-medium text-text-primary dark:text-white">{rec.confidence}%</span>
                  </div>
                  <div className="h-1.5 bg-gray-100 dark:bg-gray-800 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full bg-primary rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${rec.confidence}%` }}
                      transition={{ duration: 0.8, delay: 0.4 }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </ErrorBoundary>
  )
}
