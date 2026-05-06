import { useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { Trophy, Award } from 'lucide-react'
import {
  useReactTable, getCoreRowModel, getSortedRowModel, flexRender, createColumnHelper,
  type SortingState,
} from '@tanstack/react-table'
import MetricsBarChart from '../components/charts/MetricsBarChart'
import ModelComparisonRadar from '../components/charts/ModelComparisonRadar'
import ErrorBoundary from '../components/shared/ErrorBoundary'
import { useRainfallMetrics, useRainfallSummary } from '../api/rainfallApi'
import { MODEL_COLORS } from '../utils/formatters'
import type { ModelMetrics, ModelName } from '../types'

const columnHelper = createColumnHelper<ModelMetrics>()

export default function ModelComparisonPage() {
  const [sorting, setSorting] = useState<SortingState>([])

  const { data: rainfallMetrics } = useRainfallMetrics()
  const { data: summary } = useRainfallSummary()

  const currentMetrics = useMemo(() => {
    const metrics = rainfallMetrics ?? []
    const validModels: ModelName[] = ['LSTM', 'GRU', 'BiLSTM', 'CNN-LSTM', 'WLSTM', 'SimpleRNN']
    
    // Deduplicate and filter, keeping the most recent record (backend ordered desc)
    const latestMetricsMap = new Map<ModelName, ModelMetrics>()
    metrics.forEach((m) => {
      if (validModels.includes(m.model)) {
        if (!latestMetricsMap.has(m.model)) {
          latestMetricsMap.set(m.model, m)
        }
      }
    })

    return Array.from(latestMetricsMap.values())
  }, [rainfallMetrics])

  const bestModel = summary?.best_model ?? null

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
        return (
          <span className="font-mono text-sm">{val.toFixed(3)}</span>
        )
      },
    }),
    columnHelper.accessor('mae', {
      header: 'MAE ↓',
      cell: (info) => {
        const val = info.getValue()
        if (val == null) return <span className="text-text-muted text-sm">—</span>
        return (
          <span className="font-mono text-sm">{val.toFixed(3)}</span>
        )
      },
    }),
    columnHelper.accessor('r2', {
      header: 'R² ↑',
      cell: (info) => {
        const val = info.getValue() ?? 0
        return (
          <span className="font-mono text-sm">{val.toFixed(3)}</span>
        )
      },
    }),
    columnHelper.accessor('nse', {
      header: 'NSE ↑',
      cell: (info) => {
        const val = info.getValue() ?? 0
        return (
          <span className="font-mono text-sm">{val.toFixed(3)}</span>
        )
      },
    }),
    columnHelper.accessor('f1', {
      header: 'F1',
      cell: (info) => {
        const val = info.getValue() ?? 0
        return (
          <span className="font-mono text-sm">{(val * 100).toFixed(1)}%</span>
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

  const recommendations = summary?.recommendations ?? []

  return (
    <ErrorBoundary>
      <div className="space-y-6">
        {/* Best model badge */}
        {bestModel && (
          <motion.div
            key="rainfall-best"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="inline-flex items-center gap-2 px-4 py-2 bg-primary/10 dark:bg-primary/20 rounded-xl"
          >
            <Award className="w-5 h-5 text-primary" />
            <span className="text-sm font-semibold text-primary">
              Best Rainfall Model: {bestModel}
            </span>
            <span className="text-xs text-text-muted dark:text-text-dark-muted">(lowest RMSE)</span>
          </motion.div>
        )}

        {/* Metrics table */}
        <motion.div
          key="table-rainfall"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="card p-5"
        >
          <h3 className="text-sm font-semibold text-text-primary dark:text-white mb-4">
            Rainfall Model Metrics
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
          <MetricsBarChart data={currentMetrics} metric="rmse" title="Rainfall — RMSE Comparison" />
          <ModelComparisonRadar data={currentMetrics} title="Rainfall — Multi-Metric Radar" />
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
                    <span className="text-text-muted dark:text-text-dark-muted">Reliability Score</span>
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
