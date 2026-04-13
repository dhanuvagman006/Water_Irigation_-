import { useState, useMemo } from 'react'
import { motion } from 'framer-motion'
import { Play, Trophy } from 'lucide-react'
import {
  ComposedChart, Line, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, Legend,
} from 'recharts'
import {
  useReactTable, getCoreRowModel, flexRender, createColumnHelper, getSortedRowModel,
  type SortingState,
} from '@tanstack/react-table'
import ModelSelector from '../components/shared/ModelSelector'
import DateRangePicker from '../components/shared/DateRangePicker'
import ModelComparisonRadar from '../components/charts/ModelComparisonRadar'
import ErrorBoundary from '../components/shared/ErrorBoundary'
import { ChartSkeleton } from '../components/shared/LoadingSpinner'
import { useRainfallPrediction, useRainfallMetrics } from '../api/rainfallApi'
import { formatDate } from '../utils/formatters'
import type { ModelName, ModelMetrics } from '../types'

const columnHelper = createColumnHelper<ModelMetrics>()

interface TooltipEntry {
  dataKey?: string
  value?: number
  name?: string
  color?: string
}

interface CustomTooltipProps {
  active?: boolean
  payload?: TooltipEntry[]
  label?: string
}

function CustomTooltip({ active, payload, label }: CustomTooltipProps) {
  if (!active || !payload?.length) return null
  const predicted = payload.find((p: TooltipEntry) => p.dataKey === 'predicted_mm')
  const actual = payload.find((p: TooltipEntry) => p.dataKey === 'actual_mm')
  const diff = predicted?.value != null && actual?.value != null
    ? (predicted.value - actual.value).toFixed(1)
    : null

  return (
    <div className="bg-white dark:bg-surface-dark border border-gray-100 dark:border-gray-700 rounded-lg shadow-lg p-3 text-xs">
      <p className="font-semibold text-text-primary dark:text-white mb-1.5">
        {formatDate(label, 'EEEE, MMM dd yyyy')}
      </p>
      {predicted?.value != null && (
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-primary" />
          <span className="text-text-muted dark:text-text-dark-muted">Predicted:</span>
          <span className="font-mono font-medium">{predicted.value.toFixed(1)} mm</span>
        </div>
      )}
      {actual?.value != null && (
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-accent" />
          <span className="text-text-muted dark:text-text-dark-muted">Actual:</span>
          <span className="font-mono font-medium">{actual.value.toFixed(1)} mm</span>
        </div>
      )}
      {diff && (
        <div className="mt-1 pt-1 border-t border-gray-100 dark:border-gray-700">
          <span className="text-text-muted dark:text-text-dark-muted">Error: </span>
          <span className={`font-mono font-medium ${parseFloat(diff) > 0 ? 'text-warning' : 'text-success'}`}>
            {parseFloat(diff) > 0 ? '+' : ''}{diff} mm
          </span>
        </div>
      )}
    </div>
  )
}

export default function RainfallPage() {
  const [model, setModel] = useState<ModelName>('LSTM')
  const [startDate, setStartDate] = useState(new Date())
  const [endDate, setEndDate] = useState(new Date(Date.now() + 14 * 24 * 60 * 60 * 1000))
  const [sorting, setSorting] = useState<SortingState>([])

  const days = useMemo(() => {
    const diffTime = Math.abs(endDate.getTime() - startDate.getTime());
    return Math.ceil(diffTime / (1000 * 60 * 60 * 24));
  }, [startDate, endDate]);

  const { data: rainfall, isLoading, isFetching, refetch } = useRainfallPrediction(model, startDate, days || 1)
  const { data: metrics } = useRainfallMetrics()

  const filteredMetrics = useMemo(() => {
    if (!metrics) return []
    const validModels: ModelName[] = ['LSTM', 'GRU', 'BiLSTM', 'CNN-LSTM', 'Transformer', 'StackedLSTM']
    
    // Deduplicate and filter, keeping the most recent record
    const latestMetricsMap = new Map<ModelName, ModelMetrics>()
    metrics.forEach((m) => {
      if (validModels.includes(m.model)) {
        latestMetricsMap.set(m.model, m)
      }
    })

    return Array.from(latestMetricsMap.values())
  }, [metrics])

  const metricColumns = useMemo(() => [
    columnHelper.accessor('model', {
      header: 'Model',
      cell: (info) => <span className="font-semibold text-sm">{info.getValue()}</span>,
    }),
    columnHelper.accessor('rmse', {
      header: 'RMSE',
      cell: (info) => {
        const val = info.getValue()
        const best = filteredMetrics.length > 0 ? Math.min(...filteredMetrics.map((m) => m.rmse)) : -1
        if (val == null) return <span className="text-text-muted text-sm">—</span>
        return (
          <span className={`font-mono text-sm ${val === best ? 'text-primary font-bold' : ''}`}>
            {val.toFixed(3)}
          </span>
        )
      },
    }),
    columnHelper.accessor('mae', {
      header: 'MAE',
      cell: (info) => {
        const val = info.getValue()
        const best = filteredMetrics.length > 0 ? Math.min(...filteredMetrics.map((m) => m.mae)) : -1
        if (val == null) return <span className="text-text-muted text-sm">—</span>
        return (
          <span className={`font-mono text-sm ${val === best ? 'text-primary font-bold' : ''}`}>
            {val.toFixed(3)}
          </span>
        )
      },
    }),
    columnHelper.accessor('r2', {
      header: 'R²',
      cell: (info) => {
        const val = info.getValue()
        const best = filteredMetrics.length > 0 ? Math.max(...filteredMetrics.map((m) => m.r2)) : -1
        if (val == null) return <span className="text-text-muted text-sm">—</span>
        return (
          <span className={`font-mono text-sm ${val === best ? 'text-primary font-bold' : ''}`}>
            {val.toFixed(3)}
          </span>
        )
      },
    }),
    columnHelper.accessor('nse', {
      header: 'NSE',
      cell: (info) => {
        const val = info.getValue()
        const best = filteredMetrics.length > 0 ? Math.max(...filteredMetrics.map((m) => m.nse)) : -1
        if (val == null) return <span className="text-text-muted text-sm">—</span>
        return (
          <span className={`font-mono text-sm ${val === best ? 'text-primary font-bold' : ''}`}>
            {val.toFixed(3)}
          </span>
        )
      },
    }),
  ], [filteredMetrics])

  // eslint-disable-next-line react-hooks/incompatible-library
  const table = useReactTable({
    data: filteredMetrics,
    columns: metricColumns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  })

  return (
    <ErrorBoundary>
      <div className="space-y-6">
        {/* Controls bar */}
        <motion.div
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          className="card p-4 flex flex-wrap items-center gap-4"
        >
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium text-text-muted dark:text-text-dark-muted">Model:</label>
            <ModelSelector value={model} onChange={setModel} />
          </div>
          <DateRangePicker
            startDate={startDate}
            endDate={endDate}
            onStartChange={setStartDate}
            onEndChange={setEndDate}
          />
          <button onClick={() => refetch()} disabled={isFetching} className="btn-primary ml-auto flex items-center gap-2">
            {isFetching ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Predicting...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Run Prediction
              </>
            )}
          </button>
        </motion.div>

        {/* Main chart */}
        {(isLoading || isFetching) ? (
          <ChartSkeleton />
        ) : (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.1 }}
            className="card p-5"
          >
            <h3 className="text-sm font-semibold text-text-primary dark:text-white mb-4">
              Rainfall Prediction — {model}
            </h3>
            <ResponsiveContainer width="100%" height={350}>
              <ComposedChart data={rainfall} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" className="text-gray-100 dark:text-gray-800" stroke="currentColor" />
                <XAxis dataKey="date" tickFormatter={(d) => formatDate(d)} tick={{ fontSize: 11 }} stroke="#6B7280" />
                <YAxis tick={{ fontSize: 11 }} stroke="#6B7280" label={{ value: 'mm', angle: -90, position: 'insideLeft', style: { fontSize: 11, fill: '#6B7280' } }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <ReferenceLine y={10} stroke="#BA7517" strokeDasharray="6 3" label={{ value: '10mm threshold', position: 'right', style: { fontSize: 10, fill: '#BA7517' } }} />
                <Bar dataKey="actual_mm" name="Actual" fill="#378ADD" fillOpacity={0.4} radius={[4, 4, 0, 0]} maxBarSize={32} />
                <Line type="monotone" dataKey="predicted_mm" name="Predicted" stroke="#0F6E56" strokeWidth={2.5} dot={{ r: 4, fill: '#0F6E56', stroke: '#fff', strokeWidth: 2 }} />
              </ComposedChart>
            </ResponsiveContainer>
          </motion.div>
        )}

        {/* Metrics + Radar */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Metrics table */}
          <motion.div
            initial={{ opacity: 0, x: -12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="card p-5"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-text-primary dark:text-white">Model Metrics</h3>
              {filteredMetrics.length > 0 && (
                <div className="flex items-center gap-1.5 text-xs text-primary font-medium">
                  <Trophy className="w-3.5 h-3.5" />
                  Best values highlighted
                </div>
              )}
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
                    <tr key={row.id} className="border-b border-gray-50 dark:border-gray-800/50 hover:bg-gray-50/50 dark:hover:bg-gray-800/30">
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

          {/* Radar chart */}
          {filteredMetrics.length > 0 && (
            <motion.div
              initial={{ opacity: 0, x: 12 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
            >
              <ModelComparisonRadar data={filteredMetrics} title="Rainfall Model Comparison" />
            </motion.div>
          )}
        </div>
      </div>
    </ErrorBoundary>
  )
}
