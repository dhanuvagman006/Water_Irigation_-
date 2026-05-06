import { useQuery } from '@tanstack/react-query'
import api from './axiosConfig'
import type { RainfallPrediction, ModelMetrics, ModelName, RainfallSummary } from '../types'

import { normalizeModelName } from '../utils/formatters'

type RainfallMetricsApiRecord = {
  model_name: string
  rmse: number | null
  mae: number | null
  r2: number | null
  nse: number | null
  accuracy?: number | null
  f1?: number | null
}

function formatLocalDateKey(date: Date): string {
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

function daysToHorizon(days: number): string {
  if (days <= 1) return 'short'
  if (days <= 7) return 'medium'
  return 'long'
}

export function useRainfallPrediction(
  model: ModelName,
  startDate?: Date,
  days: number = 14,
  options?: { enabled?: boolean }
) {
  return useQuery<RainfallPrediction[]>({
    queryKey: ['rainfall', model, startDate ? formatLocalDateKey(startDate) : null, days],
    queryFn: async () => {
      const payload: Record<string, string | number> = {
        model,
        days,
        horizon: daysToHorizon(days),
      }
      if (startDate) {
        payload.start_date = formatLocalDateKey(startDate)
      }

      const { data } = await api.post('/rainfall/predict', payload)
      return data.predictions
    },
    enabled: options?.enabled ?? true,
    staleTime: 5 * 60 * 1000,
  })
}

export function useRainfallMetrics() {
  return useQuery<ModelMetrics[]>({
    queryKey: ['rainfallMetrics'],
    queryFn: async () => {
      const { data } = await api.get('/rainfall/metrics')
      return (data as RainfallMetricsApiRecord[]).map((item) => ({
        model: normalizeModelName(item.model_name),
        rmse: item.rmse ?? null,
        mae: item.mae ?? null,
        r2: item.r2 ?? null,
        nse: item.nse ?? null,
        accuracy: item.accuracy ?? undefined,
        f1: item.f1 ?? undefined,
      }))
    },
    staleTime: 10 * 60 * 1000,
  })
}

export function useRainfallSummary() {
  return useQuery<RainfallSummary>({
    queryKey: ['rainfall-summary'],
    queryFn: async () => {
      const { data } = await api.get('/rainfall/summary')
      return data
    },
    staleTime: 10 * 60 * 1000,
  })
}
