import { useQuery } from '@tanstack/react-query'
import api from './axiosConfig'
import type { RainfallPrediction, ModelMetrics, ModelName } from '../types'

import { normalizeModelName } from '../utils/formatters'

export function useRainfallPrediction(model: ModelName, startDate?: Date, days: number = 14) {
  return useQuery<RainfallPrediction[]>({
    queryKey: ['rainfall', model, startDate?.toISOString(), days],
    queryFn: async () => {
      const payload: Record<string, string | number> = { model, days }
      if (startDate) {
        // format Date as YYYY-MM-DD
        payload.start_date = startDate.toISOString().split('T')[0]
      }

      const { data } = await api.post('/rainfall/predict', payload)
      return data.predictions
    },
    staleTime: 5 * 60 * 1000,
  })
}

export function useRainfallMetrics() {
  return useQuery<ModelMetrics[]>({
    queryKey: ['rainfallMetrics'],
    queryFn: async () => {
      const { data } = await api.get('/rainfall/metrics')
      return data.map((item: Record<string, unknown>) => ({
        ...item,
        model: normalizeModelName(String(item.model_name))
      })) as ModelMetrics[]
    },
    staleTime: 10 * 60 * 1000,
  })
}
