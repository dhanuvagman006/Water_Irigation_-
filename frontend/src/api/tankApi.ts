import { useQuery, useMutation } from '@tanstack/react-query'
import api from './axiosConfig'
import type { TankPrediction, ModelMetrics, TankInput } from '../types'
import { normalizeModelName } from '../utils/formatters'

export function useTankPrediction() {
  return useMutation<TankPrediction[], Error, TankInput>({
    mutationFn: async (input) => {
      const { data } = await api.post('/tank/predict', input)
      const predictions = (data.predictions as Array<Record<string, unknown>> | undefined) ?? []
      return predictions.map((p) => ({
        date: String(p.date),
        level: p.level as TankPrediction['level'],
        percentage: Number(p.percentage),
        model: input.model,
      }))
    },
  })
}

export function useTankMetrics() {
  return useQuery<ModelMetrics[]>({
    queryKey: ['tankMetrics'],
    queryFn: async () => {
      const { data } = await api.get('/tank/metrics')
      return data.map((item: Record<string, unknown>) => ({
        ...item,
        model: normalizeModelName(String(item.model_name))
      })) as ModelMetrics[]
    },
    staleTime: 10 * 60 * 1000,
  })
}
