import { useQuery, useMutation } from '@tanstack/react-query'
import api from './axiosConfig'
import type { IrrigationPlan, ModelMetrics, IrrigationInput } from '../types'
import { normalizeModelName } from '../utils/formatters'

export function useIrrigationPrediction() {
  return useMutation<IrrigationPlan[], Error, IrrigationInput>({
    mutationFn: async (input) => {
      const payload = {
        ...input,
        growth_stages: Object.fromEntries((input.crop_types || []).map((c) => [c, 'Vegetative'])),
        num_plants: Object.fromEntries((input.crop_types || []).map((c) => [c, 50])),
      }
      const { data } = await api.post('/irrigation/predict', payload)
      return data.plan.map((p: any) => ({
        ...p,
        soil_moisture: p.soil_moisture_forecast
      })) as IrrigationPlan[]
    },
  })
}

export function useIrrigationMetrics() {
  return useQuery<ModelMetrics[]>({
    queryKey: ['irrigationMetrics'],
    queryFn: async () => {
      const { data } = await api.get('/irrigation/metrics')
      return data.map((item: Record<string, unknown>) => ({
        ...item,
        model: normalizeModelName(String(item.model_name))
      })) as ModelMetrics[]
    },
    staleTime: 10 * 60 * 1000,
  })
}
