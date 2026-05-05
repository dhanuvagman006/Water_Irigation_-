import { useMutation } from '@tanstack/react-query'
import api from './axiosConfig'
import type { IrrigationPlan, IrrigationInput } from '../types'

export function useIrrigationPrediction() {
  return useMutation<IrrigationPlan[], Error, IrrigationInput>({
    mutationFn: async (input) => {
      const payload = {
        ...input,
        growth_stages: Object.fromEntries((input.crop_types || []).map((c) => [c, 'Vegetative'])),
        num_plants: Object.fromEntries((input.crop_types || []).map((c) => [c, input.plants_per_crop])),
      }
      const { data } = await api.post('/irrigation/predict', payload)
      const plan = (data.plan as Array<Record<string, unknown>> | undefined) ?? []
      return plan.map((p) => ({
        date: String(p.date),
        crop: p.crop as IrrigationPlan['crop'],
        decision: p.decision as IrrigationPlan['decision'],
        water_liters: Number(p.water_liters),
        reason: String(p.reason ?? ''),
        soil_moisture: Number(p.soil_moisture_forecast ?? 0),
      }))
    },
  })
}
