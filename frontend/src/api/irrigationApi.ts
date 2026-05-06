import { useMutation } from '@tanstack/react-query'
import api from './axiosConfig'
import type { IrrigationPlan, IrrigationInput } from '../types'

type IrrigationPlanApiRecord = {
  date: string
  crop: string
  decision: string
  water_liters: number
  reason?: string
  soil_moisture_forecast?: number
}

function isCropType(value: string): value is IrrigationPlan['crop'] {
  return value === 'Arecanut' || value === 'Coconut' || value === 'Pepper'
}

function isDecision(value: string): value is IrrigationPlan['decision'] {
  return value === 'Irrigate' || value === 'No Irrigate' || value === 'Monitor'
}

function isIrrigationPlanRecord(
  value: IrrigationPlanApiRecord
): value is IrrigationPlanApiRecord & { crop: IrrigationPlan['crop']; decision: IrrigationPlan['decision'] } {
  return isCropType(value.crop) && isDecision(value.decision)
}

export function useIrrigationPrediction() {
  return useMutation<IrrigationPlan[], Error, IrrigationInput>({
    mutationFn: async (input) => {
      const payload = {
        ...input,
        growth_stages: Object.fromEntries((input.crop_types || []).map((c) => [c, 'Vegetative'])),
        num_plants: Object.fromEntries((input.crop_types || []).map((c) => [c, input.plants_per_crop])),
      }
      const { data } = await api.post('/irrigation/predict', payload)
      const plan = (data.plan as IrrigationPlanApiRecord[] | undefined) ?? []
      return plan
        .filter(isIrrigationPlanRecord)
        .map((p) => ({
          date: p.date,
          crop: p.crop,
          decision: p.decision,
          water_liters: Number(p.water_liters),
          reason: p.reason ?? '',
          soil_moisture: Number(p.soil_moisture_forecast ?? 0),
        }))
    },
  })
}
