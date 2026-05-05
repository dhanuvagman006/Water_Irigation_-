import { useMutation } from '@tanstack/react-query'
import api from './axiosConfig'
import type { TankPrediction, TankInput } from '../types'

export function useTankPrediction() {
  return useMutation<TankPrediction[], Error, TankInput>({
    mutationFn: async (input) => {
      const { data } = await api.post('/tank/predict', input)
      const predictions = (data.predictions as Array<Record<string, unknown>> | undefined) ?? []
      return predictions.map((p) => ({
        date: String(p.date),
        level: p.level as TankPrediction['level'],
        percentage: Number(p.percentage),
      }))
    },
  })
}
