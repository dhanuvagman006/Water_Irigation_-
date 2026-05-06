// ============================================================
// AquaAI — Water Management System Types
// ============================================================

export type ModelName = 'LSTM' | 'GRU' | 'BiLSTM' | 'CNN-LSTM' | 'WLSTM' | 'SimpleRNN'

export type RainfallPrediction = {
  date: string            // ISO date string
  predicted_mm: number
  actual_mm?: number      // optional, if available
  model?: ModelName
  confidence_low?: number
  confidence_high?: number
}

export type ModelMetrics = {
  model: ModelName
  rmse: number | null
  mae: number | null
  r2: number | null
  nse: number | null
  f1?: number
  accuracy?: number
}

export type RainfallRecommendation = {
  tab: string
  model: ModelName
  confidence: number
}

export type RainfallSummary = {
  best_model: ModelName
  confidence: number
  rmse?: number
  nse?: number
  r2?: number
  recommendations: RainfallRecommendation[]
}

export type TankLevel = 'Low' | 'Medium' | 'Full'

export type TankPrediction = {
  date: string
  level: TankLevel
  percentage: number
}

export type CropType = 'Arecanut' | 'Coconut' | 'Pepper'

export type IrrigationDecision = 'Irrigate' | 'No Irrigate' | 'Monitor'

export type IrrigationPlan = {
  date: string
  crop: CropType
  decision: IrrigationDecision
  water_liters: number
  reason: string
  soil_moisture: number
}

export type AppState = {
  selectedModel: ModelName
  setSelectedModel: (m: ModelName) => void
  dateRange: [Date, Date]
  setDateRange: (r: [Date, Date]) => void
  sidebarCollapsed: boolean
  toggleSidebar: () => void
  darkMode: boolean
  toggleDarkMode: () => void
}

export type TankInput = {
  roof_area: number
  tank_capacity: number
  current_level: number
  daily_consumption: number
}

export type IrrigationInput = {
  soil_moisture: number
  crop_types: CropType[]
  plants_per_crop: number
}
