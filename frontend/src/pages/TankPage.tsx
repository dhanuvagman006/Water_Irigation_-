import { useState } from 'react'
import { motion } from 'framer-motion'
import { Play } from 'lucide-react'
import ModelSelector from '../components/shared/ModelSelector'
import TankLevelBarChart from '../components/charts/TankLevelBarChart'
import AlertCard from '../components/cards/AlertCard'
import ErrorBoundary from '../components/shared/ErrorBoundary'
import { ChartSkeleton } from '../components/shared/LoadingSpinner'
import { useTankPrediction } from '../api/tankApi'
import type { ModelName, TankPrediction } from '../types'

export default function TankPage() {
  const [model, setModel] = useState<ModelName>('LSTM')
  const [roofArea, setRoofArea] = useState(120)
  const [tankCapacity, setTankCapacity] = useState(5000)
  const [currentLevel, setCurrentLevel] = useState(75)
  const [dailyConsumption, setDailyConsumption] = useState(200)

  const mutation = useTankPrediction()
  const predictions = (mutation.data ?? []) as TankPrediction[]

  const handlePredict = () => {
    mutation.mutate({
      roof_area: roofArea,
      tank_capacity: tankCapacity,
      current_level: currentLevel,
      daily_consumption: dailyConsumption,
      model,
    })
  }

  // Compute alerts
  const lowDay = predictions.findIndex((p) => p.level === 'Low')
  const daysRemaining = predictions.filter((p) => p.level !== 'Low').length
  const avgRainCollection = Math.round(roofArea * 15 * 0.8) // rough estimate

  return (
    <ErrorBoundary>
      <div className="grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-6">
        {/* Input panel */}
        <motion.div
          initial={{ opacity: 0, x: -12 }}
          animate={{ opacity: 1, x: 0 }}
          className="card p-5 space-y-5 lg:sticky lg:top-24 h-fit"
        >
          <h3 className="text-sm font-semibold text-text-primary dark:text-white">Tank Parameters</h3>

          {/* Roof area */}
          <div>
            <div className="flex justify-between text-xs mb-1.5">
              <span className="text-text-muted dark:text-text-dark-muted font-medium">Roof Area</span>
              <span className="font-mono font-semibold text-text-primary dark:text-white">{roofArea} m²</span>
            </div>
            <input
              type="range" min="10" max="500" value={roofArea}
              onChange={(e) => setRoofArea(Number(e.target.value))}
              className="w-full accent-primary h-1.5"
            />
          </div>

          {/* Tank capacity */}
          <div>
            <label className="text-xs text-text-muted dark:text-text-dark-muted font-medium block mb-1.5">
              Tank Capacity (L)
            </label>
            <input
              type="number" value={tankCapacity}
              onChange={(e) => setTankCapacity(Number(e.target.value))}
              className="input"
            />
          </div>

          {/* Current level */}
          <div>
            <div className="flex justify-between text-xs mb-1.5">
              <span className="text-text-muted dark:text-text-dark-muted font-medium">Current Level</span>
              <span className="font-mono font-semibold text-text-primary dark:text-white">{currentLevel}%</span>
            </div>
            <input
              type="range" min="0" max="100" value={currentLevel}
              onChange={(e) => setCurrentLevel(Number(e.target.value))}
              className="w-full accent-primary h-1.5"
            />
            {/* Visual gauge */}
            <div className="mt-3 h-24 w-full bg-gray-100 dark:bg-gray-800 rounded-xl overflow-hidden relative">
              <motion.div
                className="absolute bottom-0 w-full rounded-b-xl"
                initial={{ height: 0 }}
                animate={{ height: `${currentLevel}%` }}
                transition={{ duration: 0.6, ease: 'easeOut' }}
                style={{
                  background: currentLevel > 70
                    ? 'linear-gradient(to top, #3B6D11, #1D9E75)'
                    : currentLevel > 35
                    ? 'linear-gradient(to top, #BA7517, #FCD34D)'
                    : 'linear-gradient(to top, #E24B4A, #F87171)',
                }}
              />
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-lg font-bold font-mono text-white drop-shadow-md">
                  {currentLevel}%
                </span>
              </div>
            </div>
          </div>

          {/* Daily consumption */}
          <div>
            <label className="text-xs text-text-muted dark:text-text-dark-muted font-medium block mb-1.5">
              Daily Consumption (L/day)
            </label>
            <input
              type="number" value={dailyConsumption}
              onChange={(e) => setDailyConsumption(Number(e.target.value))}
              className="input"
            />
          </div>

          {/* Model */}
          <div>
            <label className="text-xs text-text-muted dark:text-text-dark-muted font-medium block mb-1.5">Model</label>
            <ModelSelector value={model} onChange={setModel} />
          </div>

          <button onClick={handlePredict} className="btn-primary w-full" disabled={mutation.isPending}>
            <Play className="w-4 h-4" />
            {mutation.isPending ? 'Predicting...' : 'Predict'}
          </button>
        </motion.div>

        {/* Results */}
        <div className="space-y-6">
          {/* Large Gauge */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            className="card p-6"
          >
            <h3 className="text-sm font-semibold text-text-primary dark:text-white mb-4">Current + Predicted Fill</h3>
            <div className="flex items-center gap-6">
              <div className="flex-1">
                <div className="h-6 bg-gray-100 dark:bg-gray-800 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${currentLevel}%` }}
                    transition={{ duration: 0.8, ease: 'easeOut' }}
                    style={{
                      background: currentLevel > 70
                        ? 'linear-gradient(to right, #0F6E56, #1D9E75)'
                        : currentLevel > 35
                        ? 'linear-gradient(to right, #BA7517, #FCD34D)'
                        : 'linear-gradient(to right, #E24B4A, #F87171)',
                    }}
                  />
                </div>
                <div className="flex justify-between mt-2">
                  <span className="text-xs text-text-muted dark:text-text-dark-muted">0%</span>
                  <span className="text-sm font-mono font-bold text-text-primary dark:text-white">{currentLevel}%</span>
                  <span className="text-xs text-text-muted dark:text-text-dark-muted">100%</span>
                </div>
              </div>
              <div className="text-center px-4 border-l border-gray-100 dark:border-gray-800">
                <p className="text-xs text-text-muted dark:text-text-dark-muted">Capacity</p>
                <p className="text-lg font-mono font-bold text-text-primary dark:text-white">{tankCapacity}L</p>
                <p className="text-xs text-text-muted dark:text-text-dark-muted mt-1">
                  Available: {Math.round(tankCapacity * currentLevel / 100)}L
                </p>
              </div>
            </div>
          </motion.div>

          {/* Bar chart */}
          {mutation.isPending ? <ChartSkeleton /> : <TankLevelBarChart data={predictions} />}

          {/* Alert cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {lowDay >= 0 && (
              <AlertCard
                type="danger"
                title="Low Tank Warning"
                message={`Tank will reach LOW on Day ${lowDay + 1}`}
                pulse
              />
            )}
            <AlertCard
              type="info"
              title="Water Remaining"
              message={`Estimated ${daysRemaining} days of water remaining at current consumption`}
            />
            <AlertCard
              type="success"
              title="Rain Collection"
              message={`Collect ~${avgRainCollection}L from next rain event`}
            />
          </div>
        </div>
      </div>
    </ErrorBoundary>
  )
}
