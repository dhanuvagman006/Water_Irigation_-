import { motion } from 'framer-motion'
import { Droplets } from 'lucide-react'
import type { CropType, IrrigationDecision } from '../../types'
import { DECISION_COLORS } from '../../utils/formatters'

interface CropStatusCardProps {
  crop: CropType
  decision: IrrigationDecision
  waterLiters: number
}

const cropEmoji: Record<CropType, string> = {
  Arecanut: '🌴',
  Coconut: '🥥',
  Pepper: '🌶️',
}

export default function CropStatusCard({ crop, decision, waterLiters }: CropStatusCardProps) {
  const borderColor = DECISION_COLORS[decision]

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="card p-5"
      style={{ borderLeft: `4px solid ${borderColor}` }}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-2xl">{cropEmoji[crop]}</span>
          <h4 className="font-semibold text-text-primary dark:text-white">{crop}</h4>
        </div>
        <span
          className="badge text-xs font-semibold"
          style={{
            backgroundColor: `${borderColor}15`,
            color: borderColor,
          }}
        >
          {decision}
        </span>
      </div>
      <div className="flex items-center gap-2 text-text-muted dark:text-text-dark-muted">
        <Droplets className="w-4 h-4" />
        <span className="text-sm font-mono">{waterLiters}L today</span>
      </div>
    </motion.div>
  )
}
