import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import type { LucideIcon } from 'lucide-react'

interface StatCardProps {
  title: string
  value: number | string
  suffix?: string
  icon: LucideIcon
  trend?: { value: number; label: string }
  color?: 'primary' | 'accent' | 'success' | 'warning' | 'danger'
  loading?: boolean
}

const colorMap = {
  primary: 'from-primary/10 to-primary/5 text-primary dark:from-primary/20 dark:to-primary/10',
  accent: 'from-accent/10 to-accent/5 text-accent dark:from-accent/20 dark:to-accent/10',
  success: 'from-success/10 to-success/5 text-success dark:from-success/20 dark:to-success/10',
  warning: 'from-warning/10 to-warning/5 text-warning dark:from-warning/20 dark:to-warning/10',
  danger: 'from-danger/10 to-danger/5 text-danger dark:from-danger/20 dark:to-danger/10',
}

export default function StatCard({ title, value, suffix, icon: Icon, trend, color = 'primary', loading }: StatCardProps) {
  const [displayValue, setDisplayValue] = useState(0)
  const numericValue = typeof value === 'number' ? value : parseFloat(value) || 0

  useEffect(() => {
    if (loading || typeof value !== 'number') return
    const duration = 800
    const startTime = Date.now()
    const animate = () => {
      const elapsed = Date.now() - startTime
      const progress = Math.min(elapsed / duration, 1)
      const eased = 1 - Math.pow(1 - progress, 3)
      setDisplayValue(Math.round(numericValue * eased * 10) / 10)
      if (progress < 1) requestAnimationFrame(animate)
    }
    animate()
  }, [numericValue, loading, value])

  if (loading) {
    return (
      <div className="card p-5">
        <div className="flex items-start justify-between">
          <div className="space-y-3 flex-1">
            <div className="skeleton h-4 w-24" />
            <div className="skeleton h-8 w-20" />
          </div>
          <div className="skeleton h-10 w-10 rounded-xl" />
        </div>
      </div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="card-hover p-5"
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-text-muted dark:text-text-dark-muted font-medium">{title}</p>
          <div className="flex items-baseline gap-1.5 mt-1">
            <span className="text-2xl font-semibold text-text-primary dark:text-white font-mono">
              {typeof value === 'number' ? displayValue : value}
            </span>
            {suffix && (
              <span className="text-sm text-text-muted dark:text-text-dark-muted">{suffix}</span>
            )}
          </div>
          {trend && (
            <p className={`text-xs mt-2 ${trend.value >= 0 ? 'text-success' : 'text-danger'}`}>
              {trend.value >= 0 ? '↑' : '↓'} {Math.abs(trend.value)}% {trend.label}
            </p>
          )}
        </div>
        <div className={`p-2.5 rounded-xl bg-gradient-to-br ${colorMap[color]}`}>
          <Icon className="w-5 h-5" />
        </div>
      </div>
    </motion.div>
  )
}
