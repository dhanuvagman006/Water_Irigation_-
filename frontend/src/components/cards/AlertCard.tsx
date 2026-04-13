import { motion } from 'framer-motion'
import { AlertTriangle, Info, CheckCircle } from 'lucide-react'

interface AlertCardProps {
  type: 'danger' | 'warning' | 'info' | 'success'
  title: string
  message: string
  pulse?: boolean
}

const config = {
  danger: {
    icon: AlertTriangle,
    bg: 'bg-danger/5 border-danger/20 dark:bg-danger/10',
    text: 'text-danger',
    iconBg: 'bg-danger/10',
  },
  warning: {
    icon: AlertTriangle,
    bg: 'bg-warning/5 border-warning/20 dark:bg-warning/10',
    text: 'text-warning',
    iconBg: 'bg-warning/10',
  },
  info: {
    icon: Info,
    bg: 'bg-accent/5 border-accent/20 dark:bg-accent/10',
    text: 'text-accent',
    iconBg: 'bg-accent/10',
  },
  success: {
    icon: CheckCircle,
    bg: 'bg-success/5 border-success/20 dark:bg-success/10',
    text: 'text-success',
    iconBg: 'bg-success/10',
  },
}

export default function AlertCard({ type, title, message, pulse }: AlertCardProps) {
  const c = config[type]
  const Icon = c.icon

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      className={`rounded-xl border p-4 ${c.bg} ${pulse ? 'animate-pulse-slow' : ''}`}
    >
      <div className="flex gap-3">
        <div className={`p-2 rounded-lg ${c.iconBg} shrink-0 h-fit`}>
          <Icon className={`w-4 h-4 ${c.text}`} />
        </div>
        <div>
          <h4 className={`text-sm font-semibold ${c.text}`}>{title}</h4>
          <p className="text-xs text-text-muted dark:text-text-dark-muted mt-1">{message}</p>
        </div>
      </div>
    </motion.div>
  )
}
