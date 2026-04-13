interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg'
}

export function LoadingSpinner({ size = 'md' }: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: 'w-4 h-4 border-2',
    md: 'w-8 h-8 border-3',
    lg: 'w-12 h-12 border-4',
  }

  return (
    <div className="flex items-center justify-center p-4">
      <div
        className={`${sizeClasses[size]} border-gray-200 dark:border-gray-700 border-t-primary rounded-full animate-spin`}
      />
    </div>
  )
}

export function ChartSkeleton() {
  return (
    <div className="card p-5 space-y-4">
      <div className="flex justify-between items-center">
        <div className="skeleton h-5 w-32" />
        <div className="skeleton h-5 w-20" />
      </div>
      <div className="skeleton h-64 w-full rounded-lg" />
    </div>
  )
}

export function TableSkeleton({ rows = 5 }: { rows?: number }) {
  return (
    <div className="card p-5 space-y-3">
      <div className="skeleton h-5 w-40 mb-4" />
      <div className="skeleton h-10 w-full rounded" />
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="skeleton h-8 w-full rounded" />
      ))}
    </div>
  )
}

export function StatCardSkeleton() {
  return (
    <div className="card p-5">
      <div className="flex items-start justify-between">
        <div className="space-y-3 flex-1">
          <div className="skeleton h-4 w-24" />
          <div className="skeleton h-8 w-20" />
          <div className="skeleton h-3 w-16" />
        </div>
        <div className="skeleton h-10 w-10 rounded-xl" />
      </div>
    </div>
  )
}
