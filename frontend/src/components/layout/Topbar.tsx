import { useLocation } from 'react-router-dom'
import { MapPin, RefreshCw, Sun, Moon } from 'lucide-react'
import { format } from 'date-fns'
import useAppStore from '../../store/useAppStore'

const pageTitles: Record<string, string> = {
  '/': 'Dashboard',
  '/rainfall': 'Rainfall Prediction',
  '/tank': 'Tank Storage',
  '/irrigation': 'Irrigation Planner',
  '/models': 'Model Comparison',
}

export default function Topbar() {
  const location = useLocation()
  const { darkMode, toggleDarkMode } = useAppStore()
  const title = pageTitles[location.pathname] || 'Dashboard'

  return (
    <header className="h-16 bg-white/80 dark:bg-[#141821]/80 backdrop-blur-md border-b border-gray-100 dark:border-gray-800 flex items-center justify-between px-6 sticky top-0 z-30">
      {/* Left: Title */}
      <div>
        <h1 className="text-xl font-semibold text-text-primary dark:text-white">
          {title}
        </h1>
      </div>

      {/* Right: Info + Controls */}
      <div className="flex items-center gap-4">
        {/* Location badge */}
        <div className="hidden md:flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-primary/5 dark:bg-primary/10 text-xs font-medium text-primary dark:text-secondary">
          <MapPin className="w-3.5 h-3.5" />
          12.87°N 74.88°E
        </div>

        {/* Last updated */}
        <div className="hidden lg:flex items-center gap-2 text-xs text-text-muted dark:text-text-dark-muted">
          <span>Updated {format(new Date(), 'HH:mm:ss')}</span>
          <button
            onClick={() => window.location.reload()}
            className="p-1.5 rounded-md hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            title="Refresh data"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Dark mode toggle */}
        <button
          onClick={toggleDarkMode}
          className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors text-text-muted dark:text-text-dark-muted"
          title="Toggle dark mode"
        >
          {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
        </button>
      </div>
    </header>
  )
}
