import { NavLink, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  LayoutDashboard,
  CloudRain,
  Database,
  Sprout,
  BarChart2,
  Settings,
  Droplets,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react'
import useAppStore from '../../store/useAppStore'

const navItems = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/rainfall', label: 'Rainfall', icon: CloudRain },
  { path: '/tank', label: 'Tank Storage', icon: Database },
  { path: '/irrigation', label: 'Irrigation', icon: Sprout },
  { path: '/models', label: 'Model Comparison', icon: BarChart2 },
]

export default function Sidebar() {
  const { sidebarCollapsed, toggleSidebar } = useAppStore()
  const location = useLocation()

  return (
    <motion.aside
      initial={false}
      animate={{ width: sidebarCollapsed ? 60 : 240 }}
      transition={{ duration: 0.2, ease: 'easeInOut' }}
      className="fixed left-0 top-0 h-screen bg-white dark:bg-[#141821] border-r border-gray-100 dark:border-gray-800 z-40 flex flex-col overflow-hidden"
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 h-16 border-b border-gray-100 dark:border-gray-800 shrink-0">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary to-secondary flex items-center justify-center shrink-0">
          <Droplets className="w-5 h-5 text-white" />
        </div>
        <AnimatePresence>
          {!sidebarCollapsed && (
            <motion.span
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -10 }}
              className="text-lg font-semibold text-gradient whitespace-nowrap"
            >
              AquaAI
            </motion.span>
          )}
        </AnimatePresence>
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-4 px-2 space-y-1 overflow-y-auto custom-scrollbar">
        {navItems.map((item) => {
          const isActive =
            location.pathname === item.path ||
            (item.path !== '/' && location.pathname.startsWith(item.path))
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={`
                flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200
                ${isActive 
                  ? 'bg-primary/10 text-primary dark:text-secondary border-l-[3px] border-primary' 
                  : 'text-text-muted hover:bg-gray-50 dark:hover:bg-gray-800/50 hover:text-text-primary dark:hover:text-text-dark border-l-[3px] border-transparent'}
              `}
            >
              <item.icon className={`w-5 h-5 shrink-0 ${isActive ? 'text-primary dark:text-secondary' : ''}`} />
              <AnimatePresence>
                {!sidebarCollapsed && (
                  <motion.span
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="whitespace-nowrap"
                  >
                    {item.label}
                  </motion.span>
                )}
              </AnimatePresence>
            </NavLink>
          )
        })}
      </nav>

      {/* Bottom section */}
      <div className="border-t border-gray-100 dark:border-gray-800 p-3 space-y-2 shrink-0">
        <div className="flex items-center gap-3 px-2 py-2 text-text-muted dark:text-text-dark-muted">
          <Settings className="w-4 h-4 shrink-0" />
          <AnimatePresence>
            {!sidebarCollapsed && (
              <motion.span
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="text-xs whitespace-nowrap"
              >
                Mangaluru, Karnataka
              </motion.span>
            )}
          </AnimatePresence>
        </div>

        <button
          onClick={toggleSidebar}
          className="w-full flex items-center justify-center p-2 rounded-lg text-text-muted hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
        >
          {sidebarCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>
    </motion.aside>
  )
}
