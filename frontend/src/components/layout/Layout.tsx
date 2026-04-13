import { Outlet } from 'react-router-dom'
import { motion } from 'framer-motion'
import Sidebar from './Sidebar'
import Topbar from './Topbar'
import useAppStore from '../../store/useAppStore'

export default function Layout() {
  const { sidebarCollapsed } = useAppStore()

  return (
    <div className="min-h-screen bg-background dark:bg-background-dark">
      <Sidebar />

      {/* Main content area */}
      <div
        className="transition-all duration-200"
        style={{ marginLeft: sidebarCollapsed ? 60 : 240 }}
      >
        <Topbar />

        {/* Mobile warning */}
        <div className="md:hidden bg-warning/10 text-warning text-xs text-center py-2 px-4">
          Best viewed on desktop (1280px+)
        </div>

        <main className="p-6">
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2, ease: 'easeOut' }}
          >
            <Outlet />
          </motion.div>
        </main>
      </div>
    </div>
  )
}
