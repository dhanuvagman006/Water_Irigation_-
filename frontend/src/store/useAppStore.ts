import { create } from 'zustand'
import type { AppState } from '../types'

const useAppStore = create<AppState>((set) => ({
  selectedModel: 'LSTM',
  setSelectedModel: (m) => set({ selectedModel: m }),

  dateRange: [new Date(), new Date(Date.now() + 14 * 24 * 60 * 60 * 1000)],
  setDateRange: (r) => set({ dateRange: r }),

  sidebarCollapsed: false,
  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),

  darkMode: localStorage.getItem('aqua_dark') === 'true',
  toggleDarkMode: () =>
    set((s) => {
      const next = !s.darkMode
      localStorage.setItem('aqua_dark', String(next))
      if (next) {
        document.documentElement.classList.add('dark')
      } else {
        document.documentElement.classList.remove('dark')
      }
      return { darkMode: next }
    }),
}))

export default useAppStore
