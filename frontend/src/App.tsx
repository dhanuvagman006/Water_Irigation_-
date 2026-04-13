import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Toaster } from 'react-hot-toast'
import Layout from './components/layout/Layout'
import Dashboard from './pages/Dashboard'
import RainfallPage from './pages/RainfallPage'
import TankPage from './pages/TankPage'
import IrrigationPage from './pages/IrrigationPage'
import ModelComparisonPage from './pages/ModelComparisonPage'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      refetchOnWindowFocus: false,
    },
  },
})

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<Dashboard />} />
            <Route path="/rainfall" element={<RainfallPage />} />
            <Route path="/tank" element={<TankPage />} />
            <Route path="/irrigation" element={<IrrigationPage />} />
            <Route path="/models" element={<ModelComparisonPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            borderRadius: '12px',
            background: '#1A1A1A',
            color: '#fff',
            fontSize: '13px',
          },
        }}
      />
    </QueryClientProvider>
  )
}
