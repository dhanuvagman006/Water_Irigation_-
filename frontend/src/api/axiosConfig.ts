import axios, { AxiosError } from 'axios'
import toast from 'react-hot-toast'

const RAW_API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? ''

function resolveApiBaseUrl(): string {
  if (!RAW_API_BASE_URL) return '/api'

  if (typeof window !== 'undefined') {
    const host = window.location.hostname
    const isLocalHost = host === 'localhost' || host === '127.0.0.1'
    const isLocalBase = RAW_API_BASE_URL.startsWith('http://localhost:8001') ||
      RAW_API_BASE_URL.startsWith('http://127.0.0.1:8001')

    if (!isLocalHost && isLocalBase) {
      return '/api'
    }
  }

  return RAW_API_BASE_URL
}

const API_BASE_URL = resolveApiBaseUrl()
const API_KEY = (import.meta.env.VITE_API_KEY as string | undefined) ?? ''

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
  },
})

api.interceptors.request.use(
  (config) => {
    if (API_KEY) {
      config.headers['X-API-Key'] = API_KEY
    }
    return config
  },
  (error) => Promise.reject(error)
)

api.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    type ErrorBody = { error?: string; detail?: string; message?: string }
    const data = error.response?.data as ErrorBody | undefined
    const message = data?.error || data?.detail || data?.message || error.message || 'An error occurred'
    console.error('[API Error]', message)

    if (error.response?.status === 403) {
      toast.error('Backend connection failed: Invalid API key or server not running')
    } else if (error.response?.status === 422) {
      toast.error('Server needs more weather data — check backend logs')
    } else if (error.response?.status >= 500) {
      toast.error('Server error — please check the backend')
    } else if (error.code === 'ECONNABORTED') {
      toast.error('Backend is not responding — is it running?')
    } else if (error.code === 'ERR_NETWORK') {
      toast.error('Cannot connect to backend at ' + API_BASE_URL)
    }

    return Promise.reject(new Error(message))
  }
)

export default api
