import axios, { AxiosError } from 'axios'

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:8000/api'
const API_KEY = (import.meta.env.VITE_API_KEY as string | undefined) ?? ''

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add API key for the backend
    if (API_KEY) {
      config.headers['X-API-Key'] = API_KEY
    }
    return config
  },
  (error) => Promise.reject(error)
)

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    type ErrorBody = { error?: string; detail?: string; message?: string }
    const data = error.response?.data as ErrorBody | undefined
    const message = data?.error || data?.detail || data?.message || error.message || 'An error occurred'
    console.error('[API Error]', message)
    return Promise.reject(new Error(message))
  }
)

export default api
