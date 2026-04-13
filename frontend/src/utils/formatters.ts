import { format, parseISO } from 'date-fns'

export function formatDate(dateStr: string, fmt: string = 'MMM dd'): string {
  try {
    return format(parseISO(dateStr), fmt)
  } catch {
    return dateStr
  }
}

export function formatNumber(num: number, decimals: number = 1): string {
  return num.toFixed(decimals)
}

export function formatPercentage(num: number): string {
  return `${Math.round(num)}%`
}

export function formatMetric(value: number, metric: string): string {
  if (metric === 'r2' || metric === 'nse') return value.toFixed(3)
  if (metric === 'accuracy' || metric === 'f1') return (value * 100).toFixed(1) + '%'
  return value.toFixed(2)
}

export function exportToCSV(data: Record<string, unknown>[], filename: string): void {
  if (!data.length) return
  const headers = Object.keys(data[0])
  const rows = data.map((row) =>
    headers.map((h) => {
      const val = row[h]
      if (typeof val === 'string' && val.includes(',')) return `"${val}"`
      return String(val ?? '')
    }).join(',')
  )
  const csv = [headers.join(','), ...rows].join('\n')
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `${filename}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

export const MODEL_COLORS: Record<string, string> = {
  LSTM: '#0F6E56',
  GRU: '#378ADD',
  BiLSTM: '#8B5CF6',
  'CNN-LSTM': '#E24B4A',
  Transformer: '#BA7517',
  StackedLSTM: '#3B6D11',
}

export function normalizeModelName(backendName: string): string {
  const map: Record<string, string> = {
    lstm: 'LSTM',
    gru: 'GRU',
    bilstm: 'BiLSTM',
    cnn_lstm: 'CNN-LSTM',
    transformer: 'Transformer',
    stacked_lstm: 'StackedLSTM'
  };
  return map[backendName.toLowerCase()] || backendName;
}

export const TANK_LEVEL_COLORS = {
  Low: '#E24B4A',
  Medium: '#BA7517',
  Full: '#3B6D11',
}

export const DECISION_COLORS = {
  Irrigate: '#3B6D11',
  'No Irrigate': '#6B7280',
  Monitor: '#BA7517',
}
