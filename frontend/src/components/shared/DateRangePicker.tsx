import { format } from 'date-fns'

interface DateRangePickerProps {
  startDate: Date
  endDate: Date
  onStartChange: (d: Date) => void
  onEndChange: (d: Date) => void
}

export default function DateRangePicker({ startDate, endDate, onStartChange, onEndChange }: DateRangePickerProps) {
  const formatInputDate = (value: Date) => format(value, 'yyyy-MM-dd')
  const parseInputDate = (value: string) => {
    const [year, month, day] = value.split('-').map(Number)
    return new Date(year, month - 1, day)
  }

  return (
    <div className="flex items-center gap-2">
      <input
        type="date"
        value={formatInputDate(startDate)}
        onChange={(e) => onStartChange(parseInputDate(e.target.value))}
        className="input text-sm"
      />
      <span className="text-text-muted dark:text-text-dark-muted text-sm">to</span>
      <input
        type="date"
        value={formatInputDate(endDate)}
        onChange={(e) => onEndChange(parseInputDate(e.target.value))}
        className="input text-sm"
      />
    </div>
  )
}
