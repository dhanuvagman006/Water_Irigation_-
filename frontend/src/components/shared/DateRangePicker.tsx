import { format } from 'date-fns'

interface DateRangePickerProps {
  startDate: Date
  endDate: Date
  onStartChange: (d: Date) => void
  onEndChange: (d: Date) => void
}

export default function DateRangePicker({ startDate, endDate, onStartChange, onEndChange }: DateRangePickerProps) {
  return (
    <div className="flex items-center gap-2">
      <input
        type="date"
        value={format(startDate, 'yyyy-MM-dd')}
        onChange={(e) => onStartChange(new Date(e.target.value))}
        className="input text-sm"
      />
      <span className="text-text-muted dark:text-text-dark-muted text-sm">to</span>
      <input
        type="date"
        value={format(endDate, 'yyyy-MM-dd')}
        onChange={(e) => onEndChange(new Date(e.target.value))}
        className="input text-sm"
      />
    </div>
  )
}
