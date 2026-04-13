import type { ModelName } from '../../types'
import useAppStore from '../../store/useAppStore'

const models: ModelName[] = ['LSTM', 'GRU', 'BiLSTM', 'CNN-LSTM', 'Transformer', 'StackedLSTM']

interface ModelSelectorProps {
  value?: ModelName
  onChange?: (model: ModelName) => void
}

export default function ModelSelector({ value, onChange }: ModelSelectorProps) {
  const { selectedModel, setSelectedModel } = useAppStore()
  const currentModel = value ?? selectedModel
  const handleChange = onChange ?? setSelectedModel

  return (
    <select
      value={currentModel}
      onChange={(e) => handleChange(e.target.value as ModelName)}
      className="select pr-8"
    >
      {models.map((m) => (
        <option key={m} value={m}>
          {m}
        </option>
      ))}
    </select>
  )
}
