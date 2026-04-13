from pydantic import BaseModel, Field, field_validator
from datetime import date, datetime
from typing import Optional

class DayPrediction(BaseModel):
    date: date
    predicted_mm: float
    confidence_low: float   # 10th percentile estimate
    confidence_high: float  # 90th percentile estimate

class RainfallPredictRequest(BaseModel):
    model: str = "LSTM"     # model name
    days: int = Field(default=14, ge=1, le=30)  # forecast horizon
    start_date: Optional[date] = None

    @field_validator("model")
    @classmethod
    def validate_model(cls, model: str) -> str:
        m = model.strip()
        if not m:
            raise ValueError("model must not be empty")
        return m

class RainfallPredictResponse(BaseModel):
    predictions: list[DayPrediction]
    model_used: str
    generated_at: datetime
