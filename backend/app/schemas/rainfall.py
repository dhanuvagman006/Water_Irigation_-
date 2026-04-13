from pydantic import BaseModel
from datetime import date, datetime
from typing import Optional

class DayPrediction(BaseModel):
    date: date
    predicted_mm: float
    confidence_low: float   # 10th percentile estimate
    confidence_high: float  # 90th percentile estimate

class RainfallPredictRequest(BaseModel):
    model: str = "LSTM"     # model name
    days: int = 14          # forecast horizon
    start_date: Optional[date] = None

class RainfallPredictResponse(BaseModel):
    predictions: list[DayPrediction]
    model_used: str
    generated_at: datetime
