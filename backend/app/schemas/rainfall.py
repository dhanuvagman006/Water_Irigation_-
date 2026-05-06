from pydantic import BaseModel, Field, field_validator
from datetime import date, datetime
from typing import Optional

class DayPrediction(BaseModel):
    date: date
    predicted_mm: float
    confidence_low: float   # 10th percentile estimate
    confidence_high: float  # 90th percentile estimate

class RainfallPredictRequest(BaseModel):
    model: str = "LSTM"
    days: int = Field(default=7, ge=1, le=15)
    horizon: str = Field(default="medium", pattern="^(short|medium|long)$")
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

class RainfallRecommendation(BaseModel):
    tab: str
    model: str
    confidence: int

class RainfallSummaryResponse(BaseModel):
    best_model: str
    confidence: int
    rmse: Optional[float] = None
    nse: Optional[float] = None
    r2: Optional[float] = None
    recommendations: list[RainfallRecommendation]
