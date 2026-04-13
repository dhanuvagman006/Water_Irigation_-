from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import date
from typing import Optional

class TankDayPrediction(BaseModel):
    date: date
    level: str              # Low/Medium/Full
    percentage: float
    estimated_liters: float

class TankPredictRequest(BaseModel):
    roof_area: float = Field(..., gt=0.0)          # m²
    tank_capacity: float = Field(..., gt=0.0)      # liters
    current_level: float = Field(..., ge=0.0)      # liters
    daily_consumption: float = Field(..., ge=0.0)  # liters/day
    model: str = "LSTM"
    rainfall_predictions: Optional[list[float]] = None

    @field_validator("model")
    @classmethod
    def validate_model(cls, model: str) -> str:
        m = model.strip()
        if not m:
            raise ValueError("model must not be empty")
        return m

    @field_validator("rainfall_predictions")
    @classmethod
    def validate_rainfall_predictions(cls, preds: Optional[list[float]]) -> Optional[list[float]]:
        if preds is None:
            return None
        if not preds:
            return None
        if len(preds) > 30:
            raise ValueError("rainfall_predictions must have at most 30 values")
        if any((p is None) or (p < 0) for p in preds):
            raise ValueError("rainfall_predictions must be non-negative")
        return preds

    @model_validator(mode="after")
    def validate_level_with_capacity(self):
        if self.current_level > self.tank_capacity:
            raise ValueError("current_level must be <= tank_capacity")
        return self

class TankPredictResponse(BaseModel):
    predictions: list[TankDayPrediction]
    alert: Optional[str]    # "Tank LOW on Day 5"
    days_remaining: int
    model_used: str
