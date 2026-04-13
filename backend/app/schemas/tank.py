from pydantic import BaseModel
from datetime import date
from typing import Optional

class TankDayPrediction(BaseModel):
    date: date
    level: str              # Low/Medium/Full
    percentage: float
    estimated_liters: float

class TankPredictRequest(BaseModel):
    roof_area: float        # m²
    tank_capacity: float    # liters
    current_level: float    # liters
    daily_consumption: float # liters/day
    model: str = "LSTM"
    rainfall_predictions: Optional[list[float]] = None

class TankPredictResponse(BaseModel):
    predictions: list[TankDayPrediction]
    alert: Optional[str]    # "Tank LOW on Day 5"
    days_remaining: int
    model_used: str
