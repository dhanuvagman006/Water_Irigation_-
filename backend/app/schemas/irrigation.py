from pydantic import BaseModel
from datetime import date

class IrrigationDayPlan(BaseModel):
    date: date
    crop: str
    decision: str
    water_liters: float
    reason: str
    soil_moisture_forecast: float

class IrrigationPredictRequest(BaseModel):
    soil_moisture: float    # 0.0 - 1.0
    crop_types: list[str]   # ["Arecanut", "Coconut", "Pepper"]
    growth_stages: dict[str, str]     # {"Arecanut": "Vegetative", ...}
    model: str = "LSTM"
    num_plants: dict[str, int] = {}   # {"Arecanut": 50, "Coconut": 30}

class IrrigationPredictResponse(BaseModel):
    plan: list[IrrigationDayPlan]
    total_water_liters: dict[str, float]  # per crop, 14-day total
    model_used: str
