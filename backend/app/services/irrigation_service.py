import numpy as np
from datetime import date, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import logging
logger = logging.getLogger(__name__)
from fastapi import HTTPException
from app.schemas.irrigation import IrrigationPredictRequest, IrrigationPredictResponse, IrrigationDayPlan
from app.schemas.rainfall import RainfallPredictRequest
from app.services.rainfall_service import rainfall_service
from app.services.model_loader import ModelLoader
from app.config import settings

BASE_WATER_REQUIREMENTS = {
    "Arecanut": {"Vegetative": 18, "Flowering": 20, "Fruiting": 22, "Dormant": 10},
    "Coconut":  {"Vegetative": 12, "Flowering": 15, "Fruiting": 14, "Dormant": 8},
    "Pepper":   {"Vegetative": 6,  "Flowering": 8,  "Fruiting": 7,  "Dormant": 3},
}

TARGET_SOIL_MOISTURE = 0.72
SATURATION_THRESHOLD = 0.8

CROP_INDEX = {"Arecanut": 0, "Coconut": 1, "Pepper": 2}
STAGE_INDEX = {"Vegetative": 0, "Flowering": 1, "Fruiting": 2, "Dormant": 3}
DECISION_INDEX = {0: "Irrigate", 1: "No Irrigate", 2: "Monitor"}


def get_moisture_factor(soil_moisture: float) -> float:
    if soil_moisture >= SATURATION_THRESHOLD:
        return 0.0

    deficit_ratio = (TARGET_SOIL_MOISTURE - soil_moisture) / TARGET_SOIL_MOISTURE
    return float(np.clip(deficit_ratio, 0.0, 1.15))


class IrrigationService:
    async def predict(self, request: IrrigationPredictRequest, model_loader: ModelLoader, db_session: AsyncSession) -> IrrigationPredictResponse:
        rain_req = RainfallPredictRequest(model=settings.DEFAULT_MODEL, days=14)
        rain_resp = await rainfall_service.predict(rain_req, model_loader, db_session)
        rain_preds = [p.predicted_mm for p in rain_resp.predictions]

        temp_max = 30.0

        model_used = f"{request.model} (rule-based)"
        
        plan = []
        total_water_liters = {crop: 0.0 for crop in request.crop_types}
        
        for crop in request.crop_types:
            stage = request.growth_stages.get(crop, "Vegetative")
            base_need = BASE_WATER_REQUIREMENTS.get(crop, {}).get(stage, 10)
            num_plants = request.num_plants.get(crop, 1)
            
            current_moisture = request.soil_moisture
            
            for i in range(14):
                rain_today = rain_preds[i]
                rain_d2 = rain_preds[i + 1] if i + 1 < len(rain_preds) else 0.0
                rain_d3 = rain_preds[i + 2] if i + 2 < len(rain_preds) else 0.0

                roof_area = getattr(request, "roof_area", 100)
                efficiency = 0.8
                rainfall_contribution = (rain_today * roof_area * efficiency) / num_plants if num_plants > 0 else 0

                moisture_factor = get_moisture_factor(current_moisture)
                moisture_adjusted_need = base_need * moisture_factor
                irrigation_needed = max(0, moisture_adjusted_need - rainfall_contribution)

                if current_moisture >= 0.8:
                    decision = "No Irrigate"
                    liters = 0.0
                    reason = "Soil already saturated"
                elif rain_today > 10:
                    decision = "No Irrigate"
                    liters = 0.0
                    reason = "Sufficient rainfall today"
                elif irrigation_needed <= 0:
                    decision = "No Irrigate"
                    liters = 0.0
                    reason = "Rainfall covers requirement"
                elif irrigation_needed < (base_need * 0.5):
                    decision = "Monitor"
                    liters = round(irrigation_needed * num_plants, 2)
                    reason = "Partial irrigation needed"
                else:
                    decision = "Irrigate"
                    liters = round(irrigation_needed * num_plants, 2)
                    reason = "Soil moisture deficit"

                total_water_liters[crop] += liters
                
                rainfall_infiltration = (rain_today * 0.15) / 100
                irrigation_infiltration = (liters / 1000) * 0.3 / 100 if liters > 0 else 0
                evapotranspiration = 0.15 / 100
                
                current_moisture += rainfall_infiltration + irrigation_infiltration - evapotranspiration
                current_moisture = float(np.clip(current_moisture, 0.0, 1.0))
                
                plan.append(IrrigationDayPlan(
                    date=date.today() + timedelta(days=i+1),
                    crop=crop,
                    decision=decision,
                    water_liters=liters,
                    reason=reason,
                    soil_moisture_forecast=round(current_moisture, 4)
                ))
            
        return IrrigationPredictResponse(
            plan=plan,
            total_water_liters={crop: round(total, 2) for crop, total in total_water_liters.items()},
            model_used=model_used
        )

irrigation_service = IrrigationService()