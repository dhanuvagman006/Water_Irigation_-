import numpy as np
from datetime import date, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.irrigation import IrrigationPredictRequest, IrrigationPredictResponse, IrrigationDayPlan
from app.schemas.rainfall import RainfallPredictRequest
from app.services.rainfall_service import rainfall_service
from app.services.model_loader import ModelLoader

BASE_WATER_REQUIREMENTS = {
    "Arecanut": {"Vegetative": 18, "Flowering": 20, "Fruiting": 22, "Dormant": 10},
    "Coconut":  {"Vegetative": 12, "Flowering": 15, "Fruiting": 14, "Dormant": 8},
    "Pepper":   {"Vegetative": 6,  "Flowering": 8,  "Fruiting": 7,  "Dormant": 3},
}

class IrrigationService:
    async def predict(self, request: IrrigationPredictRequest, model_loader: ModelLoader, db_session: AsyncSession) -> IrrigationPredictResponse:
        # 1. Base water need is already in BASE_WATER_REQUIREMENTS
        # 2. Fetch rainfall predictions for 14 days
        rain_req = RainfallPredictRequest(model="LSTM", days=14)
        rain_resp = await rainfall_service.predict(rain_req, model_loader, db_session)
        rain_preds = [p.predicted_mm for p in rain_resp.predictions]
        
        plan = []
        total_water_liters = {crop: 0.0 for crop in request.crop_types}
        
        for crop in request.crop_types:
            stage = request.growth_stages.get(crop, "Vegetative")
            base_need = BASE_WATER_REQUIREMENTS.get(crop, {}).get(stage, 10)
            num_plants = request.num_plants.get(crop, 1)
            
            current_moisture = request.soil_moisture
            
            for i in range(14):
                rain_today = rain_preds[i]
                
                # CRITICAL FIX: Account for rainfall in water need calculation
                # Assume roof_area = 100 m² for rain-to-water conversion
                # 1 mm rainfall × area × efficiency = liters per plant
                roof_area = getattr(request, 'roof_area', 100)  # default 100 m²
                efficiency = 0.8  # 20% loss in collection/infiltration
                rainfall_contribution = (rain_today * roof_area * efficiency) / num_plants if num_plants > 0 else 0
                
                # Calculate actual irrigation needed (base need minus rainfall)
                irrigation_needed = max(0, base_need - rainfall_contribution)
                
                # CRITICAL FIX: Check soil saturation - do NOT irrigate if already wet
                should_skip_saturation = current_moisture > 0.8
                
                # Determine decision based on irrigation need and soil state
                if should_skip_saturation:
                    decision = "No Irrigate"
                    liters = 0.0
                    reason = f"Soil saturated (moisture={current_moisture:.2f}), skipping irrigation."
                elif irrigation_needed <= 0:
                    decision = "No Irrigate"
                    liters = 0.0
                    reason = f"Rainfall sufficient ({rain_today:.1f}mm covers need), no irrigation needed."
                elif irrigation_needed < (base_need * 0.5):
                    decision = "Monitor"
                    liters = irrigation_needed * num_plants
                    reason = f"Rainfall partially covers need ({rain_today:.1f}mm), reducing irrigation to {liters:.1f}L."
                else:
                    decision = "Irrigate"
                    liters = irrigation_needed * num_plants
                    reason = f"Low rainfall ({rain_today:.1f}mm) and moisture ({current_moisture:.2f}), irrigation needed."

                total_water_liters[crop] += liters
                
                # Update soil moisture based on rainfall and irrigation
                # Simplified physics: infiltration from rain + irrigation, evapotranspiration loss
                rainfall_infiltration = (rain_today * 0.15) / 100  # normalized to 0-1 scale
                irrigation_infiltration = (liters / 1000) * 0.3 / 100 if liters > 0 else 0  # some irrigation water infiltrates
                evapotranspiration = 0.15 / 100  # daily ET loss normalized
                
                current_moisture += rainfall_infiltration + irrigation_infiltration - evapotranspiration
                current_moisture = float(np.clip(current_moisture, 0.0, 1.0))
                
                plan.append(IrrigationDayPlan(
                    date=date.today() + timedelta(days=i+1),
                    crop=crop,
                    decision=decision,
                    water_liters=liters,
                    reason=reason,
                    soil_moisture_forecast=current_moisture
                ))
                
        return IrrigationPredictResponse(
            plan=plan,
            total_water_liters=total_water_liters,
            model_used=request.model
        )

irrigation_service = IrrigationService()
