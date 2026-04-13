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
        
        # Determine crop and stage encodings (simplified for the demo)
        crop_map = {"Arecanut": 0, "Coconut": 1, "Pepper": 2}
        stage_map = {"Vegetative": 0, "Flowering": 1, "Fruiting": 2, "Dormant": 3}
        decision_map = {0: "Irrigate", 1: "No Irrigate", 2: "Monitor"}
        
        for crop in request.crop_types:
            stage = request.growth_stages.get(crop, "Vegetative")
            base_need = BASE_WATER_REQUIREMENTS.get(crop, {}).get(stage, 10)
            num_plants = request.num_plants.get(crop, 1)
            
            crop_encoded = crop_map.get(crop, 0)
            stage_encoded = stage_map.get(stage, 0)
            
            current_moisture = request.soil_moisture
            
            for i in range(14):
                rain_today = rain_preds[i]
                # Dummy simulation of future rainfall impacts
                rain_day2 = rain_preds[i+1] if i+1 < 14 else 0.0
                rain_day3 = rain_preds[i+2] if i+2 < 14 else 0.0
                
                # Mock high temp for feature
                temp_max = 30.0 
                
                # 3. Build feature array
                features = np.array([[current_moisture, rain_today, rain_day2, rain_day3, temp_max, crop_encoded, stage_encoded]])
                
                try:
                    scaler = model_loader.get_scaler("irrigation")
                    scaled = scaler.transform(features)
                    model = model_loader.get_model("irrigation", request.model)
                    out = model.predict(scaled, verbose=0)
                    decision_idx = int(np.argmax(out, axis=1)[0])
                    decision = decision_map.get(decision_idx, "Monitor")
                except Exception:
                    # Fallback logic if model fails/missing
                    if current_moisture < 0.3 and rain_today < 5.0:
                        decision = "Irrigate"
                    elif current_moisture > 0.7 or rain_today > 10.0:
                        decision = "No Irrigate"
                    else:
                        decision = "Monitor"

                if decision == "Irrigate":
                    liters = base_need * num_plants
                    reason = f"Low moisture ({current_moisture:.2f}) and low rain predicted."
                elif decision == "Monitor":
                    liters = 0.5 * base_need * num_plants
                    reason = f"Moderate moisture ({current_moisture:.2f}), reducing water."
                else: # No Irrigate
                    liters = 0.0
                    reason = f"Sufficient moisture/rain ({rain_today:.1f}mm), skipping irrigation."

                total_water_liters[crop] += liters
                
                # Simulate moisture change
                if decision == "Irrigate":
                    current_moisture += 0.2
                current_moisture += (rain_today * 0.05) - 0.1 # EVAP
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
