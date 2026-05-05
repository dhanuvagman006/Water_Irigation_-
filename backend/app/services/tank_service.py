import numpy as np
from datetime import date, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.tank import TankPredictRequest, TankPredictResponse, TankDayPrediction
from app.schemas.rainfall import RainfallPredictRequest
from app.services.rainfall_service import rainfall_service
from app.services.model_loader import ModelLoader

TANK_LEVELS = {0: "Low", 1: "Medium", 2: "Full"}

class TankService:
    async def predict(self, request: TankPredictRequest, model_loader: ModelLoader, db_session: AsyncSession) -> TankPredictResponse:
        rainfall_predictions = request.rainfall_predictions
        
        # 1. If rainfall predictions not provided, call RainfallService internally
        if not rainfall_predictions:
            rain_req = RainfallPredictRequest(model="LSTM", days=14)
            rain_resp = await rainfall_service.predict(rain_req, model_loader, db_session)
            rainfall_predictions = [p.predicted_mm for p in rain_resp.predictions]
        
        model_used = "physics model"

        days_ahead = len(rainfall_predictions)
        predictions = []
        current_level = request.current_level
        alert = None

        # We process day by day
        for i in range(days_ahead):
            rain_mm = rainfall_predictions[i]

            level_status = None
            # Convert rainfall to liters
            water_in_liters = rain_mm * request.roof_area * 0.8

            # Update tank
            current_level += (water_in_liters - request.daily_consumption)

            # Clamp values
            current_level = float(np.clip(current_level, 0.0, request.tank_capacity))

            # Percentage
            percentage = (current_level / request.tank_capacity) * 100.0

            # Classification
            if percentage < 25:
                level_status = "Low"
            elif percentage < 75:
                level_status = "Medium"
            else:
                level_status = "Full"

            if level_status == "Low" and not alert:
                alert = f"Tank LOW on Day {i+1}"

            predictions.append(TankDayPrediction(
                date=date.today() + timedelta(days=i+1),
                level=level_status,
                percentage=percentage,
                estimated_liters=current_level
            ))

        days_remaining = int(current_level / request.daily_consumption) if request.daily_consumption > 0 else 999

        return TankPredictResponse(
            predictions=predictions,
            alert=alert,
            days_remaining=days_remaining,
            model_used=model_used
        )

tank_service = TankService()
