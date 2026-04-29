import numpy as np
from datetime import date, datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException
from app.schemas.rainfall import RainfallPredictRequest, RainfallPredictResponse, DayPrediction
from app.services.nasa_service import nasa_service
from app.services.preprocessor import preprocessor
from app.services.model_loader import ModelLoader

class RainfallService:
    async def predict(self, request: RainfallPredictRequest, model_loader: ModelLoader, db_session: AsyncSession) -> RainfallPredictResponse:
        # 1. Fetch last 30 days of NASA data
        df = await nasa_service.get_recent(days=30, db=db_session)
        if len(df) < 30:
            raise HTTPException(status_code=422, detail="Insufficient NASA data (less than 30 days). Please trigger a data fetch.")
            
        # 2. Build feature matrix
        raw_features = preprocessor.prepare_rainfall_features(df)
        
        # 3. Normalize with rainfall scaler
        scaler = model_loader.get_scaler("rainfall")
        # Ensure our features match the scaler's training shape
        # In a real app we might only scale certain columns, but here we assume the scaler expects all 8 features
        scaled_features = scaler.transform(raw_features)
        
        # 4. Create sliding window [1, 30, 8]
        X = preprocessor.create_sliding_window(scaled_features, window_size=30)
        
        # 5. Load selected model
        model = model_loader.get_model("rainfall", request.model)
        
        # 6. Predict -> raw output (usually 14 days)
        # Note: We assume the model predicts limited days directly.
        raw_output = model.predict(X, verbose=0)
        
        # Determine how many days we can actually extract
        output_len = raw_output.shape[1] if len(raw_output.shape) == 2 else len(raw_output.flatten())
        actual_days = min(request.days, output_len)
        
        # 7. Inverse transform (requires dummy structure if scaler expects 8 features)
        dummy = np.zeros((actual_days, raw_features.shape[1]))
        if len(raw_output.shape) == 2:
            dummy[:, 0] = raw_output[0, :actual_days]
        else:
            dummy[:, 0] = raw_output.flatten()[:actual_days] # Fallback
            
        unscaled_dummy = scaler.inverse_transform(dummy)
        predicted_mm = unscaled_dummy[:, 0]
        
        # 8. Clip negative values to 0
        predicted_mm = np.clip(predicted_mm, 0.0, None)
        
        # 9. Confidence interval +/- 15%
        predictions = []
        start_date = request.start_date or (date.today() + timedelta(days=1))
        
        for i in range(actual_days):
            val = float(predicted_mm[i])
            day_date = start_date + timedelta(days=i)
            predictions.append(DayPrediction(
                date=day_date,
                predicted_mm=val,
                confidence_low=val * 0.85,
                confidence_high=val * 1.15
            ))
            
        return RainfallPredictResponse(
            predictions=predictions,
            model_used=request.model,
            generated_at=datetime.utcnow()
        )

rainfall_service = RainfallService()
