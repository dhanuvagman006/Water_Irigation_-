import numpy as np
from datetime import date, datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException
from app.schemas.rainfall import RainfallPredictRequest, RainfallPredictResponse, DayPrediction
from app.services.nasa_service import nasa_service
from app.services.preprocessor import preprocessor
from app.services.model_loader import ModelLoader

HORIZON_MAP = {"short": 1, "medium": 7, "long": 15}

class RainfallService:
    async def predict(self, request: RainfallPredictRequest, model_loader: ModelLoader, db_session: AsyncSession) -> RainfallPredictResponse:
        df = await nasa_service.get_recent(days=30, db=db_session)
        if len(df) < 30:
            raise HTTPException(status_code=422, detail="Insufficient NASA data (less than 30 days). Please trigger a data fetch.")

        try:
            predicted_mm, model_used = self._predict_with_model(request, model_loader, df)
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
            predicted_mm = self._predict_from_recent_rainfall(df, request.days)
            model_used = f"{request.model} (recent-rainfall fallback)"
        except Exception:
            predicted_mm = self._predict_from_recent_rainfall(df, request.days)
            model_used = f"{request.model} (recent-rainfall fallback)"

        actual_days = min(request.days, len(predicted_mm))

        predicted_mm = np.clip(predicted_mm, 0.0, 200.0)

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
            model_used=model_used,
            generated_at=datetime.utcnow()
        )

    def _predict_with_model(self, request: RainfallPredictRequest, model_loader: ModelLoader, df):
        raw_features = preprocessor.prepare_rainfall_features(df)
        scaler = model_loader.get_scaler("rainfall")
        scaled_features = scaler.transform(raw_features)
        X = preprocessor.create_sliding_window(scaled_features, window_size=30)

        horizon = getattr(request, "horizon", "medium")
        model = model_loader.get_model("rainfall", request.model, horizon=horizon)
        raw_output = model.predict(X, verbose=0)

        if len(raw_output.shape) == 2:
            output = raw_output[0]
        else:
            output = raw_output.flatten()

        output_len = len(output)
        actual_days = min(request.days, output_len)

        dummy = np.zeros((actual_days, raw_features.shape[1]))
        dummy[:, 0] = output[:actual_days]

        return scaler.inverse_transform(dummy)[:, 0], f"{request.model} ({horizon})"

    def _predict_from_recent_rainfall(self, df, days: int) -> np.ndarray:
        rainfall = df["precipitation_mm"].astype(float).to_numpy()
        recent = rainfall[-14:] if len(rainfall) >= 14 else rainfall
        weekly_pattern = rainfall[-7:] if len(rainfall) >= 7 else recent
        baseline = float(np.mean(recent)) if len(recent) else 0.0

        forecast = []
        for i in range(days):
            pattern_value = float(weekly_pattern[i % len(weekly_pattern)]) if len(weekly_pattern) else baseline
            forecast.append((pattern_value * 0.65) + (baseline * 0.35))
        return np.array(forecast, dtype=float)

rainfall_service = RainfallService()
