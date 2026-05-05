import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from fastapi import HTTPException
from app.schemas.rainfall import RainfallPredictRequest, RainfallPredictResponse, DayPrediction
from app.services.preprocessor import preprocessor
from app.services.model_loader import ModelLoader
from app.config import settings
HORIZON_MAP = {"short": 1, "medium": 7, "long": 15}
def load_local_csv(path: str):
    import pandas as pd

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df

class RainfallService:
    def predict(self, request: RainfallPredictRequest, model_loader: ModelLoader, db_session: AsyncSession) -> RainfallPredictResponse:
        context_end = (request.start_date or date.today()) - timedelta(days=1)
        # df = await self._get_local_data(db_session, end_date=context_end, days=60)
        df = load_local_csv(settings.DATASET_PATH)
        
        prediction_date = request.start_date or (date.today() + timedelta(days=1))

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
        X = preprocessor.create_sliding_window(scaled_features, window_size=60)

        horizon = getattr(request, "horizon", "medium")
        model = model_loader.get_model("rainfall", request.model, horizon=horizon)
        raw_output = model.predict(X, verbose=0)

        if len(raw_output.shape) == 2:
            output = raw_output[-1]
        else:
            output = raw_output.flatten()

        output_len = len(output)
        actual_days = min(request.days, output_len)

        dummy = np.zeros((actual_days, raw_features.shape[1]))
        dummy[:, 0] = output[:actual_days]

        predicted = scaler.inverse_transform(dummy)[:, 0]

        prediction_date = request.start_date or (date.today() + timedelta(days=1))
        month = prediction_date.month
        if month in [12, 1, 2]:
            seasonal_factor = 0.2
        elif month in [3, 4]:
            seasonal_factor = 0.5
        elif month == 5:
            seasonal_factor = 0.8
        elif month in [6, 7, 8]:
            seasonal_factor = 1.5
        elif month == 9:
            seasonal_factor = 1.2
        elif month == 10:
            seasonal_factor = 0.9
        else:
            seasonal_factor = 0.4

        predicted = predicted * seasonal_factor

        return predicted, f"{request.model} ({horizon})"

    def _predict_from_recent_rainfall(self, df, days: int) -> np.ndarray:
        rainfall = df["precipitation_mm"].astype(float).to_numpy()
        recent = rainfall[-14:] if len(rainfall) >= 14 else rainfall
        weekly_pattern = rainfall[-7:] if len(rainfall) >= 7 else recent
        baseline = float(np.mean(recent)) if len(recent) else 0.0

        prediction_date = (date.today() + timedelta(days=1))
        month = prediction_date.month
        if month in [12, 1, 2]:
            seasonal_factor = 0.2
        elif month in [3, 4]:
            seasonal_factor = 0.5
        elif month == 5:
            seasonal_factor = 0.8
        elif month in [6, 7, 8]:
            seasonal_factor = 1.5
        elif month == 9:
            seasonal_factor = 1.2
        elif month == 10:
            seasonal_factor = 0.9
        else:
            seasonal_factor = 0.4

        forecast = []
        for i in range(days):
            pattern_value = float(weekly_pattern[i % len(weekly_pattern)]) if len(weekly_pattern) else baseline
            forecast.append(((pattern_value * 0.65) + (baseline * 0.35)) * seasonal_factor)
        return np.array(forecast, dtype=float)

rainfall_service = RainfallService()
