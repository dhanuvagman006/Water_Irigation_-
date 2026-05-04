import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from fastapi import HTTPException
from app.schemas.rainfall import RainfallPredictRequest, RainfallPredictResponse, DayPrediction
from app.services.preprocessor import preprocessor
from app.services.model_loader import ModelLoader
from app.database.models import NASADataRecord

HORIZON_MAP = {"short": 1, "medium": 7, "long": 15}

class RainfallService:
    async def predict(self, request: RainfallPredictRequest, model_loader: ModelLoader, db_session: AsyncSession) -> RainfallPredictResponse:
        context_end = (request.start_date or date.today()) - timedelta(days=1)
        df = await self._get_local_data(db_session, end_date=context_end, days=60)

        prediction_date = request.start_date or (date.today() + timedelta(days=1))
        df = await self._inject_seasonal_context(df, db_session, prediction_date)

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

    async def _get_local_data(self, db_session: AsyncSession, end_date: date = None, days: int = 60) -> pd.DataFrame:
        """Fetch weather data from local database (populated from CSV)"""
        if end_date is None:
            end_date = date.today()
        start_date = end_date - timedelta(days=days)

        stmt = select(NASADataRecord).where(
            NASADataRecord.date >= start_date,
            NASADataRecord.date <= end_date
        ).order_by(NASADataRecord.date)

        result = await db_session.execute(stmt)
        records = list(result.scalars().all())

        if not records:
            return pd.DataFrame()

        data = []
        for r in records:
            data.append({
                "date": r.date,
                "precipitation_mm": r.precipitation_mm,
                "temp_max": r.temp_max,
                "temp_min": r.temp_min,
                "humidity": r.humidity,
                "wind_speed": r.wind_speed,
                "solar_radiation": r.solar_radiation,
                "pressure": r.pressure
            })

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        return df

    async def _inject_seasonal_context(self, df, db_session: AsyncSession, target_date: date):
        df = df.copy()
        if "date" in df.columns:
            current_month = target_date.month
            current_doy = target_date.timetuple().tm_yday

            stmt = select(NASADataRecord).order_by(NASADataRecord.date.desc()).limit(3650)
            result = await db_session.execute(stmt)
            historical = list(result.scalars().all())

            if historical:
                hist_data = []
                for r in historical:
                    if r.precipitation_mm is not None:
                        hist_data.append({
                            "date": r.date,
                            "precipitation_mm": r.precipitation_mm,
                            "temp_max": r.temp_max,
                            "temp_min": r.temp_min,
                            "humidity": r.humidity,
                            "wind_speed": r.wind_speed,
                            "solar_radiation": r.solar_radiation,
                            "pressure": r.pressure
                        })

                if hist_data:
                    hist_df = pd.DataFrame(hist_data)
                    hist_df["datetime"] = pd.to_datetime(hist_df["date"])
                    hist_df["month"] = hist_df["datetime"].dt.month
                    hist_df["doy"] = hist_df["datetime"].dt.dayofyear

                    month_mask = hist_df["month"] == current_month
                    doy_mask = abs(hist_df["doy"] - current_doy) <= 15
                    seasonal_mask = month_mask | doy_mask

                    seasonal = hist_df[seasonal_mask]
                    if len(seasonal) < 5:
                        seasonal = hist_df[abs(hist_df["doy"] - current_doy) <= 30]

                    if len(seasonal) >= 5:
                        seasonal_avg = {
                            "precipitation_mm": seasonal["precipitation_mm"].mean(),
                            "temp_max": seasonal["temp_max"].mean() if seasonal["temp_max"].notna().any() else df["temp_max"].mean(),
                            "temp_min": seasonal["temp_min"].mean() if seasonal["temp_min"].notna().any() else df["temp_min"].mean(),
                            "humidity": seasonal["humidity"].mean() if seasonal["humidity"].notna().any() else df["humidity"].mean(),
                            "wind_speed": seasonal["wind_speed"].mean() if seasonal["wind_speed"].notna().any() else df["wind_speed"].mean(),
                            "solar_radiation": seasonal["solar_radiation"].mean() if seasonal["solar_radiation"].notna().any() else df["solar_radiation"].mean(),
                            "pressure": seasonal["pressure"].mean() if seasonal["pressure"].notna().any() else df["pressure"].mean(),
                        }

                        alpha = 0.3
                        for col in seasonal_avg:
                            if col in df.columns:
                                df[col] = df[col].fillna(seasonal_avg[col])
                                recent_vals = df[col].dropna()
                                if len(recent_vals) > 0:
                                    df.loc[df.index[-1], col] = (1 - alpha) * df.loc[df.index[-1], col] + alpha * seasonal_avg[col]

        return df

    def _predict_with_model(self, request: RainfallPredictRequest, model_loader: ModelLoader, df):
        raw_features = preprocessor.prepare_rainfall_features(df)
        scaler = model_loader.get_scaler("rainfall")
        scaled_features = scaler.transform(raw_features)
        X = preprocessor.create_sliding_window(scaled_features, window_size=30)

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
