import numpy as np
import pandas as pd

class Preprocessor:
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "date" in df.columns:
            df["datetime"] = pd.to_datetime(df["date"], dayfirst=True, errors='coerce')
            df = df.dropna(subset=["datetime"])
            day_of_year = df["datetime"].dt.dayofyear
            df["sin_day"] = np.sin(2 * np.pi * day_of_year / 365.25)
            df["cos_day"] = np.cos(2 * np.pi * day_of_year / 365.25)
            df["month"] = df["datetime"].dt.month
            df["day_of_year"] = day_of_year
            df.drop(columns=["datetime"], inplace=True)
        return df

    def add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        precip = df["precipitation_mm"]
        df["rainfall_lag_1"] = precip.shift(1)
        df["rainfall_lag_3"] = precip.shift(3)
        df["rainfall_lag_7"] = precip.shift(7)
        df["rolling_mean_7"] = precip.shift(1).rolling(window=7, min_periods=1).mean()
        df["rolling_std_7"] = precip.shift(1).rolling(window=7, min_periods=1).std()
        df["rolling_std_7"] = df["rolling_std_7"].fillna(0)
        df = df.dropna(subset=["rainfall_lag_1", "rainfall_lag_3", "rainfall_lag_7"])
        df = df.reset_index(drop=True)
        return df

    def create_sliding_window(self, data: np.ndarray, window_size: int = 60) -> np.ndarray:
        if len(data) < window_size:
            raise ValueError(f"Not enough data. Expected at least {window_size} days, got {len(data)}")
        window = data[-window_size:]
        return np.expand_dims(window, axis=0)

    def prepare_rainfall_features(self, df: pd.DataFrame) -> np.ndarray:
        df = self.add_time_features(df)
        df = self.add_engineered_features(df)
        cols = [
            "precipitation_mm", "temp_max", "temp_min",
            "humidity", "wind_speed", "solar_radiation",
            "pressure", "sin_day", "cos_day",
            "month", "day_of_year",
            "rainfall_lag_1", "rainfall_lag_3", "rainfall_lag_7",
            "rolling_mean_7", "rolling_std_7"
        ]
        return df[cols].values

preprocessor = Preprocessor()
