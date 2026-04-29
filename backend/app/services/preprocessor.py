import numpy as np
import pandas as pd

class Preprocessor:
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "date" in df.columns:
            # Convert date to datetime, handle new NASA format DD-MM-YYYY
            df["datetime"] = pd.to_datetime(df["date"], dayfirst=True, errors='coerce')
            # Drop rows with invalid dates if any
            df = df.dropna(subset=["datetime"])
            day_of_year = df["datetime"].dt.dayofyear
            df["sin_day"] = np.sin(2 * np.pi * day_of_year / 365.25)
            df["cos_day"] = np.cos(2 * np.pi * day_of_year / 365.25)
            df.drop(columns=["datetime"], inplace=True)
        return df

    def create_sliding_window(self, data: np.ndarray, window_size: int = 30) -> np.ndarray:
        """
        Takes an array of shape (N, features) and returns shape (1, window_size, features)
        Assumes N == window_size for a single inference step.
        """
        if len(data) < window_size:
            raise ValueError(f"Not enough data. Expected at least {window_size} days, got {len(data)}")
        
        # Take the most recent window_size days
        window = data[-window_size:]
        return np.expand_dims(window, axis=0)

    def prepare_rainfall_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Build feature matrix:
        [precipitation, temp_max, temp_min,
         humidity, wind_speed, solar_rad,
         sin(day_of_year), cos(day_of_year)]
        """
        df = self.add_time_features(df)
        cols = [
            "precipitation_mm", "temp_max", "temp_min",
            "humidity", "wind_speed", "solar_radiation",
            "pressure", "sin_day", "cos_day"
        ]
        return df[cols].values

preprocessor = Preprocessor()
