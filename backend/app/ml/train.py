import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, SimpleRNN

# ---------------- CONFIG ----------------
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "weather.csv"
MODELS_DIR = "models/rainfall"
SCALER_PATH = "scalers/rainfall_scaler.pkl"

WINDOW_SIZE = 60
HORIZONS = [1, 7, 15]
EPOCHS = 2
BATCH_SIZE = 32

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs("scalers", exist_ok=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["Date"])
df = df.sort_values("date")

# ---------------- FEATURE ENGINEERING ----------------
day_of_year = df["date"].dt.dayofyear
df["sin_day"] = np.sin(2 * np.pi * day_of_year / 365.25)
df["cos_day"] = np.cos(2 * np.pi * day_of_year / 365.25)
df["month"] = df["date"].dt.month
df["day_of_year"] = day_of_year

df["rainfall_lag_1"] = df["precipitation_mm"].shift(1)
df["rainfall_lag_3"] = df["precipitation_mm"].shift(3)
df["rainfall_lag_7"] = df["precipitation_mm"].shift(7)
df["rolling_mean_7"] = df["precipitation_mm"].shift(1).rolling(window=7, min_periods=1).mean()
df["rolling_std_7"] = df["precipitation_mm"].shift(1).rolling(window=7, min_periods=1).std().fillna(0)

df = df.dropna()

FEATURES = [
    "precipitation_mm", "temp_max", "temp_min",
    "humidity", "wind_speed", "solar_radiation",
    "pressure", "sin_day", "cos_day",
    "month", "day_of_year",
    "rainfall_lag_1", "rainfall_lag_3", "rainfall_lag_7",
    "rolling_mean_7", "rolling_std_7"
]

# ---------------- SCALING ----------------
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[FEATURES])
joblib.dump(scaler, SCALER_PATH)

def make_sequences(scaled_data: np.ndarray, window_size: int, horizon: int):
    X, y = [], []
    for i in range(len(scaled_data) - window_size - horizon):
        X.append(scaled_data[i:i + window_size])
        y.append(scaled_data[i + window_size:i + window_size + horizon, 0])
    return np.array(X), np.array(y)

# ---------------- MODELS ----------------
def build_lstm(input_shape, horizon: int):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dropout(0.2),
        Dense(horizon)
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.Huber(), metrics=['mae'])
    return model

def build_gru(input_shape, horizon: int):
    model = Sequential([
        GRU(64, input_shape=input_shape),
        Dropout(0.3),
        Dense(horizon)
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.Huber(), metrics=['mae'])
    return model

def build_bilstm(input_shape, horizon: int):
    model = Sequential([
        Bidirectional(LSTM(32), input_shape=input_shape),
        Dropout(0.25),
        Dense(horizon)
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.Huber(), metrics=['mae'])
    return model

def build_cnn_lstm(input_shape, horizon: int):
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        LSTM(32),
        Dense(horizon)
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.Huber(), metrics=['mae'])
    return model

def build_simplernn(input_shape, horizon: int):
    model = Sequential([
        SimpleRNN(32, input_shape=input_shape),
        Dropout(0.4),
        Dense(horizon)
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.Huber(), metrics=['mae'])
    return model

def build_wlstm(input_shape, horizon: int):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(32),
        Dense(horizon)
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.Huber(), metrics=['mae'])
    return model

MODELS = {
    "lstm": build_lstm,
    "gru": build_gru,
    "bilstm": build_bilstm,
    "cnn_lstm": build_cnn_lstm,
    "simplernn": build_simplernn,
    "wlstm": build_wlstm
}

# ---------------- TRAIN ----------------
results = {}

for horizon in HORIZONS:
    X, y = make_sequences(scaled, WINDOW_SIZE, horizon)
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    for name, builder in MODELS.items():
        print(f"\nTraining {name} ({horizon}d)")

        model = builder((X_train.shape[1], X_train.shape[2]), horizon)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=3)
            ]
        )

        model.save(f"{MODELS_DIR}/{name}_{horizon}d.keras")

        plt.figure()
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='val')
        plt.title(f"{name} Loss ({horizon}d)")
        plt.legend()
        plt.savefig(f"{MODELS_DIR}/{name}_{horizon}d_loss.png")
        plt.close()

        pred = model.predict(X_val)
        pred_flat = pred.flatten()
        y_flat = y_val.flatten()

        rmse = np.sqrt(np.mean((pred_flat - y_flat) ** 2))
        mae = np.mean(np.abs(pred_flat - y_flat))
        r2 = r2_score(y_flat, pred_flat)

        mean_obs = np.mean(y_flat)
        nse = 1 - (np.sum((pred_flat - y_flat)**2) / np.sum((y_flat - mean_obs)**2))

        print(f"{name} ({horizon}d) -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, NSE: {nse:.4f}")

        results[f"{name}_{horizon}d"] = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "nse": nse
        }

print("\n=== FINAL RESULTS ===")
for k, v in results.items():
    print(k, v)