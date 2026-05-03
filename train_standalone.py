import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_NUM_INTEROP_THREADS"] = "4"
os.environ["TF_NUM_INTRA_OP_THREADS"] = "4"

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import joblib

np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow version: {tf.__version__}")
print(f"Using {os.cpu_count()} CPU cores")

# Load dataset
dataset_path = "./Dakshina_Kannada_Weather_2000_2024.csv"
print(f"\nLoading dataset: {dataset_path}")
raw_df = pd.read_csv(dataset_path)
print(f"Dataset shape: {raw_df.shape}")
print(f"Date range: {raw_df['Date'].min()} to {raw_df['Date'].max()}")

# Rename Date column
raw_df.rename(columns={"Date": "date"}, inplace=True)
raw_df["date"] = pd.to_datetime(raw_df["date"])
raw_df = raw_df.sort_values("date").reset_index(drop=True)

# Add time features
day_of_year = raw_df["date"].dt.dayofyear
raw_df["sin_day"] = np.sin(2 * np.pi * day_of_year / 365.25)
raw_df["cos_day"] = np.cos(2 * np.pi * day_of_year / 365.25)
raw_df["month"] = raw_df["date"].dt.month
raw_df["day_of_year"] = day_of_year

# Add engineered features
precip = raw_df["precipitation_mm"]
raw_df["rainfall_lag_1"] = precip.shift(1)
raw_df["rainfall_lag_3"] = precip.shift(3)
raw_df["rainfall_lag_7"] = precip.shift(7)
raw_df["rolling_mean_7"] = precip.shift(1).rolling(window=7, min_periods=1).mean()
raw_df["rolling_std_7"] = precip.shift(1).rolling(window=7, min_periods=1).std()
raw_df["rolling_std_7"] = raw_df["rolling_std_7"].fillna(0)
raw_df = raw_df.dropna(subset=["rainfall_lag_1", "rainfall_lag_3", "rainfall_lag_7"])
raw_df = raw_df.reset_index(drop=True)

print(f"After feature engineering: {raw_df.shape}")

# Build dataset
cols = ["precipitation_mm", "temp_max", "temp_min", "humidity", "wind_speed", "solar_radiation", "pressure", "sin_day", "cos_day", "month", "day_of_year", "rainfall_lag_1", "rainfall_lag_3", "rainfall_lag_7", "rolling_mean_7", "rolling_std_7"]

window_size = 30
horizon = 15

features = raw_df[cols].values
precip_vals = raw_df["precipitation_mm"].values

# Scale features
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(raw_df[cols].values)

X, y = [], []
n_samples = len(scaled_data) - window_size - horizon + 1
print(f"\nTotal samples: {n_samples}")

for i in range(n_samples):
    X.append(scaled_data[i : i + window_size])
    y.append(scaled_data[i + window_size : i + window_size + horizon, 0])

X = np.array(X)
y = np.array(y)

train_size = int(0.8 * len(X))
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

print(f"Train: {X_train.shape}, Val: {X_val.shape}")

# Save scaler
scalers_dir = "./backend/scalers"
os.makedirs(scalers_dir, exist_ok=True)
joblib.dump(scaler, os.path.join(scalers_dir, "rainfall_scaler.pkl"))
print(f"Scaler saved to {scalers_dir}/rainfall_scaler.pkl")

# Define models
def build_lstm(input_shape, out_dim):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dropout(0.2),
        Dense(out_dim)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_gru(input_shape, out_dim):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout
    model = Sequential([
        GRU(64, input_shape=input_shape),
        Dropout(0.3),
        Dense(out_dim)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_bilstm(input_shape, out_dim):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
    model = Sequential([
        Bidirectional(LSTM(32), input_shape=input_shape),
        Dropout(0.25),
        Dense(out_dim)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_cnn_lstm(input_shape, out_dim):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
    ks = 3 if input_shape[0] > 2 else 1
    model = Sequential([
        Conv1D(filters=32, kernel_size=ks, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2) if input_shape[0] >= 2 else lambda x: x,
        LSTM(32),
        Dropout(0.2),
        Dense(out_dim)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_simplernn(input_shape, out_dim):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
    model = Sequential([
        SimpleRNN(32, input_shape=input_shape),
        Dropout(0.4),
        Dense(out_dim)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_lstm_attention(input_shape, out_dim):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D, Input
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.1)(x)
    att = Attention()([x, x])
    x = GlobalAveragePooling1D()(att)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(out_dim)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_transformer(input_shape, out_dim):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Dropout, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Input
    inputs = Input(shape=input_shape)
    x = MultiHeadAttention(num_heads=2, key_dim=16)(inputs, inputs)
    x = Dropout(0.2)(x)
    x = LayerNormalization(epsilon=1e-6)(x + inputs)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(out_dim)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

BUILD_FNS = {
    "lstm": build_lstm,
    "gru": build_gru,
    "bilstm": build_bilstm,
    "cnn_lstm": build_cnn_lstm,
    "simplernn": build_simplernn,
    "lstm_attention": build_lstm_attention,
    "transformer": build_transformer,
}

AUTOTUNE = tf.data.AUTOTUNE

def make_dataset(X, y, batch_size=32, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(y), seed=42)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds

# Train for each horizon
HORIZONS = {"short": 1, "medium": 7, "long": 15}

models_dir = "./backend/models"

for h_name, horizon in HORIZONS.items():
    h_suffix = f"{horizon}d"
    print(f"\n{'='*60}")
    print(f" HORIZON: {h_name.upper()} ({horizon}-day ahead)")
    print(f"{'='*60}")
    
    y_train_h = y_train[:, :horizon]
    y_val_h = y_val[:, :horizon]
    
    for name, build_fn in BUILD_FNS.items():
        print(f"\n  Training {name} ({h_suffix})...")
        
        model = build_fn(X_train.shape[1:], horizon)
        
        train_ds = make_dataset(X_train, y_train_h, batch_size=64, shuffle=True)
        val_ds = make_dataset(X_val, y_val_h, batch_size=64)
        
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=100,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5),
            ]
        )
        
        # Save model
        model_dir = os.path.join(models_dir, "rainfall")
        os.makedirs(model_dir, exist_ok=True)
        model.save(os.path.join(model_dir, f"{name}_{h_suffix}.keras"))
        
        # Evaluate
        y_pred = model.predict(X_val, verbose=0)
        
        # Inverse transform for metrics
        n_features = X_train.shape[2]
        dummy_pred = np.zeros((len(y_pred.flatten()), n_features))
        dummy_true = np.zeros((len(y_val_h.flatten()), n_features))
        dummy_pred[:, 0] = y_pred.flatten()
        dummy_true[:, 0] = y_val_h.flatten()
        
        inv_pred = scaler.inverse_transform(dummy_pred)[:, 0]
        inv_true = scaler.inverse_transform(dummy_true)[:, 0]
        
        rmse = float(np.sqrt(np.mean((inv_pred - inv_true) ** 2)))
        mae = float(np.mean(np.abs(inv_pred - inv_true)))
        mean_obs = np.mean(inv_true)
        ss_res = np.sum((inv_pred - inv_true) ** 2)
        ss_var = np.sum((inv_true - mean_obs) ** 2)
        nse = 1 - (ss_res / ss_var) if ss_var > 0 else 0
        r2 = float(r2_score(inv_true, inv_pred))
        
        print(f"    RMSE: {rmse:.4f}, MAE: {mae:.4f}, NSE: {nse:.4f}, R2: {r2:.4f}")

print("\n\nTraining complete! Models saved.")
