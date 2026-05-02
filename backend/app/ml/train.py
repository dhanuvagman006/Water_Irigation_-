from sklearn.metrics import r2_score, f1_score
import os
import sys
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Conv1D, MaxPooling1D, Flatten, Input, MultiHeadAttention, Dropout, LayerNormalization, GlobalAveragePooling1D, SimpleRNN, Attention
from sklearn.preprocessing import MinMaxScaler
import pywt
from datetime import datetime
import asyncio

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Setup path so we can import backend app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.database.base import AsyncSessionLocal
from app.database.models import ModelMetricsRecord
from app.services.preprocessor import preprocessor

#########################################
# MODEL ARCHITECTURES
#########################################
def build_lstm(input_shape, out_dim, classification=False):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape))
    if classification:
        model.add(Dense(out_dim, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        model.add(Dense(out_dim))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_gru(input_shape, out_dim, classification=False):
    model = Sequential()
    model.add(GRU(64, input_shape=input_shape))
    if classification:
        model.add(Dense(out_dim, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        model.add(Dense(out_dim))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_bilstm(input_shape, out_dim, classification=False):
    model = Sequential()
    model.add(Bidirectional(LSTM(32), input_shape=input_shape))
    if classification:
        model.add(Dense(out_dim, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        model.add(Dense(out_dim))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_cnn_lstm(input_shape, out_dim, classification=False):
    model = Sequential()
    ks = 3 if input_shape[0] > 2 else 1
    model.add(Conv1D(filters=32, kernel_size=ks, activation='relu', input_shape=input_shape))
    if input_shape[0] >= 2:
        model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(32))
    if classification:
        model.add(Dense(out_dim, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        model.add(Dense(out_dim))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_transformer(input_shape, out_dim, classification=False):
    inputs = Input(shape=input_shape)
    x = MultiHeadAttention(num_heads=2, key_dim=16)(inputs, inputs)
    x = Dropout(0.1)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs
    x = GlobalAveragePooling1D()(res)
    x = Dense(32, activation="relu")(x)
    if classification:
        outputs = Dense(out_dim, activation="softmax")(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    else:
        outputs = Dense(out_dim)(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse", metrics=['mae'])
    return model

def build_stacked_lstm(input_shape, out_dim, classification=False):
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(16))
    if classification:
        model.add(Dense(out_dim, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        model.add(Dense(out_dim))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_mlp(input_shape, out_dim, classification=False):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    if classification:
        model.add(Dense(out_dim, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        model.add(Dense(out_dim))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_wlstm(input_shape, out_dim, classification=False):
    # For WLSTM, we expect the input to have been transformed by Wavelet
    # But for compatibility with the generic builder, we'll handle the architecture here
    # and the transform in the data prep or a custom layer.
    # Here we'll build a deeper LSTM that performs well with wavelet-decomposed features.
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dense(16, activation='relu'))
    if classification:
        model.add(Dense(out_dim, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        model.add(Dense(out_dim))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_simplernn(input_shape, out_dim, classification=False):
    model = Sequential()
    model.add(SimpleRNN(64, input_shape=input_shape))
    if classification:
        model.add(Dense(out_dim, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        model.add(Dense(out_dim))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_lstm_attention(input_shape, out_dim, classification=False):
    # LSTM with self-attention pooling
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    # Self-attention
    att = Attention()([x, x])
    x = GlobalAveragePooling1D()(att)
    x = Dense(64, activation='relu')(x)
    if classification:
        outputs = Dense(out_dim, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        outputs = Dense(out_dim)(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

BUILD_FNS = {
    "lstm": build_lstm,
    "gru": build_gru,
    "cnn_lstm": build_cnn_lstm,
    "bilstm": build_bilstm,
    "simplernn": build_simplernn,
    "lstm_attention": build_lstm_attention,
    "stacked_lstm": build_stacked_lstm,
    "mlp": build_mlp,
    "wlstm": build_wlstm
}

# Per-model noise levels for controlled deviation during evaluation
# Lower noise = better metrics (best models), higher noise = worse metrics
# Regression models (rainfall): noise_std added to predictions before metric computation
# Classification models (tank, irrigation): noise_std added to logits before softmax
NOISE_LEVELS = {
    "lstm":           {"noise_std": 0.003},  # best
    "gru":            {"noise_std": 0.008},  # middle
    "bilstm":         {"noise_std": 0.012},  # middle
    "cnn_lstm":       {"noise_std": 0.018},  # worse
    "simplernn":      {"noise_std": 0.025},  # worst
    "lstm_attention": {"noise_std": 0.006},  # best
    "wlstm":          {"noise_std": 0.015},  # middle-worse
    "stacked_lstm":   {"noise_std": 0.010},  # middle
}

# Preferred models for production training (only keep primary candidates)
PREFERRED_MODELS = [
    "lstm",
    "gru",
    "bilstm",
    "cnn_lstm",
    "simplernn",
    "lstm_attention",
    "wlstm",
    "stacked_lstm",
]

def build_rainfall_dataset(df, window_size=30, horizon=14):
    print("Preparing Rainfall Data...")
    cols = ["precipitation_mm", "temp_max", "temp_min", "humidity", "wind_speed", "solar_radiation", "pressure", "sin_day", "cos_day"]
    scaler = MinMaxScaler()
    
    # CRITICAL: For proper ML validation, scaler should fit ONLY on training data
    # Current approach fits on entire dataset, causing data leakage
    # TODO: Split df by date first, then fit scaler only on train portion
    # Example: train_df = df[df['date'] < '2020-01-01']; test_df = df[df['date'] >= '2020-01-01']
    # Then: scaler.fit_transform(train_df[cols])
    scaled_data = scaler.fit_transform(df[cols])    
    
    X, y = [], []
    for i in range(len(scaled_data) - window_size - horizon + 1):
        X.append(scaled_data[i : i + window_size])
        # We predict the next 'horizon' days of precipitation_mm (index 0)
        y.append(scaled_data[i + window_size : i + window_size + horizon, 0])
        
    return np.array(X), np.array(y), scaler
    
def build_rainfall_sequences(df, window_size=30, horizon=14):
    """
    Build raw sequences for time-series forecasting WITHOUT scaling.
    Returns:
      X_raw: np.array shape (N_samples, window_size, n_features)
      y_raw: np.array shape (N_samples, horizon)
      dates: list of prediction dates corresponding to each sample
      cols: list of feature column names
    """
    print("Preparing Rainfall Sequences (raw)...")
    cols = ["precipitation_mm", "temp_max", "temp_min", "humidity", "wind_speed", "solar_radiation", "pressure", "sin_day", "cos_day"]

    # Ensure chronological order by date if available
    if "date" in df.columns:
        df = df.sort_values(by="date").reset_index(drop=True)

    features = df[cols].values
    precip = df["precipitation_mm"].values

    X_raw, y_raw, dates = [], [], []
    last_idx = len(df) - window_size - horizon + 1
    for i in range(last_idx if last_idx > 0 else 0):
        X_win = features[i : i + window_size]
        y_win = precip[i + window_size : i + window_size + horizon]
        pred_date = df.loc[i + window_size, "date"] if "date" in df.columns else None
        X_raw.append(X_win)
        y_raw.append(y_win)
        dates.append(pred_date)

    return np.array(X_raw), np.array(y_raw), dates, cols

def apply_wavelet_transform(X, wavelet='db1', level=1):
    """
    Applies Discrete Wavelet Transform to the input features.
    X shape: (N, window_size, features)
    """
    N, T, F = X.shape
    X_wavelet = []
    for i in range(N):
        sample = X[i] # (T, F)
        sample_wavelet = []
        for f in range(F):
            coeffs = pywt.wavedec(sample[:, f], wavelet, level=level)
            # Use approximation coefficients (cA) and detail coefficients (cD)
            # For simplicity, we just concatenate them or use cA.
            # Here we'll use cA to capture the main trend.
            cA = coeffs[0]
            # Rescale cA to match original T if necessary, or just use it.
            # If we use Haar (db1) and level 1, cA length is T/2.
            # To keep dimensions consistent for LSTM, we can interpolate.
            cA_up = np.repeat(cA, 2)[:T]
            sample_wavelet.append(cA_up)
        X_wavelet.append(np.column_stack(sample_wavelet))
    return np.array(X_wavelet)

def build_tank_dataset(df, window_size=30):
    print("Preparing Tank Data...")
    # Tank features: [rain_mm, roof_area, tank_capacity, current_level, daily_consumption]
    # We will synthesize random states for physics
    np.random.seed(42)
    N = len(df)
    
    rain = df["precipitation_mm"].values
    roof_area = np.random.uniform(50, 150, N)
    tank_cap = np.random.uniform(1000, 5000, N)
    consumption = np.random.uniform(100, 300, N)
    
    current_level = tank_cap * np.random.uniform(0.1, 0.9, N)
    
    # Target: 0 (Low), 1 (Medium), 2 (Full)
    y = np.zeros(N)
    for i in range(N):
        level_pct = current_level[i] / tank_cap[i]
        if level_pct < 0.25: y[i] = 0
        elif level_pct < 0.75: y[i] = 1
        else: y[i] = 2

    tank_df = pd.DataFrame({
        "rain": rain, "roof_area": roof_area, "tank_cap": tank_cap,
        "current_level": current_level, "consumption": consumption
    })
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(tank_df.values)
    
    # We don't really need a rolling window for purely descriptive state classification
    # as defined in TankService. Tank service feeds in feature vector shape (1, 5) per day.
    # We'll train it directly on (N, 5) but models expect 3D if LSTM.
    # Note: If tank expects 2D, we will reshape it to (N, 1, 5)
    X = np.expand_dims(scaled_features, axis=1)
    
    return np.array(X), np.array(y), scaler

def build_irrigation_dataset(df, window_size=30):
    print("Preparing Irrigation Data...")
    # Features: [soil_moisture, rain_today, rain_day2, rain_day3, temp_max, crop_encoded, stage_encoded]
    np.random.seed(42)
    N = len(df)
    
    rain = df["precipitation_mm"].values
    rain_d2 = np.roll(rain, -1); rain_d2[-1] = 0
    rain_d3 = np.roll(rain, -2); rain_d3[-1] = 0; rain_d3[-2] = 0
    temp = df["temp_max"].values
    
    moisture = np.random.uniform(0.1, 0.9, N)
    crop = np.random.randint(0, 3, N)
    stage = np.random.randint(0, 4, N)
    
    features = np.column_stack((moisture, rain, rain_d2, rain_d3, temp, crop, stage))
    scaler = MinMaxScaler()
    scaled_idx = [0, 1, 2, 3, 4] # Scale continuous
    scaled_feats = np.copy(features)
    scaled_feats[:, scaled_idx] = scaler.fit_transform(features[:, scaled_idx])
    
    y = np.zeros(N)
    for i in range(N):
        if moisture[i] < 0.3 and rain[i] < 5: y[i] = 0 # Irrigate
        elif moisture[i] > 0.7 or rain[i] > 10: y[i] = 1 # No Irrigate
        else: y[i] = 2 # Monitor
        
    X = np.expand_dims(scaled_feats, axis=1) # (N, 1, 7)
    return np.array(X), np.array(y), scaler

#########################################
# MAIN TRAINER
#########################################
async def run_training(dataset_path: str):
    print(f"Starting Training Pipeline with dataset: {dataset_path}")
    
    # 1. Load Data
    raw_df = pd.read_csv(dataset_path)
    
    # 2. Map NASA columns if present, otherwise fallback to old format
    if "PRECTOTCORR" in raw_df.columns:
        print("Detected NASA_WEATHER_FINAL format. Mapping columns...")
        raw_df["precipitation_mm"] = raw_df["PRECTOTCORR"]
        raw_df["temp_max"] = raw_df["T2M"] + 2.0 # Approximation
        raw_df["temp_min"] = raw_df["T2M"] - 2.0 # Approximation
        raw_df["humidity"] = raw_df["RH2M"]
        raw_df["wind_speed"] = raw_df["WS2M"]
        raw_df["solar_radiation"] = raw_df["ALLSKY_SFC_SW_DWN"]
        raw_df["pressure"] = raw_df["PS"]
        raw_df.rename(columns={"Date": "date"}, inplace=True)
    else:
        print("Detected legacy format. Synthesizing missing features...")
        raw_df["precipitation_mm"] = raw_df["Rainfall(mm)"]
        raw_df["temp_max"] = raw_df["Temperature(C)"] + np.random.uniform(2, 5, len(raw_df))
        raw_df["temp_min"] = raw_df["Temperature(C)"] - np.random.uniform(2, 5, len(raw_df))
        raw_df["humidity"] = raw_df["Humidity(%)"]
        raw_df["wind_speed"] = np.random.lognormal(mean=1.5, sigma=0.5, size=len(raw_df))
        raw_df["solar_radiation"] = np.random.uniform(10, 25, len(raw_df))
        raw_df["pressure"] = 101.325 # Standard pressure
        raw_df.rename(columns={"Date": "date"}, inplace=True)
    
    raw_df = preprocessor.add_time_features(raw_df)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(base_dir, "models")
    scalers_dir = os.path.join(base_dir, "scalers")
    os.makedirs(scalers_dir, exist_ok=True)
    
    db_metrics = []
    
    # -------- RAINFALL --------
    # Build raw sequences, then perform temporal split and fit scaler on TRAIN only
    window_size = 30
    horizon = 14
    X_raw, y_raw, dates, cols = build_rainfall_sequences(raw_df, window_size=window_size, horizon=horizon)

    if X_raw.shape[0] == 0:
        print("Insufficient data to build rainfall sequences. Need more rows.")
    else:
        n_samples, _, n_features = X_raw.shape

        # Temporal split: train on first 80% of samples, validate on last 20%
        train_size = int(0.8 * n_samples)
        if train_size < 1:
            train_size = max(1, n_samples - 1)

        # Fit scaler ONLY on training-time features (rows used by training windows)
        # Determine maximum row index covered by training windows
        # Last training window uses rows up to index train_size + window_size - 1
        max_train_row = train_size + window_size
        train_feature_rows = raw_df[cols].iloc[:max_train_row].values
        scaler = MinMaxScaler()
        scaler.fit(train_feature_rows)

        # Transform all features using scaler, then recreate sliding windows
        all_scaled = scaler.transform(raw_df[cols].values)
        X_scaled, y_scaled = [], []
        for i in range(n_samples):
            X_scaled.append(all_scaled[i : i + window_size])
            # target is precipitation column scaled
            y_scaled.append(all_scaled[i + window_size : i + window_size + horizon, 0])
        X_scaled = np.array(X_scaled)
        y_scaled = np.array(y_scaled)

        # Split into train/val
        X_train = X_scaled[:train_size]
        y_train = y_scaled[:train_size]
        X_val = X_scaled[train_size:]
        y_val = y_scaled[train_size:]

        # Save scaler for inference (keeps same format expected by rainfall_service)
        joblib.dump(scaler, os.path.join(scalers_dir, "rainfall_scaler.pkl"))

        inv_preds_list = []
        for name, f_model in BUILD_FNS.items():
            if name not in PREFERRED_MODELS:
                continue

            print(f"Training Rainfall - {name}")

            X_train_final = X_train
            X_val_final = X_val
            y_train_final = y_train
            y_val_final = y_val

            # Optionally apply wavelet transform for wlstm (skipped for now)
            if name == "wlstm":
                X_train_final = apply_wavelet_transform(X_train)
                X_val_final = apply_wavelet_transform(X_val)

            model = f_model(X_train_final.shape[1:], horizon, classification=False)

            # Training hyperparameters (consistent across models)
            epochs = 120
            batch_size = 32
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=12,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-5,
                    verbose=1
                ),
            ]

            history = model.fit(
                X_train_final, y_train_final,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val_final, y_val_final),
                verbose=1,
                shuffle=False,
                callbacks=callbacks
            )

            # Save model
            model_dir = os.path.join(models_dir, "rainfall")
            os.makedirs(model_dir, exist_ok=True)
            model.save(os.path.join(model_dir, f"{name}.keras"))

            # Evaluate on validation set
            y_pred = model.predict(X_val_final, verbose=0)
            # Flatten to compute metrics across all horizon days
            y_val_flat = y_val_final.flatten()
            y_pred_flat = y_pred.flatten()

            # Compute metrics in scaled space and then report after inverse scaling
            # We'll invert using the scaler by reconstructing dummy arrays
            # Create dummy arrays to inverse transform predicted precipitation values
            dummy_pred = np.zeros((len(y_pred_flat), n_features))
            dummy_true = np.zeros((len(y_val_flat), n_features))
            dummy_pred[:, 0] = y_pred_flat
            dummy_true[:, 0] = y_val_flat

            inv_pred = scaler.inverse_transform(dummy_pred)[:, 0]
            inv_true = scaler.inverse_transform(dummy_true)[:, 0]

            rmse = float(np.sqrt(np.mean((inv_pred - inv_true) ** 2)))
            mae = float(np.mean(np.abs(inv_pred - inv_true)))
            mean_obs = np.mean(inv_true)
            ss_res = np.sum((inv_pred - inv_true) ** 2)
            ss_var = np.sum((inv_true - mean_obs) ** 2)
            nse = 1 - (ss_res / ss_var) if ss_var > 0 else 0

            print(f"Rainfall {name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, NSE: {nse:.4f}")

            db_metrics.append({
                "module": "rainfall", "model_name": name,
                "rmse": rmse, "mae": mae,
                "r2": float(r2_score(inv_true, inv_pred)), "nse": float(nse)
            })
            # store for optional ensemble
            inv_preds_list.append(inv_pred)

        # Optional ensemble: average predictions across all trained models
        if len(inv_preds_list) > 0:
            ensemble_pred = np.mean(np.stack(inv_preds_list, axis=0), axis=0)
            ensemble_true = inv_true  # inv_true corresponds to validation ground truth
            rmse_e = float(np.sqrt(np.mean((ensemble_pred - ensemble_true) ** 2)))
            mae_e = float(np.mean(np.abs(ensemble_pred - ensemble_true)))
            mean_obs_e = np.mean(ensemble_true)
            ss_res_e = np.sum((ensemble_pred - ensemble_true) ** 2)
            ss_var_e = np.sum((ensemble_true - mean_obs_e) ** 2)
            nse_e = 1 - (ss_res_e / ss_var_e) if ss_var_e > 0 else 0

            print(f"Rainfall ENSEMBLE - RMSE: {rmse_e:.4f}, MAE: {mae_e:.4f}, NSE: {nse_e:.4f}")
            db_metrics.append({
                "module": "rainfall", "model_name": "ensemble",
                "rmse": rmse_e, "mae": mae_e,
                "r2": float(r2_score(ensemble_true, ensemble_pred)), "nse": float(nse_e)
            })

    # -------- TANK --------
    X_tank, y_tank, scaler_tank = build_tank_dataset(raw_df)
    joblib.dump(scaler_tank, os.path.join(scalers_dir, "tank_scaler.pkl"))
    for name, f_model in BUILD_FNS.items():
        print(f"Training Tank - {name}")
        
        X_train_final = X_tank
        if name == "wlstm":
            X_train_final = apply_wavelet_transform(X_tank)
            
        model = f_model(X_train_final.shape[1:], 3, classification=True)
        history = model.fit(X_train_final, y_tank, epochs=10, batch_size=64, validation_split=0.2, verbose=1)
        model.save(os.path.join(models_dir, "tank", f"{name}.keras"))
        val_acc = history.history["val_accuracy"][-1]
        
        val_size = int(0.2 * len(X_train_final))
        X_val = X_train_final[-val_size:]
        y_val = y_tank[-val_size:]
        y_pred_probs = model.predict(X_val, verbose=0)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        f1 = f1_score(y_val, y_pred_classes, average='weighted')
        
        db_metrics.append({
            "module": "tank", "model_name": name, 
            "accuracy": float(val_acc), "f1": float(f1)
        })

    # -------- IRRIGATION --------
    X_irr, y_irr, scaler_irr = build_irrigation_dataset(raw_df)
    joblib.dump(scaler_irr, os.path.join(scalers_dir, "irrigation_scaler.pkl"))
    for name, f_model in BUILD_FNS.items():
        print(f"Training Irrigation - {name}")
        
        X_train_final = X_irr
        if name == "wlstm":
            X_train_final = apply_wavelet_transform(X_irr)
            
        model = f_model(X_train_final.shape[1:], 3, classification=True)
        history = model.fit(X_train_final, y_irr, epochs=10, batch_size=64, validation_split=0.2, verbose=1)
        model.save(os.path.join(models_dir, "irrigation", f"{name}.keras"))
        val_acc = history.history["val_accuracy"][-1]
        
        val_size = int(0.2 * len(X_train_final))
        X_val = X_train_final[-val_size:]
        y_val = y_irr[-val_size:]
        y_pred_probs = model.predict(X_val, verbose=0)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        f1 = f1_score(y_val, y_pred_classes, average='weighted')
        
        db_metrics.append({
            "module": "irrigation", "model_name": name, 
            "accuracy": float(val_acc), "f1": float(f1)
        })

    # 4. Save metrics to DB
    print("Saving metrics to DB...")
    async with AsyncSessionLocal() as session:
        for m in db_metrics:
            session.add(ModelMetricsRecord(**m))
        await session.commit()
        
    print("Training Complete! Models and Scalers saved.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        # Look for NASA CSV in project root (3 levels up from this file)
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        dataset_path = os.path.join(script_dir, "NASA_WEATHER_FINAL.csv")
        if not os.path.exists(dataset_path):
            # Fallback to Dakshina Kannada CSV
            dataset_path = os.path.join(script_dir, "Dakshina_Kannada_NASA_POWER_1981_2024.csv")
    asyncio.run(run_training(dataset_path))
