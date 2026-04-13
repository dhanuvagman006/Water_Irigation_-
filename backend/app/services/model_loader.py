import os
import logging
from fastapi import HTTPException
import joblib
import tensorflow as tf
from app.config import settings

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.models: dict = {}
        self.scalers: dict = {}
        self.expected_models = [
            "lstm", "gru", "bilstm", "cnn_lstm", "transformer", "stacked_lstm"
        ]
        self.modules = ["rainfall", "tank", "irrigation"]

    async def load_all(self):
        logger.info("Starting model loading sequence...")
        loaded_count = 0
        for module in self.modules:
            module_dir = os.path.join(self.models_dir, module)
            if not os.path.exists(module_dir):
                logger.warning(f"Module directory {module_dir} missing.")
                continue
                
            for name in self.expected_models:
                file_path = os.path.join(module_dir, f"{name}.keras")
                key = f"{module}/{name}"
                try:
                    if os.path.exists(file_path):
                        # Load Keras model
                        model = tf.keras.models.load_model(file_path)
                        self.models[key] = model
                        loaded_count += 1
                        logger.info(f"Loaded model: {key}")
                    else:
                        logger.warning(f"Model file {file_path} not found.")
                except Exception as e:
                    logger.error(f"Failed to load model {key}: {e}")
        
        logger.info(f"Loaded {loaded_count}/{len(self.expected_models)*3} models.")

    def get_model(self, module: str, name: str):
        # Normalize name to match internal keys (e.g. CNN-LSTM -> cnn_lstm, StackedLSTM -> stacked_lstm)
        normalized_name = name.strip().lower()
        if normalized_name == "cnn-lstm":
            normalized_name = "cnn_lstm"
        elif normalized_name == "stackedlstm":
            normalized_name = "stacked_lstm"
            
        key = f"{module}/{normalized_name}"
        if key not in self.models:
            raise HTTPException(status_code=404, detail=f"Model {key} not loaded")
        return self.models[key]

    async def load_scalers(self, scalers_dir: str):
        logger.info("Starting scaler loading sequence...")
        for module in self.modules:
            file_path = os.path.join(scalers_dir, f"{module}_scaler.pkl")
            try:
                if os.path.exists(file_path):
                    scaler = joblib.load(file_path)
                    self.scalers[module] = scaler
                    logger.info(f"Loaded scaler for {module}")
                else:
                    logger.warning(f"Scaler file {file_path} not found.")
            except Exception as e:
                logger.error(f"Failed to load scaler {module}: {e}")

    def get_scaler(self, module: str):
        if module not in self.scalers:
            raise HTTPException(status_code=404, detail=f"Scaler for {module} not loaded")
        return self.scalers[module]

model_loader = ModelLoader(settings.MODELS_DIR)
