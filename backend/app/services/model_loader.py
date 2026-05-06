import os
import logging
from fastapi import HTTPException
import joblib
from app.config import settings

logger = logging.getLogger(__name__)

HORIZON_SUFFIXES = ["1d", "7d", "15d"]

class ModelLoader:
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.models: dict = {}
        self.scalers: dict = {}
        self.load_errors: dict[str, str] = {}
        self.expected_models = [
            "lstm", "gru", "bilstm",
            "cnn_lstm", "simplernn", "wlstm"
        ]
        self.modules = ["rainfall"]

    async def load_all(self):
        if not settings.LOAD_MODELS:
            logger.info("Model loading disabled; prediction endpoints will use fallback logic where available.")
            self.load_errors = {
                f"{module}/{name}": "model loading disabled"
                for module in self.modules
                for name in self.expected_models
            }
            return

        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf

        tf.get_logger().setLevel(logging.ERROR)
        logger.info("Starting model loading sequence...")
        loaded_count = 0
        self.load_errors.clear()
        for module in self.modules:
            module_dir = os.path.join(self.models_dir, module)
            if not os.path.exists(module_dir):
                logger.warning("Model module directory missing: %s", module_dir)
                continue

            if module == "rainfall":
                for name in self.expected_models:
                    loaded_any = False
                    for suffix in HORIZON_SUFFIXES:
                        file_path = os.path.join(module_dir, f"{name}_{suffix}.keras")
                        key = f"{module}/{name}_{suffix}"
                        try:
                            if os.path.exists(file_path):
                                model = tf.keras.models.load_model(file_path)
                                self.models[key] = model
                                loaded_count += 1
                                loaded_any = True
                                logger.info("Loaded model: %s", key)
                            else:
                                self.load_errors[key] = "model file not found"
                        except Exception as e:
                            self.load_errors[key] = str(e).splitlines()[0]
                    if not loaded_any:
                        base_path = os.path.join(module_dir, f"{name}.keras")
                        if os.path.exists(base_path):
                            try:
                                model = tf.keras.models.load_model(base_path)
                                for suffix in HORIZON_SUFFIXES:
                                    key = f"{module}/{name}_{suffix}"
                                    self.models[key] = model
                                    loaded_count += 1
                                    loaded_any = True
                                    self.load_errors.pop(key, None)
                                    logger.info("Loaded model (no horizon suffix): %s -> %s", name, key)
                            except Exception as e:
                                for suffix in HORIZON_SUFFIXES:
                                    key = f"{module}/{name}_{suffix}"
                                    self.load_errors[key] = str(e).splitlines()[0]
                    if not loaded_any:
                        for suffix in HORIZON_SUFFIXES:
                            key = f"{module}/{name}_{suffix}"
                            if key not in self.load_errors:
                                self.load_errors[key] = "model file not found"
            else:
                for name in self.expected_models:
                    file_path = os.path.join(module_dir, f"{name}.keras")
                    key = f"{module}/{name}"
                    try:
                        if os.path.exists(file_path):
                            model = tf.keras.models.load_model(file_path)
                            self.models[key] = model
                            loaded_count += 1
                            logger.info("Loaded model: %s", key)
                        else:
                            self.load_errors[key] = "model file not found"
                    except Exception as e:
                        self.load_errors[key] = str(e).splitlines()[0]

        total_rainfall = len(self.expected_models) * len(HORIZON_SUFFIXES)
        total_other = len(self.expected_models) * (len(self.modules) - 1)
        expected_count = total_rainfall + total_other
        logger.info("Loaded %s/%s models.", loaded_count, expected_count)
        if self.load_errors:
            logger.warning(
                "%s model artifact(s) could not be loaded; prediction endpoints will use fallback logic where available.",
                len(self.load_errors),
            )
            logger.debug("Model load errors: %s", self.load_errors)

    def get_model(self, module: str, name: str, horizon: str = "medium"):
        normalized_name = name.strip().lower()
        if normalized_name == "cnn-lstm":
            normalized_name = "cnn_lstm"
        elif normalized_name == "stackedlstm":
            normalized_name = "stacked_lstm"
        elif normalized_name == "lstm+attention":
            normalized_name = "lstm_attention"
        elif normalized_name == "simplernn":
            normalized_name = "simplernn"

        horizon_map = {"short": "1d", "medium": "7d", "long": "15d"}
        suffix = horizon_map.get(horizon, "7d")

        if module == "rainfall":
            key = f"{module}/{normalized_name}_{suffix}"
        else:
            key = f"{module}/{normalized_name}"

        if key not in self.models:
            raise HTTPException(status_code=404, detail=f"Model {key} not loaded")
        return self.models[key]

    async def load_scalers(self, scalers_dir: str):
        logger.info("Starting scaler loading sequence...")
        self.scalers.clear()
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

    def validate_required(self):
        if not settings.LOAD_MODELS:
            logger.info("Model validation skipped because LOAD_MODELS is disabled.")
            return

        missing = []
        for module in self.modules:
            if module == "rainfall":
                for name in self.expected_models:
                    for suffix in HORIZON_SUFFIXES:
                        key = f"{module}/{name}_{suffix}"
                        if key not in self.models:
                            missing.append(key)
            else:
                for name in self.expected_models:
                    key = f"{module}/{name}"
                    if key not in self.models:
                        missing.append(key)

        for module in self.modules:
            if module not in self.scalers:
                missing.append(f"{module}_scaler.pkl")

        if missing:
            raise RuntimeError(
                "Missing required model/scaler artifacts: " + ", ".join(sorted(missing))
            )

model_loader = ModelLoader(settings.MODELS_DIR)
