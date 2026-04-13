from pydantic_settings import BaseSettings, SettingsConfigDict
from logging import getLogger

logger = getLogger(__name__)

class Settings(BaseSettings):
    APP_NAME: str = "AquaAI Backend"
    API_KEY: str = "dev_super_secret_key_123"
    DATABASE_URL: str = "sqlite+aiosqlite:///./aquaai.db"
    MODELS_DIR: str = "./models"
    SCALERS_DIR: str = "./scalers"
    DATA_DIR: str = "./data"
    NASA_LAT: float = 12.87
    NASA_LON: float = 74.88
    DEFAULT_MODEL: str = "LSTM"
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    REDIS_URL: str = ""
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
