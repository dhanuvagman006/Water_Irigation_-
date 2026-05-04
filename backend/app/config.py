from logging import getLogger
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[1]


def _backend_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str(BACKEND_DIR / candidate)


def _sqlite_url(path: str) -> str:
    return f"sqlite+aiosqlite:///{Path(path).as_posix()}"


class Settings(BaseSettings):
    APP_NAME: str = "AquaAI Backend"
    API_KEY: str = "dev_super_secret_key_123"
    DATABASE_URL: str = _sqlite_url(BACKEND_DIR / "aquaai.db")
    MODELS_DIR: str = _backend_path("models")
    SCALERS_DIR: str = _backend_path("scalers")
    DATA_DIR: str = _backend_path("data")
    NASA_LAT: float = 12.87
    NASA_LON: float = 74.88
    DEFAULT_MODEL: str = "LSTM"
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8001",
        "http://127.0.0.1:8001",
    ]
    REDIS_URL: str = ""
    LOG_LEVEL: str = "INFO"
    LOAD_MODELS: bool = True

    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()

if settings.DATABASE_URL == "sqlite+aiosqlite:///./aquaai.db":
    settings.DATABASE_URL = _sqlite_url(BACKEND_DIR / "aquaai.db")

settings.MODELS_DIR = _backend_path(settings.MODELS_DIR)
settings.SCALERS_DIR = _backend_path(settings.SCALERS_DIR)
settings.DATA_DIR = _backend_path(settings.DATA_DIR)
