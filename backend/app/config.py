from dotenv import load_dotenv

# pydantic-settings reads .env into `settings` but does NOT export it to
# os.environ. vlm_extractor.py / field_classifier.py / quality_gate.py read
# OLLAMA_URL, VLM_MODEL_NAME, ENABLE_VLM, ... straight from os.environ, so
# load .env into the process environment first.
load_dotenv()

from pydantic_settings import BaseSettings  # noqa: E402


class Settings(BaseSettings):
    # PostgreSQL in production; falls back to local SQLite for zero-setup dev.
    database_url: str = "sqlite:///./packsure.db"

    jwt_secret: str = "change-this-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24

    cors_origins: str = "http://localhost:5173"

    # Load PaddleOCR / YOLO / VLM probe at startup (FastAPI lifespan) so the
    # first scan is not a cold start. Set false for quick UI-only dev.
    preload_models: bool = True

    # The variables below are consumed via os.environ by the pipeline modules;
    # they are declared here so they are documented and validated in one place.
    enable_vlm: bool = True
    ollama_url: str = "http://localhost:11434"
    vlm_model_name: str = "qwen2.5vl:3b"
    vlm_timeout_seconds: int = 120
    vlm_max_side: int = 1280
    blur_variance_threshold: float = 40.0

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
