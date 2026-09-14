from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # PostgreSQL in production (Render/Railway provide DATABASE_URL directly);
    # falls back to a local SQLite file so `uvicorn app.main:app` works with
    # zero external setup for development.
    database_url: str = "sqlite:///./packsure.db"

    jwt_secret: str = "change-this-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24

    cors_origins: str = "http://localhost:5173"

    # DistilBERT is an optional ensemble signal (see field_classifier.py) --
    # not required for the pipeline to function, and a ~650MB download.
    # Opt-in only.
    enable_ner_backbone: bool = False

    class Config:
        env_file = ".env"


settings = Settings()
