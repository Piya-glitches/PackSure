from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.config import settings

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    # Hackathon-scope schema management: create tables directly from models.
    # Production roadmap: replace with Alembic migrations for versioned,
    # zero-downtime schema changes.
    from app import models  # noqa: F401  (ensures models are registered)

    Base.metadata.create_all(bind=engine)
