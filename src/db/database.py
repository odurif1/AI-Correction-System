"""Database configuration and session management."""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from pathlib import Path
from config.constants import DATA_DIR

# Database path - stored in configured data directory
DATA_DIR_PATH = Path(DATA_DIR)
DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("AI_CORRECTION_DATABASE_URL", "").strip()
if not DATABASE_URL:
    DATABASE_URL = f"sqlite:///{DATA_DIR_PATH / 'app.db'}"

engine_kwargs = {"pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL,
    **engine_kwargs,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    from db.models import User, UsageRecord, SessionJob, SessionJobEvent  # noqa: F401
    Base.metadata.create_all(bind=engine)
