"""
Database session management.
"""

from contextlib import contextmanager, asynccontextmanager
from typing import Generator, AsyncGenerator, Optional

from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .engine import get_engine, get_async_engine

# Session factories
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=get_engine(),
    expire_on_commit=False,
)

AsyncSessionLocal = async_sessionmaker(
    bind=get_async_engine(),
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Get database session context manager for synchronous operations.
    
    Yields:
        SQLAlchemy Session instance
    """
    db: Optional[Session] = None
    try:
        db = SessionLocal()
        yield db
        db.commit()
    except Exception:
        if db:
            db.rollback()
        raise
    finally:
        if db:
            db.close()


@asynccontextmanager
async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Get async database session context manager.
    
    Yields:
        AsyncSession instance
    """
    db: Optional[AsyncSession] = None
    try:
        db = AsyncSessionLocal()
        yield db
        await db.commit()
    except Exception:
        if db:
            await db.rollback()
        raise
    finally:
        if db:
            await db.close()


def get_db_session() -> Session:
    """
    Get a database session (manually manage commit/rollback).
    
    Returns:
        SQLAlchemy Session instance
        
    Note: Caller must handle commit/rollback and close.
    """
    return SessionLocal()


async def get_async_db_session() -> AsyncSession:
    """
    Get an async database session (manually manage commit/rollback).
    
    Returns:
        AsyncSession instance
        
    Note: Caller must handle commit/rollback and close.
    """
    return AsyncSessionLocal()


# FastAPI dependency versions
def get_db_fastapi():
    """FastAPI dependency for synchronous database."""
    with get_db() as db:
        yield db


async def get_async_db_fastapi():
    """FastAPI dependency for async database."""
    async with get_async_db() as db:
        yield db