"""
Database engine configuration for PostgreSQL.
"""

import os
from typing import Optional
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool

from core.config import get_config, DatabaseType

# Global engine instances
_sync_engine: Optional[Engine] = None
_async_engine: Optional[AsyncEngine] = None


def get_engine() -> Engine:
    """
    Get synchronous SQLAlchemy engine.
    
    Returns:
        SQLAlchemy engine instance
    """
    global _sync_engine
    
    if _sync_engine is None:
        config = get_config().database
        
        if config.type == DatabaseType.POSTGRESQL:
            # PostgreSQL connection string
            if config.url:
                database_url = config.url
            else:
                database_url = (
                    f"postgresql://{config.user}:{config.password}"
                    f"@{config.host}:{config.port}/{config.name}"
                )
        elif config.type == DatabaseType.SQLITE:
            # SQLite connection string
            database_url = f"sqlite:///{config.name}"
        elif config.type == DatabaseType.MYSQL:
            # MySQL connection string
            database_url = (
                f"mysql+pymysql://{config.user}:{config.password}"
                f"@{config.host}:{config.port}/{config.name}"
            )
        else:
            raise ValueError(f"Unsupported database type: {config.type}")
        
        _sync_engine = create_engine(
            database_url,
            echo=config.echo,
            pool_size=config.pool_size,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
            poolclass=QueuePool,
            connect_args={
                "connect_timeout": 10,
                "application_name": get_config().app_name,
            } if config.type == DatabaseType.POSTGRESQL else {}
        )
    
    return _sync_engine


def get_async_engine() -> AsyncEngine:
    """
    Get asynchronous SQLAlchemy engine.
    
    Returns:
        AsyncEngine instance
    """
    global _async_engine
    
    if _async_engine is None:
        config = get_config().database
        
        if config.type == DatabaseType.POSTGRESQL:
            # PostgreSQL async connection string
            if config.url:
                # Convert postgresql:// to postgresql+asyncpg://
                database_url = config.url.replace(
                    'postgresql://', 'postgresql+asyncpg://', 1
                )
            else:
                database_url = (
                    f"postgresql+asyncpg://{config.user}:{config.password}"
                    f"@{config.host}:{config.port}/{config.name}"
                )
        elif config.type == DatabaseType.SQLITE:
            # SQLite async connection string
            database_url = f"sqlite+aiosqlite:///{config.name}"
        elif config.type == DatabaseType.MYSQL:
            # MySQL async connection string
            database_url = (
                f"mysql+aiomysql://{config.user}:{config.password}"
                f"@{config.host}:{config.port}/{config.name}"
            )
        else:
            raise ValueError(f"Unsupported database type: {config.type}")
        
        _async_engine = create_async_engine(
            database_url,
            echo=config.echo,
            pool_size=config.pool_size,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
            poolclass=QueuePool,
            connect_args={
                "command_timeout": 30,
                "server_settings": {
                    "application_name": get_config().app_name,
                }
            } if config.type == DatabaseType.POSTGRESQL else {}
        )
    
    return _async_engine


def test_connection() -> bool:
    """
    Test database connection.
    
    Returns:
        True if connection successful
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False


async def test_async_connection() -> bool:
    """
    Test async database connection.
    
    Returns:
        True if connection successful
    """
    try:
        engine = get_async_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"Async database connection failed: {e}")
        return False


def close_engines() -> None:
    """Close all database engines."""
    global _sync_engine, _async_engine
    
    if _sync_engine:
        _sync_engine.dispose()
        _sync_engine = None
    
    if _async_engine:
        # Async engine disposal is different
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule disposal
                asyncio.create_task(_async_engine.dispose())
            else:
                # Run disposal in current loop
                loop.run_until_complete(_async_engine.dispose())
        except Exception:
            pass
        finally:
            _async_engine = None


# Convenience exports
engine = get_engine()
async_engine = get_async_engine()