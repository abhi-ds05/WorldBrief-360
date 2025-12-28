"""
Database initialization and setup.
"""

import logging
from typing import List, Optional
from sqlalchemy import inspect, text

from .base import Base
from .engine import get_engine, get_async_engine
from .session import SessionLocal
from .models import *  # Import all models to register them with Base

logger = logging.getLogger(__name__)


def create_tables() -> None:
    """
    Create all database tables.
    """
    engine = get_engine()
    
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
        # Create extensions if using PostgreSQL
        inspector = inspect(engine)
        if inspector.dialect.name == 'postgresql':
            with engine.connect() as conn:
                # Enable UUID extension
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\""))
                # Enable case-insensitive text extension
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"citext\""))
                # Enable JSONB indexing
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"btree_gin\""))
                conn.commit()
                logger.info("PostgreSQL extensions created")
                
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        raise


async def create_tables_async() -> None:
    """
    Create all database tables asynchronously.
    """
    engine = get_async_engine()
    
    try:
        async with engine.begin() as conn:
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully (async)")
            
            # Create extensions if using PostgreSQL
            inspector = inspect(engine.sync_engine)
            if inspector.dialect.name == 'postgresql':
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\""))
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"citext\""))
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"btree_gin\""))
                logger.info("PostgreSQL extensions created (async)")
                
    except Exception as e:
        logger.error(f"Failed to create tables (async): {e}")
        raise


def drop_tables() -> None:
    """
    Drop all database tables.
    Warning: This will delete all data!
    """
    engine = get_engine()
    
    try:
        Base.metadata.drop_all(bind=engine)
        logger.warning("Database tables dropped")
    except Exception as e:
        logger.error(f"Failed to drop tables: {e}")
        raise


async def drop_tables_async() -> None:
    """
    Drop all database tables asynchronously.
    Warning: This will delete all data!
    """
    engine = get_async_engine()
    
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            logger.warning("Database tables dropped (async)")
    except Exception as e:
        logger.error(f"Failed to drop tables (async): {e}")
        raise


def reset_db() -> None:
    """
    Reset database (drop and recreate all tables).
    Warning: This will delete all data!
    """
    logger.warning("Resetting database...")
    drop_tables()
    create_tables()
    logger.info("Database reset complete")


async def reset_db_async() -> None:
    """
    Reset database asynchronously.
    Warning: This will delete all data!
    """
    logger.warning("Resetting database (async)...")
    await drop_tables_async()
    await create_tables_async()
    logger.info("Database reset complete (async)")


def init_db() -> None:
    """
    Initialize database (create tables if they don't exist).
    """
    engine = get_engine()
    inspector = inspect(engine)
    
    # Check if any tables exist
    existing_tables = inspector.get_table_names()
    
    if not existing_tables:
        create_tables()
        logger.info("Database initialized with tables")
    else:
        # Check if our tables exist
        our_tables = [table.__tablename__ for table in Base.metadata.tables.values()]
        missing_tables = [table for table in our_tables if table not in existing_tables]
        
        if missing_tables:
            logger.info(f"Creating missing tables: {missing_tables}")
            create_tables()
        else:
            logger.info("Database tables already exist")


async def init_db_async() -> None:
    """
    Initialize database asynchronously.
    """
    engine = get_async_engine()
    inspector = inspect(engine.sync_engine)
    
    # Check if any tables exist
    existing_tables = inspector.get_table_names()
    
    if not existing_tables:
        await create_tables_async()
        logger.info("Database initialized with tables (async)")
    else:
        # Check if our tables exist
        our_tables = [table.__tablename__ for table in Base.metadata.tables.values()]
        missing_tables = [table for table in our_tables if table not in existing_tables]
        
        if missing_tables:
            logger.info(f"Creating missing tables (async): {missing_tables}")
            await create_tables_async()
        else:
            logger.info("Database tables already exist (async)")


def check_db_health() -> bool:
    """
    Check database health and connectivity.
    
    Returns:
        True if database is healthy
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            # Simple query to check connectivity
            result = conn.execute(text("SELECT 1"))
            return result.scalar() == 1
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


async def check_db_health_async() -> bool:
    """
    Check database health asynchronously.
    
    Returns:
        True if database is healthy
    """
    try:
        engine = get_async_engine()
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            return await result.scalar() == 1
    except Exception as e:
        logger.error(f"Database health check failed (async): {e}")
        return False


def get_table_info() -> List[dict]:
    """
    Get information about all database tables.
    
    Returns:
        List of table information dictionaries
    """
    engine = get_engine()
    inspector = inspect(engine)
    
    tables_info = []
    
    for table_name in inspector.get_table_names():
        table_info = {
            'name': table_name,
            'columns': [],
            'indexes': inspector.get_indexes(table_name),
            'foreign_keys': inspector.get_foreign_keys(table_name),
            'primary_key': inspector.get_pk_constraint(table_name),
        }
        
        for column in inspector.get_columns(table_name):
            table_info['columns'].append({
                'name': column['name'],
                'type': str(column['type']),
                'nullable': column['nullable'],
                'default': column['default'],
                'primary_key': column.get('primary_key', False),
            })
        
        tables_info.append(table_info)
    
    return tables_info