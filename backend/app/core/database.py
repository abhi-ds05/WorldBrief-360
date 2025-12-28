"""
Database management and utilities.
Supports multiple database backends (PostgreSQL, MySQL, SQLite, MongoDB).
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from enum import Enum

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    Integer,
    String,
    DateTime,
    Boolean,
    Text,
    ForeignKey,
    select,
    insert,
    update,
    delete,
    func,
    text,
    inspect
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import aioredis
from redis.exceptions import RedisError

from .config import DatabaseConfig, get_config, DatabaseType
from .exceptions import DatabaseError, ValidationError
from .logging_config import get_logger

logger = get_logger(__name__)


class DatabaseBackend(str, Enum):
    """Supported database backends."""
    SQLALCHEMY = "sqlalchemy"
    MONGODB = "mongodb"
    REDIS = "redis"


class ConnectionStatus(str, Enum):
    """Database connection status."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"


class DatabaseHealth:
    """Database health status."""
    
    def __init__(self):
        self.status = ConnectionStatus.DISCONNECTED
        self.last_check: Optional[datetime] = None
        self.latency: Optional[float] = None
        self.error: Optional[str] = None
        self.version: Optional[str] = None
        self.tables_count: Optional[int] = None
        self.connections_count: Optional[int] = None


class BaseModel:
    """Base model with common fields and methods."""
    
    def to_dict(self, exclude: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Convert model to dictionary.
        
        Args:
            exclude: Fields to exclude
            
        Returns:
            Dictionary representation
        """
        exclude = exclude or []
        result = {}
        
        for column in self.__table__.columns:
            if column.name not in exclude:
                value = getattr(self, column.name)
                if isinstance(value, datetime):
                    result[column.name] = value.isoformat()
                else:
                    result[column.name] = value
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Any:
        """
        Create model instance from dictionary.
        
        Args:
            data: Dictionary data
            
        Returns:
            Model instance
        """
        return cls(**data)
    
    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """
        Update model instance from dictionary.
        
        Args:
            data: Dictionary data
        """
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)


# SQLAlchemy Base
Base = declarative_base(cls=BaseModel)


class DatabaseConnection:
    """
    Base database connection class.
    
    Attributes:
        config: Database configuration
        engine: Database engine
        session_factory: Session factory
        health: Health status
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize database connection.
        
        Args:
            config: Database configuration
        """
        self.config = config or get_config().database
        self.engine: Optional[Any] = None
        self.session_factory: Optional[Any] = None
        self.health = DatabaseHealth()
        self._connection_pool: Optional[Any] = None
        self._connected = False
        
        # Statistics
        self._stats = {
            "queries_executed": 0,
            "transactions": 0,
            "errors": 0,
            "connection_time": None,
        }
    
    async def connect(self) -> None:
        """
        Establish database connection.
        
        Raises:
            DatabaseError: If connection fails
        """
        raise NotImplementedError
    
    async def disconnect(self) -> None:
        """Close database connection."""
        raise NotImplementedError
    
    async def health_check(self) -> DatabaseHealth:
        """
        Perform health check.
        
        Returns:
            DatabaseHealth object
        """
        raise NotImplementedError
    
    async def execute(self, query: Any, **kwargs) -> Any:
        """
        Execute database query.
        
        Args:
            query: Query to execute
            **kwargs: Additional parameters
            
        Returns:
            Query result
            
        Raises:
            DatabaseError: If query execution fails
        """
        raise NotImplementedError
    
    async def execute_many(self, queries: List[Any]) -> List[Any]:
        """
        Execute multiple queries.
        
        Args:
            queries: List of queries
            
        Returns:
            List of results
        """
        results = []
        for query in queries:
            try:
                result = await self.execute(query)
                results.append(result)
            except Exception as e:
                results.append(e)
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with statistics
        """
        return self._stats.copy()
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = {
            "queries_executed": 0,
            "transactions": 0,
            "errors": 0,
            "connection_time": datetime.now(),
        }


class SQLAlchemyConnection(DatabaseConnection):
    """SQLAlchemy database connection."""
    
    async def connect(self) -> None:
        """Establish SQLAlchemy connection."""
        try:
            if self.config.type == DatabaseType.SQLITE:
                # SQLite async requires aiosqlite
                database_url = f"sqlite+aiosqlite:///{self.config.name}"
            elif self.config.type == DatabaseType.POSTGRESQL:
                database_url = f"postgresql+asyncpg://{self.config.user}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.name}"
            elif self.config.type == DatabaseType.MYSQL:
                database_url = f"mysql+aiomysql://{self.config.user}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.name}"
            else:
                raise DatabaseError(f"Unsupported database type: {self.config.type}")
            
            # Create async engine
            self.engine = create_async_engine(
                database_url,
                echo=self.config.echo,
                pool_size=self.config.pool_size,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=3600,
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            
            # Test connection
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            
            self._connected = True
            self.health.status = ConnectionStatus.CONNECTED
            self._stats["connection_time"] = datetime.now()
            
            logger.info(f"Connected to {self.config.type} database: {self.config.name}")
            
        except Exception as e:
            self._connected = False
            self.health.status = ConnectionStatus.ERROR
            self.health.error = str(e)
            raise DatabaseError(f"Failed to connect to database: {e}")
    
    async def disconnect(self) -> None:
        """Close SQLAlchemy connection."""
        if self.engine:
            await self.engine.dispose()
            self.engine = None
            self.session_factory = None
            self._connected = False
            self.health.status = ConnectionStatus.DISCONNECTED
            logger.info("Disconnected from database")
    
    async def health_check(self) -> DatabaseHealth:
        """Perform health check."""
        self.health.last_check = datetime.now()
        
        try:
            start_time = datetime.now()
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            end_time = datetime.now()
            
            self.health.latency = (end_time - start_time).total_seconds() * 1000
            self.health.status = ConnectionStatus.CONNECTED
            self.health.error = None
            
            # Get database version
            result = await conn.execute(text("SELECT version()"))
            version = await result.scalar()
            self.health.version = str(version) if version else "unknown"
            
            # Get tables count (for SQL databases)
            inspector = inspect(self.engine)
            tables = await inspector.get_table_names()
            self.health.tables_count = len(tables)
            
        except Exception as e:
            self.health.status = ConnectionStatus.ERROR
            self.health.error = str(e)
            self._stats["errors"] += 1
        
        return self.health
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get database session context manager.
        
        Yields:
            AsyncSession instance
            
        Raises:
            DatabaseError: If session creation fails
        """
        if not self.session_factory:
            raise DatabaseError("Database not connected")
        
        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            self._stats["errors"] += 1
            raise DatabaseError(f"Database session error: {e}")
        finally:
            await session.close()
    
    async def execute(self, query: Any, **kwargs) -> Any:
        """
        Execute database query.
        
        Args:
            query: SQLAlchemy query
            **kwargs: Additional parameters
            
        Returns:
            Query result
        """
        async with self.get_session() as session:
            try:
                result = await session.execute(query, **kwargs)
                self._stats["queries_executed"] += 1
                return result
            except IntegrityError as e:
                raise DatabaseError(f"Integrity error: {e}", details={"error": "integrity_error"})
            except OperationalError as e:
                raise DatabaseError(f"Operational error: {e}", details={"error": "operational_error"})
            except SQLAlchemyError as e:
                raise DatabaseError(f"SQLAlchemy error: {e}", details={"error": "sqlalchemy_error"})
    
    async def execute_raw(self, sql: str, params: Optional[Dict] = None) -> Any:
        """
        Execute raw SQL query.
        
        Args:
            sql: SQL query string
            params: Query parameters
            
        Returns:
            Query result
        """
        async with self.get_session() as session:
            try:
                result = await session.execute(text(sql), params or {})
                self._stats["queries_executed"] += 1
                return result
            except SQLAlchemyError as e:
                raise DatabaseError(f"Raw SQL execution error: {e}")
    
    async def create_all(self, base: Any = Base) -> None:
        """
        Create all tables.
        
        Args:
            base: SQLAlchemy Base class
        """
        async with self.engine.begin() as conn:
            await conn.run_sync(base.metadata.create_all)
        logger.info("Database tables created")
    
    async def drop_all(self, base: Any = Base) -> None:
        """
        Drop all tables.
        
        Args:
            base: SQLAlchemy Base class
        """
        async with self.engine.begin() as conn:
            await conn.run_sync(base.metadata.drop_all)
        logger.warning("Database tables dropped")


class MongoDBConnection(DatabaseConnection):
    """MongoDB database connection."""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        super().__init__(config)
        self.client: Optional[MongoClient] = None
        self.db: Optional[Any] = None
    
    async def connect(self) -> None:
        """Establish MongoDB connection."""
        try:
            if self.config.url:
                self.client = MongoClient(self.config.url)
            else:
                self.client = MongoClient(
                    host=self.config.host,
                    port=self.config.port,
                    username=self.config.user,
                    password=self.config.password,
                    authSource=self.config.name,
                )
            
            self.db = self.client[self.config.name]
            
            # Test connection
            self.client.admin.command('ping')
            
            self._connected = True
            self.health.status = ConnectionStatus.CONNECTED
            self._stats["connection_time"] = datetime.now()
            
            logger.info(f"Connected to MongoDB: {self.config.name}")
            
        except PyMongoError as e:
            self._connected = False
            self.health.status = ConnectionStatus.ERROR
            self.health.error = str(e)
            raise DatabaseError(f"Failed to connect to MongoDB: {e}")
    
    async def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            self._connected = False
            self.health.status = ConnectionStatus.DISCONNECTED
            logger.info("Disconnected from MongoDB")
    
    async def health_check(self) -> DatabaseHealth:
        """Perform health check."""
        self.health.last_check = datetime.now()
        
        try:
            start_time = datetime.now()
            self.client.admin.command('ping')
            end_time = datetime.now()
            
            self.health.latency = (end_time - start_time).total_seconds() * 1000
            self.health.status = ConnectionStatus.CONNECTED
            self.health.error = None
            
            # Get database info
            db_info = self.client.server_info()
            self.health.version = db_info.get('version', 'unknown')
            
            # Get collections count
            collections = self.db.list_collection_names()
            self.health.tables_count = len(collections)
            
        except PyMongoError as e:
            self.health.status = ConnectionStatus.ERROR
            self.health.error = str(e)
            self._stats["errors"] += 1
        
        return self.health
    
    async def execute(self, query: Any, **kwargs) -> Any:
        """
        Execute MongoDB query.
        
        Args:
            query: MongoDB query dict
            **kwargs: Additional parameters
            
        Returns:
            Query result
        """
        try:
            collection = kwargs.get('collection')
            operation = kwargs.get('operation')
            
            if not collection or not operation:
                raise DatabaseError("Collection and operation must be specified")
            
            coll = self.db[collection]
            
            if operation == 'find':
                result = list(coll.find(query))
            elif operation == 'find_one':
                result = coll.find_one(query)
            elif operation == 'insert_one':
                result = coll.insert_one(query)
            elif operation == 'insert_many':
                result = coll.insert_many(query)
            elif operation == 'update_one':
                result = coll.update_one(
                    query.get('filter', {}),
                    query.get('update', {}),
                    upsert=query.get('upsert', False)
                )
            elif operation == 'update_many':
                result = coll.update_many(
                    query.get('filter', {}),
                    query.get('update', {}),
                    upsert=query.get('upsert', False)
                )
            elif operation == 'delete_one':
                result = coll.delete_one(query)
            elif operation == 'delete_many':
                result = coll.delete_many(query)
            elif operation == 'count':
                result = coll.count_documents(query)
            elif operation == 'aggregate':
                result = list(coll.aggregate(query))
            else:
                raise DatabaseError(f"Unsupported MongoDB operation: {operation}")
            
            self._stats["queries_executed"] += 1
            return result
            
        except PyMongoError as e:
            self._stats["errors"] += 1
            raise DatabaseError(f"MongoDB error: {e}")


class RedisConnection(DatabaseConnection):
    """Redis database connection."""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        super().__init__(config)
        self.redis: Optional[aioredis.Redis] = None
    
    async def connect(self) -> None:
        """Establish Redis connection."""
        try:
            if self.config.url:
                self.redis = await aioredis.from_url(
                    self.config.url,
                    password=self.config.password,
                    db=self.config.name if isinstance(self.config.name, int) else 0,
                    encoding="utf-8",
                    decode_responses=True,
                )
            else:
                self.redis = await aioredis.Redis(
                    host=self.config.host,
                    port=self.config.port,
                    password=self.config.password,
                    db=self.config.name if isinstance(self.config.name, int) else 0,
                    encoding="utf-8",
                    decode_responses=True,
                )
            
            # Test connection
            await self.redis.ping()
            
            self._connected = True
            self.health.status = ConnectionStatus.CONNECTED
            self._stats["connection_time"] = datetime.now()
            
            logger.info(f"Connected to Redis: {self.config.host}:{self.config.port}")
            
        except RedisError as e:
            self._connected = False
            self.health.status = ConnectionStatus.ERROR
            self.health.error = str(e)
            raise DatabaseError(f"Failed to connect to Redis: {e}")
    
    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            self.redis = None
            self._connected = False
            self.health.status = ConnectionStatus.DISCONNECTED
            logger.info("Disconnected from Redis")
    
    async def health_check(self) -> DatabaseHealth:
        """Perform health check."""
        self.health.last_check = datetime.now()
        
        try:
            start_time = datetime.now()
            await self.redis.ping()
            end_time = datetime.now()
            
            self.health.latency = (end_time - start_time).total_seconds() * 1000
            self.health.status = ConnectionStatus.CONNECTED
            self.health.error = None
            
            # Get Redis info
            info = await self.redis.info()
            self.health.version = info.get('redis_version', 'unknown')
            
            # Get keys count
            keys_count = await self.redis.dbsize()
            self.health.tables_count = keys_count
            
        except RedisError as e:
            self.health.status = ConnectionStatus.ERROR
            self.health.error = str(e)
            self._stats["errors"] += 1
        
        return self.health
    
    async def execute(self, command: str, *args, **kwargs) -> Any:
        """
        Execute Redis command.
        
        Args:
            command: Redis command
            *args: Command arguments
            **kwargs: Additional parameters
            
        Returns:
            Command result
        """
        try:
            if not hasattr(self.redis, command):
                raise DatabaseError(f"Invalid Redis command: {command}")
            
            method = getattr(self.redis, command)
            result = await method(*args, **kwargs)
            
            self._stats["queries_executed"] += 1
            return result
            
        except RedisError as e:
            self._stats["errors"] += 1
            raise DatabaseError(f"Redis error: {e}")


class DatabaseManager:
    """
    Main database manager supporting multiple backends.
    
    Attributes:
        config: Database configuration
        connection: Active database connection
        backend: Current backend type
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize database manager.
        
        Args:
            config: Database configuration
        """
        self.config = config or get_config().database
        self.connection: Optional[DatabaseConnection] = None
        self.backend: Optional[DatabaseBackend] = None
        
        # Connection pool for multiple databases
        self._connections: Dict[str, DatabaseConnection] = {}
        self._default_connection: Optional[str] = None
    
    async def connect(
        self,
        backend: Optional[DatabaseBackend] = None,
        config: Optional[DatabaseConfig] = None
    ) -> None:
        """
        Establish database connection.
        
        Args:
            backend: Database backend type
            config: Database configuration
        """
        if config:
            self.config = config
        
        if not backend:
            # Determine backend from config
            if self.config.type == DatabaseType.MONGODB:
                backend = DatabaseBackend.MONGODB
            elif self.config.type in [DatabaseType.POSTGRESQL, DatabaseType.MYSQL, DatabaseType.SQLITE]:
                backend = DatabaseBackend.SQLALCHEMY
            else:
                raise DatabaseError(f"Cannot determine backend for type: {self.config.type}")
        
        self.backend = backend
        
        # Create appropriate connection
        if backend == DatabaseBackend.SQLALCHEMY:
            self.connection = SQLAlchemyConnection(self.config)
        elif backend == DatabaseBackend.MONGODB:
            self.connection = MongoDBConnection(self.config)
        elif backend == DatabaseBackend.REDIS:
            self.connection = RedisConnection(self.config)
        else:
            raise DatabaseError(f"Unsupported database backend: {backend}")
        
        await self.connection.connect()
        logger.info(f"Database manager connected using {backend}")
    
    async def disconnect(self) -> None:
        """Close database connection."""
        if self.connection:
            await self.connection.disconnect()
            self.connection = None
            self.backend = None
            logger.info("Database manager disconnected")
    
    async def health_check(self) -> DatabaseHealth:
        """
        Perform health check.
        
        Returns:
            DatabaseHealth object
            
        Raises:
            DatabaseError: If no connection established
        """
        if not self.connection:
            raise DatabaseError("No database connection established")
        
        return await self.connection.health_check()
    
    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get database session (SQLAlchemy only).
        
        Yields:
            AsyncSession instance
            
        Raises:
            DatabaseError: If not using SQLAlchemy backend
        """
        if not isinstance(self.connection, SQLAlchemyConnection):
            raise DatabaseError("Session context manager only available for SQLAlchemy backend")
        
        async with self.connection.get_session() as session:
            yield session
    
    async def execute(self, query: Any, **kwargs) -> Any:
        """
        Execute database query.
        
        Args:
            query: Query to execute
            **kwargs: Additional parameters
            
        Returns:
            Query result
            
        Raises:
            DatabaseError: If no connection established
        """
        if not self.connection:
            raise DatabaseError("No database connection established")
        
        return await self.connection.execute(query, **kwargs)
    
    async def execute_raw(self, sql: str, params: Optional[Dict] = None) -> Any:
        """
        Execute raw SQL query (SQLAlchemy only).
        
        Args:
            sql: SQL query string
            params: Query parameters
            
        Returns:
            Query result
            
        Raises:
            DatabaseError: If not using SQLAlchemy backend
        """
        if not isinstance(self.connection, SQLAlchemyConnection):
            raise DatabaseError("Raw SQL execution only available for SQLAlchemy backend")
        
        return await self.connection.execute_raw(sql, params)
    
    async def create_all(self, base: Any = Base) -> None:
        """
        Create all tables (SQLAlchemy only).
        
        Args:
            base: SQLAlchemy Base class
            
        Raises:
            DatabaseError: If not using SQLAlchemy backend
        """
        if not isinstance(self.connection, SQLAlchemyConnection):
            raise DatabaseError("Table creation only available for SQLAlchemy backend")
        
        await self.connection.create_all(base)
    
    async def drop_all(self, base: Any = Base) -> None:
        """
        Drop all tables (SQLAlchemy only).
        
        Args:
            base: SQLAlchemy Base class
            
        Raises:
            DatabaseError: If not using SQLAlchemy backend
        """
        if not isinstance(self.connection, SQLAlchemyConnection):
            raise DatabaseError("Table dropping only available for SQLAlchemy backend")
        
        await self.connection.drop_all(base)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with statistics
        """
        if self.connection:
            return self.connection.get_stats()
        return {}
    
    async def add_connection(
        self,
        name: str,
        backend: DatabaseBackend,
        config: DatabaseConfig
    ) -> None:
        """
        Add a new database connection.
        
        Args:
            name: Connection name
            backend: Database backend
            config: Database configuration
        """
        if name in self._connections:
            logger.warning(f"Connection '{name}' already exists, reconnecting")
            await self.remove_connection(name)
        
        # Create connection
        if backend == DatabaseBackend.SQLALCHEMY:
            connection = SQLAlchemyConnection(config)
        elif backend == DatabaseBackend.MONGODB:
            connection = MongoDBConnection(config)
        elif backend == DatabaseBackend.REDIS:
            connection = RedisConnection(config)
        else:
            raise DatabaseError(f"Unsupported database backend: {backend}")
        
        await connection.connect()
        self._connections[name] = connection
        
        # Set as default if first connection
        if not self._default_connection:
            self._default_connection = name
        
        logger.info(f"Added database connection: {name}")
    
    async def remove_connection(self, name: str) -> None:
        """
        Remove a database connection.
        
        Args:
            name: Connection name
        """
        if name in self._connections:
            await self._connections[name].disconnect()
            del self._connections[name]
            
            # Update default connection if needed
            if self._default_connection == name:
                self._default_connection = next(iter(self._connections.keys()), None)
            
            logger.info(f"Removed database connection: {name}")
    
    def get_connection(self, name: Optional[str] = None) -> DatabaseConnection:
        """
        Get a database connection by name.
        
        Args:
            name: Connection name (None for default)
            
        Returns:
            DatabaseConnection instance
            
        Raises:
            DatabaseError: If connection not found
        """
        if not name:
            name = self._default_connection
        
        if not name or name not in self._connections:
            raise DatabaseError(f"Database connection not found: {name}")
        
        return self._connections[name]
    
    def set_default_connection(self, name: str) -> None:
        """
        Set default database connection.
        
        Args:
            name: Connection name
            
        Raises:
            DatabaseError: If connection not found
        """
        if name not in self._connections:
            raise DatabaseError(f"Database connection not found: {name}")
        
        self._default_connection = name
        logger.info(f"Set default database connection: {name}")
    
    def list_connections(self) -> Dict[str, Dict[str, Any]]:
        """
        List all database connections.
        
        Returns:
            Dictionary of connection info
        """
        connections = {}
        for name, conn in self._connections.items():
            connections[name] = {
                "backend": type(conn).__name__.replace("Connection", ""),
                "status": conn.health.status.value,
                "connected": conn._connected,
            }
        return connections


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


async def get_database_manager(
    config: Optional[DatabaseConfig] = None,
    backend: Optional[DatabaseBackend] = None
) -> DatabaseManager:
    """
    Get or create database manager instance.
    
    Args:
        config: Database configuration
        backend: Database backend
        
    Returns:
        DatabaseManager instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(config)
        await _db_manager.connect(backend)
    return _db_manager


async def close_database_manager() -> None:
    """Close database manager."""
    global _db_manager
    if _db_manager:
        await _db_manager.disconnect()
        _db_manager = None


# Context manager for database operations
@asynccontextmanager
async def database_session(
    config: Optional[DatabaseConfig] = None,
    backend: Optional[DatabaseBackend] = None
) -> AsyncGenerator[DatabaseManager, None]:
    """
    Context manager for database operations.
    
    Args:
        config: Database configuration
        backend: Database backend
        
    Yields:
        DatabaseManager instance
    """
    db_manager = await get_database_manager(config, backend)
    try:
        yield db_manager
    finally:
        await close_database_manager()


# Convenience functions for common operations
async def execute_query(
    query: Any,
    config: Optional[DatabaseConfig] = None,
    **kwargs
) -> Any:
    """Execute a database query."""
    async with database_session(config) as db:
        return await db.execute(query, **kwargs)


async def health_check(config: Optional[DatabaseConfig] = None) -> DatabaseHealth:
    """Perform database health check."""
    async with database_session(config) as db:
        return await db.health_check()


async def create_tables(base: Any = Base, config: Optional[DatabaseConfig] = None) -> None:
    """Create database tables."""
    async with database_session(config) as db:
        await db.create_all(base)


# Model mixins
class TimestampMixin:
    """Mixin for timestamp fields."""
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class SoftDeleteMixin:
    """Mixin for soft delete functionality."""
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime, nullable=True)


class AuditMixin:
    """Mixin for audit fields."""
    created_by = Column(String(100), nullable=True)
    updated_by = Column(String(100), nullable=True)


# Query utilities
class QueryBuilder:
    """SQL query builder utility."""
    
    @staticmethod
    def build_select(
        table: Table,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> select:
        """
        Build SELECT query.
        
        Args:
            table: SQLAlchemy Table
            filters: Filter conditions
            order_by: Order by columns
            limit: Result limit
            offset: Result offset
            
        Returns:
            SELECT query
        """
        query = select(table)
        
        if filters:
            for key, value in filters.items():
                if hasattr(table.c, key):
                    column = getattr(table.c, key)
                    if value is None:
                        query = query.where(column.is_(None))
                    else:
                        query = query.where(column == value)
        
        if order_by:
            order_clauses = []
            for col in order_by:
                if col.startswith('-'):
                    order_clauses.append(getattr(table.c, col[1:]).desc())
                else:
                    order_clauses.append(getattr(table.c, col).asc())
            query = query.order_by(*order_clauses)
        
        if limit:
            query = query.limit(limit)
        
        if offset:
            query = query.offset(offset)
        
        return query
    
    @staticmethod
    def build_insert(table: Table, data: Dict[str, Any]) -> insert:
        """
        Build INSERT query.
        
        Args:
            table: SQLAlchemy Table
            data: Data to insert
            
        Returns:
            INSERT query
        """
        return insert(table).values(**data)
    
    @staticmethod
    def build_update(table: Table, filters: Dict[str, Any], data: Dict[str, Any]) -> update:
        """
        Build UPDATE query.
        
        Args:
            table: SQLAlchemy Table
            filters: Filter conditions
            data: Data to update
            
        Returns:
            UPDATE query
        """
        query = update(table)
        
        for key, value in filters.items():
            if hasattr(table.c, key):
                column = getattr(table.c, key)
                query = query.where(column == value)
        
        return query.values(**data)
    
    @staticmethod
    def build_delete(table: Table, filters: Dict[str, Any]) -> delete:
        """
        Build DELETE query.
        
        Args:
            table: SQLAlchemy Table
            filters: Filter conditions
            
        Returns:
            DELETE query
        """
        query = delete(table)
        
        for key, value in filters.items():
            if hasattr(table.c, key):
                column = getattr(table.c, key)
                query = query.where(column == value)
        
        return query