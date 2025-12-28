"""
Configuration management using Pydantic settings.
Supports environment variables, .env files, and YAML configuration.
"""

import os
import yaml
from typing import Any, Dict, Optional, List, Union
from enum import Enum
from pydantic import BaseSettings, Field, validator
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Application environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class CacheBackend(str, Enum):
    """Cache backend types."""
    MEMORY = "memory"
    REDIS = "redis"
    FILE = "file"
    MULTI_LEVEL = "multi_level"


class DatabaseType(str, Enum):
    """Database types."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MONGODB = "mongodb"


class LogLevel(str, Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class CacheConfig(BaseSettings):
    """Cache configuration."""
    backend: CacheBackend = CacheBackend.MEMORY
    redis_url: Optional[str] = Field(None, env="REDIS_URL")
    redis_password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    redis_db: int = Field(0, env="REDIS_DB")
    file_cache_path: str = Field("./cache", env="FILE_CACHE_PATH")
    default_ttl: int = Field(300, env="CACHE_TTL")  # 5 minutes
    max_memory_entries: int = Field(1000, env="CACHE_MAX_MEMORY_ENTRIES")
    
    @validator("redis_url", always=True)
    def validate_redis_url(cls, v, values):
        if values.get("backend") == CacheBackend.REDIS and not v:
            raise ValueError("Redis URL is required when using Redis backend")
        return v


class DatabaseConfig(BaseSettings):
    """Database configuration."""
    type: DatabaseType = DatabaseType.SQLITE
    url: Optional[str] = Field(None, env="DATABASE_URL")
    host: str = Field("localhost", env="DB_HOST")
    port: int = Field(5432, env="DB_PORT")
    name: str = Field("app.db", env="DB_NAME")
    user: Optional[str] = Field(None, env="DB_USER")
    password: Optional[str] = Field(None, env="DB_PASSWORD")
    pool_size: int = Field(10, env="DB_POOL_SIZE")
    echo: bool = Field(False, env="DB_ECHO")
    
    @property
    def connection_string(self) -> str:
        """Generate connection string based on database type."""
        if self.url:
            return self.url
        
        if self.type == DatabaseType.SQLITE:
            return f"sqlite:///{self.name}"
        elif self.type == DatabaseType.POSTGRESQL:
            return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
        elif self.type == DatabaseType.MYSQL:
            return f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
        elif self.type == DatabaseType.MONGODB:
            return f"mongodb://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
        return ""


class SecurityConfig(BaseSettings):
    """Security configuration."""
    secret_key: str = Field(..., env="SECRET_KEY")
    algorithm: str = Field("HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(7, env="REFRESH_TOKEN_EXPIRE_DAYS")
    password_hash_algorithm: str = Field("bcrypt", env="PASSWORD_HASH_ALGORITHM")
    password_hash_rounds: int = Field(12, env="PASSWORD_HASH_ROUNDS")
    cors_origins: List[str] = Field(["*"], env="CORS_ORIGINS")
    rate_limit_per_minute: int = Field(60, env="RATE_LIMIT_PER_MINUTE")
    
    @validator("secret_key")
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        return v


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    level: LogLevel = LogLevel.INFO
    format: str = Field("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    date_format: str = Field("%Y-%m-%d %H:%M:%S")
    file_path: Optional[str] = Field(None, env="LOG_FILE_PATH")
    max_file_size: int = Field(10485760, env="LOG_MAX_FILE_SIZE")  # 10MB
    backup_count: int = Field(5, env="LOG_BACKUP_COUNT")


class APIConfig(BaseSettings):
    """API configuration."""
    title: str = Field("Application API", env="API_TITLE")
    version: str = Field("1.0.0", env="API_VERSION")
    description: str = Field("", env="API_DESCRIPTION")
    docs_url: str = Field("/docs", env="API_DOCS_URL")
    redoc_url: str = Field("/redoc", env="API_REDOC_URL")
    openapi_url: str = Field("/openapi.json", env="OPENAPI_URL")
    api_prefix: str = Field("/api", env="API_PREFIX")
    debug: bool = Field(False, env="API_DEBUG")


class Config(BaseSettings):
    """Main application configuration."""
    
    # Core
    app_name: str = Field("My Application", env="APP_NAME")
    environment: Environment = Field(Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(False, env="DEBUG")
    
    # Server
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    workers: int = Field(1, env="WORKERS")
    
    # Components
    cache: CacheConfig = Field(default_factory=CacheConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    
    # External services
    external_api_url: Optional[str] = Field(None, env="EXTERNAL_API_URL")
    external_api_key: Optional[str] = Field(None, env="EXTERNAL_API_KEY")
    
    # Feature flags
    enable_cache: bool = Field(True, env="ENABLE_CACHE")
    enable_rate_limit: bool = Field(True, env="ENABLE_RATE_LIMIT")
    enable_metrics: bool = Field(False, env="ENABLE_METRICS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        case_sensitive = False
    
    @validator("environment", pre=True)
    def normalize_environment(cls, v):
        if isinstance(v, str):
            return v.lower()
        return v
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION
    
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == Environment.TESTING


@lru_cache()
def get_config(config_file: Optional[str] = None) -> Config:
    """
    Get cached configuration instance.
    
    Args:
        config_file: Optional path to YAML configuration file
        
    Returns:
        Config instance
    """
    config_data = {}
    
    # Load YAML config if provided
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                yaml_data = yaml.safe_load(f)
                if yaml_data:
                    config_data = yaml_data
                    logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.warning(f"Failed to load YAML config from {config_file}: {e}")
    
    # Create config instance
    config = Config(_env_file=".env", **config_data)
    
    # Set logging level based on config
    logging.getLogger().setLevel(config.logging.level.value)
    
    logger.info(f"Configuration loaded for {config.app_name} in {config.environment} environment")
    return config


def load_config(config_file: Optional[str] = None) -> Config:
    """
    Load configuration (clears cache first).
    
    Args:
        config_file: Optional path to YAML configuration file
        
    Returns:
        Config instance
    """
    get_config.cache_clear()
    return get_config(config_file)