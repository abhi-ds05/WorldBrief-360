"""
Core module containing essential utilities and configurations for the application.
"""

from .config import Config, get_config, load_config
from .settings import Settings, get_settings
from .security import SecurityManager, encrypt_data, decrypt_data, hash_password
from .logging_config import setup_logging, get_logger
from .exceptions import (
    AppException,
    ConfigurationError,
    SecurityError,
    ValidationError,
    DatabaseError,
    CacheError,
    ExternalServiceError
)
from .constants import (
    AppConstants,
    CacheConstants,
    SecurityConstants,
    ErrorMessages
)
from .metadata import AppMetadata, get_app_metadata
from .version import __version__, get_version_info

__all__ = [
    # Config
    "Config",
    "get_config",
    "load_config",
    
    # Settings
    "Settings",
    "get_settings",
    
    # Security
    "SecurityManager",
    "encrypt_data",
    "decrypt_data",
    "hash_password",
    
    # Logging
    "setup_logging",
    "get_logger",
    
    # Exceptions
    "AppException",
    "ConfigurationError",
    "SecurityError",
    "ValidationError",
    "DatabaseError",
    "CacheError",
    "ExternalServiceError",
    
    # Constants
    "AppConstants",
    "CacheConstants",
    "SecurityConstants",
    "ErrorMessages",
    
    # Metadata
    "AppMetadata",
    "get_app_metadata",
    
    # Version
    "__version__",
    "get_version_info",
]