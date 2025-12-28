"""
Application constants and enumerations.
"""

from enum import Enum, IntEnum
from datetime import timedelta


class AppConstants:
    """Application-wide constants."""
    
    # Application
    APP_NAME = "My Application"
    APP_DESCRIPTION = "A modern web application"
    APP_VERSION = "1.0.0"
    
    # Paths
    ROOT_DIR = "/"
    STATIC_DIR = "static"
    TEMPLATES_DIR = "templates"
    UPLOAD_DIR = "uploads"
    
    # Time
    DEFAULT_TIMEOUT = 30  # seconds
    CACHE_TIMEOUT = 5  # seconds for cache operations
    REQUEST_TIMEOUT = 30  # seconds for HTTP requests
    
    # Limits
    MAX_PAGE_SIZE = 100
    DEFAULT_PAGE_SIZE = 20
    MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_REQUEST_SIZE = 16 * 1024 * 1024  # 16MB
    
    # Retry
    DEFAULT_RETRY_ATTEMPTS = 3
    RETRY_DELAY = 1  # second


class CacheConstants:
    """Cache-related constants."""
    
    # Keys
    KEY_PREFIX = "app"
    KEY_SEPARATOR = ":"
    
    # TTLs (in seconds)
    TTL_SHORT = 60  # 1 minute
    TTL_MEDIUM = 300  # 5 minutes
    TTL_LONG = 3600  # 1 hour
    TTL_VERY_LONG = 86400  # 24 hours
    TTL_NEVER_EXPIRE = 0  # No expiration
    
    # Namespaces
    NS_USER = "user"
    NS_SESSION = "session"
    NS_CONFIG = "config"
    NS_TOKEN = "token"
    NS_LOCK = "lock"
    
    # Patterns
    PATTERN_ALL = "*"
    PATTERN_USER = f"{KEY_PREFIX}{KEY_SEPARATOR}{NS_USER}{KEY_SEPARATOR}*"
    
    # Lock
    LOCK_TIMEOUT = 10  # seconds
    LOCK_BLOCKING_TIMEOUT = 5  # seconds


class SecurityConstants:
    """Security-related constants."""
    
    # Algorithms
    HASH_ALGORITHM = "HS256"
    ENCRYPTION_ALGORITHM = "AES-GCM"
    PASSWORD_HASH_ALGORITHM = "bcrypt"
    
    # Token
    TOKEN_TYPE_BEARER = "Bearer"
    TOKEN_HEADER = "Authorization"
    TOKEN_PREFIX = "Bearer "
    
    # Password
    PASSWORD_MIN_LENGTH = 8
    PASSWORD_MAX_LENGTH = 128
    PASSWORD_SALT_LENGTH = 32
    
    # Encryption
    ENCRYPTION_KEY_LENGTH = 32  # bytes for AES-256
    IV_LENGTH = 12  # bytes for GCM
    
    # CORS
    CORS_ALLOW_ORIGINS = ["*"]
    CORS_ALLOW_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_ALLOW_HEADERS = ["*"]
    CORS_ALLOW_CREDENTIALS = True
    CORS_MAX_AGE = 600  # seconds
    
    # Rate limiting
    RATE_LIMIT_WINDOW = 60  # seconds
    RATE_LIMIT_MAX_REQUESTS = 100


class ErrorMessages:
    """Standard error messages."""
    
    # General
    INTERNAL_ERROR = "An internal error occurred"
    NOT_FOUND = "Resource not found"
    FORBIDDEN = "Access forbidden"
    UNAUTHORIZED = "Authentication required"
    BAD_REQUEST = "Bad request"
    VALIDATION_ERROR = "Validation error"
    
    # Auth
    INVALID_CREDENTIALS = "Invalid credentials"
    TOKEN_EXPIRED = "Token has expired"
    TOKEN_INVALID = "Invalid token"
    TOKEN_MISSING = "Token missing"
    INSUFFICIENT_PERMISSIONS = "Insufficient permissions"
    
    # Cache
    CACHE_UNAVAILABLE = "Cache service unavailable"
    CACHE_TIMEOUT = "Cache operation timeout"
    CACHE_LOCK_FAILED = "Failed to acquire cache lock"
    
    # Database
    DATABASE_ERROR = "Database error occurred"
    RECORD_NOT_FOUND = "Record not found"
    DUPLICATE_ENTRY = "Duplicate entry"
    CONSTRAINT_VIOLATION = "Constraint violation"
    
    # External services
    EXTERNAL_SERVICE_ERROR = "External service error"
    EXTERNAL_SERVICE_TIMEOUT = "External service timeout"
    EXTERNAL_SERVICE_UNAVAILABLE = "External service unavailable"
    
    # Validation
    REQUIRED_FIELD = "This field is required"
    INVALID_EMAIL = "Invalid email address"
    INVALID_PHONE = "Invalid phone number"
    INVALID_URL = "Invalid URL"
    STRING_TOO_SHORT = "String is too short"
    STRING_TOO_LONG = "String is too long"
    NUMBER_TOO_SMALL = "Number is too small"
    NUMBER_TOO_LARGE = "Number is too large"
    
    # File upload
    FILE_TOO_LARGE = "File is too large"
    INVALID_FILE_TYPE = "Invalid file type"
    UPLOAD_FAILED = "File upload failed"


class StatusCode(IntEnum):
    """HTTP status codes."""
    
    # Success
    OK = 200
    CREATED = 201
    ACCEPTED = 202
    NO_CONTENT = 204
    
    # Client errors
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    CONFLICT = 409
    UNPROCESSABLE_ENTITY = 422
    TOO_MANY_REQUESTS = 429
    
    # Server errors
    INTERNAL_SERVER_ERROR = 500
    NOT_IMPLEMENTED = 501
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503
    GATEWAY_TIMEOUT = 504


class CacheStatus(str, Enum):
    """Cache operation status."""
    HIT = "hit"
    MISS = "miss"
    ERROR = "error"
    EXPIRED = "expired"
    STALE = "stale"


class DatabaseOperation(str, Enum):
    """Database operation types."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    QUERY = "query"
    TRANSACTION = "transaction"