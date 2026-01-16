"""
Security Module for WorldBrief 360

This module provides comprehensive security features including:
- Authentication and authorization middleware
- Rate limiting and request throttling
- Input validation and sanitization
- Security headers and Content Security Policy
- Audit logging and monitoring
- CORS configuration
- SSL/TLS utilities

Usage:
    from app.security import (
        SecurityMiddleware,
        RateLimiter,
        InputSanitizer,
        get_current_user,
        require_role,
        require_permission
    )
"""

__version__ = "1.0.0"
__author__ = "WorldBrief 360 Security Team"

# Import main components for easy access
from .auth_middleware import (
    SecurityMiddleware,
    JWTTokenHandler,
    get_current_user,
    require_role,
    require_permission,
    verify_api_key,
    create_access_token,
    create_refresh_token,
    decode_token,
    TokenBlacklist,
)

from .rate_limiter import (
    RateLimiter,
    TokenBucket,
    SlidingWindowLimiter,
    RateLimitExceeded,
    rate_limit,
    user_rate_limit,
    ip_rate_limit,
    get_rate_limit_headers,
)

from .input_sanitizer import (
    InputSanitizer,
    sanitize_html,
    sanitize_sql,
    sanitize_json,
    validate_email,
    validate_url,
    validate_phone,
    validate_geolocation,
    strip_xss,
    escape_html,
)

from .content_security import (
    ContentSecurityPolicy,
    SecurityHeaders,
    SecurityPolicy,
    XSSProtection,
    FrameOptions,
    HSTS,
    ReferrerPolicy,
    FeaturePolicy,
    PermissionsPolicy,
)

from .audit_logger import (
    AuditLogger,
    AuditEvent,
    audit_log,
    SecurityEvent,
    UserActivityEvent,
    SystemEvent,
    IncidentVerificationEvent,
    WalletTransactionEvent,
)

from .cors import (
    CORSConfig,
    setup_cors,
    cors_middleware,
    allowed_origins,
    allowed_methods,
    allowed_headers,
)

from .ssl import (
    SSLConfig,
    SSLContext,
    generate_self_signed_cert,
    verify_certificate_chain,
    check_certificate_expiry,
    enforce_https,
)

# Optional imports for advanced features
try:
    from .mfa import (      #type: ignore
        MFAService,
        TOTPHandler,
        EmailOTPHandler,
        generate_backup_codes,
        verify_mfa,
    )
except ImportError:
    pass

try:
    from .threat_detection import (   #type: ignore
        ThreatDetector,
        AnomalyDetector,
        BruteForceDetector,
        BotDetection,
        threat_score,
    )
except ImportError:
    pass

# Export main security utilities
__all__ = [
    # Authentication
    "SecurityMiddleware",
    "JWTTokenHandler",
    "get_current_user",
    "require_role",
    "require_permission",
    "create_access_token",
    "create_refresh_token",
    
    # Rate Limiting
    "RateLimiter",
    "rate_limit",
    "RateLimitExceeded",
    
    # Input Sanitization
    "InputSanitizer",
    "sanitize_html",
    "sanitize_sql",
    "validate_email",
    "validate_url",
    
    # Security Headers
    "ContentSecurityPolicy",
    "SecurityHeaders",
    "SecurityPolicy",
    
    # Audit Logging
    "AuditLogger",
    "audit_log",
    "AuditEvent",
    
    # CORS
    "CORSConfig",
    "setup_cors",
    
    # SSL
    "SSLConfig",
    "enforce_https",
]

# Configuration defaults
DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "https://worldbrief360.com",
]

DEFAULT_RATE_LIMITS = {
    "anonymous": {"requests": 100, "period": 3600},  # 100 requests per hour
    "authenticated": {"requests": 1000, "period": 3600},  # 1000 requests per hour
    "admin": {"requests": 10000, "period": 3600},  # 10000 requests per hour
}

DEFAULT_SECURITY_HEADERS = {
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}

# Security constants
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7
PASSWORD_HASH_ALGORITHM = "bcrypt"
PASSWORD_MIN_LENGTH = 8
API_KEY_LENGTH = 32
SESSION_TIMEOUT_MINUTES = 30

# Security event types
SECURITY_EVENTS = {
    "LOGIN_SUCCESS": "user.login.success",
    "LOGIN_FAILED": "user.login.failed",
    "PASSWORD_CHANGE": "user.password.change",
    "2FA_ENABLED": "user.2fa.enabled",
    "API_KEY_GENERATED": "api.key.generated",
    "RATE_LIMIT_EXCEEDED": "security.rate_limit.exceeded",
    "SUSPICIOUS_ACTIVITY": "security.suspicious.activity",
    "INCIDENT_REPORTED": "incident.reported",
    "INCIDENT_VERIFIED": "incident.verified",
    "WALLET_TRANSACTION": "wallet.transaction",
}


def setup_security(app, config=None):
    """
    Setup all security features for the FastAPI application.
    
    Args:
        app: FastAPI application instance
        config: Optional configuration dictionary
    
    Returns:
        Configured FastAPI app with security features
    """
    from fastapi import FastAPI
    
    # Default configuration
    if config is None:
        config = {}
    
    # Setup CORS
    cors_config = config.get("cors", {})
    origins = cors_config.get("origins", DEFAULT_CORS_ORIGINS)
    app = setup_cors(app, origins=origins)
    
    # Add security middleware
    app.add_middleware(SecurityMiddleware)
    
    # Setup rate limiting if enabled
    if config.get("enable_rate_limiting", True):
        from .rate_limiter import setup_rate_limiting
        rate_limits = config.get("rate_limits", DEFAULT_RATE_LIMITS)
        app = setup_rate_limiting(app, rate_limits)
    
    # Add security headers
    if config.get("enable_security_headers", True):
        from .content_security import add_security_headers
        headers = config.get("security_headers", DEFAULT_SECURITY_HEADERS)
        app = add_security_headers(app, headers)
    
    # Setup audit logging
    if config.get("enable_audit_logging", True):
        from .audit_logger import setup_audit_logging
        audit_config = config.get("audit", {})
        app = setup_audit_logging(app, audit_config)
    
    # Setup input sanitization
    if config.get("enable_input_sanitization", True):
        from .input_sanitizer import setup_input_sanitization
        app = setup_input_sanitization(app)
    
    return app


def get_security_config():
    """
    Get the current security configuration.
    
    Returns:
        Dictionary with security configuration
    """
    from app.core.config import get_settings
    
    settings = get_settings()
    
    return {
        "jwt_secret_key": settings.SECRET_KEY,
        "jwt_algorithm": JWT_ALGORITHM,
        "access_token_expire_minutes": ACCESS_TOKEN_EXPIRE_MINUTES,
        "refresh_token_expire_days": REFRESH_TOKEN_EXPIRE_DAYS,
        "password_hash_algorithm": PASSWORD_HASH_ALGORITHM,
        "password_min_length": PASSWORD_MIN_LENGTH,
        "enable_2fa": settings.ENABLE_2FA,
        "require_email_verification": settings.REQUIRE_EMAIL_VERIFICATION,
        "session_timeout_minutes": SESSION_TIMEOUT_MINUTES,
        "allowed_file_types": [".jpg", ".jpeg", ".png", ".pdf", ".txt"],
        "max_file_size_mb": 10,
        "enable_csrf_protection": True,
        "enable_csp": True,
        "enable_hsts": settings.ENVIRONMENT == "production",
        "enable_rate_limiting": True,
        "enable_audit_logging": True,
        "enable_threat_detection": True,
    }


# Security utilities
def hash_password(password: str) -> str:
    """
    Hash a password using the configured algorithm.
    
    Args:
        password: Plain text password
    
    Returns:
        Hashed password
    """
    from passlib.context import CryptContext
    
    pwd_context = CryptContext(schemes=[PASSWORD_HASH_ALGORITHM], deprecated="auto")
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password
    
    Returns:
        True if password matches, False otherwise
    """
    from passlib.context import CryptContext
    
    pwd_context = CryptContext(schemes=[PASSWORD_HASH_ALGORITHM], deprecated="auto")
    return pwd_context.verify(plain_password, hashed_password)


def generate_api_key() -> str:
    """
    Generate a secure API key.
    
    Returns:
        Random API key string
    """
    import secrets
    import string
    
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(API_KEY_LENGTH))


def validate_password_strength(password: str) -> dict:
    """
    Validate password strength.
    
    Args:
        password: Password to validate
    
    Returns:
        Dictionary with validation results
    """
    import re
    
    results = {
        "is_valid": True,
        "errors": [],
        "score": 0,
        "strength": "weak",
    }
    
    # Check length
    if len(password) < PASSWORD_MIN_LENGTH:
        results["is_valid"] = False
        results["errors"].append(f"Password must be at least {PASSWORD_MIN_LENGTH} characters long")
    
    # Check for uppercase
    if not re.search(r'[A-Z]', password):
        results["score"] += 1
        results["errors"].append("Password should contain at least one uppercase letter")
    
    # Check for lowercase
    if not re.search(r'[a-z]', password):
        results["score"] += 1
        results["errors"].append("Password should contain at least one lowercase letter")
    
    # Check for digits
    if not re.search(r'\d', password):
        results["score"] += 1
        results["errors"].append("Password should contain at least one digit")
    
    # Check for special characters
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        results["score"] += 1
        results["errors"].append("Password should contain at least one special character")
    
    # Calculate strength
    if results["score"] >= 4:
        results["strength"] = "strong"
    elif results["score"] >= 2:
        results["strength"] = "medium"
    else:
        results["strength"] = "weak"
    
    return results


# Security context for dependency injection
class SecurityContext:
    """Context manager for security operations."""
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.sanitizer = InputSanitizer()
        self.audit_logger = AuditLogger()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# Initialize security context
_security_context = None


def get_security_context() -> SecurityContext:
    """
    Get or create security context singleton.
    
    Returns:
        SecurityContext instance
    """
    global _security_context
    
    if _security_context is None:
        _security_context = SecurityContext()
    
    return _security_context


# Security decorators for easy use
def secure_endpoint(rate_limit: bool = True, audit: bool = True, sanitize: bool = True):
    """
    Decorator to apply security features to endpoints.
    
    Args:
        rate_limit: Apply rate limiting
        audit: Enable audit logging
        sanitize: Enable input sanitization
    
    Returns:
        Decorated function
    """
    def decorator(func):
        from functools import wraps
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get security context
            context = get_security_context()
            
            try:
                # Apply rate limiting
                if rate_limit:
                    await context.rate_limiter.check_limit()
                
                # Sanitize inputs
                if sanitize:
                    kwargs = context.sanitizer.sanitize_inputs(kwargs)
                
                # Call original function
                result = await func(*args, **kwargs)
                
                # Audit log
                if audit:
                    context.audit_logger.log_security_event(
                        event_type="endpoint.access",
                        user_id="current_user",  # Would come from request context
                        details={"endpoint": func.__name__, "result": "success"}
                    )
                
                return result
                
            except Exception as e:
                # Log security failure
                if audit:
                    context.audit_logger.log_security_event(
                        event_type="endpoint.access",
                        user_id="current_user",
                        details={"endpoint": func.__name__, "result": "failure", "error": str(e)}
                    )
                raise
        
        return wrapper
    
    return decorator


print(f"Security module {__version__} initialized successfully.")