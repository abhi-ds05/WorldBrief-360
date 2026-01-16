"""
Cross-Origin Resource Sharing (CORS) Configuration

This module provides comprehensive CORS configuration with security best practices:
- Origin validation and whitelisting
- Dynamic origin resolution
- Credential handling
- Preflight request optimization
- Security headers for CORS
- Rate limiting for CORS requests
- CORS violation logging and monitoring
"""

import re
import json
import ipaddress
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Tuple, Union
from urllib.parse import urlparse
from functools import lru_cache

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.config import get_settings
from app.security.audit_logger import AuditLogger, AuditEventType, AuditSeverity
from app.security.rate_limiter import RateLimiter

# Get settings
settings = get_settings()


class CORSPolicy(str, Enum):
    """CORS policy enforcement levels."""
    DISABLED = "disabled"  # No CORS headers
    RESTRICTIVE = "restrictive"  # Strict origin matching
    PERMISSIVE = "permissive"  # Allow most origins
    DEVELOPMENT = "development"  # Allow all in dev


class CredentialsPolicy(str, Enum):
    """Credentials handling policies."""
    ALLOW = "allow"  # Allow credentials (cookies, auth headers)
    DENY = "deny"  # Don't allow credentials
    SAME_ORIGIN = "same-origin"  # Only allow for same-origin


@dataclass
class CORSOrigin:
    """Represents an allowed CORS origin with metadata."""
    origin: str
    pattern: Optional[Pattern] = None
    allow_credentials: bool = False
    max_age: int = 86400  # 24 hours
    allowed_methods: Set[str] = field(default_factory=lambda: {"GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"})
    allowed_headers: Set[str] = field(default_factory=lambda: {"*"})
    exposed_headers: Set[str] = field(default_factory=lambda: {"Content-Length", "X-Request-ID"})
    description: str = ""
    is_regex: bool = False
    
    def __post_init__(self):
        """Compile regex pattern if origin is a regex."""
        if self.is_regex:
            try:
                self.pattern = re.compile(self.origin)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern for origin '{self.origin}': {e}")
    
    def matches(self, origin: str) -> bool:
        """Check if origin matches this CORS origin."""
        if self.is_regex and self.pattern:
            return bool(self.pattern.match(origin))
        return self.origin == origin


@dataclass
class CORSConfig:
    """Complete CORS configuration."""
    
    # Policy
    enabled: bool = True
    policy: CORSPolicy = CORSPolicy.RESTRICTIVE
    credentials_policy: CredentialsPolicy = CredentialsPolicy.ALLOW
    
    # Origins
    allowed_origins: List[Union[str, CORSOrigin]] = field(default_factory=list)
    allow_all_origins: bool = False
    origin_regex: Optional[str] = None
    
    # Methods and Headers
    allowed_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"])
    allowed_headers: List[str] = field(default_factory=lambda: ["*"])
    exposed_headers: List[str] = field(default_factory=lambda: [])
    
    # Cache
    max_age: int = 86400  # 24 hours
    preflight_max_age: int = 600  # 10 minutes
    
    # Security
    allow_credentials: bool = True
    vary_header: bool = True
    add_security_headers: bool = True
    
    # Rate limiting
    enable_cors_rate_limiting: bool = True
    cors_rate_limit: int = 100  # requests per minute
    cors_rate_limit_window: int = 60
    
    # Monitoring
    log_violations: bool = True
    block_malicious_origins: bool = True
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Convert string origins to CORSOrigin objects
        processed_origins = []
        for origin in self.allowed_origins:
            if isinstance(origin, str):
                processed_origins.append(CORSOrigin(origin=origin))
            else:
                processed_origins.append(origin)
        self.allowed_origins = processed_origins
        
        # Add default exposed headers
        if not self.exposed_headers:
            self.exposed_headers = ["Content-Length", "X-Request-ID", "X-Total-Count"]
        
        # Set policy-based defaults
        if self.policy == CORSPolicy.DEVELOPMENT:
            self.allow_all_origins = True
            self.allow_credentials = True
        elif self.policy == CORSPolicy.PERMISSIVE:
            self.allowed_origins.extend([
                CORSOrigin(origin="https://*.worldbrief360.com"),
                CORSOrigin(origin="https://worldbrief360.com"),
            ])
        elif self.policy == CORSPolicy.RESTRICTIVE:
            self.allow_credentials = True
            self.allowed_headers = [
                "Authorization",
                "Content-Type",
                "X-Requested-With",
                "X-API-Key",
                "X-CSRF-Token",
            ]
        
        # Set credentials policy
        if self.credentials_policy == CredentialsPolicy.DENY:
            self.allow_credentials = False
        elif self.credentials_policy == CredentialsPolicy.SAME_ORIGIN:
            # This is handled differently in middleware
            pass


class CORSValidator:
    """Validator for CORS origins and requests."""
    
    def __init__(self, config: CORSConfig):
        self.config = config
        self.audit_logger = AuditLogger()
        self.rate_limiter = RateLimiter()
        
        # Pre-compile patterns for faster matching
        self._compiled_patterns: List[Tuple[Pattern, CORSOrigin]] = []
        self._exact_origins: Dict[str, CORSOrigin] = {}
        self._init_patterns()
    
    def _init_patterns(self):
        """Initialize compiled patterns and exact origins."""
        for cors_origin in self.config.allowed_origins:
            if cors_origin.is_regex and cors_origin.pattern:
                self._compiled_patterns.append((cors_origin.pattern, cors_origin))
            else:
                self._exact_origins[cors_origin.origin] = cors_origin
    
    def validate_origin(self, origin: str) -> Tuple[bool, Optional[CORSOrigin]]:
        """
        Validate an origin against allowed origins.
        
        Args:
            origin: The origin to validate
            
        Returns:
            Tuple of (is_valid, matching_cors_origin)
        """
        # Fast path: allow all origins
        if self.config.allow_all_origins:
            return True, None
        
        # Fast path: exact match
        if origin in self._exact_origins:
            return True, self._exact_origins[origin]
        
        # Check regex patterns
        for pattern, cors_origin in self._compiled_patterns:
            if pattern.match(origin):
                return True, cors_origin
        
        # Check if origin is a valid URL
        if not self._is_valid_url(origin):
            return False, None
        
        # Check for subdomain patterns (e.g., *.example.com)
        for cors_origin in self.config.allowed_origins:
            if cors_origin.origin.startswith("*."):
                domain = cors_origin.origin[2:]  # Remove "*."
                if origin.endswith(domain):
                    return True, cors_origin
        
        return False, None
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if a string is a valid URL."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def is_malicious_origin(self, origin: str) -> bool:
        """
        Check if origin appears to be malicious.
        
        Args:
            origin: Origin to check
            
        Returns:
            True if origin appears malicious
        """
        if not origin:
            return False
        
        # Check for common attack patterns
        malicious_patterns = [
            r"\.(php|asp|aspx|jsp|cgi|pl|sh|exe|bat|cmd)(\?|$)",  # Script extensions
            r"(cmd|exec|system|eval|union|select|insert|update|delete|drop|alter)\b",  # SQL/command injection
            r"(alert|prompt|confirm|document\.cookie|localStorage|sessionStorage)\b",  # XSS
            r"(<script|javascript:|onload=|onerror=|onclick=)",  # More XSS
            r"\.(local|localhost|test|dev)$",  # Local domains in production
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",  # IP addresses (might be legitimate)
        ]
        
        for pattern in malicious_patterns:
            if re.search(pattern, origin, re.IGNORECASE):
                return True
        
        # Check for suspicious characters
        suspicious_chars = ['<', '>', '"', "'", '\\', ';', '&', '|', '`', '$', '(', ')']
        for char in suspicious_chars:
            if char in origin:
                return True
        
        return False
    
    def log_cors_violation(self, request: Request, origin: str, reason: str):
        """Log CORS violation for monitoring."""
        if not self.config.log_violations:
            return
        
        self.audit_logger.log_security_event(
            event_type=AuditEventType.SUSPICIOUS_ACTIVITY,
            description=f"CORS Violation: {reason}",
            details={
                "origin": origin,
                "path": request.url.path,
                "method": request.method,
                "user_agent": request.headers.get("User-Agent"),
                "ip_address": request.client.host if request.client else None,
                "referer": request.headers.get("Referer"),
                "reason": reason,
            },
            severity=AuditSeverity.WARNING
        )
    
    async def check_rate_limit(self, origin: str, request: Request) -> bool:
        """
        Check if origin has exceeded rate limit for CORS requests.
        
        Args:
            origin: Request origin
            request: FastAPI request
            
        Returns:
            True if allowed, False if rate limited
        """
        if not self.config.enable_cors_rate_limiting:
            return True
        
        identifier = f"cors:{origin}"
        allowed = await self.rate_limiter.is_allowed(
            identifier=identifier,
            endpoint=request.url.path,
            limit=self.config.cors_rate_limit,
            window=self.config.cors_rate_limit_window
        )
        
        if not allowed:
            self.log_cors_violation(
                request,
                origin,
                f"Rate limit exceeded: {self.config.cors_rate_limit} requests per {self.config.cors_rate_limit_window} seconds"
            )
        
        return allowed


class DynamicCORSResolver:
    """
    Dynamic CORS origin resolver for complex scenarios.
    
    Supports:
    - Database-driven origin whitelisting
    - User-specific origins
    - Environment-based origins
    - Origin validation callbacks
    """
    
    def __init__(self, config: CORSConfig):
        self.config = config
        self.origin_cache: Dict[str, bool] = {}
        self.cache_ttl = 300  # 5 minutes
        self.last_cache_clear = 0
        
    @lru_cache(maxsize=1000)
    def is_origin_allowed(self, origin: str, user_id: Optional[str] = None) -> bool:
        """
        Check if origin is allowed (with caching).
        
        Args:
            origin: Origin to check
            user_id: Optional user ID for user-specific origins
            
        Returns:
            True if origin is allowed
        """
        # Check cache first
        cache_key = f"{origin}:{user_id}"
        if cache_key in self.origin_cache:
            return self.origin_cache[cache_key]
        
        # Dynamic resolution logic
        is_allowed = self._resolve_origin(origin, user_id)
        
        # Cache result
        self.origin_cache[cache_key] = is_allowed
        
        return is_allowed
    
    def _resolve_origin(self, origin: str, user_id: Optional[str]) -> bool:
        """Resolve if origin is allowed dynamically."""
        # Default to validator
        validator = CORSValidator(self.config)
        is_valid, _ = validator.validate_origin(origin)
        
        if not is_valid and user_id:
            # Check user-specific origins from database
            # This is a placeholder for database integration
            is_valid = self._check_user_origin(user_id, origin)
        
        return is_valid
    
    def _check_user_origin(self, user_id: str, origin: str) -> bool:
        """Check if origin is allowed for specific user."""
        # Implement database lookup here
        # Example: return UserCORSOrigin.query.filter_by(user_id=user_id, origin=origin).first() is not None
        return False
    
    def clear_cache(self):
        """Clear origin cache."""
        self.origin_cache.clear()
        self.is_origin_allowed.cache_clear()


class SecureCORSMiddleware(BaseHTTPMiddleware):
    """
    Enhanced CORS middleware with security features.
    
    Features:
    - Origin validation with regex support
    - Dynamic origin resolution
    - Rate limiting for CORS requests
    - CORS violation logging
    - Security headers for CORS
    - Preflight optimization
    """
    
    def __init__(
        self,
        app: ASGIApp,
        config: Optional[CORSConfig] = None,
        validator: Optional[CORSValidator] = None,
        dynamic_resolver: Optional[DynamicCORSResolver] = None
    ):
        super().__init__(app)
        self.config = config or self._get_default_config()
        self.validator = validator or CORSValidator(self.config)
        self.dynamic_resolver = dynamic_resolver or DynamicCORSResolver(self.config)
        self.audit_logger = AuditLogger()
        
        # Initialize FastAPI CORS middleware for basic functionality
        self.cors_middleware = CORSMiddleware(
            app=app,
            allow_origins=[str(origin) for origin in self.config.allowed_origins] if not self.config.allow_all_origins else ["*"],
            allow_credentials=self.config.allow_credentials,
            allow_methods=self.config.allowed_methods,
            allow_headers=self.config.allowed_headers,
            expose_headers=self.config.exposed_headers,
            max_age=self.config.preflight_max_age,
        )
    
    def _get_default_config(self) -> CORSConfig:
        """Get default CORS configuration based on environment."""
        if settings.ENVIRONMENT == "development":
            return CORSConfig(
                policy=CORSPolicy.DEVELOPMENT,
                allowed_origins=[
                    "http://localhost:3000",
                    "http://localhost:8000",
                    "http://127.0.0.1:3000",
                    "http://127.0.0.1:8000",
                ],
                allow_credentials=True,
            )
        elif settings.ENVIRONMENT == "production":
            return CORSConfig(
                policy=CORSPolicy.RESTRICTIVE,
                allowed_origins=[
                    CORSOrigin(
                        origin="https://worldbrief360.com",
                        allow_credentials=True,
                        description="Production frontend"
                    ),
                    CORSOrigin(
                        origin="https://app.worldbrief360.com",
                        allow_credentials=True,
                        description="Production app"
                    ),
                    CORSOrigin(
                        origin="https://admin.worldbrief360.com",
                        allow_credentials=True,
                        description="Admin panel"
                    ),
                ],
                credentials_policy=CredentialsPolicy.ALLOW,
            )
        else:  # staging, testing
            return CORSConfig(
                policy=CORSPolicy.PERMISSIVE,
                allowed_origins=[
                    CORSOrigin(
                        origin="https://staging.worldbrief360.com",
                        allow_credentials=True,
                        description="Staging environment"
                    ),
                    CORSOrigin(
                        origin=re.escape(r"https://*.worldbrief360.com"),
                        is_regex=True,
                        allow_credentials=True,
                        description="All subdomains"
                    ),
                ],
            )
    
    async def dispatch(self, request: Request, call_next):
        """Process CORS requests with enhanced security."""
        
        # Skip CORS for non-browser requests or internal requests
        if not self._is_cors_request(request):
            return await call_next(request)
        
        # Get origin from request
        origin = request.headers.get("origin")
        
        # Handle preflight requests
        if request.method == "OPTIONS" and "access-control-request-method" in request.headers:
            return await self._handle_preflight(request, origin)
        
        # Handle actual CORS request
        return await self._handle_cors_request(request, origin, call_next)
    
    def _is_cors_request(self, request: Request) -> bool:
        """Check if request is a CORS request."""
        # Not a CORS request if no origin header
        if "origin" not in request.headers:
            return False
        
        # Skip for same-origin requests
        if self._is_same_origin(request):
            return False
        
        # Skip for certain paths (e.g., health checks, internal APIs)
        excluded_paths = [
            "/health",
            "/metrics",
            "/internal/",
            "/api/v1/health",
        ]
        
        for path in excluded_paths:
            if request.url.path.startswith(path):
                return False
        
        return True
    
    def _is_same_origin(self, request: Request) -> bool:
        """Check if request is same-origin."""
        origin = request.headers.get("origin")
        if not origin:
            return True
        
        try:
            origin_parsed = urlparse(origin)
            request_parsed = urlparse(str(request.url))
            
            return (
                origin_parsed.scheme == request_parsed.scheme and
                origin_parsed.hostname == request_parsed.hostname and
                origin_parsed.port == request_parsed.port
            )
        except:
            return False
    
    async def _handle_preflight(self, request: Request, origin: Optional[str]) -> Response:
        """Handle CORS preflight request."""
        if not origin:
            return Response(
                status_code=400,
                content=json.dumps({"error": "Missing Origin header"}),
                media_type="application/json"
            )
        
        # Validate origin
        is_valid, cors_origin = self.validator.validate_origin(origin)
        
        if not is_valid:
            self.validator.log_cors_violation(request, origin, "Origin not allowed in preflight")
            return Response(
                status_code=403,
                content=json.dumps({"error": "CORS origin not allowed"}),
                media_type="application/json"
            )
        
        # Check rate limit
        if not await self.validator.check_rate_limit(origin, request):
            return Response(
                status_code=429,
                content=json.dumps({"error": "Rate limit exceeded"}),
                headers={"Retry-After": "60"}
            )
        
        # Get requested method and headers
        requested_method = request.headers.get("access-control-request-method", "")
        requested_headers = request.headers.get("access-control-request-headers", "")
        
        # Validate requested method
        if requested_method and requested_method not in self.config.allowed_methods:
            self.validator.log_cors_violation(
                request, origin, f"Method not allowed: {requested_method}"
            )
            return Response(
                status_code=405,
                content=json.dumps({"error": f"Method {requested_method} not allowed"}),
                media_type="application/json"
            )
        
        # Create preflight response
        response = Response(
            status_code=200,
            content="",
            media_type="text/plain"
        )
        
        # Add CORS headers
        self._add_cors_headers(response, origin, cors_origin, is_preflight=True)
        
        return response
    
    async def _handle_cors_request(self, request: Request, origin: Optional[str], call_next):
        """Handle actual CORS request."""
        if not origin:
            # Not a CORS request, proceed normally
            return await call_next(request)
        
        # Check if origin is malicious
        if self.config.block_malicious_origins and self.validator.is_malicious_origin(origin):
            self.validator.log_cors_violation(request, origin, "Malicious origin detected")
            return Response(
                status_code=403,
                content=json.dumps({"error": "Origin not allowed"}),
                media_type="application/json"
            )
        
        # Validate origin
        is_valid, cors_origin = self.validator.validate_origin(origin)
        
        # Try dynamic resolution if static validation fails
        if not is_valid and self.dynamic_resolver:
            # Extract user ID from request if available
            user_id = None
            if hasattr(request.state, 'auth_context'):
                user_id = getattr(request.state.auth_context, 'user_id', None)
            
            is_valid = self.dynamic_resolver.is_origin_allowed(origin, user_id)
        
        if not is_valid:
            self.validator.log_cors_violation(request, origin, "Origin not allowed")
            return Response(
                status_code=403,
                content=json.dumps({"error": "CORS origin not allowed"}),
                media_type="application/json"
            )
        
        # Check rate limit
        if not await self.validator.check_rate_limit(origin, request):
            return Response(
                status_code=429,
                content=json.dumps({"error": "Rate limit exceeded"}),
                headers={"Retry-After": "60"}
            )
        
        # Process request
        response = await call_next(request)
        
        # Add CORS headers to response
        self._add_cors_headers(response, origin, cors_origin)
        
        return response
    
    def _add_cors_headers(self, response: Response, origin: str, cors_origin: Optional[CORSOrigin], is_preflight: bool = False):
        """Add CORS headers to response."""
        
        # Allow-Credentials
        if self.config.allow_credentials and (cors_origin is None or cors_origin.allow_credentials):
            response.headers["Access-Control-Allow-Credentials"] = "true"
        
        # Allow-Origin
        if self.config.allow_all_origins:
            response.headers["Access-Control-Allow-Origin"] = "*"
        else:
            response.headers["Access-Control-Allow-Origin"] = origin
        
        # Vary header (for caching)
        if self.config.vary_header:
            existing_vary = response.headers.get("Vary", "")
            if "Origin" not in existing_vary:
                response.headers["Vary"] = f"{existing_vary}, Origin".strip(", ")
        
        # Expose headers
        if self.config.exposed_headers:
            exposed_headers = self.config.exposed_headers
            if cors_origin and cors_origin.exposed_headers:
                exposed_headers = list(cors_origin.exposed_headers)
            response.headers["Access-Control-Expose-Headers"] = ", ".join(exposed_headers)
        
        # Preflight-specific headers
        if is_preflight:
            # Allow-Methods
            allowed_methods = self.config.allowed_methods
            if cors_origin and cors_origin.allowed_methods:
                allowed_methods = list(cors_origin.allowed_methods)
            response.headers["Access-Control-Allow-Methods"] = ", ".join(allowed_methods)
            
            # Allow-Headers
            allowed_headers = self.config.allowed_headers
            if cors_origin and cors_origin.allowed_headers:
                allowed_headers = list(cors_origin.allowed_headers)
            response.headers["Access-Control-Allow-Headers"] = ", ".join(allowed_headers)
            
            # Max-Age
            max_age = self.config.preflight_max_age
            if cors_origin:
                max_age = cors_origin.max_age
            response.headers["Access-Control-Max-Age"] = str(max_age)
        
        # Add security headers for CORS
        if self.config.add_security_headers:
            self._add_cors_security_headers(response)
    
    def _add_cors_security_headers(self, response: Response):
        """Add security-specific headers for CORS."""
        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        
        # XSS protection
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Permissions policy
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=(), payment=()"
        )


def setup_cors(
    app: FastAPI,
    config: Optional[CORSConfig] = None,
    use_enhanced_middleware: bool = True
) -> FastAPI:
    """
    Setup CORS for FastAPI application.
    
    Args:
        app: FastAPI application
        config: CORS configuration
        use_enhanced_middleware: Whether to use enhanced middleware
    
    Returns:
        Configured FastAPI app
    """
    if not config:
        # Auto-configure based on environment
        if settings.ENVIRONMENT == "development":
            config = CORSConfig(
                policy=CORSPolicy.DEVELOPMENT,
                allowed_origins=[
                    "http://localhost:3000",
                    "http://localhost:8000",
                    "http://127.0.0.1:3000",
                    "http://127.0.0.1:8000",
                ],
                allow_credentials=True,
                log_violations=True,
            )
        else:
            config = CORSConfig(
                policy=CORSPolicy.RESTRICTIVE,
                allowed_origins=[
                    CORSOrigin(
                        origin="https://worldbrief360.com",
                        allow_credentials=True,
                        description="Production frontend"
                    ),
                ],
                log_violations=True,
                enable_cors_rate_limiting=True,
            )
    
    if use_enhanced_middleware:
        # Use enhanced middleware
        app.add_middleware(SecureCORSMiddleware, config=config)
    else:
        # Use standard FastAPI CORS middleware
        origins = []
        if config.allow_all_origins:
            origins = ["*"]
        else:
            origins = [str(origin) for origin in config.allowed_origins]
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=config.allow_credentials,
            allow_methods=config.allowed_methods,
            allow_headers=config.allowed_headers,
            expose_headers=config.exposed_headers,
            max_age=config.preflight_max_age,
        )
    
    # Add CORS test endpoint
    @app.get("/api/v1/security/cors-test", include_in_schema=False)
    async def cors_test(request: Request):
        """Test endpoint for CORS configuration."""
        origin = request.headers.get("origin", "Not provided")
        is_valid, cors_origin = CORSValidator(config).validate_origin(origin)
        
        return {
            "origin": origin,
            "is_valid": is_valid,
            "cors_origin": cors_origin.origin if cors_origin else None,
            "allowed_origins": [str(o) for o in config.allowed_origins],
            "allow_credentials": config.allow_credentials,
            "policy": config.policy.value,
        }
    
    # Add CORS violation report endpoint
    @app.post("/api/v1/security/cors-violation", include_in_schema=False)
    async def report_cors_violation(request: Request):
        """Endpoint to report CORS violations (for monitoring)."""
        try:
            data = await request.json()
            audit_logger = AuditLogger()
            
            audit_logger.log_security_event(
                event_type=AuditEventType.SUSPICIOUS_ACTIVITY,
                description="CORS Violation Reported",
                details=data,
                severity=AuditSeverity.WARNING
            )
            
            return {"status": "received"}
        
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    print(f"CORS configured with policy: {config.policy.value}")
    print(f"Allowed origins: {[str(o) for o in config.allowed_origins]}")
    
    return app


def get_cors_config() -> CORSConfig:
    """Get current CORS configuration."""
    # This would typically load from database or config file
    return CORSConfig()


def validate_origin(origin: str, config: Optional[CORSConfig] = None) -> bool:
    """
    Validate an origin against CORS configuration.
    
    Args:
        origin: Origin to validate
        config: Optional CORS configuration
    
    Returns:
        True if origin is allowed
    """
    if not config:
        config = get_cors_config()
    
    validator = CORSValidator(config)
    is_valid, _ = validator.validate_origin(origin)
    
    return is_valid


def parse_origin_pattern(pattern: str) -> Optional[Pattern]:
    """
    Parse an origin pattern into a regex.
    
    Args:
        pattern: Pattern string (e.g., "*.example.com", "https://*.example.com:8080")
    
    Returns:
        Compiled regex pattern or None if invalid
    """
    if not pattern:
        return None
    
    # Convert wildcard pattern to regex
    regex_pattern = re.escape(pattern)
    regex_pattern = regex_pattern.replace(r"\*", ".*")
    regex_pattern = regex_pattern.replace(r"\?", ".")
    
    # Add start and end anchors
    if not regex_pattern.startswith("^"):
        regex_pattern = "^" + regex_pattern
    if not regex_pattern.endswith("$"):
        regex_pattern = regex_pattern + "$"
    
    try:
        return re.compile(regex_pattern)
    except re.error:
        return None


# Export main components
__all__ = [
    # Classes
    "CORSConfig",
    "CORSOrigin",
    "CORSValidator",
    "DynamicCORSResolver",
    "SecureCORSMiddleware",
    
    # Enums
    "CORSPolicy",
    "CredentialsPolicy",
    
    # Functions
    "setup_cors",
    "get_cors_config",
    "validate_origin",
    "parse_origin_pattern",
    
    # Constants
    "CORSPolicy",
    "CredentialsPolicy",
]