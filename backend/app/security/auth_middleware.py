"""
Authentication and Authorization Middleware for WorldBrief 360

This module provides:
- JWT token validation and management
- Role-based access control (RBAC)
- Permission-based authorization
- API key authentication
- OAuth2 token validation
- Request context and user extraction
- Token blacklisting
- Session management
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

import jwt
import redis
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from jose import JWTError, jwk
from jose.exceptions import JWTClaimsError, ExpiredSignatureError
from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    TokenExpiredError,
    InvalidTokenError,
    RateLimitExceededError,
)
from app.db.models.user import User, UserRole, Permission
from app.db.session import get_db_session
from app.security.audit_logger import AuditLogger, AuditEventType, AuditContext
from app.security.rate_limiter import RateLimiter

# Get settings
settings = get_settings()

# Security configurations
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES or 30
REFRESH_TOKEN_EXPIRE_DAYS = settings.REFRESH_TOKEN_EXPIRE_DAYS or 7
API_KEY_EXPIRE_DAYS = settings.API_KEY_EXPIRE_DAYS or 90
TOKEN_REFRESH_THRESHOLD = 300  # Refresh token 5 minutes before expiry

# OAuth2 configuration
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_PREFIX}/auth/login",
    auto_error=False
)

# HTTP Bearer for token extraction
bearer_scheme = HTTPBearer(auto_error=False)


class TokenType(str, Enum):
    """Types of authentication tokens."""
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    SESSION = "session"


class AuthMethod(str, Enum):
    """Authentication methods."""
    JWT = "jwt"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    SESSION = "session"
    BASIC = "basic"


@dataclass
class TokenPayload:
    """Decoded JWT token payload."""
    sub: str  # Subject (user ID)
    email: Optional[str] = None
    username: Optional[str] = None
    role: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    token_type: TokenType = TokenType.ACCESS
    exp: Optional[datetime] = None
    iat: Optional[datetime] = None
    jti: Optional[str] = None  # JWT ID for blacklisting
    iss: Optional[str] = None  # Issuer
    aud: Optional[str] = None  # Audience
    scopes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthContext:
    """Authentication context for the current request."""
    user_id: Optional[str] = None
    username: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    permissions: Set[str] = field(default_factory=set)
    token_type: Optional[TokenType] = None
    auth_method: Optional[AuthMethod] = None
    session_id: Optional[str] = None
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_authenticated: bool = False
    is_active: bool = False
    is_verified: bool = False
    requires_mfa: bool = False
    mfa_verified: bool = False
    token_expiry: Optional[datetime] = None
    scopes: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


class JWTTokenHandler:
    """Handler for JWT token creation, validation, and management."""
    
    def __init__(self, secret_key: str, algorithm: str = JWT_ALGORITHM):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.redis_client = self._init_redis()
        self.audit_logger = AuditLogger()
    
    def _init_redis(self) -> Optional[redis.Redis]:
        """Initialize Redis client for token blacklisting."""
        if not settings.REDIS_URL:
            return None
        
        try:
            return redis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
        except Exception as e:
            print(f"Redis connection failed: {e}")
            return None
    
    def create_access_token(
        self,
        user_id: str,
        email: str,
        username: str,
        role: str,
        permissions: List[str],
        expires_delta: Optional[timedelta] = None,
        metadata: Optional[Dict[str, Any]] = None,
        jti: Optional[str] = None
    ) -> str:
        """
        Create a JWT access token.
        
        Args:
            user_id: User ID
            email: User email
            username: Username
            role: User role
            permissions: List of permissions
            expires_delta: Optional expiration time delta
            metadata: Additional token metadata
            jti: JWT ID for blacklisting
            
        Returns:
            JWT token string
        """
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        payload = {
            "sub": user_id,
            "email": email,
            "username": username,
            "role": role,
            "permissions": permissions,
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": jti or str(uuid.uuid4()),
            "iss": settings.JWT_ISSUER or "worldbrief360",
            "aud": settings.JWT_AUDIENCE or "worldbrief360_users",
            "type": TokenType.ACCESS.value,
            "metadata": metadata or {},
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        # Store token in Redis for fast validation (optional)
        if self.redis_client:
            key = f"token:{payload['jti']}"
            expiry_seconds = int((expire - datetime.utcnow()).total_seconds())
            self.redis_client.setex(
                key,
                expiry_seconds,
                json.dumps({
                    "user_id": user_id,
                    "type": TokenType.ACCESS.value,
                    "valid": True
                })
            )
        
        return token
    
    def create_refresh_token(
        self,
        user_id: str,
        device_id: Optional[str] = None
    ) -> str:
        """
        Create a refresh token.
        
        Args:
            user_id: User ID
            device_id: Optional device identifier
            
        Returns:
            Refresh token string
        """
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        jti = str(uuid.uuid4())
        
        payload = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": jti,
            "type": TokenType.REFRESH.value,
            "device_id": device_id,
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        # Store refresh token in Redis
        if self.redis_client:
            key = f"refresh_token:{jti}"
            expiry_seconds = int((expire - datetime.utcnow()).total_seconds())
            self.redis_client.setex(
                key,
                expiry_seconds,
                json.dumps({
                    "user_id": user_id,
                    "device_id": device_id,
                    "valid": True
                })
            )
        
        return token
    
    def decode_token(self, token: str) -> TokenPayload:
        """
        Decode and validate a JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded token payload
            
        Raises:
            InvalidTokenError: If token is invalid
            TokenExpiredError: If token has expired
        """
        try:
            # Decode token
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_aud": True,
                    "verify_iss": True,
                },
                issuer=settings.JWT_ISSUER or "worldbrief360",
                audience=settings.JWT_AUDIENCE or "worldbrief360_users",
            )
            
            # Check blacklist
            if self.is_token_blacklisted(payload.get("jti")):
                raise InvalidTokenError("Token has been revoked")
            
            # Convert to TokenPayload
            return TokenPayload(
                sub=payload.get("sub"),
                email=payload.get("email"),
                username=payload.get("username"),
                role=payload.get("role"),
                permissions=payload.get("permissions", []),
                token_type=TokenType(payload.get("type", TokenType.ACCESS.value)),
                exp=datetime.fromtimestamp(payload.get("exp")) if payload.get("exp") else None,
                iat=datetime.fromtimestamp(payload.get("iat")) if payload.get("iat") else None,
                jti=payload.get("jti"),
                iss=payload.get("iss"),
                aud=payload.get("aud"),
                scopes=payload.get("scopes", []),
                metadata=payload.get("metadata", {})
            )
            
        except ExpiredSignatureError:
            raise TokenExpiredError("Token has expired")
        except JWTError as e:
            raise InvalidTokenError(f"Invalid token: {str(e)}")
        except Exception as e:
            raise InvalidTokenError(f"Token validation failed: {str(e)}")
    
    def verify_token(self, token: str) -> bool:
        """
        Verify if a token is valid.
        
        Args:
            token: JWT token string
            
        Returns:
            True if token is valid, False otherwise
        """
        try:
            self.decode_token(token)
            return True
        except (InvalidTokenError, TokenExpiredError):
            return False
    
    def blacklist_token(self, jti: str, expiry_seconds: int = 86400):
        """
        Blacklist a token by its JTI.
        
        Args:
            jti: JWT ID
            expiry_seconds: Blacklist expiration in seconds
        """
        if self.redis_client:
            key = f"blacklist:{jti}"
            self.redis_client.setex(key, expiry_seconds, "true")
    
    def is_token_blacklisted(self, jti: Optional[str]) -> bool:
        """
        Check if a token is blacklisted.
        
        Args:
            jti: JWT ID
            
        Returns:
            True if token is blacklisted, False otherwise
        """
        if not jti or not self.redis_client:
            return False
        
        key = f"blacklist:{jti}"
        return self.redis_client.exists(key) > 0
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """
        Refresh an access token using a refresh token.
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            New access token or None if refresh failed
        """
        try:
            # Decode refresh token
            payload = self.decode_token(refresh_token)
            
            if payload.token_type != TokenType.REFRESH:
                return None
            
            # Check if refresh token is valid in Redis
            if self.redis_client:
                key = f"refresh_token:{payload.jti}"
                if not self.redis_client.exists(key):
                    return None
            
            # Get user from database
            db_session = next(get_db_session())
            user = db_session.query(User).filter(User.id == payload.sub).first()
            
            if not user or not user.is_active:
                return None
            
            # Create new access token
            permissions = [p.name for p in user.permissions]
            new_token = self.create_access_token(
                user_id=user.id,
                email=user.email,
                username=user.username,
                role=user.role.value,
                permissions=permissions,
                metadata={"refreshed_from": payload.jti}
            )
            
            return new_token
            
        except Exception as e:
            print(f"Token refresh failed: {e}")
            return None
        finally:
            if 'db_session' in locals():
                db_session.close()


class APIKeyHandler:
    """Handler for API key authentication."""
    
    def __init__(self):
        self.redis_client = self._init_redis()
    
    def _init_redis(self) -> Optional[redis.Redis]:
        """Initialize Redis client for API key caching."""
        if not settings.REDIS_URL:
            return None
        
        try:
            return redis.from_url(settings.REDIS_URL, decode_responses=True)
        except Exception:
            return None
    
    def validate_api_key(self, api_key: str, db_session: Session) -> Optional[User]:
        """
        Validate an API key.
        
        Args:
            api_key: API key string
            db_session: Database session
            
        Returns:
            User object if valid, None otherwise
        """
        # Check cache first
        if self.redis_client:
            cached_user = self.redis_client.get(f"api_key:{api_key}")
            if cached_user:
                user_data = json.loads(cached_user)
                return User(**user_data)  # Simplified
        
        # Check database
        from app.db.models.api_key import APIKey as APIKeyModel
        
        api_key_obj = db_session.query(APIKeyModel)\
            .filter(
                APIKeyModel.key == api_key,
                APIKeyModel.is_active == True,
                APIKeyModel.expires_at > datetime.utcnow()
            )\
            .first()
        
        if not api_key_obj:
            return None
        
        user = api_key_obj.user
        if not user or not user.is_active:
            return None
        
        # Cache the result
        if self.redis_client:
            user_data = {
                "id": user.id,
                "email": user.email,
                "username": user.username,
                "role": user.role.value,
            }
            self.redis_client.setex(
                f"api_key:{api_key}",
                300,  # Cache for 5 minutes
                json.dumps(user_data)
            )
        
        # Update last used timestamp
        api_key_obj.last_used_at = datetime.utcnow()
        db_session.commit()
        
        return user


class SecurityMiddleware:
    """
    FastAPI middleware for security features.
    
    Provides:
    - Authentication validation
    - Request context extraction
    - Rate limiting
    - Security headers
    - Audit logging
    - Request/Response modification
    """
    
    def __init__(
        self,
        app,
        jwt_handler: Optional[JWTTokenHandler] = None,
        api_key_handler: Optional[APIKeyHandler] = None,
        rate_limiter: Optional[RateLimiter] = None,
        enable_auth: bool = True,
        enable_rate_limiting: bool = True,
        enable_audit_logging: bool = True,
        enable_cors: bool = True
    ):
        self.app = app
        self.jwt_handler = jwt_handler or JWTTokenHandler(settings.SECRET_KEY)
        self.api_key_handler = api_key_handler or APIKeyHandler()
        self.rate_limiter = rate_limiter or RateLimiter()
        self.enable_auth = enable_auth
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_audit_logging = enable_audit_logging
        self.enable_cors = enable_cors
        self.audit_logger = AuditLogger()
        
        # Public endpoints that don't require authentication
        self.public_endpoints = {
            "/api/v1/auth/login",
            "/api/v1/auth/register",
            "/api/v1/auth/refresh",
            "/api/v1/auth/forgot-password",
            "/api/v1/auth/reset-password",
            "/api/v1/health",
            "/api/v1/docs",
            "/api/v1/redoc",
            "/api/v1/openapi.json",
        }
        
        # Rate limit exempt endpoints
        self.rate_limit_exempt = {
            "/api/v1/health",
            "/api/v1/auth/login",  # But we'll still rate limit this differently
        }
    
    async def __call__(self, request: Request, call_next):
        """
        Process incoming request.
        """
        start_time = time.time()
        
        # Create request context
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        correlation_id = request.headers.get("X-Correlation-ID", "")
        
        # Add request ID to headers
        request.scope["headers"].append(
            (b"x-request-id", request_id.encode())
        )
        
        # Create audit context
        audit_context = AuditContext(
            correlation_id=correlation_id,
            request_id=request_id,
            session_id=request.cookies.get("session_id", ""),
            user_agent=request.headers.get("User-Agent", ""),
            ip_address=request.client.host if request.client else "",
            location=None,
            device_info=None,
            request_path=request.url.path,
            request_method=request.method,
            query_params=dict(request.query_params)
        )
        
        # Store in request state for later use
        request.state.request_id = request_id
        request.state.correlation_id = correlation_id
        request.state.audit_context = audit_context
        
        # Apply rate limiting
        if self.enable_rate_limiting and request.url.path not in self.rate_limit_exempt:
            try:
                await self._apply_rate_limiting(request)
            except RateLimitExceededError as e:
                return self._create_rate_limit_response(request, e)
        
        # Authentication and authorization
        auth_context = None
        if self.enable_auth and not self._is_public_endpoint(request.url.path):
            try:
                auth_context = await self._authenticate_request(request)
                request.state.auth_context = auth_context
                
                # Check authorization for the endpoint
                await self._authorize_request(request, auth_context)
                
            except AuthenticationError as e:
                return self._create_auth_error_response(request, e, status.HTTP_401_UNAUTHORIZED)
            except AuthorizationError as e:
                return self._create_auth_error_response(request, e, status.HTTP_403_FORBIDDEN)
            except Exception as e:
                return self._create_auth_error_response(
                    request,
                    AuthenticationError("Authentication failed"),
                    status.HTTP_401_UNAUTHORIZED
                )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Add security headers
            response = self._add_security_headers(response)
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            if correlation_id:
                response.headers["X-Correlation-ID"] = correlation_id
            
            # Audit log successful request
            if self.enable_audit_logging:
                self._log_request(
                    request=request,
                    response=response,
                    auth_context=auth_context,
                    duration_ms=duration_ms,
                    success=True
                )
            
            return response
            
        except Exception as e:
            # Audit log failed request
            if self.enable_audit_logging:
                self._log_request(
                    request=request,
                    response=None,
                    auth_context=auth_context,
                    duration_ms=(time.time() - start_time) * 1000,
                    success=False,
                    error=str(e)
                )
            raise
    
    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public."""
        # Exact match
        if path in self.public_endpoints:
            return True
        
        # Pattern matching (e.g., /api/v1/auth/*)
        for public_path in self.public_endpoints:
            if public_path.endswith("*") and path.startswith(public_path[:-1]):
                return True
        
        return False
    
    async def _apply_rate_limiting(self, request: Request):
        """Apply rate limiting to request."""
        identifier = self._get_rate_limit_identifier(request)
        endpoint = request.url.path
        
        # Different limits for different endpoints
        if endpoint.startswith("/api/v1/auth/login"):
            limit = 5  # 5 attempts per minute for login
            window = 60
        else:
            limit = 100  # 100 requests per minute for other endpoints
            window = 60
        
        if not await self.rate_limiter.is_allowed(identifier, endpoint, limit, window):
            raise RateLimitExceededError(
                f"Rate limit exceeded for {identifier}. Limit: {limit} requests per {window} seconds"
            )
    
    def _get_rate_limit_identifier(self, request: Request) -> str:
        """Get identifier for rate limiting."""
        # Try to use authenticated user ID
        if hasattr(request.state, 'auth_context') and request.state.auth_context:
            return f"user:{request.state.auth_context.user_id}"
        
        # Fall back to IP address
        return f"ip:{request.client.host if request.client else 'unknown'}"
    
    async def _authenticate_request(self, request: Request) -> AuthContext:
        """Authenticate the incoming request."""
        # Try different authentication methods
        auth_context = await self._authenticate_jwt(request)
        if auth_context and auth_context.is_authenticated:
            return auth_context
        
        auth_context = await self._authenticate_api_key(request)
        if auth_context and auth_context.is_authenticated:
            return auth_context
        
        auth_context = await self._authenticate_session(request)
        if auth_context and auth_context.is_authenticated:
            return auth_context
        
        # No valid authentication found
        raise AuthenticationError("Authentication required")
    
    async def _authenticate_jwt(self, request: Request) -> AuthContext:
        """Authenticate using JWT token."""
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return AuthContext()
        
        # Check Bearer token
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix
            
            try:
                # Decode and validate token
                payload = self.jwt_handler.decode_token(token)
                
                # Get user from database for additional validation
                db_session = next(get_db_session())
                user = db_session.query(User)\
                    .filter(User.id == payload.sub, User.is_active == True)\
                    .first()
                
                if not user:
                    raise AuthenticationError("User not found or inactive")
                
                # Create auth context
                permissions = {p.name for p in user.permissions}
                
                return AuthContext(
                    user_id=user.id,
                    username=user.username,
                    email=user.email,
                    role=user.role.value,
                    permissions=permissions,
                    token_type=payload.token_type,
                    auth_method=AuthMethod.JWT,
                    is_authenticated=True,
                    is_active=user.is_active,
                    is_verified=user.is_verified,
                    requires_mfa=user.mfa_enabled,
                    token_expiry=payload.exp,
                    scopes=set(payload.scopes),
                    metadata=payload.metadata,
                    ip_address=request.client.host if request.client else None,
                    user_agent=request.headers.get("User-Agent"),
                )
                
            except (InvalidTokenError, TokenExpiredError) as e:
                # Log failed authentication attempt
                self.audit_logger.log_security_event(
                    event_type=AuditEventType.LOGIN_FAILED,
                    description=f"JWT authentication failed: {str(e)}",
                    severity="WARNING",
                    context=request.state.audit_context
                )
                raise AuthenticationError(f"Invalid token: {str(e)}")
            finally:
                if 'db_session' in locals():
                    db_session.close()
        
        return AuthContext()
    
    async def _authenticate_api_key(self, request: Request) -> AuthContext:
        """Authenticate using API key."""
        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if not api_key:
            return AuthContext()
        
        try:
            db_session = next(get_db_session())
            user = self.api_key_handler.validate_api_key(api_key, db_session)
            
            if not user:
                raise AuthenticationError("Invalid API key")
            
            permissions = {p.name for p in user.permissions}
            
            return AuthContext(
                user_id=user.id,
                username=user.username,
                email=user.email,
                role=user.role.value,
                permissions=permissions,
                token_type=TokenType.API_KEY,
                auth_method=AuthMethod.API_KEY,
                is_authenticated=True,
                is_active=user.is_active,
                is_verified=user.is_verified,
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("User-Agent"),
            )
            
        except Exception as e:
            self.audit_logger.log_security_event(
                event_type=AuditEventType.LOGIN_FAILED,
                description=f"API key authentication failed: {str(e)}",
                severity="WARNING",
                context=request.state.audit_context
            )
            raise AuthenticationError(f"API key authentication failed: {str(e)}")
        finally:
            if 'db_session' in locals():
                db_session.close()
    
    async def _authenticate_session(self, request: Request) -> AuthContext:
        """Authenticate using session cookie."""
        session_id = request.cookies.get("session_id")
        if not session_id:
            return AuthContext()
        
        # Implement session validation logic here
        # This is a simplified example
        return AuthContext()
    
    async def _authorize_request(self, request: Request, auth_context: AuthContext):
        """Authorize the request based on user permissions."""
        if not auth_context.is_authenticated:
            raise AuthorizationError("Authentication required")
        
        # Check if user is active
        if not auth_context.is_active:
            raise AuthorizationError("User account is inactive")
        
        # Check endpoint-specific permissions
        endpoint = request.url.path
        method = request.method
        
        # Define endpoint permission requirements
        permission_map = {
            ("/api/v1/admin/", "GET"): "admin.read",
            ("/api/v1/admin/", "POST"): "admin.write",
            ("/api/v1/admin/", "PUT"): "admin.write",
            ("/api/v1/admin/", "DELETE"): "admin.delete",
            ("/api/v1/users/", "GET"): "user.read",
            ("/api/v1/users/", "POST"): "user.write",
            ("/api/v1/users/", "PUT"): "user.write",
            ("/api/v1/users/", "DELETE"): "user.delete",
        }
        
        # Check if endpoint requires specific permission
        for (path_prefix, http_method), required_permission in permission_map.items():
            if endpoint.startswith(path_prefix) and method == http_method:
                if required_permission not in auth_context.permissions:
                    raise AuthorizationError(
                        f"Insufficient permissions. Required: {required_permission}"
                    )
        
        # Additional authorization checks can be added here
        # e.g., resource ownership, time-based access, etc.
    
    def _add_security_headers(self, response: Response) -> Response:
        """Add security headers to response."""
        security_headers = {
            "X-Frame-Options": "DENY",
            "X-Content-Type-Options": "nosniff",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';",
        }
        
        if settings.ENVIRONMENT == "production":
            security_headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response
    
    def _log_request(
        self,
        request: Request,
        response: Optional[Response],
        auth_context: Optional[AuthContext],
        duration_ms: float,
        success: bool,
        error: Optional[str] = None
    ):
        """Log request details for auditing."""
        event_type = AuditEventType.API_CALL
        severity = "INFO" if success else "ERROR"
        
        details = {
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "duration_ms": duration_ms,
            "status_code": response.status_code if response else None,
            "user_agent": request.headers.get("User-Agent"),
            "ip_address": request.client.host if request.client else None,
        }
        
        if error:
            details["error"] = error
        
        self.audit_logger.log_security_event(
            event_type=event_type,
            user_id=auth_context.user_id if auth_context else None,
            username=auth_context.username if auth_context else None,
            description=f"API Request: {request.method} {request.url.path}",
            details=details,
            severity=severity,
            success=success,
            context=request.state.audit_context if hasattr(request.state, 'audit_context') else None
        )
    
    def _create_rate_limit_response(self, request: Request, error: RateLimitExceededError) -> Response:
        """Create rate limit exceeded response."""
        self.audit_logger.log_security_event(
            event_type=AuditEventType.RATE_LIMIT_EXCEEDED,
            description=f"Rate limit exceeded for {request.url.path}",
            details={
                "ip_address": request.client.host if request.client else None,
                "path": request.url.path,
                "method": request.method,
            },
            severity="WARNING",
            success=False,
            context=request.state.audit_context
        )
        
        return Response(
            content=json.dumps({
                "detail": str(error),
                "error_code": "RATE_LIMIT_EXCEEDED"
            }),
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            media_type="application/json",
            headers={
                "Retry-After": "60",
                "X-RateLimit-Limit": "100",
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(time.time() + 60))
            }
        )
    
    def _create_auth_error_response(self, request: Request, error: Exception, status_code: int) -> Response:
        """Create authentication/authorization error response."""
        self.audit_logger.log_security_event(
            event_type=AuditEventType.ACCESS_DENIED,
            description=f"Access denied: {str(error)}",
            details={
                "ip_address": request.client.host if request.client else None,
                "path": request.url.path,
                "method": request.method,
                "error": str(error),
            },
            severity="WARNING",
            success=False,
            context=request.state.audit_context
        )
        
        return Response(
            content=json.dumps({
                "detail": str(error),
                "error_code": error.__class__.__name__.upper()
            }),
            status_code=status_code,
            media_type="application/json",
            headers={"WWW-Authenticate": "Bearer"}
        )


# FastAPI Dependencies
def get_jwt_handler() -> JWTTokenHandler:
    """Dependency to get JWT token handler."""
    return JWTTokenHandler(settings.SECRET_KEY)


def get_api_key_handler() -> APIKeyHandler:
    """Dependency to get API key handler."""
    return APIKeyHandler()


async def get_current_user(
    request: Request,
    jwt_handler: JWTTokenHandler = Depends(get_jwt_handler),
    api_key_handler: APIKeyHandler = Depends(get_api_key_handler),
    db_session: Session = Depends(get_db_session)
) -> User:
    """
    FastAPI dependency to get current authenticated user.
    
    Args:
        request: FastAPI request
        jwt_handler: JWT token handler
        api_key_handler: API key handler
        db_session: Database session
        
    Returns:
        Authenticated user
        
    Raises:
        HTTPException: If authentication fails
    """
    # Try JWT authentication first
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:]
        
        try:
            payload = jwt_handler.decode_token(token)
            user = db_session.query(User)\
                .filter(User.id == payload.sub, User.is_active == True)\
                .first()
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found or inactive"
                )
            
            return user
            
        except (InvalidTokenError, TokenExpiredError) as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e)
            )
    
    # Try API key authentication
    api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if api_key:
        user = api_key_handler.validate_api_key(api_key, db_session)
        if user:
            return user
    
    # No valid authentication found
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required"
    )


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Dependency to get current active user.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Active user
        
    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


async def get_current_verified_user(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """
    Dependency to get current verified user.
    
    Args:
        current_user: Current active user
        
    Returns:
        Verified user
        
    Raises:
        HTTPException: If user is not verified
    """
    if not current_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User not verified"
        )
    return current_user


# Role-based access control decorators
def require_role(required_role: Union[str, List[str]]):
    """
    Decorator to require specific role(s).
    
    Args:
        required_role: Required role or list of roles
    """
    def role_checker(user: User = Depends(get_current_active_user)):
        if isinstance(required_role, str):
            required_roles = [required_role]
        else:
            required_roles = required_role
        
        if user.role.value not in required_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role: {', '.join(required_roles)}"
            )
        return user
    
    return role_checker


def require_permission(required_permission: Union[str, List[str]]):
    """
    Decorator to require specific permission(s).
    
    Args:
        required_permission: Required permission or list of permissions
    """
    def permission_checker(user: User = Depends(get_current_active_user)):
        if isinstance(required_permission, str):
            required_permissions = [required_permission]
        else:
            required_permissions = required_permission
        
        user_permissions = {p.name for p in user.permissions}
        
        if not any(perm in user_permissions for perm in required_permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires permission: {', '.join(required_permissions)}"
            )
        return user
    
    return permission_checker


# Utility functions
def create_access_token(
    user_id: str,
    email: str,
    username: str,
    role: str,
    permissions: List[str],
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create a JWT access token."""
    jwt_handler = JWTTokenHandler(settings.SECRET_KEY)
    return jwt_handler.create_access_token(
        user_id=user_id,
        email=email,
        username=username,
        role=role,
        permissions=permissions,
        expires_delta=expires_delta
    )


def create_refresh_token(user_id: str, device_id: Optional[str] = None) -> str:
    """Create a refresh token."""
    jwt_handler = JWTTokenHandler(settings.SECRET_KEY)
    return jwt_handler.create_refresh_token(user_id, device_id)


def decode_token(token: str) -> TokenPayload:
    """Decode and validate a JWT token."""
    jwt_handler = JWTTokenHandler(settings.SECRET_KEY)
    return jwt_handler.decode_token(token)


def verify_token(token: str) -> bool:
    """Verify if a token is valid."""
    jwt_handler = JWTTokenHandler(settings.SECRET_KEY)
    return jwt_handler.verify_token(token)


def blacklist_token(jti: str, expiry_seconds: int = 86400):
    """Blacklist a token by its JTI."""
    jwt_handler = JWTTokenHandler(settings.SECRET_KEY)
    jwt_handler.blacklist_token(jti, expiry_seconds)


def refresh_access_token(refresh_token: str) -> Optional[str]:
    """Refresh an access token."""
    jwt_handler = JWTTokenHandler(settings.SECRET_KEY)
    return jwt_handler.refresh_access_token(refresh_token)


# Export main components
__all__ = [
    # Classes
    "JWTTokenHandler",
    "APIKeyHandler",
    "SecurityMiddleware",
    
    # Models
    "TokenType",
    "AuthMethod",
    "TokenPayload",
    "AuthContext",
    
    # Dependencies
    "get_current_user",
    "get_current_active_user",
    "get_current_verified_user",
    "require_role",
    "require_permission",
    
    # Utility functions
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "verify_token",
    "blacklist_token",
    "refresh_access_token",
    
    # Constants
    "ACCESS_TOKEN_EXPIRE_MINUTES",
    "REFRESH_TOKEN_EXPIRE_DAYS",
]