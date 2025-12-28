"""
Common FastAPI dependencies used across the API.
"""
from typing import Optional, Generator, List, Dict, Any, AsyncGenerator
from datetime import datetime, timedelta

from fastapi import Depends, HTTPException, status, Query, Request
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from jose import JWTError, jwt

from app.core.config import settings
from app.core.database import get_db
from app.core.security import verify_token
from app.db.models import User, RateLimit
from app.schemas.common import PaginationParams

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login",
    auto_error=False
)

# HTTP Bearer scheme for API key authentication
http_bearer = HTTPBearer(auto_error=False)


async def get_current_user_optional(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    Get current user if authenticated, otherwise return None.
    Doesn't raise exceptions for unauthenticated users.
    """
    if not token:
        return None
    
    try:
        payload = verify_token(token)
        if payload is None:
            return None
        
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
        
        user = db.query(User).filter(User.id == int(user_id)).first()
        if user is None or not user.is_active:
            return None
        
        # Update last active timestamp
        user.last_active = datetime.utcnow()
        db.commit()
        
        return user
    
    except JWTError:
        return None


class RateLimiter:
    """
    Rate limiting dependency.
    """
    def __init__(self, requests_per_minute: int = 60, identifier: str = "ip"):
        self.requests_per_minute = requests_per_minute
        self.identifier = identifier  # "ip", "user", or "endpoint"
    
    async def __call__(
        self,
        request: Request,
        db: Session = Depends(get_db),
        current_user: Optional[User] = Depends(get_current_user_optional)
    ) -> None:
        """
        Check if request should be rate limited.
        """
        # Determine identifier
        if self.identifier == "user" and current_user:
            identifier = f"user:{current_user.id}"
        elif self.identifier == "ip":
            identifier = request.client.host
        else:
            identifier = f"endpoint:{request.url.path}"
        
        # Check rate limit
        one_minute_ago = datetime.utcnow() - timedelta(minutes=1)
        
        request_count = db.query(RateLimit).filter(
            RateLimit.identifier == identifier,
            RateLimit.endpoint == request.url.path,
            RateLimit.timestamp > one_minute_ago
        ).count()
        
        if request_count >= self.requests_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUICTS,
                detail="Rate limit exceeded",
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": "60"
                }
            )
        
        # Record this request
        rate_limit = RateLimit(
            identifier=identifier,
            endpoint=request.url.path,
            timestamp=datetime.utcnow(),
            user_id=current_user.id if current_user else None,
            ip_address=request.client.host
        )
        db.add(rate_limit)
        db.commit()
        
        # Calculate remaining requests
        remaining = self.requests_per_minute - request_count - 1
        
        # Add rate limit headers
        request.state.rate_limit_headers = {
            "X-RateLimit-Limit": str(self.requests_per_minute),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": "60"
        }


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    Get current authenticated user from JWT token.
    Raises 401 if token is invalid or user doesn't exist.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if not token:
        raise credentials_exception
    
    try:
        payload = verify_token(token)
        if payload is None:
            raise credentials_exception
        
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        
        user = db.query(User).filter(User.id == int(user_id)).first()
        if user is None or not user.is_active:
            raise credentials_exception
        
        # Update last active timestamp
        user.last_active = datetime.utcnow()
        db.commit()
        
        return user
    
    except JWTError:
        raise credentials_exception


async def get_current_admin_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current user and verify they have admin role.
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


async def get_current_admin_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current user and verify they have admin role.
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


async def get_current_moderator_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current user and verify they have moderator or admin role.
    """
    if current_user.role not in ["moderator", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Moderator access required"
        )
    return current_user


async def get_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer),
    db: Session = Depends(get_db)
) -> User:
    """
    Authenticate using API key (for server-to-server communication).
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    api_key = credentials.credentials
    
    # Look up API key in database
    from app.db.models import APIKey
    api_key_record = db.query(APIKey).filter(
        APIKey.key == api_key,
        APIKey.is_active == True,
        APIKey.expires_at > datetime.utcnow()
    ).first()
    
    if not api_key_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key"
        )
    
    # Get associated user
    user = db.query(User).filter(User.id == api_key_record.user_id).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is inactive"
        )
    
    # Update last used timestamp
    api_key_record.last_used_at = datetime.utcnow()
    db.commit()
    
    return user


def get_pagination_params(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    sort_by: Optional[str] = Query(None, description="Field to sort by"),
    sort_order: str = Query("desc", description="Sort order: asc or desc"),
    filters: Optional[str] = Query(None, description="JSON encoded filters")
) -> PaginationParams:
    """
    Common pagination parameters dependency.
    """
    # Parse filters if provided
    parsed_filters = None
    if filters:
        try:
            import json
            parsed_filters = json.loads(filters)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid filters format"
            )
    
    return PaginationParams(
        page=page,
        per_page=per_page,
        offset=(page - 1) * per_page,
        limit=per_page,
        sort_by=sort_by,
        sort_order=sort_order,
        filters=parsed_filters
    )


class PaginationDependency:
    """
    Alternative pagination dependency for more complex scenarios.
    """
    def __init__(self, default_per_page: int = 20, max_per_page: int = 100):
        self.default_per_page = default_per_page
        self.max_per_page = max_per_page
    
    def __call__(
        self,
        page: int = Query(1, ge=1, description="Page number"),
        per_page: int = Query(None, ge=1, le=100, description="Items per page"),
        cursor: Optional[str] = Query(None, description="Cursor for pagination"),
        before: Optional[str] = Query(None, description="Get items before cursor"),
        after: Optional[str] = Query(None, description="Get items after cursor")
    ) -> Dict[str, Any]:
        """
        Return pagination parameters.
        Supports both offset-based and cursor-based pagination.
        """
        if per_page is None:
            per_page = self.default_per_page
        
        per_page = min(per_page, self.max_per_page)
        
        return {
            "page": page,
            "per_page": per_page,
            "offset": (page - 1) * per_page if cursor is None else None,
            "limit": per_page,
            "cursor": cursor,
            "before": before,
            "after": after,
            "pagination_type": "cursor" if cursor else "offset"
        }


async def get_db_session() -> AsyncGenerator[Session, None]:
    """
    Get database session dependency.
    This is the same as get_db but included here for consistency.
    """
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()


async def get_request_id(request: Request) -> str:
    """
    Get request ID from headers or generate a new one.
    """
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        import uuid
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
    return request_id


async def get_user_agent(request: Request) -> str:
    """
    Get user agent from request headers.
    """
    return request.headers.get("User-Agent", "Unknown")


async def get_client_ip(request: Request) -> str:
    """
    Get client IP address.
    Handles proxies and load balancers.
    """
    if request.client is None:
        return "0.0.0.0"
    
    # Check for X-Forwarded-For header
    x_forwarded_for = request.headers.get("X-Forwarded-For")
    if x_forwarded_for:
        # Take the first IP in the list
        ip = x_forwarded_for.split(",")[0].strip()
        return ip
    
    # Check for X-Real-IP header
    x_real_ip = request.headers.get("X-Real-IP")
    if x_real_ip:
        return x_real_ip
    
    # Fall back to client host
    return request.client.host


async def get_geo_location(
    request: Request,
    db: Session = Depends(get_db)
) -> Optional[Dict[str, Any]]:
    """
    Get geographic location from IP address.
    Requires geoip2 database or external service.
    """
    ip_address = await get_client_ip(request)
    
    # Skip private IPs
    if ip_address in ["127.0.0.1", "localhost", "::1"]:
        return None
    
    try:
        # Example using geoip2 (requires geoip2 package and database)
        # import geoip2.database
        # with geoip2.database.Reader('/path/to/GeoLite2-City.mmdb') as reader:
        #     response = reader.city(ip_address)
        #     return {
        #         "country": response.country.name,
        #         "country_code": response.country.iso_code,
        #         "city": response.city.name,
        #         "latitude": response.location.latitude,
        #         "longitude": response.location.longitude,
        #     }
        
        # For now, return None (placeholder)
        return None
    
    except Exception:
        # Log error but don't fail the request
        return None


class FeatureFlagDependency:
    """
    Dependency for feature flags.
    """
    def __init__(self, feature_name: str, default_enabled: bool = False):
        self.feature_name = feature_name
        self.default_enabled = default_enabled
    
    async def __call__(
        self,
        request: Request,
        current_user: Optional[User] = Depends(get_current_user_optional)
    ) -> bool:
        """
        Check if a feature is enabled for the current request/user.
        """
        # Check feature flags (could be from database, Redis, or external service)
        # For now, return default
        return self.default_enabled


async def validate_content_type(
    request: Request,
    allowed_types: List[str] = ["application/json"]
) -> None:
    """
    Validate request content type.
    """
    content_type = request.headers.get("Content-Type", "")
    
    if not any(allowed_type in content_type for allowed_type in allowed_types):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type. Expected: {', '.join(allowed_types)}"
        )


async def validate_request_size(
    request: Request,
    max_size_mb: int = 10
) -> None:
    """
    Validate request body size.
    """
    content_length = request.headers.get("Content-Length")
    if content_length:
        size_mb = int(content_length) / (1024 * 1024)
        if size_mb > max_size_mb:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Request body too large. Maximum size is {max_size_mb}MB"
            )


async def get_audit_context(
    request: Request,
    current_user: Optional[User] = Depends(get_current_user_optional),
    client_ip: str = Depends(get_client_ip),
    user_agent: str = Depends(get_user_agent)
) -> Dict[str, Any]:
    """
    Get audit context for logging.
    """
    return {
        "user_id": current_user.id if current_user else None,
        "username": current_user.username if current_user else None,
        "ip_address": client_ip,
        "user_agent": user_agent,
        "endpoint": request.url.path,
        "method": request.method,
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": await get_request_id(request),
    }


# Export commonly used dependencies
__all__ = [
    "get_current_user",
    "get_current_user_optional",
    "get_current_admin_user",
    "get_current_moderator_user",
    "get_api_key",
    "get_pagination_params",
    "PaginationDependency",
    "RateLimiter",
    "get_db_session",
    "get_request_id",
    "get_user_agent",
    "get_client_ip",
    "get_geo_location",
    "FeatureFlagDependency",
    "validate_content_type",
    "validate_request_size",
    "get_audit_context",
]