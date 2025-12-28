"""
API Key management model.
"""

import enum
import uuid
import secrets
from datetime import datetime, timedelta
from typing import Optional, List

from sqlalchemy import (
    Column,
    DateTime,
    Boolean,
    String,
    Text,
    Integer,
    ForeignKey,
    UniqueConstraint,
    Index,
    Enum,
    CheckConstraint,
    and_,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

from db.base import Base, TimestampMixin, generate_uuid


class APIKeyType(str, enum.Enum):
    """API Key types."""
    PUBLIC = "public"      # Read-only access
    PRIVATE = "private"    # Full access
    ADMIN = "admin"        # Administrative access
    WEBHOOK = "webhook"    # Webhook authentication
    INTEGRATION = "integration"  # Third-party integration


class APIKeyStatus(str, enum.Enum):
    """API Key status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    REVOKED = "revoked"
    EXPIRED = "expired"
    SUSPENDED = "suspended"


class APIKeyScope(str, enum.Enum):
    """API Key scopes/permissions."""
    # Read operations
    READ_USERS = "read:users"
    READ_ARTICLES = "read:articles"
    READ_INCIDENTS = "read:incidents"
    READ_COMMENTS = "read:comments"
    READ_ANALYTICS = "read:analytics"
    
    # Write operations
    WRITE_USERS = "write:users"
    WRITE_ARTICLES = "write:articles"
    WRITE_INCIDENTS = "write:incidents"
    WRITE_COMMENTS = "write:comments"
    
    # Admin operations
    ADMIN_USERS = "admin:users"
    ADMIN_ARTICLES = "admin:articles"
    ADMIN_INCIDENTS = "admin:incidents"
    ADMIN_SYSTEM = "admin:system"
    
    # Special operations
    UPLOAD_FILES = "upload:files"
    SEND_NOTIFICATIONS = "send:notifications"
    PROCESS_PAYMENTS = "process:payments"
    
    # Webhook operations
    WEBHOOK_RECEIVE = "webhook:receive"
    WEBHOOK_SEND = "webhook:send"


class APIKey(Base, TimestampMixin):
    """
    API Key model for authenticating external requests.
    """
    
    __tablename__ = "api_key"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    
    # Foreign keys
    user_id = Column(UUID(as_uuid=True), ForeignKey('user.id'), nullable=False, index=True)
    
    # Key information
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    key_type = Column(Enum(APIKeyType), default=APIKeyType.PRIVATE, nullable=False)
    status = Column(Enum(APIKeyStatus), default=APIKeyStatus.ACTIVE, nullable=False)
    
    # Key values
    api_key = Column(String(255), unique=True, nullable=False, index=True)
    api_secret = Column(String(255), nullable=True)  # Only for certain key types
    public_key = Column(Text, nullable=True)  # For asymmetric encryption
    
    # Permissions
    scopes = Column(JSONB, default=list)  # List of APIKeyScope values
    
    # Rate limiting
    rate_limit_per_minute = Column(Integer, default=60, nullable=False)
    rate_limit_per_hour = Column(Integer, default=1000, nullable=False)
    rate_limit_per_day = Column(Integer, default=10000, nullable=False)
    
    # Expiration
    expires_at = Column(DateTime(timezone=True), nullable=True)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    
    # Usage tracking
    usage_count = Column(Integer, default=0, nullable=False)
    last_ip_address = Column(String(45), nullable=True)  # Supports IPv6
    last_user_agent = Column(Text, nullable=True)
    
    # Security
    allowed_ips = Column(JSONB, default=list)  # List of allowed IP addresses/CIDR
    allowed_origins = Column(JSONB, default=list)  # List of allowed CORS origins
    allowed_referers = Column(JSONB, default=list)  # List of allowed HTTP referers
    
    # Metadata
    metadata = Column(JSONB, default=dict)
    tags = Column(JSONB, default=list)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    audit_logs = relationship("AuditLog", back_populates="api_key", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_api_key_user_status', user_id, status),
        Index('idx_api_key_expires', expires_at),
        Index('idx_api_key_type_status', key_type, status),
        UniqueConstraint('api_key', name='uq_api_key'),
        CheckConstraint('rate_limit_per_minute >= 0', name='check_rate_limit_minute'),
        CheckConstraint('rate_limit_per_hour >= 0', name='check_rate_limit_hour'),
        CheckConstraint('rate_limit_per_day >= 0', name='check_rate_limit_day'),
    )
    
    @validates('api_key')
    def validate_api_key(self, key, api_key):
        """Validate API key format."""
        if not api_key or len(api_key) < 20:
            raise ValueError("API key must be at least 20 characters")
        return api_key
    
    @validates('scopes')
    def validate_scopes(self, key, scopes):
        """Validate API key scopes."""
        if not isinstance(scopes, list):
            raise ValueError("Scopes must be a list")
        
        valid_scopes = {scope.value for scope in APIKeyScope}
        for scope in scopes:
            if scope not in valid_scopes:
                raise ValueError(f"Invalid scope: {scope}")
        
        return scopes
    
    @validates('allowed_ips')
    def validate_allowed_ips(self, key, allowed_ips):
        """Validate allowed IP addresses."""
        if not isinstance(allowed_ips, list):
            raise ValueError("Allowed IPs must be a list")
        
        # Simple IP validation (could be enhanced)
        import re
        ip_pattern = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?:/\d{1,2})?$')
        
        for ip in allowed_ips:
            if not ip_pattern.match(ip):
                raise ValueError(f"Invalid IP address or CIDR: {ip}")
        
        return allowed_ips
    
    @classmethod
    def generate_api_key(cls) -> str:
        """Generate a new API key."""
        # Generate a secure random string
        return secrets.token_urlsafe(32)  # 32 bytes = 43 characters
    
    @classmethod
    def generate_api_secret(cls) -> str:
        """Generate a new API secret."""
        return secrets.token_urlsafe(64)  # 64 bytes = 86 characters
    
    def is_active(self) -> bool:
        """Check if API key is active and not expired."""
        if self.status != APIKeyStatus.ACTIVE:
            return False
        
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        
        return True
    
    def is_expired(self) -> bool:
        """Check if API key has expired."""
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False
    
    def has_scope(self, scope: str) -> bool:
        """Check if API key has a specific scope."""
        return scope in (self.scopes or [])
    
    def has_any_scope(self, scopes: List[str]) -> bool:
        """Check if API key has any of the specified scopes."""
        key_scopes = self.scopes or []
        return any(scope in key_scopes for scope in scopes)
    
    def has_all_scopes(self, scopes: List[str]) -> bool:
        """Check if API key has all of the specified scopes."""
        key_scopes = self.scopes or []
        return all(scope in key_scopes for scope in scopes)
    
    def can_access_ip(self, ip_address: str) -> bool:
        """Check if IP address is allowed."""
        if not self.allowed_ips:
            return True
        
        import ipaddress
        
        try:
            request_ip = ipaddress.ip_address(ip_address)
            
            for allowed_ip in self.allowed_ips:
                if '/' in allowed_ip:
                    # CIDR notation
                    network = ipaddress.ip_network(allowed_ip, strict=False)
                    if request_ip in network:
                        return True
                else:
                    # Single IP address
                    allowed_ip_obj = ipaddress.ip_address(allowed_ip)
                    if request_ip == allowed_ip_obj:
                        return True
        except ValueError:
            # Invalid IP address
            return False
        
        return False
    
    def can_access_origin(self, origin: str) -> bool:
        """Check if origin is allowed."""
        if not self.allowed_origins:
            return True
        
        # Simple exact match for now
        # Could be enhanced with wildcard/regex matching
        return origin in self.allowed_origins
    
    def record_usage(self, ip_address: str = None, user_agent: str = None) -> None:
        """Record API key usage."""
        self.usage_count += 1
        self.last_used_at = datetime.utcnow()
        
        if ip_address:
            self.last_ip_address = ip_address
        
        if user_agent:
            self.last_user_agent = user_agent
    
    def revoke(self, reason: str = None) -> None:
        """Revoke the API key."""
        self.status = APIKeyStatus.REVOKED
        
        if reason and self.metadata:
            if 'revocation_reason' not in self.metadata:
                self.metadata['revocation_reason'] = []
            self.metadata['revocation_reason'].append({
                'reason': reason,
                'revoked_at': datetime.utcnow().isoformat()
            })
    
    def suspend(self, reason: str = None) -> None:
        """Suspend the API key."""
        self.status = APIKeyStatus.SUSPENDED
        
        if reason and self.metadata:
            if 'suspension_reason' not in self.metadata:
                self.metadata['suspension_reason'] = []
            self.metadata['suspension_reason'].append({
                'reason': reason,
                'suspended_at': datetime.utcnow().isoformat()
            })
    
    def activate(self) -> None:
        """Activate the API key."""
        self.status = APIKeyStatus.ACTIVE
    
    def get_rate_limit_key(self) -> str:
        """Get rate limit key for this API key."""
        return f"api_key:{self.api_key}"
    
    def get_usage_stats(self) -> dict:
        """Get API key usage statistics."""
        return {
            'usage_count': self.usage_count,
            'last_used_at': self.last_used_at.isoformat() if self.last_used_at else None,
            'last_ip_address': self.last_ip_address,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'is_active': self.is_active(),
            'is_expired': self.is_expired(),
        }
    
    def to_dict(self, include_secret: bool = False) -> dict:
        """
        Convert to dictionary representation.
        
        Args:
            include_secret: Whether to include the API secret
            
        Returns:
            Dictionary representation
        """
        data = super().to_dict()
        
        # Never expose the API secret in normal to_dict()
        if 'api_secret' in data and not include_secret:
            del data['api_secret']
        
        # Add computed fields
        data['is_active'] = self.is_active()
        data['is_expired'] = self.is_expired()
        data['has_expired'] = self.is_expired()
        
        return data


# API Key usage tracking (optional extension)
class APIKeyUsageLog(Base, TimestampMixin):
    """
    Detailed API key usage logging for audit purposes.
    """
    
    __tablename__ = "api_key_usage_log"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    
    # Foreign keys
    api_key_id = Column(UUID(as_uuid=True), ForeignKey('api_key.id'), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('user.id'), nullable=True, index=True)
    
    # Request information
    request_id = Column(String(100), nullable=True, index=True)
    endpoint = Column(String(255), nullable=False, index=True)
    method = Column(String(10), nullable=False)  # GET, POST, PUT, DELETE, etc.
    status_code = Column(Integer, nullable=False)
    
    # Request details
    ip_address = Column(String(45), nullable=False)  # Supports IPv6
    user_agent = Column(Text, nullable=True)
    referer = Column(Text, nullable=True)
    origin = Column(Text, nullable=True)
    
    # Performance metrics
    response_time_ms = Column(Integer, nullable=False)  # Response time in milliseconds
    request_size_bytes = Column(Integer, nullable=True)  # Request size in bytes
    response_size_bytes = Column(Integer, nullable=True)  # Response size in bytes
    
    # Request/Response data (truncated for large payloads)
    request_headers = Column(JSONB, nullable=True)
    request_body = Column(Text, nullable=True)
    response_headers = Column(JSONB, nullable=True)
    response_body = Column(Text, nullable=True)
    
    # Error information
    error_message = Column(Text, nullable=True)
    error_type = Column(String(100), nullable=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict)
    
    # Relationships
    api_key = relationship("APIKey", backref="usage_logs")
    user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('idx_api_key_usage_timestamp', api_key_id, 'created_at'),
        Index('idx_api_key_usage_endpoint', api_key_id, endpoint, 'created_at'),
        Index('idx_api_key_usage_status', api_key_id, status_code, 'created_at'),
        Index('idx_api_key_usage_ip', api_key_id, ip_address, 'created_at'),
        Index('idx_api_key_usage_performance', api_key_id, response_time_ms.desc()),
    )
    
    @classmethod
    def create_from_request(
        cls,
        api_key: APIKey,
        endpoint: str,
        method: str,
        status_code: int,
        ip_address: str,
        response_time_ms: int,
        **kwargs
    ) -> 'APIKeyUsageLog':
        """Create a usage log entry from request data."""
        return cls(
            api_key_id=api_key.id,
            user_id=api_key.user_id,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            ip_address=ip_address,
            response_time_ms=response_time_ms,
            user_agent=kwargs.get('user_agent'),
            referer=kwargs.get('referer'),
            origin=kwargs.get('origin'),
            request_size_bytes=kwargs.get('request_size_bytes'),
            response_size_bytes=kwargs.get('response_size_bytes'),
            request_headers=kwargs.get('request_headers'),
            request_body=kwargs.get('request_body'),
            response_headers=kwargs.get('response_headers'),
            response_body=kwargs.get('response_body'),
            error_message=kwargs.get('error_message'),
            error_type=kwargs.get('error_type'),
            metadata=kwargs.get('metadata', {}),
        )
    
    def is_successful(self) -> bool:
        """Check if the request was successful."""
        return 200 <= self.status_code < 300
    
    def is_error(self) -> bool:
        """Check if the request resulted in an error."""
        return self.status_code >= 400
    
    def get_performance_category(self) -> str:
        """Get performance category based on response time."""
        if self.response_time_ms < 100:
            return "excellent"
        elif self.response_time_ms < 500:
            return "good"
        elif self.response_time_ms < 1000:
            return "average"
        elif self.response_time_ms < 5000:
            return "poor"
        else:
            return "very_poor"