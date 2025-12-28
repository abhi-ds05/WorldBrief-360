"""
Audit logging model for tracking system activities.
"""

import enum
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from sqlalchemy import (
    Column,
    DateTime,
    Boolean,
    String,
    Text,
    Integer,
    Float,
    ForeignKey,
    Index,
    Enum,
    CheckConstraint,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, INET
from sqlalchemy import Column, Integer, Float, BigInteger
from sqlalchemy.orm import relationship, validates

from db.base import Base, TimestampMixin, generate_uuid


class AuditAction(str, enum.Enum):
    """Audit action types."""
    # User actions
    USER_LOGIN = "user:login"
    USER_LOGOUT = "user:logout"
    USER_REGISTER = "user:register"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    USER_PASSWORD_CHANGE = "user:password_change"
    USER_EMAIL_CHANGE = "user:email_change"
    
    # Article actions
    ARTICLE_CREATE = "article:create"
    ARTICLE_UPDATE = "article:update"
    ARTICLE_DELETE = "article:delete"
    ARTICLE_PUBLISH = "article:publish"
    ARTICLE_VIEW = "article:view"
    
    # Incident actions
    INCIDENT_CREATE = "incident:create"
    INCIDENT_UPDATE = "incident:update"
    INCIDENT_DELETE = "incident:delete"
    INCIDENT_VERIFY = "incident:verify"
    
    # Comment actions
    COMMENT_CREATE = "comment:create"
    COMMENT_UPDATE = "comment:update"
    COMMENT_DELETE = "comment:delete"
    
    # Financial actions
    TRANSACTION_CREATE = "transaction:create"
    TRANSACTION_UPDATE = "transaction:update"
    TRANSACTION_DELETE = "transaction:delete"
    PAYMENT_PROCESS = "payment:process"
    PAYMENT_REFUND = "payment:refund"
    
    # System actions
    CONFIG_UPDATE = "config:update"
    CACHE_CLEAR = "cache:clear"
    BACKUP_CREATE = "backup:create"
    BACKUP_RESTORE = "backup:restore"
    
    # API actions
    API_KEY_CREATE = "api_key:create"
    API_KEY_UPDATE = "api_key:update"
    API_KEY_DELETE = "api_key:delete"
    API_CALL = "api:call"
    
    # Security actions
    SECURITY_ALERT = "security:alert"
    SUSPICIOUS_ACTIVITY = "security:suspicious_activity"
    PERMISSION_CHANGE = "security:permission_change"
    ROLE_CHANGE = "security:role_change"
    
    # Admin actions
    ADMIN_LOGIN = "admin:login"
    ADMIN_ACTION = "admin:action"
    BULK_OPERATION = "admin:bulk_operation"
    
    # Notification actions
    NOTIFICATION_SEND = "notification:send"
    NOTIFICATION_READ = "notification:read"
    NOTIFICATION_DELETE = "notification:delete"


class AuditSeverity(str, enum.Enum):
    """Audit log severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditStatus(str, enum.Enum):
    """Audit action status."""
    SUCCESS = "success"
    FAILURE = "failure"
    PENDING = "pending"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class AuditLog(Base, TimestampMixin):
    """
    Audit log model for tracking all system activities.
    Essential for security, compliance, and debugging.
    """
    
    __tablename__ = "audit_log"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    
    # Foreign keys (optional, as some actions might not have associated entities)
    user_id = Column(UUID(as_uuid=True), ForeignKey('user.id'), nullable=True, index=True)
    api_key_id = Column(UUID(as_uuid=True), ForeignKey('api_key.id'), nullable=True, index=True)
    
    # Entity references (for actions on specific resources)
    entity_type = Column(String(100), nullable=True, index=True)  # e.g., 'user', 'article', 'incident'
    entity_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    
    # Audit metadata
    action = Column(Enum(AuditAction), nullable=False, index=True)
    severity = Column(Enum(AuditSeverity), default=AuditSeverity.INFO, nullable=False, index=True)
    status = Column(Enum(AuditStatus), default=AuditStatus.SUCCESS, nullable=False, index=True)
    
    # Request/operation information
    request_id = Column(String(100), nullable=True, index=True)
    session_id = Column(String(100), nullable=True, index=True)
    correlation_id = Column(String(100), nullable=True, index=True)
    
    # Location and device information
    ip_address = Column(INET, nullable=True, index=True)
    user_agent = Column(Text, nullable=True)
    device_type = Column(String(50), nullable=True)  # mobile, desktop, tablet, bot
    browser = Column(String(100), nullable=True)
    browser_version = Column(String(50), nullable=True)
    operating_system = Column(String(100), nullable=True)
    platform = Column(String(50), nullable=True)
    
    # Geographic information
    country = Column(String(2), nullable=True, index=True)  # ISO country code
    region = Column(String(100), nullable=True)
    city = Column(String(100), nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    
    # Action details
    resource = Column(String(255), nullable=True, index=True)  # e.g., '/api/users', 'ArticleController'
    method = Column(String(10), nullable=True)  # HTTP method: GET, POST, PUT, DELETE, etc.
    endpoint = Column(String(500), nullable=True, index=True)
    
    # Data changes
    old_values = Column(JSONB, nullable=True)  # State before the action
    new_values = Column(JSONB, nullable=True)  # State after the action
    changed_fields = Column(JSONB, nullable=True)  # List of fields that changed
    
    # Performance metrics
    duration_ms = Column(Integer, nullable=True)  # Action duration in milliseconds
    request_size_bytes = Column(Integer, nullable=True)
    response_size_bytes = Column(Integer, nullable=True)
    
    # Error information
    error_code = Column(String(50), nullable=True, index=True)
    error_message = Column(Text, nullable=True)
    error_stack_trace = Column(Text, nullable=True)
    
    # Additional context
    tags = Column(JSONB, default=list, index=True)
    metadata = Column(JSONB, default=dict)
    custom_fields = Column(JSONB, default=dict)
    
    # Retention management
    is_archived = Column(Boolean, default=False, nullable=False, index=True)
    retention_days = Column(Integer, default=365, nullable=False)  # Default 1 year retention
    
    # Relationships
    user = relationship("User", backref="audit_logs")
    api_key = relationship("APIKey", back_populates="audit_logs")
    
    # Indexes (optimized for common query patterns)
    __table_args__ = (
        Index('idx_audit_log_timestamp_action', Column('created_at').desc(), action),
        Index('idx_audit_log_user_timestamp', user_id, Column('created_at').desc()),
        Index('idx_audit_log_entity', entity_type, entity_id, Column('created_at').desc()),
        Index('idx_audit_log_severity_status', severity, status, Column('created_at').desc()),
        Index('idx_audit_log_ip_timestamp', ip_address, Column('created_at').desc()),
        Index('idx_audit_log_country_timestamp', country, Column('created_at').desc()),
        Index('idx_audit_log_resource_method', resource, method, Column('created_at').desc()),
        Index('idx_audit_log_request_id', request_id, Column('created_at').desc()),
        Index('idx_audit_log_tags', tags, postgresql_using='gin'),
        Index('idx_audit_log_metadata', metadata, postgresql_using='gin'),
        CheckConstraint('duration_ms >= 0', name='check_duration_non_negative'),
        CheckConstraint('retention_days >= 0', name='check_retention_non_negative'),
    )
    
    @classmethod
    def create_log(
        cls,
        action: AuditAction,
        user_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        **kwargs
    ) -> 'AuditLog':
        """
        Create an audit log entry with standardized fields.
        
        Args:
            action: The audit action
            user_id: ID of the user performing the action
            entity_type: Type of entity being acted upon
            entity_id: ID of entity being acted upon
            **kwargs: Additional fields for the log
            
        Returns:
            AuditLog instance
        """
        from .user import User
        
        log = cls(
            action=action,
            user_id=user_id,
            entity_type=entity_type,
            entity_id=entity_id,
            request_id=kwargs.get('request_id'),
            session_id=kwargs.get('session_id'),
            correlation_id=kwargs.get('correlation_id'),
            ip_address=kwargs.get('ip_address'),
            user_agent=kwargs.get('user_agent'),
            device_type=kwargs.get('device_type'),
            browser=kwargs.get('browser'),
            browser_version=kwargs.get('browser_version'),
            operating_system=kwargs.get('operating_system'),
            platform=kwargs.get('platform'),
            country=kwargs.get('country'),
            region=kwargs.get('region'),
            city=kwargs.get('city'),
            latitude=kwargs.get('latitude'),
            longitude=kwargs.get('longitude'),
            resource=kwargs.get('resource'),
            method=kwargs.get('method'),
            endpoint=kwargs.get('endpoint'),
            old_values=kwargs.get('old_values'),
            new_values=kwargs.get('new_values'),
            changed_fields=kwargs.get('changed_fields'),
            duration_ms=kwargs.get('duration_ms'),
            request_size_bytes=kwargs.get('request_size_bytes'),
            response_size_bytes=kwargs.get('response_size_bytes'),
            error_code=kwargs.get('error_code'),
            error_message=kwargs.get('error_message'),
            error_stack_trace=kwargs.get('error_stack_trace'),
            tags=kwargs.get('tags', []),
            metadata=kwargs.get('metadata', {}),
            custom_fields=kwargs.get('custom_fields', {}),
            severity=kwargs.get('severity', AuditSeverity.INFO),
            status=kwargs.get('status', AuditStatus.SUCCESS),
            api_key_id=kwargs.get('api_key_id'),
        )
        
        # Extract device information from user agent
        if not log.device_type and log.user_agent:
            log.device_type = cls._detect_device_type(log.user_agent)
        
        # Extract browser information
        if not log.browser and log.user_agent:
            log.browser, log.browser_version = cls._extract_browser_info(log.user_agent)
        
        # Extract operating system
        if not log.operating_system and log.user_agent:
            log.operating_system = cls._extract_os_info(log.user_agent)
        
        return log
    
    @staticmethod
    def _detect_device_type(user_agent: str) -> str:
        """Detect device type from user agent."""
        user_agent = user_agent.lower()
        
        if 'mobile' in user_agent or 'android' in user_agent or 'iphone' in user_agent:
            return 'mobile'
        elif 'tablet' in user_agent or 'ipad' in user_agent:
            return 'tablet'
        elif 'bot' in user_agent or 'crawler' in user_agent or 'spider' in user_agent:
            return 'bot'
        else:
            return 'desktop'
    
    @staticmethod
    def _extract_browser_info(user_agent: str):
        """Extract browser name and version from user agent."""
        import re
        
        # Common browser patterns
        patterns = [
            (r'chrome/(\d+\.\d+)', 'Chrome'),
            (r'firefox/(\d+\.\d+)', 'Firefox'),
            (r'safari/(\d+\.\d+)', 'Safari'),
            (r'edge/(\d+\.\d+)', 'Edge'),
            (r'opera/(\d+\.\d+)', 'Opera'),
            (r'msie (\d+\.\d+)', 'Internet Explorer'),
        ]
        
        for pattern, browser_name in patterns:
            match = re.search(pattern, user_agent.lower())
            if match:
                return browser_name, match.group(1)
        
        return 'Unknown', None
    
    @staticmethod
    def _extract_os_info(user_agent: str) -> str:
        """Extract operating system from user agent."""
        user_agent = user_agent.lower()
        
        if 'windows' in user_agent:
            return 'Windows'
        elif 'mac os' in user_agent or 'macos' in user_agent:
            return 'macOS'
        elif 'linux' in user_agent:
            return 'Linux'
        elif 'android' in user_agent:
            return 'Android'
        elif 'ios' in user_agent or 'iphone' in user_agent:
            return 'iOS'
        else:
            return 'Unknown'
    
    def is_successful(self) -> bool:
        """Check if the action was successful."""
        return self.status == AuditStatus.SUCCESS
    
    def is_failed(self) -> bool:
        """Check if the action failed."""
        return self.status == AuditStatus.FAILURE
    
    def get_action_category(self) -> str:
        """Get action category from action type."""
        if ':' in self.action.value:
            return self.action.value.split(':')[0]
        return 'unknown'
    
    def get_action_description(self) -> str:
        """Get human-readable action description."""
        # Map action types to descriptions
        descriptions = {
            AuditAction.USER_LOGIN: "User logged in",
            AuditAction.USER_LOGOUT: "User logged out",
            AuditAction.USER_REGISTER: "User registered",
            AuditAction.USER_UPDATE: "User profile updated",
            AuditAction.ARTICLE_CREATE: "Article created",
            AuditAction.ARTICLE_UPDATE: "Article updated",
            AuditAction.ARTICLE_DELETE: "Article deleted",
            AuditAction.ARTICLE_PUBLISH: "Article published",
            AuditAction.INCIDENT_CREATE: "Incident reported",
            AuditAction.INCIDENT_UPDATE: "Incident updated",
            AuditAction.INCIDENT_VERIFY: "Incident verified",
            AuditAction.COMMENT_CREATE: "Comment created",
            AuditAction.COMMENT_UPDATE: "Comment updated",
            AuditAction.COMMENT_DELETE: "Comment deleted",
            AuditAction.TRANSACTION_CREATE: "Transaction created",
            AuditAction.TRANSACTION_UPDATE: "Transaction updated",
            AuditAction.PAYMENT_PROCESS: "Payment processed",
            AuditAction.API_KEY_CREATE: "API key created",
            AuditAction.API_KEY_UPDATE: "API key updated",
            AuditAction.API_KEY_DELETE: "API key deleted",
            AuditAction.API_CALL: "API call made",
            AuditAction.SECURITY_ALERT: "Security alert triggered",
            AuditAction.ADMIN_LOGIN: "Admin logged in",
            AuditAction.NOTIFICATION_SEND: "Notification sent",
        }
        
        return descriptions.get(self.action, self.action.value.replace(':', ' ').title())
    
    def get_changes_summary(self) -> Dict[str, Any]:
        """Get summary of changes made."""
        if not self.old_values and not self.new_values:
            return {}
        
        summary = {
            'fields_changed': self.changed_fields or [],
            'has_changes': bool(self.old_values or self.new_values),
        }
        
        # Add specific change information for common fields
        if self.old_values and self.new_values:
            for field in summary['fields_changed']:
                if field in self.old_values and field in self.new_values:
                    summary[f'{field}_from'] = self.old_values[field]
                    summary[f'{field}_to'] = self.new_values[field]
        
        return summary
    
    def should_alert(self) -> bool:
        """Determine if this log entry should trigger an alert."""
        # Alert on critical errors
        if self.severity in [AuditSeverity.ERROR, AuditSeverity.CRITICAL]:
            return True
        
        # Alert on security-related actions
        if self.action in [
            AuditAction.SECURITY_ALERT,
            AuditAction.SUSPICIOUS_ACTIVITY,
            AuditAction.ADMIN_LOGIN,
        ]:
            return True
        
        # Alert on failed admin actions
        if self.get_action_category() == 'admin' and self.is_failed():
            return True
        
        return False
    
    def get_retention_expiry_date(self) -> datetime:
        """Calculate when this log entry should be archived/removed."""
        return self.created_at + timedelta(days=self.retention_days)
    
    def is_expired(self) -> bool:
        """Check if the log entry has expired based on retention policy."""
        return datetime.utcnow() > self.get_retention_expiry_date()
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Args:
            include_sensitive: Whether to include sensitive data (IP, user agent, etc.)
            
        Returns:
            Dictionary representation
        """
        data = super().to_dict()
        
        if not include_sensitive:
            # Remove sensitive information
            sensitive_fields = [
                'ip_address', 'user_agent', 'device_type', 'browser',
                'browser_version', 'operating_system', 'platform',
                'country', 'region', 'city', 'latitude', 'longitude',
                'old_values', 'new_values', 'error_stack_trace'
            ]
            
            for field in sensitive_fields:
                if field in data:
                    del data[field]
        
        # Add computed fields
        data['action_category'] = self.get_action_category()
        data['action_description'] = self.get_action_description()
        data['is_successful'] = self.is_successful()
        data['is_failed'] = self.is_failed()
        data['should_alert'] = self.should_alert()
        data['changes_summary'] = self.get_changes_summary()
        data['retention_expiry'] = self.get_retention_expiry_date().isoformat()
        data['is_expired'] = self.is_expired()
        
        # Add user information if available
        if self.user:
            data['user'] = {
                'id': str(self.user.id),
                'username': self.user.username,
                'email': self.user.email if include_sensitive else None,
                'full_name': self.user.full_name,
            }
        
        # Add API key information if available
        if self.api_key:
            data['api_key'] = {
                'id': str(self.api_key.id),
                'name': self.api_key.name,
                'type': self.api_key.key_type.value,
            }
        
        return data


# Audit log aggregation for reporting
class AuditLogSummary(Base):
    """
    Materialized view or summary table for audit log reporting.
    This would typically be populated by a scheduled job.
    """
    
    __tablename__ = "audit_log_summary"
    
    # Primary key (date + action)
    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    
    # Aggregation dimensions
    date = Column(DateTime, nullable=False, index=True)  # Date only (time truncated to day)
    action = Column(Enum(AuditAction), nullable=False, index=True)
    severity = Column(Enum(AuditSeverity), nullable=False, index=True)
    status = Column(Enum(AuditStatus), nullable=False, index=True)
    entity_type = Column(String(100), nullable=True, index=True)
    country = Column(String(2), nullable=True, index=True)
    
    # Aggregation metrics
    total_count = Column(Integer, default=0, nullable=False)
    success_count = Column(Integer, default=0, nullable=False)
    failure_count = Column(Integer, default=0, nullable=False)
    total_duration_ms = Column(BigInteger, default=0, nullable=False)
    avg_duration_ms = Column(Float, default=0.0, nullable=False)
    
    # User statistics
    unique_users = Column(Integer, default=0, nullable=False)
    unique_ips = Column(Integer, default=0, nullable=False)
    
    # Error statistics
    error_count = Column(Integer, default=0, nullable=False)
    unique_error_codes = Column(Integer, default=0, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_audit_summary_date_action', date.desc(), action),
        Index('idx_audit_summary_entity_date', entity_type, date.desc()),
        Index('idx_audit_summary_country_date', country, date.desc()),
        UniqueConstraint('date', 'action', 'severity', 'status', 'entity_type', 'country',
                        name='uq_audit_summary_dimensions'),
        CheckConstraint('total_count >= 0', name='check_total_count_non_negative'),
        CheckConstraint('success_count >= 0', name='check_success_count_non_negative'),
        CheckConstraint('failure_count >= 0', name='check_failure_count_non_negative'),
        CheckConstraint('total_duration_ms >= 0', name='check_total_duration_non_negative'),
        CheckConstraint('avg_duration_ms >= 0', name='check_avg_duration_non_negative'),
    )