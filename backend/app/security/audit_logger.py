"""
Audit Logger for Security and Compliance Tracking

This module provides comprehensive audit logging for:
- User authentication and authorization events
- Security incidents and anomalies
- System changes and administrative actions
- Data access and modifications
- API usage and access patterns
- Compliance requirements (GDPR, HIPAA, etc.)

Features:
- Structured logging with JSON formatting
- Multiple output destinations (console, file, database, external services)
- Correlation IDs for request tracing
- Sensitive data masking
- Configurable retention policies
- Real-time alerting for security events
- Batch processing for high-volume events
"""

import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, asdict, field
from contextlib import contextmanager
from threading import Lock
from collections import deque

from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models.audit_log import AuditLog as AuditLogModel
from app.db.session import get_db_session

# Get settings
settings = get_settings()

# Configure audit logging
AUDIT_LOG_LEVEL = logging.INFO
AUDIT_RETENTION_DAYS = settings.AUDIT_LOG_RETENTION_DAYS or 90
BATCH_SIZE = settings.AUDIT_BATCH_SIZE or 100
FLUSH_INTERVAL = settings.AUDIT_FLUSH_INTERVAL or 60  # seconds


class AuditEventType(str, Enum):
    """Types of audit events."""
    
    # Authentication events
    LOGIN_SUCCESS = "LOGIN_SUCCESS"
    LOGIN_FAILED = "LOGIN_FAILED"
    LOGOUT = "LOGOUT"
    PASSWORD_CHANGE = "PASSWORD_CHANGE"
    PASSWORD_RESET = "PASSWORD_RESET"
    MFA_ENABLED = "MFA_ENABLED"
    MFA_DISABLED = "MFA_DISABLED"
    MFA_FAILED = "MFA_FAILED"
    API_KEY_CREATED = "API_KEY_CREATED"
    API_KEY_REVOKED = "API_KEY_REVOKED"
    SESSION_CREATED = "SESSION_CREATED"
    SESSION_EXPIRED = "SESSION_EXPIRED"
    SESSION_REVOKED = "SESSION_REVOKED"
    
    # User events
    USER_CREATED = "USER_CREATED"
    USER_UPDATED = "USER_UPDATED"
    USER_DELETED = "USER_DELETED"
    USER_ROLE_CHANGED = "USER_ROLE_CHANGED"
    USER_PERMISSION_CHANGED = "USER_PERMISSION_CHANGED"
    PROFILE_UPDATED = "PROFILE_UPDATED"
    
    # Security events
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    SUSPICIOUS_ACTIVITY = "SUSPICIOUS_ACTIVITY"
    BRUTE_FORCE_ATTEMPT = "BRUTE_FORCE_ATTEMPT"
    SQL_INJECTION_ATTEMPT = "SQL_INJECTION_ATTEMPT"
    XSS_ATTEMPT = "XSS_ATTEMPT"
    FILE_UPLOAD_BLOCKED = "FILE_UPLOAD_BLOCKED"
    MALICIOUS_CONTENT = "MALICIOUS_CONTENT"
    ACCESS_DENIED = "ACCESS_DENIED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    
    # Data events
    DATA_CREATED = "DATA_CREATED"
    DATA_UPDATED = "DATA_UPDATED"
    DATA_DELETED = "DATA_DELETED"
    DATA_EXPORTED = "DATA_EXPORTED"
    DATA_IMPORTED = "DATA_IMPORTED"
    DATA_ACCESSED = "DATA_ACCESSED"
    DATA_SHARED = "DATA_SHARED"
    
    # Incident events
    INCIDENT_REPORTED = "INCIDENT_REPORTED"
    INCIDENT_VERIFIED = "INCIDENT_VERIFIED"
    INCIDENT_DISMISSED = "INCIDENT_DISMISSED"
    INCIDENT_RESOLVED = "INCIDENT_RESOLVED"
    INCIDENT_COMMENT_ADDED = "INCIDENT_COMMENT_ADDED"
    INCIDENT_VOTED = "INCIDENT_VOTED"
    
    # Briefing events
    BRIEFING_GENERATED = "BRIEFING_GENERATED"
    BRIEFING_VIEWED = "BRIEFING_VIEWED"
    BRIEFING_SHARED = "BRIEFING_SHARED"
    BRIEFING_DOWNLOADED = "BRIEFING_DOWNLOADED"
    
    # Chat events
    CHAT_STARTED = "CHAT_STARTED"
    CHAT_MESSAGE_SENT = "CHAT_MESSAGE_SENT"
    CHAT_MESSAGE_RECEIVED = "CHAT_MESSAGE_RECEIVED"
    
    # Wallet events
    WALLET_CREATED = "WALLET_CREATED"
    COINS_EARNED = "COINS_EARNED"
    COINS_SPENT = "COINS_SPENT"
    TRANSACTION_COMPLETED = "TRANSACTION_COMPLETED"
    TRANSACTION_FAILED = "TRANSACTION_FAILED"
    
    # System events
    SYSTEM_STARTUP = "SYSTEM_STARTUP"
    SYSTEM_SHUTDOWN = "SYSTEM_SHUTDOWN"
    CONFIG_CHANGED = "CONFIG_CHANGED"
    FEATURE_FLAG_CHANGED = "FEATURE_FLAG_CHANGED"
    BACKUP_CREATED = "BACKUP_CREATED"
    BACKUP_RESTORED = "BACKUP_RESTORED"
    MAINTENANCE_STARTED = "MAINTENANCE_STARTED"
    MAINTENANCE_COMPLETED = "MAINTENANCE_COMPLETED"
    
    # API events
    API_CALL = "API_CALL"
    API_ERROR = "API_ERROR"
    API_RATE_LIMITED = "API_RATE_LIMITED"
    WEBHOOK_RECEIVED = "WEBHOOK_RECEIVED"
    WEBHOOK_SENT = "WEBHOOK_SENT"
    
    # Admin events
    ADMIN_ACTION = "ADMIN_ACTION"
    USER_BANNED = "USER_BANNED"
    USER_UNBANNED = "USER_UNBANNED"
    CONTENT_MODERATED = "CONTENT_MODERATED"
    SYSTEM_ALERT = "SYSTEM_ALERT"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AuditSource(str, Enum):
    """Source of audit events."""
    API = "API"
    WEB = "WEB"
    MOBILE = "MOBILE"
    CLI = "CLI"
    SCHEDULED_TASK = "SCHEDULED_TASK"
    SYSTEM = "SYSTEM"
    EXTERNAL = "EXTERNAL"


@dataclass
class AuditContext:
    """Context information for audit events."""
    correlation_id: str = ""
    request_id: str = ""
    session_id: str = ""
    user_agent: str = ""
    ip_address: str = ""
    location: Optional[Dict[str, Any]] = None
    device_info: Optional[Dict[str, Any]] = None
    request_path: str = ""
    request_method: str = ""
    query_params: Optional[Dict[str, Any]] = None


@dataclass
class AuditEvent:
    """Audit event data structure."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: AuditEventType = AuditEventType.API_CALL
    severity: AuditSeverity = AuditSeverity.INFO
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: AuditSource = AuditSource.API
    source_component: str = ""
    
    # User information
    user_id: Optional[str] = None
    username: Optional[str] = None
    user_role: Optional[str] = None
    user_email: Optional[str] = None
    
    # Event details
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Context
    context: AuditContext = field(default_factory=AuditContext)
    
    # Status
    success: bool = True
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    
    # Performance
    duration_ms: Optional[float] = None
    resource_usage: Optional[Dict[str, Any]] = None


class AuditLogSchema(BaseModel):
    """Pydantic schema for audit log validation."""
    event_id: str = Field(..., description="Unique event identifier")
    event_type: AuditEventType = Field(..., description="Type of audit event")
    severity: AuditSeverity = Field(..., description="Severity level")
    timestamp: datetime = Field(..., description="Event timestamp")
    source: AuditSource = Field(..., description="Event source")
    source_component: str = Field(..., description="Source component/namespace")
    
    user_id: Optional[str] = Field(None, description="User ID if applicable")
    username: Optional[str] = Field(None, description="Username if applicable")
    user_role: Optional[str] = Field(None, description="User role if applicable")
    
    description: str = Field(..., description="Event description")
    details: Dict[str, Any] = Field(default_factory=dict, description="Event details")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    success: bool = Field(True, description="Whether the operation was successful")
    error_message: Optional[str] = Field(None, description="Error message if any")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class AuditLogger:
    """
    Main audit logging class with multiple output destinations and batching.
    """
    
    def __init__(
        self,
        name: str = "worldbrief360_audit",
        enable_console: bool = True,
        enable_file: bool = False,
        enable_database: bool = True,
        enable_external: bool = False,
        batch_size: int = BATCH_SIZE,
        flush_interval: int = FLUSH_INTERVAL,
        retention_days: int = AUDIT_RETENTION_DAYS,
        mask_sensitive_data: bool = True
    ):
        """
        Initialize the audit logger.
        
        Args:
            name: Logger name
            enable_console: Enable console logging
            enable_file: Enable file logging
            enable_database: Enable database logging
            enable_external: Enable external service logging
            batch_size: Batch size for database operations
            flush_interval: Flush interval in seconds
            retention_days: Log retention period in days
            mask_sensitive_data: Mask sensitive data in logs
        """
        self.name = name
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.retention_days = retention_days
        self.mask_sensitive_data = mask_sensitive_data
        
        # Initialize buffers
        self._buffer: List[AuditEvent] = []
        self._buffer_lock = Lock()
        self._last_flush = time.time()
        
        # Setup logging destinations
        self.destinations = []
        
        # Console logger
        if enable_console:
            console_logger = self._setup_console_logger()
            self.destinations.append(console_logger)
        
        # File logger
        if enable_file:
            file_logger = self._setup_file_logger()
            self.destinations.append(file_logger)
        
        # Database logger
        if enable_database:
            self.destinations.append(self._log_to_database)
        
        # External service logger (e.g., SIEM, Splunk, ELK)
        if enable_external:
            external_logger = self._setup_external_logger()
            if external_logger:
                self.destinations.append(external_logger)
        
        # Initialize cleanup scheduler
        self._setup_cleanup_scheduler()
        
        # Initialize alerting
        self._setup_alerting()
        
        print(f"Audit Logger initialized with {len(self.destinations)} destinations")
    
    def _setup_console_logger(self) -> Callable:
        """Setup structured JSON console logging."""
        logger = logging.getLogger(f"{self.name}.console")
        logger.setLevel(AUDIT_LOG_LEVEL)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Add JSON formatter handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "event": %(message)s}',
            datefmt="%Y-%m-%dT%H:%M:%SZ"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return lambda event: logger.log(
            self._get_log_level(event.severity),
            json.dumps(self._event_to_dict(event), default=str)
        )
    
    def _setup_file_logger(self) -> Callable:
        """Setup file-based logging with rotation."""
        logger = logging.getLogger(f"{self.name}.file")
        logger.setLevel(AUDIT_LOG_LEVEL)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Create logs directory if it doesn't exist
        log_dir = settings.LOG_DIR / "audit"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Add rotating file handler
        from logging.handlers import RotatingFileHandler
        
        log_file = log_dir / "audit.log"
        handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=10
        )
        
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "event": %(message)s}',
            datefmt="%Y-%m-%dT%H:%M:%SZ"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return lambda event: logger.log(
            self._get_log_level(event.severity),
            json.dumps(self._event_to_dict(event), default=str)
        )
    
    def _setup_external_logger(self) -> Optional[Callable]:
        """Setup external logging service (e.g., Splunk, ELK, SIEM)."""
        # This is a placeholder for external service integration
        # Implement based on your specific external service
        
        external_type = settings.AUDIT_EXTERNAL_TYPE
        if not external_type:
            return None
        
        if external_type == "splunk":
            return self._log_to_splunk
        elif external_type == "elasticsearch":
            return self._log_to_elasticsearch
        elif external_type == "s3":
            return self._log_to_s3
        elif external_type == "custom":
            return self._log_to_custom_external
        
        return None
    
    def _log_to_database(self, event: AuditEvent):
        """Log event to database with batching."""
        with self._buffer_lock:
            self._buffer.append(event)
            
            # Check if we should flush
            should_flush = (
                len(self._buffer) >= self.batch_size or
                time.time() - self._last_flush >= self.flush_interval
            )
            
            if should_flush:
                self._flush_buffer()
    
    def _flush_buffer(self):
        """Flush buffer to database."""
        if not self._buffer:
            return
        
        buffer_copy = self._buffer.copy()
        self._buffer.clear()
        self._last_flush = time.time()
        
        # Use thread pool or async for database operations in production
        try:
            self._save_to_database(buffer_copy)
        except Exception as e:
            # Log error and retry or store in dead letter queue
            print(f"Error saving audit logs to database: {e}")
            # In production, implement retry logic or store in Redis/queue
    
    def _save_to_database(self, events: List[AuditEvent]):
        """Save events to database."""
        if not events:
            return
        
        try:
            db_session = next(get_db_session())
            audit_logs = []
            
            for event in events:
                # Convert to database model
                audit_log = AuditLogModel(
                    event_id=event.event_id,
                    event_type=event.event_type.value,
                    severity=event.severity.value,
                    timestamp=event.timestamp,
                    source=event.source.value,
                    source_component=event.source_component,
                    user_id=event.user_id,
                    username=event.username,
                    user_role=event.user_role,
                    description=event.description,
                    details=json.dumps(event.details),
                    metadata=json.dumps(event.metadata),
                    context=json.dumps(asdict(event.context)),
                    success=event.success,
                    error_message=event.error_message,
                    error_code=event.error_code,
                    duration_ms=event.duration_ms,
                    ip_address=event.context.ip_address,
                    user_agent=event.context.user_agent,
                    request_path=event.context.request_path,
                    request_method=event.context.request_method,
                )
                audit_logs.append(audit_log)
            
            db_session.bulk_save_objects(audit_logs)
            db_session.commit()
            
        except Exception as e:
            print(f"Database error in audit logging: {e}")
            raise
        finally:
            db_session.close()
    
    def _log_to_splunk(self, event: AuditEvent):
        """Log to Splunk HTTP Event Collector."""
        # Implement Splunk integration
        pass
    
    def _log_to_elasticsearch(self, event: AuditEvent):
        """Log to Elasticsearch."""
        # Implement Elasticsearch integration
        pass
    
    def _log_to_s3(self, event: AuditEvent):
        """Log to AWS S3 for long-term storage."""
        # Implement S3 integration
        pass
    
    def _log_to_custom_external(self, event: AuditEvent):
        """Log to custom external service."""
        # Implement custom external service integration
        pass
    
    def _setup_cleanup_scheduler(self):
        """Setup scheduled cleanup of old audit logs."""
        import threading
        
        def cleanup_old_logs():
            while True:
                time.sleep(24 * 3600)  # Run daily
                self.cleanup_old_entries()
        
        # Start cleanup thread
        cleanup_thread = threading.Thread(target=cleanup_old_logs, daemon=True)
        cleanup_thread.start()
    
    def _setup_alerting(self):
        """Setup real-time alerting for security events."""
        self.alerts_enabled = settings.AUDIT_ALERTS_ENABLED
        self.alert_rules = self._load_alert_rules()
    
    def _load_alert_rules(self) -> List[Dict[str, Any]]:
        """Load alert rules from configuration."""
        return [
            {
                "event_type": AuditEventType.LOGIN_FAILED,
                "threshold": 5,
                "time_window": 300,  # 5 minutes
                "severity": AuditSeverity.WARNING,
                "action": "alert_admin"
            },
            {
                "event_type": AuditEventType.SUSPICIOUS_ACTIVITY,
                "threshold": 1,
                "time_window": 60,
                "severity": AuditSeverity.CRITICAL,
                "action": "block_ip"
            },
            # Add more rules as needed
        ]
    
    def log_security_event(
        self,
        event_type: Union[AuditEventType, str],
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        description: str = "",
        details: Optional[Dict[str, Any]] = None,
        severity: AuditSeverity = AuditSeverity.INFO,
        success: bool = True,
        error_message: Optional[str] = None,
        context: Optional[AuditContext] = None,
        **kwargs
    ):
        """
        Log a security event.
        
        Args:
            event_type: Type of audit event
            user_id: User ID if applicable
            username: Username if applicable
            description: Event description
            details: Event details
            severity: Severity level
            success: Whether operation was successful
            error_message: Error message if any
            context: Audit context
            **kwargs: Additional event properties
        """
        # Convert string event type to enum
        if isinstance(event_type, str):
            try:
                event_type = AuditEventType(event_type)
            except ValueError:
                event_type = AuditEventType.API_CALL
        
        # Create audit event
        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            source=AuditSource.API,
            source_component=self.name,
            user_id=user_id,
            username=username,
            description=description,
            details=details or {},
            success=success,
            error_message=error_message,
            context=context or AuditContext(),
            **kwargs
        )
        
        # Mask sensitive data if enabled
        if self.mask_sensitive_data:
            event = self._mask_sensitive_data(event)
        
        # Log to all destinations
        for destination in self.destinations:
            try:
                destination(event)
            except Exception as e:
                print(f"Error logging to destination: {e}")
        
        # Check for alerts
        if self.alerts_enabled:
            self._check_alerts(event)
        
        # Return event ID for correlation
        return event.event_id
    
    def log_user_activity(
        self,
        user_id: str,
        username: str,
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        context: Optional[AuditContext] = None
    ):
        """
        Log user activity.
        
        Args:
            user_id: User ID
            username: Username
            action: Action performed
            resource_type: Type of resource
            resource_id: Resource ID
            details: Additional details
            context: Audit context
        """
        description = f"User {username} performed {action} on {resource_type}"
        if resource_id:
            description += f" ({resource_id})"
        
        self.log_security_event(
            event_type=AuditEventType.USER_UPDATED,
            user_id=user_id,
            username=username,
            description=description,
            details=details or {},
            severity=AuditSeverity.INFO,
            context=context
        )
    
    def log_data_access(
        self,
        user_id: str,
        username: str,
        data_type: str,
        data_id: str,
        access_type: str = "read",
        context: Optional[AuditContext] = None
    ):
        """
        Log data access events.
        
        Args:
            user_id: User ID
            username: Username
            data_type: Type of data accessed
            data_id: Data identifier
            access_type: Type of access (read, write, delete)
            context: Audit context
        """
        self.log_security_event(
            event_type=AuditEventType.DATA_ACCESSED,
            user_id=user_id,
            username=username,
            description=f"User {username} {access_type} access to {data_type}: {data_id}",
            details={
                "data_type": data_type,
                "data_id": data_id,
                "access_type": access_type
            },
            severity=AuditSeverity.INFO,
            context=context
        )
    
    def log_api_call(
        self,
        endpoint: str,
        method: str,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        status_code: int = 200,
        duration_ms: Optional[float] = None,
        context: Optional[AuditContext] = None
    ):
        """
        Log API calls.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            user_id: User ID if authenticated
            username: Username if authenticated
            status_code: HTTP status code
            duration_ms: Request duration in milliseconds
            context: Audit context
        """
        success = 200 <= status_code < 400
        
        self.log_security_event(
            event_type=AuditEventType.API_CALL,
            user_id=user_id,
            username=username,
            description=f"API Call: {method} {endpoint} - {status_code}",
            details={
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
                "duration_ms": duration_ms
            },
            severity=AuditSeverity.DEBUG if success else AuditSeverity.WARNING,
            success=success,
            duration_ms=duration_ms,
            context=context
        )
    
    def _mask_sensitive_data(self, event: AuditEvent) -> AuditEvent:
        """Mask sensitive data in audit events."""
        sensitive_fields = [
            "password", "token", "secret", "key", "credit_card",
            "ssn", "phone", "email", "address"
        ]
        
        def mask_dict(data: Dict[str, Any]) -> Dict[str, Any]:
            masked = {}
            for key, value in data.items():
                if any(sensitive in key.lower() for sensitive in sensitive_fields):
                    masked[key] = "***MASKED***"
                elif isinstance(value, dict):
                    masked[key] = mask_dict(value)
                elif isinstance(value, list):
                    masked[key] = [
                        mask_dict(item) if isinstance(item, dict) else
                        "***MASKED***" if any(sensitive in str(item).lower() 
                        for sensitive in sensitive_fields) else item
                        for item in value
                    ]
                else:
                    masked[key] = value
            return masked
        
        # Mask details and metadata
        event.details = mask_dict(event.details)
        event.metadata = mask_dict(event.metadata)
        
        return event
    
    def _event_to_dict(self, event: AuditEvent) -> Dict[str, Any]:
        """Convert audit event to dictionary."""
        event_dict = asdict(event)
        
        # Convert enums to strings
        event_dict["event_type"] = event.event_type.value
        event_dict["severity"] = event.severity.value
        event_dict["source"] = event.source.value
        
        # Convert datetime to ISO format
        if isinstance(event_dict["timestamp"], datetime):
            event_dict["timestamp"] = event_dict["timestamp"].isoformat()
        
        return event_dict
    
    def _get_log_level(self, severity: AuditSeverity) -> int:
        """Convert audit severity to logging level."""
        severity_map = {
            AuditSeverity.DEBUG: logging.DEBUG,
            AuditSeverity.INFO: logging.INFO,
            AuditSeverity.WARNING: logging.WARNING,
            AuditSeverity.ERROR: logging.ERROR,
            AuditSeverity.CRITICAL: logging.CRITICAL,
        }
        return severity_map.get(severity, logging.INFO)
    
    def _check_alerts(self, event: AuditEvent):
        """Check if event triggers any alerts."""
        for rule in self.alert_rules:
            if event.event_type == rule["event_type"]:
                # Check threshold and time window
                # Implement alerting logic here
                pass
    
    def cleanup_old_entries(self):
        """Cleanup old audit entries based on retention policy."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
            
            db_session = next(get_db_session())
            deleted_count = db_session.query(AuditLogModel)\
                .filter(AuditLogModel.timestamp < cutoff_date)\
                .delete()
            db_session.commit()
            
            print(f"Cleaned up {deleted_count} old audit entries")
            
        except Exception as e:
            print(f"Error cleaning up audit logs: {e}")
        finally:
            if 'db_session' in locals():
                db_session.close()
    
    def get_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        severity: Optional[AuditSeverity] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve audit events with filtering.
        
        Args:
            start_date: Start date filter
            end_date: End date filter
            event_type: Event type filter
            user_id: User ID filter
            severity: Severity filter
            limit: Maximum number of results
            offset: Result offset
            
        Returns:
            List of audit events
        """
        try:
            db_session = next(get_db_session())
            query = db_session.query(AuditLogModel)
            
            # Apply filters
            if start_date:
                query = query.filter(AuditLogModel.timestamp >= start_date)
            if end_date:
                query = query.filter(AuditLogModel.timestamp <= end_date)
            if event_type:
                query = query.filter(AuditLogModel.event_type == event_type.value)
            if user_id:
                query = query.filter(AuditLogModel.user_id == user_id)
            if severity:
                query = query.filter(AuditLogModel.severity == severity.value)
            
            # Order and paginate
            query = query.order_by(AuditLogModel.timestamp.desc())
            query = query.offset(offset).limit(limit)
            
            events = query.all()
            
            # Convert to dictionaries
            result = []
            for event in events:
                event_dict = {
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "severity": event.severity,
                    "timestamp": event.timestamp.isoformat() if event.timestamp else None,
                    "user_id": event.user_id,
                    "username": event.username,
                    "description": event.description,
                    "details": json.loads(event.details) if event.details else {},
                    "success": event.success,
                    "ip_address": event.ip_address,
                    "user_agent": event.user_agent,
                    "request_path": event.request_path,
                    "request_method": event.request_method,
                }
                result.append(event_dict)
            
            return result
            
        except Exception as e:
            print(f"Error retrieving audit events: {e}")
            return []
        finally:
            if 'db_session' in locals():
                db_session.close()
    
    def flush(self):
        """Force flush all buffered events."""
        with self._buffer_lock:
            self._flush_buffer()
    
    def __del__(self):
        """Destructor to ensure buffered events are flushed."""
        self.flush()


# FastAPI Integration
class AuditLogDependency:
    """FastAPI dependency for audit logging."""
    
    def __init__(self, logger: Optional[AuditLogger] = None):
        self.logger = logger or audit_logger
    
    async def __call__(
        self,
        request,
        current_user: Optional[Dict[str, Any]] = None,
        event_type: AuditEventType = AuditEventType.API_CALL,
        description: str = "",
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Create audit context and log event.
        
        Args:
            request: FastAPI request
            current_user: Current user information
            event_type: Event type
            description: Event description
            details: Event details
            
        Returns:
            Audit context
        """
        # Create audit context from request
        context = AuditContext(
            correlation_id=request.headers.get("X-Correlation-ID", ""),
            request_id=request.headers.get("X-Request-ID", str(uuid.uuid4())),
            user_agent=request.headers.get("User-Agent", ""),
            ip_address=request.client.host if request.client else "",
            request_path=request.url.path,
            request_method=request.method,
            query_params=dict(request.query_params)
        )
        
        # Log the event
        if self.logger:
            self.logger.log_security_event(
                event_type=event_type,
                user_id=current_user.get("id") if current_user else None,
                username=current_user.get("username") if current_user else None,
                user_role=current_user.get("role") if current_user else None,
                description=description,
                details=details or {},
                context=context
            )
        
        return context


# Decorator for auditing function calls
def audit_log(
    event_type: AuditEventType = AuditEventType.API_CALL,
    description: str = "",
    capture_args: bool = True,
    capture_result: bool = False,
    capture_errors: bool = True
):
    """
    Decorator to audit function calls.
    
    Args:
        event_type: Event type
        description: Event description
        capture_args: Capture function arguments
        capture_result: Capture function result
        capture_errors: Capture errors
    """
    def decorator(func):
        from functools import wraps
        import inspect
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            error_message = None
            result = None
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                
                # Extract user information from args/kwargs if available
                user_id = None
                username = None
                
                # Try to find user in arguments
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                
                for i, arg in enumerate(args):
                    if i < len(params):
                        param_name = params[i]
                        if 'user' in param_name.lower() and isinstance(arg, dict):
                            user_id = arg.get('id')
                            username = arg.get('username')
                
                # Check kwargs
                for key, value in kwargs.items():
                    if 'user' in key.lower() and isinstance(value, dict):
                        user_id = value.get('id')
                        username = value.get('username')
                
                # Prepare details
                details = {}
                if capture_args:
                    details["args"] = str(args)
                    details["kwargs"] = str(kwargs)
                if capture_result and result is not None:
                    details["result"] = str(result)
                
                # Log the event
                audit_logger.log_security_event(
                    event_type=event_type,
                    user_id=user_id,
                    username=username,
                    description=description or f"Function call: {func.__name__}",
                    details=details,
                    severity=AuditSeverity.ERROR if not success else AuditSeverity.INFO,
                    success=success,
                    error_message=error_message,
                    duration_ms=duration_ms
                )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            error_message = None
            result = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                
                # Extract user information (similar to async wrapper)
                user_id = None
                username = None
                
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                
                for i, arg in enumerate(args):
                    if i < len(params):
                        param_name = params[i]
                        if 'user' in param_name.lower() and isinstance(arg, dict):
                            user_id = arg.get('id')
                            username = arg.get('username')
                
                for key, value in kwargs.items():
                    if 'user' in key.lower() and isinstance(value, dict):
                        user_id = value.get('id')
                        username = value.get('username')
                
                details = {}
                if capture_args:
                    details["args"] = str(args)
                    details["kwargs"] = str(kwargs)
                if capture_result and result is not None:
                    details["result"] = str(result)
                
                audit_logger.log_security_event(
                    event_type=event_type,
                    user_id=user_id,
                    username=username,
                    description=description or f"Function call: {func.__name__}",
                    details=details,
                    severity=AuditSeverity.ERROR if not success else AuditSeverity.INFO,
                    success=success,
                    error_message=error_message,
                    duration_ms=duration_ms
                )
        
        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Context manager for auditing code blocks
@contextmanager
def audit_block(
    event_type: AuditEventType = AuditEventType.API_CALL,
    description: str = "",
    user_id: Optional[str] = None,
    username: Optional[str] = None,
    capture_result: bool = False
):
    """
    Context manager for auditing code blocks.
    
    Args:
        event_type: Event type
        description: Event description
        user_id: User ID
        username: Username
        capture_result: Capture the result
    """
    start_time = time.time()
    success = True
    error_message = None
    result = None
    
    try:
        yield
    except Exception as e:
        success = False
        error_message = str(e)
        raise
    finally:
        duration_ms = (time.time() - start_time) * 1000
        
        details = {}
        if capture_result and result is not None:
            details["result"] = str(result)
        
        audit_logger.log_security_event(
            event_type=event_type,
            user_id=user_id,
            username=username,
            description=description,
            details=details,
            severity=AuditSeverity.ERROR if not success else AuditSeverity.INFO,
            success=success,
            error_message=error_message,
            duration_ms=duration_ms
        )


# Create global audit logger instance
audit_logger = AuditLogger(
    name="worldbrief360",
    enable_console=settings.ENVIRONMENT == "development",
    enable_file=settings.ENVIRONMENT != "development",
    enable_database=True,
    enable_external=settings.AUDIT_EXTERNAL_TYPE is not None,
    batch_size=BATCH_SIZE,
    flush_interval=FLUSH_INTERVAL,
    retention_days=AUDIT_RETENTION_DAYS,
    mask_sensitive_data=True
)

# Export main components
__all__ = [
    "AuditLogger",
    "AuditEvent",
    "AuditEventType",
    "AuditSeverity",
    "AuditSource",
    "AuditContext",
    "AuditLogSchema",
    "AuditLogDependency",
    "audit_log",
    "audit_block",
    "audit_logger",
]