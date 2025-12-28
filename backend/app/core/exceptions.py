"""
Custom exceptions for the application.
"""

from typing import Any, Dict, Optional, List
from http import HTTPStatus


class AppException(Exception):
    """
    Base exception for all application errors.
    
    Attributes:
        message: Error message
        status_code: HTTP status code
        code: Application error code
        details: Additional error details
    """
    
    def __init__(
        self,
        message: str,
        status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.code = code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details
            }
        }
    
    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


class ConfigurationError(AppException):
    """Raised when there's a configuration error."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        config_key: Optional[str] = None
    ):
        if config_key:
            details = details or {}
            details["config_key"] = config_key
        super().__init__(
            message=message,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            code="CONFIGURATION_ERROR",
            details=details
        )


class SecurityError(AppException):
    """Raised when there's a security-related error."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        security_context: Optional[str] = None
    ):
        if security_context:
            details = details or {}
            details["security_context"] = security_context
        super().__init__(
            message=message,
            status_code=HTTPStatus.UNAUTHORIZED,
            code="SECURITY_ERROR",
            details=details
        )


class ValidationError(AppException):
    """Raised when data validation fails."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        field_errors: Optional[Dict[str, List[str]]] = None
    ):
        if field_errors:
            details = details or {}
            details["field_errors"] = field_errors
        super().__init__(
            message=message,
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            code="VALIDATION_ERROR",
            details=details
        )


class DatabaseError(AppException):
    """Raised when there's a database error."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        operation: Optional[str] = None,
        query: Optional[str] = None
    ):
        if operation:
            details = details or {}
            details["operation"] = operation
        if query:
            details = details or {}
            details["query"] = query
        super().__init__(
            message=message,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            code="DATABASE_ERROR",
            details=details
        )


class CacheError(AppException):
    """Raised when there's a cache error."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cache_key: Optional[str] = None,
        operation: Optional[str] = None
    ):
        if cache_key:
            details = details or {}
            details["cache_key"] = cache_key
        if operation:
            details = details or {}
            details["operation"] = operation
        super().__init__(
            message=message,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            code="CACHE_ERROR",
            details=details
        )


class ExternalServiceError(AppException):
    """Raised when there's an error with an external service."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        service_name: Optional[str] = None,
        status_code: Optional[int] = None
    ):
        if service_name:
            details = details or {}
            details["service_name"] = service_name
        if status_code:
            details = details or {}
            details["status_code"] = status_code
        
        http_status = HTTPStatus.BAD_GATEWAY
        if status_code and 400 <= status_code < 500:
            http_status = HTTPStatus.BAD_REQUEST
        
        super().__init__(
            message=message,
            status_code=http_status,
            code="EXTERNAL_SERVICE_ERROR",
            details=details
        )


class NotFoundError(AppException):
    """Raised when a resource is not found."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[Any] = None
    ):
        if resource_type:
            details = details or {}
            details["resource_type"] = resource_type
        if resource_id:
            details = details or {}
            details["resource_id"] = resource_id
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_FOUND,
            code="NOT_FOUND",
            details=details
        )


class AuthenticationError(SecurityError):
    """Raised when authentication fails."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            details=details,
            security_context="authentication"
        )


class AuthorizationError(SecurityError):
    """Raised when authorization fails."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        required_permissions: Optional[List[str]] = None
    ):
        if required_permissions:
            details = details or {}
            details["required_permissions"] = required_permissions
        super().__init__(
            message=message,
            details=details,
            security_context="authorization"
        )


class RateLimitError(AppException):
    """Raised when rate limit is exceeded."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        reset_time: Optional[int] = None
    ):
        if limit:
            details = details or {}
            details["limit"] = limit
        if reset_time:
            details = details or {}
            details["reset_time"] = reset_time
        super().__init__(
            message=message,
            status_code=HTTPStatus.TOO_MANY_REQUESTS,
            code="RATE_LIMIT_EXCEEDED",
            details=details
        )


class TimeoutError(AppException):
    """Raised when an operation times out."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        operation: Optional[str] = None
    ):
        if timeout:
            details = details or {}
            details["timeout"] = timeout
        if operation:
            details = details or {}
            details["operation"] = operation
        super().__init__(
            message=message,
            status_code=HTTPStatus.GATEWAY_TIMEOUT,
            code="TIMEOUT_ERROR",
            details=details
        )


# Convenience functions
def raise_not_found(
    resource_type: str,
    resource_id: Any,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """Raise a NotFoundError with a standard message."""
    raise NotFoundError(
        message=f"{resource_type} with ID {resource_id} not found",
        details=details,
        resource_type=resource_type,
        resource_id=resource_id
    )


def raise_validation_error(
    message: str,
    field_errors: Optional[Dict[str, List[str]]] = None
) -> None:
    """Raise a ValidationError with field errors."""
    raise ValidationError(
        message=message,
        field_errors=field_errors
    )