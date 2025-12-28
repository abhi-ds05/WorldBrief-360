"""
Pydantic schemas for feature flag API and data validation.
"""

from typing import Dict, Any, Optional, List, Union, Set
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, validator, root_validator
import json

from .flags import FlagType, VariantType


# ==================== REQUEST SCHEMAS ====================

class FeatureFlagCreate(BaseModel):
    """Schema for creating a new feature flag."""
    
    name: str = Field(
        ...,
        description="Unique identifier for the feature flag",
        min_length=1,
        max_length=100,
        regex=r"^[a-z0-9_]+$",  # snake_case
        example="dark_mode"
    )
    
    description: str = Field(
        ...,
        description="Human-readable description of the feature flag",
        min_length=1,
        max_length=500,
        example="Enable dark mode interface"
    )
    
    flag_type: FlagType = Field(
        FlagType.BOOLEAN,
        description="Type of feature flag"
    )
    
    enabled: bool = Field(
        False,
        description="Whether the flag is globally enabled"
    )
    
    variants: Dict[str, Any] = Field(
        default_factory=dict,
        description="Available variants for multivariate flags"
    )
    
    default_variant: Optional[Any] = Field(
        None,
        description="Default variant value when flag is disabled"
    )
    
    rollout_percentage: float = Field(
        0.0,
        description="Percentage rollout (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    target_users: List[str] = Field(
        default_factory=list,
        description="Specific user IDs to target"
    )
    
    target_segments: List[str] = Field(
        default_factory=list,
        description="User segments to target"
    )
    
    start_time: Optional[datetime] = Field(
        None,
        description="When the flag becomes active"
    )
    
    end_time: Optional[datetime] = Field(
        None,
        description="When the flag expires"
    )
    
    environments: List[str] = Field(
        default_factory=lambda: ["development", "staging", "production"],
        description="Environments where flag is available"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional flag metadata"
    )
    
    @validator('name')
    def validate_name(cls, v):
        """Validate flag name."""
        if not v:
            raise ValueError("Flag name cannot be empty")
        return v
    
    @validator('description')
    def validate_description(cls, v):
        """Validate description."""
        if not v.strip():
            raise ValueError("Description cannot be empty")
        return v.strip()
    
    @validator('variants')
    def validate_variants(cls, v, values):
        """Validate variants based on flag type."""
        flag_type = values.get('flag_type')
        
        if flag_type == FlagType.MULTIVARIATE and not v:
            raise ValueError("Multivariate flags must have at least one variant")
        
        if flag_type == FlagType.BOOLEAN and v:
            raise ValueError("Boolean flags should not have variants")
        
        return v
    
    @validator('start_time', 'end_time')
    def validate_times(cls, v, values, field):
        """Validate time constraints."""
        if field.name == 'end_time' and v:
            start_time = values.get('start_time')
            if start_time and v <= start_time:
                raise ValueError("End time must be after start time")
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
        schema_extra = {
            "example": {
                "name": "enhanced_summarization",
                "description": "Use advanced summarization models",
                "flag_type": "percentage",
                "enabled": True,
                "rollout_percentage": 0.5,
                "environments": ["production", "staging"],
                "metadata": {
                    "owner": "ai-team",
                    "jira_ticket": "AI-123"
                }
            }
        }


class FeatureFlagUpdate(BaseModel):
    """Schema for updating an existing feature flag."""
    
    description: Optional[str] = Field(
        None,
        description="Updated description",
        min_length=1,
        max_length=500
    )
    
    enabled: Optional[bool] = Field(
        None,
        description="Updated enabled state"
    )
    
    variants: Optional[Dict[str, Any]] = Field(
        None,
        description="Updated variants"
    )
    
    default_variant: Optional[Any] = Field(
        None,
        description="Updated default variant"
    )
    
    rollout_percentage: Optional[float] = Field(
        None,
        description="Updated rollout percentage",
        ge=0.0,
        le=1.0
    )
    
    target_users: Optional[List[str]] = Field(
        None,
        description="Updated target users"
    )
    
    target_segments: Optional[List[str]] = Field(
        None,
        description="Updated target segments"
    )
    
    start_time: Optional[datetime] = Field(
        None,
        description="Updated start time"
    )
    
    end_time: Optional[datetime] = Field(
        None,
        description="Updated end time"
    )
    
    environments: Optional[List[str]] = Field(
        None,
        description="Updated environments"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Updated metadata"
    )
    
    @validator('rollout_percentage')
    def validate_rollout(cls, v):
        """Validate rollout percentage."""
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("Rollout percentage must be between 0.0 and 1.0")
        return v
    
    @validator('start_time', 'end_time')
    def validate_times(cls, v, values, field):
        """Validate time constraints."""
        if field.name == 'end_time' and v:
            start_time = values.get('start_time')
            if start_time and v <= start_time:
                raise ValueError("End time must be after start time")
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class FeatureFlagBatchUpdate(BaseModel):
    """Schema for batch updating feature flags."""
    
    updates: Dict[str, FeatureFlagUpdate] = Field(
        ...,
        description="Dictionary of flag name to update data"
    )
    
    user_id: Optional[str] = Field(
        None,
        description="User performing the update"
    )


class EvaluationRequest(BaseModel):
    """Schema for evaluating a feature flag."""
    
    user_id: Optional[str] = Field(
        None,
        description="User identifier for evaluation"
    )
    
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Evaluation context"
    )
    
    environment: str = Field(
        "production",
        description="Environment for evaluation"
    )


class BatchEvaluationRequest(BaseModel):
    """Schema for batch evaluating feature flags."""
    
    flags: List[str] = Field(
        ...,
        description="List of feature flags to evaluate"
    )
    
    user_id: Optional[str] = Field(
        None,
        description="User identifier for evaluation"
    )
    
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Evaluation context"
    )
    
    environment: str = Field(
        "production",
        description="Environment for evaluation"
    )


class SegmentCreate(BaseModel):
    """Schema for creating a user segment."""
    
    name: str = Field(
        ...,
        description="Unique segment name",
        min_length=1,
        max_length=100,
        regex=r"^[a-z0-9_]+$"
    )
    
    description: str = Field(
        ...,
        description="Segment description",
        min_length=1,
        max_length=500
    )
    
    rules: List[Dict[str, Any]] = Field(
        ...,
        description="Segment rules"
    )
    
    priority: int = Field(
        0,
        description="Segment priority",
        ge=0
    )
    
    @validator('name')
    def validate_name(cls, v):
        """Validate segment name."""
        if not v:
            raise ValueError("Segment name cannot be empty")
        return v


class SegmentTestRequest(BaseModel):
    """Schema for testing segment matching."""
    
    context: Dict[str, Any] = Field(
        ...,
        description="Context to test against segment"
    )


# ==================== RESPONSE SCHEMAS ====================

class VariantAssignment(BaseModel):
    """Schema for variant assignment in evaluation."""
    
    variant: Any = Field(
        ...,
        description="Assigned variant value"
    )
    
    variant_name: Optional[str] = Field(
        None,
        description="Name of the variant (for multivariate flags)"
    )
    
    is_default: bool = Field(
        False,
        description="Whether this is the default variant"
    )


class FeatureFlagResponse(BaseModel):
    """Schema for feature flag evaluation response."""
    
    flag_name: str = Field(
        ...,
        description="Name of the feature flag"
    )
    
    enabled: bool = Field(
        ...,
        description="Whether the flag is enabled for the user"
    )
    
    variant: Optional[Any] = Field(
        None,
        description="Assigned variant (if any)"
    )
    
    variant_assignment: Optional[VariantAssignment] = Field(
        None,
        description="Detailed variant assignment"
    )
    
    reason: str = Field(
        ...,
        description="Reason for the evaluation result"
    )
    
    flag_type: FlagType = Field(
        ...,
        description="Type of feature flag"
    )
    
    environment: str = Field(
        "production",
        description="Environment used for evaluation"
    )
    
    user_id: Optional[str] = Field(
        None,
        description="User identifier"
    )
    
    segments: List[str] = Field(
        default_factory=list,
        description="User segments that matched"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Evaluation timestamp"
    )
    
    error: bool = Field(
        False,
        description="Whether an error occurred"
    )
    
    error_message: Optional[str] = Field(
        None,
        description="Error message if evaluation failed"
    )
    
    @validator('reason')
    def validate_reason(cls, v):
        """Validate reason field."""
        allowed_reasons = {
            "flag_enabled",
            "flag_disabled",
            "targeted_user",
            "segment_match",
            "percentage_rollout",
            "environment_mismatch",
            "time_constraint",
            "flag_not_found",
            "evaluation_error",
        }
        if v not in allowed_reasons:
            raise ValueError(f"Invalid reason: {v}")
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BatchEvaluationResponse(BaseModel):
    """Schema for batch evaluation response."""
    
    evaluations: Dict[str, FeatureFlagResponse] = Field(
        ...,
        description="Evaluation results for each flag"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Evaluation timestamp"
    )
    
    environment: str = Field(
        "production",
        description="Environment used for evaluation"
    )


class FeatureFlagDetail(BaseModel):
    """Schema for detailed feature flag information."""
    
    name: str = Field(
        ...,
        description="Flag name"
    )
    
    description: str = Field(
        ...,
        description="Flag description"
    )
    
    flag_type: FlagType = Field(
        ...,
        description="Flag type"
    )
    
    enabled: bool = Field(
        ...,
        description="Global enabled state"
    )
    
    variants: Dict[str, Any] = Field(
        default_factory=dict,
        description="Available variants"
    )
    
    default_variant: Optional[Any] = Field(
        None,
        description="Default variant"
    )
    
    rollout_percentage: float = Field(
        ...,
        description="Rollout percentage"
    )
    
    target_users: List[str] = Field(
        default_factory=list,
        description="Targeted users"
    )
    
    target_segments: List[str] = Field(
        default_factory=list,
        description="Targeted segments"
    )
    
    start_time: Optional[datetime] = Field(
        None,
        description="Start time"
    )
    
    end_time: Optional[datetime] = Field(
        None,
        description="End time"
    )
    
    environments: List[str] = Field(
        ...,
        description="Available environments"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Flag metadata"
    )
    
    created_at: Optional[datetime] = Field(
        None,
        description="Creation timestamp"
    )
    
    updated_at: Optional[datetime] = Field(
        None,
        description="Last update timestamp"
    )
    
    created_by: Optional[str] = Field(
        None,
        description="User who created the flag"
    )
    
    updated_by: Optional[str] = Field(
        None,
        description="User who last updated the flag"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class FeatureFlagList(BaseModel):
    """Schema for listing feature flags."""
    
    flags: List[FeatureFlagDetail] = Field(
        ...,
        description="List of feature flags"
    )
    
    total: int = Field(
        ...,
        description="Total number of flags"
    )
    
    page: int = Field(
        1,
        description="Current page"
    )
    
    page_size: int = Field(
        50,
        description="Page size"
    )
    
    has_more: bool = Field(
        ...,
        description="Whether there are more pages"
    )


class SegmentResponse(BaseModel):
    """Schema for segment response."""
    
    name: str = Field(
        ...,
        description="Segment name"
    )
    
    description: str = Field(
        ...,
        description="Segment description"
    )
    
    rules: List[Dict[str, Any]] = Field(
        ...,
        description="Segment rules"
    )
    
    priority: int = Field(
        ...,
        description="Segment priority"
    )
    
    user_count: Optional[int] = Field(
        None,
        description="Number of users in segment"
    )
    
    created_at: Optional[datetime] = Field(
        None,
        description="Creation timestamp"
    )
    
    updated_at: Optional[datetime] = Field(
        None,
        description="Last update timestamp"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class SegmentTestResponse(BaseModel):
    """Schema for segment test response."""
    
    matches: bool = Field(
        ...,
        description="Whether context matches segment"
    )
    
    matched_rules: List[str] = Field(
        ...,
        description="List of rules that matched"
    )
    
    failed_rules: List[str] = Field(
        ...,
        description="List of rules that failed"
    )


class EvaluationMetrics(BaseModel):
    """Schema for evaluation metrics."""
    
    flag_name: str = Field(
        ...,
        description="Flag name"
    )
    
    total_evaluations: int = Field(
        ...,
        description="Total number of evaluations"
    )
    
    enabled_evaluations: int = Field(
        ...,
        description="Number of enabled evaluations"
    )
    
    disabled_evaluations: int = Field(
        ...,
        description="Number of disabled evaluations"
    )
    
    enabled_percentage: float = Field(
        ...,
        description="Percentage of enabled evaluations"
    )
    
    variant_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Counts for each variant"
    )
    
    error_count: int = Field(
        ...,
        description="Number of evaluation errors"
    )
    
    error_percentage: float = Field(
        ...,
        description="Percentage of evaluation errors"
    )
    
    last_evaluated: Optional[datetime] = Field(
        None,
        description="Last evaluation timestamp"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class FlagUpdateResponse(BaseModel):
    """Schema for flag update response."""
    
    flag_name: str = Field(
        ...,
        description="Flag name"
    )
    
    old_value: Optional[Dict[str, Any]] = Field(
        None,
        description="Old flag value"
    )
    
    new_value: Optional[Dict[str, Any]] = Field(
        None,
        description="New flag value"
    )
    
    timestamp: datetime = Field(
        ...,
        description="Update timestamp"
    )
    
    source: str = Field(
        ...,
        description="Update source"
    )
    
    user_id: Optional[str] = Field(
        None,
        description="User who performed update"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SystemStats(BaseModel):
    """Schema for system statistics."""
    
    total_flags: int = Field(
        ...,
        description="Total number of feature flags"
    )
    
    enabled_flags: int = Field(
        ...,
        description="Number of enabled flags"
    )
    
    disabled_flags: int = Field(
        ...,
        description="Number of disabled flags"
    )
    
    cache_size: int = Field(
        ...,
        description="Number of flags in cache"
    )
    
    total_evaluations: int = Field(
        ...,
        description="Total number of evaluations"
    )
    
    total_errors: int = Field(
        ...,
        description="Total number of errors"
    )
    
    error_rate: float = Field(
        ...,
        description="Error rate percentage"
    )
    
    recent_updates: int = Field(
        ...,
        description="Number of recent updates"
    )
    
    evaluation_history_size: int = Field(
        ...,
        description="Size of evaluation history"
    )
    
    segments_count: int = Field(
        ...,
        description="Number of user segments"
    )
    
    backend_type: str = Field(
        ...,
        description="Type of backend"
    )
    
    update_strategy: str = Field(
        ...,
        description="Update strategy"
    )
    
    polling_interval: int = Field(
        ...,
        description="Polling interval in seconds"
    )
    
    cache_ttl: int = Field(
        ...,
        description="Cache TTL in seconds"
    )
    
    uptime: Optional[float] = Field(
        None,
        description="System uptime in seconds"
    )


class HealthCheckResponse(BaseModel):
    """Schema for health check response."""
    
    status: str = Field(
        ...,
        description="Health status"
    )
    
    version: str = Field(
        ...,
        description="System version"
    )
    
    backend_connected: bool = Field(
        ...,
        description="Whether backend is connected"
    )
    
    cache_healthy: bool = Field(
        ...,
        description="Whether cache is healthy"
    )
    
    last_update: Optional[datetime] = Field(
        None,
        description="Last successful update"
    )
    
    metrics_enabled: bool = Field(
        ...,
        description="Whether metrics are enabled"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


# ==================== WEBHOOK SCHEMAS ====================

class WebhookEvent(str, Enum):
    """Types of webhook events."""
    FLAG_CREATED = "flag.created"
    FLAG_UPDATED = "flag.updated"
    FLAG_DELETED = "flag.deleted"
    FLAG_EVALUATED = "flag.evaluated"
    SEGMENT_CREATED = "segment.created"
    SEGMENT_UPDATED = "segment.updated"
    SEGMENT_DELETED = "segment.deleted"


class WebhookPayload(BaseModel):
    """Schema for webhook payload."""
    
    event: WebhookEvent = Field(
        ...,
        description="Webhook event type"
    )
    
    data: Dict[str, Any] = Field(
        ...,
        description="Event data"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Event timestamp"
    )
    
    source: str = Field(
        "feature-flag-system",
        description="Event source"
    )
    
    signature: Optional[str] = Field(
        None,
        description="Payload signature for verification"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class WebhookRegistration(BaseModel):
    """Schema for webhook registration."""
    
    url: str = Field(
        ...,
        description="Webhook URL"
    )
    
    events: List[WebhookEvent] = Field(
        ...,
        description="Events to subscribe to"
    )
    
    secret: Optional[str] = Field(
        None,
        description="Secret for signature verification"
    )
    
    enabled: bool = Field(
        True,
        description="Whether webhook is enabled"
    )
    
    timeout: int = Field(
        5,
        description="Request timeout in seconds",
        ge=1,
        le=30
    )
    
    retry_count: int = Field(
        3,
        description="Number of retry attempts",
        ge=0,
        le=10
    )


# ==================== ADMIN SCHEMAS ====================

class AdminFlagImport(BaseModel):
    """Schema for importing flags from external source."""
    
    source: str = Field(
        ...,
        description="Import source (json, unleash, launchdarkly)"
    )
    
    data: Union[str, Dict[str, Any]] = Field(
        ...,
        description="Import data"
    )
    
    overwrite: bool = Field(
        False,
        description="Whether to overwrite existing flags"
    )
    
    @validator('data')
    def validate_data(cls, v):
        """Validate import data."""
        if isinstance(v, str):
            try:
                json.loads(v)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON string")
        return v


class AdminFlagExport(BaseModel):
    """Schema for exporting flags."""
    
    format: str = Field(
        "json",
        description="Export format",
        regex="^(json|yaml|csv)$"
    )
    
    include_disabled: bool = Field(
        True,
        description="Include disabled flags"
    )
    
    include_metadata: bool = Field(
        True,
        description="Include metadata"
    )


class AdminAuditQuery(BaseModel):
    """Schema for audit log queries."""
    
    start_time: Optional[datetime] = Field(
        None,
        description="Start time for query"
    )
    
    end_time: Optional[datetime] = Field(
        None,
        description="End time for query"
    )
    
    flag_name: Optional[str] = Field(
        None,
        description="Filter by flag name"
    )
    
    user_id: Optional[str] = Field(
        None,
        description="Filter by user ID"
    )
    
    event_type: Optional[str] = Field(
        None,
        description="Filter by event type"
    )
    
    limit: int = Field(
        100,
        description="Maximum number of results",
        ge=1,
        le=1000
    )
    
    offset: int = Field(
        0,
        description="Result offset",
        ge=0
    )


# ==================== ERROR SCHEMAS ====================

class ErrorResponse(BaseModel):
    """Schema for error responses."""
    
    error: str = Field(
        ...,
        description="Error type"
    )
    
    message: str = Field(
        ...,
        description="Error message"
    )
    
    detail: Optional[Dict[str, Any]] = Field(
        None,
        description="Error details"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp"
    )
    
    request_id: Optional[str] = Field(
        None,
        description="Request ID for tracing"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ValidationError(BaseModel):
    """Schema for validation errors."""
    
    loc: List[str] = Field(
        ...,
        description="Error location"
    )
    
    msg: str = Field(
        ...,
        description="Error message"
    )
    
    type: str = Field(
        ...,
        description="Error type"
    )


class ValidationErrorResponse(BaseModel):
    """Schema for validation error responses."""
    
    errors: List[ValidationError] = Field(
        ...,
        description="List of validation errors"
    )


# ==================== HELPER SCHEMAS ====================

class PaginationParams(BaseModel):
    """Schema for pagination parameters."""
    
    page: int = Field(
        1,
        description="Page number",
        ge=1
    )
    
    page_size: int = Field(
        50,
        description="Page size",
        ge=1,
        le=100
    )


class SortParams(BaseModel):
    """Schema for sorting parameters."""
    
    sort_by: str = Field(
        "name",
        description="Field to sort by"
    )
    
    sort_order: str = Field(
        "asc",
        description="Sort order (asc/desc)",
        regex="^(asc|desc)$"
    )


class FilterParams(BaseModel):
    """Schema for filter parameters."""
    
    enabled: Optional[bool] = Field(
        None,
        description="Filter by enabled state"
    )
    
    flag_type: Optional[FlagType] = Field(
        None,
        description="Filter by flag type"
    )
    
    environment: Optional[str] = Field(
        None,
        description="Filter by environment"
    )
    
    search: Optional[str] = Field(
        None,
        description="Search in name and description"
    )


# Export all schemas
__all__ = [
    # Request schemas
    "FeatureFlagCreate",
    "FeatureFlagUpdate",
    "FeatureFlagBatchUpdate",
    "EvaluationRequest",
    "BatchEvaluationRequest",
    "SegmentCreate",
    "SegmentTestRequest",
    
    # Response schemas
    "VariantAssignment",
    "FeatureFlagResponse",
    "BatchEvaluationResponse",
    "FeatureFlagDetail",
    "FeatureFlagList",
    "SegmentResponse",
    "SegmentTestResponse",
    "EvaluationMetrics",
    "FlagUpdateResponse",
    "SystemStats",
    "HealthCheckResponse",
    
    # Webhook schemas
    "WebhookEvent",
    "WebhookPayload",
    "WebhookRegistration",
    
    # Admin schemas
    "AdminFlagImport",
    "AdminFlagExport",
    "AdminAuditQuery",
    
    # Error schemas
    "ErrorResponse",
    "ValidationError",
    "ValidationErrorResponse",
    
    # Helper schemas
    "PaginationParams",
    "SortParams",
    "FilterParams",
]