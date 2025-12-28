"""
Background Task model for tracking long-running and deferred tasks in the system.
This model stores metadata, status, and results of background jobs executed by Celery,
async workers, or other task processing systems.
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, Float, Enum, ForeignKey, Boolean, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship, validates
from datetime import datetime
import enum
import json
from typing import Any, Dict, Optional

from app.db.base import Base
from app.core.exceptions import ValidationError


class TaskStatus(str, enum.Enum):
    """
    Status of a background task.
    
    PENDING: Task has been created but not yet started
    QUEUED: Task is in the queue waiting for execution
    RUNNING: Task is currently being executed
    PAUSED: Task execution has been paused (can be resumed)
    COMPLETED: Task has finished successfully
    FAILED: Task execution failed
    CANCELLED: Task was cancelled before completion
    RETRYING: Task failed and is being retried
    EXPIRED: Task timed out
    """
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    EXPIRED = "expired"


class TaskPriority(str, enum.Enum):
    """
    Priority of a background task.
    
    LOW: Low priority tasks (e.g., cleanup, maintenance)
    MEDIUM: Standard priority (default for most tasks)
    HIGH: High priority tasks (e.g., user-facing operations)
    CRITICAL: Critical tasks that should preempt others
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskType(str, enum.Enum):
    """
    Types of background tasks supported by the system.
    """
    # Data operations
    DATA_INGESTION = "data_ingestion"
    DATA_PROCESSING = "data_processing"
    DATA_EXPORT = "data_export"
    DATA_CLEANUP = "data_cleanup"
    
    # Content generation
    BRIEFING_GENERATION = "briefing_generation"
    IMAGE_GENERATION = "image_generation"
    AUDIO_GENERATION = "audio_generation"
    VIDEO_GENERATION = "video_generation"
    CHART_GENERATION = "chart_generation"
    
    # AI/ML operations
    EMBEDDING_GENERATION = "embedding_generation"
    MODEL_TRAINING = "model_training"
    MODEL_INFERENCE = "model_inference"
    MODEL_EVALUATION = "model_evaluation"
    
    # Incident management
    INCIDENT_VERIFICATION = "incident_verification"
    INCIDENT_ANALYSIS = "incident_analysis"
    INCIDENT_CLUSTERING = "incident_clustering"
    
    # User operations
    REPORT_GENERATION = "report_generation"
    NOTIFICATION_SENDING = "notification_sending"
    USER_DATA_EXPORT = "user_data_export"
    
    # System operations
    SYSTEM_BACKUP = "system_backup"
    SYSTEM_CLEANUP = "system_cleanup"
    CACHE_REFRESH = "cache_refresh"
    INDEX_REBUILD = "index_rebuild"
    
    # Integration tasks
    WEBHOOK_PROCESSING = "webhook_processing"
    API_SYNC = "api_sync"
    DATA_SYNC = "data_sync"


class BackgroundTask(Base):
    """
    Model for tracking background tasks.
    
    Each task represents a unit of work that can be executed asynchronously.
    Tasks can be scheduled, monitored, and managed through this model.
    """
    
    __tablename__ = "background_tasks"
    
    # Primary key and identification
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(100), unique=True, index=True, nullable=False, comment="UUID for external reference")
    task_type = Column(Enum(TaskType), nullable=False, index=True, comment="Type of task")
    name = Column(String(255), nullable=True, comment="Human-readable task name")
    description = Column(Text, nullable=True, comment="Task description")
    
    # Status and priority
    status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=False, index=True)
    priority = Column(Enum(TaskPriority), default=TaskPriority.MEDIUM, nullable=False, index=True)
    
    # Ownership and context
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    initiated_by = Column(String(100), nullable=True, comment="Service/component that initiated the task")
    parent_task_id = Column(String(100), nullable=True, index=True, comment="Parent task ID for subtasks")
    correlation_id = Column(String(100), nullable=True, index=True, comment="Correlation ID for request tracing")
    
    # Input and output data
    input_data = Column(JSONB, nullable=True, comment="JSON input parameters for the task")
    output_data = Column(JSONB, nullable=True, comment="JSON output/results from the task")
    error_data = Column(JSONB, nullable=True, comment="JSON error details if task failed")
    
    # Execution details
    worker_id = Column(String(100), nullable=True, comment="ID of worker executing the task")
    queue_name = Column(String(100), nullable=True, index=True, comment="Name of the queue the task is in")
    execution_engine = Column(String(50), nullable=True, comment="Engine used (celery, async, sync, etc.)")
    celery_task_id = Column(String(100), nullable=True, unique=True, comment="Celery task ID if applicable")
    
    # Progress tracking
    progress = Column(Float, default=0.0, comment="Progress percentage (0-100)")
    progress_message = Column(String(500), nullable=True, comment="Current progress message")
    total_steps = Column(Integer, default=1, comment="Total number of steps")
    completed_steps = Column(Integer, default=0, comment="Number of completed steps")
    
    # Time management
    estimated_duration = Column(Integer, nullable=True, comment="Estimated duration in seconds")
    timeout_seconds = Column(Integer, default=3600, comment="Timeout in seconds (1 hour default)")
    max_retries = Column(Integer, default=3, comment="Maximum number of retry attempts")
    retry_count = Column(Integer, default=0, comment="Current retry count")
    retry_delay = Column(Integer, default=60, comment="Delay between retries in seconds")
    
    # Dependencies and relationships
    dependencies = Column(JSONB, nullable=True, comment="JSON array of task IDs this task depends on")
    metadata = Column(JSONB, nullable=True, comment="Additional metadata for the task")
    
    # Resource tracking
    cpu_usage = Column(Float, nullable=True, comment="CPU usage in percentage")
    memory_usage = Column(Integer, nullable=True, comment="Memory usage in bytes")
    disk_usage = Column(Integer, nullable=True, comment="Disk usage in bytes")
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False, index=True)
    scheduled_for = Column(DateTime, nullable=True, index=True, comment="When the task is scheduled to run")
    started_at = Column(DateTime, nullable=True, comment="When the task started execution")
    completed_at = Column(DateTime, nullable=True, comment="When the task completed")
    expires_at = Column(DateTime, nullable=True, index=True, comment="When the task expires")
    
    # Flags
    is_archived = Column(Boolean, default=False, index=True, comment="Whether the task is archived")
    is_notification_sent = Column(Boolean, default=False, comment="Whether completion notification was sent")
    is_manual_review_required = Column(Boolean, default=False, comment="Whether manual review is required")
    
    # Relationships
    user = relationship("User", backref="background_tasks")
    subtasks = relationship(
        "BackgroundTask",
        primaryjoin="BackgroundTask.parent_task_id == foreign(BackgroundTask.task_id)",
        remote_side=[task_id],
        backref="parent",
        lazy="dynamic"
    )
    
    # Indexes
    __table_args__ = (
        Index('ix_background_tasks_status_priority', 'status', 'priority'),
        Index('ix_background_tasks_user_status', 'user_id', 'status'),
        Index('ix_background_tasks_created_status', 'created_at', 'status'),
        Index('ix_background_tasks_scheduled_status', 'scheduled_for', 'status'),
        Index('ix_background_tasks_expires_status', 'expires_at', 'status'),
    )
    
    @validates('task_id')
    def validate_task_id(self, key, task_id):
        """Validate task ID is not empty."""
        if not task_id or len(task_id.strip()) == 0:
            raise ValidationError("Task ID cannot be empty")
        return task_id.strip()
    
    @validates('progress')
    def validate_progress(self, key, progress):
        """Validate progress is between 0 and 100."""
        if progress is not None:
            if progress < 0 or progress > 100:
                raise ValidationError("Progress must be between 0 and 100")
        return progress
    
    @validates('timeout_seconds')
    def validate_timeout(self, key, timeout):
        """Validate timeout is reasonable."""
        if timeout is not None and timeout <= 0:
            raise ValidationError("Timeout must be positive")
        return timeout
    
    @validates('max_retries')
    def validate_max_retries(self, key, retries):
        """Validate max retries is reasonable."""
        if retries < 0:
            raise ValidationError("Max retries cannot be negative")
        if retries > 100:
            raise ValidationError("Max retries cannot exceed 100")
        return retries
    
    def __repr__(self):
        return f"<BackgroundTask {self.task_id} ({self.task_type} - {self.status})>"
    
    def to_dict(self, include_details: bool = True) -> Dict[str, Any]:
        """
        Convert task to dictionary representation.
        
        Args:
            include_details: Whether to include detailed data
            
        Returns:
            Dictionary representation of the task
        """
        result = {
            "id": self.id,
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "user_id": self.user_id,
            "initiated_by": self.initiated_by,
            "parent_task_id": self.parent_task_id,
            "correlation_id": self.correlation_id,
            "progress": self.progress,
            "progress_message": self.progress_message,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "worker_id": self.worker_id,
            "queue_name": self.queue_name,
            "execution_engine": self.execution_engine,
            "celery_task_id": self.celery_task_id,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "retry_delay": self.retry_delay,
            "timeout_seconds": self.timeout_seconds,
            "estimated_duration": self.estimated_duration,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "disk_usage": self.disk_usage,
            "is_archived": self.is_archived,
            "is_notification_sent": self.is_notification_sent,
            "is_manual_review_required": self.is_manual_review_required,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "scheduled_for": self.scheduled_for.isoformat() if self.scheduled_for else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }
        
        if include_details:
            result.update({
                "input_data": self.input_data,
                "output_data": self.output_data,
                "error_data": self.error_data,
                "dependencies": self.dependencies,
                "metadata": self.metadata,
                "subtask_count": self.subtasks.count() if hasattr(self.subtasks, 'count') else 0,
            })
        
        return result
    
    def to_public_dict(self) -> Dict[str, Any]:
        """
        Convert task to public dictionary (excluding sensitive information).
        
        Returns:
            Public-safe dictionary representation
        """
        result = self.to_dict(include_details=False)
        
        # Remove internal fields
        internal_fields = [
            'input_data', 'output_data', 'error_data', 'metadata',
            'worker_id', 'celery_task_id', 'cpu_usage', 'memory_usage',
            'disk_usage'
        ]
        
        for field in internal_fields:
            if field in result:
                del result[field]
        
        return result
    
    @property
    def duration(self) -> Optional[float]:
        """
        Calculate task duration in seconds.
        
        Returns:
            Duration in seconds or None if not measurable
        """
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.utcnow() - self.started_at).total_seconds()
        return None
    
    @property
    def is_active(self) -> bool:
        """
        Check if task is currently active (running or retrying).
        
        Returns:
            True if task is active
        """
        return self.status in [TaskStatus.RUNNING, TaskStatus.RETRYING]
    
    @property
    def is_finished(self) -> bool:
        """
        Check if task is finished (completed, failed, cancelled, or expired).
        
        Returns:
            True if task is finished
        """
        return self.status in [
            TaskStatus.COMPLETED, 
            TaskStatus.FAILED, 
            TaskStatus.CANCELLED,
            TaskStatus.EXPIRED
        ]
    
    @property
    def can_retry(self) -> bool:
        """
        Check if task can be retried.
        
        Returns:
            True if task can be retried
        """
        return (
            self.status == TaskStatus.FAILED and 
            self.retry_count < self.max_retries
        )
    
    @property
    def is_expired(self) -> bool:
        """
        Check if task has expired.
        
        Returns:
            True if task has expired
        """
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False
    
    @property
    def progress_percentage(self) -> float:
        """
        Calculate progress as percentage of completed steps.
        
        Returns:
            Progress percentage (0-100)
        """
        if self.total_steps > 0:
            return (self.completed_steps / self.total_steps) * 100
        return self.progress or 0.0
    
    def add_progress(self, steps: int = 1, message: Optional[str] = None):
        """
        Increment progress by given number of steps.
        
        Args:
            steps: Number of steps to increment
            message: Optional progress message
        """
        self.completed_steps += steps
        self.progress = self.progress_percentage
        
        if message:
            self.progress_message = message
        
        # Ensure we don't exceed total steps
        if self.completed_steps > self.total_steps:
            self.completed_steps = self.total_steps
    
    def set_error(self, error_message: str, error_details: Optional[Dict[str, Any]] = None):
        """
        Set error information for the task.
        
        Args:
            error_message: Human-readable error message
            error_details: Additional error details
        """
        self.error_data = {
            "message": error_message,
            "details": error_details or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.utcnow()
    
    def set_output(self, output_data: Dict[str, Any]):
        """
        Set output data for the task.
        
        Args:
            output_data: Output data from task execution
        """
        self.output_data = output_data
        self.status = TaskStatus.COMPLETED
        self.progress = 100.0
        self.completed_steps = self.total_steps
        self.completed_at = datetime.utcnow()
    
    def mark_for_retry(self, delay: Optional[int] = None):
        """
        Mark task for retry.
        
        Args:
            delay: Optional custom retry delay in seconds
        """
        if not self.can_retry:
            raise ValidationError("Task cannot be retried")
        
        self.status = TaskStatus.RETRYING
        self.retry_count += 1
        
        if delay is not None:
            self.retry_delay = delay
        
        # Schedule for retry
        from datetime import timedelta
        self.scheduled_for = datetime.utcnow() + timedelta(seconds=self.retry_delay)
    
    def cancel(self, reason: Optional[str] = None):
        """
        Cancel the task.
        
        Args:
            reason: Optional cancellation reason
        """
        if self.is_finished:
            raise ValidationError("Cannot cancel a finished task")
        
        self.status = TaskStatus.CANCELLED
        self.completed_at = datetime.utcnow()
        
        if reason:
            self.error_data = {
                "cancellation_reason": reason,
                "cancelled_at": datetime.utcnow().isoformat()
            }


class TaskDependency(Base):
    """
    Model for tracking dependencies between tasks.
    Allows for complex task workflows with dependencies.
    """
    
    __tablename__ = "task_dependencies"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(100), ForeignKey("background_tasks.task_id", ondelete="CASCADE"), nullable=False, index=True)
    depends_on_task_id = Column(String(100), ForeignKey("background_tasks.task_id", ondelete="CASCADE"), nullable=False, index=True)
    dependency_type = Column(String(50), default="completion", comment="Type of dependency (completion, success, etc.)")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    task = relationship("BackgroundTask", foreign_keys=[task_id], backref="dependent_tasks")
    depends_on = relationship("BackgroundTask", foreign_keys=[depends_on_task_id], backref="dependencies_of")
    
    __table_args__ = (
        Index('ix_task_dependencies_task_depends', 'task_id', 'depends_on_task_id', unique=True),
    )
    
    def __repr__(self):
        return f"<TaskDependency {self.task_id} -> {self.depends_on_task_id}>"