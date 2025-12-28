"""
Moderation Log model for tracking all moderation actions in the system.
This includes automated moderation, manual moderation, user reports, and appeals.
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, Enum, ForeignKey, Boolean, JSON, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship, validates,Float,List
from datetime import timedelta
from datetime import datetime
import enum
import json
import select
from typing import Any, Dict, Optional

from app.db.base import Base
from app.core.exceptions import ValidationError


class ModerationAction(str, enum.Enum):
    """
    Types of moderation actions that can be taken.
    
    NO_ACTION: No action was taken (just logged)
    WARN_USER: User was warned about their behavior
    REMOVE_CONTENT: Content was removed/hidden
    REDACT_CONTENT: Sensitive parts of content were redacted
    FLAG_FOR_REVIEW: Content was flagged for manual review
    SUSPEND_USER: User account was temporarily suspended
    BAN_USER: User account was permanently banned
    RESTORE_CONTENT: Previously removed content was restored
    REDACT_AND_WARN: Content was redacted and user was warned
    REMOVE_CONTENT_AND_WARN: Content removed and user warned
    REMOVE_CONTENT_AND_SUSPEND: Content removed and user suspended
    APPEAL_APPROVED: User appeal was approved
    APPEAL_REJECTED: User appeal was rejected
    ESCALATE_TO_ADMIN: Case was escalated to admin
    """
    NO_ACTION = "no_action"
    WARN_USER = "warn_user"
    REMOVE_CONTENT = "remove_content"
    REDACT_CONTENT = "redact_content"
    FLAG_FOR_REVIEW = "flag_for_review"
    SUSPEND_USER = "suspend_user"
    BAN_USER = "ban_user"
    RESTORE_CONTENT = "restore_content"
    REDACT_AND_WARN = "redact_and_warn"
    REMOVE_CONTENT_AND_WARN = "remove_content_and_warn"
    REMOVE_CONTENT_AND_SUSPEND = "remove_content_and_suspension"
    APPEAL_APPROVED = "appeal_approved"
    APPEAL_REJECTED = "appeal_rejected"
    ESCALATE_TO_ADMIN = "escalate_to_admin"


class ContentType(str, enum.Enum):
    """
    Types of content that can be moderated.
    """
    USER = "user"                     # User profile/account
    INCIDENT = "incident"             # Incident report
    INCIDENT_IMAGE = "incident_image" # Incident media
    COMMENT = "comment"               # User comment
    CHAT = "chat"                     # Chat message
    BRIEFING = "briefing"             # Generated briefing
    TOPIC = "topic"                   # Topic/page
    ARTICLE = "article"               # News/article
    IMAGE = "image"                   # Generated/uploaded image
    AUDIO = "audio"                   # Generated/uploaded audio
    VIDEO = "video"                   # Generated/uploaded video
    POLL = "poll"                     # Poll/survey
    REVIEW = "review"                 # User review
    FEEDBACK = "feedback"             # System feedback
    MESSAGE = "message"               # Private message
    GROUP = "group"                   # User group/community


class ModerationSeverity(str, enum.Enum):
    """
    Severity levels for moderation violations.
    """
    INFO = "info"          # Informational only
    LOW = "low"            # Minor violation
    MEDIUM = "medium"      # Moderate violation
    HIGH = "high"          # Serious violation
    CRITICAL = "critical"  # Severe violation requiring immediate action


class ModerationSource(str, enum.Enum):
    """
    Source of the moderation action.
    """
    AUTOMATED = "automated"      # Automated system
    MANUAL = "manual"            # Human moderator
    USER_REPORT = "user_report"  # Reported by user
    SYSTEM = "system"            # System-generated
    API = "api"                  # External API/moderation service
    APPEAL = "appeal"            # User appeal process


class AppealStatus(str, enum.Enum):
    """
    Status of a moderation appeal.
    """
    PENDING = "pending"      # Appeal submitted, awaiting review
    UNDER_REVIEW = "under_review"  # Being reviewed by moderator
    APPROVED = "approved"    # Appeal approved, action reversed
    REJECTED = "rejected"    # Appeal rejected, action stands
    ESCALATED = "escalated"  # Escalated to higher authority
    WITHDRAWN = "withdrawn"  # Appeal withdrawn by user
    EXPIRED = "expired"      # Appeal expired (time limit)


class ModerationLog(Base):
    """
    Main model for tracking all moderation actions.
    
    Each record represents a single moderation action taken on content or user.
    This provides a complete audit trail for all moderation activities.
    """
    
    __tablename__ = "moderation_logs"
    
    # Primary key and identification
    id = Column(Integer, primary_key=True, index=True)
    log_id = Column(String(100), unique=True, index=True, nullable=False, comment="UUID for external reference")
    
    # Action details
    action = Column(Enum(ModerationAction), nullable=False, index=True)
    action_taken_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Content being moderated
    content_type = Column(Enum(ContentType), nullable=False, index=True)
    content_id = Column(String(100), nullable=False, index=True, comment="ID of the moderated content")
    content_url = Column(String(500), nullable=True, comment="URL to the content if available")
    
    # Original content (for restoration/audit)
    original_content = Column(Text, nullable=True, comment="Original content before moderation")
    modified_content = Column(Text, nullable=True, comment="Content after moderation")
    
    # Users involved
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True, comment="User who created the content")
    moderator_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True, comment="Moderator who took action")
    reporter_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True, comment="User who reported the content")
    
    # Moderation details
    severity = Column(Enum(ModerationSeverity), default=ModerationSeverity.MEDIUM, nullable=False, index=True)
    source = Column(Enum(ModerationSource), default=ModerationSource.AUTOMATED, nullable=False, index=True)
    reason = Column(Text, nullable=False, comment="Reason for moderation action")
    detailed_reason = Column(Text, nullable=True, comment="Detailed explanation")
    
    # Rule matching (for automated moderation)
    matched_rules = Column(JSONB, nullable=True, comment="JSON array of rule names that were matched")
    confidence_score = Column(Integer, nullable=True, comment="Confidence score (0-100) for automated moderation")
    
    # Appeal information
    appeal_status = Column(Enum(AppealStatus), nullable=True, index=True)
    appeal_submitted_at = Column(DateTime, nullable=True)
    appeal_reviewed_at = Column(DateTime, nullable=True)
    appeal_reviewed_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    appeal_notes = Column(Text, nullable=True, comment="Notes from appeal review")
    
    # Duration (for suspensions/bans)
    duration_days = Column(Integer, nullable=True, comment="Duration in days for suspensions")
    expires_at = Column(DateTime, nullable=True, index=True, comment="When suspension/ban expires")
    
    # Evidence and context
    evidence = Column(JSONB, nullable=True, comment="JSON evidence (screenshots, logs, etc.)")
    context_data = Column(JSONB, nullable=True, comment="JSON context data at time of moderation")
    metadata = Column(JSONB, nullable=True, comment="Additional metadata")
    
    # Communication
    user_notified = Column(Boolean, default=False, comment="Whether user was notified")
    notification_sent_at = Column(DateTime, nullable=True)
    moderator_notes = Column(Text, nullable=True, comment="Internal notes from moderator")
    
    # Resolution and follow-up
    is_resolved = Column(Boolean, default=False, index=True, comment="Whether case is resolved")
    resolved_at = Column(DateTime, nullable=True)
    resolved_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    resolution_notes = Column(Text, nullable=True)
    
    # Audit trail
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Related moderation entries
    parent_log_id = Column(String(100), nullable=True, index=True, comment="Parent moderation log (for appeals, reversals)")
    related_log_ids = Column(JSONB, nullable=True, comment="JSON array of related moderation log IDs")
    
    # System flags
    is_reviewed = Column(Boolean, default=False, index=True, comment="Whether action has been reviewed by human")
    requires_human_review = Column(Boolean, default=False, index=True, comment="Whether case requires human review")
    is_test = Column(Boolean, default=False, comment="Whether this is a test entry")
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id], backref="moderation_logs_as_user")
    moderator = relationship("User", foreign_keys=[moderator_id], backref="moderation_logs_as_moderator")
    reporter = relationship("User", foreign_keys=[reporter_id], backref="moderation_logs_as_reporter")
    appeal_reviewer = relationship("User", foreign_keys=[appeal_reviewed_by], backref="moderation_logs_as_appeal_reviewer")
    resolver = relationship("User", foreign_keys=[resolved_by], backref="moderation_logs_as_resolver")
    
    # Related logs
    parent_log = relationship(
        "ModerationLog",
        remote_side=[log_id],
        primaryjoin="ModerationLog.parent_log_id == foreign(ModerationLog.log_id)",
        backref="child_logs"
    )
    
    # Indexes
    __table_args__ = (
        Index('ix_moderation_logs_user_action', 'user_id', 'action'),
        Index('ix_moderation_logs_content_action', 'content_type', 'content_id', 'action'),
        Index('ix_moderation_logs_severity_source', 'severity', 'source'),
        Index('ix_moderation_logs_action_taken', 'action_taken_at', 'action'),
        Index('ix_moderation_logs_user_severity', 'user_id', 'severity'),
        Index('ix_moderation_logs_appeal_status', 'appeal_status', 'created_at'),
        Index('ix_moderation_logs_is_resolved', 'is_resolved', 'updated_at'),
    )
    
    @validates('log_id')
    def validate_log_id(self, key, log_id):
        """Validate log ID is not empty."""
        if not log_id or len(log_id.strip()) == 0:
            raise ValidationError("Log ID cannot be empty")
        return log_id.strip()
    
    @validates('content_id')
    def validate_content_id(self, key, content_id):
        """Validate content ID is not empty."""
        if not content_id or len(content_id.strip()) == 0:
            raise ValidationError("Content ID cannot be empty")
        return content_id.strip()
    
    @validates('reason')
    def validate_reason(self, key, reason):
        """Validate reason is provided."""
        if not reason or len(reason.strip()) == 0:
            raise ValidationError("Reason is required")
        return reason.strip()
    
    @validates('confidence_score')
    def validate_confidence_score(self, key, score):
        """Validate confidence score is between 0 and 100."""
        if score is not None:
            if score < 0 or score > 100:
                raise ValidationError("Confidence score must be between 0 and 100")
        return score
    
    @validates('duration_days')
    def validate_duration_days(self, key, days):
        """Validate duration days is reasonable."""
        if days is not None and days <= 0:
            raise ValidationError("Duration days must be positive")
        if days and days > 3650:  # 10 years max
            raise ValidationError("Duration days cannot exceed 3650")
        return days
    
    def __repr__(self):
        return f"<ModerationLog {self.log_id} ({self.action} on {self.content_type}:{self.content_id})>"
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        Convert moderation log to dictionary representation.
        
        Args:
            include_sensitive: Whether to include sensitive information
            
        Returns:
            Dictionary representation
        """
        result = {
            "id": self.id,
            "log_id": self.log_id,
            "action": self.action.value,
            "action_taken_at": self.action_taken_at.isoformat() if self.action_taken_at else None,
            "content_type": self.content_type.value,
            "content_id": self.content_id,
            "content_url": self.content_url,
            "user_id": self.user_id,
            "moderator_id": self.moderator_id,
            "reporter_id": self.reporter_id,
            "severity": self.severity.value,
            "source": self.source.value,
            "reason": self.reason,
            "detailed_reason": self.detailed_reason,
            "confidence_score": self.confidence_score,
            "appeal_status": self.appeal_status.value if self.appeal_status else None,
            "appeal_submitted_at": self.appeal_submitted_at.isoformat() if self.appeal_submitted_at else None,
            "appeal_reviewed_at": self.appeal_reviewed_at.isoformat() if self.appeal_reviewed_at else None,
            "appeal_reviewed_by": self.appeal_reviewed_by,
            "appeal_notes": self.appeal_notes,
            "duration_days": self.duration_days,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "user_notified": self.user_notified,
            "notification_sent_at": self.notification_sent_at.isoformat() if self.notification_sent_at else None,
            "moderator_notes": self.moderator_notes if include_sensitive else None,
            "is_resolved": self.is_resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "resolution_notes": self.resolution_notes if include_sensitive else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "parent_log_id": self.parent_log_id,
            "is_reviewed": self.is_reviewed,
            "requires_human_review": self.requires_human_review,
            "is_test": self.is_test,
        }
        
        if include_sensitive:
            result.update({
                "original_content": self.original_content,
                "modified_content": self.modified_content,
                "matched_rules": self.matched_rules,
                "evidence": self.evidence,
                "context_data": self.context_data,
                "metadata": self.metadata,
                "related_log_ids": self.related_log_ids,
            })
        
        return result
    
    def to_public_dict(self) -> Dict[str, Any]:
        """
        Convert to public dictionary (safe for user viewing).
        
        Returns:
            Public-safe dictionary
        """
        result = self.to_dict(include_sensitive=False)
        
        # Remove internal fields
        internal_fields = [
            'moderator_id', 'reporter_id', 'moderator_notes', 'resolution_notes',
            'confidence_score', 'is_reviewed', 'requires_human_review', 'is_test',
            'appeal_reviewed_by', 'resolved_by'
        ]
        
        for field in internal_fields:
            if field in result:
                del result[field]
        
        return result
    
    @property
    def is_automated(self) -> bool:
        """
        Check if moderation was automated.
        
        Returns:
            True if automated
        """
        return self.source == ModerationSource.AUTOMATED
    
    @property
    def is_manual(self) -> bool:
        """
        Check if moderation was manual.
        
        Returns:
            True if manual
        """
        return self.source == ModerationSource.MANUAL
    
    @property
    def is_user_report(self) -> bool:
        """
        Check if moderation originated from user report.
        
        Returns:
            True if from user report
        """
        return self.source == ModerationSource.USER_REPORT
    
    @property
    def is_active_suspension(self) -> bool:
        """
        Check if this is an active suspension.
        
        Returns:
            True if active suspension
        """
        if self.action not in [ModerationAction.SUSPEND_USER, ModerationAction.REMOVE_CONTENT_AND_SUSPEND]:
            return False
        
        if not self.expires_at:
            return False
        
        return datetime.utcnow() < self.expires_at
    
    @property
    def is_permanent_ban(self) -> bool:
        """
        Check if this is a permanent ban.
        
        Returns:
            True if permanent ban
        """
        return self.action == ModerationAction.BAN_USER
    
    @property
    def appeal_pending(self) -> bool:
        """
        Check if appeal is pending.
        
        Returns:
            True if appeal pending
        """
        return self.appeal_status == AppealStatus.PENDING
    
    @property
    def can_be_appealed(self) -> bool:
        """
        Check if this action can be appealed.
        
        Returns:
            True if appealable
        """
        # Cannot appeal if already appealed or certain actions
        non_appealable_actions = [
            ModerationAction.NO_ACTION,
            ModerationAction.FLAG_FOR_REVIEW,
            ModerationAction.APPEAL_APPROVED,
            ModerationAction.APPEAL_REJECTED,
        ]
        
        if self.action in non_appealable_actions:
            return False
        
        if self.appeal_status is not None:
            return False
        
        # Check time limit (7 days to appeal)
        appeal_deadline = self.action_taken_at + timedelta(days=7)
        return datetime.utcnow() < appeal_deadline
    
    def add_evidence(self, evidence_type: str, evidence_data: Dict[str, Any], description: str = None):
        """
        Add evidence to the moderation log.
        
        Args:
            evidence_type: Type of evidence (screenshot, log, etc.)
            evidence_data: Evidence data
            description: Optional description
        """
        if self.evidence is None:
            self.evidence = []
        
        evidence_entry = {
            "type": evidence_type,
            "data": evidence_data,
            "added_at": datetime.utcnow().isoformat(),
            "description": description,
        }
        
        self.evidence.append(evidence_entry)
    
    def add_context(self, context_key: str, context_value: Any):
        """
        Add context data to the moderation log.
        
        Args:
            context_key: Context key
            context_value: Context value
        """
        if self.context_data is None:
            self.context_data = {}
        
        self.context_data[context_key] = context_value
    
    def submit_appeal(self, user_id: int, appeal_reason: str, evidence: List[Dict] = None):
        """
        Submit an appeal for this moderation action.
        
        Args:
            user_id: ID of user submitting appeal
            appeal_reason: Reason for appeal
            evidence: Optional list of evidence for appeal
        """
        if not self.can_be_appealed:
            raise ValidationError("This action cannot be appealed")
        
        if user_id != self.user_id:
            raise ValidationError("Only the affected user can submit an appeal")
        
        self.appeal_status = AppealStatus.PENDING
        self.appeal_submitted_at = datetime.utcnow()
        self.detailed_reason = appeal_reason
        
        # Store appeal evidence
        if evidence:
            if self.evidence is None:
                self.evidence = []
            self.evidence.extend(evidence)
    
    def review_appeal(self, moderator_id: int, approved: bool, notes: str = None):
        """
        Review an appeal.
        
        Args:
            moderator_id: ID of moderator reviewing
            approved: Whether appeal is approved
            notes: Review notes
        """
        if self.appeal_status != AppealStatus.PENDING:
            raise ValidationError("Appeal is not pending review")
        
        self.appeal_status = AppealStatus.APPROVED if approved else AppealStatus.REJECTED
        self.appeal_reviewed_at = datetime.utcnow()
        self.appeal_reviewed_by = moderator_id
        self.appeal_notes = notes
        
        if approved:
            # Reverse the moderation action
            self._reverse_action()
    
    def _reverse_action(self):
        """Reverse the moderation action (when appeal is approved)."""
        # Create a reversal log entry
        reversal_log = ModerationLog(
            log_id=f"reversal_{self.log_id}",
            action=ModerationAction.RESTORE_CONTENT,
            content_type=self.content_type,
            content_id=self.content_id,
            user_id=self.user_id,
            moderator_id=self.appeal_reviewed_by,
            reason=f"Appeal approved for log {self.log_id}",
            severity=ModerationSeverity.INFO,
            source=ModerationSource.MANUAL,
            parent_log_id=self.log_id,
            metadata={
                "original_action": self.action.value,
                "appeal_notes": self.appeal_notes,
            }
        )
        
        # Mark original as resolved
        self.is_resolved = True
        self.resolved_at = datetime.utcnow()
        self.resolved_by = self.appeal_reviewed_by
        self.resolution_notes = "Appeal approved, action reversed"
        
        return reversal_log
    
    def escalate_to_admin(self, moderator_id: int, reason: str):
        """
        Escalate this case to admin.
        
        Args:
            moderator_id: ID of moderator escalating
            reason: Reason for escalation
        """
        self.appeal_status = AppealStatus.ESCALATED
        self.requires_human_review = True
        self.moderator_notes = f"Escalated to admin by moderator {moderator_id}: {reason}"
    
    def resolve(self, resolver_id: int, notes: str = None):
        """
        Mark case as resolved.
        
        Args:
            resolver_id: ID of user resolving
            notes: Resolution notes
        """
        self.is_resolved = True
        self.resolved_at = datetime.utcnow()
        self.resolved_by = resolver_id
        self.resolution_notes = notes
    
    def notify_user(self):
        """Mark user as notified."""
        self.user_notified = True
        self.notification_sent_at = datetime.utcnow()


class ModerationRule(Base):
    """
    Model for storing moderation rules used by automated systems.
    """
    
    __tablename__ = "moderation_rules"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Rule configuration
    rule_type = Column(String(50), nullable=False, index=True, comment="Type of rule: regex, keyword, ml, etc.")
    pattern = Column(Text, nullable=True, comment="Pattern/regex/keywords for matching")
    config = Column(JSONB, nullable=True, comment="JSON configuration for the rule")
    
    # Content types this rule applies to
    content_types = Column(JSONB, nullable=False, comment="JSON array of content types")
    
    # Severity and actions
    default_severity = Column(Enum(ModerationSeverity), nullable=False, default=ModerationSeverity.MEDIUM)
    default_action = Column(Enum(ModerationAction), nullable=False, default=ModerationAction.FLAG_FOR_REVIEW)
    
    # Thresholds
    confidence_threshold = Column(Integer, default=80, comment="Confidence threshold (0-100)")
    occurrence_threshold = Column(Integer, default=1, comment="Minimum occurrences to trigger")
    
    # Rule metadata
    category = Column(String(50), nullable=True, index=True, comment="Rule category: spam, hate_speech, etc.")
    tags = Column(JSONB, nullable=True, comment="JSON array of tags")
    
    # Status
    is_active = Column(Boolean, default=True, index=True)
    is_system_rule = Column(Boolean, default=False, comment="Whether rule is system-defined (not user editable)")
    
    # Versioning
    version = Column(Integer, default=1, nullable=False)
    previous_version_id = Column(Integer, ForeignKey("moderation_rules.id"), nullable=True)
    
    # Statistics
    match_count = Column(Integer, default=0, comment="Number of times rule has matched")
    false_positive_count = Column(Integer, default=0, comment="False positive count")
    last_matched_at = Column(DateTime, nullable=True)
    
    # Audit
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    creator = relationship("User", foreign_keys=[created_by])
    updater = relationship("User", foreign_keys=[updated_by])
    previous_version = relationship("ModerationRule", remote_side=[id])
    
    def __repr__(self):
        return f"<ModerationRule {self.name} ({self.rule_type})>"
    
    def increment_match(self):
        """Increment match count."""
        self.match_count += 1
        self.last_matched_at = datetime.utcnow()
    
    def increment_false_positive(self):
        """Increment false positive count."""
        self.false_positive_count += 1
    
    @property
    def accuracy_rate(self) -> float:
        """
        Calculate rule accuracy rate.
        
        Returns:
            Accuracy rate as percentage (0-100)
        """
        if self.match_count == 0:
            return 100.0
        
        false_positives = self.false_positive_count
        true_positives = self.match_count - false_positives
        
        if self.match_count == 0:
            return 0.0
        
        return (true_positives / self.match_count) * 100


class UserModerationStats(Base):
    """
    Model for tracking user moderation statistics.
    """
    
    __tablename__ = "user_moderation_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False, index=True)
    
    # Violation counts
    total_violations = Column(Integer, default=0)
    critical_violations = Column(Integer, default=0)
    high_violations = Column(Integer, default=0)
    medium_violations = Column(Integer, default=0)
    low_violations = Column(Integer, default=0)
    
    # Action counts
    warnings_received = Column(Integer, default=0)
    suspensions_received = Column(Integer, default=0)
    bans_received = Column(Integer, default=0)
    content_removed = Column(Integer, default=0)
    
    # Appeal stats
    appeals_submitted = Column(Integer, default=0)
    appeals_approved = Column(Integer, default=0)
    appeals_rejected = Column(Integer, default=0)
    
    # Report stats
    reports_submitted = Column(Integer, default=0)
    reports_accepted = Column(Integer, default=0)
    reports_rejected = Column(Integer, default=0)
    
    # Time tracking
    first_violation_at = Column(DateTime, nullable=True)
    last_violation_at = Column(DateTime, nullable=True)
    last_warning_at = Column(DateTime, nullable=True)
    last_suspension_at = Column(DateTime, nullable=True)
    
    # Reputation
    moderation_score = Column(Integer, default=100, comment="Moderation reputation score (0-100)")
    trust_level = Column(Integer, default=1, comment="Trust level (1-5)")
    
    # Status
    is_under_review = Column(Boolean, default=False, index=True)
    review_required_until = Column(DateTime, nullable=True)
    
    # Calculated fields
    violation_rate = Column(Float, default=0.0, comment="Violations per day")
    appeal_success_rate = Column(Float, default=0.0, comment="Appeal success percentage")
    
    # Metadata
    metadata = Column(JSONB, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", backref="moderation_stats")
    
    @property
    def is_high_risk(self) -> bool:
        """
        Check if user is high risk.
        
        Returns:
            True if high risk
        """
        return (
            self.moderation_score < 30 or
            self.critical_violations >= 3 or
            self.suspensions_received >= 2 or
            self.is_under_review
        )
    
    @property
    def days_since_last_violation(self) -> Optional[int]:
        """
        Get days since last violation.
        
        Returns:
            Days since last violation or None
        """
        if not self.last_violation_at:
            return None
        
        delta = datetime.utcnow() - self.last_violation_at
        return delta.days
    
    def update_violation_rate(self, total_days: Optional[int] = None):
        """
        Update violation rate.
        
        Args:
            total_days: Total days to calculate rate over (default: account age)
        """
        if not self.first_violation_at:
            self.violation_rate = 0.0
            return
        
        if total_days is None:
            # Calculate account age in days
            async def get_account_age():
                from app.db.session import AsyncSessionLocal
                async with AsyncSessionLocal() as session:
                    from app.db.models.user import User
                    result = await session.execute(
                        select(User.created_at).where(User.id == self.user_id)
                    )
                    user = result.scalar_one_or_none()
                    if user and user.created_at:
                        delta = datetime.utcnow() - user.created_at
                        return max(delta.days, 1)
                    return 1
            
            # Note: This would need to be called in an async context
            # For now, we'll skip the async call
            total_days = 30  # Default
        
        if total_days <= 0:
            total_days = 1
        
        self.violation_rate = self.total_violations / total_days
    
    def update_appeal_success_rate(self):
        """Update appeal success rate."""
        if self.appeals_submitted == 0:
            self.appeal_success_rate = 0.0
        else:
            self.appeal_success_rate = (self.appeals_approved / self.appeals_submitted) * 100