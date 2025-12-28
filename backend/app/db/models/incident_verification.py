"""
incident_verification.py - Incident Verification Model

This module defines the IncidentVerification model for tracking verification
and validation processes for incidents. This includes:
- Multi-source verification (official reports, eyewitnesses, media, etc.)
- Verification scoring and confidence levels
- Verification workflow and status tracking
- Audit trail of verification activities
- Evidence collection and validation

Key Features:
- Multi-factor verification scoring
- Source reliability tracking
- Verification workflow with statuses
- Evidence linking and validation
- Confidence scoring and risk assessment
"""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from enum import Enum
from sqlalchemy import (
    Boolean,
    Column,
    String,
    Text,
    ForeignKey,
    Integer,
    DateTime,
    ARRAY,
    Enum as SQLEnum,
    JSON,
    Float,
    CheckConstraint,
    UniqueConstraint,
)

from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func

from db.base import Base
from models.mixins import TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from models.incident import Incident
    from models.user import User
    from models.article import Article
    from models.comment import Comment
    from models.image import Image


class VerificationStatus(Enum):
    """Overall verification status of an incident."""
    UNVERIFIED = "unverified"          # Not yet verified
    PENDING = "pending"                # Verification in progress
    PARTIALLY_VERIFIED = "partially_verified"  # Some aspects verified
    VERIFIED = "verified"              # Fully verified
    UNCONFIRMED = "unconfirmed"        # Cannot be confirmed
    DEBUNKED = "debunked"              # Proven false/misleading
    OUTDATED = "outdated"              # No longer relevant/current
    CONTESTED = "contested"            # Disputed by reliable sources
    INCONCLUSIVE = "inconclusive"      # Insufficient evidence


class VerificationMethod(Enum):
    """Methods used for verification."""
    OFFICIAL_SOURCE = "official_source"        # Government/official report
    EYEWITNESS = "eyewitness"                  # Eyewitness account
    MULTIPLE_SOURCES = "multiple_sources"      # Corroborating sources
    EXPERT_ANALYSIS = "expert_analysis"        # Subject matter expert
    MEDIA_CONFIRMATION = "media_confirmation"  # Media verification
    SOCIAL_MEDIA = "social_media"              # Social media verification
    GEOLOCATION = "geolocation"                # Location verification
    REVERSE_IMAGE_SEARCH = "reverse_image_search"  # Image verification
    DATA_ANALYSIS = "data_analysis"            # Data/statistical verification
    CROSS_REFERENCE = "cross_reference"        # Cross-reference checking
    FACT_CHECK = "fact_check"                  # Fact-checking organization
    USER_REPORT = "user_report"                # User-submitted verification
    AUTOMATED = "automated"                    # Automated verification
    OTHER = "other"                            # Other method


class SourceReliability(Enum):
    """Reliability rating for verification sources."""
    VERY_HIGH = "very_high"    # Official/government sources
    HIGH = "high"              # Reputable media, experts
    MEDIUM = "medium"          # Established organizations
    LOW = "low"                # Unverified sources
    VERY_LOW = "very_low"      # Anonymous/unreliable sources
    UNKNOWN = "unknown"        # Unknown reliability


class VerificationPriority(Enum):
    """Priority level for verification tasks."""
    CRITICAL = "critical"      # High-impact incidents
    HIGH = "high"              # Important incidents
    MEDIUM = "medium"          # Standard priority
    LOW = "low"               # Low priority/background
    ROUTINE = "routine"       # Routine verification


class IncidentVerification(Base, UUIDMixin, TimestampMixin):
    """
    Incident Verification model for tracking verification processes.
    
    This model manages the verification workflow for incidents, tracking
    sources, evidence, verification methods, and confidence scores.
    
    Attributes:
        id: Primary key UUID
        incident_id: Foreign key to Incident
        status: Overall verification status
        confidence_score: Overall confidence (0-1)
        verification_score: Calculated verification score (0-100)
        priority: Verification priority
        assigned_to_id: User assigned to verification
        verified_by_id: User who completed verification
        verified_at: When verification was completed
        verification_methods: Methods used for verification
        summary: Verification summary/conclusion
        notes: Detailed verification notes
        metadata: Additional JSON metadata
        sources_count: Number of verification sources
        evidence_count: Number of evidence items
        last_verified_at: Last verification update
        next_review_at: Scheduled next review
        is_active: Whether verification is active
        requires_review: Whether review is needed
    """
    
    __tablename__ = "incident_verifications"
    
    # Incident reference
    incident_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("incidents.id", ondelete="CASCADE"), 
        nullable=False,
        unique=True,  # One verification per incident
        index=True
    )
    
    # Verification status and scoring
    status = Column(
        SQLEnum(VerificationStatus),
        default=VerificationStatus.UNVERIFIED,
        nullable=False,
        index=True
    )
    confidence_score = Column(
        Float,
        default=0.0,
        nullable=False,
        index=True
    )
    verification_score = Column(
        Integer,
        default=0,
        nullable=False,
        index=True
    )
    priority = Column(
        SQLEnum(VerificationPriority),
        default=VerificationPriority.MEDIUM,
        nullable=False,
        index=True
    )
    
    # Assignment and completion
    assigned_to_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    verified_by_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    verified_at = Column(DateTime(timezone=True), nullable=True)
    
    # Verification details
    verification_methods = Column(
        ARRAY(SQLEnum(VerificationMethod)),
        default=[],
        nullable=False
    )
    summary = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Counters and timestamps
    sources_count = Column(Integer, default=0, nullable=False)
    evidence_count = Column(Integer, default=0, nullable=False)
    last_verified_at = Column(DateTime(timezone=True), nullable=True)
    next_review_at = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # Status flags
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    requires_review = Column(Boolean, default=False, nullable=False, index=True)
    is_contested = Column(Boolean, default=False, nullable=False, index=True)
    
    # Relationships
    incident = relationship("Incident", back_populates="verification")
    assigned_to = relationship("User", foreign_keys=[assigned_to_id])
    verified_by = relationship("User", foreign_keys=[verified_by_id])
    
    # Related verification items
    sources = relationship(
        "VerificationSource",
        back_populates="verification",
        cascade="all, delete-orphan"
    )
    evidence = relationship(
        "VerificationEvidence",
        back_populates="verification",
        cascade="all, delete-orphan"
    )
    logs = relationship(
        "VerificationLog",
        back_populates="verification",
        cascade="all, delete-orphan",
        order_by="VerificationLog.created_at.desc()"
    )
    
    # Check constraints
    __table_args__ = (
        CheckConstraint(
            'confidence_score >= 0 AND confidence_score <= 1',
            name='check_confidence_score_range'
        ),
        CheckConstraint(
            'verification_score >= 0 AND verification_score <= 100',
            name='check_verification_score_range'
        ),
        CheckConstraint(
            'sources_count >= 0',
            name='check_sources_count_non_negative'
        ),
        CheckConstraint(
            'evidence_count >= 0',
            name='check_evidence_count_non_negative'
        ),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<IncidentVerification(incident={self.incident_id}, status={self.status.value}, score={self.verification_score})>"
    
    @validates('confidence_score')
    def validate_confidence(self, key: str, score: float) -> float:
        """Validate confidence score."""
        if score < 0 or score > 1:
            raise ValueError("Confidence score must be between 0 and 1")
        return round(score, 3)
    
    @validates('verification_score')
    def validate_verification_score(self, key: str, score: int) -> int:
        """Validate verification score."""
        if score < 0 or score > 100:
            raise ValueError("Verification score must be between 0 and 100")
        return score
    
    @property
    def is_verified(self) -> bool:
        """Check if incident is verified."""
        return self.status in [
            VerificationStatus.VERIFIED,
            VerificationStatus.PARTIALLY_VERIFIED
        ]
    
    @property
    def is_debunked(self) -> bool:
        """Check if incident is debunked."""
        return self.status == VerificationStatus.DEBUNKED
    
    @property
    def needs_verification(self) -> bool:
        """Check if verification is needed."""
        return self.status in [
            VerificationStatus.UNVERIFIED,
            VerificationStatus.PENDING,
            VerificationStatus.INCONCLUSIVE
        ]
    
    @property
    def verification_level(self) -> str:
        """Get human-readable verification level."""
        if self.verification_score >= 90:
            return "highly_verified"
        elif self.verification_score >= 70:
            return "well_verified"
        elif self.verification_score >= 50:
            return "moderately_verified"
        elif self.verification_score >= 30:
            return "lightly_verified"
        else:
            return "unverified"
    
    @property
    def confidence_level(self) -> str:
        """Get human-readable confidence level."""
        if self.confidence_score >= 0.9:
            return "very_high"
        elif self.confidence_score >= 0.7:
            return "high"
        elif self.confidence_score >= 0.5:
            return "medium"
        elif self.confidence_score >= 0.3:
            return "low"
        else:
            return "very_low"
    
    @property
    def days_since_verification(self) -> Optional[int]:
        """Get days since last verification."""
        reference_date = self.verified_at or self.last_verified_at or self.created_at
        if reference_date:
            delta = datetime.utcnow() - reference_date
            return delta.days
        return None
    
    def update_status(
        self, 
        new_status: VerificationStatus,
        updated_by: Optional['User'] = None,
        notes: Optional[str] = None
    ) -> VerificationStatus:
        """
        Update verification status.
        
        Args:
            new_status: New verification status
            updated_by: User making the change
            notes: Optional notes about the change
            
        Returns:
            Previous status
        """
        old_status = self.status
        self.status = new_status
        self.last_verified_at = datetime.utcnow()
        
        # Log the status change
        self.log_status_change(old_status, new_status, updated_by, notes)
        
        # If verified, set verified_by and verified_at
        if new_status in [VerificationStatus.VERIFIED, VerificationStatus.PARTIALLY_VERIFIED]:
            if not self.verified_at:
                self.verified_at = datetime.utcnow()
            if updated_by and not self.verified_by_id:
                self.verified_by_id = updated_by.id
        
        return old_status
    
    def log_status_change(
        self, 
        old_status: VerificationStatus,
        new_status: VerificationStatus,
        changed_by: Optional['User'] = None,
        notes: Optional[str] = None
    ) -> None:
        """Log a status change."""
        from models.verification_log import VerificationLog, LogType
        
        log = VerificationLog(
            verification_id=self.id,
            log_type=LogType.STATUS_CHANGE,
            old_value=old_status.value,
            new_value=new_status.value,
            changed_by_id=changed_by.id if changed_by else None,
            notes=notes
        )
        self.logs.append(log)
    
    def assign_to(self, user: 'User', assigned_by: Optional['User'] = None) -> None:
        """Assign verification to a user."""
        self.assigned_to_id = user.id
        
        # Log assignment
        from models.verification_log import VerificationLog, LogType
        log = VerificationLog(
            verification_id=self.id,
            log_type=LogType.ASSIGNMENT,
            new_value=user.username,
            changed_by_id=assigned_by.id if assigned_by else None,
            notes=f"Assigned to {user.username}"
        )
        self.logs.append(log)
    
    def calculate_scores(self) -> Dict[str, float]:
        """
        Calculate verification and confidence scores based on sources and evidence.
        
        Returns:
            Dictionary with verification_score and confidence_score
        """
        # Base calculations (simplified - in practice would be more complex)
        verification_score = 0
        confidence_score = 0.0
        
        # Consider sources
        if self.sources:
            source_scores = []
            for source in self.sources:
                if source.is_verified:
                    reliability_scores = {
                        SourceReliability.VERY_HIGH: 1.0,
                        SourceReliability.HIGH: 0.8,
                        SourceReliability.MEDIUM: 0.6,
                        SourceReliability.LOW: 0.4,
                        SourceReliability.VERY_LOW: 0.2,
                        SourceReliability.UNKNOWN: 0.1,
                    }
                    score = reliability_scores.get(source.reliability, 0.1)
                    source_scores.append(score)
            
            if source_scores:
                # Weighted average of source scores
                avg_source_score = sum(source_scores) / len(source_scores)
                verification_score += int(avg_source_score * 50)  # 50% of total
        
        # Consider evidence
        if self.evidence:
            verified_evidence = [e for e in self.evidence if e.is_verified]
            if verified_evidence:
                evidence_ratio = len(verified_evidence) / len(self.evidence)
                verification_score += int(evidence_ratio * 30)  # 30% of total
        
        # Consider verification methods
        method_weights = {
            VerificationMethod.OFFICIAL_SOURCE: 20,
            VerificationMethod.EXPERT_ANALYSIS: 15,
            VerificationMethod.MULTIPLE_SOURCES: 15,
            VerificationMethod.MEDIA_CONFIRMATION: 10,
            VerificationMethod.FACT_CHECK: 10,
            VerificationMethod.GEOLOCATION: 8,
            VerificationMethod.REVERSE_IMAGE_SEARCH: 8,
            VerificationMethod.DATA_ANALYSIS: 7,
            VerificationMethod.CROSS_REFERENCE: 5,
            VerificationMethod.EYEWITNESS: 3,
            VerificationMethod.SOCIAL_MEDIA: 2,
            VerificationMethod.USER_REPORT: 1,
            VerificationMethod.AUTOMATED: 1,
            VerificationMethod.OTHER: 1,
        }
        
        method_score = 0
        for method in self.verification_methods:
            method_score += method_weights.get(method, 0)
        
        # Cap method contribution at 20% of total
        method_contribution = min(method_score / 100, 20)
        verification_score += method_contribution
        
        # Cap at 100
        verification_score = min(100, verification_score)
        
        # Calculate confidence score (simplified)
        confidence_score = verification_score / 100
        
        # Update scores
        self.verification_score = verification_score
        self.confidence_score = round(confidence_score, 3)
        
        return {
            "verification_score": verification_score,
            "confidence_score": confidence_score
        }
    
    def add_source(
        self,
        source_type: str,
        url: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        reliability: SourceReliability = SourceReliability.UNKNOWN,
        is_verified: bool = False,
        verified_by: Optional['User'] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'VerificationSource':
        """Add a verification source."""
        from models.verification_source import VerificationSource
        
        source = VerificationSource(
            verification_id=self.id,
            source_type=source_type,
            url=url,
            title=title,
            description=description,
            reliability=reliability,
            is_verified=is_verified,
            verified_by_id=verified_by.id if verified_by else None,
            metadata=metadata or {}
        )
        
        self.sources.append(source)
        self.sources_count += 1
        
        # Recalculate scores
        self.calculate_scores()
        
        return source
    
    def add_evidence(
        self,
        evidence_type: str,
        content: str,
        source_url: Optional[str] = None,
        is_verified: bool = False,
        verified_by: Optional['User'] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'VerificationEvidence':
        """Add verification evidence."""
        from models.verification_evidence import VerificationEvidence
        
        evidence = VerificationEvidence(
            verification_id=self.id,
            evidence_type=evidence_type,
            content=content,
            source_url=source_url,
            is_verified=is_verified,
            verified_by_id=verified_by.id if verified_by else None,
            metadata=metadata or {}
        )
        
        self.evidence.append(evidence)
        self.evidence_count += 1
        
        # Recalculate scores
        self.calculate_scores()
        
        return evidence
    
    def schedule_review(self, days_from_now: int = 30) -> None:
        """Schedule next review."""
        from datetime import timedelta
        self.next_review_at = datetime.utcnow() + timedelta(days=days_from_now)
        self.requires_review = True
    
    def mark_reviewed(self, reviewed_by: Optional['User'] = None) -> None:
        """Mark as reviewed."""
        self.requires_review = False
        self.last_verified_at = datetime.utcnow()
        
        if reviewed_by:
            self.verified_by_id = reviewed_by.id
            self.verified_at = datetime.utcnow()
    
    def to_dict(self, include_details: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": str(self.id),
            "incident_id": str(self.incident_id),
            "status": self.status.value,
            "confidence_score": self.confidence_score,
            "verification_score": self.verification_score,
            "verification_level": self.verification_level,
            "confidence_level": self.confidence_level,
            "priority": self.priority.value,
            "verification_methods": [m.value for m in self.verification_methods],
            "summary": self.summary,
            "sources_count": self.sources_count,
            "evidence_count": self.evidence_count,
            "is_verified": self.is_verified,
            "is_debunked": self.is_debunked,
            "needs_verification": self.needs_verification,
            "is_active": self.is_active,
            "requires_review": self.requires_review,
            "is_contested": self.is_contested,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "last_verified_at": self.last_verified_at.isoformat() if self.last_verified_at else None,
            "next_review_at": self.next_review_at.isoformat() if self.next_review_at else None,
            "days_since_verification": self.days_since_verification,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "assigned_to_id": str(self.assigned_to_id) if self.assigned_to_id else None,
            "verified_by_id": str(self.verified_by_id) if self.verified_by_id else None,
            "metadata": self.metadata
        }
        
        if include_details:
            if self.sources:
                result["sources"] = [source.to_dict() for source in self.sources]
            if self.evidence:
                result["evidence"] = [evidence.to_dict() for evidence in self.evidence]
            if self.logs:
                result["recent_logs"] = [log.to_dict() for log in self.logs[:10]]
        
        if self.assigned_to:
            result["assigned_to"] = {
                "id": str(self.assigned_to.id),
                "username": self.assigned_to.username
            }
        
        if self.verified_by:
            result["verified_by"] = {
                "id": str(self.verified_by.id),
                "username": self.verified_by.username
            }
        
        return result
    
    @classmethod
    def create(
        cls,
        incident_id: uuid.UUID,
        priority: VerificationPriority = VerificationPriority.MEDIUM,
        assigned_to_id: Optional[uuid.UUID] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'IncidentVerification':
        """Factory method to create a new verification record."""
        verification = cls(
            incident_id=incident_id,
            priority=priority,
            assigned_to_id=assigned_to_id,
            status=VerificationStatus.UNVERIFIED,
            confidence_score=0.0,
            verification_score=0,
            metadata=metadata or {}
        )
        
        # Schedule initial review
        verification.schedule_review(days_from_now=7)
        
        return verification


class VerificationSource(Base, UUIDMixin, TimestampMixin):
    """
    Verification Source model for tracking sources used in verification.
    """
    
    __tablename__ = "verification_sources"
    
    verification_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("incident_verifications.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    source_type = Column(String(100), nullable=False, index=True)
    url = Column(String(2000), nullable=True)
    title = Column(String(500), nullable=True)
    description = Column(Text, nullable=True)
    
    reliability = Column(
        SQLEnum(SourceReliability),
        default=SourceReliability.UNKNOWN,
        nullable=False,
        index=True
    )
    is_verified = Column(Boolean, default=False, nullable=False, index=True)
    
    verified_by_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True
    )
    verified_at = Column(DateTime(timezone=True), nullable=True)
    
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    verification = relationship("IncidentVerification", back_populates="sources")
    verified_by = relationship("User")
    
    def __repr__(self) -> str:
        return f"<VerificationSource(id={self.id}, type={self.source_type}, verified={self.is_verified})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "verification_id": str(self.verification_id),
            "source_type": self.source_type,
            "url": self.url,
            "title": self.title,
            "description": self.description,
            "reliability": self.reliability.value,
            "is_verified": self.is_verified,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "verified_by_id": str(self.verified_by_id) if self.verified_by_id else None,
            "metadata": self.metadata
        }


class VerificationEvidence(Base, UUIDMixin, TimestampMixin):
    """
    Verification Evidence model for tracking evidence used in verification.
    """
    
    __tablename__ = "verification_evidence"
    
    verification_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("incident_verifications.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    evidence_type = Column(String(100), nullable=False, index=True)
    content = Column(Text, nullable=False)
    source_url = Column(String(2000), nullable=True)
    
    is_verified = Column(Boolean, default=False, nullable=False, index=True)
    verified_by_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True
    )
    verified_at = Column(DateTime(timezone=True), nullable=True)
    
    confidence_score = Column(Float, nullable=True)
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    verification = relationship("IncidentVerification", back_populates="evidence")
    verified_by = relationship("User")
    
    def __repr__(self) -> str:
        return f"<VerificationEvidence(id={self.id}, type={self.evidence_type}, verified={self.is_verified})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "verification_id": str(self.verification_id),
            "evidence_type": self.evidence_type,
            "content": self.content,
            "source_url": self.source_url,
            "is_verified": self.is_verified,
            "confidence_score": self.confidence_score,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "verified_by_id": str(self.verified_by_id) if self.verified_by_id else None,
            "metadata": self.metadata
        }


class VerificationLog(Base, UUIDMixin, TimestampMixin):
    """
    Verification Log model for tracking verification activities.
    """
    
    __tablename__ = "verification_logs"
    
    verification_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("incident_verifications.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    log_type = Column(String(50), nullable=False, index=True)
    old_value = Column(Text, nullable=True)
    new_value = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    
    changed_by_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    verification = relationship("IncidentVerification", back_populates="logs")
    changed_by = relationship("User")
    
    def __repr__(self) -> str:
        return f"<VerificationLog(id={self.id}, type={self.log_type}, verification={self.verification_id})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "verification_id": str(self.verification_id),
            "log_type": self.log_type,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "notes": self.notes,
            "changed_by_id": str(self.changed_by_id) if self.changed_by_id else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": self.metadata
        }


# Pydantic schemas for API validation
"""
If you're using Pydantic, here are the schemas for verification models.
"""

from pydantic import BaseModel, Field, validator, HttpUrl
from typing import Optional, List, Dict, Any
from datetime import datetime


class VerificationBase(BaseModel):
    """Base schema for verification operations."""
    priority: VerificationPriority = Field(default=VerificationPriority.MEDIUM)
    summary: Optional[str] = None
    notes: Optional[str] = None


class VerificationCreate(VerificationBase):
    """Schema for creating verification records."""
    incident_id: str
    assigned_to_id: Optional[str] = None
    
    @validator('incident_id', 'assigned_to_id')
    def validate_uuids(cls, v):
        if v is not None:
            try:
                uuid.UUID(v)
            except ValueError:
                raise ValueError('Invalid UUID format')
        return v


class VerificationUpdate(BaseModel):
    """Schema for updating verification records."""
    status: Optional[VerificationStatus] = None
    priority: Optional[VerificationPriority] = None
    assigned_to_id: Optional[str] = None
    summary: Optional[str] = None
    notes: Optional[str] = None
    verification_methods: Optional[List[VerificationMethod]] = None


class VerificationStatusUpdate(BaseModel):
    """Schema for updating verification status."""
    status: VerificationStatus
    notes: Optional[str] = None


class VerificationInDBBase(VerificationBase):
    """Base schema for verification in database."""
    id: str
    incident_id: str
    status: VerificationStatus
    confidence_score: float
    verification_score: int
    verification_level: str
    confidence_level: str
    verification_methods: List[str]
    sources_count: int
    evidence_count: int
    is_verified: bool
    is_debunked: bool
    needs_verification: bool
    is_active: bool
    requires_review: bool
    verified_at: Optional[datetime]
    last_verified_at: Optional[datetime]
    next_review_at: Optional[datetime]
    assigned_to_id: Optional[str]
    verified_by_id: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class Verification(VerificationInDBBase):
    """Schema for verification API responses."""
    assigned_to: Optional[Dict[str, Any]] = None
    verified_by: Optional[Dict[str, Any]] = None
    sources: Optional[List[Dict[str, Any]]] = None
    evidence: Optional[List[Dict[str, Any]]] = None
    
    class Config:
        from_attributes = True


class VerificationSourceSchema(BaseModel):
    """Schema for verification sources."""
    id: str
    source_type: str
    url: Optional[str]
    title: Optional[str]
    description: Optional[str]
    reliability: str
    is_verified: bool
    verified_at: Optional[datetime]
    created_at: datetime
    
    class Config:
        from_attributes = True


class VerificationEvidenceSchema(BaseModel):
    """Schema for verification evidence."""
    id: str
    evidence_type: str
    content: str
    source_url: Optional[str]
    is_verified: bool
    confidence_score: Optional[float]
    verified_at: Optional[datetime]
    created_at: datetime
    
    class Config:
        from_attributes = True


class VerificationStats(BaseModel):
    """Schema for verification statistics."""
    total_verifications: int
    by_status: Dict[str, int]
    by_priority: Dict[str, int]
    average_verification_score: float
    average_confidence_score: float
    verification_rate: float  # Percentage verified
    pending_verifications: int
    overdue_reviews: int
    
    class Config:
        from_attributes = True


class VerificationSearchRequest(BaseModel):
    """Schema for verification search requests."""
    status: Optional[VerificationStatus] = None
    priority: Optional[VerificationPriority] = None
    min_verification_score: Optional[int] = Field(None, ge=0, le=100)
    max_verification_score: Optional[int] = Field(None, ge=0, le=100)
    min_confidence_score: Optional[float] = Field(None, ge=0, le=1)
    max_confidence_score: Optional[float] = Field(None, ge=0, le=1)
    is_active: Optional[bool] = None
    requires_review: Optional[bool] = None
    assigned_to_id: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    sort_by: str = Field(default="verification_score", pattern="^(verification_score|confidence_score|created_at|updated_at)$")
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$")
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)
    
    class Config:
        from_attributes = True