"""
incident.py - Incident Model

This module defines the Incident model, which represents events, accidents,
emergencies, or other notable occurrences that need tracking, verification,
and management within the platform.

Key Features:
- Comprehensive incident classification and categorization
- Multi-level severity and priority assessment
- Geographic and temporal tracking
- Stakeholder and organization involvement
- Verification and evidence management
- Workflow and status tracking
- Impact assessment and reporting
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from enum import Enum
from sqlalchemy import (
    Column, String, Text, ForeignKey, Integer, DateTime, 
    Boolean, Enum as SQLEnum, JSON, Float, CheckConstraint,
    Table, UniqueConstraint, Index, BigInteger
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.sql import func
from geoalchemy2 import Geography
from geoalchemy2.shape import to_shape

from db.base import Base
from models.mixins import TimestampMixin, SoftDeleteMixin, UUIDMixin, StatusMixin

if TYPE_CHECKING:
    from models.user import User
    from models.article import Article
    from models.comment import Comment
    from models.dataset import Dataset
    from models.feedback import Feedback
    from models.incident_image import IncidentImage
    from models.incident_verification import IncidentVerification
    from models.embedding import Embedding


class IncidentType(Enum):
    """Types of incidents."""
    ACCIDENT = "accident"                      # Transportation, industrial
    NATURAL_DISASTER = "natural_disaster"      # Earthquake, flood, hurricane
    PUBLIC_HEALTH = "public_health"            # Disease outbreak, contamination
    SECURITY = "security"                      # Crime, terrorism, violence
    ENVIRONMENTAL = "environmental"            # Pollution, spill, deforestation
    INFRASTRUCTURE = "infrastructure"          # Power outage, bridge collapse
    TECHNOLOGICAL = "technological"            # Cyber attack, system failure
    SOCIAL = "social"                          # Protest, riot, unrest
    POLITICAL = "political"                    # Election dispute, coup
    HUMANITARIAN = "humanitarian"              # Refugee crisis, famine
    BUSINESS = "business"                      # Corporate scandal, bankruptcy
    OTHER = "other"                            # Other incident types


class IncidentSeverity(Enum):
    """Severity levels for incidents."""
    MINOR = "minor"            # Minimal impact, localized
    MODERATE = "moderate"      # Noticeable impact, some disruption
    MAJOR = "major"            # Significant impact, widespread disruption
    SEVERE = "severe"          # Critical impact, emergency response needed
    CATASTROPHIC = "catastrophic"  # Extreme impact, national/international concern


class IncidentStatus(Enum):
    """Workflow status for incidents."""
    REPORTED = "reported"          # Initial report received
    CONFIRMED = "confirmed"        # Incident confirmed
    INVESTIGATING = "investigating"  # Under investigation
    CONTAINED = "contained"        # Situation contained
    RESOLVED = "resolved"          # Incident resolved
    CLOSED = "closed"              # Case closed
    ARCHIVED = "archived"          #Archived for reference
    CANCELLED = "cancelled"        # False report or duplicate
    ESCALATED = "escalated"        # Escalated to higher authority


class IncidentPriority(Enum):
    """Priority levels for incident handling."""
    LOW = "low"           # Routine handling
    MEDIUM = "medium"     # Standard priority
    HIGH = "high"         # Important, needs attention
    CRITICAL = "critical" # Urgent, immediate action required
    EMERGENCY = "emergency"  # Life-threatening, highest priority


class IncidentConfidence(Enum):
    """Confidence levels in incident reporting."""
    LOW = "low"           # Unverified, single source
    MEDIUM = "medium"     # Some verification, few sources
    HIGH = "high"         # Well-verified, multiple sources
    CONFIRMED = "confirmed"  # Officially confirmed


class Incident(Base, UUIDMixin, TimestampMixin, SoftDeleteMixin, StatusMixin):
    """
    Incident model representing events and occurrences.
    
    This is the central model for tracking incidents across the platform,
    integrating with verification, evidence, media, and reporting systems.
    
    Attributes:
        id: Primary key UUID
        title: Incident title/headline
        description: Detailed description
        summary: Executive summary
        incident_type: Type of incident
        severity: Severity level
        status: Workflow status
        priority: Handling priority
        confidence: Confidence level in reporting
        location: Location description
        address: Specific address
        city: City
        state: State/province
        country: Country
        postal_code: Postal/ZIP code
        latitude: GPS latitude
        longitude: GPS longitude
        geojson: GeoJSON geometry
        geometry: PostGIS geometry (for spatial queries)
        occurred_at: When incident occurred
        discovered_at: When incident was discovered
        reported_at: When incident was reported
        started_at: When incident started (for ongoing)
        ended_at: When incident ended
        duration_minutes: Duration in minutes
        affected_area_km2: Affected area in square kilometers
        estimated_impacted: Estimated number of people impacted
        confirmed_impacted: Confirmed number impacted
        estimated_damage: Estimated damage in USD
        confirmed_damage: Confirmed damage in USD
        source_count: Number of sources reporting
        media_count: Number of media items
        evidence_count: Number of evidence items
        verification_score: Verification score (0-100)
        risk_level: Risk assessment level (1-10)
        tags: Categorization tags
        categories: Incident categories
        metadata: Additional JSON metadata
        created_by_id: User who created the incident
        assigned_to_id: User assigned to handle
        organization_id: Responsible organization
        is_public: Whether incident is publicly visible
        is_sensitive: Whether incident contains sensitive info
        requires_verification: Whether verification is needed
        is_verified: Whether incident is verified
        verification_status: Verification status summary
        last_verified_at: Last verification timestamp
        next_review_at: Scheduled next review
    """
    
    __tablename__ = "incidents"
    
    __statuses__ = [s.value for s in IncidentStatus]
    
    # Basic information
    title = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    
    # Classification
    incident_type = Column(
        SQLEnum(IncidentType),
        nullable=False,
        index=True
    )
    severity = Column(
        SQLEnum(IncidentSeverity),
        default=IncidentSeverity.MODERATE,
        nullable=False,
        index=True
    )
    status = Column(
        SQLEnum(IncidentStatus),
        default=IncidentStatus.REPORTED,
        nullable=False,
        index=True
    )
    priority = Column(
        SQLEnum(IncidentPriority),
        default=IncidentPriority.MEDIUM,
        nullable=False,
        index=True
    )
    confidence = Column(
        SQLEnum(IncidentConfidence),
        default=IncidentConfidence.LOW,
        nullable=False,
        index=True
    )
    
    # Location information
    location = Column(String(500), nullable=True, index=True)
    address = Column(String(500), nullable=True)
    city = Column(String(100), nullable=True, index=True)
    state = Column(String(100), nullable=True, index=True)
    country = Column(String(2), nullable=True, index=True)  # ISO 3166-1 alpha-2
    postal_code = Column(String(20), nullable=True, index=True)
    
    # Geographic coordinates
    latitude = Column(Float, nullable=True, index=True)
    longitude = Column(Float, nullable=True, index=True)
    geojson = Column(JSONB, nullable=True)
    # Uncomment if using PostGIS:
    # geometry = Column(Geography(geometry_type='POINT', srid=4326), nullable=True)
    
    # Temporal information
    occurred_at = Column(DateTime(timezone=True), nullable=True, index=True)
    discovered_at = Column(DateTime(timezone=True), nullable=True)
    reported_at = Column(DateTime(timezone=True), nullable=False, index=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    duration_minutes = Column(Integer, nullable=True)
    
    # Impact assessment
    affected_area_km2 = Column(Float, nullable=True)
    estimated_impacted = Column(Integer, nullable=True)
    confirmed_impacted = Column(Integer, nullable=True)
    estimated_damage = Column(BigInteger, nullable=True)  # In USD
    confirmed_damage = Column(BigInteger, nullable=True)  # In USD
    
    # Counters and scores
    source_count = Column(Integer, default=0, nullable=False)
    media_count = Column(Integer, default=0, nullable=False)
    evidence_count = Column(Integer, default=0, nullable=False)
    verification_score = Column(Integer, default=0, nullable=False, index=True)
    risk_level = Column(Integer, nullable=True, index=True)  # 1-10
    
    # Categorization
    tags = Column(ARRAY(String), default=[], nullable=False, index=True)
    categories = Column(ARRAY(String), default=[], nullable=False, index=True)
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships and ownership
    created_by_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    assigned_to_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    organization_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("organizations.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    
    # Visibility and verification flags
    is_public = Column(Boolean, default=False, nullable=False, index=True)
    is_sensitive = Column(Boolean, default=False, nullable=False, index=True)
    requires_verification = Column(Boolean, default=True, nullable=False, index=True)
    is_verified = Column(Boolean, default=False, nullable=False, index=True)
    verification_status = Column(String(50), nullable=True, index=True)
    
    # Verification timestamps
    last_verified_at = Column(DateTime(timezone=True), nullable=True)
    next_review_at = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # Relationships
    created_by = relationship("User", foreign_keys=[created_by_id])
    assigned_to = relationship("User", foreign_keys=[assigned_to_id])
    organization = relationship("Organization", back_populates="incidents")
    
    # Related models
    articles = relationship(
        "Article",
        back_populates="incident",
        cascade="all, delete-orphan"
    )
    
    comments = relationship(
        "Comment",
        back_populates="incident",
        cascade="all, delete-orphan"
    )
    
    datasets = relationship(
        "Dataset",
        secondary="dataset_incidents",
        back_populates="incidents"
    )
    
    feedback = relationship(
        "Feedback",
        back_populates="incident",
        cascade="all, delete-orphan"
    )
    
    incident_images = relationship(
        "IncidentImage",
        back_populates="incident",
        cascade="all, delete-orphan"
    )
    
    verification = relationship(
        "IncidentVerification",
        back_populates="incident",
        uselist=False,
        cascade="all, delete-orphan"
    )
    
    embeddings = relationship(
        "Embedding",
        back_populates="incident",
        cascade="all, delete-orphan"
    )
    
    # Check constraints
    __table_args__ = (
        CheckConstraint(
            'latitude IS NULL OR (latitude >= -90 AND latitude <= 90)',
            name='check_latitude_range'
        ),
        CheckConstraint(
            'longitude IS NULL OR (longitude >= -180 AND longitude <= 180)',
            name='check_longitude_range'
        ),
        CheckConstraint(
            'verification_score >= 0 AND verification_score <= 100',
            name='check_verification_score_range'
        ),
        CheckConstraint(
            'risk_level IS NULL OR (risk_level >= 1 AND risk_level <= 10)',
            name='check_risk_level_range'
        ),
        CheckConstraint(
            'estimated_impacted IS NULL OR estimated_impacted >= 0',
            name='check_estimated_impacted_non_negative'
        ),
        CheckConstraint(
            'confirmed_impacted IS NULL OR confirmed_impacted >= 0',
            name='check_confirmed_impacted_non_negative'
        ),
        CheckConstraint(
            'estimated_damage IS NULL OR estimated_damage >= 0',
            name='check_estimated_damage_non_negative'
        ),
        CheckConstraint(
            'confirmed_damage IS NULL OR confirmed_damage >= 0',
            name='check_confirmed_damage_non_negative'
        ),
        Index('ix_incidents_country_severity', 'country', 'severity'),
        Index('ix_incidents_type_status', 'incident_type', 'status'),
        Index('ix_incidents_created_status', 'created_at', 'status'),
        Index('ix_incidents_location_search', 'city', 'state', 'country'),
        UniqueConstraint('title', 'occurred_at', 'latitude', 'longitude', 
                        name='uq_incident_location_time', deferrable=True),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Incident(id={self.id}, title='{self.title}', type={self.incident_type.value})>"
    
    def __init__(self, **kwargs):
        """Initialize incident with default values."""
        # Set reported_at to now if not provided
        if 'reported_at' not in kwargs:
            kwargs['reported_at'] = datetime.utcnow()
        
        super().__init__(**kwargs)
        
        # Set initial verification status
        if self.verification_status is None:
            self.verification_status = "unverified"
            self.requires_verification = True
    
    @validates('title')
    def validate_title(self, key: str, title: str) -> str:
        """Validate incident title."""
        title = title.strip()
        if not title:
            raise ValueError("Incident title cannot be empty")
        if len(title) > 255:
            raise ValueError("Incident title cannot exceed 255 characters")
        return title
    
    @validates('description')
    def validate_description(self, key: str, description: str) -> str:
        """Validate incident description."""
        description = description.strip()
        if not description:
            raise ValueError("Incident description cannot be empty")
        if len(description) > 10000:
            raise ValueError("Incident description is too long")
        return description
    
    @validates('country')
    def validate_country(self, key: str, country: Optional[str]) -> Optional[str]:
        """Validate country code."""
        if country and len(country) != 2:
            raise ValueError("Country must be 2-letter ISO code")
        return country.upper() if country else None
    
    @validates('latitude', 'longitude')
    def validate_coordinates(self, key: str, value: Optional[float]) -> Optional[float]:
        """Validate geographic coordinates."""
        if value is not None:
            if key == 'latitude' and (value < -90 or value > 90):
                raise ValueError("Latitude must be between -90 and 90")
            if key == 'longitude' and (value < -180 or value > 180):
                raise ValueError("Longitude must be between -180 and 180")
        return value
    
    @property
    def full_location(self) -> str:
        """Get full location string."""
        parts = []
        if self.address:
            parts.append(self.address)
        if self.city:
            parts.append(self.city)
        if self.state:
            parts.append(self.state)
        if self.country:
            parts.append(self.country)
        return ", ".join(parts) if parts else self.location or "Unknown location"
    
    @property
    def coordinates(self) -> Optional[Dict[str, float]]:
        """Get coordinates as dictionary."""
        if self.latitude is not None and self.longitude is not None:
            return {
                "latitude": self.latitude,
                "longitude": self.longitude
            }
        return None
    
    @property
    def has_location(self) -> bool:
        """Check if incident has location data."""
        return self.latitude is not None and self.longitude is not None
    
    @property
    def is_ongoing(self) -> bool:
        """Check if incident is ongoing."""
        if self.ended_at:
            return False
        if self.started_at and not self.ended_at:
            return True
        # If no start/end times, check status
        return self.status not in [
            IncidentStatus.RESOLVED,
            IncidentStatus.CLOSED,
            IncidentStatus.ARCHIVED,
            IncidentStatus.CANCELLED
        ]
    
    @property
    def age_hours(self) -> float:
        """Get age of incident in hours."""
        reference_date = self.occurred_at or self.reported_at or self.created_at
        if reference_date:
            delta = datetime.utcnow() - reference_date
            return delta.total_seconds() / 3600
        return 0.0
    
    @property
    def duration_hours(self) -> Optional[float]:
        """Get duration in hours."""
        if self.duration_minutes:
            return self.duration_minutes / 60
        if self.started_at and self.ended_at:
            delta = self.ended_at - self.started_at
            return delta.total_seconds() / 3600
        return None
    
    @property
    def impact_score(self) -> float:
        """Calculate impact score (0-100)."""
        score = 0.0
        
        # Severity contributes 40%
        severity_scores = {
            IncidentSeverity.MINOR: 10,
            IncidentSeverity.MODERATE: 30,
            IncidentSeverity.MAJOR: 60,
            IncidentSeverity.SEVERE: 80,
            IncidentSeverity.CATASTROPHIC: 100,
        }
        score += severity_scores.get(self.severity, 0) * 0.4
        
        # Impacted people contribute 30%
        if self.confirmed_impacted:
            if self.confirmed_impacted >= 10000:
                score += 30
            elif self.confirmed_impacted >= 1000:
                score += 20
            elif self.confirmed_impacted >= 100:
                score += 15
            elif self.confirmed_impacted >= 10:
                score += 10
            else:
                score += 5
        elif self.estimated_impacted:
            if self.estimated_impacted >= 10000:
                score += 20
            elif self.estimated_impacted >= 1000:
                score += 15
            elif self.estimated_impacted >= 100:
                score += 10
            elif self.estimated_impacted >= 10:
                score += 5
        
        # Damage contributes 20%
        if self.confirmed_damage:
            if self.confirmed_damage >= 1000000000:  # 1B+
                score += 20
            elif self.confirmed_damage >= 100000000:  # 100M+
                score += 15
            elif self.confirmed_damage >= 10000000:   # 10M+
                score += 10
            elif self.confirmed_damage >= 1000000:    # 1M+
                score += 5
        elif self.estimated_damage:
            if self.estimated_damage >= 1000000000:
                score += 15
            elif self.estimated_damage >= 100000000:
                score += 10
            elif self.estimated_damage >= 10000000:
                score += 7
            elif self.estimated_damage >= 1000000:
                score += 3
        
        # Area affected contributes 10%
        if self.affected_area_km2:
            if self.affected_area_km2 >= 1000:
                score += 10
            elif self.affected_area_km2 >= 100:
                score += 7
            elif self.affected_area_km2 >= 10:
                score += 5
            elif self.affected_area_km2 >= 1:
                score += 3
        
        return min(100.0, score)
    
    @property
    def urgency_score(self) -> float:
        """Calculate urgency score (0-100)."""
        score = 0.0
        
        # Priority contributes 40%
        priority_scores = {
            IncidentPriority.LOW: 10,
            IncidentPriority.MEDIUM: 30,
            IncidentPriority.HIGH: 60,
            IncidentPriority.CRITICAL: 80,
            IncidentPriority.EMERGENCY: 100,
        }
        score += priority_scores.get(self.priority, 0) * 0.4
        
        # Age contributes 30% (newer incidents are more urgent)
        age_hours = self.age_hours
        if age_hours <= 1:
            score += 30  # Within last hour
        elif age_hours <= 6:
            score += 25  # Within last 6 hours
        elif age_hours <= 24:
            score += 20  # Within last day
        elif age_hours <= 72:
            score += 10  # Within last 3 days
        elif age_hours <= 168:
            score += 5   # Within last week
        
        # Ongoing status contributes 20%
        if self.is_ongoing:
            score += 20
        
        # Risk level contributes 10%
        if self.risk_level:
            score += self.risk_level * 2  # Max 20, scaled to 10
        
        return min(100.0, score)
    
    @property
    def primary_image(self) -> Optional['IncidentImage']:
        """Get primary incident image."""
        if not self.incident_images:
            return None
        
        for img in self.incident_images:
            if img.is_primary and not img.is_deleted:
                return img
        
        # Return first non-deleted image
        for img in self.incident_images:
            if not img.is_deleted:
                return img
        
        return None
    
    def update_duration(self) -> None:
        """Update duration based on start and end times."""
        if self.started_at and self.ended_at:
            delta = self.ended_at - self.started_at
            self.duration_minutes = int(delta.total_seconds() / 60)
    
    def update_counters(self) -> None:
        """Update all counter fields."""
        # Update source count from verification
        if self.verification:
            self.source_count = self.verification.sources_count
        
        # Update media count
        self.media_count = len([img for img in self.incident_images if not img.is_deleted])
        
        # Update evidence count from verification
        if self.verification:
            self.evidence_count = self.verification.evidence_count
        
        # Update verification status
        if self.verification:
            self.verification_score = self.verification.verification_score
            self.is_verified = self.verification.is_verified
            self.verification_status = self.verification.status.value
            self.last_verified_at = self.verification.last_verified_at
    
    def add_tag(self, tag: str) -> bool:
        """Add a tag to incident."""
        tag = tag.strip().lower()
        if tag and tag not in self.tags:
            self.tags.append(tag)
            return True
        return False
    
    def remove_tag(self, tag: str) -> bool:
        """Remove a tag from incident."""
        tag = tag.strip().lower()
        if tag in self.tags:
            self.tags.remove(tag)
            return True
        return False
    
    def set_location(
        self,
        latitude: float,
        longitude: float,
        address: Optional[str] = None,
        city: Optional[str] = None,
        state: Optional[str] = None,
        country: Optional[str] = None,
        postal_code: Optional[str] = None
    ) -> None:
        """Set incident location with validation."""
        if latitude < -90 or latitude > 90:
            raise ValueError("Latitude must be between -90 and 90")
        if longitude < -180 or longitude > 180:
            raise ValueError("Longitude must be between -180 and 180")
        
        self.latitude = latitude
        self.longitude = longitude
        self.address = address
        self.city = city
        self.state = state
        self.country = country.upper() if country else None
        self.postal_code = postal_code
        
        # Update geojson
        self.geojson = {
            "type": "Point",
            "coordinates": [longitude, latitude]
        }
    
    def schedule_review(self, days_from_now: int = 7) -> None:
        """Schedule next review."""
        self.next_review_at = datetime.utcnow() + timedelta(days=days_from_now)
    
    def mark_as_reviewed(self, reviewed_by: Optional['User'] = None) -> None:
        """Mark incident as reviewed."""
        self.next_review_at = None
        self.last_verified_at = datetime.utcnow()
        
        if reviewed_by:
            self.assigned_to_id = reviewed_by.id
    
    def escalate(self, new_priority: IncidentPriority, reason: Optional[str] = None) -> None:
        """Escalate incident priority."""
        old_priority = self.priority
        self.priority = new_priority
        
        # Log escalation in metadata
        if 'escalation_history' not in self.metadata:
            self.metadata['escalation_history'] = []
        
        self.metadata['escalation_history'].append({
            'from_priority': old_priority.value,
            'to_priority': new_priority.value,
            'escalated_at': datetime.utcnow().isoformat(),
            'reason': reason
        })
    
    def create_verification_record(self, assigned_to: Optional['User'] = None) -> 'IncidentVerification':
        """Create a verification record for this incident."""
        from models.incident_verification import IncidentVerification, VerificationPriority
        
        # Map incident priority to verification priority
        priority_map = {
            IncidentPriority.LOW: VerificationPriority.LOW,
            IncidentPriority.MEDIUM: VerificationPriority.MEDIUM,
            IncidentPriority.HIGH: VerificationPriority.HIGH,
            IncidentPriority.CRITICAL: VerificationPriority.CRITICAL,
            IncidentPriority.EMERGENCY: VerificationPriority.CRITICAL,
        }
        
        verification = IncidentVerification.create(
            incident_id=self.id,
            priority=priority_map.get(self.priority, VerificationPriority.MEDIUM),
            assigned_to_id=assigned_to.id if assigned_to else None
        )
        
        self.verification = verification
        self.requires_verification = True
        
        return verification
    
    def to_dict(self, include_related: bool = False, include_metadata: bool = False) -> Dict[str, Any]:
        """
        Convert incident to dictionary for API responses.
        
        Args:
            include_related: Whether to include related entities
            include_metadata: Whether to include metadata
            
        Returns:
            Dictionary representation
        """
        result = {
            "id": str(self.id),
            "title": self.title,
            "description": self.description,
            "summary": self.summary,
            "incident_type": self.incident_type.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "priority": self.priority.value,
            "confidence": self.confidence.value,
            "location": self.location,
            "address": self.address,
            "city": self.city,
            "state": self.state,
            "country": self.country,
            "postal_code": self.postal_code,
            "full_location": self.full_location,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "coordinates": self.coordinates,
            "has_location": self.has_location,
            "occurred_at": self.occurred_at.isoformat() if self.occurred_at else None,
            "discovered_at": self.discovered_at.isoformat() if self.discovered_at else None,
            "reported_at": self.reported_at.isoformat() if self.reported_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_minutes": self.duration_minutes,
            "duration_hours": round(self.duration_hours, 2) if self.duration_hours else None,
            "affected_area_km2": self.affected_area_km2,
            "estimated_impacted": self.estimated_impacted,
            "confirmed_impacted": self.confirmed_impacted,
            "estimated_damage": self.estimated_damage,
            "confirmed_damage": self.confirmed_damage,
            "source_count": self.source_count,
            "media_count": self.media_count,
            "evidence_count": self.evidence_count,
            "verification_score": self.verification_score,
            "risk_level": self.risk_level,
            "impact_score": round(self.impact_score, 2),
            "urgency_score": round(self.urgency_score, 2),
            "is_ongoing": self.is_ongoing,
            "age_hours": round(self.age_hours, 2),
            "tags": self.tags,
            "categories": self.categories,
            "is_public": self.is_public,
            "is_sensitive": self.is_sensitive,
            "requires_verification": self.requires_verification,
            "is_verified": self.is_verified,
            "verification_status": self.verification_status,
            "last_verified_at": self.last_verified_at.isoformat() if self.last_verified_at else None,
            "next_review_at": self.next_review_at.isoformat() if self.next_review_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "created_by_id": str(self.created_by_id) if self.created_by_id else None,
            "assigned_to_id": str(self.assigned_to_id) if self.assigned_to_id else None,
            "organization_id": str(self.organization_id) if self.organization_id else None,
            "is_deleted": self.is_deleted,
            "deleted_at": self.deleted_at.isoformat() if self.deleted_at else None
        }
        
        if include_metadata:
            result["metadata"] = self.metadata
        
        if include_related:
            if self.created_by:
                result["created_by"] = {
                    "id": str(self.created_by.id),
                    "username": self.created_by.username,
                    "email": getattr(self.created_by, 'email', None)
                }
            
            if self.assigned_to:
                result["assigned_to"] = {
                    "id": str(self.assigned_to.id),
                    "username": self.assigned_to.username
                }
            
            if self.organization:
                result["organization"] = {
                    "id": str(self.organization.id),
                    "name": self.organization.name,
                    "type": getattr(self.organization, 'type', None)
                }
            
            if self.primary_image:
                result["primary_image"] = self.primary_image.to_dict(include_image=True)
            
            if self.verification:
                result["verification"] = self.verification.to_dict(include_details=False)
            
            # Count related entities
            result["article_count"] = len(self.articles) if self.articles else 0
            result["comment_count"] = len(self.comments) if self.comments else 0
            result["feedback_count"] = len(self.feedback) if self.feedback else 0
            result["image_count"] = len(self.incident_images) if self.incident_images else 0
        
        return result
    
    @classmethod
    def create(
        cls,
        title: str,
        description: str,
        incident_type: IncidentType,
        created_by_id: Optional[uuid.UUID] = None,
        location: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        occurred_at: Optional[datetime] = None,
        severity: IncidentSeverity = IncidentSeverity.MODERATE,
        priority: IncidentPriority = IncidentPriority.MEDIUM,
        confidence: IncidentConfidence = IncidentConfidence.LOW,
        tags: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        is_public: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'Incident':
        """
        Factory method to create a new incident.
        
        Args:
            title: Incident title
            description: Incident description
            incident_type: Type of incident
            created_by_id: User creating the incident
            location: Location description
            latitude: GPS latitude
            longitude: GPS longitude
            occurred_at: When incident occurred
            severity: Severity level
            priority: Priority level
            confidence: Confidence level
            tags: Categorization tags
            categories: Incident categories
            is_public: Whether incident is public
            metadata: Additional metadata
            
        Returns:
            A new Incident instance
        """
        # Validate coordinates
        if latitude is not None:
            if latitude < -90 or latitude > 90:
                raise ValueError("Latitude must be between -90 and 90")
        if longitude is not None:
            if longitude < -180 or longitude > 180:
                raise ValueError("Longitude must be between -180 and 180")
        
        # Create geojson if coordinates provided
        geojson = None
        if latitude is not None and longitude is not None:
            geojson = {
                "type": "Point",
                "coordinates": [longitude, latitude]
            }
        
        incident = cls(
            title=title.strip(),
            description=description.strip(),
            incident_type=incident_type,
            created_by_id=created_by_id,
            location=location,
            latitude=latitude,
            longitude=longitude,
            geojson=geojson,
            occurred_at=occurred_at,
            reported_at=datetime.utcnow(),
            severity=severity,
            priority=priority,
            confidence=confidence,
            tags=tags or [],
            categories=categories or [],
            is_public=is_public,
            metadata=metadata or {},
            status=IncidentStatus.REPORTED
        )
        
        # Schedule initial review
        incident.schedule_review(days_from_now=3)
        
        return incident
    
    @classmethod
    def create_from_article(
        cls,
        article: 'Article',
        incident_type: IncidentType,
        created_by_id: Optional[uuid.UUID] = None,
        **kwargs
    ) -> 'Incident':
        """
        Create an incident from an existing article.
        
        Args:
            article: Article instance
            incident_type: Type of incident
            created_by_id: User creating incident
            **kwargs: Additional arguments
            
        Returns:
            Incident instance
        """
        # Extract location from article metadata if available
        location = article.location or article.metadata.get('location')
        latitude = article.latitude or article.metadata.get('latitude')
        longitude = article.longitude or article.metadata.get('longitude')
        
        incident = cls.create(
            title=article.title,
            description=article.content[:1000] + "..." if len(article.content) > 1000 else article.content,
            incident_type=incident_type,
            created_by_id=created_by_id,
            location=location,
            latitude=latitude,
            longitude=longitude,
            occurred_at=article.published_at,
            tags=["article_source"] + (article.tags if hasattr(article, 'tags') else []),
            categories=["media_reported"],
            **kwargs
        )
        
        # Link article to incident
        article.incident_id = incident.id
        
        # Add article metadata
        incident.metadata.update({
            "source_article": {
                "id": str(article.id),
                "url": article.url,
                "source": article.source,
                "author": article.author,
                "published_at": article.published_at.isoformat() if article.published_at else None
            }
        })
        
        return incident


# Association table for incidents and datasets
dataset_incidents = Table(
    'dataset_incidents',
    Base.metadata,
    Column('dataset_id', UUID(as_uuid=True), ForeignKey('datasets.id', ondelete='CASCADE'), primary_key=True),
    Column('incident_id', UUID(as_uuid=True), ForeignKey('incidents.id', ondelete='CASCADE'), primary_key=True),
    Column('added_at', DateTime(timezone=True), server_default=func.now()),
    Column('metadata', JSONB, default=dict)
)


# Pydantic schemas for API validation
"""
If you're using Pydantic, here are the schemas for the Incident model.
"""

from pydantic import BaseModel, Field, validator, HttpUrl
from typing import Optional, List, Dict, Any
from datetime import datetime


class IncidentBase(BaseModel):
    """Base schema for incident operations."""
    title: str = Field(..., max_length=255)
    description: str = Field(..., min_length=10, max_length=10000)
    incident_type: IncidentType
    location: Optional[str] = Field(None, max_length=500)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    occurred_at: Optional[datetime] = None
    severity: IncidentSeverity = Field(default=IncidentSeverity.MODERATE)
    priority: IncidentPriority = Field(default=IncidentPriority.MEDIUM)
    confidence: IncidentConfidence = Field(default=IncidentConfidence.LOW)
    tags: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    is_public: bool = Field(default=False)
    
    @validator('title')
    def validate_title(cls, v):
        if not v or not v.strip():
            raise ValueError('Incident title cannot be empty')
        return v.strip()
    
    @validator('description')
    def validate_description(cls, v):
        if not v or not v.strip():
            raise ValueError('Incident description cannot be empty')
        if len(v.strip()) < 10:
            raise ValueError('Incident description must be at least 10 characters')
        return v.strip()
    
    @validator('latitude', 'longitude')
    def validate_coordinates(cls, v, values, **kwargs):
        if v is not None:
            field = kwargs.get('field')
            if field.name == 'latitude' and (v < -90 or v > 90):
                raise ValueError('Latitude must be between -90 and 90')
            if field.name == 'longitude' and (v < -180 or v > 180):
                raise ValueError('Longitude must be between -180 and 180')
        return v
    
    @validator('tags', 'categories')
    def validate_tags(cls, v):
        if len(v) > 50:
            raise ValueError('Cannot have more than 50 items')
        return [item.strip().lower() for item in v if item.strip()]


class IncidentCreate(IncidentBase):
    """Schema for creating new incidents."""
    created_by_id: Optional[str] = None
    
    @validator('created_by_id')
    def validate_uuid(cls, v):
        if v is not None:
            try:
                uuid.UUID(v)
            except ValueError:
                raise ValueError('Invalid UUID format')
        return v


class IncidentUpdate(BaseModel):
    """Schema for updating incidents."""
    title: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = Field(None, min_length=10, max_length=10000)
    summary: Optional[str] = None
    incident_type: Optional[IncidentType] = None
    severity: Optional[IncidentSeverity] = None
    status: Optional[IncidentStatus] = None
    priority: Optional[IncidentPriority] = None
    confidence: Optional[IncidentConfidence] = None
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    occurred_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    estimated_impacted: Optional[int] = Field(None, ge=0)
    confirmed_impacted: Optional[int] = Field(None, ge=0)
    estimated_damage: Optional[int] = Field(None, ge=0)
    confirmed_damage: Optional[int] = Field(None, ge=0)
    tags: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    is_public: Optional[bool] = None
    is_sensitive: Optional[bool] = None
    assigned_to_id: Optional[str] = None
    risk_level: Optional[int] = Field(None, ge=1, le=10)


class IncidentInDBBase(IncidentBase):
    """Base schema for incident in database."""
    id: str
    status: IncidentStatus
    source_count: int
    media_count: int
    evidence_count: int
    verification_score: int
    risk_level: Optional[int]
    impact_score: float
    urgency_score: float
    is_ongoing: bool
    is_verified: bool
    verification_status: Optional[str]
    created_by_id: Optional[str]
    assigned_to_id: Optional[str]
    organization_id: Optional[str]
    created_at: datetime
    updated_at: datetime
    reported_at: datetime
    last_verified_at: Optional[datetime]
    next_review_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class Incident(IncidentInDBBase):
    """Schema for incident API responses."""
    created_by: Optional[Dict[str, Any]] = None
    assigned_to: Optional[Dict[str, Any]] = None
    organization: Optional[Dict[str, Any]] = None
    full_location: str
    coordinates: Optional[Dict[str, float]]
    has_location: bool
    age_hours: float
    duration_hours: Optional[float]
    primary_image: Optional[Dict[str, Any]] = None
    verification: Optional[Dict[str, Any]] = None
    article_count: int = 0
    comment_count: int = 0
    feedback_count: int = 0
    image_count: int = 0
    
    class Config:
        from_attributes = True


class IncidentSearchRequest(BaseModel):
    """Schema for incident search requests."""
    query: Optional[str] = None
    incident_type: Optional[IncidentType] = None
    severity: Optional[IncidentSeverity] = None
    status: Optional[IncidentStatus] = None
    priority: Optional[IncidentPriority] = None
    confidence: Optional[IncidentConfidence] = None
    country: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    tags: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    is_public: Optional[bool] = None
    is_verified: Optional[bool] = None
    is_ongoing: Optional[bool] = None
    created_by_id: Optional[str] = None
    assigned_to_id: Optional[str] = None
    organization_id: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    min_verification_score: Optional[int] = Field(None, ge=0, le=100)
    max_verification_score: Optional[int] = Field(None, ge=0, le=100)
    min_impact_score: Optional[float] = Field(None, ge=0, le=100)
    max_impact_score: Optional[float] = Field(None, ge=0, le=100)
    min_urgency_score: Optional[float] = Field(None, ge=0, le=100)
    max_urgency_score: Optional[float] = Field(None, ge=0, le=100)
    sort_by: str = Field(default="created_at", pattern="^(created_at|updated_at|occurred_at|verification_score|impact_score|urgency_score|priority)$")
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$")
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)
    
    class Config:
        from_attributes = True


class IncidentStats(BaseModel):
    """Schema for incident statistics."""
    total_incidents: int
    by_type: Dict[str, int]
    by_severity: Dict[str, int]
    by_status: Dict[str, int]
    by_country: Dict[str, int]
    by_month: Dict[str, int]
    ongoing_count: int
    verified_count: int
    average_verification_score: float
    average_impact_score: float
    total_estimated_impacted: int
    total_confirmed_impacted: int
    total_estimated_damage: int
    total_confirmed_damage: int
    recent_incidents: List[Dict[str, Any]]
    high_priority_incidents: List[Dict[str, Any]]
    
    class Config:
        from_attributes = True


class IncidentExportRequest(BaseModel):
    """Schema for incident export requests."""
    format: str = Field(default="json", pattern="^(json|csv|excel|geojson)$")
    include_related: bool = Field(default=False)
    include_metadata: bool = Field(default=False)
    fields: Optional[List[str]] = None
    
    class Config:
        from_attributes = True