"""
verification_source.py - Verification Source Model

This module defines the VerificationSource model for tracking and managing
sources used in the verification of incidents. This includes:
- Official sources (government agencies, authorities)
- Media sources (news organizations, journalists)
- Expert sources (subject matter experts, analysts)
- Eyewitness sources (first-hand accounts)
- Data sources (datasets, statistics, research)
- Social media sources (platforms, influencers)
- Automated sources (APIs, feeds, monitoring tools)

Key Features:
- Source classification and typing
- Reliability and credibility scoring
- Source verification tracking
- Contact and attribution management
- Usage statistics and impact tracking
"""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from enum import Enum
from sqlalchemy import (
    Column, String, Text, ForeignKey, Integer, DateTime,ARRAY, BigInteger, 
    Boolean, Enum as SQLEnum, JSON, Float, CheckConstraint,
    UniqueConstraint, Index
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func

from db.base import Base
from models.mixins import TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from models.incident_verification import IncidentVerification
    from models.user import User
    from models.article import Article
    from models.verification_evidence import VerificationEvidence


class SourceType(Enum):
    """Types of verification sources."""
    GOVERNMENT = "government"            # Government agencies, officials
    LAW_ENFORCEMENT = "law_enforcement"  # Police, military, security
    EMERGENCY_SERVICES = "emergency_services"  # Fire, medical, rescue
    NEWS_MEDIA = "news_media"            # Newspapers, TV, radio
    ONLINE_MEDIA = "online_media"        # Digital news, blogs, portals
    SOCIAL_MEDIA = "social_media"        # Twitter, Facebook, Instagram
    EXPERT = "expert"                    # Subject matter experts
    RESEARCH_INSTITUTION = "research_institution"  # Universities, think tanks
    NGO = "ngo"                          # Non-governmental organizations
    INTERNATIONAL_ORG = "international_org"  # UN, Red Cross, WHO
    CORPORATE = "corporate"              # Companies, businesses
    EYEWITNESS = "eyewitness"            # First-hand witnesses
    AFFECTED_PARTY = "affected_party"    # Directly affected individuals
    PUBLIC_DATA = "public_data"          # Public datasets, statistics
    SATELLITE = "satellite"              # Satellite imagery/data
    SENSOR = "sensor"                    # IoT sensors, monitoring devices
    CCTV = "cctv"                        # Surveillance cameras
    AUTOMATED = "automated"              # Automated monitoring systems
    USER_GENERATED = "user_generated"    # User-submitted content
    ARCHIVE = "archive"                  # Historical archives
    OTHER = "other"                      # Other source types


class SourceStatus(Enum):
    """Status of source verification and activity."""
    ACTIVE = "active"                    # Currently active and reliable
    VERIFIED = "verified"                # Verified as reliable
    PENDING_VERIFICATION = "pending_verification"  # Awaiting verification
    UNVERIFIED = "unverified"            # Not yet verified
    SUSPENDED = "suspended"              # Temporarily suspended
    BLACKLISTED = "blacklisted"          # Permanently unreliable
    RETIRED = "retired"                  # No longer active
    ARCHIVED = "archived"                #Archived for reference
    CONTESTED = "contested"              # Reliability contested
    LIMITED = "limited"                  # Limited reliability


class SourceReliability(Enum):
    """Reliability rating for sources."""
    A_PLUS = "a_plus"        # Highest reliability (official/government)
    A = "a"                  # High reliability (established media)
    B_PLUS = "b_plus"        # Good reliability (reputable organizations)
    B = "b"                  # Moderate reliability (established sources)
    C_PLUS = "c_plus"        # Fair reliability (some verification)
    C = "c"                  # Basic reliability (limited verification)
    D = "d"                  # Low reliability (unverified/anonymous)
    E = "e"                  # Very low reliability (questionable)
    F = "f"                  # Unreliable (frequently inaccurate)
    UNKNOWN = "unknown"      # Unknown reliability


class SourceTier(Enum):
    """Tier classification for sources."""
    TIER_1 = "tier_1"        # Primary/official sources
    TIER_2 = "tier_2"        # Reputable secondary sources
    TIER_3 = "tier_3"        #Established tertiary sources
    TIER_4 = "tier_4"        # Emerging/unverified sources
    TIER_5 = "tier_5"        # Questionable/unofficial sources


class VerificationSource(Base, UUIDMixin, TimestampMixin):
    """
    Verification Source model for tracking information sources.
    
    This model manages sources used in incident verification, including
    their classification, reliability assessment, verification status,
    and usage tracking across the platform.
    
    Attributes:
        id: Primary key UUID
        verification_id: Foreign key to IncidentVerification
        source_type: Type of source
        status: Source status
        reliability: Reliability rating
        tier: Source tier classification
        name: Source name/organization
        description: Source description
        url: Source website/URL
        contact_email: Contact email
        contact_phone: Contact phone
        location: Source location
        country: Source country
        language: Primary language
        established_date: When source was established
        verified_at: When source was verified
        verified_by_id: User who verified source
        verification_notes: Verification notes
        credibility_score: Credibility score (0-100)
        usage_count: Number of times used
        last_used_at: Last time source was used
        metadata: Additional JSON metadata
        tags: Categorization tags
        categories: Source categories
        is_public: Whether source is publicly visible
        is_automated: Whether source is automated
        requires_attribution: Whether attribution is required
        attribution_format: Format for attribution
        api_endpoint: API endpoint (if automated source)
        api_key_required: Whether API key is required
        rate_limit: API rate limit (requests per minute)
        last_checked_at: Last time source was checked/updated
        update_frequency: How often source is updated
        is_active: Whether source is active
        parent_source_id: Parent source (for organizations)
        related_articles: Related articles from this source
        related_evidence: Evidence from this source
    """
    
    __tablename__ = "verification_sources"
    
    # Foreign key to verification
    verification_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("incident_verifications.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # Source classification
    source_type = Column(
        SQLEnum(SourceType),
        nullable=False,
        index=True
    )
    status = Column(
        SQLEnum(SourceStatus),
        default=SourceStatus.UNVERIFIED,
        nullable=False,
        index=True
    )
    reliability = Column(
        SQLEnum(SourceReliability),
        default=SourceReliability.UNKNOWN,
        nullable=False,
        index=True
    )
    tier = Column(
        SQLEnum(SourceTier),
        default=SourceTier.TIER_4,
        nullable=False,
        index=True
    )
    
    # Source information
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    url = Column(String(2000), nullable=True)
    contact_email = Column(String(255), nullable=True, index=True)
    contact_phone = Column(String(50), nullable=True)
    
    # Location and language
    location = Column(String(255), nullable=True, index=True)
    country = Column(String(2), nullable=True, index=True)  # ISO 3166-1 alpha-2
    language = Column(String(10), nullable=True, index=True)
    established_date = Column(DateTime(timezone=True), nullable=True)
    
    # Verification tracking
    verified_at = Column(DateTime(timezone=True), nullable=True)
    verified_by_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    verification_notes = Column(Text, nullable=True)
    
    # Scoring and usage
    credibility_score = Column(Integer, default=0, nullable=False, index=True)
    accuracy_score = Column(Float, nullable=True, index=True)  # 0-1
    response_time_score = Column(Integer, nullable=True)  # 1-10
    usage_count = Column(Integer, default=0, nullable=False)
    last_used_at = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # Metadata and categorization
    metadata = Column(JSONB, default=dict, nullable=False)
    tags = Column(ARRAY(String), default=[], nullable=False, index=True)
    categories = Column(ARRAY(String), default=[], nullable=False, index=True)
    
    # Visibility and access
    is_public = Column(Boolean, default=True, nullable=False, index=True)
    is_automated = Column(Boolean, default=False, nullable=False, index=True)
    requires_attribution = Column(Boolean, default=True, nullable=False)
    attribution_format = Column(String(500), nullable=True)
    
    # API/automation details
    api_endpoint = Column(String(2000), nullable=True)
    api_key_required = Column(Boolean, default=False, nullable=False)
    rate_limit = Column(Integer, nullable=True)  # requests per minute
    last_checked_at = Column(DateTime(timezone=True), nullable=True)
    update_frequency = Column(String(50), nullable=True)  # daily, hourly, etc.
    
    # Activity status
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    
    # Parent-child relationships (for organizations)
    parent_source_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("verification_sources.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    
    # Relationships
    verification = relationship("IncidentVerification", back_populates="sources")
    verified_by = relationship("User", foreign_keys=[verified_by_id])
    
    # Parent-child relationships
    parent = relationship(
        "VerificationSource",
        remote_side=[id],
        back_populates="children"
    )
    children = relationship(
        "VerificationSource",
        back_populates="parent",
        cascade="all, delete-orphan"
    )
    
    # Related articles and evidence
    related_articles = relationship(
        "Article",
        back_populates="source_reference",
        primaryjoin="Article.source == VerificationSource.name"
    )
    
    related_evidence = relationship(
        "VerificationEvidence",
        back_populates="related_source",
        foreign_keys="[VerificationEvidence.related_source_id]"
    )
    
    # Check constraints
    __table_args__ = (
        CheckConstraint(
            'credibility_score >= 0 AND credibility_score <= 100',
            name='check_credibility_score_range'
        ),
        CheckConstraint(
            'accuracy_score IS NULL OR (accuracy_score >= 0 AND accuracy_score <= 1)',
            name='check_accuracy_score_range'
        ),
        CheckConstraint(
            'response_time_score IS NULL OR (response_time_score >= 1 AND response_time_score <= 10)',
            name='check_response_time_score_range'
        ),
        CheckConstraint(
            'usage_count >= 0',
            name='check_usage_count_non_negative'
        ),
        UniqueConstraint('verification_id', 'name', name='uq_verification_source_name'),
        Index('ix_verification_sources_type_reliability', 'source_type', 'reliability'),
        Index('ix_verification_sources_credibility_status', 'credibility_score', 'status'),
        Index('ix_verification_sources_last_used', 'last_used_at', 'is_active'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<VerificationSource(id={self.id}, name='{self.name}', type={self.source_type.value})>"
    
    @validates('name')
    def validate_name(self, key: str, name: str) -> str:
        """Validate source name."""
        name = name.strip()
        if not name:
            raise ValueError("Source name cannot be empty")
        if len(name) > 255:
            raise ValueError("Source name cannot exceed 255 characters")
        return name
    
    @validates('country')
    def validate_country(self, key: str, country: Optional[str]) -> Optional[str]:
        """Validate country code."""
        if country and len(country) != 2:
            raise ValueError("Country must be 2-letter ISO code")
        return country.upper() if country else None
    
    @validates('credibility_score')
    def validate_credibility_score(self, key: str, score: int) -> int:
        """Validate credibility score."""
        if score < 0 or score > 100:
            raise ValueError("Credibility score must be between 0 and 100")
        return score
    
    @validates('tags', 'categories')
    def validate_tags(self, key: str, items: List[str]) -> List[str]:
        """Validate tags and categories."""
        if len(items) > 50:
            raise ValueError(f"Cannot have more than 50 {key}")
        return [item.strip().lower() for item in items if item.strip()]
    
    @property
    def is_verified(self) -> bool:
        """Check if source is verified."""
        return self.status in [SourceStatus.VERIFIED, SourceStatus.ACTIVE]
    
    @property
    def is_reliable(self) -> bool:
        """Check if source is reliable."""
        return self.reliability in [
            SourceReliability.A_PLUS,
            SourceReliability.A,
            SourceReliability.B_PLUS,
            SourceReliability.B
        ]
    
    @property
    def is_high_tier(self) -> bool:
        """Check if source is high tier."""
        return self.tier in [SourceTier.TIER_1, SourceTier.TIER_2]
    
    @property
    def needs_verification(self) -> bool:
        """Check if source needs verification."""
        return self.status in [
            SourceStatus.UNVERIFIED,
            SourceStatus.PENDING_VERIFICATION,
            SourceStatus.CONTESTED
        ]
    
    @property
    def is_suspended_or_blacklisted(self) -> bool:
        """Check if source is suspended or blacklisted."""
        return self.status in [SourceStatus.SUSPENDED, SourceStatus.BLACKLISTED]
    
    @property
    def age_years(self) -> Optional[int]:
        """Get age of source in years."""
        if self.established_date:
            delta = datetime.utcnow() - self.established_date
            return delta.days // 365
        return None
    
    @property
    def days_since_last_use(self) -> Optional[int]:
        """Get days since last use."""
        if self.last_used_at:
            delta = datetime.utcnow() - self.last_used_at
            return delta.days
        return None
    
    @property
    def source_score(self) -> float:
        """Calculate overall source quality score."""
        score = 0.0
        
        # Reliability contributes 40%
        reliability_scores = {
            SourceReliability.A_PLUS: 40,
            SourceReliability.A: 36,
            SourceReliability.B_PLUS: 32,
            SourceReliability.B: 28,
            SourceReliability.C_PLUS: 24,
            SourceReliability.C: 20,
            SourceReliability.D: 12,
            SourceReliability.E: 8,
            SourceReliability.F: 4,
            SourceReliability.UNKNOWN: 0,
        }
        score += reliability_scores.get(self.reliability, 0)
        
        # Credibility score contributes 30%
        score += (self.credibility_score / 100) * 30
        
        # Tier contributes 20%
        tier_scores = {
            SourceTier.TIER_1: 20,
            SourceTier.TIER_2: 16,
            SourceTier.TIER_3: 12,
            SourceTier.TIER_4: 8,
            SourceTier.TIER_5: 4,
        }
        score += tier_scores.get(self.tier, 0)
        
        # Age bonus (older sources get more trust)
        if self.age_years and self.age_years >= 5:
            score += min(5, self.age_years // 5)  # Max 5 points
        
        # Usage bonus (frequently used sources)
        if self.usage_count >= 10:
            score += min(5, self.usage_count // 10)  # Max 5 points
        
        return min(100.0, score)
    
    @property
    def attribution_text(self) -> Optional[str]:
        """Generate attribution text."""
        if not self.requires_attribution:
            return None
        
        if self.attribution_format:
            return self.attribution_format.format(
                name=self.name,
                url=self.url
            )
        
        # Default attribution format
        if self.url:
            return f"Source: {self.name} ({self.url})"
        return f"Source: {self.name}"
    
    def verify(
        self,
        status: SourceStatus,
        verified_by: 'User',
        reliability: Optional[SourceReliability] = None,
        tier: Optional[SourceTier] = None,
        credibility_score: Optional[int] = None,
        notes: Optional[str] = None
    ) -> None:
        """
        Verify the source.
        
        Args:
            status: New source status
            verified_by: User performing verification
            reliability: Updated reliability rating
            tier: Updated tier classification
            credibility_score: Updated credibility score
            notes: Verification notes
        """
        old_status = self.status
        self.status = status
        self.verified_by_id = verified_by.id
        self.verified_at = datetime.utcnow()
        
        if reliability is not None:
            self.reliability = reliability
        
        if tier is not None:
            self.tier = tier
        
        if credibility_score is not None:
            self.credibility_score = credibility_score
        
        if notes:
            self.verification_notes = notes
        
        # Log verification in metadata
        if 'verification_history' not in self.metadata:
            self.metadata['verification_history'] = []
        
        self.metadata['verification_history'].append({
            'from_status': old_status.value,
            'to_status': status.value,
            'verified_by': str(verified_by.id),
            'verified_at': self.verified_at.isoformat(),
            'reliability': reliability.value if reliability else None,
            'tier': tier.value if tier else None,
            'credibility_score': credibility_score,
            'notes': notes
        })
    
    def update_reliability(
        self,
        reliability: SourceReliability,
        updated_by: Optional['User'] = None,
        notes: Optional[str] = None
    ) -> None:
        """Update source reliability rating."""
        old_reliability = self.reliability
        self.reliability = reliability
        
        # Log reliability change
        if 'reliability_history' not in self.metadata:
            self.metadata['reliability_history'] = []
        
        self.metadata['reliability_history'].append({
            'from_reliability': old_reliability.value,
            'to_reliability': reliability.value,
            'updated_by': str(updated_by.id) if updated_by else None,
            'updated_at': datetime.utcnow().isoformat(),
            'notes': notes
        })
    
    def increment_usage(self) -> None:
        """Increment usage count and update last used timestamp."""
        self.usage_count += 1
        self.last_used_at = datetime.utcnow()
    
    def update_accuracy_score(self, accuracy_score: float, based_on: int = 1) -> None:
        """
        Update accuracy score based on new data.
        
        Args:
            accuracy_score: New accuracy measurement (0-1)
            based_on: Number of measurements this is based on
        """
        if self.accuracy_score is None:
            self.accuracy_score = accuracy_score
        else:
            # Weighted average
            total_measurements = self.metadata.get('accuracy_measurements', 0) + based_on
            weight = based_on / total_measurements
            self.accuracy_score = self.accuracy_score * (1 - weight) + accuracy_score * weight
            self.metadata['accuracy_measurements'] = total_measurements
    
    def add_tag(self, tag: str) -> bool:
        """Add a tag to source."""
        tag = tag.strip().lower()
        if tag and tag not in self.tags:
            self.tags.append(tag)
            return True
        return False
    
    def remove_tag(self, tag: str) -> bool:
        """Remove a tag from source."""
        tag = tag.strip().lower()
        if tag in self.tags:
            self.tags.remove(tag)
            return True
        return False
    
    def create_child_source(
        self,
        name: str,
        source_type: SourceType,
        description: Optional[str] = None,
        created_by: Optional['User'] = None
    ) -> 'VerificationSource':
        """
        Create a child source (e.g., department within organization).
        
        Args:
            name: Child source name
            source_type: Child source type
            description: Child source description
            created_by: User creating child source
            
        Returns:
            New child VerificationSource
        """
        child = VerificationSource(
            verification_id=self.verification_id,
            name=name,
            source_type=source_type,
            description=description,
            parent_source_id=self.id,
            reliability=self.reliability,
            tier=self.tier,
            credibility_score=self.credibility_score,
            status=SourceStatus.UNVERIFIED,  # Child needs separate verification
            is_public=self.is_public,
            metadata={
                'parent_source': str(self.id),
                'created_by': str(created_by.id) if created_by else None,
                'created_at': datetime.utcnow().isoformat()
            }
        )
        
        self.children.append(child)
        return child
    
    def mark_as_checked(self) -> None:
        """Mark source as checked/updated."""
        self.last_checked_at = datetime.utcnow()
    
    def suspend(self, suspended_by: Optional['User'] = None, reason: Optional[str] = None) -> None:
        """Suspend the source."""
        old_status = self.status
        self.status = SourceStatus.SUSPENDED
        self.is_active = False
        
        # Log suspension
        if 'status_history' not in self.metadata:
            self.metadata['status_history'] = []
        
        self.metadata['status_history'].append({
            'from_status': old_status.value,
            'to_status': SourceStatus.SUSPENDED.value,
            'changed_by': str(suspended_by.id) if suspended_by else None,
            'changed_at': datetime.utcnow().isoformat(),
            'reason': reason
        })
    
    def activate(self, activated_by: Optional['User'] = None, reason: Optional[str] = None) -> None:
        """Activate the source."""
        old_status = self.status
        self.status = SourceStatus.ACTIVE
        self.is_active = True
        
        # Log activation
        if 'status_history' not in self.metadata:
            self.metadata['status_history'] = []
        
        self.metadata['status_history'].append({
            'from_status': old_status.value,
            'to_status': SourceStatus.ACTIVE.value,
            'changed_by': str(activated_by.id) if activated_by else None,
            'changed_at': datetime.utcnow().isoformat(),
            'reason': reason
        })
    
    def to_dict(self, include_children: bool = False, include_metadata: bool = False) -> Dict[str, Any]:
        """Convert source to dictionary."""
        result = {
            "id": str(self.id),
            "verification_id": str(self.verification_id),
            "source_type": self.source_type.value,
            "status": self.status.value,
            "reliability": self.reliability.value,
            "tier": self.tier.value,
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "contact_email": self.contact_email,
            "contact_phone": self.contact_phone,
            "location": self.location,
            "country": self.country,
            "language": self.language,
            "established_date": self.established_date.isoformat() if self.established_date else None,
            "is_verified": self.is_verified,
            "is_reliable": self.is_reliable,
            "is_high_tier": self.is_high_tier,
            "needs_verification": self.needs_verification,
            "is_suspended_or_blacklisted": self.is_suspended_or_blacklisted,
            "credibility_score": self.credibility_score,
            "accuracy_score": self.accuracy_score,
            "response_time_score": self.response_time_score,
            "source_score": round(self.source_score, 2),
            "usage_count": self.usage_count,
            "age_years": self.age_years,
            "days_since_last_use": self.days_since_last_use,
            "is_public": self.is_public,
            "is_automated": self.is_automated,
            "requires_attribution": self.requires_attribution,
            "attribution_text": self.attribution_text,
            "api_endpoint": self.api_endpoint,
            "api_key_required": self.api_key_required,
            "rate_limit": self.rate_limit,
            "last_checked_at": self.last_checked_at.isoformat() if self.last_checked_at else None,
            "update_frequency": self.update_frequency,
            "is_active": self.is_active,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "verified_by_id": str(self.verified_by_id) if self.verified_by_id else None,
            "parent_source_id": str(self.parent_source_id) if self.parent_source_id else None,
            "tags": self.tags,
            "categories": self.categories,
            "has_children": len(self.children) > 0 if self.children else False,
            "children_count": len(self.children) if self.children else 0,
            "related_articles_count": len(self.related_articles) if self.related_articles else 0,
            "related_evidence_count": len(self.related_evidence) if self.related_evidence else 0
        }
        
        if include_children and self.children:
            result["children"] = [
                child.to_dict(include_children=False, include_metadata=False)
                for child in self.children
                if child.is_active
            ]
        
        if include_metadata:
            result["metadata"] = self.metadata
        
        if self.verified_by:
            result["verified_by"] = {
                "id": str(self.verified_by.id),
                "username": self.verified_by.username
            }
        
        if self.parent:
            result["parent"] = {
                "id": str(self.parent.id),
                "name": self.parent.name,
                "source_type": self.parent.source_type.value
            }
        
        return result
    
    @classmethod
    def create(
        cls,
        verification_id: uuid.UUID,
        name: str,
        source_type: SourceType,
        description: Optional[str] = None,
        url: Optional[str] = None,
        location: Optional[str] = None,
        country: Optional[str] = None,
        language: Optional[str] = None,
        established_date: Optional[datetime] = None,
        reliability: SourceReliability = SourceReliability.UNKNOWN,
        tier: SourceTier = SourceTier.TIER_4,
        credibility_score: int = 0,
        is_public: bool = True,
        is_automated: bool = False,
        tags: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'VerificationSource':
        """
        Factory method to create a new verification source.
        
        Args:
            verification_id: Verification ID
            name: Source name
            source_type: Type of source
            description: Source description
            url: Source website/URL
            location: Source location
            country: Source country (ISO code)
            language: Primary language
            established_date: When source was established
            reliability: Reliability rating
            tier: Source tier
            credibility_score: Initial credibility score
            is_public: Whether source is public
            is_automated: Whether source is automated
            tags: Categorization tags
            categories: Source categories
            metadata: Additional metadata
            
        Returns:
            A new VerificationSource instance
        """
        # Validate country code
        if country and len(country) != 2:
            raise ValueError("Country must be 2-letter ISO code")
        
        source = cls(
            verification_id=verification_id,
            name=name.strip(),
            source_type=source_type,
            description=description,
            url=url,
            location=location,
            country=country.upper() if country else None,
            language=language,
            established_date=established_date,
            reliability=reliability,
            tier=tier,
            credibility_score=max(0, min(100, credibility_score)),
            status=SourceStatus.UNVERIFIED,
            is_public=is_public,
            is_automated=is_automated,
            tags=tags or [],
            categories=categories or [],
            metadata=metadata or {},
            is_active=True
        )
        
        return source
    
    @classmethod
    def create_government_source(
        cls,
        verification_id: uuid.UUID,
        name: str,
        country: str,
        **kwargs
    ) -> 'VerificationSource':
        """
        Create a government source with default settings.
        
        Args:
            verification_id: Verification ID
            name: Government agency name
            country: Country code
            **kwargs: Additional arguments
            
        Returns:
            VerificationSource instance
        """
        return cls.create(
            verification_id=verification_id,
            name=name,
            source_type=SourceType.GOVERNMENT,
            country=country,
            reliability=SourceReliability.A_PLUS,
            tier=SourceTier.TIER_1,
            credibility_score=90,
            tags=["government", "official", "authority"],
            categories=["official", "primary"],
            **kwargs
        )
    
    @classmethod
    def create_news_media_source(
        cls,
        verification_id: uuid.UUID,
        name: str,
        url: str,
        country: Optional[str] = None,
        **kwargs
    ) -> 'VerificationSource':
        """
        Create a news media source with default settings.
        
        Args:
            verification_id: Verification ID
            name: News organization name
            url: News website URL
            country: Country code
            **kwargs: Additional arguments
            
        Returns:
            VerificationSource instance
        """
        return cls.create(
            verification_id=verification_id,
            name=name,
            source_type=SourceType.NEWS_MEDIA,
            url=url,
            country=country,
            reliability=SourceReliability.B,
            tier=SourceTier.TIER_2,
            credibility_score=70,
            tags=["news", "media", "journalism"],
            categories=["media", "secondary"],
            **kwargs
        )
    
    @classmethod
    def create_eyewitness_source(
        cls,
        verification_id: uuid.UUID,
        name: str,
        contact_email: Optional[str] = None,
        location: Optional[str] = None,
        **kwargs
    ) -> 'VerificationSource':
        """
        Create an eyewitness source with default settings.
        
        Args:
            verification_id: Verification ID
            name: Witness name
            contact_email: Contact email
            location: Witness location
            **kwargs: Additional arguments
            
        Returns:
            VerificationSource instance
        """
        return cls.create(
            verification_id=verification_id,
            name=name,
            source_type=SourceType.EYEWITNESS,
            contact_email=contact_email,
            location=location,
            reliability=SourceReliability.C,
            tier=SourceTier.TIER_4,
            credibility_score=40,
            tags=["eyewitness", "personal", "testimony"],
            categories=["personal", "tertiary"],
            is_public=False,  # Typically keep eyewitnesses private
            **kwargs
        )


# Pydantic schemas for API validation
"""
If you're using Pydantic, here are the schemas for the VerificationSource model.
"""

from pydantic import BaseModel, Field, validator, HttpUrl
from typing import Optional, List, Dict, Any
from datetime import datetime


class SourceBase(BaseModel):
    """Base schema for source operations."""
    name: str = Field(..., max_length=255)
    source_type: SourceType
    description: Optional[str] = None
    url: Optional[HttpUrl] = None
    location: Optional[str] = Field(None, max_length=255)
    country: Optional[str] = Field(None, min_length=2, max_length=2)
    language: Optional[str] = Field(None, max_length=10)
    established_date: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    is_public: bool = Field(default=True)
    is_automated: bool = Field(default=False)
    
    @validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Source name cannot be empty')
        return v.strip()
    
    @validator('country')
    def validate_country(cls, v):
        if v and len(v) != 2:
            raise ValueError('Country must be 2-letter ISO code')
        return v.upper() if v else v
    
    @validator('tags', 'categories')
    def validate_tags(cls, v):
        if len(v) > 50:
            raise ValueError('Cannot have more than 50 items')
        return [item.strip().lower() for item in v if item.strip()]


class SourceCreate(SourceBase):
    """Schema for creating new sources."""
    verification_id: str
    reliability: SourceReliability = Field(default=SourceReliability.UNKNOWN)
    tier: SourceTier = Field(default=SourceTier.TIER_4)
    credibility_score: int = Field(default=0, ge=0, le=100)
    
    @validator('verification_id')
    def validate_uuid(cls, v):
        try:
            uuid.UUID(v)
            return v
        except ValueError:
            raise ValueError('Invalid UUID format')


class SourceUpdate(BaseModel):
    """Schema for updating sources."""
    description: Optional[str] = None
    status: Optional[SourceStatus] = None
    reliability: Optional[SourceReliability] = None
    tier: Optional[SourceTier] = None
    credibility_score: Optional[int] = Field(None, ge=0, le=100)
    verification_notes: Optional[str] = None
    tags: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    is_public: Optional[bool] = None
    is_active: Optional[bool] = None


class SourceVerification(BaseModel):
    """Schema for verifying sources."""
    status: SourceStatus
    reliability: Optional[SourceReliability] = None
    tier: Optional[SourceTier] = None
    credibility_score: Optional[int] = Field(None, ge=0, le=100)
    verification_notes: Optional[str] = None


class SourceInDBBase(SourceBase):
    """Base schema for source in database."""
    id: str
    verification_id: str
    status: SourceStatus
    reliability: SourceReliability
    tier: SourceTier
    credibility_score: int
    source_score: float
    is_verified: bool
    is_reliable: bool
    is_high_tier: bool
    needs_verification: bool
    is_suspended_or_blacklisted: bool
    usage_count: int
    is_active: bool
    verified_at: Optional[datetime]
    last_used_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class Source(SourceInDBBase):
    """Schema for source API responses."""
    verified_by: Optional[Dict[str, Any]] = None
    parent: Optional[Dict[str, Any]] = None
    age_years: Optional[int]
    days_since_last_use: Optional[int]
    attribution_text: Optional[str]
    has_children: bool
    children_count: int
    related_articles_count: int
    related_evidence_count: int
    
    class Config:
        from_attributes = True


class SourceSearchRequest(BaseModel):
    """Schema for source search requests."""
    verification_id: Optional[str] = None
    source_type: Optional[SourceType] = None
    status: Optional[SourceStatus] = None
    reliability: Optional[SourceReliability] = None
    tier: Optional[SourceTier] = None
    is_verified: Optional[bool] = None
    is_reliable: Optional[bool] = None
    is_active: Optional[bool] = None
    is_public: Optional[bool] = None
    is_automated: Optional[bool] = None
    country: Optional[str] = None
    language: Optional[str] = None
    tags: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    min_credibility_score: Optional[int] = Field(None, ge=0, le=100)
    max_credibility_score: Optional[int] = Field(None, ge=0, le=100)
    min_source_score: Optional[float] = Field(None, ge=0, le=100)
    max_source_score: Optional[float] = Field(None, ge=0, le=100)
    name_contains: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    sort_by: str = Field(default="source_score", pattern="^(source_score|credibility_score|name|created_at|last_used_at|usage_count)$")
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$")
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)
    
    class Config:
        from_attributes = True


class SourceStats(BaseModel):
    """Schema for source statistics."""
    total_sources: int
    by_type: Dict[str, int]
    by_status: Dict[str, int]
    by_reliability: Dict[str, int]
    by_tier: Dict[str, int]
    by_country: Dict[str, int]
    verified_percentage: float
    reliable_percentage: float
    average_credibility_score: float
    average_source_score: float
    most_used_sources: List[Dict[str, Any]]
    recently_added_sources: List[Dict[str, Any]]
    
    class Config:
        from_attributes = True