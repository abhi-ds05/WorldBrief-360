"""
verification_evidence.py - Verification Evidence Model

This module defines the VerificationEvidence model for tracking and managing
evidence items used in the verification of incidents. This includes:
- Text evidence (reports, statements, articles)
- Media evidence (images, videos, documents)
- Data evidence (statistics, logs, datasets)
- Location evidence (coordinates, maps, geodata)
- Expert analysis and assessments

Key Features:
- Multi-type evidence classification
- Source attribution and reliability scoring
- Evidence verification tracking
- Metadata enrichment and tagging
- Relationship to verification sources
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
    from models.image import Image
    from models.verification_source import VerificationSource


class EvidenceType(Enum):
    """Types of evidence used in verification."""
    TEXT = "text"                      # Text documents, reports
    IMAGE = "image"                    # Photographs, screenshots
    VIDEO = "video"                    # Video recordings
    AUDIO = "audio"                    # Audio recordings
    DOCUMENT = "document"              # PDFs, Word docs, spreadsheets
    DATA = "data"                      # Datasets, statistics, logs
    LOCATION = "location"              # GPS coordinates, maps
    WITNESS_STATEMENT = "witness_statement"  # Eyewitness accounts
    EXPERT_ANALYSIS = "expert_analysis"      # Expert opinions/analysis
    OFFICIAL_REPORT = "official_report"      # Official reports
    NEWS_ARTICLE = "news_article"            # News articles
    SOCIAL_MEDIA = "social_media"            # Social media posts
    SATELLITE_IMAGERY = "satellite_imagery"  # Satellite images
    CCTV_FOOTAGE = "cctv_footage"            # Surveillance footage
    FORENSIC_REPORT = "forensic_report"      # Forensic analysis
    MEDICAL_REPORT = "medical_report"        # Medical reports
    FINANCIAL_RECORD = "financial_record"    # Financial documents
    LEGAL_DOCUMENT = "legal_document"        # Legal documents
    OTHER = "other"                          # Other evidence types


class EvidenceStatus(Enum):
    """Status of evidence verification."""
    PENDING = "pending"                # Not yet reviewed
    UNDER_REVIEW = "under_review"      # Currently being reviewed
    VERIFIED = "verified"              # Verified as authentic/accurate
    UNVERIFIED = "unverified"          # Cannot be verified
    MISLEADING = "misleading"          # Misleading or out of context
    FABRICATED = "fabricated"          # Proven to be fabricated
    OUTDATED = "outdated"              # No longer relevant/current
    INCONCLUSIVE = "inconclusive"      # Insufficient for verification
    DUPLICATE = "duplicate"            # Duplicate of existing evidence
    ARCHIVED = "archived"             # Archived for reference


class EvidenceReliability(Enum):
    """Reliability rating for evidence."""
    VERY_HIGH = "very_high"    # Official/primary sources, forensics
    HIGH = "high"              # Reputable sources, expert analysis
    MEDIUM = "medium"          # Established sources with some verification
    LOW = "low"                # Unverified or anonymous sources
    VERY_LOW = "very_low"      # Questionable/unreliable sources
    UNKNOWN = "unknown"        # Unknown reliability


class EvidenceFormat(Enum):
    """Formats of evidence content."""
    PLAIN_TEXT = "plain_text"
    RICH_TEXT = "rich_text"
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    JPEG = "jpeg"
    PNG = "png"
    GIF = "gif"
    MP4 = "mp4"
    AVI = "avi"
    MP3 = "mp3"
    WAV = "wav"
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    GEOJSON = "geojson"
    KML = "kml"
    OTHER = "other"


class VerificationEvidence(Base, UUIDMixin, TimestampMixin):
    """
    Verification Evidence model for tracking evidence items.
    
    This model manages evidence used in incident verification, including
    text, media, data, and other types of evidence with comprehensive
    metadata, verification tracking, and source attribution.
    
    Attributes:
        id: Primary key UUID
        verification_id: Foreign key to IncidentVerification
        evidence_type: Type of evidence
        status: Verification status
        reliability: Reliability rating
        format: Format of evidence content
        title: Evidence title/name
        description: Detailed description
        content: Evidence content (text or reference)
        source_url: Original source URL
        source_name: Name of source
        file_path: Path to stored file (if applicable)
        file_size: Size in bytes (if file)
        mime_type: MIME type (if file)
        checksum: File/content checksum for integrity
        language: Language of evidence content
        location: Geographic location (if applicable)
        coordinates: GPS coordinates (if applicable)
        captured_at: When evidence was captured/created
        submitted_at: When evidence was submitted
        submitted_by_id: User who submitted evidence
        verified_by_id: User who verified evidence
        verified_at: When evidence was verified
        confidence_score: Confidence in evidence (0-1)
        verification_notes: Notes about verification
        tags: Categorization tags
        metadata: Additional JSON metadata
        is_public: Whether evidence is publicly accessible
        is_archived: Whether evidence is archived
        view_count: Number of views
        citation_count: Number of citations
        related_source_id: Related verification source
        related_article_id: Related article
        related_image_id: Related image
    """
    
    __tablename__ = "verification_evidence"
    
    # Foreign key to verification
    verification_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("incident_verifications.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # Evidence classification
    evidence_type = Column(
        SQLEnum(EvidenceType),
        nullable=False,
        index=True
    )
    status = Column(
        SQLEnum(EvidenceStatus),
        default=EvidenceStatus.PENDING,
        nullable=False,
        index=True
    )
    reliability = Column(
        SQLEnum(EvidenceReliability),
        default=EvidenceReliability.UNKNOWN,
        nullable=False,
        index=True
    )
    format = Column(
        SQLEnum(EvidenceFormat),
        nullable=True,
        index=True
    )
    
    # Content information
    title = Column(String(500), nullable=True, index=True)
    description = Column(Text, nullable=True)
    content = Column(Text, nullable=True)  # For text evidence or references
    excerpt = Column(Text, nullable=True)  # Short excerpt/summary
    
    # Source and file information
    source_url = Column(String(2000), nullable=True)
    source_name = Column(String(255), nullable=True, index=True)
    attribution = Column(Text, nullable=True)
    file_path = Column(String(1024), nullable=True)
    file_size = Column(BigInteger, nullable=True)
    mime_type = Column(String(100), nullable=True)
    checksum = Column(String(64), nullable=True, index=True)
    
    # Location and timing
    language = Column(String(10), nullable=True, index=True)
    location = Column(String(255), nullable=True, index=True)
    latitude = Column(Float, nullable=True, index=True)
    longitude = Column(Float, nullable=True, index=True)
    location_accuracy = Column(Float, nullable=True)  # Meters
    captured_at = Column(DateTime(timezone=True), nullable=True, index=True)
    submitted_at = Column(DateTime(timezone=True), nullable=True)
    
    # Verification tracking
    submitted_by_id = Column(
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
    confidence_score = Column(Float, nullable=True, index=True)
    verification_notes = Column(Text, nullable=True)
    
    # Categorization and metadata
    tags = Column(ARRAY(String), default=[], nullable=False, index=True)
    categories = Column(ARRAY(String), default=[], nullable=False, index=True)
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Access and visibility
    is_public = Column(Boolean, default=False, nullable=False, index=True)
    is_archived = Column(Boolean, default=False, nullable=False, index=True)
    is_sensitive = Column(Boolean, default=False, nullable=False, index=True)
    
    # Usage statistics
    view_count = Column(Integer, default=0, nullable=False)
    citation_count = Column(Integer, default=0, nullable=False)
    download_count = Column(Integer, default=0, nullable=False)
    
    # Related entities
    related_source_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("verification_sources.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    related_article_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("articles.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    related_image_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("images.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    
    # Relationships
    verification = relationship("IncidentVerification", back_populates="evidence")
    submitted_by = relationship("User", foreign_keys=[submitted_by_id])
    verified_by = relationship("User", foreign_keys=[verified_by_id])
    related_source = relationship("VerificationSource")
    related_article = relationship("Article")
    related_image = relationship("Image")
    
    # Evidence versions (for updates/corrections)
    parent_evidence_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("verification_evidence.id", ondelete="SET NULL"), 
        nullable=True
    )
    versions = relationship(
        "VerificationEvidence",
        back_populates="parent",
        remote_side="VerificationEvidence.id",
        cascade="all, delete-orphan"
    )
    parent = relationship(
        "VerificationEvidence",
        remote_side=[id],
        back_populates="versions"
    )
    
    # Check constraints
    __table_args__ = (
        CheckConstraint(
            'confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1)',
            name='check_confidence_score_range'
        ),
        CheckConstraint(
            'file_size IS NULL OR file_size > 0',
            name='check_file_size_positive'
        ),
        CheckConstraint(
            'latitude IS NULL OR (latitude >= -90 AND latitude <= 90)',
            name='check_latitude_range'
        ),
        CheckConstraint(
            'longitude IS NULL OR (longitude >= -180 AND longitude <= 180)',
            name='check_longitude_range'
        ),
        CheckConstraint(
            'location_accuracy IS NULL OR location_accuracy >= 0',
            name='check_location_accuracy_non_negative'
        ),
        Index('ix_verification_evidence_verification_status', 'verification_id', 'status'),
        Index('ix_verification_evidence_type_reliability', 'evidence_type', 'reliability'),
        Index('ix_verification_evidence_created_status', 'created_at', 'status'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<VerificationEvidence(id={self.id}, type={self.evidence_type.value}, status={self.status.value})>"
    
    @validates('title')
    def validate_title(self, key: str, title: Optional[str]) -> Optional[str]:
        """Validate title."""
        if title and len(title) > 500:
            raise ValueError("Title cannot exceed 500 characters")
        return title
    
    @validates('confidence_score')
    def validate_confidence(self, key: str, score: Optional[float]) -> Optional[float]:
        """Validate confidence score."""
        if score is not None:
            if score < 0 or score > 1:
                raise ValueError("Confidence score must be between 0 and 1")
            score = round(score, 3)
        return score
    
    @validates('tags', 'categories')
    def validate_tags(self, key: str, items: List[str]) -> List[str]:
        """Validate tags and categories."""
        if len(items) > 50:
            raise ValueError(f"Cannot have more than 50 {key}")
        return [item.strip().lower() for item in items if item.strip()]
    
    @property
    def is_verified(self) -> bool:
        """Check if evidence is verified."""
        return self.status == EvidenceStatus.VERIFIED
    
    @property
    def is_fabricated(self) -> bool:
        """Check if evidence is fabricated."""
        return self.status == EvidenceStatus.FABRICATED
    
    @property
    def is_misleading(self) -> bool:
        """Check if evidence is misleading."""
        return self.status == EvidenceStatus.MISLEADING
    
    @property
    def needs_verification(self) -> bool:
        """Check if evidence needs verification."""
        return self.status in [
            EvidenceStatus.PENDING,
            EvidenceStatus.UNDER_REVIEW,
            EvidenceStatus.INCONCLUSIVE
        ]
    
    @property
    def has_location(self) -> bool:
        """Check if evidence has location data."""
        return self.latitude is not None and self.longitude is not None
    
    @property
    def location_data(self) -> Optional[Dict[str, Any]]:
        """Get location data as dictionary."""
        if self.has_location:
            return {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "accuracy": self.location_accuracy,
                "location": self.location
            }
        return None
    
    @property
    def is_media(self) -> bool:
        """Check if evidence is media type."""
        return self.evidence_type in [
            EvidenceType.IMAGE,
            EvidenceType.VIDEO,
            EvidenceType.AUDIO,
            EvidenceType.SATELLITE_IMAGERY,
            EvidenceType.CCTV_FOOTAGE
        ]
    
    @property
    def is_document(self) -> bool:
        """Check if evidence is document type."""
        return self.evidence_type in [
            EvidenceType.DOCUMENT,
            EvidenceType.OFFICIAL_REPORT,
            EvidenceType.FORENSIC_REPORT,
            EvidenceType.MEDICAL_REPORT,
            EvidenceType.FINANCIAL_RECORD,
            EvidenceType.LEGAL_DOCUMENT
        ]
    
    @property
    def is_text(self) -> bool:
        """Check if evidence is text type."""
        return self.evidence_type in [
            EvidenceType.TEXT,
            EvidenceType.WITNESS_STATEMENT,
            EvidenceType.EXPERT_ANALYSIS,
            EvidenceType.NEWS_ARTICLE,
            EvidenceType.SOCIAL_MEDIA
        ]
    
    @property
    def file_size_mb(self) -> Optional[float]:
        """Get file size in megabytes."""
        if self.file_size:
            return self.file_size / (1024 * 1024)
        return None
    
    @property
    def age_days(self) -> Optional[float]:
        """Get age of evidence in days."""
        reference_date = self.captured_at or self.created_at
        if reference_date:
            delta = datetime.utcnow() - reference_date
            return delta.total_seconds() / (24 * 3600)
        return None
    
    @property
    def evidence_score(self) -> float:
        """Calculate evidence quality score."""
        score = 0.0
        
        # Reliability score
        reliability_scores = {
            EvidenceReliability.VERY_HIGH: 1.0,
            EvidenceReliability.HIGH: 0.8,
            EvidenceReliability.MEDIUM: 0.6,
            EvidenceReliability.LOW: 0.4,
            EvidenceReliability.VERY_LOW: 0.2,
            EvidenceReliability.UNKNOWN: 0.1,
        }
        score += reliability_scores.get(self.reliability, 0.1) * 40  # 40%
        
        # Verification status score
        if self.is_verified:
            score += 30  # 30%
        elif self.status == EvidenceStatus.UNVERIFIED:
            score += 10
        elif self.status == EvidenceStatus.INCONCLUSIVE:
            score += 5
        
        # Confidence score contribution
        if self.confidence_score:
            score += self.confidence_score * 20  # 20%
        
        # Source completeness bonus
        if self.source_url and self.source_name:
            score += 5
        
        # Location data bonus
        if self.has_location:
            score += 5
        
        return min(100.0, score)
    
    def verify(
        self,
        status: EvidenceStatus,
        verified_by: 'User',
        confidence_score: Optional[float] = None,
        notes: Optional[str] = None
    ) -> None:
        """
        Verify the evidence.
        
        Args:
            status: New verification status
            verified_by: User performing verification
            confidence_score: Confidence in verification (0-1)
            notes: Verification notes
        """
        old_status = self.status
        self.status = status
        self.verified_by_id = verified_by.id
        self.verified_at = datetime.utcnow()
        
        if confidence_score is not None:
            self.confidence_score = confidence_score
        
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
            'confidence_score': confidence_score,
            'notes': notes
        })
    
    def update_reliability(
        self,
        reliability: EvidenceReliability,
        updated_by: Optional['User'] = None,
        notes: Optional[str] = None
    ) -> None:
        """Update evidence reliability rating."""
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
    
    def add_tag(self, tag: str) -> bool:
        """Add a tag to evidence."""
        tag = tag.strip().lower()
        if tag and tag not in self.tags:
            self.tags.append(tag)
            return True
        return False
    
    def remove_tag(self, tag: str) -> bool:
        """Remove a tag from evidence."""
        tag = tag.strip().lower()
        if tag in self.tags:
            self.tags.remove(tag)
            return True
        return False
    
    def increment_view_count(self) -> None:
        """Increment view count."""
        self.view_count += 1
    
    def increment_citation_count(self) -> None:
        """Increment citation count."""
        self.citation_count += 1
    
    def increment_download_count(self) -> None:
        """Increment download count."""
        self.download_count += 1
    
    def create_version(
        self,
        updated_by: 'User',
        changes: Dict[str, Any],
        notes: Optional[str] = None
    ) -> 'VerificationEvidence':
        """
        Create a new version of the evidence.
        
        Args:
            updated_by: User creating the version
            changes: Dictionary of changed fields
            notes: Version notes
            
        Returns:
            New VerificationEvidence version
        """
        # Create new evidence as version
        version = VerificationEvidence(
            verification_id=self.verification_id,
            evidence_type=self.evidence_type,
            status=EvidenceStatus.PENDING,  # New versions need verification
            reliability=self.reliability,
            format=self.format,
            title=self.title,
            description=self.description,
            content=self.content,
            source_url=self.source_url,
            source_name=self.source_name,
            parent_evidence_id=self.id,
            submitted_by_id=updated_by.id,
            submitted_at=datetime.utcnow(),
            metadata={
                'version_of': str(self.id),
                'version_changes': changes,
                'version_notes': notes,
                'previous_version': self.metadata.get('version', 1)
            }
        )
        
        # Update version tracking
        current_version = self.metadata.get('version', 1)
        self.metadata['version'] = current_version + 1
        self.metadata['has_versions'] = True
        
        return version
    
    def to_dict(self, include_content: bool = True, include_metadata: bool = False) -> Dict[str, Any]:
        """
        Convert evidence to dictionary.
        
        Args:
            include_content: Whether to include content field
            include_metadata: Whether to include metadata
            
        Returns:
            Dictionary representation
        """
        result = {
            "id": str(self.id),
            "verification_id": str(self.verification_id),
            "evidence_type": self.evidence_type.value,
            "status": self.status.value,
            "reliability": self.reliability.value,
            "format": self.format.value if self.format else None,
            "title": self.title,
            "description": self.description,
            "excerpt": self.excerpt,
            "source_url": self.source_url,
            "source_name": self.source_name,
            "attribution": self.attribution,
            "file_size": self.file_size,
            "file_size_mb": round(self.file_size_mb, 2) if self.file_size_mb else None,
            "mime_type": self.mime_type,
            "checksum": self.checksum,
            "language": self.language,
            "location": self.location,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "location_accuracy": self.location_accuracy,
            "has_location": self.has_location,
            "location_data": self.location_data,
            "captured_at": self.captured_at.isoformat() if self.captured_at else None,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "is_verified": self.is_verified,
            "is_fabricated": self.is_fabricated,
            "is_misleading": self.is_misleading,
            "needs_verification": self.needs_verification,
            "is_media": self.is_media,
            "is_document": self.is_document,
            "is_text": self.is_text,
            "confidence_score": self.confidence_score,
            "verification_notes": self.verification_notes,
            "tags": self.tags,
            "categories": self.categories,
            "is_public": self.is_public,
            "is_archived": self.is_archived,
            "is_sensitive": self.is_sensitive,
            "view_count": self.view_count,
            "citation_count": self.citation_count,
            "download_count": self.download_count,
            "evidence_score": round(self.evidence_score, 2),
            "age_days": round(self.age_days, 2) if self.age_days else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "submitted_by_id": str(self.submitted_by_id) if self.submitted_by_id else None,
            "verified_by_id": str(self.verified_by_id) if self.verified_by_id else None,
            "related_source_id": str(self.related_source_id) if self.related_source_id else None,
            "related_article_id": str(self.related_article_id) if self.related_article_id else None,
            "related_image_id": str(self.related_image_id) if self.related_image_id else None,
            "parent_evidence_id": str(self.parent_evidence_id) if self.parent_evidence_id else None,
            "has_versions": len(self.versions) > 0 if self.versions else False,
            "version_count": len(self.versions) if self.versions else 0
        }
        
        if include_content and self.content:
            result["content"] = self.content
        
        if include_metadata:
            result["metadata"] = self.metadata
        
        if self.submitted_by:
            result["submitted_by"] = {
                "id": str(self.submitted_by.id),
                "username": self.submitted_by.username
            }
        
        if self.verified_by:
            result["verified_by"] = {
                "id": str(self.verified_by.id),
                "username": self.verified_by.username
            }
        
        if self.related_source:
            result["related_source"] = {
                "id": str(self.related_source.id),
                "source_type": self.related_source.source_type
            }
        
        return result
    
    @classmethod
    def create(
        cls,
        verification_id: uuid.UUID,
        evidence_type: EvidenceType,
        title: Optional[str] = None,
        description: Optional[str] = None,
        content: Optional[str] = None,
        source_url: Optional[str] = None,
        source_name: Optional[str] = None,
        submitted_by_id: Optional[uuid.UUID] = None,
        reliability: EvidenceReliability = EvidenceReliability.UNKNOWN,
        format: Optional[EvidenceFormat] = None,
        language: Optional[str] = None,
        location: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        captured_at: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        is_public: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'VerificationEvidence':
        """
        Factory method to create new evidence.
        
        Args:
            verification_id: Verification ID
            evidence_type: Type of evidence
            title: Evidence title
            description: Evidence description
            content: Evidence content
            source_url: Source URL
            source_name: Source name
            submitted_by_id: User submitting evidence
            reliability: Reliability rating
            format: Evidence format
            language: Content language
            location: Location description
            latitude: GPS latitude
            longitude: GPS longitude
            captured_at: When evidence was captured
            tags: Categorization tags
            categories: Evidence categories
            is_public: Whether evidence is public
            metadata: Additional metadata
            
        Returns:
            A new VerificationEvidence instance
        """
        # Validate coordinates
        if latitude is not None:
            if latitude < -90 or latitude > 90:
                raise ValueError("Latitude must be between -90 and 90")
        if longitude is not None:
            if longitude < -180 or longitude > 180:
                raise ValueError("Longitude must be between -180 and 180")
        
        evidence = cls(
            verification_id=verification_id,
            evidence_type=evidence_type,
            title=title,
            description=description,
            content=content,
            source_url=source_url,
            source_name=source_name,
            submitted_by_id=submitted_by_id,
            reliability=reliability,
            format=format,
            language=language,
            location=location,
            latitude=latitude,
            longitude=longitude,
            captured_at=captured_at,
            submitted_at=datetime.utcnow(),
            tags=tags or [],
            categories=categories or [],
            is_public=is_public,
            metadata=metadata or {},
            status=EvidenceStatus.PENDING
        )
        
        return evidence
    
    @classmethod
    def create_from_article(
        cls,
        verification_id: uuid.UUID,
        article: 'Article',
        submitted_by_id: Optional[uuid.UUID] = None,
        excerpt: Optional[str] = None,
        **kwargs
    ) -> 'VerificationEvidence':
        """
        Create evidence from an existing article.
        
        Args:
            verification_id: Verification ID
            article: Article instance
            submitted_by_id: User submitting evidence
            excerpt: Custom excerpt (uses article excerpt if None)
            **kwargs: Additional arguments
            
        Returns:
            VerificationEvidence instance
        """
        excerpt = excerpt or article.excerpt or article.content[:500] + "..." if article.content else None
        
        evidence = cls.create(
            verification_id=verification_id,
            evidence_type=EvidenceType.NEWS_ARTICLE,
            title=article.title,
            description=article.description,
            content=article.content,
            source_url=article.url,
            source_name=article.source,
            submitted_by_id=submitted_by_id,
            reliability=EvidenceReliability.MEDIUM,  # Default for articles
            language=article.language,
            tags=["article", "news"] + (article.tags if hasattr(article, 'tags') else []),
            categories=["media", "publication"],
            related_article_id=article.id,
            **kwargs
        )
        
        evidence.excerpt = excerpt
        
        # Add article metadata
        evidence.metadata.update({
            "article_metadata": {
                "id": str(article.id),
                "published_at": article.published_at.isoformat() if article.published_at else None,
                "author": article.author,
                "category": article.category if hasattr(article, 'category') else None
            }
        })
        
        return evidence
    
    @classmethod
    def create_from_image(
        cls,
        verification_id: uuid.UUID,
        image: 'Image',
        submitted_by_id: Optional[uuid.UUID] = None,
        description: Optional[str] = None,
        **kwargs
    ) -> 'VerificationEvidence':
        """
        Create evidence from an existing image.
        
        Args:
            verification_id: Verification ID
            image: Image instance
            submitted_by_id: User submitting evidence
            description: Evidence description (uses image caption if None)
            **kwargs: Additional arguments
            
        Returns:
            VerificationEvidence instance
        """
        description = description or image.caption or f"Image: {image.filename}"
        
        evidence = cls.create(
            verification_id=verification_id,
            evidence_type=EvidenceType.IMAGE,
            title=image.filename,
            description=description,
            source_url=image.storage_url or image.cdn_url,
            source_name=image.uploaded_by.username if image.uploaded_by else None,
            submitted_by_id=submitted_by_id,
            reliability=EvidenceReliability.UNKNOWN,  # Images need verification
            format=EvidenceFormat.JPEG if image.format.value.lower() in ['jpeg', 'jpg'] else EvidenceFormat.PNG,
            file_path=image.storage_path,
            file_size=image.file_size,
            mime_type=image.mime_type,
            checksum=image.file_hash,
            latitude=image.metadata.get('latitude') if image.metadata else None,
            longitude=image.metadata.get('longitude') if image.metadata else None,
            captured_at=image.metadata.get('captured_at') if image.metadata else None,
            tags=["image", "visual"] + (image.tags if hasattr(image, 'tags') else []),
            categories=["media", "visual"],
            related_image_id=image.id,
            **kwargs
        )
        
        # Add image metadata
        evidence.metadata.update({
            "image_metadata": {
                "id": str(image.id),
                "dimensions": f"{image.width}x{image.height}" if image.width and image.height else None,
                "format": image.format.value,
                "uploaded_at": image.uploaded_at.isoformat() if image.uploaded_at else None,
                "is_verified": image.is_approved
            }
        })
        
        return evidence


# Pydantic schemas for API validation
"""
If you're using Pydantic, here are the schemas for the VerificationEvidence model.
"""

from pydantic import BaseModel, Field, validator, HttpUrl
from typing import Optional, List, Dict, Any
from datetime import datetime


class EvidenceBase(BaseModel):
    """Base schema for evidence operations."""
    evidence_type: EvidenceType
    title: Optional[str] = Field(None, max_length=500)
    description: Optional[str] = None
    content: Optional[str] = None
    source_url: Optional[HttpUrl] = None
    source_name: Optional[str] = Field(None, max_length=255)
    reliability: EvidenceReliability = Field(default=EvidenceReliability.UNKNOWN)
    language: Optional[str] = Field(None, max_length=10)
    location: Optional[str] = Field(None, max_length=255)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    captured_at: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    is_public: bool = Field(default=False)
    
    @validator('title')
    def validate_title(cls, v):
        if v and len(v) > 500:
            raise ValueError('Title cannot exceed 500 characters')
        return v
    
    @validator('tags', 'categories')
    def validate_tags(cls, v):
        if len(v) > 50:
            raise ValueError('Cannot have more than 50 items')
        return [item.strip().lower() for item in v if item.strip()]


class EvidenceCreate(EvidenceBase):
    """Schema for creating new evidence."""
    verification_id: str
    submitted_by_id: Optional[str] = None
    
    @validator('verification_id', 'submitted_by_id')
    def validate_uuids(cls, v):
        if v is not None:
            try:
                uuid.UUID(v)
            except ValueError:
                raise ValueError('Invalid UUID format')
        return v


class EvidenceUpdate(BaseModel):
    """Schema for updating evidence."""
    title: Optional[str] = Field(None, max_length=500)
    description: Optional[str] = None
    reliability: Optional[EvidenceReliability] = None
    status: Optional[EvidenceStatus] = None
    tags: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    is_public: Optional[bool] = None
    is_archived: Optional[bool] = None
    verification_notes: Optional[str] = None


class EvidenceVerification(BaseModel):
    """Schema for verifying evidence."""
    status: EvidenceStatus
    confidence_score: Optional[float] = Field(None, ge=0, le=1)
    verification_notes: Optional[str] = None


class EvidenceInDBBase(EvidenceBase):
    """Base schema for evidence in database."""
    id: str
    verification_id: str
    status: EvidenceStatus
    format: Optional[str]
    file_size: Optional[int]
    checksum: Optional[str]
    is_verified: bool
    is_fabricated: bool
    is_misleading: bool
    needs_verification: bool
    confidence_score: Optional[float]
    evidence_score: float
    view_count: int
    citation_count: int
    download_count: int
    submitted_by_id: Optional[str]
    verified_by_id: Optional[str]
    verified_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class Evidence(EvidenceInDBBase):
    """Schema for evidence API responses."""
    submitted_by: Optional[Dict[str, Any]] = None
    verified_by: Optional[Dict[str, Any]] = None
    file_size_mb: Optional[float]
    age_days: Optional[float]
    has_location: bool
    location_data: Optional[Dict[str, Any]]
    is_media: bool
    is_document: bool
    is_text: bool
    
    class Config:
        from_attributes = True


class EvidenceSearchRequest(BaseModel):
    """Schema for evidence search requests."""
    verification_id: Optional[str] = None
    evidence_type: Optional[EvidenceType] = None
    status: Optional[EvidenceStatus] = None
    reliability: Optional[EvidenceReliability] = None
    is_verified: Optional[bool] = None
    is_public: Optional[bool] = None
    tags: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    submitted_by_id: Optional[str] = None
    verified_by_id: Optional[str] = None
    has_location: Optional[bool] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    min_confidence_score: Optional[float] = Field(None, ge=0, le=1)
    max_confidence_score: Optional[float] = Field(None, ge=0, le=1)
    sort_by: str = Field(default="evidence_score", pattern="^(evidence_score|created_at|updated_at|view_count|confidence_score)$")
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$")
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)
    
    class Config:
        from_attributes = True


class EvidenceStats(BaseModel):
    """Schema for evidence statistics."""
    total_evidence: int
    by_type: Dict[str, int]
    by_status: Dict[str, int]
    by_reliability: Dict[str, int]
    verified_percentage: float
    average_confidence_score: Optional[float]
    average_evidence_score: float
    total_views: int
    total_citations: int
    
    class Config:
        from_attributes = True