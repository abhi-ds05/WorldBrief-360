"""
incident_image.py - Incident-Image Association Model

This module defines the IncidentImage model, which serves as a junction/association 
table between Incidents and Images. This allows for:
- Multiple images per incident
- Image categorization within incidents
- Image metadata specific to incident context
- Image ordering and prioritization
- Image verification and attribution

Key Features:
- Association metadata (caption, source, verification status)
- Image categorization (primary, evidence, diagram, etc.)
- Ordering and display priority
- Verification tracking
- Spatial/temporal metadata
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, TYPE_CHECKING
from enum import Enum
from sqlalchemy import (
    Column, String, Text, ForeignKey, Integer, DateTime, 
    Boolean, Enum as SQLEnum, JSON, Float, CheckConstraint
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func

from db.base import Base
from models.mixins import TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from models.incident import Incident
    from models.image import Image
    from models.user import User


class IncidentImageType(Enum):
    """Types of images in relation to an incident."""
    PRIMARY = "primary"            # Main/representative image
    EVIDENCE = "evidence"          # Supporting evidence
    DIAGRAM = "diagram"            # Diagram/schematic
    MAP = "map"                    # Map/location visualization
    CHART = "chart"                # Data chart/graph
    SCREENSHOT = "screenshot"      # Screen capture
    DOCUMENT = "document"          # Document image
    PROFILE = "profile"            # Profile picture of involved person
    LOCATION = "location"          # Location photo
    DAMAGE = "damage"              # Damage assessment
    REPAIR = "repair"              # Repair/aftermath
    BEFORE = "before"              # Before incident
    AFTER = "after"                # After incident
    CONTEXT = "context"            # Contextual/background
    OTHER = "other"                # Other type


class ImageVerificationStatus(Enum):
    """Verification status of incident images."""
    UNVERIFIED = "unverified"      # Not yet verified
    VERIFIED = "verified"          # Verified as accurate
    MISLEADING = "misleading"      # Misleading/misrepresented
    FALSE = "false"                # Proven false/fabricated
    OUTDATED = "outdated"          # Out of date/not current
    PENDING = "pending"            # Verification in progress
    INCONCLUSIVE = "inconclusive"  # Cannot be verified


class ImageSourceType(Enum):
    """Source types for incident images."""
    USER_SUBMITTED = "user_submitted"      # Submitted by user
    OFFICIAL = "official"                  # Official/government source
    NEWS_MEDIA = "news_media"              # News organization
    SOCIAL_MEDIA = "social_media"          # Social media platform
    CCTV = "cctv"                          # CCTV/surveillance
    SATELLITE = "satellite"                # Satellite imagery
    DRONE = "drone"                        # Drone/aerial
    WEBCAM = "webcam"                      # Public webcam
    ARCHIVE = "archive"                    # Archived/historical
    GENERATED = "generated"                # AI/computer generated
    UNKNOWN = "unknown"                    # Unknown source


class IncidentImage(Base, UUIDMixin, TimestampMixin):
    """
    Association model between Incidents and Images.
    
    This model adds incident-specific metadata to the image relationship,
    allowing for rich contextual information about how each image relates
    to a specific incident.
    
    Attributes:
        id: Primary key UUID
        incident_id: Foreign key to Incident
        image_id: Foreign key to Image
        image_type: Type of image in incident context
        verification_status: Verification status
        source_type: Source of the image
        caption: Incident-specific caption
        description: Detailed description
        source_url: Original source URL
        source_name: Name of source (person/organization)
        taken_at: When the image was taken (if known)
        taken_location: Where the image was taken
        display_order: Order for display (lower = earlier)
        is_primary: Whether this is the primary image
        is_featured: Whether to feature this image
        verification_notes: Notes about verification
        verified_by_id: User who verified the image
        verified_at: When verification occurred
        confidence_score: Confidence in accuracy (0-1)
        metadata: Additional JSON metadata
        is_deleted: Soft delete flag
    """
    
    __tablename__ = "incident_images"
    
    # Foreign keys
    incident_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("incidents.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    image_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("images.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # Image type and classification
    image_type = Column(
        SQLEnum(IncidentImageType),
        default=IncidentImageType.PRIMARY,
        nullable=False,
        index=True
    )
    verification_status = Column(
        SQLEnum(ImageVerificationStatus),
        default=ImageVerificationStatus.UNVERIFIED,
        nullable=False,
        index=True
    )
    source_type = Column(
        SQLEnum(ImageSourceType),
        default=ImageSourceType.UNKNOWN,
        nullable=False,
        index=True
    )
    
    # Content and attribution
    caption = Column(String(500), nullable=True)
    description = Column(Text, nullable=True)
    source_url = Column(String(2000), nullable=True)
    source_name = Column(String(255), nullable=True)
    attribution = Column(Text, nullable=True)
    copyright = Column(String(255), nullable=True)
    
    # Temporal and spatial metadata
    taken_at = Column(DateTime(timezone=True), nullable=True, index=True)
    taken_location = Column(String(255), nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    location_accuracy = Column(Float, nullable=True)  # In meters
    
    # Display and organization
    display_order = Column(Integer, default=0, nullable=False, index=True)
    is_primary = Column(Boolean, default=False, nullable=False, index=True)
    is_featured = Column(Boolean, default=False, nullable=False, index=True)
    is_public = Column(Boolean, default=True, nullable=False, index=True)
    
    # Verification details
    verification_notes = Column(Text, nullable=True)
    verified_by_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    verified_at = Column(DateTime(timezone=True), nullable=True)
    confidence_score = Column(Float, nullable=True, index=True)  # 0-1
    
    # Usage and engagement
    view_count = Column(Integer, default=0, nullable=False)
    download_count = Column(Integer, default=0, nullable=False)
    citation_count = Column(Integer, default=0, nullable=False)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Soft delete
    is_deleted = Column(Boolean, default=False, nullable=False, index=True)
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    deleted_by_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True
    )
    
    # Relationships
    incident = relationship("Incident", back_populates="incident_images")
    image = relationship("Image", back_populates="incident_associations")
    verified_by = relationship("User", foreign_keys=[verified_by_id])
    deleted_by = relationship("User", foreign_keys=[deleted_by_id])
    
    # Check constraints
    __table_args__ = (
        CheckConstraint(
            'confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1)',
            name='check_confidence_score_range'
        ),
        CheckConstraint(
            'latitude IS NULL OR (latitude >= -90 AND latitude <= 90)',
            name='check_latitude_range'
        ),
        CheckConstraint(
            'longitude IS NULL OR (longitude >= -180 AND longitude <= 180)',
            name='check_longitude_range'
        ),
        # Ensure only one primary image per incident
        # This would typically be enforced at application level or with partial index
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<IncidentImage(incident={self.incident_id}, image={self.image_id}, type={self.image_type.value})>"
    
    @validates('caption')
    def validate_caption(self, key: str, caption: Optional[str]) -> Optional[str]:
        """Validate caption length."""
        if caption and len(caption) > 500:
            raise ValueError("Caption cannot exceed 500 characters")
        return caption
    
    @validates('confidence_score')
    def validate_confidence(self, key: str, score: Optional[float]) -> Optional[float]:
        """Validate confidence score."""
        if score is not None:
            if score < 0 or score > 1:
                raise ValueError("Confidence score must be between 0 and 1")
            # Round to 3 decimal places
            score = round(score, 3)
        return score
    
    @validates('display_order')
    def validate_display_order(self, key: str, order: int) -> int:
        """Validate display order."""
        if order < 0:
            raise ValueError("Display order cannot be negative")
        return order
    
    @property
    def is_verified(self) -> bool:
        """Check if image is verified."""
        return self.verification_status == ImageVerificationStatus.VERIFIED
    
    @property
    def is_false(self) -> bool:
        """Check if image is false/fabricated."""
        return self.verification_status == ImageVerificationStatus.FALSE
    
    @property
    def is_misleading(self) -> bool:
        """Check if image is misleading."""
        return self.verification_status == ImageVerificationStatus.MISLEADING
    
    @property
    def needs_verification(self) -> bool:
        """Check if image needs verification."""
        return self.verification_status in [
            ImageVerificationStatus.UNVERIFIED,
            ImageVerificationStatus.PENDING
        ]
    
    @property
    def has_location(self) -> bool:
        """Check if image has location data."""
        return self.latitude is not None and self.longitude is not None
    
    @property
    def location(self) -> Optional[Dict[str, float]]:
        """Get location as dictionary."""
        if self.has_location:
            return {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "accuracy": self.location_accuracy
            }
        return None
    
    @property
    def age_days(self) -> Optional[float]:
        """Get age of image in days (from taken_at or created_at)."""
        reference_date = self.taken_at or self.created_at
        if reference_date:
            delta = datetime.utcnow() - reference_date
            return delta.total_seconds() / (24 * 3600)
        return None
    
    def verify(
        self, 
        status: ImageVerificationStatus,
        verified_by: 'User',
        notes: Optional[str] = None,
        confidence_score: Optional[float] = None
    ) -> None:
        """
        Verify the image.
        
        Args:
            status: Verification status
            verified_by: User performing verification
            notes: Verification notes
            confidence_score: Confidence in verification (0-1)
        """
        old_status = self.verification_status
        self.verification_status = status
        self.verified_by_id = verified_by.id
        self.verified_at = datetime.utcnow()
        self.verification_notes = notes
        self.confidence_score = confidence_score
        
        # Update metadata
        if 'verification_history' not in self.metadata:
            self.metadata['verification_history'] = []
        
        self.metadata['verification_history'].append({
            'from_status': old_status.value,
            'to_status': status.value,
            'verified_by': str(verified_by.id),
            'verified_at': self.verified_at.isoformat(),
            'notes': notes,
            'confidence_score': confidence_score
        })
    
    def mark_as_primary(self) -> None:
        """Mark this image as primary for the incident."""
        # In practice, you'd want to unset primary flag for other images
        # This would be handled at the service layer
        self.is_primary = True
        self.display_order = 0  # Primary should be first
    
    def increment_view_count(self) -> None:
        """Increment view count."""
        self.view_count += 1
    
    def increment_download_count(self) -> None:
        """Increment download count."""
        self.download_count += 1
    
    def increment_citation_count(self) -> None:
        """Increment citation count."""
        self.citation_count += 1
    
    def soft_delete(self, deleted_by: Optional['User'] = None) -> None:
        """Soft delete the association."""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()
        if deleted_by:
            self.deleted_by_id = deleted_by.id
    
    def restore(self) -> None:
        """Restore a soft-deleted association."""
        self.is_deleted = False
        self.deleted_at = None
        self.deleted_by_id = None
    
    def to_dict(self, include_image: bool = True, include_incident: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary for API responses.
        
        Args:
            include_image: Whether to include full image data
            include_incident: Whether to include incident data
            
        Returns:
            Dictionary representation
        """
        result = {
            "id": str(self.id),
            "incident_id": str(self.incident_id),
            "image_id": str(self.image_id),
            "image_type": self.image_type.value,
            "verification_status": self.verification_status.value,
            "source_type": self.source_type.value,
            "caption": self.caption,
            "description": self.description,
            "source_url": self.source_url,
            "source_name": self.source_name,
            "attribution": self.attribution,
            "copyright": self.copyright,
            "taken_at": self.taken_at.isoformat() if self.taken_at else None,
            "taken_location": self.taken_location,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "location_accuracy": self.location_accuracy,
            "display_order": self.display_order,
            "is_primary": self.is_primary,
            "is_featured": self.is_featured,
            "is_public": self.is_public,
            "is_verified": self.is_verified,
            "is_false": self.is_false,
            "is_misleading": self.is_misleading,
            "needs_verification": self.needs_verification,
            "has_location": self.has_location,
            "location": self.location,
            "age_days": round(self.age_days, 2) if self.age_days else None,
            "verification_notes": self.verification_notes,
            "confidence_score": self.confidence_score,
            "view_count": self.view_count,
            "download_count": self.download_count,
            "citation_count": self.citation_count,
            "is_deleted": self.is_deleted,
            "deleted_at": self.deleted_at.isoformat() if self.deleted_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "metadata": self.metadata
        }
        
        if include_image and self.image:
            result["image"] = self.image.to_dict(include_variants=False)
        
        if include_incident and self.incident:
            result["incident"] = {
                "id": str(self.incident.id),
                "title": self.incident.title,
                "severity": self.incident.severity,
                "status": self.incident.status
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
        image_id: uuid.UUID,
        image_type: IncidentImageType = IncidentImageType.PRIMARY,
        caption: Optional[str] = None,
        description: Optional[str] = None,
        source_type: ImageSourceType = ImageSourceType.UNKNOWN,
        source_url: Optional[str] = None,
        source_name: Optional[str] = None,
        taken_at: Optional[datetime] = None,
        taken_location: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        is_primary: bool = False,
        is_featured: bool = False,
        display_order: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'IncidentImage':
        """
        Factory method to create a new incident-image association.
        
        Args:
            incident_id: Incident ID
            image_id: Image ID
            image_type: Type of image in incident context
            caption: Incident-specific caption
            description: Detailed description
            source_type: Source type
            source_url: Source URL
            source_name: Source name
            taken_at: When image was taken
            taken_location: Where image was taken
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            is_primary: Whether this is primary image
            is_featured: Whether to feature this image
            display_order: Display order
            metadata: Additional metadata
            
        Returns:
            A new IncidentImage instance
        """
        # Validate coordinates if provided
        if latitude is not None:
            if latitude < -90 or latitude > 90:
                raise ValueError("Latitude must be between -90 and 90")
        if longitude is not None:
            if longitude < -180 or longitude > 180:
                raise ValueError("Longitude must be between -180 and 180")
        
        # Create association
        incident_image = cls(
            incident_id=incident_id,
            image_id=image_id,
            image_type=image_type,
            caption=caption,
            description=description,
            source_type=source_type,
            source_url=source_url,
            source_name=source_name,
            taken_at=taken_at,
            taken_location=taken_location,
            latitude=latitude,
            longitude=longitude,
            is_primary=is_primary,
            is_featured=is_featured,
            display_order=display_order,
            metadata=metadata or {},
            verification_status=ImageVerificationStatus.UNVERIFIED
        )
        
        return incident_image
    
    @classmethod
    def create_from_image_upload(
        cls,
        incident_id: uuid.UUID,
        image_file,  # Uploaded file object
        uploaded_by_id: uuid.UUID,
        **kwargs
    ) -> 'IncidentImage':
        """
        Create incident image association from uploaded file.
        
        Args:
            incident_id: Incident ID
            image_file: Uploaded file object
            uploaded_by_id: User uploading the image
            **kwargs: Additional arguments for Image creation
            
        Returns:
            IncidentImage instance with created Image
        """
        from models.image import Image, ImageType
        
        # First create the Image
        image = Image.create_from_upload(
            file_data=image_file,
            filename=image_file.filename,
            uploaded_by_id=uploaded_by_id,
            image_type=ImageType.CONTENT,
            **{k: v for k, v in kwargs.items() if hasattr(Image.create_from_upload, k)}
        )
        
        # Then create the association
        incident_image = cls.create(
            incident_id=incident_id,
            image_id=image.id,
            **{k: v for k, v in kwargs.items() if hasattr(cls.create, k)}
        )
        
        # Set metadata about the upload
        incident_image.metadata.update({
            "uploaded_by": str(uploaded_by_id),
            "uploaded_at": datetime.utcnow().isoformat()
        })
        
        return incident_image


# Pydantic schemas for API validation
"""
If you're using Pydantic, here are the schemas for the IncidentImage model.
"""

from pydantic import BaseModel, Field, validator, HttpUrl
from typing import Optional, Dict, Any
from datetime import datetime


class IncidentImageBase(BaseModel):
    """Base schema for incident image operations."""
    image_type: IncidentImageType = Field(default=IncidentImageType.PRIMARY)
    caption: Optional[str] = Field(None, max_length=500)
    description: Optional[str] = None
    source_type: ImageSourceType = Field(default=ImageSourceType.UNKNOWN)
    source_url: Optional[HttpUrl] = None
    source_name: Optional[str] = Field(None, max_length=255)
    taken_at: Optional[datetime] = None
    taken_location: Optional[str] = None
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    is_primary: bool = Field(default=False)
    is_featured: bool = Field(default=False)
    display_order: int = Field(default=0, ge=0)
    
    @validator('caption')
    def validate_caption(cls, v):
        if v and len(v) > 500:
            raise ValueError('Caption cannot exceed 500 characters')
        return v
    
    @validator('latitude', 'longitude')
    def validate_coordinates(cls, v, values, **kwargs):
        if v is not None:
            field = kwargs.get('field')
            if field.name == 'latitude' and (v < -90 or v > 90):
                raise ValueError('Latitude must be between -90 and 90')
            if field.name == 'longitude' and (v < -180 or v > 180):
                raise ValueError('Longitude must be between -180 and 180')
        return v


class IncidentImageCreate(IncidentImageBase):
    """Schema for creating new incident-image associations."""
    incident_id: str
    image_id: str
    
    @validator('incident_id', 'image_id')
    def validate_uuids(cls, v):
        try:
            uuid.UUID(v)
            return v
        except ValueError:
            raise ValueError('Invalid UUID format')


class IncidentImageUpdate(BaseModel):
    """Schema for updating incident-image associations."""
    image_type: Optional[IncidentImageType] = None
    caption: Optional[str] = Field(None, max_length=500)
    description: Optional[str] = None
    verification_status: Optional[ImageVerificationStatus] = None
    is_primary: Optional[bool] = None
    is_featured: Optional[bool] = None
    display_order: Optional[int] = Field(None, ge=0)
    verification_notes: Optional[str] = None
    confidence_score: Optional[float] = Field(None, ge=0, le=1)


class IncidentImageVerification(BaseModel):
    """Schema for verifying incident images."""
    verification_status: ImageVerificationStatus
    verification_notes: Optional[str] = None
    confidence_score: Optional[float] = Field(None, ge=0, le=1)


class IncidentImageInDBBase(IncidentImageBase):
    """Base schema for incident image in database."""
    id: str
    incident_id: str
    image_id: str
    verification_status: ImageVerificationStatus
    confidence_score: Optional[float]
    view_count: int
    download_count: int
    citation_count: int
    is_verified: bool
    is_false: bool
    is_misleading: bool
    needs_verification: bool
    has_location: bool
    created_at: datetime
    updated_at: datetime
    verified_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class IncidentImage(IncidentImageInDBBase):
    """Schema for incident image API responses."""
    image: Optional[Dict[str, Any]] = None
    incident: Optional[Dict[str, Any]] = None
    verified_by: Optional[Dict[str, Any]] = None
    location: Optional[Dict[str, float]] = None
    age_days: Optional[float]
    
    class Config:
        from_attributes = True


class IncidentImageUpload(BaseModel):
    """Schema for uploading images to incidents."""
    file: Any  # UploadFile in FastAPI
    caption: Optional[str] = Field(None, max_length=500)
    description: Optional[str] = None
    image_type: IncidentImageType = Field(default=IncidentImageType.PRIMARY)
    source_type: ImageSourceType = Field(default=ImageSourceType.USER_SUBMITTED)
    taken_at: Optional[datetime] = None
    taken_location: Optional[str] = None
    is_primary: bool = Field(default=False)
    is_featured: bool = Field(default=False)


class IncidentImageStats(BaseModel):
    """Schema for incident image statistics."""
    total_images: int
    by_type: Dict[str, int]
    by_verification_status: Dict[str, int]
    by_source_type: Dict[str, int]
    verified_percentage: float
    primary_images: int
    featured_images: int
    average_views_per_image: float
    total_views: int
    
    class Config:
        from_attributes = True