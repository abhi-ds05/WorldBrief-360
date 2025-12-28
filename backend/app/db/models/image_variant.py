"""
image_variant.py - Image Variant Model for Processed Image Versions
"""

import uuid
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, TYPE_CHECKING
from enum import Enum
from sqlalchemy import (
    Column, String, ForeignKey, Integer, DateTime, Text, 
    Boolean, Enum as SQLEnum, JSON, Float, BigInteger,
    CheckConstraint
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func

from db.base import Base
from models.mixins import TimestampMixin, UUIDMixin

# Import shared enums
from .image_enums import ImageFormat, StorageBackend

if TYPE_CHECKING:
    from models.image import Image
    from models.user import User


class VariantType(Enum):
    """Types of image variants."""
    THUMBNAIL = "thumbnail"
    THUMBNAIL_SMALL = "thumbnail_small"    # 150x150
    THUMBNAIL_MEDIUM = "thumbnail_medium"  # 300x300
    THUMBNAIL_LARGE = "thumbnail_large"    # 600x600
    SMALL = "small"                        # max 800px
    MEDIUM = "medium"                      # max 1200px
    LARGE = "large"                        # max 1920px
    EXTRA_LARGE = "extra_large"            # max 2560px
    WEBP = "webp"                          # WebP conversion
    AVIF = "avif"                          # AVIF conversion
    WATERMARKED = "watermarked"            # Watermarked version
    BLUR = "blur"                          # Blurred placeholder
    GRAYSCALE = "grayscale"                # Grayscale version


class VariantStatus(Enum):
    """Processing status of variants."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"


class ImageVariant(Base, UUIDMixin, TimestampMixin):
    """
    ImageVariant model for storing processed versions of images.
    """
    
    __tablename__ = "image_variants"
    
    # Original image reference
    original_image_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("images.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # Variant type and status
    variant_type = Column(
        SQLEnum(VariantType),
        nullable=False,
        index=True
    )
    status = Column(
        SQLEnum(VariantStatus),
        default=VariantStatus.PENDING,
        nullable=False,
        index=True
    )
    
    # Dimensions and format
    width = Column(Integer, nullable=True, index=True)
    height = Column(Integer, nullable=True, index=True)
    format = Column(
        SQLEnum(ImageFormat),
        nullable=False,
        index=True
    )
    quality = Column(Integer, nullable=True)
    
    # File properties
    file_size = Column(BigInteger, nullable=True)
    file_hash = Column(String(64), nullable=True, unique=True)
    
    # Storage information
    storage_backend = Column(
        SQLEnum(StorageBackend),
        default=StorageBackend.LOCAL,
        nullable=False,
        index=True
    )
    storage_path = Column(String(1024), nullable=False, index=True)
    storage_url = Column(String(2048), nullable=True)
    cdn_url = Column(String(2048), nullable=True, index=True)
    
    # Processing details
    processing_parameters = Column(JSONB, default=dict, nullable=False)
    metadata = Column(JSONB, default=dict, nullable=False)
    processing_time_ms = Column(Integer, nullable=True)
    
    # Processing metadata
    processed_by_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True
    )
    processed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Cache and usage tracking
    expires_at = Column(DateTime(timezone=True), nullable=True)
    cache_key = Column(String(255), nullable=True, unique=True)
    is_auto_generated = Column(Boolean, default=True, nullable=False)
    is_default = Column(Boolean, default=False, nullable=False, index=True)
    usage_count = Column(Integer, default=0, nullable=False)
    last_accessed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Error tracking
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)
    
    # Relationships
    original_image = relationship(
        "Image", 
        back_populates="variants"
    )
    processed_by = relationship("User")
    
    # Check constraints
    __table_args__ = (
        CheckConstraint(
            'width IS NULL OR width > 0', 
            name='check_variant_width_positive'
        ),
        CheckConstraint(
            'height IS NULL OR height > 0', 
            name='check_variant_height_positive'
        ),
        CheckConstraint(
            'quality IS NULL OR (quality >= 1 AND quality <= 100)', 
            name='check_quality_range'
        ),
    )
    
    def __repr__(self) -> str:
        return f"<ImageVariant(id={self.id}, type={self.variant_type.value}, {self.width}x{self.height})>"
    
    @property
    def dimensions(self) -> Optional[tuple]:
        """Get dimensions as tuple."""
        if self.width and self.height:
            return (self.width, self.height)
        return None
    
    @property
    def aspect_ratio(self) -> Optional[float]:
        """Calculate aspect ratio."""
        if self.width and self.height and self.height > 0:
            return round(self.width / self.height, 3)
        return None
    
    @property
    def is_ready(self) -> bool:
        """Check if variant is ready for use."""
        return (
            self.status == VariantStatus.COMPLETED and 
            self.storage_url is not None and
            not self.is_expired
        )
    
    @property
    def is_expired(self) -> bool:
        """Check if variant cache has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    @property
    def public_url(self) -> Optional[str]:
        """Get public URL."""
        return self.cdn_url or self.storage_url
    
    def record_usage(self) -> None:
        """Record that variant was accessed."""
        self.usage_count += 1
        self.last_accessed_at = datetime.utcnow()
    
    def mark_as_processed(
        self, 
        file_size: int, 
        file_hash: str, 
        storage_url: str,
        processing_time_ms: int,
        processed_by: Optional['User'] = None
    ) -> None:
        """Mark variant as successfully processed."""
        self.status = VariantStatus.COMPLETED
        self.file_size = file_size
        self.file_hash = file_hash
        self.storage_url = storage_url
        self.processing_time_ms = processing_time_ms
        self.processed_at = datetime.utcnow()
        self.processed_by_id = processed_by.id if processed_by else None
        
        # Generate cache key
        if not self.cache_key:
            self.cache_key = self.generate_cache_key()
    
    def generate_cache_key(self) -> str:
        """Generate a unique cache key."""
        components = [
            "variant",
            str(self.original_image_id),
            self.variant_type.value,
            f"{self.width}x{self.height}" if self.width and self.height else "unknown",
            self.format.value,
            f"q{self.quality}" if self.quality else "qdefault",
        ]
        
        params_hash = hashlib.md5(
            str(self.processing_parameters).encode()
        ).hexdigest()[:8]
        components.append(params_hash)
        
        return "_".join(components)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert variant to dictionary."""
        return {
            "id": str(self.id),
            "original_image_id": str(self.original_image_id),
            "variant_type": self.variant_type.value,
            "status": self.status.value,
            "width": self.width,
            "height": self.height,
            "dimensions": self.dimensions,
            "aspect_ratio": self.aspect_ratio,
            "format": self.format.value,
            "quality": self.quality,
            "file_size": self.file_size,
            "storage_url": self.storage_url,
            "cdn_url": self.cdn_url,
            "public_url": self.public_url,
            "is_ready": self.is_ready,
            "is_expired": self.is_expired,
            "is_default": self.is_default,
            "usage_count": self.usage_count,
            "processing_time_ms": self.processing_time_ms,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "cache_key": self.cache_key,
        }
    
    @classmethod
    def create(
        cls,
        original_image_id: uuid.UUID,
        variant_type: VariantType,
        width: Optional[int] = None,
        height: Optional[int] = None,
        format: ImageFormat = ImageFormat.JPEG,
        quality: Optional[int] = None,
        storage_backend: StorageBackend = StorageBackend.LOCAL,
        processing_parameters: Optional[Dict[str, Any]] = None,
        is_default: bool = False
    ) -> 'ImageVariant':
        """Factory method to create a new image variant."""
        # Set default quality
        if quality is None:
            if format in [ImageFormat.JPEG, ImageFormat.JPG, ImageFormat.WEBP]:
                quality = 85
            else:
                quality = 95
        
        # Generate storage path
        storage_path = cls.generate_storage_path(
            original_image_id=original_image_id,
            variant_type=variant_type,
            width=width,
            height=height,
            format=format
        )
        
        # Create variant
        variant = cls(
            original_image_id=original_image_id,
            variant_type=variant_type,
            width=width,
            height=height,
            format=format,
            quality=quality,
            storage_backend=storage_backend,
            storage_path=storage_path,
            processing_parameters=processing_parameters or {},
            is_default=is_default,
            status=VariantStatus.PENDING
        )
        
        return variant
    
    @staticmethod
    def generate_storage_path(
        original_image_id: uuid.UUID,
        variant_type: VariantType,
        width: Optional[int] = None,
        height: Optional[int] = None,
        format: ImageFormat = ImageFormat.JPEG
    ) -> str:
        """Generate storage path for variant."""
        base_path = f"variants/{original_image_id}/{variant_type.value}"
        
        if width and height:
            base_path += f"/{width}x{height}"
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{base_path}/{timestamp}.{format.value}"