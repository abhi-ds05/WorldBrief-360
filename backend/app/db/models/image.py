"""
image.py - Image Model for Media Management
"""

import uuid
import hashlib
import mimetypes
import json
import io
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, TYPE_CHECKING, Tuple, BinaryIO
from enum import Enum
from sqlalchemy import (
    Column, String, Text, ForeignKey, Integer, DateTime, 
    Boolean, Enum as SQLEnum, JSON, Float, LargeBinary,
    BigInteger, CheckConstraint, ARRAY
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.sql import func
from PIL import Image as PILImage
import PIL.ExifTags

from db.base import Base
from models.mixins import TimestampMixin, SoftDeleteMixin, UUIDMixin, OwnableMixin

# Import enums that are shared
from .image_enums import ImageFormat, StorageBackend, ImageType, ImageStatus

if TYPE_CHECKING:
    from models.user import User
    from models.article import Article
    from models.incident import Incident
    from models.comment import Comment
    from models.image_variant import ImageVariant


class Image(Base, UUIDMixin, TimestampMixin, SoftDeleteMixin, OwnableMixin):
    """
    Image model for managing image files and their metadata.
    """
    
    __tablename__ = "images"
    
    # File information
    filename = Column(String(255), nullable=False, index=True)
    original_filename = Column(String(255), nullable=True)
    file_extension = Column(String(10), nullable=False, index=True)
    mime_type = Column(String(100), nullable=False)
    format = Column(SQLEnum(ImageFormat), nullable=False, index=True)
    image_type = Column(SQLEnum(ImageType), default=ImageType.CONTENT, nullable=False, index=True)
    
    # Storage information
    storage_backend = Column(
        SQLEnum(StorageBackend),
        default=StorageBackend.LOCAL,
        nullable=False,
        index=True
    )
    status = Column(
        SQLEnum(ImageStatus),
        default=ImageStatus.UPLOADING,
        nullable=False,
        index=True
    )
    
    # File properties
    file_size = Column(BigInteger, nullable=False, index=True)
    file_hash = Column(String(64), nullable=False, unique=True, index=True)
    
    # Image properties
    width = Column(Integer, nullable=True, index=True)
    height = Column(Integer, nullable=True, index=True)
    aspect_ratio = Column(Float, nullable=True, index=True)
    color_mode = Column(String(20), nullable=True)
    dpi = Column(Integer, nullable=True)
    is_animated = Column(Boolean, default=False, nullable=False)
    frame_count = Column(Integer, default=1, nullable=False)
    duration_ms = Column(Integer, nullable=True)
    
    # Storage paths and URLs
    storage_path = Column(String(1024), nullable=False, index=True)
    storage_url = Column(String(2048), nullable=True)
    cdn_url = Column(String(2048), nullable=True, index=True)
    thumbnail_url = Column(String(2048), nullable=True)
    optimized_url = Column(String(2048), nullable=True)
    
    # Metadata
    metadata = Column(MutableDict.as_mutable(JSONB), default=dict, nullable=False)
    processing_metadata = Column(MutableDict.as_mutable(JSONB), default=dict, nullable=False)
    exif_data = Column(JSONB, nullable=True)
    
    # Content information
    alt_text = Column(String(500), nullable=True)
    caption = Column(Text, nullable=True)
    copyright = Column(String(255), nullable=True)
    attribution = Column(Text, nullable=True)
    tags = Column(ARRAY(String), default=[], nullable=False, index=True)
    
    # Moderation and safety
    moderation_score = Column(Float, nullable=True, index=True)
    moderation_labels = Column(ARRAY(String), default=[], nullable=False)
    is_sensitive = Column(Boolean, default=False, nullable=False, index=True)
    is_approved = Column(Boolean, default=False, nullable=False, index=True)
    nsfw_score = Column(Float, nullable=True, index=True)
    
    # Processing timestamps
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    processed_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # Foreign keys
    uploaded_by_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    article_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("articles.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    incident_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("incidents.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    comment_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("comments.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    
    # Relationships
    uploaded_by = relationship("User", back_populates="uploaded_images")
    article = relationship("Article", back_populates="images")
    incident = relationship("Incident", back_populates="images")
    comment = relationship("Comment", back_populates="images")
    
    # Variants relationship - NOTE: This uses string reference to avoid circular import
    variants = relationship(
        "ImageVariant",
        back_populates="original_image",
        foreign_keys="[ImageVariant.original_image_id]",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    # Check constraints
    __table_args__ = (
        CheckConstraint('file_size > 0', name='check_file_size_positive'),
        CheckConstraint('width IS NULL OR width > 0', name='check_width_positive'),
        CheckConstraint('height IS NULL OR height > 0', name='check_height_positive'),
    )
    
    def __repr__(self) -> str:
        return f"<Image(id={self.id}, filename='{self.filename}', size={self.file_size})>"
    
    @validates('filename')
    def validate_filename(self, key: str, filename: str) -> str:
        """Validate and sanitize filename."""
        import re
        filename = Path(filename).name
        filename = re.sub(r'[^\w\-\.]', '_', filename)
        if not filename:
            filename = f"image_{uuid.uuid4().hex[:8]}"
        return filename
    
    @property
    def is_expired(self) -> bool:
        """Check if image has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    @property
    def is_processed(self) -> bool:
        """Check if image processing is complete."""
        return self.status == ImageStatus.READY and self.processed_at is not None
    
    @property
    def public_url(self) -> Optional[str]:
        """Get the public URL for the image."""
        return self.cdn_url or self.storage_url or self.thumbnail_url
    
    @property
    def dimensions(self) -> Optional[Tuple[int, int]]:
        """Get image dimensions."""
        if self.width and self.height:
            return (self.width, self.height)
        return None
    
    @property
    def megapixels(self) -> Optional[float]:
        """Get image resolution in megapixels."""
        if self.width and self.height:
            return (self.width * self.height) / 1_000_000
        return None
    
    @property
    def file_size_mb(self) -> float:
        """Get file size in megabytes."""
        return self.file_size / (1024 * 1024)
    
    def calculate_aspect_ratio(self) -> Optional[float]:
        """Calculate and set aspect ratio."""
        if self.width and self.height and self.height > 0:
            self.aspect_ratio = round(self.width / self.height, 3)
            return self.aspect_ratio
        return None
    
    def extract_metadata_from_file(self, file_data: BinaryIO) -> Dict[str, Any]:
        """Extract metadata from image file."""
        metadata = {}
        
        try:
            file_data.seek(0)
            with PILImage.open(file_data) as img:
                metadata.update({
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                })
                
                # Set image properties
                self.width, self.height = img.size
                self.color_mode = img.mode
                self.calculate_aspect_ratio()
                
        except Exception as e:
            metadata['error'] = str(e)
        
        return metadata
    
    def generate_file_hash(self, file_data: BinaryIO) -> str:
        """Generate SHA256 hash of file data."""
        file_data.seek(0)
        sha256_hash = hashlib.sha256()
        chunk_size = 8192
        while chunk := file_data.read(chunk_size):
            sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def create_thumbnail(
        self,
        width: int = 300,
        height: int = 300,
        quality: int = 85,
        format: ImageFormat = ImageFormat.JPEG
    ):
        """
        Create a thumbnail variant of this image.
        
        Args:
            width: Thumbnail width
            height: Thumbnail height
            quality: JPEG/WebP quality (1-100)
            format: Output format
            
        Returns:
            ImageVariant instance
        """
        # Import here to avoid circular imports
        from models.image_variant import ImageVariant, VariantType
        
        # Create thumbnail variant
        thumbnail = ImageVariant.create(
            original_image_id=self.id,
            variant_type=VariantType.THUMBNAIL,
            width=width,
            height=height,
            format=format,
            quality=quality,
            storage_backend=self.storage_backend
        )
        
        return thumbnail
    
    def mark_as_processed(self) -> None:
        """Mark image as processed and ready."""
        self.status = ImageStatus.READY
        self.processed_at = datetime.utcnow()
        self.is_approved = True
    
    def to_dict(self, include_variants: bool = False) -> Dict[str, Any]:
        """Convert image to dictionary."""
        result = {
            "id": str(self.id),
            "filename": self.filename,
            "original_filename": self.original_filename,
            "file_extension": self.file_extension,
            "mime_type": self.mime_type,
            "format": self.format.value,
            "image_type": self.image_type.value,
            "storage_backend": self.storage_backend.value,
            "status": self.status.value,
            "file_size": self.file_size,
            "file_size_mb": round(self.file_size_mb, 2),
            "file_hash": self.file_hash,
            "width": self.width,
            "height": self.height,
            "dimensions": self.dimensions,
            "aspect_ratio": self.aspect_ratio,
            "storage_url": self.storage_url,
            "cdn_url": self.cdn_url,
            "public_url": self.public_url,
            "alt_text": self.alt_text,
            "caption": self.caption,
            "tags": self.tags,
            "is_approved": self.is_approved,
            "is_sensitive": self.is_sensitive,
            "uploaded_at": self.uploaded_at.isoformat() if self.uploaded_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "uploaded_by_id": str(self.uploaded_by_id) if self.uploaded_by_id else None,
        }
        
        if include_variants:
            result["variants"] = [
                variant.to_dict()
                for variant in self.variants
                if not variant.is_deleted
            ]
        
        return result
    
    @classmethod
    def create_from_upload(
        cls,
        file_data: BinaryIO,
        filename: str,
        uploaded_by_id: Optional[uuid.UUID] = None,
        image_type: ImageType = ImageType.CONTENT,
        alt_text: Optional[str] = None,
        tags: Optional[List[str]] = None,
        storage_backend: StorageBackend = StorageBackend.LOCAL,
        owner_id: Optional[uuid.UUID] = None,
        is_public: bool = True
    ) -> 'Image':
        """Factory method to create an image from uploaded file."""
        # Get file size
        file_data.seek(0, 2)
        file_size = file_data.tell()
        file_data.seek(0)
        
        # Determine file extension and format
        file_extension = Path(filename).suffix.lower().lstrip('.') or 'bin'
        mime_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
        
        # Determine format
        format_str = file_extension.upper()
        try:
            image_format = ImageFormat(format_str)
        except ValueError:
            mime_to_format = {
                'image/jpeg': ImageFormat.JPEG,
                'image/jpg': ImageFormat.JPG,
                'image/png': ImageFormat.PNG,
                'image/gif': ImageFormat.GIF,
                'image/webp': ImageFormat.WEBP,
            }
            image_format = mime_to_format.get(mime_type, ImageFormat.JPEG)
        
        # Generate unique filename and hash
        unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
        file_hash = hashlib.sha256(file_data.read()).hexdigest()
        file_data.seek(0)
        
        # Create storage path
        now = datetime.utcnow()
        storage_path = f"{now.year}/{now.month:02d}/{now.day:02d}/{unique_filename}"
        
        # Create image
        image = cls(
            filename=unique_filename,
            original_filename=filename,
            file_extension=file_extension,
            mime_type=mime_type,
            format=image_format,
            image_type=image_type,
            storage_backend=storage_backend,
            status=ImageStatus.UPLOADING,
            file_size=file_size,
            file_hash=file_hash,
            storage_path=storage_path,
            alt_text=alt_text,
            tags=tags or [],
            uploaded_by_id=uploaded_by_id,
            owner_id=owner_id,
            is_public=is_public
        )
        
        # Extract metadata
        metadata = image.extract_metadata_from_file(file_data)
        image.metadata.update(metadata)
        
        return image