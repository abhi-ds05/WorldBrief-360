"""
image_enums.py - Shared Enums for Image Models
"""

from enum import Enum


class ImageFormat(Enum):
    """Supported image formats."""
    JPEG = "jpeg"
    JPG = "jpg"
    PNG = "png"
    GIF = "gif"
    WEBP = "webp"
    SVG = "svg"
    BMP = "bmp"
    TIFF = "tiff"
    HEIC = "heic"
    AVIF = "avif"
    ICO = "ico"
    RAW = "raw"


class ImageType(Enum):
    """Types/categories of images."""
    PROFILE = "profile"
    COVER = "cover"
    CONTENT = "content"
    THUMBNAIL = "thumbnail"
    BANNER = "banner"
    ICON = "icon"
    LOGO = "logo"
    SCREENSHOT = "screenshot"
    PHOTO = "photo"
    ILLUSTRATION = "illustration"
    DIAGRAM = "diagram"
    OTHER = "other"


class StorageBackend(Enum):
    """Storage backends for images."""
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"
    CLOUDFLARE = "cloudflare"
    CDN = "cdn"
    DATABASE = "database"


class ImageStatus(Enum):
    """Image processing status."""
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"
    MODERATION_PENDING = "moderation_pending"
    MODERATION_APPROVED = "moderation_approved"
    MODERATION_REJECTED = "moderation_rejected"
    DELETED = "deleted"