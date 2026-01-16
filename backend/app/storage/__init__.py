"""
Storage Abstraction Layer for WorldBrief 360

This module provides a unified interface for file storage with support for:
- Local filesystem storage
- AWS S3 storage
- Google Cloud Storage
- Azure Blob Storage
- CDN integration
- File operations with encryption
- Versioning and lifecycle management
- Backup and disaster recovery

Features:
- Storage provider abstraction
- Automatic file type detection
- File metadata management
- Streaming support for large files
- Access control and permissions
- File integrity verification
- Cost optimization
- Monitoring and logging
"""

import os
import mimetypes
import hashlib
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union, Callable
from contextlib import contextmanager
from abc import ABC, abstractmethod

from app.core.config import get_settings
from app.security.audit_logger import AuditLogger, AuditEventType, AuditSeverity

# Get settings
settings = get_settings()

# Storage module version
__version__ = "1.0.0"
__author__ = "WorldBrief 360 Storage Team"


class StorageProvider(str, Enum):
    """Supported storage providers."""
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"
    MINIO = "minio"
    CEPH = "ceph"


class FileType(str, Enum):
    """File type categories."""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    ARCHIVE = "archive"
    CODE = "code"
    DATA = "data"
    TEXT = "text"
    BINARY = "binary"
    UNKNOWN = "unknown"


class AccessLevel(str, Enum):
    """File access levels."""
    PRIVATE = "private"
    PUBLIC_READ = "public-read"
    PUBLIC_READ_WRITE = "public-read-write"
    AUTHENTICATED_READ = "authenticated-read"
    BUCKET_OWNER_READ = "bucket-owner-read"
    BUCKET_OWNER_FULL_CONTROL = "bucket-owner-full-control"


class StorageEvent(str, Enum):
    """Storage events for monitoring."""
    FILE_UPLOADED = "file.uploaded"
    FILE_DOWNLOADED = "file.downloaded"
    FILE_DELETED = "file.deleted"
    FILE_COPIED = "file.copied"
    FILE_MOVED = "file.moved"
    STORAGE_CREATED = "storage.created"
    STORAGE_DELETED = "storage.deleted"
    QUOTA_EXCEEDED = "quota.exceeded"
    INTEGRITY_CHECK_FAILED = "integrity.check.failed"


@dataclass
class FileMetadata:
    """File metadata information."""
    filename: str
    filepath: str
    size: int
    content_type: str
    file_type: FileType
    md5_hash: str
    sha256_hash: str
    created_at: datetime
    modified_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    access_level: AccessLevel = AccessLevel.PRIVATE
    version_id: Optional[str] = None
    storage_class: Optional[str] = None
    expires_at: Optional[datetime] = None
    etag: Optional[str] = None
    encryption: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "filename": self.filename,
            "filepath": self.filepath,
            "size": self.size,
            "content_type": self.content_type,
            "file_type": self.file_type.value,
            "md5_hash": self.md5_hash,
            "sha256_hash": self.sha256_hash,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "metadata": self.metadata,
            "tags": self.tags,
            "access_level": self.access_level.value,
            "version_id": self.version_id,
            "storage_class": self.storage_class,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "etag": self.etag,
            "encryption": self.encryption,
        }


@dataclass
class StorageStats:
    """Storage statistics."""
    total_files: int
    total_size: int
    available_space: int
    used_space_percentage: float
    file_count_by_type: Dict[FileType, int]
    size_by_type: Dict[FileType, int]
    oldest_file: Optional[datetime]
    newest_file: Optional[datetime]
    cost_estimate: Optional[float] = None
    last_backup: Optional[datetime] = None


@dataclass
class StorageConfig:
    """Storage configuration."""
    provider: StorageProvider = StorageProvider.LOCAL
    bucket_name: Optional[str] = None
    base_path: str = "uploads"
    region: Optional[str] = None
    endpoint_url: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    use_ssl: bool = True
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_file_types: List[str] = field(default_factory=list)
    blocked_file_types: List[str] = field(default_factory=list)
    enable_encryption: bool = True
    encryption_key: Optional[str] = None
    enable_versioning: bool = False
    enable_lifecycle: bool = False
    lifecycle_days: int = 30
    cdn_enabled: bool = False
    cdn_url: Optional[str] = None
    audit_logging: bool = True
    compression: bool = False
    
    def __post_init__(self):
        """Set default allowed file types if not specified."""
        if not self.allowed_file_types:
            self.allowed_file_types = [
                # Images
                '.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.bmp', '.tiff',
                # Documents
                '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.txt',
                '.md', '.rtf', '.odt', '.ods', '.odp',
                # Archives
                '.zip', '.tar', '.gz', '.7z', '.rar',
                # Audio
                '.mp3', '.wav', '.ogg', '.flac', '.m4a',
                # Video
                '.mp4', '.webm', '.mov', '.avi', '.mkv',
                # Data
                '.json', '.csv', '.xml', '.yaml', '.yml',
            ]
        
        if not self.blocked_file_types:
            self.blocked_file_types = [
                '.exe', '.bat', '.cmd', '.sh', '.py', '.php', '.js', '.html',
                '.jar', '.war', '.ear', '.dll', '.so', '.bin'
            ]


# Import storage implementations
try:
    from .local_storage import LocalStorage
except ImportError:
    LocalStorage = None

try:
    from .s3_storage import S3Storage
except ImportError:
    S3Storage = None

try:
    from .gcs_storage import GCSStorage
except ImportError:
    GCSStorage = None

try:
    from .azure_storage import AzureStorage
except ImportError:
    AzureStorage = None

try:
    from .cdn_manager import CDNManager
except ImportError:
    CDNManager = None


class StorageError(Exception):
    """Base storage error."""
    pass


class FileNotFoundError(StorageError):
    """File not found error."""
    pass


class FileTooLargeError(StorageError):
    """File too large error."""
    pass


class FileTypeNotAllowedError(StorageError):
    """File type not allowed error."""
    pass


class StorageQuotaExceededError(StorageError):
    """Storage quota exceeded error."""
    pass


class StorageIntegrityError(StorageError):
    """Storage integrity error."""
    pass


# Base Storage Interface
class BaseStorage(ABC):
    """Abstract base class for storage providers."""
    
    def __init__(self, config: StorageConfig, audit_logger: Optional[AuditLogger] = None):
        self.config = config
        self.audit_logger = audit_logger or AuditLogger()
        self._validate_config()
    
    def _validate_config(self):
        """Validate storage configuration."""
        if not self.config.base_path:
            raise ValueError("base_path must be specified")
        
        if self.config.max_file_size <= 0:
            raise ValueError("max_file_size must be positive")
    
    @abstractmethod
    async def upload_file(
        self,
        file_obj: BinaryIO,
        filename: str,
        filepath: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        access_level: AccessLevel = AccessLevel.PRIVATE,
        content_type: Optional[str] = None
    ) -> FileMetadata:
        """Upload a file."""
        pass
    
    @abstractmethod
    async def download_file(
        self,
        filepath: str,
        version_id: Optional[str] = None
    ) -> BinaryIO:
        """Download a file."""
        pass
    
    @abstractmethod
    async def delete_file(
        self,
        filepath: str,
        version_id: Optional[str] = None
    ) -> bool:
        """Delete a file."""
        pass
    
    @abstractmethod
    async def file_exists(
        self,
        filepath: str,
        version_id: Optional[str] = None
    ) -> bool:
        """Check if file exists."""
        pass
    
    @abstractmethod
    async def get_file_metadata(
        self,
        filepath: str,
        version_id: Optional[str] = None
    ) -> Optional[FileMetadata]:
        """Get file metadata."""
        pass
    
    @abstractmethod
    async def list_files(
        self,
        prefix: Optional[str] = None,
        recursive: bool = True,
        max_keys: int = 1000
    ) -> List[FileMetadata]:
        """List files."""
        pass
    
    @abstractmethod
    async def copy_file(
        self,
        source_path: str,
        dest_path: str,
        source_version_id: Optional[str] = None
    ) -> FileMetadata:
        """Copy a file."""
        pass
    
    @abstractmethod
    async def move_file(
        self,
        source_path: str,
        dest_path: str,
        source_version_id: Optional[str] = None
    ) -> FileMetadata:
        """Move a file."""
        pass
    
    @abstractmethod
    async def get_presigned_url(
        self,
        filepath: str,
        expires_in: int = 3600,
        version_id: Optional[str] = None
    ) -> str:
        """Get presigned URL for temporary access."""
        pass
    
    @abstractmethod
    async def get_storage_stats(self) -> StorageStats:
        """Get storage statistics."""
        pass
    
    # Common utility methods
    def _validate_file_type(self, filename: str) -> Tuple[FileType, str]:
        """
        Validate file type and get content type.
        
        Args:
            filename: Name of the file
            
        Returns:
            Tuple of (file_type, content_type)
            
        Raises:
            FileTypeNotAllowedError: If file type is not allowed
        """
        # Get file extension
        ext = Path(filename).suffix.lower()
        
        # Check blocked file types
        if ext in self.config.blocked_file_types:
            raise FileTypeNotAllowedError(
                f"File type {ext} is not allowed"
            )
        
        # Check allowed file types
        if (self.config.allowed_file_types and 
            ext not in self.config.allowed_file_types):
            raise FileTypeNotAllowedError(
                f"File type {ext} is not in allowed list"
            )
        
        # Get content type
        content_type, _ = mimetypes.guess_type(filename)
        if not content_type:
            content_type = 'application/octet-stream'
        
        # Determine file type category
        if content_type.startswith('image/'):
            file_type = FileType.IMAGE
        elif content_type.startswith('video/'):
            file_type = FileType.VIDEO
        elif content_type.startswith('audio/'):
            file_type = FileType.AUDIO
        elif content_type in [
            'application/pdf',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.ms-powerpoint',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            'text/plain',
            'text/markdown',
            'text/rtf',
        ]:
            file_type = FileType.DOCUMENT
        elif content_type in [
            'application/zip',
            'application/x-tar',
            'application/x-gzip',
            'application/x-7z-compressed',
            'application/x-rar-compressed',
        ]:
            file_type = FileType.ARCHIVE
        elif content_type.startswith('text/') or content_type in [
            'application/json',
            'application/xml',
            'application/yaml',
        ]:
            file_type = FileType.TEXT
        elif content_type in [
            'application/octet-stream',
            'application/x-binary',
        ]:
            file_type = FileType.BINARY
        else:
            file_type = FileType.UNKNOWN
        
        return file_type, content_type
    
    def _calculate_hashes(self, file_obj: BinaryIO) -> Tuple[str, str]:
        """
        Calculate MD5 and SHA256 hashes of file.
        
        Args:
            file_obj: File object
            
        Returns:
            Tuple of (md5_hash, sha256_hash)
        """
        # Save current position
        current_pos = file_obj.tell()
        file_obj.seek(0)
        
        md5_hash = hashlib.md5()
        sha256_hash = hashlib.sha256()
        
        # Read in chunks
        chunk_size = 8192
        while chunk := file_obj.read(chunk_size):
            md5_hash.update(chunk)
            sha256_hash.update(chunk)
        
        # Restore position
        file_obj.seek(current_pos)
        
        return md5_hash.hexdigest(), sha256_hash.hexdigest()
    
    def _log_storage_event(
        self,
        event_type: StorageEvent,
        details: Dict[str, Any],
        severity: AuditSeverity = AuditSeverity.INFO
    ):
        """Log storage event."""
        if self.config.audit_logging:
            self.audit_logger.log_security_event(
                event_type=AuditEventType.DATA_ACCESSED,
                description=f"Storage event: {event_type.value}",
                details=details,
                severity=severity
            )
    
    def _get_full_path(self, filepath: str) -> str:
        """Get full path including base path."""
        if filepath.startswith('/'):
            filepath = filepath[1:]
        
        return str(Path(self.config.base_path) / filepath)
    
    def _get_relative_path(self, full_path: str) -> str:
        """Get relative path from full path."""
        base_path = str(self.config.base_path)
        if full_path.startswith(base_path):
            return full_path[len(base_path):].lstrip('/')
        return full_path


# Storage Factory
class StorageFactory:
    """Factory for creating storage instances."""
    
    @staticmethod
    def create_storage(
        config: Optional[StorageConfig] = None,
        provider: Optional[StorageProvider] = None
    ) -> BaseStorage:
        """
        Create storage instance based on configuration.
        
        Args:
            config: Storage configuration
            provider: Storage provider (overrides config.provider)
            
        Returns:
            Storage instance
            
        Raises:
            ValueError: If provider is not supported
        """
        if config is None:
            # Load from settings
            config = StorageFactory.get_default_config()
        
        if provider:
            config.provider = provider
        
        # Create storage instance based on provider
        if config.provider == StorageProvider.LOCAL:
            if LocalStorage is None:
                raise ImportError("LocalStorage not available")
            return LocalStorage(config)
        
        elif config.provider == StorageProvider.S3:
            if S3Storage is None:
                raise ImportError("S3Storage not available")
            return S3Storage(config)
        
        elif config.provider == StorageProvider.GCS:
            if GCSStorage is None:
                raise ImportError("GCSStorage not available")
            return GCSStorage(config)
        
        elif config.provider == StorageProvider.AZURE:
            if AzureStorage is None:
                raise ImportError("AzureStorage not available")
            return AzureStorage(config)
        
        elif config.provider == StorageProvider.MINIO:
            # Minio uses S3-compatible API
            if S3Storage is None:
                raise ImportError("S3Storage not available")
            return S3Storage(config)
        
        elif config.provider == StorageProvider.CEPH:
            # Ceph uses S3-compatible API
            if S3Storage is None:
                raise ImportError("S3Storage not available")
            return S3Storage(config)
        
        else:
            raise ValueError(f"Unsupported storage provider: {config.provider}")
    
    @staticmethod
    def get_default_config() -> StorageConfig:
        """Get default storage configuration from settings."""
        provider = getattr(settings, 'STORAGE_PROVIDER', 'local')
        
        # Map string to enum
        provider_map = {
            'local': StorageProvider.LOCAL,
            's3': StorageProvider.S3,
            'gcs': StorageProvider.GCS,
            'azure': StorageProvider.AZURE,
            'minio': StorageProvider.MINIO,
            'ceph': StorageProvider.CEPH,
        }
        
        storage_provider = provider_map.get(provider.lower(), StorageProvider.LOCAL)
        
        return StorageConfig(
            provider=storage_provider,
            bucket_name=getattr(settings, 'STORAGE_BUCKET_NAME', None),
            base_path=getattr(settings, 'STORAGE_BASE_PATH', 'uploads'),
            region=getattr(settings, 'STORAGE_REGION', None),
            endpoint_url=getattr(settings, 'STORAGE_ENDPOINT_URL', None),
            access_key=getattr(settings, 'STORAGE_ACCESS_KEY', None),
            secret_key=getattr(settings, 'STORAGE_SECRET_KEY', None),
            use_ssl=getattr(settings, 'STORAGE_USE_SSL', True),
            max_file_size=getattr(settings, 'STORAGE_MAX_FILE_SIZE', 100 * 1024 * 1024),
            allowed_file_types=getattr(settings, 'STORAGE_ALLOWED_FILE_TYPES', []),
            blocked_file_types=getattr(settings, 'STORAGE_BLOCKED_FILE_TYPES', []),
            enable_encryption=getattr(settings, 'STORAGE_ENABLE_ENCRYPTION', True),
            encryption_key=getattr(settings, 'STORAGE_ENCRYPTION_KEY', None),
            enable_versioning=getattr(settings, 'STORAGE_ENABLE_VERSIONING', False),
            enable_lifecycle=getattr(settings, 'STORAGE_ENABLE_LIFECYCLE', False),
            lifecycle_days=getattr(settings, 'STORAGE_LIFECYCLE_DAYS', 30),
            cdn_enabled=getattr(settings, 'STORAGE_CDN_ENABLED', False),
            cdn_url=getattr(settings, 'STORAGE_CDN_URL', None),
            audit_logging=getattr(settings, 'STORAGE_AUDIT_LOGGING', True),
            compression=getattr(settings, 'STORAGE_COMPRESSION', False),
        )


# File Operations Manager
class FileManager:
    """High-level file operations manager."""
    
    def __init__(
        self,
        storage: Optional[BaseStorage] = None,
        config: Optional[StorageConfig] = None
    ):
        self.storage = storage or StorageFactory.create_storage(config)
        self.cdn_manager = None
        
        # Initialize CDN manager if enabled
        if self.storage.config.cdn_enabled and CDNManager is not None:
            self.cdn_manager = CDNManager(self.storage.config)
    
    async def upload(
        self,
        file_obj: BinaryIO,
        filename: str,
        filepath: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        access_level: AccessLevel = AccessLevel.PRIVATE,
        content_type: Optional[str] = None,
        user_id: Optional[str] = None,
        user_ip: Optional[str] = None
    ) -> FileMetadata:
        """
        Upload a file with validation and logging.
        
        Args:
            file_obj: File object
            filename: Name of the file
            filepath: Path to store the file
            metadata: Additional metadata
            tags: File tags
            access_level: Access level
            content_type: Content type
            user_id: User ID for auditing
            user_ip: User IP for auditing
            
        Returns:
            File metadata
        """
        # Check file size
        current_pos = file_obj.tell()
        file_obj.seek(0, 2)  # Seek to end
        file_size = file_obj.tell()
        file_obj.seek(current_pos)  # Restore position
        
        if file_size > self.storage.config.max_file_size:
            raise FileTooLargeError(
                f"File size {file_size} exceeds maximum {self.storage.config.max_file_size}"
            )
        
        # Validate file type
        file_type, detected_content_type = self.storage._validate_file_type(filename)
        if content_type is None:
            content_type = detected_content_type
        
        # Upload file
        file_metadata = await self.storage.upload_file(
            file_obj=file_obj,
            filename=filename,
            filepath=filepath,
            metadata=metadata or {},
            tags=tags or [],
            access_level=access_level,
            content_type=content_type
        )
        
        # Log event
        self.storage._log_storage_event(
            event_type=StorageEvent.FILE_UPLOADED,
            details={
                "filename": filename,
                "filepath": file_metadata.filepath,
                "size": file_size,
                "file_type": file_type.value,
                "content_type": content_type,
                "user_id": user_id,
                "user_ip": user_ip,
                "access_level": access_level.value,
            }
        )
        
        return file_metadata
    
    async def download(
        self,
        filepath: str,
        version_id: Optional[str] = None,
        user_id: Optional[str] = None,
        user_ip: Optional[str] = None
    ) -> BinaryIO:
        """
        Download a file with logging.
        
        Args:
            filepath: Path to the file
            version_id: Version ID (for versioned storage)
            user_id: User ID for auditing
            user_ip: User IP for auditing
            
        Returns:
            File object
            
        Raises:
            FileNotFoundError: If file does not exist
        """
        # Check if file exists
        if not await self.storage.file_exists(filepath, version_id):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Download file
        file_obj = await self.storage.download_file(filepath, version_id)
        
        # Get file metadata
        metadata = await self.storage.get_file_metadata(filepath, version_id)
        
        # Log event
        if metadata:
            self.storage._log_storage_event(
                event_type=StorageEvent.FILE_DOWNLOADED,
                details={
                    "filepath": filepath,
                    "filename": metadata.filename,
                    "size": metadata.size,
                    "file_type": metadata.file_type.value,
                    "user_id": user_id,
                    "user_ip": user_ip,
                    "version_id": version_id,
                }
            )
        
        return file_obj
    
    async def delete(
        self,
        filepath: str,
        version_id: Optional[str] = None,
        user_id: Optional[str] = None,
        user_ip: Optional[str] = None
    ) -> bool:
        """
        Delete a file with logging.
        
        Args:
            filepath: Path to the file
            version_id: Version ID (for versioned storage)
            user_id: User ID for auditing
            user_ip: User IP for auditing
            
        Returns:
            True if deleted successfully
        """
        # Get file metadata before deletion
        metadata = await self.storage.get_file_metadata(filepath, version_id)
        
        # Delete file
        result = await self.storage.delete_file(filepath, version_id)
        
        # Log event
        if metadata:
            self.storage._log_storage_event(
                event_type=StorageEvent.FILE_DELETED,
                details={
                    "filepath": filepath,
                    "filename": metadata.filename,
                    "size": metadata.size,
                    "file_type": metadata.file_type.value,
                    "user_id": user_id,
                    "user_ip": user_ip,
                    "version_id": version_id,
                }
            )
        
        return result
    
    async def get_url(
        self,
        filepath: str,
        expires_in: int = 3600,
        version_id: Optional[str] = None,
        use_cdn: bool = True
    ) -> str:
        """
        Get URL for file access.
        
        Args:
            filepath: Path to the file
            expires_in: URL expiration in seconds
            version_id: Version ID (for versioned storage)
            use_cdn: Whether to use CDN URL
            
        Returns:
            File URL
        """
        # Check if CDN should be used
        if use_cdn and self.cdn_manager:
            url = await self.cdn_manager.get_cdn_url(filepath)
            if url:
                return url
        
        # Get presigned URL from storage
        return await self.storage.get_presigned_url(filepath, expires_in, version_id)
    
    async def get_metadata(
        self,
        filepath: str,
        version_id: Optional[str] = None
    ) -> Optional[FileMetadata]:
        """Get file metadata."""
        return await self.storage.get_file_metadata(filepath, version_id)
    
    async def list(
        self,
        prefix: Optional[str] = None,
        recursive: bool = True,
        max_keys: int = 1000
    ) -> List[FileMetadata]:
        """List files."""
        return await self.storage.list_files(prefix, recursive, max_keys)
    
    async def copy(
        self,
        source_path: str,
        dest_path: str,
        source_version_id: Optional[str] = None,
        user_id: Optional[str] = None,
        user_ip: Optional[str] = None
    ) -> FileMetadata:
        """
        Copy a file with logging.
        
        Args:
            source_path: Source file path
            dest_path: Destination file path
            source_version_id: Source version ID
            user_id: User ID for auditing
            user_ip: User IP for auditing
            
        Returns:
            Metadata of copied file
        """
        # Get source metadata
        source_metadata = await self.storage.get_file_metadata(source_path, source_version_id)
        
        # Copy file
        dest_metadata = await self.storage.copy_file(source_path, dest_path, source_version_id)
        
        # Log event
        if source_metadata:
            self.storage._log_storage_event(
                event_type=StorageEvent.FILE_COPIED,
                details={
                    "source_path": source_path,
                    "dest_path": dest_path,
                    "source_filename": source_metadata.filename,
                    "dest_filename": dest_metadata.filename,
                    "size": source_metadata.size,
                    "user_id": user_id,
                    "user_ip": user_ip,
                    "source_version_id": source_version_id,
                }
            )
        
        return dest_metadata
    
    async def move(
        self,
        source_path: str,
        dest_path: str,
        source_version_id: Optional[str] = None,
        user_id: Optional[str] = None,
        user_ip: Optional[str] = None
    ) -> FileMetadata:
        """
        Move a file with logging.
        
        Args:
            source_path: Source file path
            dest_path: Destination file path
            source_version_id: Source version ID
            user_id: User ID for auditing
            user_ip: User IP for auditing
            
        Returns:
            Metadata of moved file
        """
        # Get source metadata
        source_metadata = await self.storage.get_file_metadata(source_path, source_version_id)
        
        # Move file
        dest_metadata = await self.storage.move_file(source_path, dest_path, source_version_id)
        
        # Log event
        if source_metadata:
            self.storage._log_storage_event(
                event_type=StorageEvent.FILE_MOVED,
                details={
                    "source_path": source_path,
                    "dest_path": dest_path,
                    "source_filename": source_metadata.filename,
                    "dest_filename": dest_metadata.filename,
                    "size": source_metadata.size,
                    "user_id": user_id,
                    "user_ip": user_ip,
                    "source_version_id": source_version_id,
                }
            )
        
        return dest_metadata
    
    async def get_stats(self) -> StorageStats:
        """Get storage statistics."""
        return await self.storage.get_storage_stats()
    
    async def search_files(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        file_type: Optional[FileType] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[FileMetadata]:
        """
        Search files by various criteria.
        
        Args:
            query: Search query (filename/path contains)
            tags: Filter by tags
            file_type: Filter by file type
            min_size: Minimum file size
            max_size: Maximum file size
            start_date: Start date for created_at
            end_date: End date for created_at
            limit: Maximum results
            
        Returns:
            List of matching files
        """
        # Get all files
        all_files = await self.storage.list_files(recursive=True, max_keys=10000)
        
        # Apply filters
        results = []
        
        for file_metadata in all_files:
            # Query filter
            if query and query.lower() not in file_metadata.filename.lower():
                continue
            
            # Tags filter
            if tags and not any(tag in file_metadata.tags for tag in tags):
                continue
            
            # File type filter
            if file_type and file_metadata.file_type != file_type:
                continue
            
            # Size filter
            if min_size and file_metadata.size < min_size:
                continue
            if max_size and file_metadata.size > max_size:
                continue
            
            # Date filter
            if start_date and file_metadata.created_at < start_date:
                continue
            if end_date and file_metadata.created_at > end_date:
                continue
            
            results.append(file_metadata)
            
            if len(results) >= limit:
                break
        
        return results
    
    async def cleanup_old_files(
        self,
        days_old: int = 30,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Cleanup old files.
        
        Args:
            days_old: Delete files older than this many days
            dry_run: If True, only report what would be deleted
            
        Returns:
            Cleanup statistics
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        # Get all files
        all_files = await self.storage.list_files(recursive=True)
        
        # Find old files
        old_files = []
        total_size = 0
        
        for file_metadata in all_files:
            if file_metadata.created_at < cutoff_date:
                old_files.append(file_metadata)
                total_size += file_metadata.size
        
        # Delete files (if not dry run)
        deleted_files = []
        deleted_size = 0
        
        if not dry_run:
            for file_metadata in old_files:
                try:
                    await self.storage.delete_file(file_metadata.filepath)
                    deleted_files.append(file_metadata.filename)
                    deleted_size += file_metadata.size
                except Exception as e:
                    self.storage._log_storage_event(
                        event_type=StorageEvent.FILE_DELETED,
                        details={
                            "filepath": file_metadata.filepath,
                            "error": str(e),
                        },
                        severity=AuditSeverity.ERROR
                    )
        
        return {
            "dry_run": dry_run,
            "total_files_found": len(all_files),
            "old_files_found": len(old_files),
            "total_size_old_files": total_size,
            "files_deleted": len(deleted_files),
            "size_deleted": deleted_size,
            "deleted_filenames": deleted_files,
        }


# File Upload Handler
class FileUploadHandler:
    """Handler for file uploads with validation and processing."""
    
    def __init__(self, file_manager: Optional[FileManager] = None):
        self.file_manager = file_manager or FileManager()
    
    async def handle_upload(
        self,
        file: BinaryIO,
        filename: str,
        filepath: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        access_level: AccessLevel = AccessLevel.PRIVATE,
        content_type: Optional[str] = None,
        user_id: Optional[str] = None,
        user_ip: Optional[str] = None,
        process_image: bool = False,
        generate_thumbnails: bool = False,
        compress: bool = False
    ) -> Dict[str, Any]:
        """
        Handle file upload with optional processing.
        
        Args:
            file: File object
            filename: Original filename
            filepath: Destination path
            metadata: Additional metadata
            tags: File tags
            access_level: Access level
            content_type: Content type
            user_id: User ID
            user_ip: User IP
            process_image: Process image if it's an image
            generate_thumbnails: Generate thumbnails for images
            compress: Compress file if supported
            
        Returns:
            Upload result with metadata and processing info
        """
        result = {
            "success": False,
            "metadata": None,
            "thumbnails": [],
            "processing_errors": [],
        }
        
        try:
            # Upload original file
            file_metadata = await self.file_manager.upload(
                file_obj=file,
                filename=filename,
                filepath=filepath,
                metadata=metadata,
                tags=tags,
                access_level=access_level,
                content_type=content_type,
                user_id=user_id,
                user_ip=user_ip
            )
            
            result["metadata"] = file_metadata
            result["success"] = True
            
            # Process image if requested
            if process_image and file_metadata.file_type == FileType.IMAGE:
                try:
                    await self._process_image(file_metadata)
                except Exception as e:
                    result["processing_errors"].append(f"Image processing failed: {str(e)}")
            
            # Generate thumbnails if requested
            if generate_thumbnails and file_metadata.file_type == FileType.IMAGE:
                try:
                    thumbnails = await self._generate_thumbnails(file_metadata)
                    result["thumbnails"] = thumbnails
                except Exception as e:
                    result["processing_errors"].append(f"Thumbnail generation failed: {str(e)}")
            
            # Compress if requested
            if compress:
                try:
                    compressed = await self._compress_file(file_metadata)
                    if compressed:
                        result["compressed"] = True
                except Exception as e:
                    result["processing_errors"].append(f"Compression failed: {str(e)}")
        
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    async def _process_image(self, file_metadata: FileMetadata):
        """Process image (optimize, strip metadata, etc.)."""
        # This would be implemented with PIL/Pillow or similar
        pass
    
    async def _generate_thumbnails(self, file_metadata: FileMetadata) -> List[FileMetadata]:
        """Generate thumbnails for image."""
        # This would generate different sizes and return their metadata
        return []
    
    async def _compress_file(self, file_metadata: FileMetadata) -> bool:
        """Compress file if supported."""
        # This would compress files based on type
        return False


# Storage Monitor
class StorageMonitor:
    """Monitor storage usage and health."""
    
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager
        self.audit_logger = AuditLogger()
        
    async def check_storage_health(self) -> Dict[str, Any]:
        """Check storage health and report issues."""
        health_report = {
            "status": "healthy",
            "checks": {},
            "warnings": [],
            "errors": [],
        }
        
        try:
            # Get storage stats
            stats = await self.file_manager.get_stats()
            health_report["stats"] = stats.to_dict() if hasattr(stats, 'to_dict') else stats
            
            # Check disk usage
            if stats.used_space_percentage > 90:
                health_report["status"] = "warning"
                health_report["warnings"].append(
                    f"Storage usage is high: {stats.used_space_percentage:.1f}%"
                )
            
            if stats.used_space_percentage > 95:
                health_report["status"] = "error"
                health_report["errors"].append(
                    f"Storage usage is critical: {stats.used_space_percentage:.1f}%"
                )
            
            # Check for very old files
            if stats.oldest_file:
                age_days = (datetime.utcnow() - stats.oldest_file).days
                if age_days > 365:
                    health_report["warnings"].append(
                        f"Oldest file is {age_days} days old"
                    )
            
            # Check backup status
            if not stats.last_backup or (datetime.utcnow() - stats.last_backup).days > 7:
                health_report["warnings"].append("No recent backup found")
        
        except Exception as e:
            health_report["status"] = "error"
            health_report["errors"].append(f"Health check failed: {str(e)}")
        
        return health_report
    
    async def monitor_upload_activity(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Monitor upload activity for given time period."""
        # This would query audit logs for upload events
        # For now, return placeholder
        return {
            "period_hours": hours,
            "total_uploads": 0,
            "total_size": 0,
            "uploads_by_type": {},
            "top_uploaders": [],
        }
    
    async def generate_usage_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate storage usage report for date range."""
        # This would analyze storage usage patterns
        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "total_files_added": 0,
            "total_files_removed": 0,
            "net_storage_change": 0,
            "cost_estimate": 0.0,
            "usage_by_user": {},
            "usage_by_file_type": {},
        }


# FastAPI Dependencies
def get_storage() -> BaseStorage:
    """Dependency to get storage instance."""
    config = StorageFactory.get_default_config()
    return StorageFactory.create_storage(config)


def get_file_manager() -> FileManager:
    """Dependency to get file manager."""
    return FileManager()


def get_upload_handler() -> FileUploadHandler:
    """Dependency to get upload handler."""
    return FileUploadHandler()


# Context manager for temporary files
@contextmanager
def temporary_upload(
    filename: str,
    content: bytes,
    cleanup: bool = True
):
    """
    Context manager for temporary file uploads.
    
    Args:
        filename: Temporary filename
        content: File content
        cleanup: Whether to cleanup after
        
    Yields:
        Temporary file path
    """
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / filename
    
    try:
        with open(temp_path, 'wb') as f:
            f.write(content)
        
        yield temp_path
    
    finally:
        if cleanup:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


# Utility functions
def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """Validate file extension against allowed list."""
    ext = Path(filename).suffix.lower()
    return ext in allowed_extensions


def get_file_type_from_extension(filename: str) -> FileType:
    """Get file type from extension."""
    ext = Path(filename).suffix.lower()
    
    image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.bmp', '.tiff'}
    video_exts = {'.mp4', '.webm', '.mov', '.avi', '.mkv'}
    audio_exts = {'.mp3', '.wav', '.ogg', '.flac', '.m4a'}
    document_exts = {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.txt'}
    archive_exts = {'.zip', '.tar', '.gz', '.7z', '.rar'}
    
    if ext in image_exts:
        return FileType.IMAGE
    elif ext in video_exts:
        return FileType.VIDEO
    elif ext in audio_exts:
        return FileType.AUDIO
    elif ext in document_exts:
        return FileType.DOCUMENT
    elif ext in archive_exts:
        return FileType.ARCHIVE
    else:
        return FileType.UNKNOWN


def calculate_file_hash(file_path: Path, algorithm: str = 'sha256') -> str:
    """Calculate file hash."""
    hash_func = getattr(hashlib, algorithm)()
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def get_human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


# Export main components
__all__ = [
    # Classes
    "BaseStorage",
    "StorageFactory",
    "FileManager",
    "FileUploadHandler",
    "StorageMonitor",
    
    # Data classes
    "StorageConfig",
    "FileMetadata",
    "StorageStats",
    
    # Enums
    "StorageProvider",
    "FileType",
    "AccessLevel",
    "StorageEvent",
    
    # Exceptions
    "StorageError",
    "FileNotFoundError",
    "FileTooLargeError",
    "FileTypeNotAllowedError",
    "StorageQuotaExceededError",
    "StorageIntegrityError",
    
    # FastAPI Dependencies
    "get_storage",
    "get_file_manager",
    "get_upload_handler",
    
    # Context managers
    "temporary_upload",
    
    # Utility functions
    "validate_file_extension",
    "get_file_type_from_extension",
    "calculate_file_hash",
    "get_human_readable_size",
    
    # Storage implementations (conditionally exported)
    "LocalStorage",
    "S3Storage",
    "GCSStorage",
    "AzureStorage",
    "CDNManager",
]

# Print initialization message
print(f"Storage module {__version__} initialized")
print(f"Default provider: {StorageFactory.get_default_config().provider.value}")