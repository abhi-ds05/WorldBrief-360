"""
Azure Blob Storage Implementation for WorldBrief 360

This module provides Azure Blob Storage integration with:
- Blob upload/download operations
- Container management
- SAS (Shared Access Signature) token generation
- Blob metadata and properties management
- Blob versioning and snapshots
- Blob lifecycle management
- Azure CDN integration
- Data encryption at rest and in transit
"""

import asyncio
import time
import uuid
import hashlib
import base64
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager

from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError
from azure.storage.blob import (
    BlobServiceClient,
    ContainerClient,
    BlobClient,
    BlobSasPermissions,
    generate_blob_sas,
    BlobType,
    StandardBlobTier,
    PremiumPageBlobTier,
    ContentSettings,
)
from azure.storage.blob.aio import (
    BlobServiceClient as AsyncBlobServiceClient,
    ContainerClient as AsyncContainerClient,
    BlobClient as AsyncBlobClient,
)
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from cryptography.fernet import Fernet

from app.core.config import get_settings
from app.security.audit_logger import AuditLogger, AuditEventType, AuditSeverity
from .base_storage import BaseStorage, FileMetadata, StorageConfig, AccessLevel
from . import FileType, StorageEvent

# Get settings
settings = get_settings()


class AzureBlobTier(str, Enum):
    """Azure Blob storage tiers."""
    HOT = "Hot"
    COOL = "Cool"
    ARCHIVE = "Archive"


class AzureEncryptionScope(str, Enum):
    """Azure encryption scopes."""
    DEFAULT = "Default"
    CUSTOM = "Custom"


@dataclass
class AzureStorageConfig(StorageConfig):
    """Azure Blob Storage specific configuration."""
    
    # Azure-specific settings
    connection_string: Optional[str] = None
    account_name: Optional[str] = None
    account_key: Optional[str] = None
    sas_token: Optional[str] = None
    use_managed_identity: bool = False
    tenant_id: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    
    # Blob-specific settings
    default_tier: AzureBlobTier = AzureBlobTier.HOT
    enable_blob_soft_delete: bool = True
    soft_delete_retention_days: int = 7
    enable_versioning: bool = False
    enable_change_feed: bool = False
    enable_container_delete_retention: bool = True
    container_delete_retention_days: int = 7
    
    # Encryption
    encryption_scope: str = AzureEncryptionScope.DEFAULT
    customer_provided_key: Optional[str] = None
    
    # Performance
    max_concurrency: int = 5
    max_single_put_size: int = 256 * 1024 * 1024  # 256MB
    max_block_size: int = 100 * 1024 * 1024  # 100MB
    
    # Replication
    replication_type: str = "LRS"  # LRS, GRS, RA-GRS, ZRS, GZRS, RA-GZRS
    
    # CDN
    cdn_profile_name: Optional[str] = None
    cdn_endpoint_name: Optional[str] = None
    
    def __post_init__(self):
        """Validate Azure-specific configuration."""
        super().__post_init__()
        
        # Check for required authentication method
        auth_methods = [
            self.connection_string,
            (self.account_name and (self.account_key or self.sas_token)),
            self.use_managed_identity,
            (self.tenant_id and self.client_id and self.client_secret)
        ]
        
        if not any(auth_methods):
            raise ValueError(
                "Azure Storage requires one of: connection_string, "
                "account_name with account_key/sas_token, "
                "managed identity, or service principal credentials"
            )
        
        # Set default bucket_name if not provided
        if not self.bucket_name:
            self.bucket_name = f"worldbrief360-{self.provider.value}"


@dataclass
class AzureBlobProperties:
    """Azure Blob specific properties."""
    blob_type: BlobType
    tier: AzureBlobTier
    access_tier_change_time: Optional[datetime]
    archive_status: Optional[str]
    content_length: int
    content_type: str
    content_encoding: Optional[str]
    content_language: Optional[str]
    content_md5: Optional[str]
    content_disposition: Optional[str]
    cache_control: Optional[str]
    last_modified: datetime
    creation_time: Optional[datetime]
    etag: str
    lease_state: Optional[str]
    lease_status: Optional[str]
    lease_duration: Optional[str]
    copy_id: Optional[str]
    copy_status: Optional[str]
    copy_source: Optional[str]
    copy_progress: Optional[str]
    copy_completion_time: Optional[datetime]
    copy_status_description: Optional[str]
    is_incremental_copy: Optional[bool]
    destination_snapshot: Optional[str]
    deleted_time: Optional[datetime]
    remaining_retention_days: Optional[int]
    access_tier_inferred: Optional[bool]
    customer_provided_key_sha256: Optional[str]
    encryption_key_sha256: Optional[str]
    encryption_scope: Optional[str]
    access_tier: Optional[str]
    blob_sequence_number: Optional[int]
    committed_block_count: Optional[int]
    is_server_encrypted: bool
    is_append_blob_sealed: Optional[bool]
    metadata: Dict[str, str]
    tag_count: Optional[int]
    version_id: Optional[str]
    is_current_version: Optional[bool]
    object_replication_source_properties: Optional[List[Dict[str, Any]]]
    object_replication_destination_policy_id: Optional[str]
    rehydrate_priority: Optional[str]
    last_accessed_on: Optional[datetime]


class AzureStorage(BaseStorage):
    """Azure Blob Storage implementation."""
    
    def __init__(self, config: AzureStorageConfig, audit_logger: Optional[AuditLogger] = None):
        super().__init__(config, audit_logger)
        self.config: AzureStorageConfig = config
        self.service_client: Optional[AsyncBlobServiceClient] = None
        self.container_client: Optional[AsyncContainerClient] = None
        self.sync_service_client: Optional[BlobServiceClient] = None
        self.encryption_key: Optional[bytes] = None
        self._initialize_encryption()
    
    def _initialize_encryption(self):
        """Initialize encryption if enabled."""
        if self.config.enable_encryption and self.config.encryption_key:
            key = self.config.encryption_key.encode()
            # Ensure key is 32 bytes for Fernet
            if len(key) < 32:
                key = key.ljust(32, b'0')[:32]
            self.encryption_key = base64.urlsafe_b64encode(key)
    
    async def _get_service_client(self) -> AsyncBlobServiceClient:
        """Get or create Azure Blob Service client."""
        if self.service_client is None:
            try:
                if self.config.connection_string:
                    self.service_client = AsyncBlobServiceClient.from_connection_string(
                        self.config.connection_string,
                        max_single_put_size=self.config.max_single_put_size,
                        max_block_size=self.config.max_block_size,
                    )
                
                elif self.config.account_name and (self.config.account_key or self.config.sas_token):
                    account_url = f"https://{self.config.account_name}.blob.core.windows.net"
                    
                    if self.config.account_key:
                        credential = self.config.account_key
                    else:
                        credential = self.config.sas_token
                    
                    self.service_client = AsyncBlobServiceClient(
                        account_url=account_url,
                        credential=credential,
                        max_single_put_size=self.config.max_single_put_size,
                        max_block_size=self.config.max_block_size,
                    )
                
                elif self.config.use_managed_identity:
                    credential = DefaultAzureCredential()
                    account_url = f"https://{self.config.account_name}.blob.core.windows.net"
                    
                    self.service_client = AsyncBlobServiceClient(
                        account_url=account_url,
                        credential=credential,
                        max_single_put_size=self.config.max_single_put_size,
                        max_block_size=self.config.max_block_size,
                    )
                
                elif self.config.tenant_id and self.config.client_id and self.config.client_secret:
                    credential = ClientSecretCredential(
                        tenant_id=self.config.tenant_id,
                        client_id=self.config.client_id,
                        client_secret=self.config.client_secret,
                    )
                    account_url = f"https://{self.config.account_name}.blob.core.windows.net"
                    
                    self.service_client = AsyncBlobServiceClient(
                        account_url=account_url,
                        credential=credential,
                        max_single_put_size=self.config.max_single_put_size,
                        max_block_size=self.config.max_block_size,
                    )
                
                else:
                    raise ValueError("No valid authentication method provided for Azure Storage")
                
                # Also create sync client for some operations
                self.sync_service_client = BlobServiceClient.from_connection_string(
                    self.config.connection_string
                ) if self.config.connection_string else None
                
            except Exception as e:
                self._log_storage_event(
                    StorageEvent.STORAGE_CREATED,
                    {"error": str(e), "provider": "azure"},
                    AuditSeverity.ERROR
                )
                raise
        
        return self.service_client
    
    async def _get_container_client(self) -> AsyncContainerClient:
        """Get or create container client."""
        if self.container_client is None:
            service_client = await self._get_service_client()
            self.container_client = service_client.get_container_client(self.config.bucket_name)
            
            # Create container if it doesn't exist
            try:
                await self.container_client.create_container(
                    metadata={
                        "created_by": "WorldBrief360",
                        "environment": settings.ENVIRONMENT,
                    }
                )
                
                # Configure container properties
                await self._configure_container()
                
                self._log_storage_event(
                    StorageEvent.STORAGE_CREATED,
                    {
                        "container": self.config.bucket_name,
                        "provider": "azure",
                        "region": self.config.region,
                    }
                )
                
            except ResourceExistsError:
                # Container already exists
                pass
            except Exception as e:
                self._log_storage_event(
                    StorageEvent.STORAGE_CREATED,
                    {"error": str(e), "container": self.config.bucket_name},
                    AuditSeverity.ERROR
                )
                raise
        
        return self.container_client
    
    async def _configure_container(self):
        """Configure container properties."""
        try:
            container_client = await self._get_container_client()
            
            # Set access policy
            from azure.storage.blob import PublicAccess
            
            # Map AccessLevel to Azure PublicAccess
            access_mapping = {
                AccessLevel.PRIVATE: PublicAccess.OFF,
                AccessLevel.PUBLIC_READ: PublicAccess.BLOB,
                AccessLevel.PUBLIC_READ_WRITE: PublicAccess.CONTAINER,
            }
            
            public_access = access_mapping.get(
                self.config.access_level or AccessLevel.PRIVATE,
                PublicAccess.OFF
            )
            
            await container_client.set_container_access_policy(
                public_access=public_access
            )
            
            # Set metadata
            await container_client.set_container_metadata(
                metadata={
                    "owner": "WorldBrief360",
                    "purpose": "file-storage",
                    "version": "1.0",
                }
            )
            
        except Exception as e:
            self._log_storage_event(
                StorageEvent.STORAGE_CREATED,
                {"warning": f"Container configuration failed: {str(e)}"},
                AuditSeverity.WARNING
            )
    
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
        """
        Upload a file to Azure Blob Storage.
        
        Args:
            file_obj: File object
            filename: Original filename
            filepath: Path to store the file
            metadata: Additional metadata
            tags: File tags
            access_level: Access level
            content_type: Content type
            
        Returns:
            FileMetadata object
        """
        start_time = time.time()
        
        # Validate file type
        file_type, detected_content_type = self._validate_file_type(filename)
        if content_type is None:
            content_type = detected_content_type
        
        # Calculate hashes
        md5_hash, sha256_hash = self._calculate_hashes(file_obj)
        
        # Determine blob path
        if filepath is None:
            # Generate unique filepath
            file_ext = Path(filename).suffix
            unique_id = str(uuid.uuid4())
            filepath = f"{unique_id}{file_ext}"
        
        full_path = self._get_full_path(filepath)
        
        # Prepare blob metadata
        blob_metadata = {
            "original_filename": filename,
            "file_type": file_type.value,
            "upload_timestamp": datetime.utcnow().isoformat(),
            "md5_hash": md5_hash,
            "sha256_hash": sha256_hash,
            "access_level": access_level.value,
        }
        
        if metadata:
            blob_metadata.update(metadata)
        
        if tags:
            blob_metadata["tags"] = ",".join(tags)
        
        # Encrypt file if enabled
        file_content = self._encrypt_data(file_obj) if self.encryption_key else file_obj
        
        try:
            container_client = await self._get_container_client()
            blob_client = container_client.get_blob_client(full_path)
            
            # Configure content settings
            content_settings = ContentSettings(
                content_type=content_type,
                content_encoding=None,
                content_language=None,
                content_disposition=f'inline; filename="{filename}"',
                cache_control="max-age=3600",
                content_md5=base64.b64encode(bytes.fromhex(md5_hash)).decode()
            )
            
            # Upload based on file size
            file_obj.seek(0, 2)  # Seek to end
            file_size = file_obj.tell()
            file_obj.seek(0)  # Reset position
            
            if file_size <= self.config.max_single_put_size:
                # Single upload for small files
                await blob_client.upload_blob(
                    data=file_content,
                    metadata=blob_metadata,
                    content_settings=content_settings,
                    overwrite=True,
                    standard_blob_tier=self.config.default_tier.value,
                    encryption_scope=self.config.encryption_scope,
                )
            else:
                # Chunked upload for large files
                await blob_client.upload_blob(
                    data=file_content,
                    metadata=blob_metadata,
                    content_settings=content_settings,
                    overwrite=True,
                    max_concurrency=self.config.max_concurrency,
                    standard_blob_tier=self.config.default_tier.value,
                    encryption_scope=self.config.encryption_scope,
                )
            
            # Get blob properties
            properties = await blob_client.get_blob_properties()
            
            # Create file metadata
            file_metadata = FileMetadata(
                filename=filename,
                filepath=full_path,
                size=properties.size,
                content_type=content_type,
                file_type=file_type,
                md5_hash=md5_hash,
                sha256_hash=sha256_hash,
                created_at=properties.creation_time or datetime.utcnow(),
                modified_at=properties.last_modified,
                metadata=blob_metadata,
                tags=tags or [],
                access_level=access_level,
                version_id=properties.get('version_id'),
                storage_class=self.config.default_tier.value,
                expires_at=None,
                etag=properties.etag,
                encryption="Azure SSE" if properties.is_server_encrypted else None,
            )
            
            # Log successful upload
            upload_duration = time.time() - start_time
            self._log_storage_event(
                StorageEvent.FILE_UPLOADED,
                {
                    "filename": filename,
                    "filepath": full_path,
                    "size": file_size,
                    "duration": upload_duration,
                    "blob_tier": self.config.default_tier.value,
                    "encrypted": bool(self.encryption_key),
                }
            )
            
            return file_metadata
            
        except Exception as e:
            self._log_storage_event(
                StorageEvent.FILE_UPLOADED,
                {
                    "error": str(e),
                    "filename": filename,
                    "filepath": full_path,
                },
                AuditSeverity.ERROR
            )
            raise
    
    async def download_file(
        self,
        filepath: str,
        version_id: Optional[str] = None
    ) -> BinaryIO:
        """
        Download a file from Azure Blob Storage.
        
        Args:
            filepath: Path to the file
            version_id: Version ID (for versioned blobs)
            
        Returns:
            File object
            
        Raises:
            FileNotFoundError: If blob does not exist
        """
        full_path = self._get_full_path(filepath)
        
        try:
            container_client = await self._get_container_client()
            blob_client = container_client.get_blob_client(full_path)
            
            if version_id:
                blob_client = blob_client.get_blob_client(version_id=version_id)
            
            # Download blob
            download_stream = await blob_client.download_blob()
            
            # Read content
            content = await download_stream.readall()
            
            # Decrypt if encrypted
            if self.encryption_key:
                content = self._decrypt_data(content)
            
            # Create BytesIO object
            import io
            file_obj = io.BytesIO(content)
            file_obj.name = Path(full_path).name
            
            # Log download
            self._log_storage_event(
                StorageEvent.FILE_DOWNLOADED,
                {
                    "filepath": full_path,
                    "version_id": version_id,
                    "size": len(content),
                }
            )
            
            return file_obj
            
        except ResourceNotFoundError:
            raise FileNotFoundError(f"Blob not found: {full_path}")
        except Exception as e:
            self._log_storage_event(
                StorageEvent.FILE_DOWNLOADED,
                {
                    "error": str(e),
                    "filepath": full_path,
                },
                AuditSeverity.ERROR
            )
            raise
    
    async def delete_file(
        self,
        filepath: str,
        version_id: Optional[str] = None
    ) -> bool:
        """
        Delete a file from Azure Blob Storage.
        
        Args:
            filepath: Path to the file
            version_id: Version ID (for versioned blobs)
            
        Returns:
            True if deleted successfully
        """
        full_path = self._get_full_path(filepath)
        
        try:
            container_client = await self._get_container_client()
            blob_client = container_client.get_blob_client(full_path)
            
            # Get metadata before deletion for logging
            metadata = await self.get_file_metadata(filepath, version_id)
            
            # Delete blob
            if version_id and self.config.enable_versioning:
                await blob_client.delete_blob(version_id=version_id)
            else:
                await blob_client.delete_blob()
            
            # Log deletion
            self._log_storage_event(
                StorageEvent.FILE_DELETED,
                {
                    "filepath": full_path,
                    "version_id": version_id,
                    "filename": metadata.filename if metadata else None,
                    "size": metadata.size if metadata else None,
                }
            )
            
            return True
            
        except ResourceNotFoundError:
            raise FileNotFoundError(f"Blob not found: {full_path}")
        except Exception as e:
            self._log_storage_event(
                StorageEvent.FILE_DELETED,
                {
                    "error": str(e),
                    "filepath": full_path,
                },
                AuditSeverity.ERROR
            )
            raise
    
    async def file_exists(
        self,
        filepath: str,
        version_id: Optional[str] = None
    ) -> bool:
        """
        Check if a file exists in Azure Blob Storage.
        
        Args:
            filepath: Path to the file
            version_id: Version ID (for versioned blobs)
            
        Returns:
            True if file exists
        """
        full_path = self._get_full_path(filepath)
        
        try:
            container_client = await self._get_container_client()
            blob_client = container_client.get_blob_client(full_path)
            
            if version_id:
                blob_client = blob_client.get_blob_client(version_id=version_id)
            
            await blob_client.get_blob_properties()
            return True
            
        except ResourceNotFoundError:
            return False
        except Exception:
            return False
    
    async def get_file_metadata(
        self,
        filepath: str,
        version_id: Optional[str] = None
    ) -> Optional[FileMetadata]:
        """
        Get file metadata from Azure Blob Storage.
        
        Args:
            filepath: Path to the file
            version_id: Version ID (for versioned blobs)
            
        Returns:
            FileMetadata object or None if not found
        """
        full_path = self._get_full_path(filepath)
        
        try:
            container_client = await self._get_container_client()
            blob_client = container_client.get_blob_client(full_path)
            
            if version_id:
                blob_client = blob_client.get_blob_client(version_id=version_id)
            
            properties = await blob_client.get_blob_properties()
            
            # Extract metadata from blob properties
            blob_metadata = properties.metadata or {}
            
            # Parse file type from metadata or content type
            file_type_str = blob_metadata.get('file_type', FileType.UNKNOWN.value)
            try:
                file_type = FileType(file_type_str)
            except ValueError:
                file_type = FileType.UNKNOWN
            
            # Parse access level
            access_level_str = blob_metadata.get('access_level', AccessLevel.PRIVATE.value)
            try:
                access_level = AccessLevel(access_level_str)
            except ValueError:
                access_level = AccessLevel.PRIVATE
            
            # Parse tags
            tags_str = blob_metadata.get('tags', '')
            tags = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
            
            # Get hashes from metadata
            md5_hash = blob_metadata.get('md5_hash', '')
            sha256_hash = blob_metadata.get('sha256_hash', '')
            
            # Get original filename
            filename = blob_metadata.get('original_filename', Path(full_path).name)
            
            return FileMetadata(
                filename=filename,
                filepath=full_path,
                size=properties.size,
                content_type=properties.content_settings.content_type,
                file_type=file_type,
                md5_hash=md5_hash,
                sha256_hash=sha256_hash,
                created_at=properties.creation_time or datetime.utcnow(),
                modified_at=properties.last_modified,
                metadata=blob_metadata,
                tags=tags,
                access_level=access_level,
                version_id=properties.get('version_id'),
                storage_class=properties.blob_tier or self.config.default_tier.value,
                expires_at=None,
                etag=properties.etag,
                encryption="Azure SSE" if properties.is_server_encrypted else None,
            )
            
        except ResourceNotFoundError:
            return None
        except Exception as e:
            self._log_storage_event(
                StorageEvent.FILE_DOWNLOADED,
                {
                    "error": str(e),
                    "filepath": full_path,
                    "operation": "get_metadata",
                },
                AuditSeverity.ERROR
            )
            return None
    
    async def list_files(
        self,
        prefix: Optional[str] = None,
        recursive: bool = True,
        max_keys: int = 1000
    ) -> List[FileMetadata]:
        """
        List files in Azure Blob Storage.
        
        Args:
            prefix: Prefix to filter files
            recursive: Whether to list recursively
            max_keys: Maximum number of files to return
            
        Returns:
            List of FileMetadata objects
        """
        try:
            container_client = await self._get_container_client()
            
            # Build prefix
            list_prefix = self._get_full_path(prefix) if prefix else None
            
            # List blobs
            blobs = []
            async for blob in container_client.list_blobs(
                name_starts_with=list_prefix,
                include=['metadata', 'tags']
            ):
                if len(blobs) >= max_keys:
                    break
                
                # Skip directories if not recursive
                if not recursive and '/' in blob.name[len(list_prefix or ''):]:
                    continue
                
                # Parse metadata
                blob_metadata = blob.metadata or {}
                
                # Parse file type
                file_type_str = blob_metadata.get('file_type', FileType.UNKNOWN.value)
                try:
                    file_type = FileType(file_type_str)
                except ValueError:
                    file_type = FileType.UNKNOWN
                
                # Parse access level
                access_level_str = blob_metadata.get('access_level', AccessLevel.PRIVATE.value)
                try:
                    access_level = AccessLevel(access_level_str)
                except ValueError:
                    access_level = AccessLevel.PRIVATE
                
                # Parse tags
                tags = list(blob.tags.keys()) if blob.tags else []
                
                # Get hashes
                md5_hash = blob_metadata.get('md5_hash', '')
                sha256_hash = blob_metadata.get('sha256_hash', '')
                
                # Get original filename
                filename = blob_metadata.get('original_filename', Path(blob.name).name)
                
                blobs.append(FileMetadata(
                    filename=filename,
                    filepath=blob.name,
                    size=blob.size,
                    content_type=blob.content_settings.content_type if blob.content_settings else None,
                    file_type=file_type,
                    md5_hash=md5_hash,
                    sha256_hash=sha256_hash,
                    created_at=blob.creation_time or datetime.utcnow(),
                    modified_at=blob.last_modified,
                    metadata=blob_metadata,
                    tags=tags,
                    access_level=access_level,
                    version_id=getattr(blob, 'version_id', None),
                    storage_class=blob.blob_tier or self.config.default_tier.value,
                    expires_at=None,
                    etag=blob.etag,
                    encryption="Azure SSE" if blob.is_current_version else None,
                ))
            
            return blobs
            
        except Exception as e:
            self._log_storage_event(
                StorageEvent.FILE_DOWNLOADED,
                {
                    "error": str(e),
                    "operation": "list_files",
                    "prefix": prefix,
                },
                AuditSeverity.ERROR
            )
            return []
    
    async def copy_file(
        self,
        source_path: str,
        dest_path: str,
        source_version_id: Optional[str] = None
    ) -> FileMetadata:
        """
        Copy a file within Azure Blob Storage.
        
        Args:
            source_path: Source file path
            dest_path: Destination file path
            source_version_id: Source version ID
            
        Returns:
            Metadata of copied file
        """
        source_full_path = self._get_full_path(source_path)
        dest_full_path = self._get_full_path(dest_path)
        
        try:
            container_client = await self._get_container_client()
            
            # Get source blob client
            source_blob = container_client.get_blob_client(source_full_path)
            if source_version_id:
                source_blob = source_blob.get_blob_client(version_id=source_version_id)
            
            # Get source properties
            source_props = await source_blob.get_blob_properties()
            
            # Create destination blob client
            dest_blob = container_client.get_blob_client(dest_full_path)
            
            # Start copy operation
            copy_source = source_blob.url
            if source_version_id:
                copy_source += f"?versionId={source_version_id}"
            
            await dest_blob.start_copy_from_url(copy_source)
            
            # Wait for copy to complete
            while True:
                dest_props = await dest_blob.get_blob_properties()
                if dest_props.copy.status != "pending":
                    break
                await asyncio.sleep(0.5)
            
            # Update metadata with new filename
            metadata = source_props.metadata or {}
            metadata['original_filename'] = Path(dest_path).name
            metadata['copied_from'] = source_full_path
            metadata['copied_at'] = datetime.utcnow().isoformat()
            
            await dest_blob.set_blob_metadata(metadata)
            
            # Get new metadata
            new_metadata = await self.get_file_metadata(dest_path)
            
            # Log copy operation
            self._log_storage_event(
                StorageEvent.FILE_COPIED,
                {
                    "source": source_full_path,
                    "destination": dest_full_path,
                    "source_version": source_version_id,
                    "size": source_props.size,
                }
            )
            
            return new_metadata
            
        except Exception as e:
            self._log_storage_event(
                StorageEvent.FILE_COPIED,
                {
                    "error": str(e),
                    "source": source_full_path,
                    "destination": dest_full_path,
                },
                AuditSeverity.ERROR
            )
            raise
    
    async def move_file(
        self,
        source_path: str,
        dest_path: str,
        source_version_id: Optional[str] = None
    ) -> FileMetadata:
        """
        Move a file within Azure Blob Storage.
        
        Args:
            source_path: Source file path
            dest_path: Destination file path
            source_version_id: Source version ID
            
        Returns:
            Metadata of moved file
        """
        # Copy file
        new_metadata = await self.copy_file(source_path, dest_path, source_version_id)
        
        # Delete source
        await self.delete_file(source_path, source_version_id)
        
        # Log move operation
        self._log_storage_event(
            StorageEvent.FILE_MOVED,
            {
                "source": self._get_full_path(source_path),
                "destination": self._get_full_path(dest_path),
                "source_version": source_version_id,
            }
        )
        
        return new_metadata
    
    async def get_presigned_url(
        self,
        filepath: str,
        expires_in: int = 3600,
        version_id: Optional[str] = None
    ) -> str:
        """
        Get a presigned URL for temporary access to a file.
        
        Args:
            filepath: Path to the file
            expires_in: URL expiration in seconds
            version_id: Version ID
            
        Returns:
            Presigned URL
        """
        full_path = self._get_full_path(filepath)
        
        try:
            if not self.sync_service_client:
                raise ValueError("Sync service client not available for SAS generation")
            
            # Generate SAS token
            blob_client = self.sync_service_client.get_blob_client(
                container=self.config.bucket_name,
                blob=full_path
            )
            
            # Create SAS permissions
            sas_permissions = BlobSasPermissions(
                read=True,
                write=False,
                delete=False,
                add=False,
                create=False,
                tag=False,
            )
            
            # Generate SAS token
            sas_token = generate_blob_sas(
                account_name=self.config.account_name or "",
                container_name=self.config.bucket_name,
                blob_name=full_path,
                account_key=self.config.account_key,
                permission=sas_permissions,
                expiry=datetime.utcnow() + timedelta(seconds=expires_in),
            )
            
            # Construct URL
            url = blob_client.url + "?" + sas_token
            
            if version_id:
                url += f"&versionId={version_id}"
            
            return url
            
        except Exception as e:
            self._log_storage_event(
                StorageEvent.FILE_DOWNLOADED,
                {
                    "error": str(e),
                    "filepath": full_path,
                    "operation": "generate_sas",
                },
                AuditSeverity.ERROR
            )
            raise
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get Azure Blob Storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        try:
            service_client = await self._get_service_client()
            container_client = await self._get_container_client()
            
            # Get container properties
            container_props = await container_client.get_container_properties()
            
            # List all blobs to calculate stats
            total_size = 0
            file_count = 0
            file_count_by_type = {}
            size_by_type = {}
            
            async for blob in container_client.list_blobs(include=['metadata']):
                file_count += 1
                total_size += blob.size
                
                # Get file type from metadata
                file_type_str = (blob.metadata or {}).get('file_type', FileType.UNKNOWN.value)
                try:
                    file_type = FileType(file_type_str)
                except ValueError:
                    file_type = FileType.UNKNOWN
                
                # Update counts
                file_count_by_type[file_type] = file_count_by_type.get(file_type, 0) + 1
                size_by_type[file_type] = size_by_type.get(file_type, 0) + blob.size
            
            # Get available space (Azure doesn't provide this directly)
            # We'll estimate based on subscription limits or return None
            available_space = None
            
            # Calculate cost estimate (simplified)
            cost_per_gb_per_month = {
                AzureBlobTier.HOT: 0.018,  # $0.018 per GB/month
                AzureBlobTier.COOL: 0.01,   # $0.01 per GB/month
                AzureBlobTier.ARCHIVE: 0.00099,  # $0.00099 per GB/month
            }
            
            monthly_cost = (total_size / (1024 ** 3)) * cost_per_gb_per_month.get(
                self.config.default_tier, AzureBlobTier.HOT
            )
            
            return {
                "total_files": file_count,
                "total_size": total_size,
                "available_space": available_space,
                "used_space_percentage": None,  # Azure doesn't provide container limits
                "file_count_by_type": {k.value: v for k, v in file_count_by_type.items()},
                "size_by_type": {k.value: v for k, v in size_by_type.items()},
                "oldest_file": None,  # Would need to track in metadata
                "newest_file": None,  # Would need to track in metadata
                "cost_estimate": monthly_cost,
                "last_backup": None,  # Would need backup tracking
                "container_name": self.config.bucket_name,
                "container_last_modified": container_props.last_modified.isoformat(),
                "replication_type": self.config.replication_type,
            }
            
        except Exception as e:
            self._log_storage_event(
                StorageEvent.STORAGE_CREATED,
                {
                    "error": str(e),
                    "operation": "get_stats",
                },
                AuditSeverity.ERROR
            )
            return {
                "total_files": 0,
                "total_size": 0,
                "available_space": None,
                "used_space_percentage": None,
                "file_count_by_type": {},
                "size_by_type": {},
                "oldest_file": None,
                "newest_file": None,
                "cost_estimate": 0.0,
                "last_backup": None,
                "container_name": self.config.bucket_name,
            }
    
    # Azure-specific methods
    async def change_blob_tier(
        self,
        filepath: str,
        new_tier: AzureBlobTier,
        version_id: Optional[str] = None
    ) -> bool:
        """
        Change the storage tier of a blob.
        
        Args:
            filepath: Path to the file
            new_tier: New storage tier
            version_id: Version ID
            
        Returns:
            True if tier was changed successfully
        """
        full_path = self._get_full_path(filepath)
        
        try:
            container_client = await self._get_container_client()
            blob_client = container_client.get_blob_client(full_path)
            
            if version_id:
                blob_client = blob_client.get_blob_client(version_id=version_id)
            
            await blob_client.set_standard_blob_tier(
                standard_blob_tier=new_tier.value
            )
            
            self._log_storage_event(
                StorageEvent.FILE_UPLOADED,
                {
                    "filepath": full_path,
                    "operation": "change_tier",
                    "new_tier": new_tier.value,
                    "old_tier": self.config.default_tier.value,
                }
            )
            
            return True
            
        except Exception as e:
            self._log_storage_event(
                StorageEvent.FILE_UPLOADED,
                {
                    "error": str(e),
                    "filepath": full_path,
                    "operation": "change_tier",
                },
                AuditSeverity.ERROR
            )
            return False
    
    async def create_snapshot(
        self,
        filepath: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """
        Create a snapshot of a blob.
        
        Args:
            filepath: Path to the file
            metadata: Snapshot metadata
            
        Returns:
            Snapshot ID or None if failed
        """
        full_path = self._get_full_path(filepath)
        
        try:
            container_client = await self._get_container_client()
            blob_client = container_client.get_blob_client(full_path)
            
            snapshot = await blob_client.create_snapshot(metadata=metadata)
            
            self._log_storage_event(
                StorageEvent.FILE_COPIED,
                {
                    "filepath": full_path,
                    "operation": "create_snapshot",
                    "snapshot_id": snapshot['snapshot'],
                }
            )
            
            return snapshot['snapshot']
            
        except Exception as e:
            self._log_storage_event(
                StorageEvent.FILE_COPIED,
                {
                    "error": str(e),
                    "filepath": full_path,
                    "operation": "create_snapshot",
                },
                AuditSeverity.ERROR
            )
            return None
    
    async def list_snapshots(
        self,
        filepath: str
    ) -> List[Dict[str, Any]]:
        """
        List snapshots for a blob.
        
        Args:
            filepath: Path to the file
            
        Returns:
            List of snapshot information
        """
        full_path = self._get_full_path(filepath)
        
        try:
            container_client = await self._get_container_client()
            
            snapshots = []
            async for blob in container_client.list_blobs(
                name_starts_with=full_path,
                include=['snapshots']
            ):
                if blob.snapshot:
                    snapshots.append({
                        'snapshot_id': blob.snapshot,
                        'size': blob.size,
                        'last_modified': blob.last_modified,
                        'metadata': blob.metadata,
                    })
            
            return snapshots
            
        except Exception as e:
            self._log_storage_event(
                StorageEvent.FILE_DOWNLOADED,
                {
                    "error": str(e),
                    "filepath": full_path,
                    "operation": "list_snapshots",
                },
                AuditSeverity.ERROR
            )
            return []
    
    async def restore_from_snapshot(
        self,
        filepath: str,
        snapshot_id: str
    ) -> bool:
        """
        Restore a blob from a snapshot.
        
        Args:
            filepath: Path to the file
            snapshot_id: Snapshot ID to restore from
            
        Returns:
            True if restored successfully
        """
        full_path = self._get_full_path(filepath)
        
        try:
            container_client = await self._get_container_client()
            blob_client = container_client.get_blob_client(full_path)
            
            # Copy snapshot over current blob
            snapshot_blob = blob_client.get_blob_client(snapshot=snapshot_id)
            await blob_client.start_copy_from_url(snapshot_blob.url)
            
            self._log_storage_event(
                StorageEvent.FILE_COPIED,
                {
                    "filepath": full_path,
                    "operation": "restore_snapshot",
                    "snapshot_id": snapshot_id,
                }
            )
            
            return True
            
        except Exception as e:
            self._log_storage_event(
                StorageEvent.FILE_COPIED,
                {
                    "error": str(e),
                    "filepath": full_path,
                    "operation": "restore_snapshot",
                    "snapshot_id": snapshot_id,
                },
                AuditSeverity.ERROR
            )
            return False
    
    async def set_blob_tags(
        self,
        filepath: str,
        tags: Dict[str, str],
        version_id: Optional[str] = None
    ) -> bool:
        """
        Set tags for a blob.
        
        Args:
            filepath: Path to the file
            tags: Tags to set
            version_id: Version ID
            
        Returns:
            True if tags were set successfully
        """
        full_path = self._get_full_path(filepath)
        
        try:
            container_client = await self._get_container_client()
            blob_client = container_client.get_blob_client(full_path)
            
            if version_id:
                blob_client = blob_client.get_blob_client(version_id=version_id)
            
            await blob_client.set_blob_tags(tags)
            
            self._log_storage_event(
                StorageEvent.FILE_UPLOADED,
                {
                    "filepath": full_path,
                    "operation": "set_tags",
                    "tags": tags,
                }
            )
            
            return True
            
        except Exception as e:
            self._log_storage_event(
                StorageEvent.FILE_UPLOADED,
                {
                    "error": str(e),
                    "filepath": full_path,
                    "operation": "set_tags",
                },
                AuditSeverity.ERROR
            )
            return False
    
    async def get_blob_tags(
        self,
        filepath: str,
        version_id: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Get tags for a blob.
        
        Args:
            filepath: Path to the file
            version_id: Version ID
            
        Returns:
            Dictionary of tags
        """
        full_path = self._get_full_path(filepath)
        
        try:
            container_client = await self._get_container_client()
            blob_client = container_client.get_blob_client(full_path)
            
            if version_id:
                blob_client = blob_client.get_blob_client(version_id=version_id)
            
            tags = await blob_client.get_blob_tags()
            return tags
            
        except Exception as e:
            self._log_storage_event(
                StorageEvent.FILE_DOWNLOADED,
                {
                    "error": str(e),
                    "filepath": full_path,
                    "operation": "get_tags",
                },
                AuditSeverity.ERROR
            )
            return {}
    
    # Helper methods
    def _encrypt_data(self, data: BinaryIO) -> bytes:
        """Encrypt data using Fernet."""
        if not self.encryption_key:
            return data.read()
        
        fernet = Fernet(self.encryption_key)
        return fernet.encrypt(data.read())
    
    def _decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using Fernet."""
        if not self.encryption_key:
            return encrypted_data
        
        fernet = Fernet(self.encryption_key)
        return fernet.decrypt(encrypted_data)
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.service_client:
            await self.service_client.close()
        if self.container_client:
            await self.container_client.close()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            asyncio.get_event_loop().create_task(self.cleanup())
        except:
            pass


# Utility functions
def create_azure_storage_config(
    account_name: str,
    account_key: Optional[str] = None,
    connection_string: Optional[str] = None,
    sas_token: Optional[str] = None,
    use_managed_identity: bool = False,
    tenant_id: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    bucket_name: Optional[str] = None,
    **kwargs
) -> AzureStorageConfig:
    """
    Create Azure Storage configuration.
    
    Args:
        account_name: Azure Storage account name
        account_key: Azure Storage account key
        connection_string: Azure Storage connection string
        sas_token: SAS token for authentication
        use_managed_identity: Use managed identity authentication
        tenant_id: Azure AD tenant ID
        client_id: Azure AD client ID
        client_secret: Azure AD client secret
        bucket_name: Container name
        **kwargs: Additional configuration options
        
    Returns:
        AzureStorageConfig object
    """
    # Determine authentication method
    if connection_string:
        auth_method = "connection_string"
    elif account_key:
        auth_method = "account_key"
    elif sas_token:
        auth_method = "sas_token"
    elif use_managed_identity:
        auth_method = "managed_identity"
    elif tenant_id and client_id and client_secret:
        auth_method = "service_principal"
    else:
        raise ValueError("No valid authentication method provided")
    
    # Generate default bucket name if not provided
    if not bucket_name:
        env = settings.ENVIRONMENT if hasattr(settings, 'ENVIRONMENT') else 'development'
        bucket_name = f"worldbrief360-{env}-{account_name.lower()}"
    
    return AzureStorageConfig(
        provider="azure",
        account_name=account_name,
        account_key=account_key,
        connection_string=connection_string,
        sas_token=sas_token,
        use_managed_identity=use_managed_identity,
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
        bucket_name=bucket_name,
        **kwargs
    )


async def test_azure_connection(config: AzureStorageConfig) -> Tuple[bool, str]:
    """
    Test Azure Storage connection.
    
    Args:
        config: Azure Storage configuration
        
    Returns:
        Tuple of (success, message)
    """
    try:
        storage = AzureStorage(config)
        await storage._get_service_client()
        await storage._get_container_client()
        
        # Test write/read
        test_content = b"WorldBrief360 Azure Storage Test"
        test_filename = f"test-{uuid.uuid4()}.txt"
        
        import io
        file_obj = io.BytesIO(test_content)
        
        # Upload test file
        metadata = await storage.upload_file(
            file_obj=file_obj,
            filename=test_filename,
            filepath=f"tests/{test_filename}"
        )
        
        # Download and verify
        downloaded = await storage.download_file(metadata.filepath)
        downloaded_content = downloaded.read()
        
        if downloaded_content != test_content:
            raise ValueError("Downloaded content doesn't match original")
        
        # Cleanup
        await storage.delete_file(metadata.filepath)
        
        await storage.cleanup()
        
        return True, "Azure Storage connection test successful"
        
    except Exception as e:
        return False, f"Azure Storage connection test failed: {str(e)}"


# Export main components
__all__ = [
    # Classes
    "AzureStorage",
    
    # Data classes
    "AzureStorageConfig",
    "AzureBlobProperties",
    
    # Enums
    "AzureBlobTier",
    "AzureEncryptionScope",
    
    # Utility functions
    "create_azure_storage_config",
    "test_azure_connection",
]