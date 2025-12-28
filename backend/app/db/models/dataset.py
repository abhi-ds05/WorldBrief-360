"""
dataset.py - Dataset Model for Data Management

This module defines the Dataset model for organizing and managing 
collections of data (articles, incidents, embeddings, etc.) for 
training, analysis, or export purposes.

Key Features:
- Dataset versioning and snapshots
- Support for different dataset types (training, validation, test, analysis)
- Data lineage and provenance tracking
- Integration with embeddings and articles
- Export to various formats
"""

import uuid
import json
import hashlib
from datetime import datetime
from typing import Optional, Dict, List, Any, TYPE_CHECKING
from enum import Enum
from sqlalchemy import (
    Column, String, Text, ForeignKey, Integer, DateTime, 
    Boolean, Enum as SQLEnum, JSON, LargeBinary, Float
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func

from db.base import Base
from models.mixins import TimestampMixin, SoftDeleteMixin

if TYPE_CHECKING:
    from models.user import User
    from models.article import Article
    from models.incident import Incident
    from models.embedding import Embedding
    from models.comment import Comment


class DatasetType(Enum):
    """Types of datasets."""
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"
    ANALYSIS = "analysis"
    PRODUCTION = "production"
    ARCHIVE = "archive"
    EXPORT = "export"


class DatasetFormat(Enum):
    """Supported dataset export formats."""
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"
    HDF5 = "hdf5"
    PICKLE = "pickle"
    TF_RECORD = "tf_record"
    TORCH = "torch"


class DatasetStatus(Enum):
    """Dataset processing status."""
    DRAFT = "draft"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"
    EXPORTING = "exporting"
    EXPORTED = "exported"
    DELETED = "deleted"


class Dataset(Base, TimestampMixin):
    """
    Dataset model for managing collections of data items.
    
    A dataset can contain multiple data sources (articles, incidents, embeddings)
    and supports versioning, snapshots, and export functionality.
    
    Attributes:
        id: Primary key UUID
        name: Human-readable dataset name
        description: Detailed description
        version: Dataset version (semantic versioning)
        type: Type of dataset (training, validation, etc.)
        format: Storage/export format
        status: Current processing status
        metadata: JSON field for additional configuration
        statistics: JSON field for dataset statistics
        checksum: MD5 checksum for data integrity
        file_path: Path to exported file (if applicable)
        file_size: Size of exported file in bytes
        is_public: Whether dataset is publicly accessible
        owner_id: Foreign key to User who owns the dataset
        parent_id: Foreign key to parent dataset (for versioning)
    """
    
    __tablename__ = "datasets"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Basic information
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    version = Column(String(50), default="1.0.0", nullable=False)
    
    # Type and format
    type = Column(
        SQLEnum(DatasetType),
        default=DatasetType.ANALYSIS,
        nullable=False,
        index=True
    )
    format = Column(
        SQLEnum(DatasetFormat),
        default=DatasetFormat.JSONL,
        nullable=True
    )
    
    # Status
    status = Column(
        SQLEnum(DatasetStatus),
        default=DatasetStatus.DRAFT,
        nullable=False,
        index=True
    )
    
    # Metadata and statistics
    metadata = Column(JSONB, default=dict, nullable=False, comment="Dataset configuration and metadata")
    statistics = Column(JSONB, default=dict, nullable=False, comment="Dataset statistics and metrics")
    
    # File information (for exported datasets)
    checksum = Column(String(64), nullable=True, index=True, comment="MD5 checksum for data integrity")
    file_path = Column(String(1024), nullable=True, comment="Path to exported dataset file")
    file_size = Column(Integer, nullable=True, comment="File size in bytes")
    exported_at = Column(DateTime(timezone=True), nullable=True)
    
    # Access control
    is_public = Column(Boolean, default=False, nullable=False, index=True)
    is_published = Column(Boolean, default=False, nullable=False, index=True)
    
    # Foreign keys
    owner_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    parent_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("datasets.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    
    # Relationships
    owner = relationship("User", back_populates="datasets")
    parent = relationship("Dataset", remote_side=[id], back_populates="versions")
    versions = relationship("Dataset", back_populates="parent")
    
    # Many-to-many relationships with data items
    articles = relationship(
        "Article",
        secondary="dataset_articles",
        back_populates="datasets",
        cascade="all, delete"
    )
    incidents = relationship(
        "Incident",
        secondary="dataset_incidents",
        back_populates="datasets",
        cascade="all, delete"
    )
    embeddings = relationship(
        "Embedding",
        secondary="dataset_embeddings",
        back_populates="datasets"
    )
    comments = relationship(
        "Comment",
        secondary="dataset_comments",
        back_populates="datasets"
    )
    
    # Processing logs
    processing_logs = relationship("DatasetProcessingLog", back_populates="dataset", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        """String representation of the Dataset."""
        return f"<Dataset(id={self.id}, name='{self.name}', type={self.type.value}, version={self.version})>"
    
    @validates('name')
    def validate_name(self, key: str, name: str) -> str:
        """Validate dataset name."""
        name = name.strip()
        if not name:
            raise ValueError("Dataset name cannot be empty")
        if len(name) > 255:
            raise ValueError("Dataset name is too long")
        return name
    
    @validates('version')
    def validate_version(self, key: str, version: str) -> str:
        """Validate semantic versioning."""
        # Basic semantic version validation
        import re
        pattern = r'^\d+\.\d+\.\d+$'
        if not re.match(pattern, version):
            raise ValueError("Version must follow semantic versioning (X.Y.Z)")
        return version
    
    @property
    def full_name(self) -> str:
        """Get the full dataset name with version."""
        return f"{self.name}-v{self.version}"
    
    @property
    def item_count(self) -> int:
        """Get total number of items in the dataset."""
        count = 0
        if self.articles:
            count += len(self.articles)
        if self.incidents:
            count += len(self.incidents)
        if self.embeddings:
            count += len(self.embeddings)
        if self.comments:
            count += len(self.comments)
        return count
    
    @property
    def is_exportable(self) -> bool:
        """Check if dataset can be exported."""
        return self.status in [DatasetStatus.READY, DatasetStatus.EXPORTED]
    
    @property
    def is_latest_version(self) -> bool:
        """Check if this is the latest version of the dataset."""
        if not self.versions:
            return True
        # Find the version with highest semantic version
        versions = [self] + [v for v in self.versions]
        return self == max(versions, key=lambda x: tuple(map(int, x.version.split('.'))))
    
    def update_statistics(self) -> None:
        """Update dataset statistics based on current items."""
        stats = {
            "total_items": self.item_count,
            "article_count": len(self.articles) if self.articles else 0,
            "incident_count": len(self.incidents) if self.incidents else 0,
            "embedding_count": len(self.embeddings) if self.embeddings else 0,
            "comment_count": len(self.comments) if self.comments else 0,
            "updated_at": datetime.utcnow().isoformat(),
        }
        
        # Add article statistics if available
        if self.articles:
            stats["articles"] = {
                "sources": list(set(a.source for a in self.articles if a.source)),
                "languages": list(set(a.language for a in self.articles if a.language)),
                "date_range": {
                    "min": min(a.published_at for a in self.articles if a.published_at).isoformat(),
                    "max": max(a.published_at for a in self.articles if a.published_at).isoformat(),
                } if any(a.published_at for a in self.articles) else None
            }
        
        # Add incident statistics if available
        if self.incidents:
            stats["incidents"] = {
                "severity_distribution": {},
                "status_distribution": {},
                "type_distribution": {}
            }
        
        self.statistics = stats
    
    def calculate_checksum(self) -> str:
        """Calculate checksum for dataset integrity."""
        import hashlib
        
        # Create a deterministic representation of the dataset
        data = {
            "id": str(self.id),
            "name": self.name,
            "version": self.version,
            "articles": sorted([str(a.id) for a in self.articles]) if self.articles else [],
            "incidents": sorted([str(i.id) for i in self.incidents]) if self.incidents else [],
            "embeddings": sorted([str(e.id) for e in self.embeddings]) if self.embeddings else [],
            "metadata": self.metadata
        }
        
        data_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def add_article(self, article: 'Article') -> None:
        """Add an article to the dataset."""
        if article not in self.articles:
            self.articles.append(article)
            self.update_statistics()
    
    def add_incident(self, incident: 'Incident') -> None:
        """Add an incident to the dataset."""
        if incident not in self.incidents:
            self.incidents.append(incident)
            self.update_statistics()
    
    def add_embedding(self, embedding: 'Embedding') -> None:
        """Add an embedding to the dataset."""
        if embedding not in self.embeddings:
            self.embeddings.append(embedding)
            self.update_statistics()
    
    def remove_article(self, article: 'Article') -> None:
        """Remove an article from the dataset."""
        if article in self.articles:
            self.articles.remove(article)
            self.update_statistics()
    
    def remove_incident(self, incident: 'Incident') -> None:
        """Remove an incident from the dataset."""
        if incident in self.incidents:
            self.incidents.remove(incident)
            self.update_statistics()
    
    def create_version(self, new_version: str, description: Optional[str] = None) -> 'Dataset':
        """
        Create a new version of the dataset.
        
        Args:
            new_version: New semantic version (e.g., "1.1.0")
            description: Optional description for the new version
        
        Returns:
            New Dataset instance
        """
        from sqlalchemy import inspect
        
        # Create a copy of the current dataset
        inspector = inspect(self)
        
        # Create new dataset with same attributes
        new_dataset = Dataset(
            name=self.name,
            description=description or self.description,
            version=new_version,
            type=self.type,
            format=self.format,
            metadata=self.metadata.copy(),
            statistics=self.statistics.copy(),
            is_public=self.is_public,
            owner_id=self.owner_id,
            parent_id=self.id
        )
        
        # Copy relationships
        new_dataset.articles = self.articles.copy()
        new_dataset.incidents = self.incidents.copy()
        new_dataset.embeddings = self.embeddings.copy()
        new_dataset.comments = self.comments.copy()
        
        return new_dataset
    
    def to_dict(self, include_items: bool = False, include_details: bool = False) -> Dict[str, Any]:
        """
        Convert dataset to dictionary for API responses.
        
        Args:
            include_items: Whether to include item IDs in the output
            include_details: Whether to include detailed item information
        
        Returns:
            Dictionary representation of the dataset
        """
        result = {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "type": self.type.value,
            "format": self.format.value if self.format else None,
            "status": self.status.value,
            "metadata": self.metadata,
            "statistics": self.statistics,
            "is_public": self.is_public,
            "is_published": self.is_published,
            "is_latest_version": self.is_latest_version,
            "item_count": self.item_count,
            "is_exportable": self.is_exportable,
            "checksum": self.checksum,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "exported_at": self.exported_at.isoformat() if self.exported_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "owner_id": str(self.owner_id) if self.owner_id else None
        }
        
        if include_items:
            result["articles"] = [str(a.id) for a in self.articles] if self.articles else []
            result["incidents"] = [str(i.id) for i in self.incidents] if self.incidents else []
            result["embeddings"] = [str(e.id) for e in self.embeddings] if self.embeddings else []
            result["comments"] = [str(c.id) for c in self.comments] if self.comments else []
        
        if include_details and self.owner:
            result["owner"] = {
                "id": str(self.owner.id),
                "username": self.owner.username,
                "email": self.owner.email
            }
        
        return result
    
    def export_metadata(self) -> Dict[str, Any]:
        """Export dataset metadata for documentation."""
        return {
            "dataset": {
                "name": self.name,
                "version": self.version,
                "description": self.description,
                "type": self.type.value,
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "updated_at": self.updated_at.isoformat() if self.updated_at else None,
                "checksum": self.checksum
            },
            "statistics": self.statistics,
            "metadata": self.metadata
        }
    
    @classmethod
    def create(
        cls,
        name: str,
        owner_id: uuid.UUID,
        dataset_type: DatasetType = DatasetType.ANALYSIS,
        description: Optional[str] = None,
        version: str = "1.0.0",
        is_public: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'Dataset':
        """
        Factory method to create a new dataset.
        
        Args:
            name: Dataset name
            owner_id: ID of the user creating the dataset
            dataset_type: Type of dataset
            description: Optional description
            version: Dataset version
            is_public: Whether dataset is public
            metadata: Additional metadata
        
        Returns:
            A new Dataset instance
        """
        dataset = cls(
            name=name,
            description=description,
            version=version,
            type=dataset_type,
            owner_id=owner_id,
            is_public=is_public,
            metadata=metadata or {},
            statistics={}
        )
        
        return dataset


class DatasetProcessingLog(Base, TimestampMixin):
    """
    Log for dataset processing operations.
    
    Tracks changes, exports, imports, and other operations on datasets.
    """
    
    __tablename__ = "dataset_processing_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    
    operation = Column(String(100), nullable=False, comment="Operation type (export, import, update, etc.)")
    status = Column(String(50), nullable=False, comment="Operation status (success, failed, etc.)")
    message = Column(Text, nullable=True, comment="Operation message or error")
    details = Column(JSONB, default=dict, nullable=True, comment="Operation details")
    
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationship
    dataset = relationship("Dataset", back_populates="processing_logs")
    
    def __repr__(self) -> str:
        return f"<DatasetProcessingLog(id={self.id}, operation={self.operation}, status={self.status})>"


# Association tables for many-to-many relationships
"""
These would typically be in a separate models/associations.py file,
but included here for completeness.
"""

from sqlalchemy import Table

dataset_articles = Table(
    'dataset_articles',
    Base.metadata,
    Column('dataset_id', UUID(as_uuid=True), ForeignKey('datasets.id', ondelete='CASCADE'), primary_key=True),
    Column('article_id', UUID(as_uuid=True), ForeignKey('articles.id', ondelete='CASCADE'), primary_key=True),
    Column('added_at', DateTime(timezone=True), server_default=func.now()),
    Column('metadata', JSONB, default=dict)
)

dataset_incidents = Table(
    'dataset_incidents',
    Base.metadata,
    Column('dataset_id', UUID(as_uuid=True), ForeignKey('datasets.id', ondelete='CASCADE'), primary_key=True),
    Column('incident_id', UUID(as_uuid=True), ForeignKey('incidents.id', ondelete='CASCADE'), primary_key=True),
    Column('added_at', DateTime(timezone=True), server_default=func.now()),
    Column('metadata', JSONB, default=dict)
)

dataset_embeddings = Table(
    'dataset_embeddings',
    Base.metadata,
    Column('dataset_id', UUID(as_uuid=True), ForeignKey('datasets.id', ondelete='CASCADE'), primary_key=True),
    Column('embedding_id', UUID(as_uuid=True), ForeignKey('embeddings.id', ondelete='CASCADE'), primary_key=True),
    Column('added_at', DateTime(timezone=True), server_default=func.now()),
    Column('metadata', JSONB, default=dict)
)

dataset_comments = Table(
    'dataset_comments',
    Base.metadata,
    Column('dataset_id', UUID(as_uuid=True), ForeignKey('datasets.id', ondelete='CASCADE'), primary_key=True),
    Column('comment_id', UUID(as_uuid=True), ForeignKey('comments.id', ondelete='CASCADE'), primary_key=True),
    Column('added_at', DateTime(timezone=True), server_default=func.now()),
    Column('metadata', JSONB, default=dict)
)


# Pydantic schemas for API validation
"""
If you're using Pydantic, here are the schemas for the Dataset model.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime


class DatasetBase(BaseModel):
    """Base schema for dataset operations."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    type: DatasetType = Field(default=DatasetType.ANALYSIS)
    is_public: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('name')
    def name_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Dataset name cannot be empty')
        return v.strip()


class DatasetCreate(DatasetBase):
    """Schema for creating a new dataset."""
    version: str = Field(default="1.0.0")
    
    @validator('version')
    def validate_version(cls, v):
        import re
        pattern = r'^\d+\.\d+\.\d+$'
        if not re.match(pattern, v):
            raise ValueError("Version must follow semantic versioning (X.Y.Z)")
        return v


class DatasetUpdate(BaseModel):
    """Schema for updating an existing dataset."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    is_public: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


class DatasetExportRequest(BaseModel):
    """Schema for dataset export requests."""
    format: DatasetFormat = Field(default=DatasetFormat.JSONL)
    include_embeddings: bool = Field(default=False)
    compression: Optional[str] = Field(None, description="Compression type (gzip, bzip2, etc.)")


class DatasetInDBBase(DatasetBase):
    """Base schema for dataset in database."""
    id: str
    version: str
    status: DatasetStatus
    statistics: Dict[str, Any]
    owner_id: Optional[str]
    item_count: int
    is_latest_version: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class Dataset(DatasetInDBBase):
    """Schema for dataset API responses."""
    owner: Optional[Dict[str, Any]] = None
    articles: Optional[List[str]] = None
    incidents: Optional[List[str]] = None
    embeddings: Optional[List[str]] = None
    
    class Config:
        from_attributes = True


class DatasetProcessingLogSchema(BaseModel):
    """Schema for dataset processing logs."""
    id: str
    operation: str
    status: str
    message: Optional[str]
    details: Optional[Dict[str, Any]]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime
    
    class Config:
        from_attributes = True