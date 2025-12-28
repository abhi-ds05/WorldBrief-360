"""
embedding.py - Embedding Model for Vector Representations

This module defines the Embedding model for storing vector embeddings 
generated from text data (articles, incidents, comments, etc.) to enable
semantic search, similarity analysis, and machine learning applications.

Key Features:
- Support for multiple embedding models (OpenAI, Hugging Face, etc.)
- Vector storage with dimension management
- Metadata for embedding provenance
- Integration with various content types
- Cache management for performance
"""

import uuid
import json
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING, Union
from enum import Enum
from sqlalchemy import (
    Column, String, Text, ForeignKey, Integer, DateTime, 
    Boolean, Enum as SQLEnum, JSON, Float, ARRAY
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.mutable import MutableDict, MutableList
from sqlalchemy.sql import func

# For vector operations, you might use pgvector or similar
# Uncomment if using pgvector extension
# from sqlalchemy.dialects.postgresql import VECTOR

from db.base import Base
from models.mixins import TimestampMixin

if TYPE_CHECKING:
    from models.article import Article
    from models.incident import Incident
    from models.comment import Comment
    from models.user import User
    from models.dataset import Dataset


class EmbeddingModel(Enum):
    """Supported embedding models."""
    OPENAI_ADA_002 = "text-embedding-ada-002"
    OPENAI_ADA_003 = "text-embedding-3-small"
    OPENAI_ADA_003_LARGE = "text-embedding-3-large"
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    BERT = "bert"
    USE = "universal-sentence-encoder"
    CUSTOM = "custom"
    HUGGING_FACE = "hugging-face"


class EmbeddingType(Enum):
    """Types of content being embedded."""
    ARTICLE = "article"
    ARTICLE_TITLE = "article_title"
    ARTICLE_CONTENT = "article_content"
    ARTICLE_SUMMARY = "article_summary"
    INCIDENT = "incident"
    INCIDENT_DESCRIPTION = "incident_description"
    COMMENT = "comment"
    USER_QUERY = "user_query"
    SEARCH_QUERY = "search_query"
    DATASET = "dataset"
    OTHER = "other"


class EmbeddingStatus(Enum):
    """Embedding processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"
    EXPIRED = "expired"


class Embedding(Base, TimestampMixin):
    """
    Embedding model for storing vector representations of text.
    
    Each embedding represents a high-dimensional vector that captures
    semantic meaning of text content for similarity search and ML tasks.
    
    Attributes:
        id: Primary key UUID
        vector: The embedding vector (stored as array or pgvector)
        dimension: Dimension of the embedding vector
        model: Model used to generate the embedding
        model_version: Specific version of the model
        embedding_type: Type of content embedded
        text_hash: Hash of the original text for deduplication
        text_preview: First 500 chars of original text
        status: Processing status
        metadata: JSON field for additional information
        cache_key: Key for caching embeddings
        expires_at: When cached embedding expires
        is_normalized: Whether vector is normalized (unit length)
        similarity_threshold: Default similarity threshold for this embedding
        source_id: Foreign key to source content (article, incident, etc.)
        source_type: Type of source (article, incident, etc.)
        owner_id: Foreign key to User who generated the embedding
    """
    
    __tablename__ = "embeddings"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Vector storage options:
    # Option 1: Using pgvector extension (recommended for PostgreSQL)
    # vector = Column(VECTOR(1536), nullable=False)  # Adjust dimension as needed
    
    # Option 2: Using PostgreSQL array (good for moderate dimensions)
    vector = Column(ARRAY(Float), nullable=False, index=False)
    
    # Option 3: Using JSONB (flexible but less performant for vector ops)
    # vector = Column(JSONB, nullable=False)
    
    # Vector metadata
    dimension = Column(Integer, nullable=False, index=True)
    is_normalized = Column(Boolean, default=False, nullable=False)
    similarity_threshold = Column(Float, default=0.7, nullable=False)
    
    # Model information
    model = Column(
        SQLEnum(EmbeddingModel),
        nullable=False,
        index=True
    )
    model_version = Column(String(100), nullable=True)
    
    # Content information
    embedding_type = Column(
        SQLEnum(EmbeddingType),
        nullable=False,
        index=True
    )
    text_hash = Column(String(64), nullable=False, index=True, unique=True, comment="SHA256 hash of original text")
    text_preview = Column(String(500), nullable=True, comment="First 500 characters of original text")
    original_text_length = Column(Integer, nullable=True, comment="Length of original text in characters")
    
    # Status and cache management
    status = Column(
        SQLEnum(EmbeddingStatus),
        default=EmbeddingStatus.COMPLETED,
        nullable=False,
        index=True
    )
    cache_key = Column(String(255), nullable=True, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    metadata = Column(MutableDict.as_mutable(JSONB), default=dict, nullable=False)
    
    # Foreign keys (polymorphic relationship pattern)
    source_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    source_type = Column(String(50), nullable=True, index=True)
    
    # Direct foreign keys for common relationships
    article_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("articles.id", ondelete="CASCADE"), 
        nullable=True,
        index=True
    )
    incident_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("incidents.id", ondelete="CASCADE"), 
        nullable=True,
        index=True
    )
    comment_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("comments.id", ondelete="CASCADE"), 
        nullable=True,
        index=True
    )
    owner_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    
    # Relationships
    article = relationship("Article", back_populates="embeddings")
    incident = relationship("Incident", back_populates="embeddings")
    comment = relationship("Comment", back_populates="embeddings")
    owner = relationship("User", back_populates="embeddings")
    
    # Many-to-many with datasets
    datasets = relationship(
        "Dataset",
        secondary="dataset_embeddings",
        back_populates="embeddings"
    )
    
    # Similarity relationships (for storing precomputed similarities)
    # similar_to = relationship(
    #     "EmbeddingSimilarity",
    #     foreign_keys="EmbeddingSimilarity.embedding_a_id",
    #     back_populates="embedding_a"
    # )
    
    def __repr__(self) -> str:
        """String representation of the Embedding."""
        return f"<Embedding(id={self.id}, model={self.model.value}, type={self.embedding_type.value})>"
    
    def __init__(self, **kwargs):
        """Initialize embedding with vector validation."""
        vector = kwargs.get('vector')
        if vector is not None:
            if isinstance(vector, list):
                kwargs['dimension'] = len(vector)
            elif isinstance(vector, np.ndarray):
                kwargs['dimension'] = vector.shape[0]
                kwargs['vector'] = vector.tolist()
        
        super().__init__(**kwargs)
    
    @validates('vector')
    def validate_vector(self, key: str, vector: Union[List[float], np.ndarray]) -> List[float]:
        """Validate and convert embedding vector."""
        if vector is None:
            raise ValueError("Embedding vector cannot be None")
        
        # Convert numpy array to list
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        
        # Ensure it's a list of floats
        if not isinstance(vector, list):
            raise ValueError("Vector must be a list or numpy array")
        
        if not all(isinstance(x, (int, float)) for x in vector):
            raise ValueError("Vector must contain only numbers")
        
        # Check dimension consistency
        if hasattr(self, 'dimension') and self.dimension and len(vector) != self.dimension:
            raise ValueError(f"Vector dimension {len(vector)} doesn't match declared dimension {self.dimension}")
        
        return vector
    
    @validates('dimension')
    def validate_dimension(self, key: str, dimension: int) -> int:
        """Validate embedding dimension."""
        if dimension <= 0:
            raise ValueError("Embedding dimension must be positive")
        if dimension > 10000:  # Reasonable upper limit
            raise ValueError("Embedding dimension is too large")
        return dimension
    
    @property
    def numpy_vector(self) -> np.ndarray:
        """Get embedding as numpy array."""
        return np.array(self.vector, dtype=np.float32)
    
    @property
    def is_expired(self) -> bool:
        """Check if cached embedding has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    @property
    def is_cacheable(self) -> bool:
        """Check if embedding can be cached."""
        return self.status == EmbeddingStatus.COMPLETED and not self.is_expired
    
    @property
    def source_info(self) -> Dict[str, Any]:
        """Get information about the source content."""
        info = {
            "source_type": self.source_type,
            "source_id": str(self.source_id) if self.source_id else None,
            "article_id": str(self.article_id) if self.article_id else None,
            "incident_id": str(self.incident_id) if self.incident_id else None,
            "comment_id": str(self.comment_id) if self.comment_id else None,
        }
        return info
    
    def normalize(self) -> 'Embedding':
        """Normalize the embedding vector to unit length."""
        if not self.is_normalized:
            vector_np = self.numpy_vector
            norm = np.linalg.norm(vector_np)
            if norm > 0:
                self.vector = (vector_np / norm).tolist()
                self.is_normalized = True
        return self
    
    def calculate_similarity(self, other: 'Embedding', normalized: bool = True) -> float:
        """
        Calculate cosine similarity with another embedding.
        
        Args:
            other: Another Embedding instance
            normalized: Whether to use normalized vectors
        
        Returns:
            Cosine similarity score between -1 and 1
        """
        if self.dimension != other.dimension:
            raise ValueError(f"Dimension mismatch: {self.dimension} != {other.dimension}")
        
        # Get vectors
        vec1 = self.numpy_vector
        vec2 = other.numpy_vector
        
        # Normalize if needed
        if normalized:
            if not self.is_normalized:
                vec1 = vec1 / np.linalg.norm(vec1)
            if not other.is_normalized:
                vec2 = vec2 / np.linalg.norm(vec2)
            similarity = np.dot(vec1, vec2)
        else:
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        
        return float(similarity)
    
    def is_similar_to(self, other: 'Embedding', threshold: Optional[float] = None) -> bool:
        """
        Check if this embedding is similar to another based on threshold.
        
        Args:
            other: Another Embedding instance
            threshold: Similarity threshold (uses instance threshold if None)
        
        Returns:
            True if similarity >= threshold
        """
        if threshold is None:
            threshold = self.similarity_threshold
        
        similarity = self.calculate_similarity(other)
        return similarity >= threshold
    
    def update_cache_info(self, cache_key: str, ttl_seconds: int = 86400) -> None:
        """Update cache information for this embedding."""
        self.cache_key = cache_key
        self.expires_at = datetime.utcnow() + datetime.timedelta(seconds=ttl_seconds)
        self.status = EmbeddingStatus.CACHED
    
    def mark_as_failed(self, error_message: str) -> None:
        """Mark embedding generation as failed."""
        self.status = EmbeddingStatus.FAILED
        if 'errors' not in self.metadata:
            self.metadata['errors'] = []
        self.metadata['errors'].append({
            'message': error_message,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def to_dict(self, include_vector: bool = True, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Convert embedding to dictionary for API responses.
        
        Args:
            include_vector: Whether to include the vector in output
            include_metadata: Whether to include metadata
        
        Returns:
            Dictionary representation of the embedding
        """
        result = {
            "id": str(self.id),
            "model": self.model.value,
            "model_version": self.model_version,
            "embedding_type": self.embedding_type.value,
            "dimension": self.dimension,
            "is_normalized": self.is_normalized,
            "similarity_threshold": self.similarity_threshold,
            "text_preview": self.text_preview,
            "original_text_length": self.original_text_length,
            "status": self.status.value,
            "is_expired": self.is_expired,
            "is_cacheable": self.is_cacheable,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "source_info": self.source_info,
            "owner_id": str(self.owner_id) if self.owner_id else None
        }
        
        if include_vector:
            result["vector"] = self.vector
        
        if include_metadata:
            result["metadata"] = self.metadata
        
        return result
    
    def to_serializable(self) -> Dict[str, Any]:
        """Convert to serializable format for storage/export."""
        return {
            "id": str(self.id),
            "vector": self.vector,
            "model": self.model.value,
            "embedding_type": self.embedding_type.value,
            "text_hash": self.text_hash,
            "text_preview": self.text_preview,
            "metadata": self.metadata,
            "source_info": self.source_info,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def calculate_text_hash(cls, text: str) -> str:
        """Calculate SHA256 hash of text for deduplication."""
        import hashlib
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    @classmethod
    def create(
        cls,
        vector: Union[List[float], np.ndarray],
        model: EmbeddingModel,
        embedding_type: EmbeddingType,
        text: str,
        text_hash: Optional[str] = None,
        model_version: Optional[str] = None,
        article_id: Optional[uuid.UUID] = None,
        incident_id: Optional[uuid.UUID] = None,
        comment_id: Optional[uuid.UUID] = None,
        owner_id: Optional[uuid.UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        normalize: bool = True
    ) -> 'Embedding':
        """
        Factory method to create a new embedding.
        
        Args:
            vector: Embedding vector
            model: Model used
            embedding_type: Type of content
            text: Original text
            text_hash: Pre-calculated hash (will calculate if None)
            model_version: Model version
            article_id: Associated article ID
            incident_id: Associated incident ID
            comment_id: Associated comment ID
            owner_id: Owner user ID
            metadata: Additional metadata
            normalize: Whether to normalize the vector
        
        Returns:
            A new Embedding instance
        """
        # Calculate text hash if not provided
        if text_hash is None:
            text_hash = cls.calculate_text_hash(text)
        
        # Determine source information
        source_type = None
        source_id = None
        
        if article_id:
            source_type = "article"
            source_id = article_id
        elif incident_id:
            source_type = "incident"
            source_id = incident_id
        elif comment_id:
            source_type = "comment"
            source_id = comment_id
        
        # Create embedding
        embedding = cls(
            vector=vector,
            model=model,
            model_version=model_version,
            embedding_type=embedding_type,
            text_hash=text_hash,
            text_preview=text[:500],
            original_text_length=len(text),
            article_id=article_id,
            incident_id=incident_id,
            comment_id=comment_id,
            owner_id=owner_id,
            source_type=source_type,
            source_id=source_id,
            metadata=metadata or {},
            status=EmbeddingStatus.COMPLETED
        )
        
        # Normalize if requested
        if normalize:
            embedding.normalize()
        
        return embedding
    
    @classmethod
    def batch_create(
        cls,
        vectors: List[Union[List[float], np.ndarray]],
        model: EmbeddingModel,
        embedding_type: EmbeddingType,
        texts: List[str],
        **kwargs
    ) -> List['Embedding']:
        """
        Create multiple embeddings in batch.
        
        Args:
            vectors: List of embedding vectors
            model: Model used
            embedding_type: Type of content
            texts: List of original texts
            **kwargs: Additional arguments passed to create()
        
        Returns:
            List of Embedding instances
        """
        if len(vectors) != len(texts):
            raise ValueError("Number of vectors must match number of texts")
        
        embeddings = []
        for vector, text in zip(vectors, texts):
            embedding = cls.create(
                vector=vector,
                model=model,
                embedding_type=embedding_type,
                text=text,
                **kwargs
            )
            embeddings.append(embedding)
        
        return embeddings


class EmbeddingSimilarity(Base, TimestampMixin):
    """
    Stores precomputed similarities between embeddings for performance.
    
    This is useful for frequently queried similarities or for building
    similarity graphs/clusters.
    """
    
    __tablename__ = "embedding_similarities"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    embedding_a_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("embeddings.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    embedding_b_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("embeddings.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    similarity_score = Column(Float, nullable=False, index=True)
    calculation_method = Column(String(100), default="cosine", nullable=False)
    is_reciprocal = Column(Boolean, default=True, nullable=False, 
                          comment="Whether similarity(a,b) == similarity(b,a)")
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=True)
    
    # Relationships
    embedding_a = relationship("Embedding", foreign_keys=[embedding_a_id])
    embedding_b = relationship("Embedding", foreign_keys=[embedding_b_id])
    
    # Unique constraint to prevent duplicates
    __table_args__ = (
        # Ensure we don't store both (a,b) and (b,a) if is_reciprocal is True
        # This would depend on your use case
    )
    
    def __repr__(self) -> str:
        return f"<EmbeddingSimilarity(a={self.embedding_a_id}, b={self.embedding_b_id}, score={self.similarity_score:.3f})>"


# Pydantic schemas for API validation
"""
If you're using Pydantic, here are the schemas for the Embedding model.
"""

from pydantic import BaseModel, Field, validator, conlist
from typing import Optional, List, Dict, Any, Union
from datetime import datetime


class EmbeddingBase(BaseModel):
    """Base schema for embedding operations."""
    model: EmbeddingModel
    embedding_type: EmbeddingType
    text: str = Field(..., min_length=1)
    model_version: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    normalize: bool = Field(default=True)
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()


class EmbeddingCreate(EmbeddingBase):
    """Schema for creating a new embedding."""
    # Vector can be provided or generated from text
    vector: Optional[List[float]] = None
    
    article_id: Optional[str] = None
    incident_id: Optional[str] = None
    comment_id: Optional[str] = None
    
    @validator('vector', pre=True, always=True)
    def validate_vector_dimensions(cls, v, values):
        if v is not None:
            if not isinstance(v, list):
                raise ValueError('Vector must be a list')
            if len(v) == 0:
                raise ValueError('Vector cannot be empty')
            if not all(isinstance(x, (int, float)) for x in v):
                raise ValueError('Vector must contain only numbers')
        return v


class EmbeddingSimilarityRequest(BaseModel):
    """Schema for similarity calculation requests."""
    embedding_id: str
    other_embedding_id: Optional[str] = None
    other_text: Optional[str] = None
    other_vector: Optional[List[float]] = None
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    normalized: bool = Field(default=True)
    
    @validator('other_text')
    def validate_other_text(cls, v, values):
        if v is not None and not v.strip():
            raise ValueError('Text cannot be empty if provided')
        return v.strip() if v else v


class EmbeddingSearchRequest(BaseModel):
    """Schema for embedding similarity search."""
    query: Optional[str] = None
    query_embedding: Optional[List[float]] = None
    embedding_type: Optional[EmbeddingType] = None
    model: Optional[EmbeddingModel] = None
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    
    @validator('query')
    def validate_query(cls, v, values):
        if v is None and values.get('query_embedding') is None:
            raise ValueError('Either query or query_embedding must be provided')
        return v


class EmbeddingInDBBase(EmbeddingBase):
    """Base schema for embedding in database."""
    id: str
    dimension: int
    text_hash: str
    text_preview: Optional[str]
    status: EmbeddingStatus
    is_normalized: bool
    similarity_threshold: float
    source_info: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class Embedding(EmbeddingInDBBase):
    """Schema for embedding API responses."""
    vector: Optional[List[float]] = None
    is_expired: bool
    is_cacheable: bool
    expires_at: Optional[datetime]
    owner_id: Optional[str]
    
    class Config:
        from_attributes = True


class EmbeddingSimilaritySchema(BaseModel):
    """Schema for embedding similarity responses."""
    embedding_a_id: str
    embedding_b_id: str
    similarity_score: float
    calculation_method: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class EmbeddingBatchResponse(BaseModel):
    """Schema for batch embedding responses."""
    embeddings: List[Embedding]
    total: int
    processing_time: Optional[float]
    
    class Config:
        from_attributes = True