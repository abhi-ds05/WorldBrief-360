"""
Base classes for embedding models.
Defines the abstract interface that all embedding models must implement.
"""
import asyncio
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Callable,
    Iterator, AsyncIterator
)

import numpy as np
from pydantic import BaseModel, Field, validator

from ...base import ModelType, ModelFramework, ModelDevice

logger = logging.getLogger(__name__)


class EmbeddingType(Enum):
    """Types of embeddings."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"


class EmbeddingNormalization(Enum):
    """Embedding normalization methods."""
    NONE = "none"
    L2 = "l2"
    L1 = "l1"
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"


class EmbeddingPooling(Enum):
    """Pooling strategies for token embeddings."""
    MEAN = "mean"
    MAX = "max"
    CLS = "cls"  # Use [CLS] token
    WEIGHTED_MEAN = "weighted_mean"
    ATTENTION = "attention"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str
    model_type: EmbeddingType = EmbeddingType.TEXT
    device: ModelDevice = ModelDevice.AUTO
    batch_size: int = 32
    max_length: int = 512
    normalize: bool = True
    normalization_method: EmbeddingNormalization = EmbeddingNormalization.L2
    pooling_method: EmbeddingPooling = EmbeddingPooling.MEAN
    truncation: bool = True
    padding: bool = True
    return_tensors: str = "np"  # "np", "pt", "tf"
    cache_embeddings: bool = True
    cache_size: int = 10000
    use_gpu: bool = True
    precision: str = "float32"  # "float32", "float16", "bfloat16"
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    embeddings: np.ndarray
    model_name: str
    model_version: str
    input_texts: List[str]
    input_tokens: Optional[List[List[int]]] = None
    attention_masks: Optional[np.ndarray] = None
    embedding_dim: int = 0
    processing_time_ms: float = 0.0
    batch_size: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize derived fields."""
        if self.embeddings is not None and len(self.embeddings.shape) == 2:
            self.embedding_dim = self.embeddings.shape[1]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else [],
            "model_name": self.model_name,
            "model_version": self.model_version,
            "input_texts": self.input_texts,
            "input_tokens": self.input_tokens,
            "attention_masks": self.attention_masks.tolist() if self.attention_masks is not None else [],
            "embedding_dim": self.embedding_dim,
            "processing_time_ms": self.processing_time_ms,
            "batch_size": self.batch_size,
            "metadata": self.metadata
        }
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get shape of embeddings."""
        return self.embeddings.shape if self.embeddings is not None else (0, 0)
    
    @property
    def size(self) -> int:
        """Get total number of embedding vectors."""
        return self.embeddings.shape[0] if self.embeddings is not None else 0


class EmbeddingMetadata(BaseModel):
    """Metadata for embedding models."""
    model_name: str
    model_type: EmbeddingType
    embedding_dim: int
    max_tokens: int
    supported_languages: List[str] = Field(default_factory=lambda: ["en"])
    requires_tokenization: bool = True
    is_multilingual: bool = False
    is_contextual: bool = True  # Whether embeddings are context-dependent
    produces_normalized_embeddings: bool = False
    license: Optional[str] = None
    citation: Optional[str] = None
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True
        json_encoders = {EmbeddingType: lambda x: x.value}


class BaseEmbeddingModel(ABC):
    """
    Abstract base class for all embedding models.
    
    This class defines the interface that all embedding models must implement.
    It provides common functionality for embedding generation, caching, and utilities.
    """
    
    def __init__(
        self,
        model_name: str,
        config: Optional[EmbeddingConfig] = None,
        **kwargs
    ):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the embedding model
            config: Configuration for embedding generation
            **kwargs: Additional arguments passed to configuration
        """
        self.model_name = model_name
        
        # Create config with defaults and overrides
        if config is None:
            config = EmbeddingConfig(model_name=model_name)
        
        # Update config with kwargs
        config_dict = {**config.__dict__, **kwargs}
        self.config = EmbeddingConfig(**config_dict)
        
        # Initialize model
        self.model = None
        self.tokenizer = None
        self.processor = None
        
        # Metadata
        self.metadata = EmbeddingMetadata(
            model_name=model_name,
            model_type=self.config.model_type,
            embedding_dim=0,  # Will be set after model loading
            max_tokens=self.config.max_length
        )
        
        # Cache for embeddings
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Model state
        self._is_loaded = False
        self._device = None
        
        logger.info(f"Initialized embedding model: {model_name}")
    
    @abstractmethod
    def load(self) -> None:
        """
        Load the embedding model and necessary components.
        
        This method should be implemented by subclasses to load the actual model,
        tokenizer, and any other required components.
        """
        pass
    
    @abstractmethod
    def embed(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> EmbeddingResult:
        """
        Generate embeddings for input texts.
        
        Args:
            texts: Single text or list of texts to embed
            **kwargs: Additional arguments for embedding generation
            
        Returns:
            EmbeddingResult containing the embeddings and metadata
            
        Raises:
            ValueError: If model is not loaded
            RuntimeError: If embedding generation fails
        """
        pass
    
    async def embed_async(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> EmbeddingResult:
        """
        Asynchronously generate embeddings for input texts.
        
        Args:
            texts: Single text or list of texts to embed
            **kwargs: Additional arguments for embedding generation
            
        Returns:
            EmbeddingResult containing the embeddings and metadata
        """
        # Run sync version in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.embed(texts, **kwargs)
        )
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """
        Get the dimensionality of embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        pass
    
    @abstractmethod
    def tokenize(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Tokenize input texts.
        
        Args:
            texts: Single text or list of texts to tokenize
            **kwargs: Additional arguments for tokenization
            
        Returns:
            Dictionary containing tokenized inputs
        """
        pass
    
    def normalize_embeddings(
        self,
        embeddings: np.ndarray,
        method: EmbeddingNormalization = None
    ) -> np.ndarray:
        """
        Normalize embeddings using the specified method.
        
        Args:
            embeddings: Input embeddings
            method: Normalization method (uses config if None)
            
        Returns:
            Normalized embeddings
        """
        if method is None:
            method = self.config.normalization_method
        
        if method == EmbeddingNormalization.NONE or not self.config.normalize:
            return embeddings
        
        if method == EmbeddingNormalization.L2:
            # L2 normalization
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)  # Avoid division by zero
            return embeddings / norms
        
        elif method == EmbeddingNormalization.L1:
            # L1 normalization
            norms = np.sum(np.abs(embeddings), axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            return embeddings / norms
        
        elif method == EmbeddingNormalization.MIN_MAX:
            # Min-max normalization per dimension
            min_vals = np.min(embeddings, axis=0)
            max_vals = np.max(embeddings, axis=0)
            ranges = max_vals - min_vals
            ranges = np.where(ranges == 0, 1.0, ranges)  # Avoid division by zero
            return (embeddings - min_vals) / ranges
        
        elif method == EmbeddingNormalization.Z_SCORE:
            # Z-score normalization
            mean = np.mean(embeddings, axis=0)
            std = np.std(embeddings, axis=0)
            std = np.where(std == 0, 1.0, std)  # Avoid division by zero
            return (embeddings - mean) / std
        
        else:
            logger.warning(f"Unknown normalization method: {method}, using L2")
            return self.normalize_embeddings(embeddings, EmbeddingNormalization.L2)
    
    def compute_similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
        metric: str = "cosine"
    ) -> np.ndarray:
        """
        Compute similarity between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            metric: Similarity metric ("cosine", "dot", "euclidean", "manhattan")
            
        Returns:
            Similarity matrix
            
        Raises:
            ValueError: If embeddings have different dimensions
        """
        if embeddings1.shape[1] != embeddings2.shape[1]:
            raise ValueError(
                f"Embedding dimensions must match: "
                f"{embeddings1.shape[1]} != {embeddings2.shape[1]}"
            )
        
        if metric == "cosine":
            # Cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            return cosine_similarity(embeddings1, embeddings2)
        
        elif metric == "dot":
            # Dot product similarity
            return np.dot(embeddings1, embeddings2.T)
        
        elif metric == "euclidean":
            # Euclidean distance (converted to similarity: 1 / (1 + distance))
            from sklearn.metrics.pairwise import euclidean_distances
            distances = euclidean_distances(embeddings1, embeddings2)
            return 1.0 / (1.0 + distances)
        
        elif metric == "manhattan":
            # Manhattan distance (converted to similarity)
            from sklearn.metrics.pairwise import manhattan_distances
            distances = manhattan_distances(embeddings1, embeddings2)
            return 1.0 / (1.0 + distances)
        
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    
    def find_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        candidate_texts: Optional[List[str]] = None,
        top_k: int = 5,
        metric: str = "cosine",
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Find most similar embeddings to a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: Candidate embeddings to search
            candidate_texts: Optional texts corresponding to candidates
            top_k: Number of top results to return
            metric: Similarity metric
            threshold: Optional similarity threshold
            
        Returns:
            List of dictionaries containing similarity results
        """
        # Ensure query_embedding is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Compute similarities
        similarities = self.compute_similarity(
            query_embedding, candidate_embeddings, metric
        )[0]  # Get first row for single query
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Prepare results
        results = []
        for idx in top_indices:
            similarity = float(similarities[idx])
            
            # Skip if below threshold
            if threshold is not None and similarity < threshold:
                continue
            
            result = {
                "index": int(idx),
                "similarity": similarity,
                "embedding": candidate_embeddings[idx] if candidate_embeddings is not None else None,
            }
            
            if candidate_texts is not None and idx < len(candidate_texts):
                result["text"] = candidate_texts[idx]
            
            results.append(result)
        
        return results
    
    def batch_embed(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        **kwargs
    ) -> EmbeddingResult:
        """
        Generate embeddings in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size (uses config if None)
            show_progress: Whether to show progress bar
            **kwargs: Additional arguments for embedding
            
        Returns:
            Combined EmbeddingResult for all batches
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        if len(texts) == 0:
            return EmbeddingResult(
                embeddings=np.array([]),
                model_name=self.model_name,
                model_version=self.get_version(),
                input_texts=[],
                embedding_dim=self.get_embedding_dim()
            )
        
        # Initialize results
        all_embeddings = []
        all_input_tokens = []
        all_attention_masks = []
        total_time = 0.0
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Check cache first
            cached_embeddings = []
            texts_to_embed = []
            cache_indices = []
            
            for j, text in enumerate(batch_texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self._embedding_cache:
                    cached_embeddings.append(self._embedding_cache[cache_key])
                    self._cache_hits += 1
                else:
                    texts_to_embed.append(text)
                    cache_indices.append(j)
            
            # Generate embeddings for non-cached texts
            if texts_to_embed:
                result = self.embed(texts_to_embed, **kwargs)
                self._cache_misses += len(texts_to_embed)
                
                # Cache new embeddings
                for text, embedding in zip(texts_to_embed, result.embeddings):
                    cache_key = self._get_cache_key(text)
                    self._embedding_cache[cache_key] = embedding
                
                # Store in appropriate positions
                batch_embeddings = np.zeros((len(batch_texts), result.embedding_dim))
                batch_input_tokens = [None] * len(batch_texts) if result.input_tokens is None else []
                batch_attention_masks = None if result.attention_masks is None else np.zeros((len(batch_texts), result.attention_masks.shape[1]))
                
                # Fill with cached embeddings
                cache_idx = 0
                for j in range(len(batch_texts)):
                    if j in cache_indices:
                        # New embedding
                        rel_idx = cache_indices.index(j)
                        batch_embeddings[j] = result.embeddings[rel_idx]
                        if batch_input_tokens is not None and result.input_tokens is not None:
                            batch_input_tokens.append(result.input_tokens[rel_idx])
                        if batch_attention_masks is not None:
                            batch_attention_masks[j] = result.attention_masks[rel_idx]
                        cache_idx += 1
                    else:
                        # Cached embedding
                        rel_idx = j - cache_idx
                        batch_embeddings[j] = cached_embeddings[rel_idx]
                        if batch_input_tokens is not None:
                            batch_input_tokens.append(None)
            else:
                # All embeddings were cached
                batch_embeddings = np.array(cached_embeddings)
                batch_input_tokens = None
                batch_attention_masks = None
            
            # Accumulate results
            all_embeddings.append(batch_embeddings)
            if batch_input_tokens is not None:
                all_input_tokens.extend(batch_input_tokens)
            if batch_attention_masks is not None:
                all_attention_masks.append(batch_attention_masks)
            
            total_time += result.processing_time_ms
        
        # Combine all batches
        final_embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
        final_attention_masks = np.vstack(all_attention_masks) if all_attention_masks else None
        
        return EmbeddingResult(
            embeddings=final_embeddings,
            model_name=self.model_name,
            model_version=self.get_version(),
            input_texts=texts,
            input_tokens=all_input_tokens if all_input_tokens else None,
            attention_masks=final_attention_masks,
            embedding_dim=self.get_embedding_dim(),
            processing_time_ms=total_time,
            batch_size=batch_size,
            metadata={
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "total_batches": (len(texts) + batch_size - 1) // batch_size
            }
        )
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key for a text.
        
        Args:
            text: Input text
            
        Returns:
            Cache key
        """
        import hashlib
        # Include model name and config in key to avoid collisions
        base = f"{self.model_name}_{self.config.max_length}_{text}"
        return hashlib.sha256(base.encode()).hexdigest()[:32]
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Cleared embedding cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "cache_size": len(self._embedding_cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": self._cache_hits / max(self._cache_hits + self._cache_misses, 1),
            "cache_memory_mb": self._estimate_cache_memory() / (1024 * 1024)
        }
    
    def _estimate_cache_memory(self) -> int:
        """
        Estimate memory usage of cache.
        
        Returns:
            Estimated memory in bytes
        """
        total_bytes = 0
        for embedding in self._embedding_cache.values():
            total_bytes += embedding.nbytes
        return total_bytes
    
    def is_loaded(self) -> bool:
        """
        Check if model is loaded.
        
        Returns:
            True if model is loaded
        """
        return self._is_loaded
    
    def get_metadata(self) -> EmbeddingMetadata:
        """
        Get model metadata.
        
        Returns:
            EmbeddingMetadata object
        """
        return self.metadata
    
    def get_config(self) -> EmbeddingConfig:
        """
        Get model configuration.
        
        Returns:
            EmbeddingConfig object
        """
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """
        Update model configuration.
        
        Args:
            **kwargs: Configuration updates
        """
        current_config = self.config.__dict__
        updated_config = {**current_config, **kwargs}
        self.config = EmbeddingConfig(**updated_config)
        logger.info(f"Updated embedding model config: {kwargs}")
    
    @abstractmethod
    def get_version(self) -> str:
        """
        Get model version.
        
        Returns:
            Model version string
        """
        pass
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        filepath: str,
        format: str = "numpy"
    ) -> None:
        """
        Save embeddings to file.
        
        Args:
            embeddings: Embeddings to save
            filepath: Path to save file
            format: File format ("numpy", "json", "csv")
        """
        if format == "numpy":
            np.save(filepath, embeddings)
        elif format == "json":
            import json
            with open(filepath, 'w') as f:
                json.dump(embeddings.tolist(), f)
        elif format == "csv":
            np.savetxt(filepath, embeddings, delimiter=',')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved embeddings to {filepath} in {format} format")
    
    @staticmethod
    def load_embeddings(
        filepath: str,
        format: str = "numpy"
    ) -> np.ndarray:
        """
        Load embeddings from file.
        
        Args:
            filepath: Path to embeddings file
            format: File format ("numpy", "json", "csv")
            
        Returns:
            Loaded embeddings
        """
        if format == "numpy":
            return np.load(filepath)
        elif format == "json":
            import json
            with open(filepath, 'r') as f:
                return np.array(json.load(f))
        elif format == "csv":
            return np.loadtxt(filepath, delimiter=',')
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def __call__(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> EmbeddingResult:
        """
        Make the model callable.
        
        Args:
            texts: Input texts
            **kwargs: Additional arguments
            
        Returns:
            EmbeddingResult
        """
        return self.embed(texts, **kwargs)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"BaseEmbeddingModel(model_name={self.model_name}, loaded={self._is_loaded})"
    
    def __len__(self) -> int:
        """Get embedding dimension."""
        return self.get_embedding_dim()
    
    def __enter__(self):
        """Context manager entry."""
        if not self._is_loaded:
            self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()
    
    def unload(self) -> None:
        """Unload model from memory."""
        if self._is_loaded:
            del self.model
            del self.tokenizer
            del self.processor
            self.model = None
            self.tokenizer = None
            self.processor = None
            self._is_loaded = False
            
            # Clear cache
            self.clear_cache()
            
            # Free GPU memory if applicable
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Unloaded embedding model: {self.model_name}")


class EmbeddingModelFactory:
    """
    Factory for creating embedding model instances.
    
    This class provides a unified interface for creating different types
    of embedding models based on configuration.
    """
    
    # Registry of embedding model classes
    _model_registry: Dict[str, type] = {}
    
    @classmethod
    def register_model(cls, model_type: str, model_class: type) -> None:
        """
        Register an embedding model class.
        
        Args:
            model_type: Type identifier for the model
            model_class: The model class to register
        """
        cls._model_registry[model_type] = model_class
        logger.info(f"Registered embedding model type: {model_type} -> {model_class.__name__}")
    
    @classmethod
    def create(
        cls,
        model_type: str,
        model_name: str,
        config: Optional[EmbeddingConfig] = None,
        **kwargs
    ) -> BaseEmbeddingModel:
        """
        Create an embedding model instance.
        
        Args:
            model_type: Type of embedding model
            model_name: Name of the model
            config: Model configuration
            **kwargs: Additional arguments
            
        Returns:
            Embedding model instance
            
        Raises:
            ValueError: If model_type is not registered
        """
        if model_type not in cls._model_registry:
            raise ValueError(
                f"Unknown embedding model type: {model_type}. "
                f"Available types: {list(cls._model_registry.keys())}"
            )
        
        model_class = cls._model_registry[model_type]
        
        if config is None:
            config = EmbeddingConfig(model_name=model_name)
        
        # Create model instance
        instance = model_class(model_name=model_name, config=config, **kwargs)
        
        # Load the model
        if not instance.is_loaded():
            instance.load()
        
        return instance
    
    @classmethod
    def list_available_models(cls) -> Dict[str, List[str]]:
        """
        List available model types and their implementations.
        
        Returns:
            Dictionary mapping model types to available implementations
        """
        result = {}
        for model_type, model_class in cls._model_registry.items():
            result[model_type] = model_class.__name__
        return result


# Default model implementations will be registered by their respective modules
__all__ = [
    'BaseEmbeddingModel',
    'EmbeddingConfig',
    'EmbeddingResult',
    'EmbeddingType',
    'EmbeddingNormalization',
    'EmbeddingPooling',
    'EmbeddingMetadata',
    'EmbeddingModelFactory',
]