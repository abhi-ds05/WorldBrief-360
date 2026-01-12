"""
Embedding models module.
Provides unified interface for various embedding models including:
- Sentence Transformers
- OpenAI Embeddings
- Multilingual embeddings
- Custom embedding models
"""

from .base import BaseEmbeddingModel, EmbeddingResult
from .sentence_transformers import SentenceTransformerEmbedding
from .openai_embeddings import OpenAIEmbedding
from .multilingual import MultilingualEmbedding

__all__ = [
    'BaseEmbeddingModel',
    'EmbeddingResult',
    'SentenceTransformerEmbedding',
    'OpenAIEmbedding',
    'MultilingualEmbedding',
]


def get_embedding_model(
    model_name: str,
    model_type: str = "sentence_transformers",
    **kwargs
):
    """
    Factory function to get an embedding model instance.
    
    Args:
        model_name: Name of the embedding model
        model_type: Type of embedding model ("sentence_transformers", "openai", "multilingual")
        **kwargs: Additional arguments for model initialization
        
    Returns:
        An instance of the embedding model
        
    Raises:
        ValueError: If model_type is not supported
    """
    model_type = model_type.lower()
    
    if model_type == "sentence_transformers":
        return SentenceTransformerEmbedding(model_name=model_name, **kwargs)
    elif model_type == "openai":
        return OpenAIEmbedding(model_name=model_name, **kwargs)
    elif model_type == "multilingual":
        return MultilingualEmbedding(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported embedding model type: {model_type}")


# Default embedding models configuration
DEFAULT_EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "type": "sentence_transformers",
        "description": "Lightweight general-purpose embedding model",
        "dimensions": 384,
        "max_tokens": 256,
    },
    "all-mpnet-base-v2": {
        "type": "sentence_transformers",
        "description": "High-quality general-purpose embedding model",
        "dimensions": 768,
        "max_tokens": 384,
    },
    "text-embedding-ada-002": {
        "type": "openai",
        "description": "OpenAI's Ada embedding model",
        "dimensions": 1536,
        "max_tokens": 8191,
    },
    "paraphrase-multilingual-MiniLM-L12-v2": {
        "type": "multilingual",
        "description": "Multilingual embedding model",
        "dimensions": 384,
        "max_tokens": 128,
        "languages": ["en", "es", "de", "fr", "it", "nl", "pl", "pt", "ru", "zh"],
    },
}


def list_available_models() -> dict:
    """
    List all available embedding models and their configurations.
    
    Returns:
        Dictionary of available models
    """
    return DEFAULT_EMBEDDING_MODELS.copy()


def get_model_info(model_name: str) -> dict:
    """
    Get information about a specific embedding model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model configuration dictionary
        
    Raises:
        KeyError: If model is not found
    """
    if model_name not in DEFAULT_EMBEDDING_MODELS:
        raise KeyError(f"Model not found: {model_name}")
    
    return DEFAULT_EMBEDDING_MODELS[model_name].copy()