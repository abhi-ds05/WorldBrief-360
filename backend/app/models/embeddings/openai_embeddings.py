"""
OpenAI embedding models integration.
Supports various OpenAI embedding models including:
- text-embedding-ada-002
- text-embedding-3-small
- text-embedding-3-large
"""
import asyncio
import logging
import time
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import openai
from openai import AsyncOpenAI, OpenAI

from .base import (
    BaseEmbeddingModel, EmbeddingConfig, EmbeddingResult,
    EmbeddingMetadata, EmbeddingType, EmbeddingNormalization,
    EmbeddingPooling
)
from ...base import ModelDevice

logger = logging.getLogger(__name__)


class OpenAIEmbedding(BaseEmbeddingModel):
    """
    OpenAI embedding model wrapper.
    
    Supports both synchronous and asynchronous API calls to OpenAI's embedding models.
    """
    
    # OpenAI embedding model information
    MODEL_INFO = {
        'text-embedding-ada-002': {
            'embedding_dim': 1536,
            'max_tokens': 8191,
            'description': 'Second generation embedding model from OpenAI',
            'context_window': 8191,
            'input_cost_per_1k_tokens': 0.0001,  # USD
            'output_cost_per_1k_tokens': 0.0001,  # USD
            'performance': 'good',
        },
        'text-embedding-3-small': {
            'embedding_dim': 1536,  # Can be reduced to 512, 256, etc.
            'max_tokens': 8191,
            'description': 'Third generation small embedding model',
            'context_window': 8191,
            'input_cost_per_1k_tokens': 0.00002,  # USD
            'output_cost_per_1k_tokens': 0.00002,  # USD
            'performance': 'excellent',
        },
        'text-embedding-3-large': {
            'embedding_dim': 3072,  # Can be reduced to 1024, 256, etc.
            'max_tokens': 8191,
            'description': 'Third generation large embedding model',
            'context_window': 8191,
            'input_cost_per_1k_tokens': 0.00013,  # USD
            'output_cost_per_1k_tokens': 0.00013,  # USD
            'performance': 'excellent',
        },
        'text-embedding-ada-002-v2': {
            'embedding_dim': 1536,
            'max_tokens': 8191,
            'description': 'Updated version of text-embedding-ada-002',
            'context_window': 8191,
            'input_cost_per_1k_tokens': 0.0001,
            'output_cost_per_1k_tokens': 0.0001,
            'performance': 'good',
        },
    }
    
    def __init__(
        self,
        model_name: str = 'text-embedding-ada-002',
        config: Optional[EmbeddingConfig] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        **kwargs
    ):
        """
        Initialize OpenAI embedding model.
        
        Args:
            model_name: Name of the OpenAI embedding model
            config: Configuration for embedding generation
            api_key: OpenAI API key (uses environment variable if not provided)
            organization: OpenAI organization ID
            base_url: Custom base URL for API (for Azure OpenAI or proxies)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            **kwargs: Additional arguments passed to configuration
        """
        # Set default config for OpenAI models
        if config is None:
            config = EmbeddingConfig(
                model_name=model_name,
                model_type=EmbeddingType.TEXT,
                max_length=8191,  # OpenAI's maximum
                normalize=False,  # OpenAI returns normalized embeddings
                normalization_method=EmbeddingNormalization.NONE,
                pooling_method=EmbeddingPooling.NONE,  # No pooling needed for OpenAI
                batch_size=2048,  # OpenAI's recommended batch size for embeddings
                cache_embeddings=True,
                cache_size=10000,
            )
        
        # Update with model-specific defaults from MODEL_INFO
        if model_name in self.MODEL_INFO:
            model_info = self.MODEL_INFO[model_name]
            config.max_length = model_info.get('max_tokens', config.max_length)
        
        super().__init__(model_name, config, **kwargs)
        
        # OpenAI client configuration
        self.api_key = api_key
        self.organization = organization
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        
        # OpenAI clients
        self._client: Optional[OpenAI] = None
        self._async_client: Optional[AsyncOpenAI] = None
        
        # API usage tracking
        self._total_tokens_used = 0
        self._total_requests = 0
        self._total_cost = 0.0
        
        # Rate limiting
        self._rate_limit_remaining = None
        self._rate_limit_reset = None
        
        # Update metadata
        self._update_metadata()
        
        logger.info(f"Initialized OpenAI embedding model: {model_name}")
    
    def _update_metadata(self) -> None:
        """Update metadata based on model name."""
        model_info = self.MODEL_INFO.get(
            self.model_name,
            {
                'embedding_dim': 1536,
                'max_tokens': self.config.max_length,
                'description': 'OpenAI embedding model',
                'context_window': 8191,
                'input_cost_per_1k_tokens': 0.0001,
                'output_cost_per_1k_tokens': 0.0001,
                'performance': 'unknown',
            }
        )
        
        self.metadata = EmbeddingMetadata(
            model_name=self.model_name,
            model_type=self.config.model_type,
            embedding_dim=model_info['embedding_dim'],
            max_tokens=model_info['max_tokens'],
            supported_languages=['en'],  # OpenAI models are English-optimized
            requires_tokenization=False,  # OpenAI handles tokenization
            is_multilingual=False,
            is_contextual=True,
            produces_normalized_embeddings=True,  # OpenAI returns normalized embeddings
            license='OpenAI',
            citation='OpenAI Embedding Models',
        )
    
    def load(self) -> None:
        """
        Initialize the OpenAI client.
        
        Note: OpenAI models don't need to be "loaded" like local models,
        but we need to initialize the API client.
        """
        if self._is_loaded:
            logger.warning(f"OpenAI client for {self.model_name} is already initialized")
            return
        
        try:
            logger.info(f"Initializing OpenAI client for model: {self.model_name}")
            
            # Initialize synchronous client
            self._client = OpenAI(
                api_key=self.api_key,
                organization=self.organization,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
            
            # Initialize asynchronous client
            self._async_client = AsyncOpenAI(
                api_key=self.api_key,
                organization=self.organization,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
            
            self._is_loaded = True
            
            logger.info(f"Successfully initialized OpenAI client for {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
    
    def embed(
        self,
        texts: Union[str, List[str]],
        dimensions: Optional[int] = None,
        encoding_format: str = "float",
        user: Optional[str] = None,
        **kwargs
    ) -> EmbeddingResult:
        """
        Generate embeddings using OpenAI API.
        
        Args:
            texts: Single text or list of texts to embed
            dimensions: Optional dimensions for embedding (only for v3 models)
            encoding_format: Format of embeddings ("float" or "base64")
            user: Unique identifier for end-user (for abuse monitoring)
            **kwargs: Additional arguments for embedding generation
            
        Returns:
            EmbeddingResult containing the embeddings and metadata
            
        Raises:
            ValueError: If client is not initialized
            RuntimeError: If embedding generation fails
        """
        if not self._is_loaded:
            raise ValueError(f"OpenAI client for {self.model_name} is not initialized. Call load() first.")
        
        # Convert single text to list
        single_text = False
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        
        if len(texts) == 0:
            return EmbeddingResult(
                embeddings=np.array([]),
                model_name=self.model_name,
                model_version=self.get_version(),
                input_texts=[],
                embedding_dim=self.get_embedding_dim(dimensions),
            )
        
        try:
            import time
            start_time = time.time()
            
            # Prepare API parameters
            params = {
                'model': self.model_name,
                'input': texts,
                'encoding_format': encoding_format,
            }
            
            # Add dimensions for v3 models if specified
            if dimensions is not None and self.model_name.startswith('text-embedding-3'):
                params['dimensions'] = dimensions
            
            # Add user if provided
            if user:
                params['user'] = user
            
            # Make API call
            response = self._client.embeddings.create(**params)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Extract embeddings
            if encoding_format == "float":
                embeddings = np.array([data.embedding for data in response.data])
            elif encoding_format == "base64":
                # Decode base64 embeddings
                import base64
                embeddings = []
                for data in response.data:
                    decoded = base64.b64decode(data.embedding)
                    # Convert bytes to float array
                    # This depends on the exact format - assuming float32 little-endian
                    arr = np.frombuffer(decoded, dtype=np.float32)
                    embeddings.append(arr)
                embeddings = np.array(embeddings)
            else:
                raise ValueError(f"Unsupported encoding format: {encoding_format}")
            
            # Update usage statistics
            self._update_usage_stats(response.usage)
            
            # Update rate limiting information
            self._update_rate_limits(response)
            
            # Prepare result
            result = EmbeddingResult(
                embeddings=embeddings,
                model_name=self.model_name,
                model_version=self.get_version(),
                input_texts=texts if not single_text else texts[0],
                embedding_dim=embeddings.shape[1] if len(embeddings) > 0 else 0,
                processing_time_ms=processing_time_ms,
                batch_size=len(texts),
                metadata={
                    'api_usage': self._get_usage_metadata(response.usage),
                    'rate_limit_info': self._get_rate_limit_metadata(),
                    'encoding_format': encoding_format,
                    'dimensions': dimensions,
                    'model_info': self.MODEL_INFO.get(self.model_name, {}),
                }
            )
            
            # Cache embeddings if enabled (for float format only)
            if (self.config.cache_embeddings and 
                encoding_format == "float" and 
                len(self._embedding_cache) < self.config.cache_size):
                for i, text in enumerate(texts):
                    cache_key = self._get_cache_key(text)
                    if cache_key not in self._embedding_cache:
                        self._embedding_cache[cache_key] = embeddings[i]
            
            return result
            
        except openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded: {e}")
            raise RuntimeError(f"Rate limit exceeded: {e}")
        except openai.APIConnectionError as e:
            logger.error(f"OpenAI API connection error: {e}")
            raise RuntimeError(f"API connection error: {e}")
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"API error: {e}")
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {e}")
    
    async def embed_async(
        self,
        texts: Union[str, List[str]],
        dimensions: Optional[int] = None,
        encoding_format: str = "float",
        user: Optional[str] = None,
        **kwargs
    ) -> EmbeddingResult:
        """
        Asynchronously generate embeddings using OpenAI API.
        
        Args:
            texts: Single text or list of texts to embed
            dimensions: Optional dimensions for embedding (only for v3 models)
            encoding_format: Format of embeddings ("float" or "base64")
            user: Unique identifier for end-user (for abuse monitoring)
            **kwargs: Additional arguments for embedding generation
            
        Returns:
            EmbeddingResult containing the embeddings and metadata
        """
        if not self._is_loaded:
            raise ValueError(f"OpenAI client for {self.model_name} is not initialized. Call load() first.")
        
        # Convert single text to list
        single_text = False
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        
        if len(texts) == 0:
            return EmbeddingResult(
                embeddings=np.array([]),
                model_name=self.model_name,
                model_version=self.get_version(),
                input_texts=[],
                embedding_dim=self.get_embedding_dim(dimensions),
            )
        
        try:
            import time
            start_time = time.time()
            
            # Prepare API parameters
            params = {
                'model': self.model_name,
                'input': texts,
                'encoding_format': encoding_format,
            }
            
            # Add dimensions for v3 models if specified
            if dimensions is not None and self.model_name.startswith('text-embedding-3'):
                params['dimensions'] = dimensions
            
            # Add user if provided
            if user:
                params['user'] = user
            
            # Make async API call
            response = await self._async_client.embeddings.create(**params)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Extract embeddings
            if encoding_format == "float":
                embeddings = np.array([data.embedding for data in response.data])
            elif encoding_format == "base64":
                # Decode base64 embeddings
                import base64
                embeddings = []
                for data in response.data:
                    decoded = base64.b64decode(data.embedding)
                    arr = np.frombuffer(decoded, dtype=np.float32)
                    embeddings.append(arr)
                embeddings = np.array(embeddings)
            else:
                raise ValueError(f"Unsupported encoding format: {encoding_format}")
            
            # Update usage statistics
            self._update_usage_stats(response.usage)
            
            # Update rate limiting information
            self._update_rate_limits(response)
            
            # Prepare result
            result = EmbeddingResult(
                embeddings=embeddings,
                model_name=self.model_name,
                model_version=self.get_version(),
                input_texts=texts if not single_text else texts[0],
                embedding_dim=embeddings.shape[1] if len(embeddings) > 0 else 0,
                processing_time_ms=processing_time_ms,
                batch_size=len(texts),
                metadata={
                    'api_usage': self._get_usage_metadata(response.usage),
                    'rate_limit_info': self._get_rate_limit_metadata(),
                    'encoding_format': encoding_format,
                    'dimensions': dimensions,
                    'model_info': self.MODEL_INFO.get(self.model_name, {}),
                }
            )
            
            # Cache embeddings if enabled
            if (self.config.cache_embeddings and 
                encoding_format == "float" and 
                len(self._embedding_cache) < self.config.cache_size):
                for i, text in enumerate(texts):
                    cache_key = self._get_cache_key(text)
                    if cache_key not in self._embedding_cache:
                        self._embedding_cache[cache_key] = embeddings[i]
            
            return result
            
        except openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded: {e}")
            raise RuntimeError(f"Rate limit exceeded: {e}")
        except openai.APIConnectionError as e:
            logger.error(f"OpenAI API connection error: {e}")
            raise RuntimeError(f"API connection error: {e}")
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"API error: {e}")
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {e}")
    
    def _update_usage_stats(self, usage) -> None:
        """Update usage statistics from API response."""
        if hasattr(usage, 'total_tokens'):
            self._total_tokens_used += usage.total_tokens
        
        self._total_requests += 1
        
        # Calculate cost
        model_info = self.MODEL_INFO.get(self.model_name, {})
        input_cost_per_token = model_info.get('input_cost_per_1k_tokens', 0) / 1000
        output_cost_per_token = model_info.get('output_cost_per_1k_tokens', 0) / 1000
        
        # For embeddings, we only have total_tokens, not separated by input/output
        # Assume all tokens are input tokens for cost calculation
        if hasattr(usage, 'total_tokens'):
            self._total_cost += usage.total_tokens * input_cost_per_token
    
    def _update_rate_limits(self, response) -> None:
        """Update rate limiting information from response headers."""
        try:
            # Extract rate limit info from response headers
            if hasattr(response, '_headers'):
                headers = response._headers
                
                if 'x-ratelimit-remaining-requests' in headers:
                    self._rate_limit_remaining = int(headers['x-ratelimit-remaining-requests'])
                
                if 'x-ratelimit-reset-requests' in headers:
                    self._rate_limit_reset = headers['x-ratelimit-reset-requests']
        
        except Exception as e:
            logger.debug(f"Could not extract rate limit info: {e}")
    
    def _get_usage_metadata(self, usage) -> Dict[str, Any]:
        """Extract usage metadata from API response."""
        if usage is None:
            return {}
        
        metadata = {}
        
        if hasattr(usage, 'prompt_tokens'):
            metadata['prompt_tokens'] = usage.prompt_tokens
        
        if hasattr(usage, 'total_tokens'):
            metadata['total_tokens'] = usage.total_tokens
        
        # Add cost estimate
        model_info = self.MODEL_INFO.get(self.model_name, {})
        input_cost_per_1k = model_info.get('input_cost_per_1k_tokens', 0)
        
        if 'total_tokens' in metadata:
            metadata['estimated_cost_usd'] = (metadata['total_tokens'] / 1000) * input_cost_per_1k
        
        return metadata
    
    def _get_rate_limit_metadata(self) -> Dict[str, Any]:
        """Get current rate limit metadata."""
        metadata = {}
        
        if self._rate_limit_remaining is not None:
            metadata['remaining_requests'] = self._rate_limit_remaining
        
        if self._rate_limit_reset is not None:
            metadata['reset_time'] = self._rate_limit_reset
        
        return metadata
    
    def get_embedding_dim(self, dimensions: Optional[int] = None) -> int:
        """
        Get the dimensionality of embeddings.
        
        Args:
            dimensions: Optional target dimensions (for v3 models)
            
        Returns:
            Embedding dimension
        """
        base_dim = self.metadata.embedding_dim
        
        # For v3 models, dimensions can be specified
        if dimensions is not None and self.model_name.startswith('text-embedding-3'):
            # Validate dimensions
            model_info = self.MODEL_INFO.get(self.model_name, {})
            max_dim = model_info.get('embedding_dim', 3072)
            min_dim = 256  # Minimum for v3 models
            
            if dimensions < min_dim or dimensions > max_dim:
                logger.warning(
                    f"Dimensions {dimensions} outside valid range [{min_dim}, {max_dim}]. "
                    f"Using default: {base_dim}"
                )
                return base_dim
            
            return dimensions
        
        return base_dim
    
    def tokenize(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Estimate token count using tiktoken (OpenAI's tokenizer).
        
        Args:
            texts: Single text or list of texts
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing token information
            
        Raises:
            ImportError: If tiktoken is not installed
        """
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken is required for token counting with OpenAI models. "
                "Install with: pip install tiktoken"
            )
        
        single_text = False
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        
        # Get encoding for the model
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            # Fallback to cl100k_base (used by most newer models)
            encoding = tiktoken.get_encoding("cl100k_base")
        
        # Tokenize texts
        token_counts = []
        tokens_list = []
        
        for text in texts:
            tokens = encoding.encode(text)
            token_counts.append(len(tokens))
            tokens_list.append(tokens)
        
        result = {
            'input_ids': tokens_list,
            'token_counts': token_counts,
            'texts': texts if not single_text else texts[0],
            'model_name': self.model_name,
            'encoding_name': encoding.name,
        }
        
        if single_text:
            result['token_count'] = token_counts[0]
        
        return result
    
    def get_version(self) -> str:
        """
        Get model version.
        
        Returns:
            Model version string
        """
        # Extract version from model name if possible
        if 'ada-002' in self.model_name:
            return 'ada-002'
        elif '3-small' in self.model_name:
            return '3-small'
        elif '3-large' in self.model_name:
            return '3-large'
        else:
            return 'unknown'
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get API usage statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        model_info = self.MODEL_INFO.get(self.model_name, {})
        input_cost_per_1k = model_info.get('input_cost_per_1k_tokens', 0)
        
        stats = {
            'total_requests': self._total_requests,
            'total_tokens_used': self._total_tokens_used,
            'total_cost_usd': self._total_cost,
            'average_tokens_per_request': (
                self._total_tokens_used / max(self._total_requests, 1)
            ),
            'cost_per_1k_tokens_usd': input_cost_per_1k,
            'estimated_monthly_cost_usd': self._total_cost * 30,  # Rough estimate
        }
        
        # Add rate limit info
        if self._rate_limit_remaining is not None:
            stats['rate_limit_remaining'] = self._rate_limit_remaining
        
        if self._rate_limit_reset is not None:
            stats['rate_limit_reset'] = self._rate_limit_reset
        
        return stats
    
    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self._total_tokens_used = 0
        self._total_requests = 0
        self._total_cost = 0.0
        logger.info("Reset OpenAI usage statistics")
    
    def estimate_cost(
        self,
        texts: Union[str, List[str]],
        dimensions: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Estimate the cost of embedding generation.
        
        Args:
            texts: Single text or list of texts
            dimensions: Optional dimensions for v3 models
            
        Returns:
            Dictionary with cost estimates
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
        
        # Get token counts
        token_info = self.tokenize(texts)
        
        if isinstance(token_info['token_counts'], list):
            total_tokens = sum(token_info['token_counts'])
        else:
            total_tokens = token_info['token_counts']
        
        # Get cost per token
        model_info = self.MODEL_INFO.get(self.model_name, {})
        input_cost_per_token = model_info.get('input_cost_per_1k_tokens', 0) / 1000
        
        # Calculate cost
        estimated_cost = total_tokens * input_cost_per_token
        
        return {
            'total_texts': len(texts),
            'total_tokens': total_tokens,
            'cost_per_token_usd': input_cost_per_token,
            'estimated_cost_usd': estimated_cost,
            'model': self.model_name,
            'dimensions': dimensions if dimensions else self.get_embedding_dim(),
        }
    
    def get_model_capabilities(self) -> Dict[str, Any]:
        """
        Get capabilities of the OpenAI embedding model.
        
        Returns:
            Dictionary with model capabilities
        """
        model_info = self.MODEL_INFO.get(self.model_name, {})
        
        capabilities = {
            'model': self.model_name,
            'max_tokens': model_info.get('max_tokens', 8191),
            'max_batch_size': 2048,  # OpenAI's recommended batch size
            'supports_dimension_reduction': self.model_name.startswith('text-embedding-3'),
            'supports_base64_encoding': True,
            'returns_normalized_embeddings': True,
            'input_token_cost_per_1k': model_info.get('input_cost_per_1k_tokens', 0),
            'context_window': model_info.get('context_window', 8191),
            'description': model_info.get('description', 'OpenAI embedding model'),
        }
        
        # Add dimension options for v3 models
        if self.model_name.startswith('text-embedding-3'):
            base_dim = model_info.get('embedding_dim', 1536)
            capabilities['available_dimensions'] = {
                'min': 256,
                'max': base_dim,
                'recommended': [256, 512, 1024, base_dim] if base_dim > 1024 else [256, 512, base_dim]
            }
        
        return capabilities
    
    def validate_input(
        self,
        texts: Union[str, List[str]],
        raise_on_error: bool = True
    ) -> Dict[str, Any]:
        """
        Validate input texts for OpenAI API constraints.
        
        Args:
            texts: Single text or list of texts
            raise_on_error: Whether to raise exceptions on validation errors
            
        Returns:
            Dictionary with validation results
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
        
        validation_results = {
            'total_texts': len(texts),
            'valid': True,
            'errors': [],
            'warnings': [],
            'token_counts': [],
            'total_tokens': 0,
        }
        
        model_info = self.MODEL_INFO.get(self.model_name, {})
        max_tokens = model_info.get('max_tokens', 8191)
        max_batch_size = 2048  # OpenAI's recommended batch size
        
        # Check batch size
        if len(texts) > max_batch_size:
            error_msg = f"Batch size {len(texts)} exceeds maximum {max_batch_size}"
            if raise_on_error:
                raise ValueError(error_msg)
            validation_results['valid'] = False
            validation_results['errors'].append(error_msg)
        elif len(texts) > 1000:
            validation_results['warnings'].append(
                f"Large batch size: {len(texts)} texts. Consider batching."
            )
        
        # Check individual texts
        for i, text in enumerate(texts):
            # Tokenize to check length
            try:
                token_info = self.tokenize([text])
                token_count = token_info['token_counts'][0] if isinstance(token_info['token_counts'], list) else token_info['token_counts']
                validation_results['token_counts'].append(token_count)
                validation_results['total_tokens'] += token_count
                
                # Check token limit
                if token_count > max_tokens:
                    error_msg = f"Text {i} exceeds {max_tokens} token limit: {token_count} tokens"
                    if raise_on_error:
                        raise ValueError(error_msg)
                    validation_results['valid'] = False
                    validation_results['errors'].append(error_msg)
                elif token_count > max_tokens * 0.9:
                    validation_results['warnings'].append(
                        f"Text {i} is close to token limit: {token_count}/{max_tokens}"
                    )
                
                # Check for empty text
                if not text.strip():
                    validation_results['warnings'].append(f"Text {i} is empty or whitespace-only")
            
            except Exception as e:
                error_msg = f"Failed to tokenize text {i}: {str(e)}"
                if raise_on_error:
                    raise ValueError(error_msg)
                validation_results['valid'] = False
                validation_results['errors'].append(error_msg)
        
        # Check total tokens for batch
        if validation_results['total_tokens'] > max_tokens * 100:  # Arbitrary limit for batch
            validation_results['warnings'].append(
                f"Large total token count: {validation_results['total_tokens']}"
            )
        
        return validation_results
    
    def batch_embed_with_retry(
        self,
        texts: List[str],
        max_retries: int = 3,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0,
        **kwargs
    ) -> EmbeddingResult:
        """
        Generate embeddings with automatic retry on failure.
        
        Args:
            texts: List of texts to embed
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries in seconds
            backoff_factor: Multiplier for delay between retries
            **kwargs: Additional arguments for embedding
            
        Returns:
            EmbeddingResult containing the embeddings
            
        Raises:
            RuntimeError: If all retry attempts fail
        """
        import time
        
        last_exception = None
        
        for attempt in range(max_retries + 1):  # +1 for the initial attempt
            try:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt}/{max_retries} for embedding {len(texts)} texts")
                    time.sleep(retry_delay * (backoff_factor ** (attempt - 1)))
                
                return self.embed(texts, **kwargs)
                
            except (openai.RateLimitError, openai.APIConnectionError, openai.APIError) as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt == max_retries:
                    logger.error(f"All {max_retries + 1} attempts failed")
                    break
        
        # If we get here, all attempts failed
        raise RuntimeError(
            f"Failed to generate embeddings after {max_retries + 1} attempts. "
            f"Last error: {last_exception}"
        )
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """
        List all available OpenAI embedding models.
        
        Returns:
            Dictionary of model names to model information
        """
        return cls.MODEL_INFO.copy()
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific OpenAI embedding model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information dictionary or None if not found
        """
        return cls.MODEL_INFO.get(model_name)
    
    def unload(self) -> None:
        """Clean up OpenAI clients."""
        if self._is_loaded:
            # Close clients if they have close methods
            if hasattr(self._client, 'close'):
                self._client.close()
            if hasattr(self._async_client, 'close'):
                # Async close should be done in async context
                pass
            
            self._client = None
            self._async_client = None
            self._is_loaded = False
            
            # Clear cache
            self.clear_cache()
            
            logger.info(f"Unloaded OpenAI client for {self.model_name}")
    
    def __repr__(self) -> str:
        """String representation."""
        loaded = self._is_loaded
        dim = self.metadata.embedding_dim
        return (
            f"OpenAIEmbedding(model_name={self.model_name}, "
            f"loaded={loaded}, dim={dim}, requests={self._total_requests})"
        )


# Register with factory
try:
    from .base import EmbeddingModelFactory
    EmbeddingModelFactory.register_model('openai', OpenAIEmbedding)
    logger.info("Registered OpenAIEmbedding with EmbeddingModelFactory")
except ImportError:
    logger.warning("Could not register OpenAIEmbedding with factory")


__all__ = [
    'OpenAIEmbedding',
    'MODEL_INFO',
]