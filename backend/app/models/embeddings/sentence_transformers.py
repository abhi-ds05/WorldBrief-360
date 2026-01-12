"""
Sentence Transformers embedding models.
Supports a wide range of pre-trained sentence embedding models from:
- sentence-transformers library
- Hugging Face Transformers
- Custom fine-tuned models
"""
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

from .base import (
    BaseEmbeddingModel, EmbeddingConfig, EmbeddingResult,
    EmbeddingMetadata, EmbeddingType, EmbeddingNormalization,
    EmbeddingPooling
)
from ...base import ModelDevice

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """
    Sentence Transformers embedding model wrapper.
    
    Supports various pre-trained sentence embedding models with efficient
    pooling strategies and batch processing.
    """
    
    # Popular Sentence Transformers models with their specifications
    MODEL_INFO = {
        # Lightweight models
        'all-MiniLM-L6-v2': {
            'embedding_dim': 384,
            'max_tokens': 256,
            'description': 'Lightweight general-purpose model, good balance of speed and quality',
            'pooling': 'mean',
            'architecture': 'MiniLM',
            'parameters': 22_700_000,
            'performance': 'fast',
            'memory_mb': 90,
            'languages': ['en'],
        },
        'all-MiniLM-L12-v2': {
            'embedding_dim': 384,
            'max_tokens': 256,
            'description': 'Lightweight 12-layer model, slightly better quality than L6',
            'pooling': 'mean',
            'architecture': 'MiniLM',
            'parameters': 33_400_000,
            'performance': 'balanced',
            'memory_mb': 130,
            'languages': ['en'],
        },
        
        # High-quality models
        'all-mpnet-base-v2': {
            'embedding_dim': 768,
            'max_tokens': 384,
            'description': 'High-quality model based on MPNet, best general-purpose model',
            'pooling': 'mean',
            'architecture': 'MPNet',
            'parameters': 109_000_000,
            'performance': 'accurate',
            'memory_mb': 420,
            'languages': ['en'],
        },
        'multi-qa-mpnet-base-dot-v1': {
            'embedding_dim': 768,
            'max_tokens': 512,
            'description': 'Optimized for question-answering and retrieval tasks',
            'pooling': 'dot',
            'architecture': 'MPNet',
            'parameters': 109_000_000,
            'performance': 'accurate',
            'memory_mb': 420,
            'languages': ['en'],
        },
        
        # Multilingual models
        'paraphrase-multilingual-MiniLM-L12-v2': {
            'embedding_dim': 384,
            'max_tokens': 128,
            'description': 'Multilingual model supporting 50+ languages',
            'pooling': 'mean',
            'architecture': 'MiniLM',
            'parameters': 118_000_000,
            'performance': 'fast',
            'memory_mb': 450,
            'languages': ['en', 'es', 'de', 'fr', 'it', 'nl', 'pl', 'pt', 'ru', 'zh', 'ja', 'ko', 'ar'],
        },
        'paraphrase-multilingual-mpnet-base-v2': {
            'embedding_dim': 768,
            'max_tokens': 128,
            'description': 'High-quality multilingual model based on MPNet',
            'pooling': 'mean',
            'architecture': 'MPNet',
            'parameters': 278_000_000,
            'performance': 'accurate',
            'memory_mb': 1100,
            'languages': ['en', 'es', 'de', 'fr', 'it', 'nl', 'pl', 'pt', 'ru', 'zh', 'ja', 'ko', 'ar'],
        },
        
        # Domain-specific models
        'all-distilroberta-v1': {
            'embedding_dim': 768,
            'max_tokens': 512,
            'description': 'Distilled version of RoBERTa, good for longer texts',
            'pooling': 'mean',
            'architecture': 'DistilRoBERTa',
            'parameters': 82_000_000,
            'performance': 'balanced',
            'memory_mb': 330,
            'languages': ['en'],
        },
        'msmarco-distilbert-base-v3': {
            'embedding_dim': 768,
            'max_tokens': 512,
            'description': 'Optimized for MS MARCO passage ranking',
            'pooling': 'cls',
            'architecture': 'DistilBERT',
            'parameters': 66_000_000,
            'performance': 'fast',
            'memory_mb': 270,
            'languages': ['en'],
        },
        
        # Code models
        'all-codegen-350m': {
            'embedding_dim': 1024,
            'max_tokens': 512,
            'description': 'Code embedding model for programming languages',
            'pooling': 'mean',
            'architecture': 'CodeGen',
            'parameters': 350_000_000,
            'performance': 'accurate',
            'memory_mb': 1400,
            'languages': ['code'],
        },
        
        # Biomedical models
        'biobert-base-cased-v1.1': {
            'embedding_dim': 768,
            'max_tokens': 512,
            'description': 'Biomedical text embeddings',
            'pooling': 'mean',
            'architecture': 'BioBERT',
            'parameters': 110_000_000,
            'performance': 'accurate',
            'memory_mb': 440,
            'languages': ['en'],
        },
    }
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        config: Optional[EmbeddingConfig] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize Sentence Transformers embedding model.
        
        Args:
            model_name: Name of the Sentence Transformers model
            config: Configuration for embedding generation
            model_kwargs: Additional arguments for SentenceTransformer initialization
            tokenizer_kwargs: Additional arguments for tokenizer initialization
            **kwargs: Additional arguments passed to configuration
        """
        # Set default config for Sentence Transformers models
        if config is None:
            config = EmbeddingConfig(
                model_name=model_name,
                model_type=EmbeddingType.TEXT,
                max_length=256,  # Conservative default
                normalize=True,
                normalization_method=EmbeddingNormalization.L2,
                pooling_method=EmbeddingPooling.MEAN,
                batch_size=32,
                cache_embeddings=True,
                cache_size=10000,
                use_gpu=True,
            )
        
        # Update with model-specific defaults from MODEL_INFO
        if model_name in self.MODEL_INFO:
            model_info = self.MODEL_INFO[model_name]
            config.max_length = model_info.get('max_tokens', config.max_length)
            
            # Set pooling method from model info
            if 'pooling' in model_info:
                try:
                    config.pooling_method = EmbeddingPooling(model_info['pooling'])
                except ValueError:
                    logger.warning(f"Unknown pooling method in model info: {model_info['pooling']}")
        
        super().__init__(model_name, config, **kwargs)
        
        # Additional initialization parameters
        self.model_kwargs = model_kwargs or {}
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        
        # Device management
        self._device_str = None
        
        # Model components (for custom implementation)
        self._transformer_model = None
        self._custom_tokenizer = None
        
        # Performance tracking
        self._inference_times = []
        self._batch_sizes = []
        
        # Update metadata
        self._update_metadata()
        
        logger.info(f"Initialized Sentence Transformer embedding model: {model_name}")
    
    def _update_metadata(self) -> None:
        """Update metadata based on model name."""
        model_info = self.MODEL_INFO.get(
            self.model_name,
            {
                'embedding_dim': 384,
                'max_tokens': self.config.max_length,
                'description': 'Sentence Transformer model',
                'pooling': 'mean',
                'architecture': 'Transformer',
                'parameters': 0,
                'performance': 'unknown',
                'memory_mb': 0,
                'languages': ['en'],
            }
        )
        
        # Check if it's a multilingual model
        is_multilingual = 'multilingual' in self.model_name.lower() or len(model_info.get('languages', ['en'])) > 1
        
        self.metadata = EmbeddingMetadata(
            model_name=self.model_name,
            model_type=self.config.model_type,
            embedding_dim=model_info['embedding_dim'],
            max_tokens=model_info['max_tokens'],
            supported_languages=model_info.get('languages', ['en']),
            requires_tokenization=True,
            is_multilingual=is_multilingual,
            is_contextual=True,
            produces_normalized_embeddings=False,  # SentenceTransformers doesn't normalize by default
            license=None,  # Would need to be populated from Hugging Face
            citation=None,
        )
    
    def load(self) -> None:
        """
        Load the Sentence Transformers model.
        
        Raises:
            RuntimeError: If model loading fails
        """
        if self._is_loaded:
            logger.warning(f"Model {self.model_name} is already loaded")
            return
        
        try:
            logger.info(f"Loading Sentence Transformer model: {self.model_name}")
            
            # Determine device
            if self.config.use_gpu and torch.cuda.is_available():
                self._device_str = 'cuda'
                device_count = torch.cuda.device_count()
                if device_count > 1:
                    logger.info(f"Found {device_count} GPUs, using first GPU")
            else:
                self._device_str = 'cpu'
            
            # Configure device for SentenceTransformer
            device_config = {}
            if self._device_str == 'cuda':
                device_config['device'] = self._device_str
            
            # Merge with user-provided kwargs
            all_model_kwargs = {**device_config, **self.model_kwargs}
            
            # Load SentenceTransformer model
            self.model = SentenceTransformer(
                self.model_name,
                **all_model_kwargs
            )
            
            # Update max_length from model if available
            if hasattr(self.model, 'max_seq_length'):
                model_max_length = self.model.max_seq_length
                if self.config.max_length > model_max_length:
                    logger.warning(
                        f"Config max_length ({self.config.max_length}) exceeds "
                        f"model max_seq_length ({model_max_length}). "
                        f"Using model max_seq_length."
                    )
                    self.config.max_length = model_max_length
            
            self._is_loaded = True
            
            logger.info(f"Successfully loaded model {self.model_name} on device {self._device_str}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            
            # Try fallback to transformers if SentenceTransformer fails
            try:
                logger.info(f"Trying fallback to transformers for {self.model_name}")
                self._load_with_transformers()
                self._is_loaded = True
                logger.info(f"Successfully loaded model {self.model_name} using transformers")
            except Exception as e2:
                logger.error(f"Fallback also failed: {e2}")
                raise RuntimeError(f"Failed to load model {self.model_name}: {e}")
    
    def _load_with_transformers(self) -> None:
        """
        Load model using transformers library as fallback.
        """
        # Load tokenizer
        self._custom_tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            **self.tokenizer_kwargs
        )
        
        # Load model
        self._transformer_model = AutoModel.from_pretrained(
            self.model_name,
            **self.model_kwargs
        )
        
        # Move to device
        if self._device_str == 'cuda':
            self._transformer_model = self._transformer_model.to(self._device_str)
        
        # Set model to evaluation mode
        self._transformer_model.eval()
        
        # Update max_length from tokenizer
        if hasattr(self._custom_tokenizer, 'model_max_length'):
            tokenizer_max_length = self._custom_tokenizer.model_max_length
            if tokenizer_max_length > 0 and self.config.max_length > tokenizer_max_length:
                logger.warning(
                    f"Config max_length ({self.config.max_length}) exceeds "
                    f"tokenizer max_length ({tokenizer_max_length}). "
                    f"Using tokenizer max_length."
                )
                self.config.max_length = tokenizer_max_length
    
    def embed(
        self,
        texts: Union[str, List[str]],
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        **kwargs
    ) -> EmbeddingResult:
        """
        Generate embeddings for input texts.
        
        Args:
            texts: Single text or list of texts to embed
            convert_to_numpy: Whether to convert embeddings to numpy array
            show_progress_bar: Whether to show progress bar for batch processing
            **kwargs: Additional arguments for embedding generation
            
        Returns:
            EmbeddingResult containing the embeddings and metadata
            
        Raises:
            ValueError: If model is not loaded
            RuntimeError: If embedding generation fails
        """
        if not self._is_loaded:
            raise ValueError(f"Model {self.model_name} is not loaded. Call load() first.")
        
        # Convert single text to list
        single_text = False
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        
        try:
            import time
            start_time = time.time()
            
            # Use SentenceTransformer if available
            if self.model is not None:
                embeddings = self.model.encode(
                    texts,
                    batch_size=self.config.batch_size,
                    show_progress_bar=show_progress_bar,
                    convert_to_numpy=convert_to_numpy,
                    normalize_embeddings=self.config.normalize,
                    device=self._device_str,
                )
                
                # Apply additional normalization if configured
                if self.config.normalize and not self.config.normalization_method == EmbeddingNormalization.NONE:
                    embeddings = self.normalize_embeddings(embeddings, self.config.normalization_method)
            
            # Use custom transformer model as fallback
            else:
                embeddings = self._embed_with_transformers(
                    texts,
                    convert_to_numpy=convert_to_numpy,
                    **kwargs
                )
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Track performance
            self._inference_times.append(processing_time_ms)
            self._batch_sizes.append(len(texts))
            
            # Prepare result
            result = EmbeddingResult(
                embeddings=embeddings,
                model_name=self.model_name,
                model_version=self.get_version(),
                input_texts=texts if not single_text else texts[0],
                embedding_dim=self.get_embedding_dim(),
                processing_time_ms=processing_time_ms,
                batch_size=len(texts),
                metadata={
                    'device': self._device_str,
                    'batch_size_used': len(texts),
                    'model_info': self.MODEL_INFO.get(self.model_name, {}),
                    'pooling_method': self.config.pooling_method.value,
                }
            )
            
            # Cache embeddings if enabled
            if self.config.cache_embeddings and len(self._embedding_cache) < self.config.cache_size:
                for i, text in enumerate(texts):
                    cache_key = self._get_cache_key(text)
                    if cache_key not in self._embedding_cache:
                        self._embedding_cache[cache_key] = embeddings[i]
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {e}")
    
    def _embed_with_transformers(
        self,
        texts: List[str],
        convert_to_numpy: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Generate embeddings using transformers library directly.
        
        Args:
            texts: List of texts to embed
            convert_to_numpy: Whether to convert to numpy array
            **kwargs: Additional arguments
            
        Returns:
            Embeddings array
        """
        if self._transformer_model is None or self._custom_tokenizer is None:
            raise RuntimeError("Transformer model or tokenizer not loaded")
        
        # Tokenize texts
        encoded_input = self._custom_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        if self._device_str == 'cuda':
            encoded_input = {k: v.to(self._device_str) for k, v in encoded_input.items()}
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self._transformer_model(**encoded_input)
        
        # Apply pooling
        embeddings = self._apply_pooling(
            model_output,
            encoded_input['attention_mask'],
            self.config.pooling_method
        )
        
        # Normalize if requested
        if self.config.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Convert to numpy if requested
        if convert_to_numpy:
            embeddings = embeddings.cpu().numpy()
        
        # Apply additional normalization if configured
        if self.config.normalize and not self.config.normalization_method == EmbeddingNormalization.NONE:
            if convert_to_numpy:
                embeddings = self.normalize_embeddings(embeddings, self.config.normalization_method)
            else:
                # Convert to numpy for normalization, then back to tensor
                embeddings_np = embeddings.cpu().numpy()
                embeddings_np = self.normalize_embeddings(embeddings_np, self.config.normalization_method)
                embeddings = torch.from_numpy(embeddings_np).to(embeddings.device)
        
        return embeddings
    
    def _apply_pooling(
        self,
        model_output,
        attention_mask,
        pooling_method: EmbeddingPooling
    ) -> torch.Tensor:
        """
        Apply pooling strategy to token embeddings.
        
        Args:
            model_output: Output from transformer model
            attention_mask: Attention mask
            pooling_method: Pooling method to use
            
        Returns:
            Pooled embeddings
        """
        # Get token embeddings
        token_embeddings = model_output[0]  # First element contains token embeddings
        
        # Expand attention mask for broadcasting
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        if pooling_method == EmbeddingPooling.MEAN:
            # Mean pooling with attention mask
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        
        elif pooling_method == EmbeddingPooling.MAX:
            # Max pooling with attention mask
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set masked tokens to large negative value
            return torch.max(token_embeddings, 1)[0]
        
        elif pooling_method == EmbeddingPooling.CLS:
            # Use [CLS] token embedding
            return token_embeddings[:, 0]
        
        elif pooling_method == EmbeddingPooling.WEIGHTED_MEAN:
            # Weighted mean pooling (simple version - weights by position)
            batch_size, seq_len, hidden_dim = token_embeddings.shape
            weights = torch.arange(1, seq_len + 1, device=token_embeddings.device).float()
            weights = weights.view(1, seq_len, 1).expand(batch_size, seq_len, hidden_dim)
            
            weighted_sum = torch.sum(token_embeddings * weights * input_mask_expanded, 1)
            weight_sum = torch.sum(weights * input_mask_expanded, 1)
            return weighted_sum / weight_sum
        
        else:
            # Default to mean pooling
            logger.warning(f"Unknown pooling method: {pooling_method}, using MEAN")
            return self._apply_pooling(model_output, attention_mask, EmbeddingPooling.MEAN)
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimensionality of embeddings.
        
        Returns:
            Embedding dimension
        """
        if not self._is_loaded:
            # Return from metadata if model not loaded
            return self.metadata.embedding_dim
        
        # Get from model if available
        if self.model is not None:
            if hasattr(self.model, 'get_sentence_embedding_dimension'):
                return self.model.get_sentence_embedding_dimension()
            elif hasattr(self.model, 'word_embedding_dimension'):
                return self.model.word_embedding_dimension
        
        # Get from transformer model
        if self._transformer_model is not None:
            if hasattr(self._transformer_model, 'config'):
                return self._transformer_model.config.hidden_size
        
        return self.metadata.embedding_dim
    
    def tokenize(
        self,
        texts: Union[str, List[str]],
        return_tensors: str = 'pt',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Tokenize input texts.
        
        Args:
            texts: Single text or list of texts to tokenize
            return_tensors: Format to return tensors in ('pt', 'np', None)
            **kwargs: Additional arguments for tokenization
            
        Returns:
            Dictionary containing tokenized inputs
        """
        if not self._is_loaded:
            raise ValueError(f"Model {self.model_name} is not loaded. Call load() first.")
        
        single_text = False
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        
        try:
            # Use SentenceTransformer tokenizer if available
            if self.model is not None and hasattr(self.model, 'tokenize'):
                tokenized = self.model.tokenize(texts)
            else:
                # Use custom tokenizer
                if self._custom_tokenizer is None:
                    raise RuntimeError("Tokenizer not available")
                
                tokenized = self._custom_tokenizer(
                    texts,
                    padding=True if len(texts) > 1 else False,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors=return_tensors,
                    **kwargs
                )
            
            # Convert to dictionary format
            result = {
                'input_ids': tokenized['input_ids'].tolist() if hasattr(tokenized['input_ids'], 'tolist') else tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'].tolist() if hasattr(tokenized['attention_mask'], 'tolist') else tokenized['attention_mask'],
                'texts': texts if not single_text else texts[0],
                'model_name': self.model_name,
            }
            
            # Add token count
            if 'input_ids' in tokenized:
                if single_text:
                    result['token_count'] = len(tokenized['input_ids'][0])
                else:
                    result['token_counts'] = [len(ids) for ids in tokenized['input_ids']]
            
            # Add special tokens info if available
            if self._custom_tokenizer is not None:
                result['special_tokens'] = {
                    'cls_token': self._custom_tokenizer.cls_token,
                    'sep_token': self._custom_tokenizer.sep_token,
                    'pad_token': self._custom_tokenizer.pad_token,
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to tokenize texts: {e}")
            raise RuntimeError(f"Failed to tokenize texts: {e}")
    
    def get_version(self) -> str:
        """
        Get model version.
        
        Returns:
            Model version string
        """
        if not self._is_loaded:
            return "unknown"
        
        try:
            # Try to get version from model
            if self.model is not None and hasattr(self.model, '__version__'):
                return str(self.model.__version__)
            
            # Check Hugging Face model card
            if hasattr(self.model, '_model_card'):
                # Extract version from model card or config
                if hasattr(self.model, 'model_card_data'):
                    return self.model.model_card_data.get('version', '1.0.0')
            
            # Check transformer model config
            if self._transformer_model is not None and hasattr(self._transformer_model, 'config'):
                config = self._transformer_model.config
                if hasattr(config, '_commit_hash'):
                    return config._commit_hash[:8]
            
            return "1.0.0"
            
        except:
            return "unknown"
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        info = self.MODEL_INFO.get(self.model_name, {}).copy()
        
        # Add runtime information
        info.update({
            'loaded': self._is_loaded,
            'device': self._device_str,
            'embedding_dim': self.get_embedding_dim(),
            'max_length': self.config.max_length,
            'pooling_method': self.config.pooling_method.value,
            'normalization': self.config.normalization_method.value if self.config.normalize else 'none',
        })
        
        # Add performance statistics if available
        if self._inference_times:
            info['performance_stats'] = {
                'total_batches': len(self._inference_times),
                'total_texts': sum(self._batch_sizes),
                'avg_inference_time_ms': sum(self._inference_times) / len(self._inference_times),
                'avg_batch_size': sum(self._batch_sizes) / len(self._batch_sizes),
                'throughput_texts_per_second': sum(self._batch_sizes) / (sum(self._inference_times) / 1000),
            }
        
        return info
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        if not self._inference_times:
            return {'message': 'No inference performed yet'}
        
        stats = {
            'total_batches': len(self._inference_times),
            'total_texts': sum(self._batch_sizes),
            'inference_times_ms': self._inference_times.copy(),
            'batch_sizes': self._batch_sizes.copy(),
            'avg_inference_time_ms': sum(self._inference_times) / len(self._inference_times),
            'avg_batch_size': sum(self._batch_sizes) / len(self._batch_sizes),
            'throughput_texts_per_second': sum(self._batch_sizes) / (sum(self._inference_times) / 1000),
            'min_inference_time_ms': min(self._inference_times),
            'max_inference_time_ms': max(self._inference_times),
        }
        
        # Calculate percentiles
        if len(self._inference_times) >= 10:
            import numpy as np
            stats['p50_inference_time_ms'] = np.percentile(self._inference_times, 50)
            stats['p90_inference_time_ms'] = np.percentile(self._inference_times, 90)
            stats['p95_inference_time_ms'] = np.percentile(self._inference_times, 95)
            stats['p99_inference_time_ms'] = np.percentile(self._inference_times, 99)
        
        return stats
    
    def clear_performance_stats(self) -> None:
        """Clear performance statistics."""
        self._inference_times.clear()
        self._batch_sizes.clear()
        logger.info("Cleared performance statistics")
    
    def optimize_for_inference(
        self,
        use_half_precision: bool = True,
        use_bettertransformer: bool = True,
        compile_model: bool = False
    ) -> None:
        """
        Optimize model for inference performance.
        
        Args:
            use_half_precision: Whether to use half precision (FP16)
            use_bettertransformer: Whether to use BetterTransformer optimization
            compile_model: Whether to compile model with torch.compile (PyTorch 2.0+)
        """
        if not self._is_loaded:
            raise ValueError(f"Model {self.model_name} is not loaded. Call load() first.")
        
        logger.info(f"Optimizing model {self.model_name} for inference")
        
        try:
            # Apply half precision
            if use_half_precision and self._device_str == 'cuda':
                if self.model is not None:
                    self.model = self.model.half()
                elif self._transformer_model is not None:
                    self._transformer_model = self._transformer_model.half()
                logger.info("Applied half precision (FP16)")
            
            # Apply BetterTransformer optimization
            if use_bettertransformer:
                try:
                    from optimum.bettertransformer import BetterTransformer # type: ignore
                    
                    if self.model is not None and hasattr(self.model, '_first_module'):
                        # Apply to the underlying transformer model
                        transformer = self.model._first_module().auto_model
                        transformer = BetterTransformer.transform(transformer)
                        self.model._first_module().auto_model = transformer
                        logger.info("Applied BetterTransformer optimization")
                    elif self._transformer_model is not None:
                        self._transformer_model = BetterTransformer.transform(self._transformer_model)
                        logger.info("Applied BetterTransformer optimization")
                except ImportError:
                    logger.warning(
                        "optimum not installed. BetterTransformer optimization skipped. "
                        "Install with: pip install optimum"
                    )
                except Exception as e:
                    logger.warning(f"Failed to apply BetterTransformer: {e}")
            
            # Compile model with torch.compile
            if compile_model:
                try:
                    import torch
                    if hasattr(torch, 'compile'):
                        if self._transformer_model is not None:
                            self._transformer_model = torch.compile(self._transformer_model)
                            logger.info("Applied torch.compile optimization")
                    else:
                        logger.warning("torch.compile not available (requires PyTorch 2.0+)")
                except Exception as e:
                    logger.warning(f"Failed to compile model: {e}")
        
        except Exception as e:
            logger.error(f"Failed to optimize model: {e}")
            raise RuntimeError(f"Failed to optimize model: {e}")
    
    def save_model(
        self,
        output_path: str,
        save_format: str = 'sentence_transformers'
    ) -> None:
        """
        Save the model to disk.
        
        Args:
            output_path: Path to save the model
            save_format: Format to save in ('sentence_transformers', 'transformers')
            
        Raises:
            ValueError: If model is not loaded or format is invalid
        """
        if not self._is_loaded:
            raise ValueError(f"Model {self.model_name} is not loaded. Call load() first.")
        
        import os
        from pathlib import Path
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            if save_format == 'sentence_transformers' and self.model is not None:
                self.model.save(str(output_path))
                logger.info(f"Saved model to {output_path} in Sentence Transformers format")
            
            elif save_format == 'transformers':
                if self._transformer_model is not None and self._custom_tokenizer is not None:
                    self._transformer_model.save_pretrained(output_path)
                    self._custom_tokenizer.save_pretrained(output_path)
                    logger.info(f"Saved model to {output_path} in Transformers format")
                else:
                    raise ValueError("Transformers model components not available")
            
            else:
                raise ValueError(f"Unsupported save format: {save_format}")
        
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise RuntimeError(f"Failed to save model: {e}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        model_name: Optional[str] = None,
        **kwargs
    ) -> 'SentenceTransformerEmbedding':
        """
        Load a model from a local path.
        
        Args:
            model_path: Path to the saved model
            model_name: Optional name for the model (defaults to path name)
            **kwargs: Additional arguments for initialization
            
        Returns:
            SentenceTransformerEmbedding instance
        """
        from pathlib import Path
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        
        # Use path name as model name if not provided
        if model_name is None:
            model_name = model_path.name
        
        # Check if it's a Sentence Transformers model
        if (model_path / 'config.json').exists() and (model_path / 'pytorch_model.bin').exists():
            # It's a transformers model
            instance = cls(
                model_name=model_name,
                **kwargs
            )
            
            # Load using transformers
            instance._custom_tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            instance._transformer_model = AutoModel.from_pretrained(str(model_path))
            instance._is_loaded = True
            
            logger.info(f"Loaded model from {model_path} using transformers")
            
        else:
            # Try as Sentence Transformers model
            try:
                instance = cls(
                    model_name=model_name,
                    **kwargs
                )
                
                # Load SentenceTransformer model
                instance.model = SentenceTransformer(str(model_path))
                instance._is_loaded = True
                
                logger.info(f"Loaded model from {model_path} using Sentence Transformers")
                
            except Exception as e:
                raise ValueError(f"Could not load model from {model_path}: {e}")
        
        return instance
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """
        List all available Sentence Transformers models.
        
        Returns:
            Dictionary of model names to model information
        """
        return cls.MODEL_INFO.copy()
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific Sentence Transformers model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information dictionary or None if not found
        """
        # Try exact match
        if model_name in cls.MODEL_INFO:
            return cls.MODEL_INFO[model_name].copy()
        
        # Try without sentence-transformers/ prefix
        short_name = model_name.replace('sentence-transformers/', '')
        if short_name in cls.MODEL_INFO:
            return cls.MODEL_INFO[short_name].copy()
        
        return None
    
    def unload(self) -> None:
        """Unload model from memory."""
        if self._is_loaded:
            del self.model
            del self._transformer_model
            del self._custom_tokenizer
            self.model = None
            self._transformer_model = None
            self._custom_tokenizer = None
            self._is_loaded = False
            
            # Clear cache
            self.clear_cache()
            
            # Clear performance stats
            self.clear_performance_stats()
            
            # Free GPU memory if applicable
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info(f"Unloaded model: {self.model_name}")
    
    def __repr__(self) -> str:
        """String representation."""
        loaded = self._is_loaded
        dim = self.get_embedding_dim() if loaded else self.metadata.embedding_dim
        device = self._device_str if loaded else 'unknown'
        return (
            f"SentenceTransformerEmbedding(model_name={self.model_name}, "
            f"loaded={loaded}, dim={dim}, device={device})"
        )


# Register with factory
try:
    from .base import EmbeddingModelFactory
    EmbeddingModelFactory.register_model('sentence_transformers', SentenceTransformerEmbedding)
    logger.info("Registered SentenceTransformerEmbedding with EmbeddingModelFactory")
except ImportError:
    logger.warning("Could not register SentenceTransformerEmbedding with factory")


__all__ = [
    'SentenceTransformerEmbedding',
    'MODEL_INFO',
]