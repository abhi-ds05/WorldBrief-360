"""
Multilingual embedding models.
Supports embedding generation for multiple languages using models like:
- paraphrase-multilingual-MiniLM-L12-v2
- sentence-transformers/paraphrase-multilingual-mpnet-base-v2
- intfloat/multilingual-e5-large
"""
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import (
    BaseEmbeddingModel, EmbeddingConfig, EmbeddingResult,
    EmbeddingMetadata, EmbeddingType, EmbeddingNormalization,
    EmbeddingPooling
)
from ...base import ModelDevice

logger = logging.getLogger(__name__)


class MultilingualEmbedding(BaseEmbeddingModel):
    """
    Multilingual embedding model using Sentence Transformers.
    
    Supports embedding generation for multiple languages with a single model.
    """
    
    # Language code mapping for common languages
    LANGUAGE_MAP = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'nl': 'Dutch',
        'pl': 'Polish',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'bn': 'Bengali',
        'tr': 'Turkish',
        'vi': 'Vietnamese',
        'th': 'Thai',
        'fa': 'Persian',
        'he': 'Hebrew',
        'uk': 'Ukrainian',
        'cs': 'Czech',
        'sv': 'Swedish',
        'da': 'Danish',
        'fi': 'Finnish',
        'no': 'Norwegian',
        'hu': 'Hungarian',
        'el': 'Greek',
        'ro': 'Romanian',
        'sk': 'Slovak',
        'bg': 'Bulgarian',
        'hr': 'Croatian',
        'sr': 'Serbian',
        'sl': 'Slovenian',
        'et': 'Estonian',
        'lv': 'Latvian',
        'lt': 'Lithuanian',
        'mt': 'Maltese',
        'ga': 'Irish',
        'cy': 'Welsh',
        'eu': 'Basque',
        'ca': 'Catalan',
        'gl': 'Galician',
    }
    
    # Model information for popular multilingual models
    MODEL_INFO = {
        'paraphrase-multilingual-MiniLM-L12-v2': {
            'embedding_dim': 384,
            'max_tokens': 128,
            'supported_languages': ['en', 'es', 'de', 'fr', 'it', 'nl', 'pl', 'pt', 'ru', 'zh'],
            'description': 'Lightweight multilingual model optimized for paraphrase detection',
            'performance': 'fast',
            'memory_mb': 200,
        },
        'sentence-transformers/paraphrase-multilingual-mpnet-base-v2': {
            'embedding_dim': 768,
            'max_tokens': 128,
            'supported_languages': ['en', 'es', 'de', 'fr', 'it', 'nl', 'pl', 'pt', 'ru', 'zh'],
            'description': 'High-quality multilingual model based on MPNet',
            'performance': 'accurate',
            'memory_mb': 500,
        },
        'intfloat/multilingual-e5-large': {
            'embedding_dim': 1024,
            'max_tokens': 512,
            'supported_languages': ['en', 'zh', 'es', 'fr', 'de', 'it', 'ja', 'ko', 'nl', 'pl', 'pt', 'ru'],
            'description': 'State-of-the-art multilingual embeddings (E5)',
            'performance': 'excellent',
            'memory_mb': 1200,
        },
        'sentence-transformers/distiluse-base-multilingual-cased-v2': {
            'embedding_dim': 512,
            'max_tokens': 128,
            'supported_languages': ['en', 'es', 'de', 'fr', 'it', 'nl', 'pl', 'pt', 'ru', 'zh'],
            'description': 'Distilled multilingual Universal Sentence Encoder',
            'performance': 'balanced',
            'memory_mb': 300,
        },
        'sentence-transformers/paraphrase-xlm-r-multilingual-v1': {
            'embedding_dim': 768,
            'max_tokens': 128,
            'supported_languages': ['en', 'es', 'de', 'fr', 'it', 'nl', 'pl', 'pt', 'ru', 'zh', 'ja', 'ko', 'ar'],
            'description': 'Multilingual model based on XLM-RoBERTa',
            'performance': 'accurate',
            'memory_mb': 600,
        },
    }
    
    def __init__(
        self,
        model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
        config: Optional[EmbeddingConfig] = None,
        **kwargs
    ):
        """
        Initialize multilingual embedding model.
        
        Args:
            model_name: Name of the multilingual model
            config: Configuration for embedding generation
            **kwargs: Additional arguments passed to configuration
        """
        # Set default config for multilingual models
        if config is None:
            config = EmbeddingConfig(
                model_name=model_name,
                model_type=EmbeddingType.TEXT,
                max_length=128,  # Conservative default for multilingual models
                normalize=True,
                normalization_method=EmbeddingNormalization.L2,
                pooling_method=EmbeddingPooling.MEAN,
                cache_embeddings=True,
                cache_size=10000,
            )
        
        # Update with model-specific defaults from MODEL_INFO
        if model_name in self.MODEL_INFO:
            model_info = self.MODEL_INFO[model_name]
            config.max_length = model_info.get('max_tokens', config.max_length)
        
        super().__init__(model_name, config, **kwargs)
        
        # Language detection model (loaded on demand)
        self._lang_detector = None
        
        # Language-specific settings
        self._detected_languages: Dict[str, str] = {}  # text -> language code
        self._language_stats: Dict[str, int] = {}  # language code -> count
        
        # Update metadata
        self._update_metadata()
        
        logger.info(f"Initialized multilingual embedding model: {model_name}")
    
    def _update_metadata(self) -> None:
        """Update metadata based on model name."""
        model_info = self.MODEL_INFO.get(
            self.model_name,
            self.MODEL_INFO.get(
                self.model_name.replace('sentence-transformers/', ''),
                {
                    'embedding_dim': 384,
                    'max_tokens': self.config.max_length,
                    'supported_languages': ['en'],
                    'description': 'Multilingual embedding model',
                    'performance': 'unknown',
                    'memory_mb': 0,
                }
            )
        )
        
        self.metadata = EmbeddingMetadata(
            model_name=self.model_name,
            model_type=self.config.model_type,
            embedding_dim=model_info['embedding_dim'],
            max_tokens=model_info['max_tokens'],
            supported_languages=model_info['supported_languages'],
            requires_tokenization=True,
            is_multilingual=True,
            is_contextual=True,
            produces_normalized_embeddings=False,
            license=None,  # Would need to be populated from Hugging Face
            citation=None,
        )
    
    def load(self) -> None:
        """
        Load the multilingual embedding model.
        
        Raises:
            RuntimeError: If model loading fails
        """
        if self._is_loaded:
            logger.warning(f"Model {self.model_name} is already loaded")
            return
        
        try:
            logger.info(f"Loading multilingual embedding model: {self.model_name}")
            
            # Configure device
            device = self.config.device.value
            if device == ModelDevice.AUTO.value:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Load SentenceTransformer model
            self.model = SentenceTransformer(
                self.model_name,
                device=device,
                cache_folder=None,  # Use default cache
            )
            
            # Update max_length from model if not explicitly set
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
            self._device = device
            
            logger.info(f"Successfully loaded model {self.model_name} on device {device}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")
    
    def embed(
        self,
        texts: Union[str, List[str]],
        languages: Optional[Union[str, List[str]]] = None,
        detect_language: bool = False,
        **kwargs
    ) -> EmbeddingResult:
        """
        Generate embeddings for input texts in multiple languages.
        
        Args:
            texts: Single text or list of texts to embed
            languages: Optional language codes for each text
            detect_language: Whether to automatically detect language
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
        
        if languages is not None:
            if isinstance(languages, str):
                languages = [languages]
            
            if len(languages) != len(texts):
                raise ValueError(
                    f"Number of languages ({len(languages)}) must match "
                    f"number of texts ({len(texts)})"
                )
        
        # Detect languages if requested
        detected_languages = None
        if detect_language:
            detected_languages = self.detect_languages(texts)
        
        # Prepare texts and track language information
        processed_texts = []
        text_languages = []
        language_info = []
        
        for i, text in enumerate(texts):
            processed_texts.append(text)
            
            # Determine language for this text
            lang = None
            if languages is not None:
                lang = languages[i]
            elif detect_language and detected_languages is not None:
                lang = detected_languages[i]
            
            text_languages.append(lang)
            
            # Store language info for metadata
            if lang:
                lang_name = self.LANGUAGE_MAP.get(lang, lang)
                language_info.append({
                    'text_index': i,
                    'language_code': lang,
                    'language_name': lang_name,
                    'detected': detect_language and detected_languages is not None
                })
                
                # Update language statistics
                self._language_stats[lang] = self._language_stats.get(lang, 0) + 1
        
        try:
            import time
            start_time = time.time()
            
            # Generate embeddings using SentenceTransformers
            embeddings = self.model.encode(
                processed_texts,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.config.normalize,
                convert_to_numpy=True,
                device=self._device,
            )
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Apply additional normalization if configured
            if self.config.normalize and not self.config.normalization_method == EmbeddingNormalization.NONE:
                embeddings = self.normalize_embeddings(embeddings, self.config.normalization_method)
            
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
                    'languages': text_languages if not single_text else text_languages[0],
                    'language_info': language_info,
                    'language_detection_used': detect_language,
                    'model_info': self.MODEL_INFO.get(self.model_name, {}),
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
    
    def detect_languages(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32
    ) -> Union[str, List[str]]:
        """
        Detect languages of input texts.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for language detection
            
        Returns:
            Language codes for each text
            
        Raises:
            ImportError: If language detection library is not available
        """
        single_text = False
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        
        # Lazy load language detector
        if self._lang_detector is None:
            try:
                from langdetect import detect, DetectorFactory, lang_detect_exception
                from langdetect import detect_langs
                
                # Set seed for reproducibility
                DetectorFactory.seed = 42
                
                self._lang_detector = {
                    'detect': detect,
                    'detect_langs': detect_langs,
                    'lang_detect_exception': lang_detect_exception
                }
                
                logger.info("Loaded langdetect for language detection")
                
            except ImportError:
                logger.warning(
                    "langdetect not installed. Language detection disabled. "
                    "Install with: pip install langdetect"
                )
                self._lang_detector = False
        
        if self._lang_detector is False:
            # Return 'unknown' for all texts if detection is disabled
            results = ['unknown'] * len(texts)
            return results[0] if single_text else results
        
        # Detect languages
        results = []
        detect_func = self._lang_detector['detect']
        detect_langs_func = self._lang_detector['detect_langs']
        lang_exception = self._lang_detector['lang_detect_exception']
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            for text in batch_texts:
                try:
                    # Try to detect language
                    lang = detect_func(text)
                    results.append(lang)
                    
                    # Store detection
                    self._detected_languages[text] = lang
                    
                except lang_exception.LangDetectException:
                    # Fallback: try to get probabilities
                    try:
                        langs = detect_langs_func(text)
                        if langs:
                            lang = langs[0].lang  # Most probable language
                            results.append(lang)
                            self._detected_languages[text] = lang
                        else:
                            results.append('unknown')
                            self._detected_languages[text] = 'unknown'
                    except:
                        results.append('unknown')
                        self._detected_languages[text] = 'unknown'
                except Exception:
                    results.append('unknown')
                    self._detected_languages[text] = 'unknown'
        
        return results[0] if single_text else results
    
    def get_language_probabilities(
        self,
        text: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get language probabilities for a text.
        
        Args:
            text: Input text
            top_k: Number of top languages to return
            
        Returns:
            List of dictionaries with language codes and probabilities
        """
        if self._lang_detector is None or self._lang_detector is False:
            raise ImportError("Language detection not available. Install langdetect.")
        
        try:
            from langdetect import detect_langs
            
            lang_probabilities = detect_langs(text)
            
            # Convert to list of dictionaries
            results = []
            for lp in lang_probabilities[:top_k]:
                lang_code = lp.lang
                lang_name = self.LANGUAGE_MAP.get(lang_code, lang_code)
                results.append({
                    'language_code': lang_code,
                    'language_name': lang_name,
                    'probability': lp.prob,
                    'is_supported': lang_code in self.metadata.supported_languages
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get language probabilities: {e}")
            return []
    
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
        if hasattr(self.model, 'get_sentence_embedding_dimension'):
            return self.model.get_sentence_embedding_dimension()
        elif hasattr(self.model, 'word_embedding_dimension'):
            return self.model.word_embedding_dimension
        else:
            return self.metadata.embedding_dim
    
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
        if not self._is_loaded:
            raise ValueError(f"Model {self.model_name} is not loaded. Call load() first.")
        
        single_text = False
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        
        try:
            # Use the model's tokenizer
            if hasattr(self.model, 'tokenize'):
                tokenized = self.model.tokenize(texts)
            else:
                # Fallback to manual tokenization
                tokenized = self.model._first_module().tokenize(texts)
            
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
            if hasattr(self.model, '__version__'):
                return str(self.model.__version__)
            
            # Check if it's a SentenceTransformer model
            if hasattr(self.model, '_model_card'):
                # Extract version from model card or config
                if hasattr(self.model, 'model_card_data'):
                    return self.model.model_card_data.get('version', '1.0.0')
            
            return "1.0.0"
            
        except:
            return "unknown"
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages.
        
        Returns:
            List of language codes
        """
        return self.metadata.supported_languages.copy()
    
    def is_language_supported(self, language_code: str) -> bool:
        """
        Check if a language is supported by the model.
        
        Args:
            language_code: ISO 639-1 language code
            
        Returns:
            True if language is supported
        """
        return language_code in self.metadata.supported_languages
    
    def get_language_name(self, language_code: str) -> str:
        """
        Get human-readable language name.
        
        Args:
            language_code: ISO 639-1 language code
            
        Returns:
            Language name or code if not found
        """
        return self.LANGUAGE_MAP.get(language_code, language_code)
    
    def get_language_stats(self) -> Dict[str, Any]:
        """
        Get statistics about processed languages.
        
        Returns:
            Dictionary with language statistics
        """
        total_texts = sum(self._language_stats.values())
        
        stats = {
            'total_texts_processed': total_texts,
            'unique_languages': len(self._language_stats),
            'language_distribution': self._language_stats.copy(),
            'detected_languages_count': len(self._detected_languages),
        }
        
        # Add percentages
        if total_texts > 0:
            stats['language_percentages'] = {
                lang: (count / total_texts) * 100
                for lang, count in self._language_stats.items()
            }
        
        return stats
    
    def clear_language_stats(self) -> None:
        """Clear language statistics."""
        self._language_stats.clear()
        self._detected_languages.clear()
        logger.info("Cleared language statistics")
    
    def embed_by_language(
        self,
        texts_by_language: Dict[str, List[str]],
        **kwargs
    ) -> Dict[str, EmbeddingResult]:
        """
        Generate embeddings organized by language.
        
        Args:
            texts_by_language: Dictionary mapping language codes to lists of texts
            **kwargs: Additional arguments for embedding
            
        Returns:
            Dictionary mapping language codes to EmbeddingResults
            
        Raises:
            ValueError: If any language is not supported
        """
        # Check if all languages are supported
        unsupported_languages = []
        for lang in texts_by_language.keys():
            if not self.is_language_supported(lang):
                unsupported_languages.append(lang)
        
        if unsupported_languages:
            raise ValueError(
                f"Unsupported languages: {unsupported_languages}. "
                f"Supported languages: {self.get_supported_languages()}"
            )
        
        # Generate embeddings for each language
        results = {}
        for lang, texts in texts_by_language.items():
            if texts:  # Only process if there are texts
                results[lang] = self.embed(
                    texts,
                    languages=[lang] * len(texts),
                    detect_language=False,
                    **kwargs
                )
        
        return results
    
    def get_cross_lingual_similarity(
        self,
        text1: str,
        text2: str,
        language1: Optional[str] = None,
        language2: Optional[str] = None,
        metric: str = "cosine"
    ) -> float:
        """
        Compute similarity between texts in different languages.
        
        Args:
            text1: First text
            text2: Second text
            language1: Language of first text (auto-detected if None)
            language2: Language of second text (auto-detected if None)
            metric: Similarity metric
            
        Returns:
            Similarity score
        """
        # Generate embeddings
        texts = [text1, text2]
        languages = None
        
        if language1 is not None or language2 is not None:
            languages = [language1, language2]
            detect_language = False
        else:
            detect_language = True
        
        result = self.embed(
            texts,
            languages=languages,
            detect_language=detect_language,
        )
        
        # Compute similarity
        similarities = self.compute_similarity(
            result.embeddings[0:1],  # First text
            result.embeddings[1:2],  # Second text
            metric=metric
        )
        
        return float(similarities[0, 0])
    
    def get_best_translation_match(
        self,
        query_text: str,
        candidate_texts: List[str],
        candidate_languages: Optional[List[str]] = None,
        query_language: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Find the best translation matches for a query across different languages.
        
        Args:
            query_text: Query text to match
            candidate_texts: Candidate texts in potentially different languages
            candidate_languages: Optional language codes for candidates
            query_language: Optional language of query (auto-detected if None)
            top_k: Number of top matches to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of dictionaries with match information
        """
        # Generate query embedding
        query_result = self.embed(
            [query_text],
            languages=[query_language] if query_language else None,
            detect_language=query_language is None,
        )
        query_embedding = query_result.embeddings[0]
        detected_query_lang = query_result.metadata['languages'][0]
        
        # Generate candidate embeddings
        candidate_result = self.embed(
            candidate_texts,
            languages=candidate_languages,
            detect_language=candidate_languages is None,
        )
        
        # Find similar candidates
        matches = self.find_similar(
            query_embedding,
            candidate_result.embeddings,
            candidate_texts=candidate_texts,
            top_k=top_k,
            metric="cosine",
            threshold=similarity_threshold
        )
        
        # Add language information to matches
        for match in matches:
            idx = match['index']
            match['query_language'] = detected_query_lang
            match['candidate_language'] = candidate_result.metadata['languages'][idx]
            
            # Add language names
            match['query_language_name'] = self.get_language_name(detected_query_lang)
            match['candidate_language_name'] = self.get_language_name(match['candidate_language'])
            
            # Check if languages are different
            match['cross_lingual'] = detected_query_lang != match['candidate_language']
        
        return matches
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """
        List all available multilingual models.
        
        Returns:
            Dictionary of model names to model information
        """
        return cls.MODEL_INFO.copy()
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific multilingual model.
        
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
    
    def __repr__(self) -> str:
        """String representation."""
        loaded = self._is_loaded
        dim = self.get_embedding_dim() if loaded else self.metadata.embedding_dim
        langs = len(self.metadata.supported_languages)
        return (
            f"MultilingualEmbedding(model_name={self.model_name}, "
            f"loaded={loaded}, dim={dim}, languages={langs})"
        )


# Register with factory
try:
    from .base import EmbeddingModelFactory
    EmbeddingModelFactory.register_model('multilingual', MultilingualEmbedding)
    logger.info("Registered MultilingualEmbedding with EmbeddingModelFactory")
except ImportError:
    logger.warning("Could not register MultilingualEmbedding with factory")


__all__ = [
    'MultilingualEmbedding',
    'LANGUAGE_MAP',
]