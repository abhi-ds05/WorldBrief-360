"""
Base classes for image-to-text models.
Defines the abstract interface that all vision-language models must implement.
"""
import asyncio
import base64
import io
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Callable,
    Iterator, AsyncIterator
)

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, Field, validator

from ...base import ModelType, ModelFramework, ModelDevice #type: ignore

logger = logging.getLogger(__name__)


class ImageTaskType(Enum):
    """Types of image-to-text tasks."""
    CAPTIONING = "captioning"  # Generate descriptive captions
    VQA = "visual_question_answering"  # Answer questions about images
    DOCUMENT_UNDERSTANDING = "document_understanding"  # Extract text and structure from documents
    OCR = "optical_character_recognition"  # Extract text from images
    VISUAL_REASONING = "visual_reasoning"  # Complex reasoning about images
    REFERRING_EXPRESSION = "referring_expression"  # Ground referring expressions
    VISUAL_INSTRUCTION = "visual_instruction"  # Follow instructions about images
    MULTIMODAL_CHAT = "multimodal_chat"  # Conversational interactions with images
    IMAGE_CLASSIFICATION = "image_classification"  # Classify images


class ImageFormat(Enum):
    """Supported image formats."""
    PIL = "pil"
    NUMPY = "numpy"
    TENSOR = "tensor"
    BYTES = "bytes"
    BASE64 = "base64"
    FILE_PATH = "file_path"
    URL = "url"


class TextGenerationConfig(BaseModel):
    """Configuration for text generation."""
    max_new_tokens: int = 512
    min_new_tokens: int = 1
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.0
    do_sample: bool = True
    num_beams: int = 1
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    early_stopping: bool = True
    seed: Optional[int] = None
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True


class ImageCaptionConfig(BaseModel):
    """Configuration for image captioning."""
    task_type: ImageTaskType = ImageTaskType.CAPTIONING
    prompt: Optional[str] = None
    context: Optional[str] = None
    question: Optional[str] = None  # For VQA tasks
    generation_config: TextGenerationConfig = Field(default_factory=TextGenerationConfig)
    image_size: Tuple[int, int] = (384, 384)
    normalize_image: bool = True
    return_attention: bool = False
    return_logits: bool = False
    return_scores: bool = False
    language: str = "en"
    custom_config: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True
    
    @validator('image_size')
    def validate_image_size(cls, v):
        """Validate image size."""
        if len(v) != 2:
            raise ValueError("image_size must be a tuple of (width, height)")
        if v[0] <= 0 or v[1] <= 0:
            raise ValueError("image_size dimensions must be positive")
        return v


@dataclass
class ImageTextResult:
    """Result of image-to-text processing."""
    text: str
    model_name: str
    model_version: str
    input_image_info: Dict[str, Any]
    task_type: ImageTaskType
    processing_time_ms: float = 0.0
    scores: Optional[Dict[str, float]] = None
    logits: Optional[np.ndarray] = None
    attention_weights: Optional[np.ndarray] = None
    generated_tokens: Optional[List[str]] = None
    token_probabilities: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "text": self.text,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "input_image_info": self.input_image_info,
            "task_type": self.task_type.value,
            "processing_time_ms": self.processing_time_ms,
            "scores": self.scores,
            "metadata": self.metadata,
        }
        
        # Convert numpy arrays to lists
        if self.logits is not None:
            result["logits"] = self.logits.tolist()
        if self.attention_weights is not None:
            result["attention_weights"] = self.attention_weights.tolist()
        
        # Add other optional fields
        if self.generated_tokens is not None:
            result["generated_tokens"] = self.generated_tokens
        if self.token_probabilities is not None:
            result["token_probabilities"] = self.token_probabilities
        
        return result
    
    @property
    def confidence(self) -> Optional[float]:
        """Get overall confidence score if available."""
        if self.scores and 'confidence' in self.scores:
            return self.scores['confidence']
        if self.token_probabilities:
            return np.mean(self.token_probabilities) if self.token_probabilities else None
        return None


class BaseImageToTextModel(ABC):
    """
    Abstract base class for all image-to-text models.
    
    This class defines the interface that all vision-language models must implement.
    It provides common functionality for image processing, text generation, and utilities.
    """
    
    def __init__(
        self,
        model_name: str,
        config: Optional[ImageCaptionConfig] = None,
        **kwargs
    ):
        """
        Initialize the image-to-text model.
        
        Args:
            model_name: Name of the model or path to local model
            config: Configuration for image captioning
            **kwargs: Additional arguments passed to configuration
        """
        self.model_name = model_name
        
        # Create config with defaults and overrides
        if config is None:
            config = ImageCaptionConfig()
        
        # Update config with kwargs
        config_dict = {**config.dict(), **kwargs}
        self.config = ImageCaptionConfig(**config_dict)
        
        # Initialize model components
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.feature_extractor = None
        
        # Image preprocessing
        self.image_size = self.config.image_size
        self.normalize_image = self.config.normalize_image
        
        # Device management
        self._device = None
        self._is_loaded = False
        
        # Cache for processed images
        self._image_cache: Dict[str, Any] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Performance tracking
        self._processing_times: List[float] = []
        self._total_images_processed = 0
        
        logger.info(f"Initialized image-to-text model: {model_name}")
    
    @abstractmethod
    def load(self) -> None:
        """
        Load the model and necessary components.
        
        This method should be implemented by subclasses to load the actual model,
        processor, tokenizer, and any other required components.
        """
        pass
    
    @abstractmethod
    def process_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process an image for model input.
        
        Args:
            image: Input image in various formats
            **kwargs: Additional processing arguments
            
        Returns:
            Dictionary containing processed image tensors and metadata
            
        Raises:
            ValueError: If image format is not supported
            RuntimeError: If image processing fails
        """
        pass
    
    @abstractmethod
    def generate_text(
        self,
        image_input: Dict[str, Any],
        config: Optional[ImageCaptionConfig] = None,
        **kwargs
    ) -> ImageTextResult:
        """
        Generate text from processed image input.
        
        Args:
            image_input: Processed image input from process_image()
            config: Configuration for text generation (uses instance config if None)
            **kwargs: Additional generation arguments
            
        Returns:
            ImageTextResult containing generated text and metadata
            
        Raises:
            ValueError: If model is not loaded
            RuntimeError: If text generation fails
        """
        pass
    
    def caption_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        prompt: Optional[str] = None,
        **kwargs
    ) -> ImageTextResult:
        """
        Generate caption for an image.
        
        Args:
            image: Input image in various formats
            prompt: Optional prompt/context for captioning
            **kwargs: Additional arguments for processing and generation
            
        Returns:
            ImageTextResult containing caption and metadata
        """
        # Update config with prompt if provided
        config = self.config
        if prompt is not None:
            config_dict = self.config.dict()
            config_dict['prompt'] = prompt
            config_dict.update(kwargs)
            config = ImageCaptionConfig(**config_dict)
        elif kwargs:
            config_dict = self.config.dict()
            config_dict.update(kwargs)
            config = ImageCaptionConfig(**config_dict)
        
        # Process image
        image_input = self.process_image(image, **kwargs)
        
        # Generate caption
        return self.generate_text(image_input, config)
    
    async def caption_image_async(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        prompt: Optional[str] = None,
        **kwargs
    ) -> ImageTextResult:
        """
        Asynchronously generate caption for an image.
        
        Args:
            image: Input image in various formats
            prompt: Optional prompt/context for captioning
            **kwargs: Additional arguments
            
        Returns:
            ImageTextResult containing caption and metadata
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.caption_image(image, prompt, **kwargs)
        )
    
    def answer_question(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        question: str,
        **kwargs
    ) -> ImageTextResult:
        """
        Answer a question about an image (Visual Question Answering).
        
        Args:
            image: Input image
            question: Question about the image
            **kwargs: Additional arguments
            
        Returns:
            ImageTextResult containing answer and metadata
        """
        config_dict = self.config.dict()
        config_dict['task_type'] = ImageTaskType.VQA
        config_dict['question'] = question
        config_dict.update(kwargs)
        config = ImageCaptionConfig(**config_dict)
        
        return self.caption_image(image, **config.dict())
    
    async def answer_question_async(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        question: str,
        **kwargs
    ) -> ImageTextResult:
        """
        Asynchronously answer a question about an image.
        
        Args:
            image: Input image
            question: Question about the image
            **kwargs: Additional arguments
            
        Returns:
            ImageTextResult containing answer and metadata
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.answer_question(image, question, **kwargs)
        )
    
    def batch_caption(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes]],
        prompts: Optional[List[str]] = None,
        batch_size: int = 4,
        show_progress: bool = False,
        **kwargs
    ) -> List[ImageTextResult]:
        """
        Generate captions for multiple images in batches.
        
        Args:
            images: List of input images
            prompts: Optional list of prompts (one per image)
            batch_size: Number of images to process at once
            show_progress: Whether to show progress bar
            **kwargs: Additional arguments
            
        Returns:
            List of ImageTextResults
        """
        if prompts is not None and len(prompts) != len(images):
            raise ValueError("Number of prompts must match number of images")
        
        results = []
        total_batches = (len(images) + batch_size - 1) // batch_size
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_prompts = prompts[i:i + batch_size] if prompts else None
            
            if show_progress:
                batch_num = i // batch_size + 1
                logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            for j, image in enumerate(batch_images):
                prompt = batch_prompts[j] if batch_prompts else None
                try:
                    result = self.caption_image(image, prompt, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process image {i + j}: {e}")
                    # Create error result
                    error_result = ImageTextResult(
                        text="[ERROR] Failed to generate caption",
                        model_name=self.model_name,
                        model_version=self.get_version(),
                        input_image_info={"error": str(e)},
                        task_type=ImageTaskType.CAPTIONING,
                        metadata={"error": True, "error_message": str(e)}
                    )
                    results.append(error_result)
        
        return results
    
    def _load_image(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        format: Optional[ImageFormat] = None
    ) -> Image.Image:
        """
        Load image from various input formats.
        
        Args:
            image_input: Input image in various formats
            format: Optional hint about input format
            
        Returns:
            PIL Image object
            
        Raises:
            ValueError: If image format is not supported
            RuntimeError: If image loading fails
        """
        try:
            # Check cache first
            cache_key = self._get_image_cache_key(image_input)
            if cache_key in self._image_cache:
                self._cache_hits += 1
                return self._image_cache[cache_key]
            
            self._cache_misses += 1
            image = None
            
            # Determine format if not provided
            if format is None:
                format = self._detect_image_format(image_input)
            
            # Load based on format
            if format == ImageFormat.PIL:
                image = image_input
                
            elif format == ImageFormat.NUMPY:
                # Convert numpy array to PIL Image
                if isinstance(image_input, np.ndarray):
                    if image_input.ndim == 2:  # Grayscale
                        image = Image.fromarray(image_input, mode='L')
                    elif image_input.ndim == 3:  # RGB/RGBA
                        if image_input.shape[2] == 3:
                            image = Image.fromarray(image_input, mode='RGB')
                        elif image_input.shape[2] == 4:
                            image = Image.fromarray(image_input, mode='RGBA')
                        else:
                            raise ValueError(f"Unsupported numpy array shape: {image_input.shape}")
                    else:
                        raise ValueError(f"Unsupported numpy array dimension: {image_input.ndim}")
                else:
                    raise ValueError(f"Expected numpy array for format {format}")
                
            elif format == ImageFormat.TENSOR:
                # Convert PyTorch tensor to PIL Image
                if isinstance(image_input, torch.Tensor):
                    # Detach, move to CPU, convert to numpy
                    tensor = image_input.detach().cpu()
                    
                    # Handle different tensor formats
                    if tensor.ndim == 2:  # Grayscale
                        # Scale from [0, 1] or [0, 255]
                        if tensor.max() <= 1.0:
                            tensor = tensor * 255
                        tensor = tensor.byte()
                        array = tensor.numpy()
                        image = Image.fromarray(array, mode='L')
                        
                    elif tensor.ndim == 3:  # RGB
                        # Handle channel-first vs channel-last
                        if tensor.shape[0] in [1, 3, 4]:  # Channel-first
                            tensor = tensor.permute(1, 2, 0)
                        
                        # Scale from [0, 1] or [0, 255]
                        if tensor.max() <= 1.0:
                            tensor = tensor * 255
                        tensor = tensor.byte()
                        array = tensor.numpy()
                        
                        if array.shape[2] == 1:  # Grayscale in 3D
                            image = Image.fromarray(array[:, :, 0], mode='L')
                        elif array.shape[2] == 3:
                            image = Image.fromarray(array, mode='RGB')
                        elif array.shape[2] == 4:
                            image = Image.fromarray(array, mode='RGBA')
                        else:
                            raise ValueError(f"Unsupported tensor shape: {tensor.shape}")
                    else:
                        raise ValueError(f"Unsupported tensor dimension: {tensor.ndim}")
                else:
                    raise ValueError(f"Expected torch.Tensor for format {format}")
                
            elif format == ImageFormat.BYTES:
                # Load from bytes
                if isinstance(image_input, bytes):
                    image = Image.open(io.BytesIO(image_input))
                else:
                    raise ValueError(f"Expected bytes for format {format}")
                
            elif format == ImageFormat.BASE64:
                # Decode base64 string
                if isinstance(image_input, str):
                    # Remove data URL prefix if present
                    if image_input.startswith('data:image'):
                        # Extract base64 part after comma
                        image_input = image_input.split(',', 1)[1]
                    
                    image_data = base64.b64decode(image_input)
                    image = Image.open(io.BytesIO(image_data))
                else:
                    raise ValueError(f"Expected string for base64 format")
                
            elif format == ImageFormat.FILE_PATH:
                # Load from file path
                path = Path(str(image_input))
                if not path.exists():
                    raise FileNotFoundError(f"Image file not found: {path}")
                image = Image.open(path)
                
            elif format == ImageFormat.URL:
                # Load from URL
                import requests
                from io import BytesIO
                
                if isinstance(image_input, str):
                    response = requests.get(image_input, timeout=10)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content))
                else:
                    raise ValueError(f"Expected string URL for format {format}")
                
            else:
                raise ValueError(f"Unsupported image format: {format}")
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Cache the loaded image
            self._image_cache[cache_key] = image
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise RuntimeError(f"Failed to load image: {e}")
    
    def _detect_image_format(
        self,
        image_input: Any
    ) -> ImageFormat:
        """
        Detect the format of an image input.
        
        Args:
            image_input: Input image
            
        Returns:
            Detected ImageFormat
            
        Raises:
            ValueError: If format cannot be detected
        """
        if isinstance(image_input, Image.Image):
            return ImageFormat.PIL
        elif isinstance(image_input, np.ndarray):
            return ImageFormat.NUMPY
        elif isinstance(image_input, torch.Tensor):
            return ImageFormat.TENSOR
        elif isinstance(image_input, bytes):
            return ImageFormat.BYTES
        elif isinstance(image_input, str):
            # Check if it's a base64 string
            if (len(image_input) > 100 and 
                (image_input.startswith('/9j/') or  # JPEG
                 image_input.startswith('iVBOR') or  # PNG
                 image_input.startswith('R0lGOD') or  # GIF
                 image_input.startswith('Qk') or  # BMP
                 image_input.startswith('data:image'))):
                return ImageFormat.BASE64
            # Check if it's a URL
            elif image_input.startswith(('http://', 'https://')):
                return ImageFormat.URL
            # Assume it's a file path
            else:
                return ImageFormat.FILE_PATH
        elif isinstance(image_input, Path):
            return ImageFormat.FILE_PATH
        else:
            raise ValueError(f"Cannot detect format for input type: {type(image_input)}")
    
    def _get_image_cache_key(
        self,
        image_input: Any
    ) -> str:
        """
        Generate cache key for an image input.
        
        Args:
            image_input: Input image
            
        Returns:
            Cache key string
        """
        import hashlib
        
        # Create a string representation of the input
        if isinstance(image_input, (str, Path)):
            # For files and URLs, use the path/URL
            key = str(image_input)
        elif isinstance(image_input, bytes):
            # For bytes, use hash
            key = hashlib.sha256(image_input).hexdigest()
        elif isinstance(image_input, (Image.Image, np.ndarray, torch.Tensor)):
            # For images/arrays/tensors, use a hash of a summary
            if isinstance(image_input, Image.Image):
                summary = f"{image_input.mode}{image_input.size}"
            elif isinstance(image_input, np.ndarray):
                summary = f"{image_input.shape}{image_input.dtype}{image_input.sum()}"
            elif isinstance(image_input, torch.Tensor):
                summary = f"{image_input.shape}{image_input.dtype}{image_input.sum().item()}"
            key = hashlib.sha256(summary.encode()).hexdigest()
        else:
            # Fallback: use string representation
            key = str(image_input)
        
        # Include model name in key to avoid collisions between models
        return f"{self.model_name}_{key}"
    
    def _preprocess_image(
        self,
        image: Image.Image,
        target_size: Optional[Tuple[int, int]] = None,
        normalize: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Preprocess image for model input.
        
        Args:
            image: PIL Image
            target_size: Target size for resizing
            normalize: Whether to normalize pixel values
            
        Returns:
            Dictionary with preprocessed image and metadata
        """
        if target_size is None:
            target_size = self.image_size
        if normalize is None:
            normalize = self.normalize_image
        
        original_size = image.size
        original_mode = image.mode
        
        # Resize image
        if image.size != target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to tensor if model expects it
        # This is model-specific and should be overridden by subclasses
        # Base implementation returns PIL image
        return {
            'image': image,
            'original_size': original_size,
            'original_mode': original_mode,
            'processed_size': target_size,
            'normalized': normalize,
        }
    
    def clear_image_cache(self) -> None:
        """Clear the image cache."""
        self._image_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Cleared image cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "cache_size": len(self._image_cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": self._cache_hits / max(self._cache_hits + self._cache_misses, 1),
            "total_images_processed": self._total_images_processed,
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        if not self._processing_times:
            return {"message": "No images processed yet"}
        
        stats = {
            "total_images": self._total_images_processed,
            "processing_times_ms": self._processing_times.copy(),
            "avg_processing_time_ms": sum(self._processing_times) / len(self._processing_times),
            "min_processing_time_ms": min(self._processing_times),
            "max_processing_time_ms": max(self._processing_times),
        }
        
        # Calculate percentiles
        if len(self._processing_times) >= 10:
            stats["p50_processing_time_ms"] = np.percentile(self._processing_times, 50)
            stats["p90_processing_time_ms"] = np.percentile(self._processing_times, 90)
            stats["p95_processing_time_ms"] = np.percentile(self._processing_times, 95)
            stats["p99_processing_time_ms"] = np.percentile(self._processing_times, 99)
        
        return stats
    
    def is_loaded(self) -> bool:
        """
        Check if model is loaded.
        
        Returns:
            True if model is loaded
        """
        return self._is_loaded
    
    def get_device(self) -> Optional[str]:
        """
        Get current device.
        
        Returns:
            Device string or None if not loaded
        """
        return self._device
    
    def move_to_device(self, device: Union[str, ModelDevice]) -> None:
        """
        Move model to specified device.
        
        Args:
            device: Target device
            
        Raises:
            ValueError: If model is not loaded
        """
        if not self._is_loaded:
            raise ValueError("Model must be loaded before moving to device")
        
        if isinstance(device, ModelDevice):
            device_str = device.value
        else:
            device_str = device
        
        if device_str == 'auto':
            device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if self._device != device_str:
            logger.info(f"Moving model from {self._device} to {device_str}")
            
            try:
                if hasattr(self.model, 'to'):
                    self.model.to(device_str)
                self._device = device_str
                
                # Update processor/feature_extractor if applicable
                if self.processor is not None and hasattr(self.processor, 'to'):
                    self.processor.to(device_str)
                if self.feature_extractor is not None and hasattr(self.feature_extractor, 'to'):
                    self.feature_extractor.to(device_str)
                    
            except Exception as e:
                logger.error(f"Failed to move model to {device_str}: {e}")
                raise RuntimeError(f"Failed to move model to {device_str}: {e}")
    
    @abstractmethod
    def get_version(self) -> str:
        """
        Get model version.
        
        Returns:
            Model version string
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "model_version": self.get_version(),
            "loaded": self._is_loaded,
            "device": self._device,
            "config": self.config.dict(),
            "image_size": self.image_size,
            "task_type": self.config.task_type.value,
        }
    
    def save_result(
        self,
        result: ImageTextResult,
        output_path: Union[str, Path],
        format: str = "json"
    ) -> None:
        """
        Save result to file.
        
        Args:
            result: ImageTextResult to save
            output_path: Path to save file
            format: File format ("json", "txt", "yaml")
            
        Raises:
            ValueError: If format is not supported
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        elif format == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result.text)
        elif format == "yaml":
            import yaml
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(result.to_dict(), f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved result to {output_path} in {format} format")
    
    def __call__(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        prompt: Optional[str] = None,
        **kwargs
    ) -> ImageTextResult:
        """
        Make the model callable for captioning.
        
        Args:
            image: Input image
            prompt: Optional prompt
            **kwargs: Additional arguments
            
        Returns:
            ImageTextResult
        """
        return self.caption_image(image, prompt, **kwargs)
    
    def __repr__(self) -> str:
        """String representation."""
        loaded = self._is_loaded
        device = self._device if loaded else "not loaded"
        return f"BaseImageToTextModel(model_name={self.model_name}, loaded={loaded}, device={device})"
    
    def __len__(self) -> int:
        """Get number of images processed."""
        return self._total_images_processed
    
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
            # Clear model components
            del self.model
            del self.processor
            del self.tokenizer
            del self.feature_extractor
            
            self.model = None
            self.processor = None
            self.tokenizer = None
            self.feature_extractor = None
            self._is_loaded = False
            
            # Clear caches
            self.clear_image_cache()
            
            # Free GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info(f"Unloaded model: {self.model_name}")


class ImageToTextModelFactory:
    """
    Factory for creating image-to-text model instances.
    """
    
    # Registry of model classes
    _model_registry: Dict[str, type] = {}
    
    @classmethod
    def register_model(cls, model_type: str, model_class: type) -> None:
        """
        Register an image-to-text model class.
        
        Args:
            model_type: Type identifier for the model
            model_class: The model class to register
        """
        cls._model_registry[model_type] = model_class
        logger.info(f"Registered image-to-text model type: {model_type} -> {model_class.__name__}")
    
    @classmethod
    def create(
        cls,
        model_type: str,
        model_name: str,
        config: Optional[ImageCaptionConfig] = None,
        **kwargs
    ) -> BaseImageToTextModel:
        """
        Create an image-to-text model instance.
        
        Args:
            model_type: Type of model
            model_name: Name of the model
            config: Model configuration
            **kwargs: Additional arguments
            
        Returns:
            Image-to-text model instance
            
        Raises:
            ValueError: If model_type is not registered
        """
        if model_type not in cls._model_registry:
            raise ValueError(
                f"Unknown image-to-text model type: {model_type}. "
                f"Available types: {list(cls._model_registry.keys())}"
            )
        
        model_class = cls._model_registry[model_type]
        
        if config is None:
            config = ImageCaptionConfig()
        
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
    'BaseImageToTextModel',
    'ImageTextResult',
    'ImageCaptionConfig',
    'ImageTaskType',
    'ImageFormat',
    'TextGenerationConfig',
    'ImageToTextModelFactory',
]