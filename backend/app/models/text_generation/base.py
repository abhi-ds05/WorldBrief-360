"""
Base classes and interfaces for image-to-text models.
"""

import abc
import base64
import io
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ImageTaskType(str, Enum):
    """Types of image understanding tasks."""
    
    IMAGE_CAPTIONING = "image_captioning"
    VISUAL_QUESTION_ANSWERING = "visual_question_answering"
    OBJECT_DETECTION = "object_detection"
    IMAGE_CLASSIFICATION = "image_classification"
    TEXT_EXTRACTION = "text_extraction"  # OCR
    DOCUMENT_UNDERSTANDING = "document_understanding"
    VISUAL_REASONING = "visual_reasoning"
    REFERRING_EXPRESSION = "referring_expression"  # Describe specific region
    VISUAL_GROUNDING = "visual_grounding"  # Find region from description
    MULTIMODAL_CHAT = "multimodal_chat"
    VISUAL_INSTRUCTION = "visual_instruction"  # Follow visual instructions
    IMAGE_GENERATION = "image_generation"  # Text-to-image
    IMAGE_EDITING = "image_editing"
    IMAGE_SEGMENTATION = "image_segmentation"


class ImageFormat:
    """Supported image formats."""
    PIL = "pil"
    NUMPY = "numpy"
    TORCH = "torch"
    BYTES = "bytes"
    BASE64 = "base64"
    FILE = "file"


class TextGenerationParameters(BaseModel):
    """Parameters for text generation."""
    
    max_new_tokens: int = Field(default=512, description="Maximum new tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Top-p sampling parameter")
    top_k: int = Field(default=50, description="Top-k sampling parameter")
    repetition_penalty: float = Field(default=1.0, description="Repetition penalty")
    do_sample: bool = Field(default=True, description="Whether to use sampling")
    num_beams: int = Field(default=1, description="Number of beams for beam search")
    length_penalty: float = Field(default=1.0, description="Length penalty for beam search")
    no_repeat_ngram_size: int = Field(default=0, description="No repeat n-gram size")
    early_stopping: bool = Field(default=False, description="Whether to stop early in beam search")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    
    class Config:
        arbitrary_types_allowed = True


class ImageCaptionConfig(BaseModel):
    """Configuration for image captioning/understanding."""
    
    task_type: ImageTaskType = Field(
        default=ImageTaskType.IMAGE_CAPTIONING,
        description="Type of image understanding task"
    )
    image_size: Tuple[int, int] = Field(
        default=(224, 224),
        description="Target image size for processing"
    )
    generation_config: TextGenerationParameters = Field(
        default_factory=TextGenerationParameters,
        description="Text generation parameters"
    )
    return_confidence: bool = Field(
        default=False,
        description="Whether to return confidence scores"
    )
    return_embeddings: bool = Field(
        default=False,
        description="Whether to return image/text embeddings"
    )
    return_attention: bool = Field(
        default=False,
        description="Whether to return attention maps"
    )
    language: Optional[str] = Field(
        default=None,
        description="Language for output (if multilingual)"
    )
    
    class Config:
        arbitrary_types_allowed = True


@dataclass
class ImageTextResult:
    """Result from image-to-text model processing."""
    
    # Main generated text
    text: str
    
    # Model information
    model_name: str
    model_version: str = "unknown"
    
    # Task information
    task_type: ImageTaskType = ImageTaskType.IMAGE_CAPTIONING
    
    # Performance metrics
    processing_time_ms: float = 0.0
    tokens_generated: Optional[int] = None
    tokens_processed: Optional[int] = None
    memory_used_mb: Optional[float] = None
    
    # Input information
    input_image_info: Dict[str, Any] = field(default_factory=dict)
    
    # Additional outputs
    confidence: Optional[float] = None
    embeddings: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None
    attention_maps: Optional[Union[np.ndarray, torch.Tensor, List[List[float]]]] = None
    bounding_boxes: Optional[List[Dict[str, Any]]] = None  # For object detection
    ocr_results: Optional[List[Dict[str, Any]]] = None  # For text extraction
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        result_dict = {
            "text": self.text,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "task_type": self.task_type.value,
            "processing_time_ms": self.processing_time_ms,
            "tokens_generated": self.tokens_generated,
            "tokens_processed": self.tokens_processed,
            "memory_used_mb": self.memory_used_mb,
            "input_image_info": self.input_image_info,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
        
        # Handle embeddings
        if self.embeddings is not None:
            if isinstance(self.embeddings, (np.ndarray, torch.Tensor)):
                result_dict["embeddings"] = self.embeddings.tolist()
            else:
                result_dict["embeddings"] = self.embeddings
        
        # Handle attention maps
        if self.attention_maps is not None:
            if isinstance(self.attention_maps, (np.ndarray, torch.Tensor)):
                result_dict["attention_maps"] = self.attention_maps.tolist()
            else:
                result_dict["attention_maps"] = self.attention_maps
        
        # Handle bounding boxes
        if self.bounding_boxes is not None:
            result_dict["bounding_boxes"] = self.bounding_boxes
        
        # Handle OCR results
        if self.ocr_results is not None:
            result_dict["ocr_results"] = self.ocr_results
        
        return result_dict
    
    def to_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the result."""
        return {
            "text_preview": self.text[:200] + "..." if len(self.text) > 200 else self.text,
            "model": self.model_name,
            "task": self.task_type.value,
            "processing_time_ms": self.processing_time_ms,
            "confidence": self.confidence,
            "tokens_generated": self.tokens_generated,
        }


class BaseImageToTextModel(ABC):
    """
    Abstract base class for image-to-text models.
    
    This class defines the interface for all image understanding models in WorldBrief360.
    """
    
    def __init__(
        self,
        model_name: str,
        config: Optional[ImageCaptionConfig] = None,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the image-to-text model.
        
        Args:
            model_name: Name or identifier of the model
            config: Configuration for image understanding
            device: Device to run model on (cpu, cuda, etc.)
            **kwargs: Additional model-specific arguments
        """
        self.model_name = model_name
        
        # Default configuration
        if config is None:
            config = ImageCaptionConfig()
        self.config = config
        
        # Device setup
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._is_loaded = False
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.processor = None
        
        # Image processing
        self.image_size = config.image_size
        
        # Performance tracking
        self._processing_times: List[float] = []
        self._total_images_processed = 0
        
        # Additional configuration
        self._extra_config = kwargs
        
        logger.info(f"Initialized {self.__class__.__name__} with model: {model_name}")
    
    @abstractmethod
    def load(self) -> None:
        """
        Load the model and processor/tokenizer.
        
        Raises:
            RuntimeError: If model loading fails
        """
        pass
    
    @abstractmethod
    def process_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process an image for model input.
        
        Args:
            image: Input image in various formats
            prompt: Optional text prompt for the image
            **kwargs: Additional processing arguments
            
        Returns:
            Dictionary containing processed inputs and metadata
            
        Raises:
            ValueError: If model is not loaded
            RuntimeError: If processing fails
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
            image_input: Processed image and prompt input
            config: Configuration for text generation
            **kwargs: Additional generation arguments
            
        Returns:
            ImageTextResult containing generated text and metadata
            
        Raises:
            ValueError: If model is not loaded
            RuntimeError: If generation fails
        """
        pass
    
    def caption_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        prompt: Optional[str] = None,
        config: Optional[ImageCaptionConfig] = None,
        **kwargs
    ) -> ImageTextResult:
        """
        Generate caption/description for an image.
        
        This is a convenience method that combines process_image and generate_text.
        
        Args:
            image: Input image
            prompt: Optional prompt for the image
            config: Configuration for captioning
            **kwargs: Additional arguments
            
        Returns:
            ImageTextResult containing caption and metadata
        """
        # Process image
        image_input = self.process_image(image, prompt, **kwargs)
        
        # Generate text
        return self.generate_text(image_input, config, **kwargs)
    
    def answer_question(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        question: str,
        config: Optional[ImageCaptionConfig] = None,
        **kwargs
    ) -> ImageTextResult:
        """
        Answer a question about an image (VQA).
        
        Args:
            image: Input image
            question: Question about the image
            config: Configuration for VQA
            **kwargs: Additional arguments
            
        Returns:
            ImageTextResult containing answer and metadata
        """
        # Update config for VQA
        if config is None:
            config = self.config.copy()
        config.task_type = ImageTaskType.VISUAL_QUESTION_ANSWERING
        
        return self.caption_image(image, question, config, **kwargs)
    
    def extract_text(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        config: Optional[ImageCaptionConfig] = None,
        **kwargs
    ) -> ImageTextResult:
        """
        Extract text from image (OCR).
        
        Args:
            image: Input image with text
            config: Configuration for text extraction
            **kwargs: Additional arguments
            
        Returns:
            ImageTextResult containing extracted text and metadata
        """
        # Update config for OCR
        if config is None:
            config = self.config.copy()
        config.task_type = ImageTaskType.TEXT_EXTRACTION
        
        # Use OCR-specific prompt if not provided
        kwargs.setdefault("prompt", "Extract all text from this image.")
        
        return self.caption_image(image, config=config, **kwargs)
    
    @abstractmethod
    def unload(self) -> None:
        """Unload the model from memory."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "model_type": self.__class__.__name__,
            "task_type": self.config.task_type.value,
            "device": self._device,
            "is_loaded": self._is_loaded,
            "image_size": self.image_size,
            "total_images_processed": self._total_images_processed,
            "average_processing_time_ms": self.get_average_processing_time(),
        }
    
    def get_model_capabilities(self) -> Dict[str, Any]:
        """
        Get capabilities of the model.
        
        Returns:
            Dictionary with model capabilities
        """
        return {
            "supports_captioning": False,
            "supports_vqa": False,
            "supports_ocr": False,
            "supports_object_detection": False,
            "supports_multimodal_chat": False,
            "max_image_size": self.image_size,
            "max_text_length": None,
            "multilingual": False,
            "model_parameters": 0,
            "memory_requirements_mb": 0,
        }
    
    def _load_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        target_size: Optional[Tuple[int, int]] = None,
        format: str = ImageFormat.PIL
    ) -> Image.Image:
        """
        Load and normalize an image from various input formats.
        
        Args:
            image: Input image in various formats
            target_size: Target size for resizing (width, height)
            format: Expected format of the input
            
        Returns:
            PIL Image object
            
        Raises:
            ValueError: If image cannot be loaded
        """
        try:
            pil_image = None
            
            if isinstance(image, (str, Path)):
                # Load from file path
                pil_image = Image.open(str(image))
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
            
            elif isinstance(image, Image.Image):
                # Already a PIL Image
                pil_image = image.copy()
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
            
            elif isinstance(image, np.ndarray):
                # Convert numpy array to PIL Image
                if image.ndim == 2:
                    # Grayscale
                    pil_image = Image.fromarray(image).convert('RGB')
                elif image.ndim == 3:
                    # RGB or similar
                    if image.shape[2] == 3:
                        pil_image = Image.fromarray(image)
                    elif image.shape[2] == 4:
                        pil_image = Image.fromarray(image[:, :, :3])
                    elif image.shape[2] == 1:
                        pil_image = Image.fromarray(image[:, :, 0]).convert('RGB')
                    else:
                        raise ValueError(f"Unsupported numpy array shape: {image.shape}")
                else:
                    raise ValueError(f"Unsupported numpy array dimension: {image.ndim}")
            
            elif isinstance(image, torch.Tensor):
                # Convert torch tensor to PIL Image
                image_np = image.cpu().numpy()
                if image_np.ndim == 3:
                    # Remove batch dimension if present
                    if image_np.shape[0] == 3:
                        # CHW to HWC
                        image_np = image_np.transpose(1, 2, 0)
                    elif image_np.shape[2] == 3:
                        # Already HWC
                        pass
                    else:
                        raise ValueError(f"Unsupported tensor shape: {image.shape}")
                
                # Normalize if needed
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                else:
                    image_np = image_np.astype(np.uint8)
                
                pil_image = Image.fromarray(image_np)
            
            elif isinstance(image, bytes):
                # Load from bytes
                pil_image = Image.open(io.BytesIO(image))
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
            
            elif format == ImageFormat.BASE64 and isinstance(image, str):
                # Load from base64 string
                image_bytes = base64.b64decode(image)
                pil_image = Image.open(io.BytesIO(image_bytes))
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
            
            else:
                raise ValueError(f"Unsupported image type: {type(image)} with format: {format}")
            
            # Resize if target size is specified
            if target_size and pil_image.size != target_size:
                pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
            
            return pil_image
            
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise ValueError(f"Failed to load image: {e}")
    
    def get_average_processing_time(self) -> float:
        """
        Get average processing time for all images.
        
        Returns:
            Average processing time in milliseconds
        """
        if not self._processing_times:
            return 0.0
        return sum(self._processing_times) / len(self._processing_times)
    
    def get_min_processing_time(self) -> float:
        """
        Get minimum processing time.
        
        Returns:
            Minimum processing time in milliseconds
        """
        if not self._processing_times:
            return 0.0
        return min(self._processing_times)
    
    def get_max_processing_time(self) -> float:
        """
        Get maximum processing time.
        
        Returns:
            Maximum processing time in milliseconds
        """
        if not self._processing_times:
            return 0.0
        return max(self._processing_times)
    
    def reset_performance_stats(self) -> None:
        """Reset performance statistics."""
        self._processing_times.clear()
        self._total_images_processed = 0
        logger.info("Reset performance statistics")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.unload()
        except:
            pass
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(model_name={self.model_name}, " \
               f"task_type={self.config.task_type}, loaded={self._is_loaded})"


class ModelDevice(str, Enum):
    """Available devices for model execution."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    AUTO = "auto"
    
    @classmethod
    def get_best_device(cls) -> str:
        """Get the best available device."""
        if torch.cuda.is_available():
            return cls.CUDA
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return cls.MPS
        else:
            return cls.CPU