"""
Base classes and interfaces for multimodal models.
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


class MultimodalTaskType(str, Enum):
    """Types of multimodal tasks."""
    
    # Image-related tasks
    IMAGE_CAPTIONING = "image_captioning"
    VISUAL_QUESTION_ANSWERING = "visual_question_answering"
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    IMAGE_SEGMENTATION = "image_segmentation"
    IMAGE_GENERATION = "image_generation"
    
    # Text-related tasks
    TEXT_CLASSIFICATION = "text_classification"
    TEXT_GENERATION = "text_generation"
    TEXT_SUMMARIZATION = "text_summarization"
    
    # Audio-related tasks
    AUDIO_TRANSCRIPTION = "audio_transcription"
    AUDIO_CLASSIFICATION = "audio_classification"
    TEXT_TO_SPEECH = "text_to_speech"
    
    # Cross-modal tasks
    IMAGE_TEXT_MATCHING = "image_text_matching"
    TEXT_IMAGE_RETRIEVAL = "text_image_retrieval"
    IMAGE_TEXT_RETRIEVAL = "image_text_retrieval"
    MULTIMODAL_CLASSIFICATION = "multimodal_classification"
    MULTIMODAL_GENERATION = "multimodal_generation"
    MULTIMODAL_EMBEDDING = "multimodal_embedding"
    
    # Specialized tasks
    DOCUMENT_UNDERSTANDING = "document_understanding"
    VISUAL_REASONING = "visual_reasoning"
    MULTIMODAL_CHAT = "multimodal_chat"
    REFERRING_EXPRESSION = "referring_expression"
    VISUAL_GROUNDING = "visual_grounding"


class ImageFormat:
    """Supported image formats."""
    PIL = "pil"
    NUMPY = "numpy"
    TORCH = "torch"
    BYTES = "bytes"
    BASE64 = "base64"
    FILE = "file"


class AudioFormat:
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    NUMPY = "numpy"
    TORCH = "torch"
    BYTES = "bytes"
    FILE = "file"


class TextFormat:
    """Supported text formats."""
    STRING = "string"
    TOKENS = "tokens"
    EMBEDDING = "embedding"


@dataclass
class ImageInput:
    """Container for image input data."""
    
    image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes]
    format: str = ImageFormat.PIL
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize image input."""
        if self.format not in [ImageFormat.PIL, ImageFormat.NUMPY, 
                              ImageFormat.TORCH, ImageFormat.BYTES,
                              ImageFormat.BASE64, ImageFormat.FILE]:
            raise ValueError(f"Unsupported image format: {self.format}")


@dataclass
class TextInput:
    """Container for text input data."""
    
    text: Union[str, List[str], List[int], np.ndarray, torch.Tensor]
    format: str = TextFormat.STRING
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize text input."""
        if self.format not in [TextFormat.STRING, TextFormat.TOKENS, TextFormat.EMBEDDING]:
            raise ValueError(f"Unsupported text format: {self.format}")


@dataclass
class AudioInput:
    """Container for audio input data."""
    
    audio: Union[str, Path, np.ndarray, torch.Tensor, bytes]
    format: str = AudioFormat.WAV
    sample_rate: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize audio input."""
        if self.format not in [AudioFormat.WAV, AudioFormat.MP3, AudioFormat.NUMPY,
                              AudioFormat.TORCH, AudioFormat.BYTES, AudioFormat.FILE]:
            raise ValueError(f"Unsupported audio format: {self.format}")


@dataclass
class MultimodalInput:
    """Container for multimodal input data."""
    
    images: Optional[List[ImageInput]] = None
    texts: Optional[List[TextInput]] = None
    audios: Optional[List[AudioInput]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate that at least one modality is provided."""
        if not self.images and not self.texts and not self.audios:
            raise ValueError("At least one modality (images, texts, or audios) must be provided")


@dataclass
class MultimodalResult:
    """Result from multimodal model processing."""
    
    # Main result
    result: Any
    
    # Model information
    model_name: str
    model_version: str = "unknown"
    
    # Task information
    task_type: MultimodalTaskType
    
    # Performance metrics
    processing_time_ms: float = 0.0
    tokens_processed: Optional[int] = None
    memory_used_mb: Optional[float] = None
    
    # Input information
    input_info: Dict[str, Any] = field(default_factory=dict)
    
    # Output details
    confidence: Optional[float] = None
    embeddings: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None
    attention_maps: Optional[Union[np.ndarray, torch.Tensor, List[List[float]]]] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        result_dict = {
            "result": self.result,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "task_type": self.task_type.value,
            "processing_time_ms": self.processing_time_ms,
            "tokens_processed": self.tokens_processed,
            "memory_used_mb": self.memory_used_mb,
            "input_info": self.input_info,
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
        
        return result_dict
    
    def to_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


class BaseMultimodalModel(ABC):
    """
    Abstract base class for multimodal models.
    
    This class defines the interface for all multimodal models in WorldBrief360.
    """
    
    def __init__(
        self,
        model_name: str,
        task_type: MultimodalTaskType = MultimodalTaskType.MULTIMODAL_CLASSIFICATION,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the multimodal model.
        
        Args:
            model_name: Name or identifier of the model
            task_type: Type of multimodal task
            device: Device to run model on (cpu, cuda, etc.)
            **kwargs: Additional model-specific arguments
        """
        self.model_name = model_name
        self.task_type = task_type
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._is_loaded = False
        self._model = None
        self._processor = None
        
        # Performance tracking
        self._processing_times: List[float] = []
        self._total_requests_processed = 0
        
        # Model configuration
        self.config = kwargs
        
        logger.info(f"Initialized {self.__class__.__name__} with model: {model_name}")
    
    @abstractmethod
    def load(self) -> None:
        """
        Load the model and processor.
        
        Raises:
            RuntimeError: If model loading fails
        """
        pass
    
    @abstractmethod
    def process(
        self,
        inputs: MultimodalInput,
        task_type: Optional[MultimodalTaskType] = None,
        **kwargs
    ) -> MultimodalResult:
        """
        Process multimodal inputs.
        
        Args:
            inputs: Multimodal input data
            task_type: Override task type for this request
            **kwargs: Additional processing arguments
            
        Returns:
            MultimodalResult containing the processed result
            
        Raises:
            ValueError: If model is not loaded
            RuntimeError: If processing fails
        """
        pass
    
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
            "task_type": self.task_type.value,
            "device": self._device,
            "is_loaded": self._is_loaded,
            "total_requests_processed": self._total_requests_processed,
            "average_processing_time_ms": self.get_average_processing_time(),
        }
    
    def get_model_capabilities(self) -> Dict[str, Any]:
        """
        Get capabilities of the model.
        
        Returns:
            Dictionary with model capabilities
        """
        return {
            "supports_images": False,
            "supports_text": False,
            "supports_audio": False,
            "supports_video": False,
            "max_image_size": None,
            "max_text_length": None,
            "max_audio_length": None,
            "supported_tasks": [],
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
            
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # Resize if target size is specified
            if target_size and pil_image.size != target_size:
                pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
            
            return pil_image
            
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise ValueError(f"Failed to load image: {e}")
    
    def _load_audio(
        self,
        audio: Union[str, Path, np.ndarray, torch.Tensor, bytes],
        target_sample_rate: Optional[int] = None,
        format: str = AudioFormat.WAV
    ) -> Tuple[np.ndarray, int]:
        """
        Load and normalize audio from various input formats.
        
        Args:
            audio: Input audio in various formats
            target_sample_rate: Target sample rate for resampling
            format: Expected format of the input
            
        Returns:
            Tuple of (audio_array, sample_rate)
            
        Raises:
            ValueError: If audio cannot be loaded
            ImportError: If audio dependencies are not installed
        """
        try:
            import librosa
            import soundfile as sf
            
            audio_array = None
            sample_rate = None
            
            if isinstance(audio, (str, Path)):
                # Load from file
                if str(audio).endswith('.mp3'):
                    audio_array, sample_rate = librosa.load(str(audio), sr=target_sample_rate)
                else:
                    audio_array, sample_rate = sf.read(str(audio))
                    
                    # Convert stereo to mono if needed
                    if audio_array.ndim > 1:
                        audio_array = np.mean(audio_array, axis=1)
            
            elif isinstance(audio, np.ndarray):
                # Already a numpy array
                audio_array = audio
                sample_rate = target_sample_rate or 16000  # Default
            
            elif isinstance(audio, torch.Tensor):
                # Convert torch tensor to numpy
                audio_array = audio.cpu().numpy()
                sample_rate = target_sample_rate or 16000
            
            elif isinstance(audio, bytes):
                # Load from bytes
                audio_array, sample_rate = sf.read(io.BytesIO(audio))
                
                # Convert stereo to mono if needed
                if audio_array.ndim > 1:
                    audio_array = np.mean(audio_array, axis=1)
            
            else:
                raise ValueError(f"Unsupported audio type: {type(audio)}")
            
            # Resample if needed
            if target_sample_rate and sample_rate != target_sample_rate:
                audio_array = librosa.resample(
                    audio_array,
                    orig_sr=sample_rate,
                    target_sr=target_sample_rate
                )
                sample_rate = target_sample_rate
            
            return audio_array, sample_rate
            
        except ImportError as e:
            logger.error(f"Audio dependencies not installed: {e}")
            raise ImportError("Audio processing requires librosa and soundfile. "
                            "Install with: pip install librosa soundfile")
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise ValueError(f"Failed to load audio: {e}")
    
    def _preprocess_text(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Preprocess text input.
        
        Args:
            text: Input text or list of texts
            max_length: Maximum length for truncation
            **kwargs: Additional preprocessing arguments
            
        Returns:
            Preprocessed text
        """
        if isinstance(text, list):
            if max_length:
                return [t[:max_length] for t in text]
            return text
        else:
            if max_length:
                return text[:max_length]
            return text
    
    def get_average_processing_time(self) -> float:
        """
        Get average processing time for all requests.
        
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
        self._total_requests_processed = 0
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
               f"task_type={self.task_type}, loaded={self._is_loaded})"


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


class ModelConfig(BaseModel):
    """Base configuration for models."""
    
    model_name: str = Field(..., description="Name or identifier of the model")
    task_type: MultimodalTaskType = Field(
        default=MultimodalTaskType.MULTIMODAL_CLASSIFICATION,
        description="Type of task the model performs"
    )
    device: ModelDevice = Field(
        default_factory=ModelDevice.get_best_device,
        description="Device to run the model on"
    )
    batch_size: int = Field(default=1, description="Batch size for processing")
    max_sequence_length: Optional[int] = Field(default=None, description="Maximum sequence length")
    max_image_size: Optional[Tuple[int, int]] = Field(default=None, description="Maximum image size")
    
    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True