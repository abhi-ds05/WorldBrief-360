"""
Base classes and interfaces for text-to-image models.
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
    """Types of image generation tasks."""
    
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_IMAGE = "image_to_image"  # img2img
    INPAINTING = "inpainting"  # Mask-based editing
    OUTPAINTING = "outpainting"  # Extending images
    SUPER_RESOLUTION = "super_resolution"  # Upscaling
    IMAGE_VARIATION = "image_variation"  # Generate variations
    STYLE_TRANSFER = "style_transfer"  # Apply style to image
    CONTROLNET = "controlnet"  # Conditioned generation
    DEPTH_TO_IMAGE = "depth_to_image"  # Depth conditioning
    SKETCH_TO_IMAGE = "sketch_to_image"  # Sketch conditioning
    POSE_TO_IMAGE = "pose_to_image"  # Pose conditioning


class ImageFormat:
    """Supported image formats."""
    PIL = "pil"
    NUMPY = "numpy"
    TORCH = "torch"
    BYTES = "bytes"
    BASE64 = "base64"
    FILE = "file"


class ImageGenerationParameters(BaseModel):
    """Parameters for image generation."""
    
    # Generation parameters
    num_inference_steps: int = Field(default=50, description="Number of denoising steps")
    guidance_scale: float = Field(default=7.5, description="Classifier-free guidance scale")
    negative_prompt: Optional[str] = Field(default=None, description="Negative prompt")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    
    # Image parameters
    width: int = Field(default=512, description="Image width")
    height: int = Field(default=512, description="Image height")
    num_images: int = Field(default=1, description="Number of images to generate")
    
    # Quality parameters
    quality: Optional[float] = Field(default=None, description="Overall quality (0-1)")
    detail: Optional[float] = Field(default=None, description="Detail level (0-1)")
    
    # Style parameters
    style: Optional[str] = Field(default=None, description="Style preset")
    artistic_style: Optional[str] = Field(default=None, description="Artistic style")
    
    # Advanced parameters
    strength: Optional[float] = Field(default=None, description="Strength for img2img (0-1)")
    eta: Optional[float] = Field(default=None, description="DDIM eta parameter")
    
    class Config:
        arbitrary_types_allowed = True


class ImageGenerationConfig(BaseModel):
    """Configuration for image generation."""
    
    task_type: ImageTaskType = Field(
        default=ImageTaskType.TEXT_TO_IMAGE,
        description="Type of image generation task"
    )
    output_format: str = Field(
        default=ImageFormat.PIL,
        description="Format for output images"
    )
    generation_params: ImageGenerationParameters = Field(
        default_factory=ImageGenerationParameters,
        description="Image generation parameters"
    )
    save_to_file: bool = Field(
        default=False,
        description="Whether to save generated images to files"
    )
    output_dir: Optional[str] = Field(
        default=None,
        description="Directory to save generated images"
    )
    return_metadata: bool = Field(
        default=True,
        description="Whether to return generation metadata"
    )
    safety_check: bool = Field(
        default=True,
        description="Whether to perform safety checks on generated images"
    )
    watermark: bool = Field(
        default=False,
        description="Whether to add watermark to generated images"
    )
    
    class Config:
        arbitrary_types_allowed = True


@dataclass
class ImageGenerationResult:
    """Result from text-to-image model generation."""
    
    # Generated images
    images: List[Union[Image.Image, np.ndarray, torch.Tensor, bytes, str]]
    
    # Model information
    model_name: str
    model_version: str = "unknown"
    
    # Task information
    task_type: ImageTaskType = ImageTaskType.TEXT_TO_IMAGE
    
    # Performance metrics
    processing_time_ms: float = 0.0
    images_generated: int = 0
    memory_used_mb: Optional[float] = None
    
    # Input information
    input_prompt: str = ""
    input_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Additional outputs
    seeds: Optional[List[int]] = None  # Seeds used for generation
    latents: Optional[List[np.ndarray]] = None  # Latent representations
    attention_maps: Optional[List[np.ndarray]] = None  # Attention maps
    scores: Optional[List[float]] = None  # Quality scores
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        result_dict = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "task_type": self.task_type.value,
            "processing_time_ms": self.processing_time_ms,
            "images_generated": self.images_generated,
            "memory_used_mb": self.memory_used_mb,
            "input_prompt": self.input_prompt,
            "input_parameters": self.input_parameters,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
        
        # Add seeds if available
        if self.seeds is not None:
            result_dict["seeds"] = self.seeds
        
        # Add scores if available
        if self.scores is not None:
            result_dict["scores"] = self.scores
        
        # Handle images based on format
        result_dict["images"] = []
        for img in self.images:
            if isinstance(img, Image.Image):
                # Convert PIL Image to base64
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                result_dict["images"].append({
                    "format": "base64",
                    "data": img_base64,
                    "size": img.size,
                    "mode": img.mode,
                })
            elif isinstance(img, np.ndarray):
                # Convert numpy array to base64
                pil_img = Image.fromarray(img)
                buffered = io.BytesIO()
                pil_img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                result_dict["images"].append({
                    "format": "base64",
                    "data": img_base64,
                    "shape": img.shape,
                })
            elif isinstance(img, bytes):
                # Already bytes
                img_base64 = base64.b64encode(img).decode('utf-8')
                result_dict["images"].append({
                    "format": "base64",
                    "data": img_base64,
                })
            elif isinstance(img, str):
                # Could be base64 or file path
                if img.startswith("data:image"):
                    result_dict["images"].append({
                        "format": "base64",
                        "data": img,
                    })
                else:
                    result_dict["images"].append({
                        "format": "file",
                        "path": img,
                    })
            else:
                # Unknown format
                result_dict["images"].append({
                    "format": "unknown",
                    "data": str(type(img)),
                })
        
        return result_dict
    
    def to_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def get_images_as_pil(self) -> List[Image.Image]:
        """Get all images as PIL Image objects."""
        pil_images = []
        for img in self.images:
            if isinstance(img, Image.Image):
                pil_images.append(img)
            elif isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img))
            elif isinstance(img, bytes):
                pil_images.append(Image.open(io.BytesIO(img)))
            elif isinstance(img, str) and img.startswith("data:image"):
                # Parse base64 string
                img_data = base64.b64decode(img.split(",")[1])
                pil_images.append(Image.open(io.BytesIO(img_data)))
            elif isinstance(img, str):
                # Assume file path
                pil_images.append(Image.open(img))
            elif isinstance(img, torch.Tensor):
                # Convert tensor to numpy
                img_np = img.cpu().numpy()
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img_np))
        return pil_images
    
    def save_images(self, output_dir: Union[str, Path], prefix: str = "generated") -> List[str]:
        """
        Save generated images to files.
        
        Args:
            output_dir: Directory to save images
            prefix: Prefix for filenames
            
        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        pil_images = self.get_images_as_pil()
        
        for i, img in enumerate(pil_images):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}_{i:03d}.png"
            filepath = output_dir / filename
            
            # Save image
            img.save(filepath, format="PNG", quality=95)
            saved_paths.append(str(filepath))
            
            logger.info(f"Saved image to {filepath}")
        
        return saved_paths
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the result."""
        return {
            "model": self.model_name,
            "task": self.task_type.value,
            "images_generated": self.images_generated,
            "processing_time_ms": self.processing_time_ms,
            "input_prompt": self.input_prompt[:100] + "..." if len(self.input_prompt) > 100 else self.input_prompt,
            "image_sizes": [img.size if hasattr(img, 'size') else str(type(img)) for img in self.images[:3]],
        }


class BaseTextToImageModel(ABC):
    """
    Abstract base class for text-to-image models.
    
    This class defines the interface for all image generation models in WorldBrief360.
    """
    
    def __init__(
        self,
        model_name: str,
        config: Optional[ImageGenerationConfig] = None,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the text-to-image model.
        
        Args:
            model_name: Name or identifier of the model
            config: Configuration for image generation
            device: Device to run model on (cpu, cuda, etc.)
            **kwargs: Additional model-specific arguments
        """
        self.model_name = model_name
        
        # Default configuration
        if config is None:
            config = ImageGenerationConfig()
        self.config = config
        
        # Device setup
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._is_loaded = False
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.scheduler = None
        self.feature_extractor = None
        self.safety_checker = None
        
        # Performance tracking
        self._processing_times: List[float] = []
        self._total_images_generated = 0
        
        # Additional configuration
        self._extra_config = kwargs
        
        logger.info(f"Initialized {self.__class__.__name__} with model: {model_name}")
    
    @abstractmethod
    def load(self) -> None:
        """
        Load the model and related components.
        
        Raises:
            RuntimeError: If model loading fails
        """
        pass
    
    @abstractmethod
    def generate_image(
        self,
        prompt: str,
        config: Optional[ImageGenerationConfig] = None,
        **kwargs
    ) -> ImageGenerationResult:
        """
        Generate image from text prompt.
        
        Args:
            prompt: Text prompt for image generation
            config: Configuration for image generation
            **kwargs: Additional generation arguments
            
        Returns:
            ImageGenerationResult containing generated images and metadata
            
        Raises:
            ValueError: If model is not loaded
            RuntimeError: If generation fails
        """
        pass
    
    def edit_image(
        self,
        image_path: Union[str, Path],
        prompt: str,
        mask_path: Optional[Union[str, Path]] = None,
        config: Optional[ImageGenerationConfig] = None,
        **kwargs
    ) -> ImageGenerationResult:
        """
        Edit an existing image (inpainting or img2img).
        
        Args:
            image_path: Path to input image
            prompt: Text prompt for editing
            mask_path: Optional mask for inpainting
            config: Configuration for image editing
            **kwargs: Additional arguments
            
        Returns:
            ImageGenerationResult containing edited image
            
        Raises:
            ValueError: If model is not loaded or doesn't support editing
            RuntimeError: If editing fails
        """
        if not self._is_loaded:
            raise ValueError(f"Model {self.model_name} is not loaded. Call load() first.")
        
        # Check if model supports editing
        capabilities = self.get_model_capabilities()
        if not capabilities.get("supports_inpainting", False) and not capabilities.get("supports_img2img", False):
            raise ValueError(f"Model {self.model_name} does not support image editing")
        
        # Load image
        image = self._load_image(image_path)
        
        # Load mask if provided
        mask = None
        if mask_path:
            mask = self._load_image(mask_path)
            task_type = ImageTaskType.INPAINTING
        else:
            task_type = ImageTaskType.IMAGE_TO_IMAGE
        
        # Update config
        if config is None:
            config = self.config.copy()
        config.task_type = task_type
        
        # Call specialized method if implemented
        if hasattr(self, '_edit_image_internal'):
            return self._edit_image_internal(image, prompt, mask, config, **kwargs)
        
        # Default implementation
        raise NotImplementedError(f"Image editing not implemented for {self.model_name}")
    
    def upscale_image(
        self,
        image_path: Union[str, Path],
        scale_factor: float = 2.0,
        config: Optional[ImageGenerationConfig] = None,
        **kwargs
    ) -> ImageGenerationResult:
        """
        Upscale an image using super-resolution.
        
        Args:
            image_path: Path to input image
            scale_factor: Scaling factor (e.g., 2.0 for 2x)
            config: Configuration for upscaling
            **kwargs: Additional arguments
            
        Returns:
            ImageGenerationResult containing upscaled image
            
        Raises:
            ValueError: If model is not loaded or doesn't support upscaling
            RuntimeError: If upscaling fails
        """
        if not self._is_loaded:
            raise ValueError(f"Model {self.model_name} is not loaded. Call load() first.")
        
        # Check if model supports upscaling
        capabilities = self.get_model_capabilities()
        if not capabilities.get("supports_upscaling", False):
            raise ValueError(f"Model {self.model_name} does not support upscaling")
        
        # Load image
        image = self._load_image(image_path)
        
        # Update config
        if config is None:
            config = self.config.copy()
        config.task_type = ImageTaskType.SUPER_RESOLUTION
        
        # Call specialized method if implemented
        if hasattr(self, '_upscale_image_internal'):
            return self._upscale_image_internal(image, scale_factor, config, **kwargs)
        
        # Default implementation
        raise NotImplementedError(f"Image upscaling not implemented for {self.model_name}")
    
    def generate_variations(
        self,
        image_path: Union[str, Path],
        num_variations: int = 4,
        config: Optional[ImageGenerationConfig] = None,
        **kwargs
    ) -> ImageGenerationResult:
        """
        Generate variations of an existing image.
        
        Args:
            image_path: Path to input image
            num_variations: Number of variations to generate
            config: Configuration for variation generation
            **kwargs: Additional arguments
            
        Returns:
            ImageGenerationResult containing image variations
            
        Raises:
            ValueError: If model is not loaded or doesn't support variations
            RuntimeError: If generation fails
        """
        if not self._is_loaded:
            raise ValueError(f"Model {self.model_name} is not loaded. Call load() first.")
        
        # Check if model supports variations
        capabilities = self.get_model_capabilities()
        if not capabilities.get("supports_variations", False):
            raise ValueError(f"Model {self.model_name} does not support image variations")
        
        # Load image
        image = self._load_image(image_path)
        
        # Update config
        if config is None:
            config = self.config.copy()
        config.task_type = ImageTaskType.IMAGE_VARIATION
        
        # Call specialized method if implemented
        if hasattr(self, '_generate_variations_internal'):
            return self._generate_variations_internal(image, num_variations, config, **kwargs)
        
        # Default implementation
        raise NotImplementedError(f"Image variation generation not implemented for {self.model_name}")
    
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
            "total_images_generated": self._total_images_generated,
            "average_processing_time_ms": self.get_average_processing_time(),
        }
    
    def get_model_capabilities(self) -> Dict[str, Any]:
        """
        Get capabilities of the model.
        
        Returns:
            Dictionary with model capabilities
        """
        return {
            "supports_text_to_image": False,
            "supports_img2img": False,
            "supports_inpainting": False,
            "supports_upscaling": False,
            "supports_variations": False,
            "supports_controlnet": False,
            "max_image_size": (512, 512),
            "min_image_size": (64, 64),
            "aspect_ratios": ["1:1"],
            "multilingual_prompts": False,
            "model_parameters": 0,
            "memory_requirements_mb": 0,
            "recommended_batch_size": 1,
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
    
    def _prepare_prompt(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        style: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prepare prompt for generation.
        
        Args:
            prompt: Main text prompt
            negative_prompt: Optional negative prompt
            style: Optional style preset
            **kwargs: Additional prompt parameters
            
        Returns:
            Dictionary with prepared prompts
        """
        prepared = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "style": style,
        }
        
        # Apply style templates if style is specified
        if style and hasattr(self, '_apply_style_template'):
            try:
                prepared = self._apply_style_template(prepared, style)
            except Exception as e:
                logger.warning(f"Failed to apply style template {style}: {e}")
        
        # Update with kwargs
        prepared.update(kwargs)
        
        return prepared
    
    def _perform_safety_check(self, images: List[Image.Image]) -> Tuple[List[Image.Image], List[bool]]:
        """
        Perform safety check on generated images.
        
        Args:
            images: List of generated images
            
        Returns:
            Tuple of (safe_images, safety_scores)
        """
        if not self.config.safety_check:
            return images, [True] * len(images)
        
        try:
            # Simple safety check based on image content analysis
            # In production, you would use a proper safety checker
            safety_scores = []
            safe_images = []
            
            for img in images:
                # Convert to numpy for analysis
                img_np = np.array(img)
                
                # Simple checks (can be expanded)
                is_safe = True
                
                # Check image statistics
                if img_np.mean() < 10 or img_np.mean() > 245:
                    # Very dark or very bright images might be problematic
                    is_safe = False
                
                # Add more sophisticated checks here
                
                if is_safe:
                    safe_images.append(img)
                safety_scores.append(is_safe)
            
            if len(safe_images) < len(images):
                logger.warning(f"Filtered {len(images) - len(safe_images)} images for safety")
            
            return safe_images, safety_scores
            
        except Exception as e:
            logger.error(f"Safety check failed: {e}")
            return images, [True] * len(images)
    
    def get_average_processing_time(self) -> float:
        """
        Get average processing time for all generations.
        
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
        self._total_images_generated = 0
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