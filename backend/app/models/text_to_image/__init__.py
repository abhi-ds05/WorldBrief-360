"""
Text-to-image model implementations for WorldBrief360.

This package contains models for generating images from text descriptions,
including diffusion models, GANs, and other image generation architectures.
"""

from pathlib import Path
from typing import Any, List, List, Optional, Union

from joblib import Logger
from sympy import Dict
from .base import (
    BaseTextToImageModel, ImageGenerationResult, ImageGenerationConfig,
    ImageTaskType, ImageGenerationParameters, ImageFormat
)

try:
    from .sdxl import SDXLModel
    HAS_SDXL = True
except ImportError:
    HAS_SDXL = False
    Logger.warning("SDXL dependencies not available")

try:
    from .flux import FluxModel
    HAS_FLUX = True
except ImportError:
    HAS_FLUX = False
    logging.Logger.warning("Flux dependencies not available")

try:
    from .stable_diffusion import StableDiffusionModel
    HAS_STABLE_DIFFUSION = True
except ImportError:
    HAS_STABLE_DIFFUSION = False
    logging.Logger.warning("Stable Diffusion dependencies not available")

try:
    from .dall_e import DalleModel
    HAS_DALLE = True
except ImportError:
    HAS_DALLE = False
    logging.Logger.warning("DALL-E dependencies not available")

from ..factory import TextToImageFactory # pyright: ignore[reportMissingImports]

# Re-export main classes
__all__ = [
    # Base classes
    "BaseTextToImageModel",
    "ImageGenerationResult",
    "ImageGenerationConfig",
    "ImageTaskType",
    "ImageGenerationParameters",
    "ImageFormat",
    
    # Model implementations (conditionally exported)
    "SDXLModel",
    "FluxModel",
    "StableDiffusionModel",
    "DalleModel",
    
    # Factory
    "TextToImageFactory",
]

# Initialize available models list
_AVAILABLE_MODELS = []

if HAS_SDXL:
    _AVAILABLE_MODELS.append("sdxl")
if HAS_FLUX:
    _AVAILABLE_MODELS.append("flux")
if HAS_STABLE_DIFFUSION:
    _AVAILABLE_MODELS.append("stable_diffusion")
if HAS_DALLE:
    _AVAILABLE_MODELS.append("dalle")


def get_text_to_image_model(
    model_name: str,
    config: Optional[ImageGenerationConfig] = None,
    **kwargs
) -> BaseTextToImageModel:
    """
    Factory function to get a text-to-image model instance.
    
    Args:
        model_name: Name of the model or model identifier
        config: Configuration for image generation
        **kwargs: Additional arguments to pass to model constructor
        
    Returns:
        Instance of the text-to-image model
        
    Raises:
        ValueError: If model_name is not supported
    """
    return TextToImageFactory.get_model(model_name, config, **kwargs)


def list_available_models() -> List[str]:
    """
    List all available text-to-image models.
    
    Returns:
        List of model identifiers
    """
    return _AVAILABLE_MODELS.copy()


def get_model_capabilities(model_name: str) -> Dict[str, Any]:
    """
    Get capabilities of a specific text-to-image model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with model capabilities
        
    Raises:
        ValueError: If model_name is not supported
    """
    try:
        model = get_text_to_image_model(model_name)
        return model.get_model_capabilities()
    except Exception as e:
        raise ValueError(f"Failed to get capabilities for model {model_name}: {e}")


def generate_image(
    model_name: str,
    prompt: str,
    config: Optional[ImageGenerationConfig] = None,
    **kwargs
) -> ImageGenerationResult:
    """
    Convenience function to generate an image with a model.
    
    Args:
        model_name: Name of the model
        prompt: Text prompt to generate from
        config: Configuration for image generation
        **kwargs: Additional generation arguments
        
    Returns:
        ImageGenerationResult with generated image
        
    Raises:
        ValueError: If model_name is not supported or generation fails
    """
    try:
        model = get_text_to_image_model(model_name, config, **kwargs)
        
        # Load model if not loaded
        if not getattr(model, '_is_loaded', False):
            model.load()
        
        # Generate image
        return model.generate_image(prompt, **kwargs)
        
    except Exception as e:
        raise ValueError(f"Failed to generate image with model {model_name}: {e}")


def batch_generate(
    model_name: str,
    prompts: List[str],
    config: Optional[ImageGenerationConfig] = None,
    batch_size: int = 1,
    **kwargs
) -> List[ImageGenerationResult]:
    """
    Generate images for multiple prompts in batch.
    
    Args:
        model_name: Name of the model
        prompts: List of text prompts
        config: Configuration for image generation
        batch_size: Batch size for processing
        **kwargs: Additional generation arguments
        
    Returns:
        List of ImageGenerationResult objects
        
    Raises:
        ValueError: If model_name is not supported or generation fails
    """
    try:
        model = get_text_to_image_model(model_name, config, **kwargs)
        
        # Load model if not loaded
        if not getattr(model, '_is_loaded', False):
            model.load()
        
        results = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            for prompt in batch_prompts:
                try:
                    result = model.generate_image(prompt, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to generate for prompt {i}: {e}")
                    # Add error result
                    results.append(ImageGenerationResult(
                        images=[],
                        model_name=model_name,
                        processing_time_ms=0,
                        metadata={"error": str(e)}
                    ))
            
            # Small delay between batches
            if i + batch_size < len(prompts):
                import time
                time.sleep(0.5)
        
        return results
        
    except Exception as e:
        raise ValueError(f"Failed to batch generate with model {model_name}: {e}")


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a text-to-image model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with model information
        
    Raises:
        ValueError: If model_name is not supported
    """
    try:
        model = get_text_to_image_model(model_name)
        return model.get_model_info()
    except Exception as e:
        raise ValueError(f"Failed to get info for model {model_name}: {e}")


def edit_image(
    model_name: str,
    image_path: Union[str, Path],
    prompt: str,
    mask_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> ImageGenerationResult:
    """
    Edit an existing image using inpainting or image-to-image.
    
    Args:
        model_name: Name of the model
        image_path: Path to input image
        prompt: Text prompt for editing
        mask_path: Optional mask for inpainting
        **kwargs: Additional arguments
        
    Returns:
        ImageGenerationResult with edited image
        
    Raises:
        ValueError: If model_name is not supported or editing fails
    """
    try:
        model = get_text_to_image_model(model_name, **kwargs)
        
        # Load model if not loaded
        if not getattr(model, '_is_loaded', False):
            model.load()
        
        # Check if model supports editing
        capabilities = model.get_model_capabilities()
        if not capabilities.get("supports_inpainting", False) and not capabilities.get("supports_img2img", False):
            raise ValueError(f"Model {model_name} does not support image editing")
        
        # Perform editing
        return model.edit_image(image_path, prompt, mask_path, **kwargs)
        
    except Exception as e:
        raise ValueError(f"Failed to edit image with model {model_name}: {e}")


def upscale_image(
    model_name: str,
    image_path: Union[str, Path],
    scale_factor: float = 2.0,
    **kwargs
) -> ImageGenerationResult:
    """
    Upscale an image using super-resolution.
    
    Args:
        model_name: Name of the model
        image_path: Path to input image
        scale_factor: Scaling factor (e.g., 2.0 for 2x)
        **kwargs: Additional arguments
        
    Returns:
        ImageGenerationResult with upscaled image
        
    Raises:
        ValueError: If model_name is not supported or upscaling fails
    """
    try:
        model = get_text_to_image_model(model_name, **kwargs)
        
        # Load model if not loaded
        if not getattr(model, '_is_loaded', False):
            model.load()
        
        # Check if model supports upscaling
        capabilities = model.get_model_capabilities()
        if not capabilities.get("supports_upscaling", False):
            raise ValueError(f"Model {model_name} does not support upscaling")
        
        # Perform upscaling
        return model.upscale_image(image_path, scale_factor, **kwargs)
        
    except Exception as e:
        raise ValueError(f"Failed to upscale image with model {model_name}: {e}")


# Package metadata
__version__ = "0.1.0"
__author__ = "WorldBrief360 Team"
__description__ = "Text-to-image models for WorldBrief360 AI platform"


# Initialize logging
import logging
logger = logging.getLogger(__name__)