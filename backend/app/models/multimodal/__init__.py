"""
Multimodal model implementations for WorldBrief360.

This package contains multimodal models that can process multiple input types
(text, images, audio, etc.) together.
"""

from ast import List
from typing import Dict, Any
from .base import BaseMultimodalModel, MultimodalResult, MultimodalTaskType
from .clip import CLIPModel
from .imagebind import ImageBindModel
from .qwen_vl import QwenVLModel # type: ignore

# Re-export main classes
__all__ = [
    # Base classes
    "BaseMultimodalModel",
    "MultimodalResult", 
    "MultimodalTaskType",
    
    # Model implementations
    "CLIPModel",
    "ImageBindModel",
    "QwenVLModel",
]

# Model factory registry
_MODEL_REGISTRY = {
    "clip": CLIPModel,
    "imagebind": ImageBindModel,
    "qwen_vl": QwenVLModel,
    "qwen2-vl": QwenVLModel,
}


def get_multimodal_model(
    model_name: str,
    **kwargs
):
    """
    Factory function to get a multimodal model instance.
    
    Args:
        model_name: Name of the model or model identifier
        **kwargs: Additional arguments to pass to model constructor
        
    Returns:
        Instance of the multimodal model
        
    Raises:
        ValueError: If model_name is not supported
    """
    model_name_lower = model_name.lower()
    
    # Check for exact matches
    for key, model_class in _MODEL_REGISTRY.items():
        if key in model_name_lower or model_name_lower in key:
            return model_class(model_name=model_name, **kwargs)
    
    # Try to infer from model name
    if "clip" in model_name_lower:
        return CLIPModel(model_name=model_name, **kwargs)
    elif "imagebind" in model_name_lower:
        return ImageBindModel(model_name=model_name, **kwargs)
    elif "qwen" in model_name_lower and ("vl" in model_name_lower or "vision" in model_name_lower):
        return QwenVLModel(model_name=model_name, **kwargs)
    
    raise ValueError(f"Unsupported multimodal model: {model_name}. "
                     f"Available models: {list(_MODEL_REGISTRY.keys())}")


def list_available_models() -> List[str]:
    """
    List all available multimodal models.
    
    Returns:
        List of model identifiers
    """
    return list(_MODEL_REGISTRY.keys())


def get_model_capabilities(model_name: str) -> Dict[str, Any]:
    """
    Get capabilities of a specific multimodal model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with model capabilities
        
    Raises:
        ValueError: If model_name is not supported
    """
    try:
        model = get_multimodal_model(model_name)
        return model.get_model_capabilities()
    except Exception as e:
        raise ValueError(f"Failed to get capabilities for model {model_name}: {e}")


# Package metadata
__version__ = "0.1.0"
__author__ = "WorldBrief360 Team"
__description__ = "Multimodal models for WorldBrief360 AI platform"