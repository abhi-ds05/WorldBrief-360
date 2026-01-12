"""
Image-to-Text models module.
Provides unified interface for various vision-language models including:
- BLIP (Bootstrapping Language-Image Pre-training)
- Donut (Document Understanding Transformer)
- Qwen-VL (Qwen Vision Language)
- LLaVA (Large Language and Vision Assistant)
- Custom vision-language models
"""

from .base import BaseImageToTextModel, ImageTextResult, ImageCaptionConfig
from .blip import BLIPModel
from .donut import DonutModel
from .qwen_vl import QwenVLModel
from .llava import LLaVAModel

__all__ = [
    'BaseImageToTextModel',
    'ImageTextResult',
    'ImageCaptionConfig',
    'BLIPModel',
    'DonutModel',
    'QwenVLModel',
    'LLaVAModel',
]


def get_image_to_text_model(
    model_name: str,
    model_type: str = "blip",
    **kwargs
):
    """
    Factory function to get an image-to-text model instance.
    
    Args:
        model_name: Name of the model or path to local model
        model_type: Type of vision-language model ("blip", "donut", "qwen_vl", "llava")
        **kwargs: Additional arguments for model initialization
        
    Returns:
        An instance of the image-to-text model
        
    Raises:
        ValueError: If model_type is not supported
    """
    model_type = model_type.lower()
    
    if model_type == "blip":
        return BLIPModel(model_name=model_name, **kwargs)
    elif model_type == "donut":
        return DonutModel(model_name=model_name, **kwargs)
    elif model_type == "qwen_vl" or model_type == "qwen-vl":
        return QwenVLModel(model_name=model_name, **kwargs)
    elif model_type == "llava":
        return LLaVAModel(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported image-to-text model type: {model_type}")


# Default image-to-text models configuration
DEFAULT_MODELS = {
    # BLIP models
    "Salesforce/blip-image-captioning-base": {
        "type": "blip",
        "description": "BLIP base model for image captioning",
        "max_tokens": 32,
        "image_size": 384,
        "tasks": ["captioning", "vqa"],
    },
    "Salesforce/blip-image-captioning-large": {
        "type": "blip",
        "description": "BLIP large model for detailed image captioning",
        "max_tokens": 64,
        "image_size": 384,
        "tasks": ["captioning", "vqa"],
    },
    "Salesforce/blip-vqa-base": {
        "type": "blip",
        "description": "BLIP base model for visual question answering",
        "max_tokens": 32,
        "image_size": 384,
        "tasks": ["vqa"],
    },
    
    # Donut models
    "naver-clova-ix/donut-base": {
        "type": "donut",
        "description": "Donut base model for document understanding",
        "max_tokens": 768,
        "image_size": 2560,  # Can process high-resolution documents
        "tasks": ["document_understanding", "ocr"],
    },
    "naver-clova-ix/donut-base-finetuned-cord-v2": {
        "type": "donut",
        "description": "Donut fine-tuned for CORD document parsing",
        "max_tokens": 768,
        "image_size": 2560,
        "tasks": ["document_parsing", "receipt_understanding"],
    },
    
    # Qwen-VL models
    "Qwen/Qwen-VL": {
        "type": "qwen_vl",
        "description": "Qwen Vision-Language model with strong multimodal capabilities",
        "max_tokens": 2048,
        "image_size": 448,
        "tasks": ["captioning", "vqa", "referring_expression", "visual_reasoning"],
    },
    "Qwen/Qwen-VL-Chat": {
        "type": "qwen_vl",
        "description": "Qwen VL Chat model for conversational multimodal interactions",
        "max_tokens": 2048,
        "image_size": 448,
        "tasks": ["conversation", "vqa", "detailed_description"],
    },
    
    # LLaVA models
    "llava-hf/llava-1.5-7b-hf": {
        "type": "llava",
        "description": "LLaVA 1.5 7B model for visual instruction following",
        "max_tokens": 2048,
        "image_size": 336,
        "tasks": ["visual_instruction", "vqa", "detailed_description"],
    },
    "llava-hf/llava-1.5-13b-hf": {
        "type": "llava",
        "description": "LLaVA 1.5 13B model with enhanced capabilities",
        "max_tokens": 2048,
        "image_size": 336,
        "tasks": ["visual_instruction", "vqa", "complex_reasoning"],
    },
}


def list_available_models() -> dict:
    """
    List all available image-to-text models and their configurations.
    
    Returns:
        Dictionary of available models
    """
    return DEFAULT_MODELS.copy()


def get_model_info(model_name: str) -> dict:
    """
    Get information about a specific image-to-text model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model configuration dictionary
        
    Raises:
        KeyError: If model is not found
    """
    if model_name not in DEFAULT_MODELS:
        raise KeyError(f"Model not found: {model_name}")
    
    return DEFAULT_MODELS[model_name].copy()


def get_models_by_task(task: str) -> list:
    """
    Get models that support a specific task.
    
    Args:
        task: Task name (e.g., "captioning", "vqa", "document_understanding")
        
    Returns:
        List of model names that support the task
    """
    models = []
    for model_name, info in DEFAULT_MODELS.items():
        if task in info.get("tasks", []):
            models.append(model_name)
    return models


# Helper function for common tasks
def create_captioning_model(**kwargs):
    """
    Create a model optimized for image captioning.
    
    Args:
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Image-to-text model instance
    """
    # Prefer BLIP for captioning if not specified
    if 'model_name' not in kwargs:
        kwargs['model_name'] = 'Salesforce/blip-image-captioning-large'
    if 'model_type' not in kwargs:
        kwargs['model_type'] = 'blip'
    
    return get_image_to_text_model(**kwargs)


def create_vqa_model(**kwargs):
    """
    Create a model optimized for visual question answering.
    
    Args:
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Image-to-text model instance
    """
    # Prefer Qwen-VL or LLaVA for VQA if not specified
    if 'model_name' not in kwargs:
        kwargs['model_name'] = 'Qwen/Qwen-VL-Chat'
    if 'model_type' not in kwargs:
        kwargs['model_type'] = 'qwen_vl'
    
    return get_image_to_text_model(**kwargs)


def create_document_model(**kwargs):
    """
    Create a model optimized for document understanding.
    
    Args:
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Image-to-text model instance
    """
    # Prefer Donut for document understanding
    if 'model_name' not in kwargs:
        kwargs['model_name'] = 'naver-clova-ix/donut-base'
    if 'model_type' not in kwargs:
        kwargs['model_type'] = 'donut'
    
    return get_image_to_text_model(**kwargs)