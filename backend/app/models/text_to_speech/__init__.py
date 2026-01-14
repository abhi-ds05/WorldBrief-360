"""
Text-to-Speech Models Module

This module provides implementations for various TTS models:
- Bark TTS (from Suno)
- XTTS (from Coqui)
- Coqui TTS (various models)

Each model implements the BaseTTSModel interface for consistent usage.
"""

from .base import (
    BaseTTSModel,
    TTSConfig,
    TTSResult,
    VoiceProfile,
    VoiceGender,
    VoiceStyle,
    AudioFormat,
    TTSModelError,
    ModelNotInitializedError,
    VoiceNotFoundError,
    InvalidConfigError,
    TextTooLongError,
    create_tts_model,
)

from .bark import BarkTTSModel
from .xtts import XTTSModel
from .coqui import CoquiTTSModel

# Factory function exports
def get_tts_model(
    model_type: str,
    model_name: str = None,
    model_version: str = "1.0",
    **kwargs
) -> BaseTTSModel:
    """
    Get a TTS model instance with automatic initialization.
    
    This is a convenience wrapper around create_tts_model that
    optionally initializes the model.
    
    Args:
        model_type: Type of TTS model ('bark', 'xtts', 'coqui')
        model_name: Name of the specific model (optional)
        model_version: Version of the model
        **kwargs: Passed to create_tts_model and initialize()
        
    Returns:
        BaseTTSModel: Initialized TTS model instance
        
    Example:
        >>> model = get_tts_model('bark', device='cuda')
        >>> result = model.generate("Hello world", config)
    """
    # Set default model names based on type
    if model_name is None:
        model_names = {
            'bark': 'suno/bark',
            'xtts': 'tts_models/multilingual/multi-dataset/xtts_v2',
            'coqui': 'tts_models/en/ljspeech/tacotron2-DDC'
        }
        model_name = model_names.get(model_type.lower(), model_type)
    
    # Create the model
    model = create_tts_model(
        model_type=model_type,
        model_name=model_name,
        model_version=model_version,
        **{k: v for k, v in kwargs.items() if k not in ['device', 'init_device']}
    )
    
    # Auto-initialize if device is provided
    device = kwargs.get('device', 'cpu')
    init_device = kwargs.get('init_device', device)
    
    if init_device:
        model.initialize(device=init_device, **kwargs)
    
    return model


# Available model types
AVAILABLE_MODELS = {
    'bark': {
        'description': 'Bark TTS from Suno - Highly realistic, multilingual, with speaker prompts',
        'class': BarkTTSModel,
        'default_model': 'suno/bark',
        'features': ['multilingual', 'speaker prompts', 'music generation', 'sound effects']
    },
    'xtts': {
        'description': 'XTTS from Coqui - High-quality, multilingual, voice cloning',
        'class': XTTSModel,
        'default_model': 'tts_models/multilingual/multi-dataset/xtts_v2',
        'features': ['multilingual', 'voice cloning', 'high quality', 'cross-lingual']
    },
    'coqui': {
        'description': 'Coqui TTS - Various high-quality TTS models',
        'class': CoquiTTSModel,
        'default_model': 'tts_models/en/ljspeech/tacotron2-DDC',
        'features': ['high quality', 'multiple voices', 'customizable', 'real-time']
    }
}


def list_available_models() -> dict:
    """
    List all available TTS models and their capabilities.
    
    Returns:
        dict: Dictionary of available models with their descriptions
    """
    return {
        name: {
            'description': info['description'],
            'default_model': info['default_model'],
            'features': info['features'],
            'supported_languages': info['class'].get_supported_languages() if hasattr(info['class'], 'get_supported_languages') else []
        }
        for name, info in AVAILABLE_MODELS.items()
    }


# Constants for common voice profiles
COMMON_VOICES = {
    'en_male': VoiceProfile(
        name="English Male",
        id="en_male_001",
        gender=VoiceGender.MALE,
        language="en",
        accent="us",
        style=VoiceStyle.NEUTRAL,
        sample_rate=24000,
        description="Standard American English male voice"
    ),
    'en_female': VoiceProfile(
        name="English Female",
        id="en_female_001",
        gender=VoiceGender.FEMALE,
        language="en",
        accent="us",
        style=VoiceStyle.NEUTRAL,
        sample_rate=24000,
        description="Standard American English female voice"
    ),
    'es_male': VoiceProfile(
        name="Spanish Male",
        id="es_male_001",
        gender=VoiceGender.MALE,
        language="es",
        accent="es",
        style=VoiceStyle.NEUTRAL,
        sample_rate=24000,
        description="Standard Spanish male voice"
    ),
    'fr_female': VoiceProfile(
        name="French Female",
        id="fr_female_001",
        gender=VoiceGender.FEMALE,
        language="fr",
        accent="fr",
        style=VoiceStyle.NEUTRAL,
        sample_rate=24000,
        description="Standard French female voice"
    ),
}


def get_voice_profile(voice_id: str) -> VoiceProfile:
    """
    Get a predefined voice profile by ID.
    
    Args:
        voice_id: Voice identifier (e.g., 'en_male', 'en_female')
        
    Returns:
        VoiceProfile: The voice profile
        
    Raises:
        KeyError: If voice_id is not found in COMMON_VOICES
    """
    if voice_id not in COMMON_VOICES:
        available = list(COMMON_VOICES.keys())
        raise KeyError(f"Voice '{voice_id}' not found. Available voices: {available}")
    
    return COMMON_VOICES[voice_id]


# Configuration presets
PRESET_CONFIGS = {
    'news_anchor': TTSConfig(
        voice_profile=COMMON_VOICES['en_male'],
        speed=1.1,
        pitch=1.0,
        volume=1.0,
        audio_format=AudioFormat.MP3,
        style=VoiceStyle.FORMAL
    ),
    'casual_chat': TTSConfig(
        voice_profile=COMMON_VOICES['en_female'],
        speed=1.0,
        pitch=1.05,
        volume=0.9,
        audio_format=AudioFormat.MP3,
        style=VoiceStyle.CASUAL
    ),
    'audiobook': TTSConfig(
        voice_profile=COMMON_VOICES['en_male'],
        speed=0.9,
        pitch=0.95,
        volume=1.0,
        audio_format=AudioFormat.MP3,
        style=VoiceStyle.CALM,
        add_silence_ms=500
    ),
    'excited_announcement': TTSConfig(
        voice_profile=COMMON_VOICES['en_female'],
        speed=1.3,
        pitch=1.1,
        volume=1.1,
        audio_format=AudioFormat.MP3,
        style=VoiceStyle.EXCITED
    ),
}


def get_preset_config(preset_name: str, **overrides) -> TTSConfig:
    """
    Get a preset TTS configuration.
    
    Args:
        preset_name: Name of the preset ('news_anchor', 'casual_chat', etc.)
        **overrides: Configuration values to override
        
    Returns:
        TTSConfig: The preset configuration with overrides applied
    """
    if preset_name not in PRESET_CONFIGS:
        available = list(PRESET_CONFIGS.keys())
        raise KeyError(f"Preset '{preset_name}' not found. Available presets: {available}")
    
    config = PRESET_CONFIGS[preset_name]
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


# Version information
__version__ = "1.0.0"
__author__ = "WorldBrief 360 Team"
__description__ = "Text-to-Speech module for WorldBrief 360 with multiple model backends"

# Module exports
__all__ = [
    # Base classes and types
    'BaseTTSModel',
    'TTSConfig',
    'TTSResult',
    'VoiceProfile',
    
    # Enums
    'VoiceGender',
    'VoiceStyle',
    'AudioFormat',
    
    # Exceptions
    'TTSModelError',
    'ModelNotInitializedError',
    'VoiceNotFoundError',
    'InvalidConfigError',
    'TextTooLongError',
    
    # Model implementations
    'BarkTTSModel',
    'XTTSModel',
    'CoquiTTSModel',
    
    # Factory functions
    'create_tts_model',
    'get_tts_model',
    
    # Utility functions
    'list_available_models',
    'get_voice_profile',
    'get_preset_config',
    
    # Constants
    'AVAILABLE_MODELS',
    'COMMON_VOICES',
    'PRESET_CONFIGS',
]