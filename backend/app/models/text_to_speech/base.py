"""
Base Text-to-Speech Model Interface

Defines the abstract base class and common types for all TTS model implementations.
"""
from abc import ABC, abstractmethod
from typing import Optional, BinaryIO, Union, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class VoiceGender(Enum):
    """Gender of the voice"""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class VoiceStyle(Enum):
    """Style or emotion of the voice"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    EXCITED = "excited"
    CALM = "calm"
    FORMAL = "formal"
    CASUAL = "casual"


class AudioFormat(Enum):
    """Supported audio formats"""
    MP3 = "mp3"
    WAV = "wav"
    FLAC = "flac"
    OGG = "ogg"
    WEBM = "webm"


@dataclass
class VoiceProfile:
    """Profile of a specific voice"""
    name: str
    id: str
    gender: VoiceGender
    language: str  # ISO 639-1 code
    accent: Optional[str] = None
    style: VoiceStyle = VoiceStyle.NEUTRAL
    sample_rate: int = 24000  # Default sample rate in Hz
    bit_depth: int = 16  # Default bit depth
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class TTSConfig:
    """Configuration for TTS generation"""
    voice_profile: VoiceProfile
    speed: float = 1.0  # Speech rate (0.5 = half speed, 2.0 = double speed)
    pitch: float = 1.0  # Pitch adjustment
    volume: float = 1.0  # Volume adjustment
    audio_format: AudioFormat = AudioFormat.MP3
    sample_rate: Optional[int] = None  # If None, use voice profile default
    bit_depth: Optional[int] = None  # If None, use voice profile default
    add_silence_ms: int = 0  # Add silence at the beginning/end in milliseconds
    language: Optional[str] = None  # Override language if supported


@dataclass
class TTSResult:
    """Result of TTS generation"""
    audio_data: bytes
    config: TTSConfig
    duration_ms: int  # Duration in milliseconds
    text_length: int  # Length of input text
    model_name: str
    model_version: str
    generation_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTTSModel(ABC):
    """
    Abstract base class for all TTS model implementations.
    
    All TTS models (Bark, XTTS, Coqui) should inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, model_name: str, model_version: str = "1.0"):
        """
        Initialize the TTS model.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
        """
        self.model_name = model_name
        self.model_version = model_version
        self.initialized = False
        self.supported_languages: List[str] = []
        self.available_voices: List[VoiceProfile] = []
        
    @abstractmethod
    def initialize(self, device: str = "cpu", **kwargs) -> bool:
        """
        Initialize the model with the given parameters.
        
        Args:
            device: Device to run the model on ('cpu', 'cuda', 'mps')
            **kwargs: Additional model-specific initialization parameters
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def generate(
        self, 
        text: str, 
        config: TTSConfig,
        **kwargs
    ) -> TTSResult:
        """
        Generate speech from text.
        
        Args:
            text: Input text to convert to speech
            config: TTS configuration
            **kwargs: Additional model-specific parameters
            
        Returns:
            TTSResult: The generated audio and metadata
            
        Raises:
            ValueError: If text is empty or invalid
            ModelNotInitializedError: If model is not initialized
            VoiceNotFoundError: If requested voice is not available
        """
        pass
    
    @abstractmethod
    def stream_generate(
        self, 
        text: str, 
        config: TTSConfig,
        chunk_size: int = 1024,
        **kwargs
    ) -> TTSResult:
        """
        Generate speech and stream the audio in chunks.
        
        Args:
            text: Input text to convert to speech
            config: TTS configuration
            chunk_size: Size of audio chunks to yield
            **kwargs: Additional model-specific parameters
            
        Returns:
            TTSResult: The generated audio and metadata
        """
        pass
    
    @abstractmethod
    def get_available_voices(self) -> List[VoiceProfile]:
        """
        Get list of available voices for this model.
        
        Returns:
            List[VoiceProfile]: Available voice profiles
        """
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages (ISO 639-1 codes).
        
        Returns:
            List[str]: Supported language codes
        """
        pass
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess input text before TTS generation.
        
        Args:
            text: Raw input text
            
        Returns:
            str: Preprocessed text
        """
        # Default preprocessing: trim whitespace
        text = text.strip()
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        return text
    
    def validate_config(self, config: TTSConfig) -> Tuple[bool, Optional[str]]:
        """
        Validate TTS configuration.
        
        Args:
            config: TTS configuration to validate
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        # Check if voice is available
        available_voices = self.get_available_voices()
        voice_ids = [v.id for v in available_voices]
        
        if config.voice_profile.id not in voice_ids:
            return False, f"Voice '{config.voice_profile.id}' not available"
        
        # Check language support
        supported_languages = self.get_supported_languages()
        if config.voice_profile.language not in supported_languages:
            return False, f"Language '{config.voice_profile.language}' not supported"
        
        # Validate speed parameter
        if config.speed <= 0 or config.speed > 3.0:
            return False, f"Speed must be between 0.1 and 3.0, got {config.speed}"
        
        # Validate pitch parameter
        if config.pitch <= 0 or config.pitch > 2.0:
            return False, f"Pitch must be between 0.5 and 2.0, got {config.pitch}"
        
        return True, None
    
    def estimate_duration(self, text: str, config: TTSConfig) -> int:
        """
        Estimate the duration of the generated speech in milliseconds.
        
        Args:
            text: Input text
            config: TTS configuration
            
        Returns:
            int: Estimated duration in milliseconds
        """
        # Basic estimation: ~150 words per minute, adjusted by speed
        words_per_minute = 150 * config.speed
        word_count = len(text.split())
        
        if word_count == 0:
            return 0
        
        # Calculate duration in milliseconds
        duration_minutes = word_count / words_per_minute
        duration_ms = int(duration_minutes * 60 * 1000)
        
        # Add configured silence
        duration_ms += config.add_silence_ms
        
        return duration_ms
    
    def cleanup(self) -> None:
        """
        Clean up model resources.
        
        This method should be called when the model is no longer needed
        to free up memory and resources.
        """
        self.initialized = False
        self.available_voices.clear()
        self.supported_languages.clear()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "initialized": self.initialized,
            "supported_languages": self.get_supported_languages(),
            "available_voices": [v.name for v in self.get_available_voices()],
            "class_name": self.__class__.__name__,
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.cleanup()


class TTSModelError(Exception):
    """Base exception for TTS model errors."""
    pass


class ModelNotInitializedError(TTSModelError):
    """Raised when trying to use a model that hasn't been initialized."""
    pass


class VoiceNotFoundError(TTSModelError):
    """Raised when a requested voice is not available."""
    pass


class InvalidConfigError(TTSModelError):
    """Raised when TTS configuration is invalid."""
    pass


class TextTooLongError(TTSModelError):
    """Raised when input text exceeds model limits."""
    pass


# Factory function for creating TTS models
def create_tts_model(
    model_type: str,
    model_name: str,
    model_version: str = "1.0",
    **init_kwargs
) -> BaseTTSModel:
    """
    Factory function to create TTS model instances.
    
    Args:
        model_type: Type of model ('bark', 'xtts', 'coqui')
        model_name: Name of the model
        model_version: Version of the model
        **init_kwargs: Additional initialization parameters
        
    Returns:
        BaseTTSModel: Initialized TTS model instance
        
    Raises:
        ValueError: If model_type is not supported
    """
    model_type = model_type.lower()
    
    if model_type == "bark":
        from .bark import BarkTTSModel
        return BarkTTSModel(model_name, model_version, **init_kwargs)
    
    elif model_type == "xtts":
        from .xtts import XTTSModel
        return XTTSModel(model_name, model_version, **init_kwargs)
    
    elif model_type == "coqui":
        from .coqui import CoquiTTSModel
        return CoquiTTSModel(model_name, model_version, **init_kwargs)
    
    else:
        raise ValueError(f"Unsupported TTS model type: {model_type}. "
                         f"Supported types: 'bark', 'xtts', 'coqui'")