"""
Bark TTS Model Implementation

Bark is a transformer-based text-to-speech model from Suno that can
generate highly realistic, multilingual speech with speaker prompts,
and can also generate music and sound effects.

Key features:
- Highly realistic speech generation
- Multilingual support
- Speaker prompts for voice control
- Can generate music and sound effects
- Zero-shot voice cloning (to some extent)
"""

import os
import io
import time
import tempfile
import warnings
from typing import List, Optional, Dict, Any, Generator, Union, Tuple
from pathlib import Path
import numpy as np
import torch
import torchaudio

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
)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class BarkTTSModel(BaseTTSModel):
    """
    Bark TTS model implementation.
    
    Bark is a transformer-based text-to-speech model that can generate
    highly realistic, multilingual speech with speaker prompts.
    """
    
    # Bark-specific voice presets
    BARK_VOICE_PRESETS = {
        # English voices
        "en_speaker_0": "v2/en_speaker_0",
        "en_speaker_1": "v2/en_speaker_1",
        "en_speaker_2": "v2/en_speaker_2",
        "en_speaker_3": "v2/en_speaker_3",
        "en_speaker_4": "v2/en_speaker_4",
        "en_speaker_5": "v2/en_speaker_5",
        "en_speaker_6": "v2/en_speaker_6",
        "en_speaker_7": "v2/en_speaker_7",
        "en_speaker_8": "v2/en_speaker_8",
        "en_speaker_9": "v2/en_speaker_9",
        
        # Non-English voices
        "es_speaker_0": "v2/es_speaker_0",
        "fr_speaker_0": "v2/fr_speaker_0",
        "de_speaker_0": "v2/de_speaker_0",
        "hi_speaker_0": "v2/hi_speaker_0",
        "it_speaker_0": "v2/it_speaker_0",
        "ja_speaker_0": "v2/ja_speaker_0",
        "ko_speaker_0": "v2/ko_speaker_0",
        "pl_speaker_0": "v2/pl_speaker_0",
        "pt_speaker_0": "v2/pt_speaker_0",
        "ru_speaker_0": "v2/ru_speaker_0",
        "tr_speaker_0": "v2/tr_speaker_0",
        "zh_speaker_0": "v2/zh_speaker_0",
        
        # Special voices
        "announcer": "v2/en_speaker_0",  # Clear announcer voice
        "narrator": "v2/en_speaker_6",   # Warm narrator voice
        "news": "v2/en_speaker_4",       # News anchor voice
    }
    
    # Language code mapping
    LANGUAGE_MAPPING = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "hi": "Hindi",
        "it": "Italian",
        "ja": "Japanese",
        "ko": "Korean",
        "pl": "Polish",
        "pt": "Portuguese",
        "ru": "Russian",
        "tr": "Turkish",
        "zh": "Chinese",
    }
    
    def __init__(self, model_name: str = "suno/bark", model_version: str = "1.0"):
        """
        Initialize the Bark TTS model.
        
        Args:
            model_name: Name/path of the Bark model (default: "suno/bark")
            model_version: Version of the model
        """
        super().__init__(model_name, model_version)
        self.model = None
        self.processor = None
        self.device = None
        self.sample_rate = 24000  # Bark's fixed sample rate
        
        # Set supported languages
        self.supported_languages = list(self.LANGUAGE_MAPPING.keys())
        
        # Initialize available voices
        self._setup_default_voices()
    
    def _setup_default_voices(self):
        """Setup default voice profiles for Bark."""
        self.available_voices = []
        
        # Create voice profiles for each Bark preset
        for voice_id, preset_path in self.BARK_VOICE_PRESETS.items():
            # Extract language from voice_id
            lang_code = voice_id.split("_")[0]
            if lang_code not in self.LANGUAGE_MAPPING:
                lang_code = "en"  # Default to English
            
            # Determine gender (simple heuristic)
            if "speaker_0" in voice_id or "speaker_2" in voice_id:
                gender = VoiceGender.FALE
            elif "speaker_1" in voice_id or "speaker_3" in voice_id:
                gender = VoiceGender.FEMALE
            else:
                gender = VoiceGender.MIXED
            
            # Determine style
            if "announcer" in voice_id:
                style = VoiceStyle.FORMAL
            elif "narrator" in voice_id:
                style = VoiceStyle.CALM
            elif "news" in voice_id:
                style = VoiceStyle.FORMAL
            else:
                style = VoiceStyle.NEUTRAL
            
            voice_profile = VoiceProfile(
                name=f"Bark {self.LANGUAGE_MAPPING.get(lang_code, 'Unknown')} Voice",
                id=voice_id,
                gender=gender,
                language=lang_code,
                accent=None,
                style=style,
                sample_rate=self.sample_rate,
                bit_depth=16,
                description=f"Bark TTS voice: {preset_path}",
                tags=["bark", lang_code, preset_path.split("/")[0]]
            )
            self.available_voices.append(voice_profile)
    
    def initialize(self, device: str = "cpu", **kwargs) -> bool:
        """
        Initialize the Bark model.
        
        Args:
            device: Device to run the model on ('cpu', 'cuda', 'mps')
            **kwargs: Additional parameters:
                - model_name: Override model name
                - processor_name: Override processor name
                - use_small: Use small model variant
                - use_fp16: Use half precision
                - cache_dir: Directory for model cache
                
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            import torch
            from transformers import AutoProcessor, BarkModel
            
            self.device = device
            
            # Check if CUDA is available
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                self.device = "cpu"
            
            # Check if MPS is available (Apple Silicon)
            if device == "mps" and not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                logger.warning("MPS requested but not available. Falling back to CPU.")
                self.device = "cpu"
            
            # Get model parameters
            model_name = kwargs.get("model_name", self.model_name)
            processor_name = kwargs.get("processor_name", model_name)
            use_small = kwargs.get("use_small", False)
            use_fp16 = kwargs.get("use_fp16", False)
            cache_dir = kwargs.get("cache_dir", None)
            
            # Use small model if requested
            if use_small:
                model_name = "suno/bark-small"
                processor_name = "suno/bark-small"
            
            logger.info(f"Loading Bark model: {model_name} on device: {self.device}")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                processor_name,
                cache_dir=cache_dir
            )
            
            # Load model
            self.model = BarkModel.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            
            # Move to device
            self.model = self.model.to(self.device)
            
            # Use half precision if requested and device supports it
            if use_fp16 and self.device in ["cuda", "mps"]:
                self.model = self.model.half()
            
            # Enable eval mode
            self.model.eval()
            
            self.initialized = True
            logger.info(f"Bark model initialized successfully on {self.device}")
            return True
            
        except ImportError as e:
            logger.error(f"Required packages not installed: {e}")
            logger.error("Please install: pip install transformers torch torchaudio")
            return False
            
        except Exception as e:
            logger.error(f"Failed to initialize Bark model: {e}")
            return False
    
    def generate(
        self, 
        text: str, 
        config: TTSConfig,
        **kwargs
    ) -> TTSResult:
        """
        Generate speech from text using Bark.
        
        Args:
            text: Input text to convert to speech
            config: TTS configuration
            **kwargs: Additional parameters:
                - voice_preset: Bark voice preset (overrides config.voice_profile.id)
                - temperature: Sampling temperature (default: 0.7)
                - semantic_temperature: Semantic temperature (default: 0.7)
                - waveform_temperature: Waveform temperature (default: 0.7)
                - min_eos_p: Minimum EOS probability (default: 0.05)
                - voice_diversity: Voice diversity (0.0 to 1.0, default: 0.7)
                - return_waveform: Return raw waveform instead of bytes (default: False)
                
        Returns:
            TTSResult: The generated audio and metadata
            
        Raises:
            ModelNotInitializedError: If model is not initialized
            InvalidConfigError: If configuration is invalid
            TextTooLongError: If text is too long for the model
        """
        if not self.initialized:
            raise ModelNotInitializedError("Bark model not initialized. Call initialize() first.")
        
        start_time = time.time()
        
        try:
            # Validate configuration
            is_valid, error_msg = self.validate_config(config)
            if not is_valid:
                raise InvalidConfigError(error_msg)
            
            # Preprocess text
            text = self.preprocess_text(text)
            
            # Check text length
            max_length = kwargs.get("max_length", 256)
            if len(text.split()) > max_length:
                raise TextTooLongError(
                    f"Text too long ({len(text.split())} words). "
                    f"Maximum allowed: {max_length} words."
                )
            
            # Get voice preset
            voice_preset = kwargs.get("voice_preset", config.voice_profile.id)
            
            # Map voice preset to Bark's format
            if voice_preset in self.BARK_VOICE_PRESETS:
                bark_preset = self.BARK_VOICE_PRESETS[voice_preset]
            else:
                # Use default if not found
                logger.warning(f"Voice preset '{voice_preset}' not found. Using default.")
                bark_preset = self.BARK_VOICE_PRESETS.get("en_speaker_0", "v2/en_speaker_0")
            
            # Prepare inputs
            inputs = self.processor(
                text,
                voice_preset=bark_preset,
                return_tensors="pt"
            ).to(self.device)
            
            # Generation parameters
            temperature = kwargs.get("temperature", 0.7)
            semantic_temperature = kwargs.get("semantic_temperature", 0.7)
            waveform_temperature = kwargs.get("waveform_temperature", 0.7)
            min_eos_p = kwargs.get("min_eos_p", 0.05)
            
            # Generate audio
            with torch.no_grad():
                audio_array = self.model.generate(
                    **inputs,
                    temperature=temperature,
                    semantic_temperature=semantic_temperature,
                    waveform_temperature=waveform_temperature,
                    min_eos_p=min_eos_p,
                    do_sample=True,
                )
            
            # Convert to numpy array
            audio_array = audio_array.cpu().numpy().squeeze()
            
            # Apply speed adjustment (simple resampling)
            if config.speed != 1.0:
                audio_array = self._adjust_speed(audio_array, config.speed)
            
            # Convert to target format
            audio_bytes = self._convert_audio_format(
                audio_array,
                config.audio_format,
                config.sample_rate or self.sample_rate
            )
            
            # Calculate duration
            duration_ms = int(len(audio_array) / self.sample_rate * 1000)
            
            # Add silence if requested
            if config.add_silence_ms > 0:
                audio_bytes = self._add_silence(audio_bytes, config.add_silence_ms)
            
            generation_time_ms = (time.time() - start_time) * 1000
            
            return TTSResult(
                audio_data=audio_bytes,
                config=config,
                duration_ms=duration_ms,
                text_length=len(text),
                model_name=self.model_name,
                model_version=self.model_version,
                generation_time_ms=generation_time_ms,
                metadata={
                    "voice_preset": bark_preset,
                    "temperature": temperature,
                    "semantic_temperature": semantic_temperature,
                    "waveform_temperature": waveform_temperature,
                    "device": str(self.device),
                    "sample_rate": self.sample_rate,
                    "bark_specific": {
                        "preset_used": bark_preset,
                        "original_text": text,
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating speech with Bark: {e}")
            raise TTSModelError(f"Failed to generate speech: {str(e)}")
    
    def stream_generate(
        self, 
        text: str, 
        config: TTSConfig,
        chunk_size: int = 1024,
        **kwargs
    ) -> Generator[bytes, None, TTSResult]:
        """
        Generate speech and stream the audio in chunks.
        
        Note: Bark doesn't natively support streaming, so we generate
        the full audio first and then stream it in chunks.
        
        Args:
            text: Input text to convert to speech
            config: TTS configuration
            chunk_size: Size of audio chunks to yield
            **kwargs: Additional parameters (same as generate())
            
        Yields:
            bytes: Audio chunks
        
        Returns:
            TTSResult: The complete result metadata
        """
        # Generate the full audio first
        result = self.generate(text, config, **kwargs)
        audio_data = result.audio_data
        
        # Stream in chunks
        for i in range(0, len(audio_data), chunk_size):
            yield audio_data[i:i + chunk_size]
        
        # Return the result
        return result
    
    def get_available_voices(self) -> List[VoiceProfile]:
        """
        Get list of available voices for Bark.
        
        Returns:
            List[VoiceProfile]: Available voice profiles
        """
        return self.available_voices.copy()
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages by Bark.
        
        Returns:
            List[str]: Supported language codes
        """
        return self.supported_languages.copy()
    
    def _adjust_speed(self, audio_array: np.ndarray, speed: float) -> np.ndarray:
        """
        Adjust audio speed by resampling.
        
        Args:
            audio_array: Input audio array
            speed: Speed factor (0.5 = half speed, 2.0 = double speed)
            
        Returns:
            np.ndarray: Speed-adjusted audio array
        """
        if speed == 1.0:
            return audio_array
        
        try:
            import librosa
            
            # Calculate new length
            new_length = int(len(audio_array) / speed)
            
            # Resample using librosa
            adjusted = librosa.effects.time_stretch(
                y=audio_array,
                rate=speed
            )
            
            # Ensure consistent length
            if len(adjusted) < new_length:
                # Pad with zeros
                adjusted = np.pad(adjusted, (0, new_length - len(adjusted)))
            elif len(adjusted) > new_length:
                # Truncate
                adjusted = adjusted[:new_length]
            
            return adjusted
            
        except ImportError:
            logger.warning("librosa not installed. Speed adjustment will be approximate.")
            # Simple approach: adjust sample rate
            from scipy import signal
            new_length = int(len(audio_array) / speed)
            adjusted = signal.resample(audio_array, new_length)
            return adjusted
    
    def _convert_audio_format(
        self, 
        audio_array: np.ndarray, 
        format: AudioFormat,
        sample_rate: int
    ) -> bytes:
        """
        Convert audio array to target format.
        
        Args:
            audio_array: Input audio array
            format: Target audio format
            sample_rate: Sample rate in Hz
            
        Returns:
            bytes: Audio data in target format
        """
        try:
            import io
            import soundfile as sf
            
            # Normalize audio
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            # Handle different formats
            buffer = io.BytesIO()
            
            if format == AudioFormat.WAV:
                sf.write(buffer, audio_array, sample_rate, format='WAV')
            elif format == AudioFormat.MP3:
                # MP3 requires pydub or similar
                try:
                    from pydub import AudioSegment
                    from pydub.utils import which
                    
                    # Convert numpy array to AudioSegment
                    audio_segment = AudioSegment(
                        audio_array.tobytes(),
                        frame_rate=sample_rate,
                        sample_width=audio_array.dtype.itemsize,
                        channels=1
                    )
                    
                    # Export as MP3
                    audio_segment.export(buffer, format="mp3")
                    
                except ImportError:
                    logger.warning("pydub not installed. Falling back to WAV for MP3 request.")
                    sf.write(buffer, audio_array, sample_rate, format='WAV')
            
            elif format == AudioFormat.FLAC:
                sf.write(buffer, audio_array, sample_rate, format='FLAC')
            
            elif format == AudioFormat.OGG:
                sf.write(buffer, audio_array, sample_rate, format='OGG')
            
            elif format == AudioFormat.WEBM:
                sf.write(buffer, audio_array, sample_rate, format='WEBM')
            
            else:
                # Default to WAV
                sf.write(buffer, audio_array, sample_rate, format='WAV')
            
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error converting audio format: {e}")
            # Fallback to raw WAV
            import wave
            import io
            
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes((audio_array * 32767).astype(np.int16).tobytes())
            
            return buffer.getvalue()
    
    def _add_silence(self, audio_bytes: bytes, silence_ms: int) -> bytes:
        """
        Add silence to the beginning of audio.
        
        Args:
            audio_bytes: Input audio bytes
            silence_ms: Silence duration in milliseconds
            
        Returns:
            bytes: Audio with added silence
        """
        try:
            from pydub import AudioSegment
            from pydub.utils import which
            
            # Load audio
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            
            # Create silence
            silence = AudioSegment.silent(duration=silence_ms)
            
            # Add silence to beginning
            audio_with_silence = silence + audio
            
            # Export back to bytes
            buffer = io.BytesIO()
            audio_with_silence.export(buffer, format="wav")
            
            return buffer.getvalue()
            
        except ImportError:
            logger.warning("pydub not installed. Cannot add silence.")
            return audio_bytes
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess input text for Bark.
        
        Bark can handle special tags for laughter, music, etc.
        This method adds appropriate tags if needed.
        
        Args:
            text: Raw input text
            
        Returns:
            str: Preprocessed text with Bark tags
        """
        text = super().preprocess_text(text)
        
        # Add laughter tags if not present and context suggests it
        laughter_triggers = ["haha", "lol", "lmao", "😂", "😄"]
        if any(trigger in text.lower() for trigger in laughter_triggers):
            if "[laughter]" not in text:
                text = text.replace("haha", "[laughter]")
                text = text.replace("lol", "[laughter]")
                text = text.replace("lmao", "[laughter]")
        
        # Add music tags if context suggests it
        music_triggers = ["♪", "🎵", "🎶", "music", "song"]
        if any(trigger in text.lower() for trigger in music_triggers):
            if "[music]" not in text:
                text = text.replace("♪", "[music]")
                text = text.replace("🎵", "[music]")
                text = text.replace("🎶", "[music]")
        
        return text
    
    def cleanup(self) -> None:
        """Clean up Bark model resources."""
        super().cleanup()
        
        if self.model is not None:
            # Move to CPU first to free GPU memory
            if self.device != "cpu":
                self.model = self.model.cpu()
            
            # Clear cache
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Delete model
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        logger.info("Bark model cleaned up")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the Bark model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        info = super().get_model_info()
        
        # Add Bark-specific info
        info.update({
            "device": str(self.device),
            "sample_rate": self.sample_rate,
            "voice_presets": list(self.BARK_VOICE_PRESETS.keys()),
            "model_architecture": "Bark (transformer-based)",
            "supports_music": True,
            "supports_sound_effects": True,
            "supports_speaker_prompts": True,
        })
        
        return info


# Example usage
if __name__ == "__main__":
    # Test the Bark model
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and initialize model
    model = BarkTTSModel()
    if model.initialize(device="cpu"):
        print("Model initialized successfully")
        print(f"Available voices: {[v.name for v in model.get_available_voices()[:5]]}")
        
        # Create a simple voice profile
        voice = VoiceProfile(
            name="Test Voice",
            id="en_speaker_0",
            gender=VoiceGender.MALE,
            language="en",
            style=VoiceStyle.NEUTRAL
        )
        
        # Create config
        config = TTSConfig(
            voice_profile=voice,
            speed=1.0,
            audio_format=AudioFormat.WAV
        )
        
        # Generate speech
        try:
            result = model.generate(
                text="Hello, this is a test of the Bark TTS system.",
                config=config
            )
            
            print(f"Generation successful!")
            print(f"Duration: {result.duration_ms}ms")
            print(f"Audio size: {len(result.audio_data)} bytes")
            
            # Save to file
            with open("test_bark.wav", "wb") as f:
                f.write(result.audio_data)
            print("Audio saved to test_bark.wav")
            
        except Exception as e:
            print(f"Error: {e}")
        
        # Cleanup
        model.cleanup()
    else:
        print("Failed to initialize model")