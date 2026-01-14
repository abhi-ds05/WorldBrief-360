"""
Coqui TTS Model Implementation

Coqui TTS is a deep learning toolkit for Text-to-Speech with a focus on:
- High-quality speech synthesis
- Multiple language support
- Efficient real-time inference
- Easy voice cloning
- Customizable models

This implementation supports various Coqui TTS models including:
- Tacotron2, FastSpeech2, Glow-TTS, etc.
- Multi-speaker models
- Multilingual models
"""

import os
import time
import warnings
import tempfile
import io
import logging
from typing import List, Optional, Dict, Any, Generator, Union, Tuple
from pathlib import Path
import numpy as np
import torch

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


class CoquiTTSModel(BaseTTSModel):
    """
    Coqui TTS model implementation.
    
    Coqui TTS provides various high-quality TTS models with support for
    multiple languages, speakers, and real-time inference.
    """
    
    # Default Coqui TTS models
    DEFAULT_MODELS = {
        # English models
        "en_ljspeech_tacotron2": "tts_models/en/ljspeech/tacotron2-DDC",
        "en_ljspeech_glow-tts": "tts_models/en/ljspeech/glow-tts",
        "en_ljspeech_speedy-speech": "tts_models/en/ljspeech/speedy-speech",
        "en_ljspeech_tacotron2-DDC_ph": "tts_models/en/ljspeech/tacotron2-DDC_ph",
        
        # Multilingual models
        "multilingual_multi-dataset_xtts": "tts_models/multilingual/multi-dataset/xtts_v2",
        "multilingual_multi-dataset_bark": "tts_models/multilingual/multi-dataset/bark",
        
        # Other languages
        "fr_css10_vits": "tts_models/fr/css10/vits",
        "de_css10_vits": "tts_models/de/css10/vits",
        "es_css10_vits": "tts_models/es/css10/vits",
        "it_css10_vits": "tts_models/it/css10/vits",
        "nl_css10_vits": "tts_models/nl/css10/vits",
        "ru_css10_vits": "tts_models/ru/css10/vits",
        "zh-cn_css10_vits": "tts_models/zh-cn/css10/vits",
        "ja_css10_vits": "tts_models/ja/css10/vits",
        "ko_css10_vits": "tts_models/ko/css10/vits",
    }
    
    # Supported languages with their codes
    LANGUAGE_MAPPING = {
        "en": "English",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "it": "Italian",
        "nl": "Dutch",
        "ru": "Russian",
        "zh-cn": "Chinese (Simplified)",
        "ja": "Japanese",
        "ko": "Korean",
        "pt": "Portuguese",
        "hi": "Hindi",
        "ar": "Arabic",
    }
    
    # Model architecture types
    MODEL_ARCHITECTURES = {
        "tacotron2": "Tacotron2",
        "glow-tts": "Glow-TTS",
        "speedy-speech": "Speedy-Speech",
        "vits": "VITS",
        "fastspeech2": "FastSpeech2",
        "xtts": "XTTS",
        "bark": "Bark",
    }
    
    def __init__(self, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC", model_version: str = "1.0"):
        """
        Initialize the Coqui TTS model.
        
        Args:
            model_name: Name/path of the Coqui TTS model
            model_version: Version of the model
        """
        super().__init__(model_name, model_version)
        self.tts = None
        self.device = None
        self.sample_rate = 22050  # Common Coqui TTS sample rate
        self.speakers = []
        self.languages = []
        self.model_type = self._infer_model_type(model_name)
        
        # Initialize default voices
        self._setup_default_voices()
    
    def _infer_model_type(self, model_name: str) -> str:
        """
        Infer the model architecture type from model name.
        
        Args:
            model_name: Full model name/path
            
        Returns:
            str: Model architecture type
        """
        model_name_lower = model_name.lower()
        
        for arch_type in self.MODEL_ARCHITECTURES:
            if arch_type in model_name_lower:
                return arch_type
        
        # Default based on common patterns
        if "tacotron" in model_name_lower:
            return "tacotron2"
        elif "vits" in model_name_lower:
            return "vits"
        elif "glow" in model_name_lower:
            return "glow-tts"
        elif "speedy" in model_name_lower:
            return "speedy-speech"
        elif "fastspeech" in model_name_lower:
            return "fastspeech2"
        else:
            return "unknown"
    
    def _setup_default_voices(self):
        """Setup default voice profiles for Coqui TTS."""
        self.available_voices = []
        
        # Create generic voice profiles based on model language
        lang_code = self._extract_language_from_model(self.model_name)
        
        # Create male and female voices
        for gender in [VoiceGender.MALE, VoiceGender.FEMALE]:
            voice_id = f"{lang_code}_{gender.value}_default"
            
            voice_profile = VoiceProfile(
                name=f"Coqui {self.LANGUAGE_MAPPING.get(lang_code, lang_code)} {gender.value.title()}",
                id=voice_id,
                gender=gender,
                language=lang_code,
                accent=None,
                style=VoiceStyle.NEUTRAL,
                sample_rate=self.sample_rate,
                bit_depth=16,
                description=f"Default Coqui TTS {gender.value} voice for {lang_code}",
                tags=["coqui", lang_code, "default", gender.value]
            )
            self.available_voices.append(voice_profile)
    
    def _extract_language_from_model(self, model_name: str) -> str:
        """
        Extract language code from model name.
        
        Args:
            model_name: Full model name/path
            
        Returns:
            str: Language code (e.g., 'en', 'fr')
        """
        # Parse model name to extract language
        # Format: tts_models/{lang}/...
        parts = model_name.split('/')
        if len(parts) >= 3:
            lang_part = parts[1]
            # Handle multilingual models
            if lang_part == "multilingual":
                return "en"  # Default to English for multilingual
            return lang_part
        return "en"  # Default to English
    
    def initialize(self, device: str = "cpu", **kwargs) -> bool:
        """
        Initialize the Coqui TTS model.
        
        Args:
            device: Device to run the model on ('cpu', 'cuda', 'mps')
            **kwargs: Additional parameters:
                - model_name: Override model name
                - gpu: Use GPU (deprecated, use device instead)
                - progress_bar: Show download progress
                - num_threads: Number of CPU threads
                - use_cuda: Force CUDA (deprecated)
                - cache_dir: Directory for model cache
                - speaker_idx: Speaker index for multi-speaker models
                - language_idx: Language index for multilingual models
                
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            import torch
            try:
                from TTS.api import TTS # type: ignore
            except ImportError:
                logger.error("TTS module not installed. Please install it with: pip install TTS")
                raise
            
            # Handle deprecated parameters
            if kwargs.get("gpu", False):
                device = "cuda"
            if kwargs.get("use_cuda", False):
                device = "cuda"
            
            self.device = device
            
            # Check device availability
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                self.device = "cpu"
            
            if device == "mps" and not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                logger.warning("MPS requested but not available. Falling back to CPU.")
                self.device = "cpu"
            
            # Get model parameters
            model_name = kwargs.get("model_name", self.model_name)
            progress_bar = kwargs.get("progress_bar", True)
            num_threads = kwargs.get("num_threads", 4)
            cache_dir = kwargs.get("cache_dir", None)
            
            logger.info(f"Loading Coqui TTS model: {model_name} on device: {self.device}")
            
            # Set CPU threads for better performance
            if self.device == "cpu" and num_threads:
                torch.set_num_threads(num_threads)
            
            # Initialize TTS
            self.tts = TTS(
                model_name=model_name,
                progress_bar=progress_bar,
                gpu=(self.device == "cuda")
            )
            
            # Update model name if it was changed
            self.model_name = model_name
            self.model_type = self._infer_model_type(model_name)
            
            # Get sample rate from model
            try:
                self.sample_rate = self.tts.synthesizer.output_sample_rate
            except:
                pass  # Keep default if not available
            
            # Get available speakers if multi-speaker model
            self._load_speakers()
            
            # Get available languages if multilingual model
            self._load_languages()
            
            # Update available voices with actual model capabilities
            self._update_voices_from_model()
            
            self.initialized = True
            logger.info(f"Coqui TTS model initialized successfully on {self.device}")
            logger.info(f"Model type: {self.model_type}, Sample rate: {self.sample_rate}")
            
            return True
            
        except ImportError as e:
            logger.error(f"Required packages not installed: {e}")
            logger.error("Please install: pip install TTS")
            return False
            
        except Exception as e:
            logger.error(f"Failed to initialize Coqui TTS model: {e}")
            return False
    
    def _load_speakers(self):
        """Load available speakers from the model."""
        try:
            if hasattr(self.tts, 'speakers'):
                self.speakers = list(self.tts.speakers)
                logger.info(f"Found {len(self.speakers)} speakers in model")
            else:
                self.speakers = []
        except:
            self.speakers = []
    
    def _load_languages(self):
        """Load available languages from the model."""
        try:
            if hasattr(self.tts, 'languages'):
                self.languages = list(self.tts.languages)
                logger.info(f"Found {len(self.languages)} languages in model")
            else:
                self.languages = []
        except:
            self.languages = []
    
    def _update_voices_from_model(self):
        """Update available voices based on actual model capabilities."""
        # Clear default voices
        self.available_voices.clear()
        
        lang_code = self._extract_language_from_model(self.model_name)
        
        # If model has speakers, create voices for each
        if self.speakers:
            for i, speaker in enumerate(self.speakers):
                # Try to infer gender from speaker name
                gender = self._infer_gender_from_speaker(speaker)
                
                voice_profile = VoiceProfile(
                    name=f"Coqui Speaker {i}: {speaker}",
                    id=f"coqui_{lang_code}_speaker_{i}",
                    gender=gender,
                    language=lang_code,
                    accent=None,
                    style=VoiceStyle.NEUTRAL,
                    sample_rate=self.sample_rate,
                    bit_depth=16,
                    description=f"Coqui TTS speaker: {speaker}",
                    tags=["coqui", lang_code, f"speaker_{i}", gender.value]
                )
                self.available_voices.append(voice_profile)
        else:
            # Create generic voices
            for gender in [VoiceGender.MALE, VoiceGender.FEMALE]:
                voice_id = f"coqui_{lang_code}_{gender.value}_default"
                
                voice_profile = VoiceProfile(
                    name=f"Coqui {self.LANGUAGE_MAPPING.get(lang_code, lang_code)} {gender.value.title()}",
                    id=voice_id,
                    gender=gender,
                    language=lang_code,
                    accent=None,
                    style=VoiceStyle.NEUTRAL,
                    sample_rate=self.sample_rate,
                    bit_depth=16,
                    description=f"Coqui TTS {gender.value} voice",
                    tags=["coqui", lang_code, "default", gender.value]
                )
                self.available_voices.append(voice_profile)
    
    def _infer_gender_from_speaker(self, speaker_name: str) -> VoiceGender:
        """
        Infer gender from speaker name.
        
        Args:
            speaker_name: Name or ID of the speaker
            
        Returns:
            VoiceGender: Inferred gender
        """
        speaker_lower = speaker_name.lower()
        
        male_indicators = ["male", "man", "boy", "m_", "_m", "p", "p"]
        female_indicators = ["female", "woman", "girl", "f_", "_f", "l", "l"]
        
        if any(indicator in speaker_lower for indicator in male_indicators):
            return VoiceGender.MALE
        elif any(indicator in speaker_lower for indicator in female_indicators):
            return VoiceGender.FEMALE
        else:
            return VoiceGender.NEUTRAL
    
    def generate(
        self, 
        text: str, 
        config: TTSConfig,
        **kwargs
    ) -> TTSResult:
        """
        Generate speech from text using Coqui TTS.
        
        Args:
            text: Input text to convert to speech
            config: TTS configuration
            **kwargs: Additional parameters:
                - speaker: Speaker name/ID (for multi-speaker models)
                - language: Language code (for multilingual models)
                - emotion: Emotion/style (if supported)
                - speed: Override config.speed
                - speaker_wav: Path to reference audio for voice cloning
                - split_sentences: Split text into sentences (default: True)
                - sentence_silence: Silence between sentences in seconds (default: 0.2)
                
        Returns:
            TTSResult: The generated audio and metadata
            
        Raises:
            ModelNotInitializedError: If model is not initialized
            InvalidConfigError: If configuration is invalid
            TextTooLongError: If text is too long
        """
        if not self.initialized:
            raise ModelNotInitializedError("Coqui TTS model not initialized. Call initialize() first.")
        
        start_time = time.time()
        
        try:
            # Validate configuration
            is_valid, error_msg = self.validate_config(config)
            if not is_valid:
                raise InvalidConfigError(error_msg)
            
            # Preprocess text
            text = self.preprocess_text(text)
            
            # Check text length
            max_length = kwargs.get("max_length", 1000)
            if len(text) > max_length:
                raise TextTooLongError(
                    f"Text too long ({len(text)} characters). "
                    f"Maximum allowed: {max_length} characters."
                )
            
            # Prepare generation parameters
            generation_kwargs = {}
            
            # Speaker selection
            if self.speakers:
                speaker = kwargs.get("speaker")
                if speaker:
                    if speaker in self.speakers:
                        generation_kwargs["speaker"] = speaker
                    else:
                        logger.warning(f"Speaker '{speaker}' not found. Using default.")
                elif len(self.speakers) > 0:
                    # Use first speaker as default
                    generation_kwargs["speaker"] = self.speakers[0]
            
            # Language selection
            if self.languages:
                language = kwargs.get("language", config.voice_profile.language)
                if language in self.languages:
                    generation_kwargs["language"] = language
                elif len(self.languages) > 0:
                    # Use first language as default
                    generation_kwargs["language"] = self.languages[0]
            
            # Voice cloning (if supported and speaker_wav provided)
            speaker_wav = kwargs.get("speaker_wav")
            if speaker_wav and Path(speaker_wav).exists():
                generation_kwargs["speaker_wav"] = speaker_wav
            
            # Text splitting
            split_sentences = kwargs.get("split_sentences", True)
            sentence_silence = kwargs.get("sentence_silence", 0.2)
            
            # Generate audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Generate speech to file
                self.tts.tts_to_file(
                    text=text,
                    file_path=temp_path,
                    split_sentences=split_sentences,
                    sentence_silence=sentence_silence,
                    **generation_kwargs
                )
                
                # Read the generated audio
                import soundfile as sf
                audio_array, sample_rate = sf.read(temp_path)
                
                # Apply speed adjustment
                speed = kwargs.get("speed", config.speed)
                if speed != 1.0:
                    audio_array = self._adjust_speed(audio_array, speed, sample_rate)
                
                # Convert to target format
                audio_bytes = self._convert_audio_format(
                    audio_array,
                    config.audio_format,
                    config.sample_rate or sample_rate
                )
                
                # Calculate duration
                duration_ms = int(len(audio_array) / sample_rate * 1000)
                
                # Add silence if requested
                if config.add_silence_ms > 0:
                    audio_bytes = self._add_silence(audio_bytes, config.add_silence_ms, sample_rate)
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            
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
                    "speaker": generation_kwargs.get("speaker", "default"),
                    "language": generation_kwargs.get("language", config.voice_profile.language),
                    "model_type": self.model_type,
                    "sample_rate": sample_rate,
                    "device": str(self.device),
                    "coqui_specific": {
                        "split_sentences": split_sentences,
                        "sentence_silence": sentence_silence,
                        "has_speaker_wav": bool(speaker_wav),
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating speech with Coqui TTS: {e}")
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
        
        Note: Coqui TTS generates to file, so we generate first then stream.
        
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
        Get list of available voices for Coqui TTS.
        
        Returns:
            List[VoiceProfile]: Available voice profiles
        """
        return self.available_voices.copy()
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages by the model.
        
        Returns:
            List[str]: Supported language codes
        """
        if self.languages:
            return self.languages.copy()
        
        # Fallback to model language inference
        lang_code = self._extract_language_from_model(self.model_name)
        if lang_code == "multilingual":
            return list(self.LANGUAGE_MAPPING.keys())
        
        return [lang_code]
    
    def _adjust_speed(self, audio_array: np.ndarray, speed: float, sample_rate: int) -> np.ndarray:
        """
        Adjust audio speed using resampling.
        
        Args:
            audio_array: Input audio array
            speed: Speed factor
            sample_rate: Original sample rate
            
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
            
            return adjusted
            
        except ImportError:
            logger.warning("librosa not installed. Using simple speed adjustment.")
            # Simple resampling
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
            import soundfile as sf
            import io
            
            # Normalize audio if needed
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            # Handle mono/stereo
            if len(audio_array.shape) == 1:
                audio_array = audio_array.reshape(-1, 1)
            
            buffer = io.BytesIO()
            
            if format == AudioFormat.WAV:
                sf.write(buffer, audio_array, sample_rate, format='WAV')
            elif format == AudioFormat.MP3:
                try:
                    from pydub import AudioSegment
                    
                    # Convert to AudioSegment
                    if len(audio_array.shape) > 1:
                        # Stereo
                        audio_segment = AudioSegment(
                            audio_array.tobytes(),
                            frame_rate=sample_rate,
                            sample_width=audio_array.dtype.itemsize,
                            channels=audio_array.shape[1]
                        )
                    else:
                        # Mono
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
            
            else:
                # Default to WAV
                sf.write(buffer, audio_array, sample_rate, format='WAV')
            
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error converting audio format: {e}")
            # Fallback to simple WAV
            import wave
            import io
            
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes((audio_array * 32767).astype(np.int16).tobytes())
            
            return buffer.getvalue()
    
    def _add_silence(self, audio_bytes: bytes, silence_ms: int, sample_rate: int) -> bytes:
        """
        Add silence to audio.
        
        Args:
            audio_bytes: Input audio bytes
            silence_ms: Silence duration in milliseconds
            sample_rate: Sample rate
            
        Returns:
            bytes: Audio with added silence
        """
        try:
            from pydub import AudioSegment
            import io
            
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
    
    def list_models(self) -> Dict[str, str]:
        """
        List available Coqui TTS models.
        
        Returns:
            Dict[str, str]: Model names and descriptions
        """
        return self.DEFAULT_MODELS.copy()
    
    def clone_voice(self, reference_audio_path: str, text: str = "Hello world") -> Optional[VoiceProfile]:
        """
        Test voice cloning with a reference audio.
        
        Args:
            reference_audio_path: Path to reference audio file
            text: Text to synthesize with cloned voice
            
        Returns:
            Optional[VoiceProfile]: Voice profile for cloned voice if successful
        """
        if not self.initialized:
            raise ModelNotInitializedError("Model not initialized")
        
        if not Path(reference_audio_path).exists():
            logger.error(f"Reference audio not found: {reference_audio_path}")
            return None
        
        try:
            # Test voice cloning
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            self.tts.tts_to_file(
                text=text,
                file_path=temp_path,
                speaker_wav=reference_audio_path
            )
            
            # Create voice profile for cloned voice
            voice_profile = VoiceProfile(
                name="Cloned Voice",
                id=f"cloned_{int(time.time())}",
                gender=VoiceGender.NEUTRAL,
                language=self._extract_language_from_model(self.model_name),
                accent=None,
                style=VoiceStyle.NEUTRAL,
                sample_rate=self.sample_rate,
                description="Voice cloned from reference audio",
                tags=["cloned", "custom"]
            )
            
            os.unlink(temp_path)
            return voice_profile
            
        except Exception as e:
            logger.error(f"Voice cloning failed: {e}")
            return None
    
    def cleanup(self) -> None:
        """Clean up Coqui TTS model resources."""
        super().cleanup()
        
        if self.tts is not None:
            # Clean up TTS resources
            del self.tts
            self.tts = None
        
        # Clear cache
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        logger.info("Coqui TTS model cleaned up")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the Coqui TTS model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        info = super().get_model_info()
        
        info.update({
            "device": str(self.device),
            "sample_rate": self.sample_rate,
            "model_type": self.model_type,
            "model_architecture": self.MODEL_ARCHITECTURES.get(self.model_type, "unknown"),
            "speakers": self.speakers,
            "languages": self.languages,
            "supports_voice_cloning": "xtts" in self.model_type or self.model_type == "xtts",
            "is_multilingual": len(self.languages) > 1,
            "is_multi_speaker": len(self.speakers) > 0,
        })
        
        return info


# Example usage
if __name__ == "__main__":
    # Test the Coqui TTS model
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and initialize model
    model = CoquiTTSModel()
    if model.initialize(device="cpu"):
        print("Model initialized successfully")
        print(f"Model info: {model.get_model_info()}")
        
        # Get a voice profile
        voices = model.get_available_voices()
        if voices:
            voice = voices[0]
            
            # Create config
            config = TTSConfig(
                voice_profile=voice,
                speed=1.0,
                audio_format=AudioFormat.WAV
            )
            
            # Generate speech
            try:
                result = model.generate(
                    text="Hello, this is a test of the Coqui TTS system.",
                    config=config
                )
                
                print(f"Generation successful!")
                print(f"Duration: {result.duration_ms}ms")
                print(f"Audio size: {len(result.audio_data)} bytes")
                
                # Save to file
                with open("test_coqui.wav", "wb") as f:
                    f.write(result.audio_data)
                print("Audio saved to test_coqui.wav")
                
            except Exception as e:
                print(f"Error: {e}")
        
        # Cleanup
        model.cleanup()
    else:
        print("Failed to initialize model")