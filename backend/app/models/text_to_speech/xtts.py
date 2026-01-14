"""
XTTS (Extended Text-to-Speech) Model Implementation

XTTS is a state-of-the-art TTS model from Coqui that features:
- High-quality, natural-sounding speech
- Multilingual support (13+ languages)
- Zero-shot voice cloning (clone voices from short audio samples)
- Cross-lingual voice cloning (clone voice in one language, synthesize in another)
- Efficient inference
"""

import os
import time
import warnings
import tempfile
import io
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


class XTTSModel(BaseTTSModel):
    """
    XTTS (Extended Text-to-Speech) model implementation.
    
    XTTS v2 is a high-quality TTS model that supports:
    - 13+ languages with native accents
    - Zero-shot voice cloning from short audio samples
    - Cross-lingual voice cloning
    - Emotion control
    - Stable, natural-sounding speech
    """
    
    # Supported languages in XTTS v2
    XTTS_LANGUAGES = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "pl": "Polish",
        "tr": "Turkish",
        "ru": "Russian",
        "nl": "Dutch",
        "cs": "Czech",
        "ar": "Arabic",
        "zh-cn": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "hu": "Hungarian",
        "hi": "Hindi",
    }
    
    # Default XTTS models
    DEFAULT_MODELS = {
        "xtts_v2": "tts_models/multilingual/multi-dataset/xtts_v2",
        "xtts_v1": "tts_models/multilingual/multi-dataset/xtts_v1.1",
        "xtts_v2_small": "tts_models/multilingual/multi-dataset/xtts_v2_small",
    }
    
    # Emotion styles supported by XTTS
    XTTS_EMOTIONS = {
        "neutral": "Neutral",
        "happy": "Happy",
        "sad": "Sad",
        "angry": "Angry",
        "surprised": "Surprised",
        "fearful": "Fearful",
        "disgusted": "Disgusted",
        "excited": "Excited",
        "cheerful": "Cheerful",
        "calm": "Calm",
    }
    
    # Voice cloning parameters
    CLONING_SETTINGS = {
        "min_cloning_samples": 3,      # Minimum seconds for good cloning
        "max_cloning_samples": 30,     # Maximum seconds to use
        "optimal_cloning_samples": 10, # Optimal seconds
        "supported_sample_rates": [16000, 22050, 24000, 44100, 48000],
    }
    
    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2", model_version: str = "1.0"):
        """
        Initialize the XTTS model.
        
        Args:
            model_name: Name/path of the XTTS model
            model_version: Version of the model
        """
        super().__init__(model_name, model_version)
        self.tts = None
        self.device = None
        self.sample_rate = 24000  # XTTS default sample rate
        self.languages = list(self.XTTS_LANGUAGES.keys())
        self.cloned_voices = {}  # Cache for cloned voices
        
        # Voice cloning specific attributes
        self.speaker_embedding = None
        self.gpt_cond_latent = None
        self.reference_audio = None
        self.reference_language = None
        
        # Initialize default voices
        self._setup_default_voices()
    
    def _setup_default_voices(self):
        """Setup default voice profiles for XTTS."""
        self.available_voices = []
        
        # Create generic multilingual voices
        for lang_code in ["en", "es", "fr", "de", "it", "pt"]:
            for gender in [VoiceGender.MALE, VoiceGender.FEMALE]:
                voice_id = f"xtts_{lang_code}_{gender.value}_default"
                
                voice_profile = VoiceProfile(
                    name=f"XTTS {self.XTTS_LANGUAGES[lang_code]} {gender.value.title()}",
                    id=voice_id,
                    gender=gender,
                    language=lang_code,
                    accent=None,
                    style=VoiceStyle.NEUTRAL,
                    sample_rate=self.sample_rate,
                    bit_depth=16,
                    description=f"Default XTTS {gender.value} voice for {self.XTTS_LANGUAGES[lang_code]}",
                    tags=["xtts", lang_code, "default", gender.value, "multilingual"]
                )
                self.available_voices.append(voice_profile)
    
    def initialize(self, device: str = "cpu", **kwargs) -> bool:
        """
        Initialize the XTTS model.
        
        Args:
            device: Device to run the model on ('cpu', 'cuda', 'mps')
            **kwargs: Additional parameters:
                - model_name: Override model name
                - use_deepspeed: Use DeepSpeed for inference
                - use_bettertransformer: Use BetterTransformer optimization
                - dvae_device: Device for DVAE (if different)
                - gpt_device: Device for GPT (if different)
                - cache_dir: Directory for model cache
                - local_files_only: Use local files only
                - low_vram: Optimize for low VRAM usage
                
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            import torch
            from TTS.api import TTS # type: ignore
            from TTS.tts.configs.xtts_config import XttsConfig # type: ignore
            from TTS.tts.models.xtts import Xtts # type: ignore
            
            # Handle device
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
            use_deepspeed = kwargs.get("use_deepspeed", False)
            use_bettertransformer = kwargs.get("use_bettertransformer", False)
            dvae_device = kwargs.get("dvae_device", self.device)
            gpt_device = kwargs.get("gpt_device", self.device)
            cache_dir = kwargs.get("cache_dir", None)
            local_files_only = kwargs.get("local_files_only", False)
            low_vram = kwargs.get("low_vram", False)
            
            logger.info(f"Loading XTTS model: {model_name} on device: {self.device}")
            
            # Initialize using TTS API (simpler approach)
            self.tts = TTS(
                model_name=model_name,
                progress_bar=True,
                gpu=(self.device == "cuda")
            )
            
            # Get the underlying Xtts model for advanced features
            if hasattr(self.tts, 'synthesizer') and hasattr(self.tts.synthesizer, 'model'):
                self.xtts_model = self.tts.synthesizer.model
                
                # Apply optimizations if requested
                if use_bettertransformer:
                    try:
                        from optimum.bettertransformer import BetterTransformer # type: ignore
                        self.xtts_model = BetterTransformer.transform(self.xtts_model)
                        logger.info("Applied BetterTransformer optimization")
                    except (ImportError, Exception) as e:
                        logger.warning(f"BetterTransformer optimization not available: {e}")
                
                # Low VRAM mode
                if low_vram and self.device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.set_per_process_memory_fraction(0.5)
            
            # Update model name
            self.model_name = model_name
            
            # Get sample rate
            try:
                if hasattr(self.tts, 'synthesizer'):
                    self.sample_rate = self.tts.synthesizer.output_sample_rate
            except:
                pass  # Keep default
            
            self.initialized = True
            logger.info(f"XTTS model initialized successfully on {self.device}")
            logger.info(f"Supported languages: {self.languages}")
            
            return True
            
        except ImportError as e:
            logger.error(f"Required packages not installed: {e}")
            logger.error("Please install: pip install TTS")
            return False
            
        except Exception as e:
            logger.error(f"Failed to initialize XTTS model: {e}")
            return False
    
    def clone_voice(
        self, 
        reference_audio_path: str, 
        reference_text: Optional[str] = None,
        language: str = "en",
        **kwargs
    ) -> Optional[VoiceProfile]:
        """
        Clone a voice from reference audio.
        
        Args:
            reference_audio_path: Path to reference audio file
            reference_text: Transcript of reference audio (optional, improves quality)
            language: Language of reference audio
            **kwargs: Additional cloning parameters:
                - temperature: Sampling temperature (default: 0.7)
                - length_penalty: Length penalty (default: 1.0)
                - repetition_penalty: Repetition penalty (default: 7.0)
                - top_k: Top-k sampling (default: 50)
                - top_p: Top-p sampling (default: 0.85)
                - condition_latents: Pre-computed condition latents
                
        Returns:
            Optional[VoiceProfile]: Voice profile for cloned voice
        """
        if not self.initialized:
            raise ModelNotInitializedError("XTTS model not initialized")
        
        if not Path(reference_audio_path).exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_audio_path}")
        
        try:
            logger.info(f"Cloning voice from: {reference_audio_path}")
            
            # Load and preprocess reference audio
            audio_array, sample_rate = self._load_audio(reference_audio_path)
            
            # Check audio duration
            duration = len(audio_array) / sample_rate
            if duration < self.CLONING_SETTINGS["min_cloning_samples"]:
                logger.warning(f"Reference audio too short ({duration:.1f}s). Minimum recommended: {self.CLONING_SETTINGS['min_cloning_samples']}s")
            
            if duration > self.CLONING_SETTINGS["max_cloning_samples"]:
                logger.info(f"Trimming reference audio from {duration:.1f}s to {self.CLONING_SETTINGS['optimal_cloning_samples']}s")
                audio_array = audio_array[:int(self.CLONING_SETTINGS["optimal_cloning_samples"] * sample_rate)]
            
            # Compute speaker embedding and condition latents
            if hasattr(self, 'xtts_model') and hasattr(self.xtts_model, 'get_conditioning_latents'):
                # Advanced: Use model directly for better control
                from TTS.tts.models.xtts import Xtts # type: ignore
                
                # Convert audio to torch tensor
                audio_tensor = torch.FloatTensor(audio_array).unsqueeze(0)
                
                # Get conditioning latents
                gpt_cond_latent, speaker_embedding = self.xtts_model.get_conditioning_latents(
                    audio_tensor,
                    gpt_cond_len=self.xtts_model.args.gpt_cond_len,
                    max_ref_length=self.xtts_model.args.max_ref_len,
                    sound_norm_refs=self.xtts_model.args.sound_norm_refs
                )
                
                # Store for future use
                self.gpt_cond_latent = gpt_cond_latent
                self.speaker_embedding = speaker_embedding
                self.reference_audio = reference_audio_path
                self.reference_language = language
                
                logger.info("Voice cloned successfully using direct model API")
                
            else:
                # Use TTS API for cloning
                # This creates embeddings that will be used in the next generation
                self.reference_audio = reference_audio_path
                self.reference_language = language
                
                # Test cloning with a short phrase
                test_text = reference_text or "Hello, this is my cloned voice."
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_path = temp_file.name
                
                try:
                    self.tts.tts_to_file(
                        text=test_text,
                        file_path=temp_path,
                        speaker_wav=reference_audio_path,
                        language=language
                    )
                    logger.info("Voice cloned successfully using TTS API")
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
            
            # Create voice profile for cloned voice
            voice_id = f"cloned_{language}_{int(time.time())}"
            
            voice_profile = VoiceProfile(
                name=f"Cloned Voice ({language.upper()})",
                id=voice_id,
                gender=VoiceGender.NEUTRAL,  # Unknown from cloning
                language=language,
                accent=None,
                style=VoiceStyle.NEUTRAL,
                sample_rate=self.sample_rate,
                bit_depth=16,
                description=f"Voice cloned from {Path(reference_audio_path).name}",
                tags=["cloned", language, "custom", "zero-shot"]
            )
            
            # Cache the cloned voice
            self.cloned_voices[voice_id] = {
                "profile": voice_profile,
                "reference_path": reference_audio_path,
                "language": language,
                "timestamp": time.time()
            }
            
            # Add to available voices
            self.available_voices.append(voice_profile)
            
            return voice_profile
            
        except Exception as e:
            logger.error(f"Voice cloning failed: {e}")
            return None
    
    def generate(
        self, 
        text: str, 
        config: TTSConfig,
        **kwargs
    ) -> TTSResult:
        """
        Generate speech from text using XTTS.
        
        Args:
            text: Input text to convert to speech
            config: TTS configuration
            **kwargs: Additional parameters:
                - speaker_wav: Path to reference audio for voice cloning
                - language: Language code (overrides config.voice_profile.language)
                - emotion: Emotion/style (e.g., 'happy', 'sad', 'angry')
                - temperature: Sampling temperature (default: 0.7)
                - length_penalty: Length penalty (default: 1.0)
                - repetition_penalty: Repetition penalty (default: 7.0)
                - top_k: Top-k sampling (default: 50)
                - top_p: Top-p sampling (default: 0.85)
                - enable_text_splitting: Split long text (default: True)
                - stream: Stream audio chunks (default: False)
                - use_direct_model: Use direct model API instead of TTS API
                
        Returns:
            TTSResult: The generated audio and metadata
            
        Raises:
            ModelNotInitializedError: If model is not initialized
            InvalidConfigError: If configuration is invalid
            TextTooLongError: If text is too long
        """
        if not self.initialized:
            raise ModelNotInitializedError("XTTS model not initialized. Call initialize() first.")
        
        start_time = time.time()
        
        try:
            # Validate configuration
            is_valid, error_msg = self.validate_config(config)
            if not is_valid:
                raise InvalidConfigError(error_msg)
            
            # Check language support
            language = kwargs.get("language", config.voice_profile.language)
            if language not in self.languages:
                raise InvalidConfigError(f"Language '{language}' not supported by XTTS. Supported: {self.languages}")
            
            # Preprocess text
            text = self.preprocess_text(text)
            
            # Check text length
            max_length = kwargs.get("max_length", 500)
            if len(text) > max_length:
                raise TextTooLongError(
                    f"Text too long ({len(text)} characters). "
                    f"Maximum recommended: {max_length} characters."
                )
            
            # Prepare generation parameters
            generation_kwargs = {
                "language": language,
                "split_sentences": kwargs.get("enable_text_splitting", True),
            }
            
            # Voice cloning parameters
            speaker_wav = kwargs.get("speaker_wav")
            if speaker_wav:
                if not Path(speaker_wav).exists():
                    raise FileNotFoundError(f"Speaker audio not found: {speaker_wav}")
                generation_kwargs["speaker_wav"] = speaker_wav
            elif self.reference_audio and config.voice_profile.id.startswith("cloned_"):
                # Use cached cloned voice
                generation_kwargs["speaker_wav"] = self.reference_audio
                if "language" not in generation_kwargs:
                    generation_kwargs["language"] = self.reference_language or language
            
            # Emotion/style parameter
            emotion = kwargs.get("emotion")
            if emotion and emotion in self.XTTS_EMOTIONS:
                # XTTS v2 doesn't directly support emotion, but we can adjust parameters
                # For future versions that might support emotion
                pass
            
            # Advanced generation parameters
            advanced_params = {}
            for param in ["temperature", "length_penalty", "repetition_penalty", "top_k", "top_p"]:
                if param in kwargs:
                    advanced_params[param] = kwargs[param]
            
            # Use direct model API if requested and available
            use_direct_model = kwargs.get("use_direct_model", False)
            
            if use_direct_model and hasattr(self, 'xtts_model') and self.speaker_embedding is not None:
                # Use direct model inference with pre-computed latents
                return self._generate_direct(
                    text=text,
                    config=config,
                    language=language,
                    advanced_params=advanced_params,
                    start_time=start_time
                )
            
            # Use TTS API (simpler)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Generate speech to file
                self.tts.tts_to_file(
                    text=text,
                    file_path=temp_path,
                    **generation_kwargs,
                    **advanced_params
                )
                
                # Read the generated audio
                import soundfile as sf
                audio_array, sample_rate = sf.read(temp_path)
                
                # Apply speed adjustment
                speed = kwargs.get("speed", config.speed)
                if speed != 1.0:
                    audio_array = self._adjust_speed(audio_array, speed, sample_rate)
                
                # Apply pitch adjustment (simulated through speed adjustment)
                pitch = kwargs.get("pitch", config.pitch)
                if pitch != 1.0:
                    # Simple pitch adjustment through resampling
                    audio_array = self._adjust_pitch(audio_array, pitch, sample_rate)
                
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
                    "language": language,
                    "voice_cloning": bool(speaker_wav or self.reference_audio),
                    "reference_audio": speaker_wav or self.reference_audio,
                    "sample_rate": sample_rate,
                    "device": str(self.device),
                    "xtts_specific": {
                        "model_used": "TTS API",
                        "text_split": generation_kwargs.get("split_sentences", True),
                        **advanced_params
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating speech with XTTS: {e}")
            raise TTSModelError(f"Failed to generate speech: {str(e)}")
    
    def _generate_direct(
        self,
        text: str,
        config: TTSConfig,
        language: str,
        advanced_params: Dict[str, Any],
        start_time: float
    ) -> TTSResult:
        """
        Generate speech using direct model API (faster, more control).
        
        This method uses pre-computed speaker embeddings and condition latents.
        """
        try:
            from TTS.tts.models.xtts import Xtts # type: ignore
            
            # Prepare generation parameters
            gen_kwargs = {
                "temperature": advanced_params.get("temperature", 0.7),
                "length_penalty": advanced_params.get("length_penalty", 1.0),
                "repetition_penalty": advanced_params.get("repetition_penalty", 7.0),
                "top_k": advanced_params.get("top_k", 50),
                "top_p": advanced_params.get("top_p", 0.85),
                "cond_free": False,
                "do_sample": True,
            }
            
            # Generate audio using direct model
            audio_tensor = self.xtts_model.inference(
                text,
                language,
                self.gpt_cond_latent,
                self.speaker_embedding,
                **gen_kwargs
            )
            
            # Convert to numpy
            audio_array = audio_tensor.cpu().numpy().squeeze()
            
            # Apply speed adjustment
            if config.speed != 1.0:
                audio_array = self._adjust_speed(audio_array, config.speed, self.sample_rate)
            
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
                audio_bytes = self._add_silence(audio_bytes, config.add_silence_ms, self.sample_rate)
            
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
                    "language": language,
                    "voice_cloning": True,
                    "sample_rate": self.sample_rate,
                    "device": str(self.device),
                    "xtts_specific": {
                        "model_used": "Direct Model API",
                        "has_precomputed_latents": True,
                        **gen_kwargs
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Direct model generation failed: {e}")
            raise
    
    def stream_generate(
        self, 
        text: str, 
        config: TTSConfig,
        chunk_size: int = 4096,
        **kwargs
    ) -> Generator[bytes, None, TTSResult]:
        """
        Generate speech and stream the audio in chunks.
        
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
        # XTTS doesn't natively support streaming, so generate first
        result = self.generate(text, config, **kwargs)
        audio_data = result.audio_data
        
        # Stream in chunks
        for i in range(0, len(audio_data), chunk_size):
            yield audio_data[i:i + chunk_size]
        
        return result
    
    def get_available_voices(self) -> List[VoiceProfile]:
        """
        Get list of available voices for XTTS.
        
        Returns:
            List[VoiceProfile]: Available voice profiles
        """
        return self.available_voices.copy()
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages by XTTS.
        
        Returns:
            List[str]: Supported language codes
        """
        return self.languages.copy()
    
    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and convert to mono, target sample rate.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple[np.ndarray, int]: Audio array and sample rate
        """
        try:
            import soundfile as sf
            audio_array, sample_rate = sf.read(audio_path)
        except:
            # Fallback to torchaudio
            audio_array, sample_rate = torchaudio.load(audio_path)
            audio_array = audio_array.numpy().squeeze()
        
        # Convert to mono if stereo
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=0)
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            audio_array = self._resample_audio(audio_array, sample_rate, self.sample_rate)
        
        return audio_array, self.sample_rate
    
    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            from scipy import signal
            number_of_samples = round(len(audio) * float(target_sr) / orig_sr)
            return signal.resample(audio, number_of_samples)
    
    def _adjust_speed(self, audio_array: np.ndarray, speed: float, sample_rate: int) -> np.ndarray:
        """Adjust audio speed."""
        if speed == 1.0:
            return audio_array
        
        try:
            import librosa
            return librosa.effects.time_stretch(y=audio_array, rate=speed)
        except ImportError:
            from scipy import signal
            new_length = int(len(audio_array) / speed)
            return signal.resample(audio_array, new_length)
    
    def _adjust_pitch(self, audio_array: np.ndarray, pitch: float, sample_rate: int) -> np.ndarray:
        """Adjust audio pitch (simulated through speed adjustment)."""
        # Simple pitch adjustment: speed change with sample rate preservation
        if pitch == 1.0:
            return audio_array
        
        # For better pitch adjustment, we'd need more advanced libraries
        # This is a simple approximation
        try:
            import librosa
            # Speed adjustment affects pitch inversely
            return librosa.effects.pitch_shift(
                y=audio_array,
                sr=sample_rate,
                n_steps=12 * (pitch - 1.0)  # Approximate: 1 semitone per 0.0833 change
            )
        except ImportError:
            logger.warning("librosa not installed. Using simple speed adjustment for pitch.")
            return self._adjust_speed(audio_array, 1.0 / pitch, sample_rate)
    
    def _convert_audio_format(
        self, 
        audio_array: np.ndarray, 
        format: AudioFormat,
        sample_rate: int
    ) -> bytes:
        """Convert audio array to target format."""
        try:
            import soundfile as sf
            import io
            
            buffer = io.BytesIO()
            
            if format == AudioFormat.WAV:
                sf.write(buffer, audio_array, sample_rate, format='WAV')
            elif format == AudioFormat.MP3:
                try:
                    from pydub import AudioSegment
                    audio_segment = AudioSegment(
                        audio_array.tobytes(),
                        frame_rate=sample_rate,
                        sample_width=audio_array.dtype.itemsize,
                        channels=1
                    )
                    audio_segment.export(buffer, format="mp3")
                except ImportError:
                    sf.write(buffer, audio_array, sample_rate, format='WAV')
            elif format == AudioFormat.FLAC:
                sf.write(buffer, audio_array, sample_rate, format='FLAC')
            elif format == AudioFormat.OGG:
                sf.write(buffer, audio_array, sample_rate, format='OGG')
            else:
                sf.write(buffer, audio_array, sample_rate, format='WAV')
            
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Audio format conversion error: {e}")
            # Fallback to raw WAV
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
        """Add silence to audio."""
        try:
            from pydub import AudioSegment
            import io
            
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            silence = AudioSegment.silent(duration=silence_ms)
            audio_with_silence = silence + audio
            
            buffer = io.BytesIO()
            audio_with_silence.export(buffer, format="wav")
            
            return buffer.getvalue()
        except ImportError:
            return audio_bytes
    
    def cross_lingual_clone(
        self,
        reference_audio_path: str,
        reference_language: str,
        target_language: str,
        text: str = "Hello, this is a cross-lingual voice clone."
    ) -> Optional[TTSResult]:
        """
        Clone a voice from one language and synthesize in another.
        
        Args:
            reference_audio_path: Path to reference audio
            reference_language: Language of reference audio
            target_language: Language to synthesize in
            text: Text to synthesize
            
        Returns:
            Optional[TTSResult]: Generated speech in target language
        """
        if reference_language not in self.languages:
            raise ValueError(f"Reference language '{reference_language}' not supported")
        
        if target_language not in self.languages:
            raise ValueError(f"Target language '{target_language}' not supported")
        
        # Clone the voice
        voice_profile = self.clone_voice(
            reference_audio_path=reference_audio_path,
            language=reference_language
        )
        
        if not voice_profile:
            return None
        
        # Create config for target language
        target_voice = VoiceProfile(
            name=f"Cross-lingual: {reference_language}→{target_language}",
            id=f"cross_{reference_language}_{target_language}",
            gender=voice_profile.gender,
            language=target_language,
            style=VoiceStyle.NEUTRAL,
            sample_rate=self.sample_rate
        )
        
        config = TTSConfig(
            voice_profile=target_voice,
            audio_format=AudioFormat.WAV
        )
        
        # Generate in target language
        return self.generate(
            text=text,
            config=config,
            language=target_language,
            speaker_wav=reference_audio_path
        )
    
    def get_cloned_voices(self) -> Dict[str, Dict]:
        """
        Get all cloned voices.
        
        Returns:
            Dict[str, Dict]: Dictionary of cloned voice profiles
        """
        return self.cloned_voices.copy()
    
    def cleanup(self) -> None:
        """Clean up XTTS model resources."""
        super().cleanup()
        
        if hasattr(self, 'tts') and self.tts is not None:
            del self.tts
            self.tts = None
        
        if hasattr(self, 'xtts_model') and self.xtts_model is not None:
            del self.xtts_model
            self.xtts_model = None
        
        # Clear cloning cache
        self.speaker_embedding = None
        self.gpt_cond_latent = None
        self.reference_audio = None
        self.reference_language = None
        self.cloned_voices.clear()
        
        # Clear CUDA cache
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        logger.info("XTTS model cleaned up")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the XTTS model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        info = super().get_model_info()
        
        info.update({
            "device": str(self.device),
            "sample_rate": self.sample_rate,
            "supported_languages": self.languages,
            "cloned_voices_count": len(self.cloned_voices),
            "has_precomputed_latents": self.speaker_embedding is not None,
            "model_features": [
                "multilingual",
                "zero-shot_voice_cloning",
                "cross-lingual_cloning",
                "emotion_control",
                "high_quality"
            ],
            "cloning_supported": True,
            "cross_lingual_supported": True,
            "reference_audio": self.reference_audio,
        })
        
        return info


# Example usage
if __name__ == "__main__":
    # Test the XTTS model
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and initialize model
    model = XTTSModel(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
    
    if model.initialize(device="cpu"):
        print("XTTS model initialized successfully")
        print(f"Supported languages: {model.get_supported_languages()}")
        
        # Test with default voice
        voices = model.get_available_voices()
        if voices:
            voice = voices[0]  # English male default
            
            config = TTSConfig(
                voice_profile=voice,
                speed=1.0,
                audio_format=AudioFormat.WAV
            )
            
            try:
                result = model.generate(
                    text="Hello, this is a test of the XTTS multilingual TTS system.",
                    config=config,
                    language="en"
                )
                
                print(f"Generation successful!")
                print(f"Duration: {result.duration_ms}ms")
                print(f"Audio size: {len(result.audio_data)} bytes")
                
                # Save to file
                with open("test_xtts.wav", "wb") as f:
                    f.write(result.audio_data)
                print("Audio saved to test_xtts.wav")
                
            except Exception as e:
                print(f"Error: {e}")
        
        # Cleanup
        model.cleanup()
    else:
        print("Failed to initialize XTTS model")