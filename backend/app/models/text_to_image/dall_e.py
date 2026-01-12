"""
DALL-E model implementation for text-to-image generation.

DALL-E models are advanced text-to-image models from OpenAI that can generate
high-quality images from text descriptions with remarkable creativity and coherence.
"""

import base64
import io
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from .base import (
    BaseTextToImageModel, ImageGenerationResult, ImageGenerationConfig,
    ImageTaskType, ImageGenerationParameters, ImageFormat, ModelDevice
)

logger = logging.getLogger(__name__)


class DalleModel(BaseTextToImageModel):
    """
    DALL-E model wrapper for WorldBrief360.
    
    DALL-E models are state-of-the-art text-to-image models with:
    - Remarkable image quality and coherence
    - Strong understanding of complex prompts
    - Ability to combine concepts in novel ways
    - Support for various artistic styles
    - Good handling of text in images
    
    Model variants:
    - DALL-E 2: 3.5B parameters, 1024x1024 resolution
    - DALL-E 3: More advanced, better prompt understanding
    """
    
    # DALL-E model variants and their specifications
    MODEL_INFO = {
        # OpenAI API models (requires API key)
        "dall-e-2": {
            "description": "DALL-E 2 via OpenAI API",
            "max_tokens": 1000,
            "architecture": "diffusion",
            "parameters": 3_500_000_000,
            "performance": "excellent",
            "memory_mb": 0,  # API-based, no local memory
            "tasks": ["text_to_image", "image_variation", "inpainting", "outpainting"],
            "supported_sizes": ["256x256", "512x512", "1024x1024"],
            "max_images_per_request": 10,
            "requires_api": True,
            "api_endpoint": "https://api.openai.com/v1/images/generations",
        },
        "dall-e-3": {
            "description": "DALL-E 3 via OpenAI API",
            "max_tokens": 4000,
            "architecture": "diffusion",
            "parameters": "unknown",  # Not publicly disclosed
            "performance": "state_of_the_art",
            "memory_mb": 0,  # API-based
            "tasks": ["text_to_image", "image_editing"],
            "supported_sizes": ["1024x1024", "1024x1792", "1792x1024"],
            "styles": ["vivid", "natural"],
            "quality": ["standard", "hd"],
            "max_images_per_request": 1,
            "requires_api": True,
            "api_endpoint": "https://api.openai.com/v1/images/generations",
        },
        
        # Open-source implementations (when available)
        "openai/dall-e-mini": {
            "description": "DALL-E Mini (open-source implementation)",
            "max_tokens": 256,
            "architecture": "vqgan",
            "parameters": 400_000_000,
            "performance": "good",
            "memory_mb": 2000,
            "tasks": ["text_to_image"],
            "supported_sizes": ["256x256"],
            "requires_api": False,
        },
        "openai/dall-e-mega": {
            "description": "DALL-E Mega (larger open-source implementation)",
            "max_tokens": 256,
            "architecture": "vqgan",
            "parameters": 1_300_000_000,
            "performance": "very_good",
            "memory_mb": 4000,
            "tasks": ["text_to_image"],
            "supported_sizes": ["256x256"],
            "requires_api": False,
        },
        
        # Community models
        "kakaobrain/minDALL-E": {
            "description": "minDALL-E from Kakao Brain",
            "max_tokens": 256,
            "architecture": "transformer",
            "parameters": 1_300_000_000,
            "performance": "good",
            "memory_mb": 3000,
            "tasks": ["text_to_image"],
            "supported_sizes": ["256x256"],
            "requires_api": False,
        },
    }
    
    # Style presets for DALL-E 3
    STYLE_PRESETS = {
        "vivid": "hyper-realistic, dramatic lighting, vibrant colors",
        "natural": "realistic, natural lighting, authentic colors",
        "artistic": "painterly, artistic, brush strokes visible",
        "photographic": "photorealistic, detailed, professional photography",
        "cartoon": "animated, cartoon style, bold outlines",
        "watercolor": "watercolor painting, soft edges, translucent colors",
        "oil_painting": "oil painting, textured brush strokes, rich colors",
        "sketch": "sketch, pencil drawing, rough lines",
        "digital_art": "digital art, clean lines, vibrant colors",
        "minimalist": "minimalist, simple, clean design",
        "fantasy": "fantasy art, magical, imaginative",
        "sci_fi": "sci-fi, futuristic, technological",
        "retro": "retro, vintage, nostalgic",
        "cyberpunk": "cyberpunk, neon, futuristic dystopia",
        "steampunk": "steampunk, Victorian, mechanical",
    }
    
    # Quality presets for DALL-E 3
    QUALITY_PRESETS = {
        "standard": "standard quality, good detail",
        "hd": "high definition, extremely detailed",
        "ultra": "ultra detailed, photorealistic",
    }
    
    # Prompt enhancement templates
    PROMPT_ENHANCEMENTS = {
        "photorealistic": "photorealistic, 8k, ultra detailed, professional photography",
        "artistic": "masterpiece, trending on artstation, elegant, highly detailed",
        "cinematic": "cinematic, dramatic lighting, film grain, movie still",
        "concept_art": "concept art, digital painting, matte painting, environmental design",
        "illustration": "illustration, children's book illustration, whimsical, colorful",
        "product": "product photography, studio lighting, clean background, professional",
        "portrait": "portrait photography, professional model, detailed face, expressive",
        "landscape": "landscape photography, golden hour, panoramic, breathtaking",
        "macro": "macro photography, extremely detailed, shallow depth of field",
        "dark": "dark, moody, dramatic, chiaroscuro lighting",
        "bright": "bright, cheerful, vibrant, happy",
    }
    
    def __init__(
        self,
        model_name: str = "dall-e-3",
        config: Optional[ImageGenerationConfig] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize DALL-E model.
        
        Args:
            model_name: Name of the DALL-E model
            config: Configuration for image generation
            api_key: OpenAI API key (required for OpenAI models)
            api_base: Custom API base URL
            **kwargs: Additional arguments
        """
        # Update config with DALL-E-specific defaults
        if config is None:
            config = ImageGenerationConfig()
        
        # Get model info
        model_info = self.MODEL_INFO.get(model_name, {})
        requires_api = model_info.get("requires_api", True)
        
        # For API models, we need an API key
        if requires_api and not api_key:
            logger.warning(
                f"Model {model_name} requires an OpenAI API key. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )
        
        super().__init__(model_name, config, **kwargs)
        
        # DALL-E specific initialization
        self.api_key = api_key
        self.api_base = api_base or "https://api.openai.com/v1"
        
        # Check model type
        self._requires_api = requires_api
        self._is_dalle2 = "2" in model_name
        self._is_dalle3 = "3" in model_name
        self._is_open_source = not requires_api
        
        # API client (for OpenAI models)
        self._client = None
        
        # For open-source models
        self.model = None
        self.tokenizer = None
        self.processor = None
        
        # Generation parameters optimized for DALL-E
        self._default_generation_params = {
            "num_images": 1,
            "quality": "standard" if self._is_dalle3 else None,
            "style": "vivid" if self._is_dalle3 else None,
            "response_format": "url",  # or "b64_json"
        }
        
        logger.info(f"Initialized DALL-E model: {model_name} "
                   f"(API: {self._requires_api}, DALL-E 2: {self._is_dalle2}, DALL-E 3: {self._is_dalle3})")
    
    def load(self) -> None:
        """
        Load the DALL-E model or API client.
        
        Raises:
            RuntimeError: If model loading fails
        """
        if self._is_loaded:
            logger.warning(f"Model {self.model_name} is already loaded")
            return
        
        try:
            if self._requires_api:
                # Load API client
                logger.info(f"Setting up OpenAI API client for {self.model_name}")
                self._setup_api_client()
            else:
                # Load open-source model
                logger.info(f"Loading open-source DALL-E model: {self.model_name}")
                self._load_open_source_model()
            
            self._is_loaded = True
            logger.info(f"Successfully loaded DALL-E model {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load DALL-E model {self.model_name}: {e}")
            raise RuntimeError(f"Failed to load DALL-E model {self.model_name}: {e}")
    
    def _setup_api_client(self) -> None:
        """Set up OpenAI API client."""
        try:
            import openai
            
            if self.api_key:
                openai.api_key = self.api_key
            elif hasattr(openai, 'api_key'):
                # Check if API key is already set
                if not openai.api_key:
                    raise ValueError("OpenAI API key is required for DALL-E API models")
            
            # Set API base if provided
            if self.api_base and self.api_base != "https://api.openai.com/v1":
                openai.api_base = self.api_base
            
            self._client = openai
            
        except ImportError:
            raise ImportError(
                "OpenAI Python package is required for DALL-E API models. "
                "Install with: pip install openai"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to set up OpenAI API client: {e}")
    
    def _load_open_source_model(self) -> None:
        """Load open-source DALL-E implementation."""
        model_name = self.model_name
        
        if "minDALL-E" in model_name:
            try:
                from mindallele import MinDalle # type: ignore
                
                logger.info(f"Loading minDALL-E model")
                
                # Determine device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # Load model
                self.model = MinDalle(
                    models_root='./pretrained',
                    dtype=torch.float32,
                    device=device,
                    is_mega='mega' in model_name,
                    is_reusable=True,
                    is_verbose=False
                )
                
                logger.info("Successfully loaded minDALL-E model")
                
            except ImportError:
                raise ImportError(
                    "minDALL-E is not installed. Install with: "
                    "pip install git+https://github.com/kakaobrain/min-dalle.git"
                )
        
        elif "dall-e" in model_name.lower() and ("mini" in model_name.lower() or "mega" in model_name.lower()):
            try:
                from dalle_pytorch import DALLE # type: ignore
                from dalle_pytorch.tokenizer import tokenizer # type: ignore
                
                logger.info(f"Loading DALL-E {model_name}")
                
                # Load tokenizer
                self.tokenizer = tokenizer
                
                # Determine model size
                is_mega = "mega" in model_name.lower()
                
                # Create model configuration
                # Note: This is simplified - actual loading would require weights
                logger.warning(f"DALL-E {model_name} weights need to be downloaded separately")
                
            except ImportError:
                raise ImportError(
                    "DALL-E PyTorch is not installed. Install with: "
                    "pip install dalle-pytorch"
                )
        
        else:
            raise ValueError(f"Unsupported open-source DALL-E model: {model_name}")
    
    def generate_image(
        self,
        prompt: str,
        config: Optional[ImageGenerationConfig] = None,
        **kwargs
    ) -> ImageGenerationResult:
        """
        Generate image from text prompt using DALL-E.
        
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
        if not self._is_loaded:
            raise ValueError(f"Model {self.model_name} is not loaded. Call load() first.")
        
        if config is None:
            config = self.config
        
        try:
            start_time = time.time()
            
            # Prepare prompt
            enhanced_prompt = self._enhance_prompt(prompt, **kwargs)
            
            # Prepare generation parameters
            gen_params = self._prepare_generation_params(config, **kwargs)
            
            # Generate images
            if self._requires_api:
                images, metadata = self._generate_via_api(enhanced_prompt, gen_params)
            else:
                images, metadata = self._generate_locally(enhanced_prompt, gen_params)
            
            # Perform safety check if configured
            if config.safety_check:
                images, safety_scores = self._perform_safety_check(images)
                metadata["safety_scores"] = safety_scores
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Track performance
            self._processing_times.append(processing_time_ms)
            self._total_images_generated += len(images)
            
            # Prepare result
            result = ImageGenerationResult(
                images=images,
                model_name=self.model_name,
                model_version=self.get_version(),
                task_type=config.task_type,
                processing_time_ms=processing_time_ms,
                images_generated=len(images),
                input_prompt=prompt,
                input_parameters=gen_params,
                metadata={
                    "enhanced_prompt": enhanced_prompt,
                    "generation_params": gen_params,
                    "is_dalle2": self._is_dalle2,
                    "is_dalle3": self._is_dalle3,
                    "requires_api": self._requires_api,
                    "model_info": self.MODEL_INFO.get(self.model_name, {}),
                    **metadata,
                }
            )
            
            # Save images if configured
            if config.save_to_file and config.output_dir:
                saved_paths = result.save_images(config.output_dir, prefix="dalle")
                result.metadata["saved_paths"] = saved_paths
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate image with DALL-E: {e}")
            raise RuntimeError(f"Failed to generate image with DALL-E: {e}")
    
    def _enhance_prompt(self, prompt: str, **kwargs) -> str:
        """
        Enhance prompt with style, quality, and other improvements.
        
        Args:
            prompt: Original prompt
            **kwargs: Enhancement parameters
            
        Returns:
            Enhanced prompt
        """
        enhanced = prompt
        
        # Apply style if specified
        style = kwargs.get('style')
        if style and style in self.STYLE_PRESETS:
            style_text = self.STYLE_PRESETS[style]
            enhanced = f"{enhanced}, {style_text}"
        
        # Apply quality enhancement
        quality = kwargs.get('quality')
        if quality and quality in self.QUALITY_PRESETS:
            quality_text = self.QUALITY_PRESETS[quality]
            enhanced = f"{enhanced}, {quality_text}"
        
        # Apply additional enhancements
        enhancement = kwargs.get('enhancement')
        if enhancement and enhancement in self.PROMPT_ENHANCEMENTS:
            enhancement_text = self.PROMPT_ENHANCEMENTS[enhancement]
            enhanced = f"{enhancement_text}, {enhanced}"
        
        # For DALL-E 3, we can add more specific instructions
        if self._is_dalle3:
            # Add detailed description if not already detailed
            if len(enhanced.split()) < 30:
                enhanced = f"A detailed image of {enhanced}"
            
            # Ensure prompt is within token limits
            max_tokens = self.MODEL_INFO.get(self.model_name, {}).get("max_tokens", 1000)
            if len(enhanced) > max_tokens * 4:  # Rough estimate
                enhanced = enhanced[:max_tokens * 4]
        
        return enhanced
    
    def _prepare_generation_params(self, config: ImageGenerationConfig, **kwargs) -> Dict[str, Any]:
        """
        Prepare generation parameters.
        
        Args:
            config: Image generation configuration
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of generation parameters
        """
        params = self._default_generation_params.copy()
        
        # Update with config
        if config.generation_params:
            params.update({
                "num_images": config.generation_params.num_images,
                "width": config.generation_params.width,
                "height": config.generation_params.height,
                "seed": config.generation_params.seed,
                "negative_prompt": config.generation_params.negative_prompt,
            })
        
        # Update with kwargs
        params.update(kwargs)
        
        # Validate parameters
        if self._is_dalle2:
            # DALL-E 2 has specific size requirements
            size = f"{params['width']}x{params['height']}"
            supported_sizes = self.MODEL_INFO.get(self.model_name, {}).get("supported_sizes", [])
            if size not in supported_sizes:
                # Use closest supported size
                params['width'], params['height'] = 1024, 1024
        
        elif self._is_dalle3:
            # DALL-E 3 has specific size and quality requirements
            if 'quality' not in params:
                params['quality'] = 'standard'
            if 'style' not in params:
                params['style'] = 'vivid'
            
            # Limit number of images
            max_images = self.MODEL_INFO.get(self.model_name, {}).get("max_images_per_request", 1)
            if params['num_images'] > max_images:
                params['num_images'] = max_images
        
        return params
    
    def _generate_via_api(self, prompt: str, params: Dict[str, Any]) -> Tuple[List[Image.Image], Dict[str, Any]]:
        """
        Generate images via OpenAI API.
        
        Args:
            prompt: Enhanced prompt
            params: Generation parameters
            
        Returns:
            Tuple of (images, metadata)
        """
        import openai
        
        try:
            # Prepare API request
            request_params = {
                "prompt": prompt,
                "n": params['num_images'],
                "size": f"{params['width']}x{params['height']}",
                "response_format": params['response_format'],
            }
            
            # Add DALL-E 3 specific parameters
            if self._is_dalle3:
                request_params["model"] = "dall-e-3"
                request_params["quality"] = params['quality']
                request_params["style"] = params['style']
            else:
                request_params["model"] = "dall-e-2"
            
            # Make API request
            logger.info(f"Making DALL-E API request with prompt: {prompt[:100]}...")
            
            response = openai.images.generate(**request_params)
            
            # Process response
            images = []
            metadata = {
                "api_response": response.to_dict() if hasattr(response, 'to_dict') else str(response),
                "created": getattr(response, 'created', None),
                "revised_prompt": getattr(response, 'revised_prompt', None),
            }
            
            # Extract images
            for item in response.data:
                if params['response_format'] == "b64_json":
                    # Decode base64 image
                    image_data = base64.b64decode(item.b64_json)
                    image = Image.open(io.BytesIO(image_data))
                    images.append(image)
                else:
                    # URL response (we can't download without additional requests)
                    # For now, store the URL
                    images.append(item.url)
            
            return images, metadata
            
        except openai.OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"OpenAI API error: {e}")
        except Exception as e:
            logger.error(f"Failed to generate images via API: {e}")
            raise RuntimeError(f"Failed to generate images via API: {e}")
    
    def _generate_locally(self, prompt: str, params: Dict[str, Any]) -> Tuple[List[Image.Image], Dict[str, Any]]:
        """
        Generate images using local open-source model.
        
        Args:
            prompt: Enhanced prompt
            params: Generation parameters
            
        Returns:
            Tuple of (images, metadata)
        """
        if self.model_name == "kakaobrain/minDALL-E":
            return self._generate_with_mindalle(prompt, params)
        elif "dall-e" in self.model_name.lower():
            return self._generate_with_dalle_pytorch(prompt, params)
        else:
            raise ValueError(f"Unsupported local model: {self.model_name}")
    
    def _generate_with_mindalle(self, prompt: str, params: Dict[str, Any]) -> Tuple[List[Image.Image], Dict[str, Any]]:
        """
        Generate images using minDALL-E.
        
        Args:
            prompt: Enhanced prompt
            params: Generation parameters
            
        Returns:
            Tuple of (images, metadata)
        """
        try:
            # Generate images
            logger.info(f"Generating with minDALL-E: {prompt[:100]}...")
            
            # Prepare generation parameters
            seed = params.get('seed')
            if seed is not None:
                torch.manual_seed(seed)
            
            # Generate
            images = self.model.generate_images(
                text=prompt,
                seed=seed,
                grid_size=1,  # Single image
                progressive_outputs=False,
                top_k=256,
                temperature=1.0,
                supercondition_factor=16.0,
            )
            
            # Convert to PIL Images
            pil_images = []
            for img in images:
                # Convert tensor to PIL
                if isinstance(img, torch.Tensor):
                    img_np = img.cpu().numpy().transpose(1, 2, 0)
                    img_np = (img_np * 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_np)
                else:
                    pil_img = img
                pil_images.append(pil_img)
            
            metadata = {
                "model": "minDALL-E",
                "seed": seed,
                "grid_size": 1,
            }
            
            return pil_images, metadata
            
        except Exception as e:
            logger.error(f"Failed to generate with minDALL-E: {e}")
            raise RuntimeError(f"Failed to generate with minDALL-E: {e}")
    
    def _generate_with_dalle_pytorch(self, prompt: str, params: Dict[str, Any]) -> Tuple[List[Image.Image], Dict[str, Any]]:
        """
        Generate images using DALL-E PyTorch.
        
        Args:
            prompt: Enhanced prompt
            params: Generation parameters
            
        Returns:
            Tuple of (images, metadata)
        """
        # This is a placeholder - actual implementation would depend on the specific model
        logger.warning(f"DALL-E PyTorch generation not fully implemented for {self.model_name}")
        
        # Create a placeholder image
        width, height = params.get('width', 256), params.get('height', 256)
        placeholder = Image.new('RGB', (width, height), color='gray')
        
        metadata = {
            "model": "DALL-E PyTorch",
            "note": "Placeholder implementation - requires model weights",
        }
        
        return [placeholder], metadata
    
    def edit_image(
        self,
        image_path: Union[str, Path],
        prompt: str,
        mask_path: Optional[Union[str, Path]] = None,
        config: Optional[ImageGenerationConfig] = None,
        **kwargs
    ) -> ImageGenerationResult:
        """
        Edit an existing image using DALL-E's editing capabilities.
        
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
        if not self._requires_api or not self._is_dalle2:
            raise ValueError(f"Model {self.model_name} does not support image editing via API")
        
        try:
            import openai
            
            # Load image
            image = self._load_image(image_path)
            
            # Convert image to bytes
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            image_bytes = buffered.getvalue()
            
            # Prepare request
            request_data = {
                "image": image_bytes,
                "prompt": prompt,
                "n": 1,
                "size": f"{image.width}x{image.height}",
                "response_format": "b64_json",
            }
            
            # Add mask if provided
            if mask_path:
                mask = self._load_image(mask_path)
                mask_buffered = io.BytesIO()
                mask.save(mask_buffered, format="PNG")
                mask_bytes = mask_buffered.getvalue()
                request_data["mask"] = mask_bytes
            
            # Make API request
            logger.info(f"Editing image with DALL-E: {prompt[:100]}...")
            
            response = openai.images.edit(**request_data)
            
            # Process response
            images = []
            for item in response.data:
                image_data = base64.b64decode(item.b64_json)
                edited_image = Image.open(io.BytesIO(image_data))
                images.append(edited_image)
            
            # Create result
            if config is None:
                config = self.config.copy()
            config.task_type = ImageTaskType.INPAINTING if mask_path else ImageTaskType.IMAGE_TO_IMAGE
            
            result = ImageGenerationResult(
                images=images,
                model_name=self.model_name,
                model_version=self.get_version(),
                task_type=config.task_type,
                processing_time_ms=0,  # Would need timing
                images_generated=len(images),
                input_prompt=prompt,
                input_parameters={"has_mask": mask_path is not None},
                metadata={
                    "api_response": response.to_dict() if hasattr(response, 'to_dict') else str(response),
                    "original_image_size": image.size,
                    "mask_provided": mask_path is not None,
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to edit image with DALL-E: {e}")
            raise RuntimeError(f"Failed to edit image with DALL-E: {e}")
    
    def create_variations(
        self,
        image_path: Union[str, Path],
        num_variations: int = 4,
        config: Optional[ImageGenerationConfig] = None,
        **kwargs
    ) -> ImageGenerationResult:
        """
        Create variations of an existing image using DALL-E.
        
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
        if not self._requires_api or not self._is_dalle2:
            raise ValueError(f"Model {self.model_name} does not support image variations via API")
        
        try:
            import openai
            
            # Load image
            image = self._load_image(image_path)
            
            # Convert image to bytes
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            image_bytes = buffered.getvalue()
            
            # Prepare request
            request_data = {
                "image": image_bytes,
                "n": min(num_variations, 10),  # API limit
                "size": f"{image.width}x{image.height}",
                "response_format": "b64_json",
            }
            
            # Make API request
            logger.info(f"Creating image variations with DALL-E")
            
            response = openai.images.create_variation(**request_data)
            
            # Process response
            images = []
            for item in response.data:
                image_data = base64.b64decode(item.b64_json)
                variation = Image.open(io.BytesIO(image_data))
                images.append(variation)
            
            # Create result
            if config is None:
                config = self.config.copy()
            config.task_type = ImageTaskType.IMAGE_VARIATION
            
            result = ImageGenerationResult(
                images=images,
                model_name=self.model_name,
                model_version=self.get_version(),
                task_type=config.task_type,
                processing_time_ms=0,  # Would need timing
                images_generated=len(images),
                input_prompt="Image variations",
                input_parameters={"num_variations": num_variations},
                metadata={
                    "api_response": response.to_dict() if hasattr(response, 'to_dict') else str(response),
                    "original_image_size": image.size,
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to create image variations with DALL-E: {e}")
            raise RuntimeError(f"Failed to create image variations with DALL-E: {e}")
    
    def unload(self) -> None:
        """Unload the model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        self._client = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._is_loaded = False
        logger.info(f"Unloaded DALL-E model {self.model_name}")
    
    def get_model_capabilities(self) -> Dict[str, Any]:
        """
        Get capabilities of the DALL-E model.
        
        Returns:
            Dictionary with model capabilities
        """
        model_info = self.MODEL_INFO.get(self.model_name, {}).copy()
        
        capabilities = {
            "model": self.model_name,
            "architecture": model_info.get("architecture", "unknown"),
            "tasks": model_info.get("tasks", []),
            "supported_sizes": model_info.get("supported_sizes", []),
            "requires_api": self._requires_api,
            "is_dalle2": self._is_dalle2,
            "is_dalle3": self._is_dalle3,
            "is_open_source": self._is_open_source,
            "supports_text_to_image": True,
            "supports_img2img": self._is_dalle2 and self._requires_api,
            "supports_inpainting": self._is_dalle2 and self._requires_api,
            "supports_variations": self._is_dalle2 and self._requires_api,
            "supports_upscaling": False,
            "supports_controlnet": False,
            "max_image_size": (1024, 1024) if self._is_dalle3 else (1024, 1024),
            "min_image_size": (256, 256),
            "aspect_ratios": ["1:1", "16:9", "9:16"] if self._is_dalle3 else ["1:1"],
            "multilingual_prompts": True,
            "model_parameters": model_info.get("parameters", 0),
            "memory_requirements_mb": model_info.get("memory_mb", 0),
            "recommended_batch_size": 1,
            "max_images_per_request": model_info.get("max_images_per_request", 1),
            "style_presets": list(self.STYLE_PRESETS.keys()) if self._is_dalle3 else [],
            "quality_presets": list(self.QUALITY_PRESETS.keys()) if self._is_dalle3 else [],
        }
        
        return capabilities
    
    def get_version(self) -> str:
        """
        Get model version.
        
        Returns:
            Model version string
        """
        if self._is_dalle3:
            return "dall-e-3"
        elif self._is_dalle2:
            return "dall-e-2"
        elif "minDALL-E" in self.model_name:
            return "mindalle"
        elif "mini" in self.model_name.lower():
            return "dall-e-mini"
        elif "mega" in self.model_name.lower():
            return "dall-e-mega"
        else:
            return "dall-e"
    
    def list_style_presets(self) -> List[str]:
        """
        List available style presets (DALL-E 3 only).
        
        Returns:
            List of style preset names
        """
        if self._is_dalle3:
            return list(self.STYLE_PRESETS.keys())
        else:
            return []
    
    def list_quality_presets(self) -> List[str]:
        """
        List available quality presets (DALL-E 3 only).
        
        Returns:
            List of quality preset names
        """
        if self._is_dalle3:
            return list(self.QUALITY_PRESETS.keys())
        else:
            return []
    
    def list_prompt_enhancements(self) -> List[str]:
        """
        List available prompt enhancement templates.
        
        Returns:
            List of enhancement names
        """
        return list(self.PROMPT_ENHANCEMENTS.keys())
    
    def add_style_preset(self, name: str, description: str) -> None:
        """
        Add a custom style preset.
        
        Args:
            name: Name of the style preset
            description: Style description
        """
        self.STYLE_PRESETS[name] = description
        logger.info(f"Added style preset: {name}")
    
    def add_prompt_enhancement(self, name: str, enhancement: str) -> None:
        """
        Add a custom prompt enhancement template.
        
        Args:
            name: Name of the enhancement
            enhancement: Enhancement text
        """
        self.PROMPT_ENHANCEMENTS[name] = enhancement
        logger.info(f"Added prompt enhancement: {name}")
    
    def set_api_key(self, api_key: str) -> None:
        """
        Set OpenAI API key.
        
        Args:
            api_key: OpenAI API key
        """
        self.api_key = api_key
        logger.info("Updated OpenAI API key")
        
        # Update client if already loaded
        if self._is_loaded and self._client:
            try:
                import openai
                openai.api_key = api_key
            except:
                pass
    
    def get_usage_info(self) -> Dict[str, Any]:
        """
        Get API usage information (for API models only).
        
        Returns:
            Dictionary with usage information
            
        Raises:
            ValueError: If model is not API-based
        """
        if not self._requires_api:
            raise ValueError(f"Model {self.model_name} is not API-based")
        
        # This would require additional API calls to get usage stats
        # For now, return placeholder
        return {
            "model": self.model_name,
            "requires_api": True,
            "note": "Usage information requires additional API calls",
        }