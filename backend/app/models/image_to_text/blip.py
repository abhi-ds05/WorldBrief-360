"""
BLIP (Bootstrapping Language-Image Pre-training) model implementation.
BLIP models are excellent for image captioning and visual question answering.
"""
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import (
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
    BlipProcessor,
    BlipConfig,
)

from .base import (
    BaseImageToTextModel, ImageTextResult, ImageCaptionConfig,
    ImageTaskType, ImageFormat
)
from ...base import ModelDevice

logger = logging.getLogger(__name__)


class BLIPModel(BaseImageToTextModel):
    """
    BLIP (Bootstrapping Language-Image Pre-training) model wrapper.
    
    BLIP models are state-of-the-art for image captioning and VQA tasks.
    They combine vision and language understanding through bootstrapped pre-training.
    """
    
    # BLIP model variants and their specifications
    MODEL_INFO = {
        # Base models for captioning
        "Salesforce/blip-image-captioning-base": {
            "type": "captioning",
            "description": "BLIP base model for image captioning",
            "max_tokens": 32,
            "image_size": 384,
            "parameters": 224_000_000,
            "performance": "fast",
            "memory_mb": 900,
            "tasks": ["captioning"],
        },
        "Salesforce/blip-image-captioning-large": {
            "type": "captioning",
            "description": "BLIP large model for detailed image captioning",
            "max_tokens": 64,
            "image_size": 384,
            "parameters": 446_000_000,
            "performance": "accurate",
            "memory_mb": 1800,
            "tasks": ["captioning"],
        },
        
        # VQA models
        "Salesforce/blip-vqa-base": {
            "type": "vqa",
            "description": "BLIP base model for visual question answering",
            "max_tokens": 32,
            "image_size": 384,
            "parameters": 224_000_000,
            "performance": "fast",
            "memory_mb": 900,
            "tasks": ["vqa"],
        },
        "Salesforce/blip-vqa-capfilt-large": {
            "type": "vqa",
            "description": "BLIP large VQA model with caption filtering",
            "max_tokens": 64,
            "image_size": 384,
            "parameters": 446_000_000,
            "performance": "accurate",
            "memory_mb": 1800,
            "tasks": ["vqa"],
        },
        
        # Multitask models
        "Salesforce/blip-base": {
            "type": "multitask",
            "description": "BLIP base model for multiple vision-language tasks",
            "max_tokens": 32,
            "image_size": 384,
            "parameters": 224_000_000,
            "performance": "balanced",
            "memory_mb": 900,
            "tasks": ["captioning", "vqa", "retrieval"],
        },
        "Salesforce/blip-large": {
            "type": "multitask",
            "description": "BLIP large model for multiple vision-language tasks",
            "max_tokens": 64,
            "image_size": 384,
            "parameters": 446_000_000,
            "performance": "accurate",
            "memory_mb": 1800,
            "tasks": ["captioning", "vqa", "retrieval"],
        },
        
        # Fine-tuned models
        "Salesforce/blip-image-captioning-base-flickr": {
            "type": "captioning",
            "description": "BLIP base fine-tuned on Flickr30k",
            "max_tokens": 32,
            "image_size": 384,
            "parameters": 224_000_000,
            "performance": "fast",
            "memory_mb": 900,
            "tasks": ["captioning"],
        },
        "Salesforce/blip-itm-base-coco": {
            "type": "retrieval",
            "description": "BLIP base fine-tuned for image-text matching on COCO",
            "max_tokens": 32,
            "image_size": 384,
            "parameters": 224_000_000,
            "performance": "fast",
            "memory_mb": 900,
            "tasks": ["retrieval", "matching"],
        },
    }
    
    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        config: Optional[ImageCaptionConfig] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        processor_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize BLIP model.
        
        Args:
            model_name: Name of the BLIP model
            config: Configuration for image captioning
            model_kwargs: Additional arguments for model initialization
            processor_kwargs: Additional arguments for processor initialization
            **kwargs: Additional arguments passed to configuration
        """
        # Update config with BLIP-specific defaults
        if config is None:
            config = ImageCaptionConfig()
        
        # Update with model-specific defaults
        model_info = self.MODEL_INFO.get(model_name, {})
        config.image_size = (model_info.get("image_size", 384), model_info.get("image_size", 384))
        
        # Set task type based on model name
        if "vqa" in model_name.lower():
            config.task_type = ImageTaskType.VQA
        elif "captioning" in model_name.lower():
            config.task_type = ImageTaskType.CAPTIONING
        
        super().__init__(model_name, config, **kwargs)
        
        # BLIP-specific initialization
        self.model_kwargs = model_kwargs or {}
        self.processor_kwargs = processor_kwargs or {}
        
        # Model type detection
        self._model_type = self._detect_model_type()
        
        # Generation parameters
        self._default_generation_params = {
            "max_length": self.config.generation_config.max_new_tokens,
            "min_length": self.config.generation_config.min_new_tokens,
            "temperature": self.config.generation_config.temperature,
            "top_p": self.config.generation_config.top_p,
            "top_k": self.config.generation_config.top_k,
            "repetition_penalty": self.config.generation_config.repetition_penalty,
            "do_sample": self.config.generation_config.do_sample,
            "num_beams": self.config.generation_config.num_beams,
            "length_penalty": self.config.generation_config.length_penalty,
            "no_repeat_ngram_size": self.config.generation_config.no_repeat_ngram_size,
            "early_stopping": self.config.generation_config.early_stopping,
        }
        
        logger.info(f"Initialized BLIP model: {model_name} (type: {self._model_type})")
    
    def _detect_model_type(self) -> str:
        """Detect BLIP model type from name."""
        model_name_lower = self.model_name.lower()
        
        if "vqa" in model_name_lower:
            return "vqa"
        elif "captioning" in model_name_lower:
            return "captioning"
        elif "itm" in model_name_lower:
            return "itm"  # Image-Text Matching
        elif "base" in model_name_lower or "large" in model_name_lower:
            return "multitask"
        else:
            return "unknown"
    
    def load(self) -> None:
        """
        Load the BLIP model and processor.
        
        Raises:
            RuntimeError: If model loading fails
        """
        if self._is_loaded:
            logger.warning(f"Model {self.model_name} is already loaded")
            return
        
        try:
            logger.info(f"Loading BLIP model: {self.model_name}")
            
            # Determine device
            if torch.cuda.is_available():
                self._device = "cuda"
                device_count = torch.cuda.device_count()
                if device_count > 1:
                    logger.info(f"Found {device_count} GPUs, using first GPU")
            else:
                self._device = "cpu"
            
            # Load processor first
            logger.info(f"Loading BLIP processor for {self.model_name}")
            self.processor = BlipProcessor.from_pretrained(
                self.model_name,
                **self.processor_kwargs
            )
            
            # Load appropriate model based on detected type
            logger.info(f"Loading BLIP model for {self.model_name}")
            
            if self._model_type == "vqa":
                self.model = BlipForQuestionAnswering.from_pretrained(
                    self.model_name,
                    **self.model_kwargs
                )
            else:  # captioning, multitask, or unknown
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.model_name,
                    **self.model_kwargs
                )
            
            # Move model to device
            self.model.to(self._device)
            self.model.eval()  # Set to evaluation mode
            
            self._is_loaded = True
            
            # Update image size from processor
            if hasattr(self.processor, 'image_processor'):
                image_processor = self.processor.image_processor
                if hasattr(image_processor, 'size'):
                    size = image_processor.size
                    if isinstance(size, dict):
                        size = size.get('shortest_edge', 384)
                    elif isinstance(size, (list, tuple)):
                        size = min(size)
                    self.image_size = (size, size)
                    logger.info(f"Updated image size from processor: {self.image_size}")
            
            logger.info(f"Successfully loaded BLIP model {self.model_name} on device {self._device}")
            
        except Exception as e:
            logger.error(f"Failed to load BLIP model {self.model_name}: {e}")
            raise RuntimeError(f"Failed to load BLIP model {self.model_name}: {e}")
    
    def process_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process an image for BLIP model input.
        
        Args:
            image: Input image in various formats
            **kwargs: Additional processing arguments
            
        Returns:
            Dictionary containing processed image tensors and metadata
            
        Raises:
            ValueError: If model is not loaded
            RuntimeError: If image processing fails
        """
        if not self._is_loaded:
            raise ValueError(f"Model {self.model_name} is not loaded. Call load() first.")
        
        try:
            # Load image
            pil_image = self._load_image(image)
            
            # Process with BLIP processor
            # BLIP expects raw PIL images, not pre-processed tensors
            inputs = self.processor(
                images=pil_image,
                return_tensors="pt",
                padding=True,
                **kwargs
            )
            
            # Move inputs to device
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            # Prepare result
            result = {
                'pixel_values': inputs.get('pixel_values'),
                'attention_mask': inputs.get('attention_mask'),
                'original_image': pil_image,
                'image_size': pil_image.size,
                'model_inputs': inputs,
            }
            
            # Add text inputs if provided (for VQA or conditional generation)
            if 'text' in kwargs:
                text_inputs = self.processor(
                    text=kwargs['text'],
                    return_tensors="pt",
                    padding=True
                )
                text_inputs = {k: v.to(self._device) for k, v in text_inputs.items()}
                result['text_inputs'] = text_inputs
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process image for BLIP: {e}")
            raise RuntimeError(f"Failed to process image for BLIP: {e}")
    
    def generate_text(
        self,
        image_input: Dict[str, Any],
        config: Optional[ImageCaptionConfig] = None,
        **kwargs
    ) -> ImageTextResult:
        """
        Generate text from processed image input.
        
        Args:
            image_input: Processed image input from process_image()
            config: Configuration for text generation (uses instance config if None)
            **kwargs: Additional generation arguments
            
        Returns:
            ImageTextResult containing generated text and metadata
            
        Raises:
            ValueError: If model is not loaded
            RuntimeError: If text generation fails
        """
        if not self._is_loaded:
            raise ValueError(f"Model {self.model_name} is not loaded. Call load() first.")
        
        if config is None:
            config = self.config
        
        try:
            import time
            start_time = time.time()
            
            # Extract model inputs
            pixel_values = image_input.get('pixel_values')
            if pixel_values is None:
                raise ValueError("Processed image input missing 'pixel_values'")
            
            # Prepare generation parameters
            gen_params = self._default_generation_params.copy()
            
            # Update with config
            if config.generation_config:
                gen_params.update({
                    "max_length": config.generation_config.max_new_tokens,
                    "min_length": config.generation_config.min_new_tokens,
                    "temperature": config.generation_config.temperature,
                    "top_p": config.generation_config.top_p,
                    "top_k": config.generation_config.top_k,
                    "repetition_penalty": config.generation_config.repetition_penalty,
                    "do_sample": config.generation_config.do_sample,
                    "num_beams": config.generation_config.num_beams,
                    "length_penalty": config.generation_config.length_penalty,
                    "no_repeat_ngram_size": config.generation_config.no_repeat_ngram_size,
                    "early_stopping": config.generation_config.early_stopping,
                })
            
            # Update with kwargs
            gen_params.update(kwargs)
            
            # Set seed if provided
            if config.generation_config.seed is not None:
                torch.manual_seed(config.generation_config.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(config.generation_config.seed)
            
            # Generate based on task type
            with torch.no_grad():
                if config.task_type == ImageTaskType.VQA:
                    # VQA task
                    if 'text_inputs' not in image_input:
                        raise ValueError("VQA requires question in 'text_inputs'")
                    
                    text_inputs = image_input['text_inputs']
                    input_ids = text_inputs.get('input_ids')
                    attention_mask = text_inputs.get('attention_mask')
                    
                    # Generate answer
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        **gen_params
                    )
                    
                else:
                    # Captioning or general generation
                    # Handle conditional generation with prompt
                    if config.prompt:
                        # Process prompt
                        text_inputs = self.processor(
                            text=config.prompt,
                            return_tensors="pt",
                            padding=True
                        ).to(self._device)
                        
                        input_ids = text_inputs.get('input_ids')
                        attention_mask = text_inputs.get('attention_mask')
                        
                        outputs = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            **gen_params
                        )
                    else:
                        # Unconditional generation
                        outputs = self.model.generate(
                            pixel_values=pixel_values,
                            **gen_params
                        )
            
            # Decode generated tokens
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Track performance
            self._processing_times.append(processing_time_ms)
            self._total_images_processed += 1
            
            # Prepare result
            result = ImageTextResult(
                text=generated_text.strip(),
                model_name=self.model_name,
                model_version=self.get_version(),
                input_image_info={
                    "original_size": image_input.get('image_size'),
                    "processed_size": self.image_size,
                    "format": "pil",
                },
                task_type=config.task_type,
                processing_time_ms=processing_time_ms,
                metadata={
                    "generation_params": gen_params,
                    "model_type": self._model_type,
                    "device": self._device,
                    "prompt": config.prompt,
                    "question": config.question if config.task_type == ImageTaskType.VQA else None,
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate text with BLIP: {e}")
            raise RuntimeError(f"Failed to generate text with BLIP: {e}")
    
    def caption_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        prompt: Optional[str] = None,
        **kwargs
    ) -> ImageTextResult:
        """
        Generate caption for an image using BLIP.
        
        Args:
            image: Input image in various formats
            prompt: Optional prompt/context for captioning
            **kwargs: Additional arguments for processing and generation
            
        Returns:
            ImageTextResult containing caption and metadata
        """
        # Update config with prompt
        config_dict = self.config.dict()
        config_dict['task_type'] = ImageTaskType.CAPTIONING
        if prompt is not None:
            config_dict['prompt'] = prompt
        config_dict.update(kwargs)
        config = ImageCaptionConfig(**config_dict)
        
        # Process image
        image_input = self.process_image(image, **kwargs)
        
        # Generate caption
        return self.generate_text(image_input, config)
    
    def answer_question(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        question: str,
        **kwargs
    ) -> ImageTextResult:
        """
        Answer a question about an image using BLIP VQA.
        
        Args:
            image: Input image
            question: Question about the image
            **kwargs: Additional arguments
            
        Returns:
            ImageTextResult containing answer and metadata
        """
        if self._model_type != "vqa":
            logger.warning(f"Model {self.model_name} is not specifically a VQA model, but attempting VQA anyway")
        
        # Update config for VQA
        config_dict = self.config.dict()
        config_dict['task_type'] = ImageTaskType.VQA
        config_dict['question'] = question
        config_dict.update(kwargs)
        config = ImageCaptionConfig(**config_dict)
        
        # Process image with question
        image_input = self.process_image(image, text=question, **kwargs)
        
        # Generate answer
        return self.generate_text(image_input, config)
    
    def generate_detailed_caption(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        include_context: bool = True,
        include_objects: bool = True,
        include_attributes: bool = True,
        include_relations: bool = False,
        max_length: int = 128,
        **kwargs
    ) -> ImageTextResult:
        """
        Generate a detailed caption for an image.
        
        Args:
            image: Input image
            include_context: Include scene context
            include_objects: Include object descriptions
            include_attributes: Include object attributes
            include_relations: Include object relations
            max_length: Maximum caption length
            **kwargs: Additional arguments
            
        Returns:
            ImageTextResult containing detailed caption
        """
        # Build prompt for detailed captioning
        prompt_parts = []
        
        if include_context:
            prompt_parts.append("Describe the scene")
        
        if include_objects:
            prompt_parts.append("list the main objects")
        
        if include_attributes:
            prompt_parts.append("describe their attributes")
        
        if include_relations:
            prompt_parts.append("and their spatial relationships")
        
        prompt = f"Please {', '.join(prompt_parts)}."
        
        # Update generation parameters for longer output
        kwargs['max_length'] = max_length
        kwargs['num_beams'] = kwargs.get('num_beams', 3)
        kwargs['length_penalty'] = kwargs.get('length_penalty', 1.2)
        
        return self.caption_image(image, prompt=prompt, **kwargs)
    
    def get_image_text_similarity(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        text: str,
        **kwargs
    ) -> float:
        """
        Get similarity score between image and text using BLIP's ITM head.
        
        Args:
            image: Input image
            text: Text to compare
            **kwargs: Additional arguments
            
        Returns:
            Similarity score (higher means more similar)
            
        Raises:
            RuntimeError: If model doesn't support ITM
        """
        if not self._is_loaded:
            raise ValueError(f"Model {self.model_name} is not loaded. Call load() first.")
        
        # Check if model supports ITM
        if not hasattr(self.model, 'itm_head'):
            raise RuntimeError(f"Model {self.model_name} does not support image-text matching")
        
        try:
            # Process image and text
            image_input = self.process_image(image, **kwargs)
            pixel_values = image_input.get('pixel_values')
            
            # Process text
            text_inputs = self.processor(
                text=text,
                return_tensors="pt",
                padding=True
            ).to(self._device)
            
            # Get ITM score
            with torch.no_grad():
                # Get image features
                image_outputs = self.model.vision_model(pixel_values=pixel_values)
                image_embeds = image_outputs[0]
                
                # Get text features
                text_outputs = self.model.text_encoder(
                    input_ids=text_inputs.input_ids,
                    attention_mask=text_inputs.attention_mask,
                )
                text_embeds = text_outputs[0]
                
                # Get multimodal features
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self._device)
                
                # Concatenate and get ITM score
                multimodal_input = torch.cat(
                    [image_embeds, text_embeds[:, 0, :].unsqueeze(1)],
                    dim=1
                )
                multimodal_attention_mask = torch.cat(
                    [image_atts, text_inputs.attention_mask[:, :1]],
                    dim=1
                )
                
                itm_output = self.model.itm_head(
                    multimodal_input,
                    attention_mask=multimodal_attention_mask
                )
                
                # Get similarity score (softmax over [not_match, match])
                itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
                similarity_score = itm_scores[:, 1].item()  # Probability of match
            
            return similarity_score
            
        except Exception as e:
            logger.error(f"Failed to compute image-text similarity: {e}")
            raise RuntimeError(f"Failed to compute image-text similarity: {e}")
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimensionality of image/text embeddings.
        
        Returns:
            Embedding dimension
        """
        if not self._is_loaded:
            # Return from model info if available
            model_info = self.MODEL_INFO.get(self.model_name, {})
            if "embedding_dim" in model_info:
                return model_info["embedding_dim"]
            return 256  # Default for BLIP
        
        # Get from model config
        if hasattr(self.model, 'config'):
            return self.model.config.hidden_size
        
        return 256
    
    def get_version(self) -> str:
        """
        Get model version.
        
        Returns:
            Model version string
        """
        if not self._is_loaded:
            return "unknown"
        
        try:
            # Try to get version from config
            if hasattr(self.model, 'config'):
                config = self.model.config
                if hasattr(config, '_commit_hash'):
                    return config._commit_hash[:8]
                elif hasattr(config, 'model_type'):
                    return f"blip-{config.model_type}"
            
            return "1.0.0"
            
        except:
            return "unknown"
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the loaded BLIP model.
        
        Returns:
            Dictionary with model information
        """
        info = super().get_model_info()
        
        # Add BLIP-specific information
        blip_info = self.MODEL_INFO.get(self.model_name, {}).copy()
        
        info.update({
            "blip_model_type": self._model_type,
            "supports_vqa": self._model_type in ["vqa", "multitask"],
            "supports_captioning": self._model_type in ["captioning", "multitask"],
            "supports_itm": hasattr(self.model, 'itm_head') if self._is_loaded else False,
            "model_specs": blip_info,
        })
        
        return info
    
    def optimize_for_inference(
        self,
        use_half_precision: bool = True,
        use_bettertransformer: bool = True,
        **kwargs
    ) -> None:
        """
        Optimize BLIP model for inference performance.
        
        Args:
            use_half_precision: Whether to use half precision (FP16)
            use_bettertransformer: Whether to use BetterTransformer optimization
            **kwargs: Additional optimization arguments
        """
        if not self._is_loaded:
            raise ValueError(f"Model {self.model_name} is not loaded. Call load() first.")
        
        logger.info(f"Optimizing BLIP model {self.model_name} for inference")
        
        try:
            # Apply half precision
            if use_half_precision and self._device == "cuda":
                self.model = self.model.half()
                logger.info("Applied half precision (FP16)")
            
            # Apply BetterTransformer optimization
            if use_bettertransformer:
                try:
                    from optimum.bettertransformer import BetterTransformer # type: ignore
                    self.model = BetterTransformer.transform(self.model)
                    logger.info("Applied BetterTransformer optimization")
                except ImportError:
                    logger.warning(
                        "optimum not installed. BetterTransformer optimization skipped. "
                        "Install with: pip install optimum"
                    )
                except Exception as e:
                    logger.warning(f"Failed to apply BetterTransformer: {e}")
        
        except Exception as e:
            logger.error(f"Failed to optimize model: {e}")
            raise RuntimeError(f"Failed to optimize model: {e}")
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """
        List all available BLIP models.
        
        Returns:
            Dictionary of model names to model information
        """
        return cls.MODEL_INFO.copy()
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific BLIP model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information dictionary or None if not found
        """
        return cls.MODEL_INFO.get(model_name)
    
    def __repr__(self) -> str:
        """String representation."""
        loaded = self._is_loaded
        device = self._device if loaded else "not loaded"
        model_type = self._model_type
        return (
            f"BLIPModel(model_name={self.model_name}, "
            f"type={model_type}, loaded={loaded}, device={device})"
        )


# Register with factory
try:
    from .base import ImageToTextModelFactory
    ImageToTextModelFactory.register_model('blip', BLIPModel)
    logger.info("Registered BLIPModel with ImageToTextModelFactory")
except ImportError:
    logger.warning("Could not register BLIPModel with factory")


__all__ = [
    'BLIPModel',
    'MODEL_INFO',
]