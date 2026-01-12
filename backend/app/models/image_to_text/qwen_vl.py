"""
Qwen-VL (Qwen Vision Language) model implementation.
Qwen-VL models are powerful multimodal models with strong visual understanding
and reasoning capabilities, supporting high-resolution images and complex tasks.
"""

import base64
import io
import json
import logging
import re
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
)

from .base import (
    BaseImageToTextModel, ImageTextResult, ImageCaptionConfig,
    ImageTaskType, ImageFormat, TextGenerationConfig
)
from ...base import ModelDevice

logger = logging.getLogger(__name__)


class QwenVLModel(BaseImageToTextModel):
    """
    Qwen-VL (Qwen Vision Language) model wrapper.
    
    Qwen-VL models support high-resolution images (up to 448x448 in base, 1024x1024 in plus)
    and excel at detailed image understanding, document parsing, and complex reasoning.
    """
    
    # Qwen-VL model variants and their specifications
    MODEL_INFO = {
        # Qwen-VL base models
        "Qwen/Qwen-VL": {
            "description": "Qwen-VL base model with strong multimodal capabilities",
            "max_tokens": 2048,
            "image_size": 448,
            "vision_model": "ViT-g/14",
            "language_model": "Qwen-7B",
            "parameters": 9_600_000_000,
            "performance": "excellent",
            "memory_mb": 19000,
            "tasks": ["captioning", "vqa", "referring_expression", "visual_reasoning", "document_understanding"],
            "max_resolution": (448, 448),
            "aspect_ratios": ["1:1", "4:3", "3:4", "16:9", "9:16"],
        },
        "Qwen/Qwen-VL-Chat": {
            "description": "Qwen-VL Chat model for conversational multimodal interactions",
            "max_tokens": 2048,
            "image_size": 448,
            "vision_model": "ViT-g/14",
            "language_model": "Qwen-7B",
            "parameters": 9_600_000_000,
            "performance": "excellent",
            "memory_mb": 19000,
            "tasks": ["conversation", "vqa", "detailed_description", "multimodal_chat", "visual_instruction"],
            "max_resolution": (448, 448),
            "aspect_ratios": ["1:1", "4:3", "3:4", "16:9", "9:16"],
        },
        
        # Qwen-VL-Plus models (more powerful)
        "Qwen/Qwen-VL-Plus": {
            "description": "Qwen-VL Plus model with enhanced capabilities and higher resolution",
            "max_tokens": 4096,
            "image_size": 1024,
            "vision_model": "ViT-g/14",
            "language_model": "Qwen-14B",
            "parameters": 14_000_000_000,
            "performance": "state_of_the_art",
            "memory_mb": 28000,
            "tasks": ["captioning", "vqa", "document_understanding", "complex_reasoning", "visual_grounding"],
            "max_resolution": (1024, 1024),
            "aspect_ratios": ["1:1", "4:3", "3:4", "16:9", "9:16", "2:1", "1:2"],
        },
        "Qwen/Qwen-VL-Plus-Chat": {
            "description": "Qwen-VL Plus Chat model for advanced multimodal conversations",
            "max_tokens": 4096,
            "image_size": 1024,
            "vision_model": "ViT-g/14",
            "language_model": "Qwen-14B",
            "parameters": 14_000_000_000,
            "performance": "state_of_the_art",
            "memory_mb": 28000,
            "tasks": ["conversation", "vqa", "detailed_analysis", "multimodal_chat", "visual_instruction", "reasoning"],
            "max_resolution": (1024, 1024),
            "aspect_ratios": ["1:1", "4:3", "3:4", "16:9", "9:16", "2:1", "1:2"],
        },
        
        # Qwen2-VL models (latest generation)
        "Qwen/Qwen2-VL-2B-Instruct": {
            "description": "Qwen2-VL 2B Instruct model - efficient and capable",
            "max_tokens": 32768,
            "image_size": 448,
            "vision_model": "ViT-g/14",
            "language_model": "Qwen2-2B",
            "parameters": 2_700_000_000,
            "performance": "efficient",
            "memory_mb": 5500,
            "tasks": ["captioning", "vqa", "conversation", "document_understanding"],
            "max_resolution": (448, 448),
            "aspect_ratios": ["1:1", "4:3", "3:4", "16:9", "9:16"],
        },
        "Qwen/Qwen2-VL-7B-Instruct": {
            "description": "Qwen2-VL 7B Instruct model - balanced performance",
            "max_tokens": 32768,
            "image_size": 448,
            "vision_model": "ViT-g/14",
            "language_model": "Qwen2-7B",
            "parameters": 9_600_000_000,
            "performance": "excellent",
            "memory_mb": 19000,
            "tasks": ["captioning", "vqa", "conversation", "document_understanding", "reasoning"],
            "max_resolution": (448, 448),
            "aspect_ratios": ["1:1", "4:3", "3:4", "16:9", "9:16"],
        },
        "Qwen/Qwen2-VL-72B-Instruct": {
            "description": "Qwen2-VL 72B Instruct model - most powerful",
            "max_tokens": 32768,
            "image_size": 1024,
            "vision_model": "ViT-g/14",
            "language_model": "Qwen2-72B",
            "parameters": 86_000_000_000,
            "performance": "state_of_the_art",
            "memory_mb": 144000,
            "tasks": ["captioning", "vqa", "conversation", "complex_reasoning", "detailed_analysis", "document_understanding"],
            "max_resolution": (1024, 1024),
            "aspect_ratios": ["1:1", "4:3", "3:4", "16:9", "9:16", "2:1", "1:2"],
        },
    }
    
    # Special tokens and prompts for Qwen-VL
    SPECIAL_TOKENS = {
        "image_start": "<img>",
        "image_end": "</img>",
        "ref_start": "<ref>",
        "ref_end": "</ref>",
        "box_start": "<box>",
        "box_end": "</box>",
    }
    
    # Task-specific prompt templates
    PROMPT_TEMPLATES = {
        "captioning": "Provide a detailed description of this image.",
        "detailed_captioning": "Provide a comprehensive description of this image including objects, their attributes, spatial relationships, and the overall scene.",
        "vqa": "{question}",
        "document_understanding": "Extract and understand the content of this document.",
        "ocr": "Extract all text from this image with accurate formatting.",
        "referring_expression": "Describe the object at location {coordinates}.",
        "visual_grounding": "Identify the object described as '{description}'.",
        "multimodal_chat": "You are Qwen-VL, a helpful AI assistant. {message}",
    }
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen-VL-Chat",
        config: Optional[ImageCaptionConfig] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        processor_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize Qwen-VL model.
        
        Args:
            model_name: Name of the Qwen-VL model
            config: Configuration for image understanding
            model_kwargs: Additional arguments for model initialization
            processor_kwargs: Additional arguments for processor initialization
            **kwargs: Additional arguments passed to configuration
        """
        # Update config with Qwen-VL-specific defaults
        if config is None:
            config = ImageCaptionConfig()
        
        # Qwen-VL models support higher resolution images
        model_info = self.MODEL_INFO.get(model_name, {})
        config.image_size = (model_info.get("image_size", 448), model_info.get("image_size", 448))
        config.task_type = ImageTaskType.VISUAL_INSTRUCTION
        
        # Increase max tokens for detailed responses
        if config.generation_config.max_new_tokens < 512:
            config.generation_config.max_new_tokens = 1024
        
        super().__init__(model_name, config, **kwargs)
        
        # Qwen-VL-specific initialization
        self.model_kwargs = model_kwargs or {}
        self.processor_kwargs = processor_kwargs or {}
        
        # Check if it's a Qwen2-VL model (newer architecture)
        self._is_qwen2 = "qwen2" in model_name.lower()
        
        # Conversation history for chat models
        self._conversation_history: List[Dict[str, str]] = []
        self._max_history_length = 20
        
        # Generation parameters optimized for Qwen-VL
        self._default_generation_params = {
            "max_new_tokens": self.config.generation_config.max_new_tokens,
            "temperature": 0.1,  # Lower temperature for more factual responses
            "top_p": 0.9,
            "top_k": 40,
            "repetition_penalty": 1.05,
            "do_sample": True,
            "num_beams": 1,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
        }
        
        # For Qwen2-VL, adjust parameters
        if self._is_qwen2:
            self._default_generation_params.update({
                "temperature": 0.7,
                "top_p": 0.8,
                "repetition_penalty": 1.1,
            })
        
        # Initialize model and processor
        self.model = None
        self.tokenizer = None
        self.processor = None
        
        logger.info(f"Initialized Qwen-VL model: {model_name} (Qwen2: {self._is_qwen2})")
    
    def load(self) -> None:
        """
        Load the Qwen-VL model and processor/tokenizer.
        
        Raises:
            RuntimeError: If model loading fails
        """
        if self._is_loaded:
            logger.warning(f"Model {self.model_name} is already loaded")
            return
        
        try:
            logger.info(f"Loading Qwen-VL model: {self.model_name}")
            
            # Determine device with memory considerations
            if torch.cuda.is_available():
                self._device = "cuda"
                
                # Check memory requirements
                model_info = self.MODEL_INFO.get(self.model_name, {})
                required_memory_mb = model_info.get("memory_mb", 19000)
                
                try:
                    free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                    free_memory_mb = free_memory / (1024 * 1024)
                    
                    if free_memory_mb < required_memory_mb:
                        logger.warning(
                            f"GPU memory may be insufficient: {free_memory_mb:.0f}MB available, "
                            f"{required_memory_mb:.0f}MB recommended. "
                            "Will attempt to load with memory optimizations."
                        )
                        
                        # Enable memory optimizations
                        self.model_kwargs["low_cpu_mem_usage"] = True
                        self.model_kwargs["torch_dtype"] = torch.float16
                        
                        # For very large models, use device_map
                        if required_memory_mb > 30000:
                            self.model_kwargs["device_map"] = "auto"
                
                except Exception as e:
                    logger.warning(f"Could not check GPU memory: {e}")
            else:
                self._device = "cpu"
                logger.warning(
                    "Qwen-VL models are large and run slowly on CPU. "
                    "Consider using GPU for reasonable performance."
                )
            
            # Load processor/tokenizer based on model type
            if self._is_qwen2:
                # Qwen2-VL uses a combined processor
                logger.info(f"Loading Qwen2-VL processor for {self.model_name}")
                self.processor = Qwen2VLProcessor.from_pretrained(
                    self.model_name,
                    **self.processor_kwargs
                )
            else:
                # Original Qwen-VL uses separate tokenizer
                logger.info(f"Loading tokenizer for {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    **self.processor_kwargs
                )
            
            # Load model
            logger.info(f"Loading Qwen-VL model for {self.model_name}")
            
            if self._is_qwen2:
                # Load Qwen2-VL model
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    **self.model_kwargs
                )
            else:
                # Load original Qwen-VL model
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    **self.model_kwargs
                )
            
            # Move to device if not using device_map
            if "device_map" not in self.model_kwargs:
                self.model.to(self._device)
            
            self.model.eval()  # Set to evaluation mode
            
            self._is_loaded = True
            
            # Update image size from model info
            model_info = self.MODEL_INFO.get(self.model_name, {})
            max_res = model_info.get("max_resolution", (448, 448))
            self.image_size = max_res
            
            logger.info(f"Successfully loaded Qwen-VL model {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen-VL model {self.model_name}: {e}")
            raise RuntimeError(f"Failed to load Qwen-VL model {self.model_name}: {e}")
    
    def process_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process an image and prompt for Qwen-VL model input.
        
        Args:
            image: Input image in various formats
            prompt: Optional text prompt
            conversation_history: Optional previous conversation turns
            **kwargs: Additional processing arguments
            
        Returns:
            Dictionary containing processed inputs and metadata
            
        Raises:
            ValueError: If model is not loaded
            RuntimeError: If processing fails
        """
        if not self._is_loaded:
            raise ValueError(f"Model {self.model_name} is not loaded. Call load() first.")
        
        try:
            # Load image
            pil_image = self._load_image(image)
            
            # Format prompt
            formatted_prompt = self._format_prompt(
                image=pil_image,
                prompt=prompt,
                conversation_history=conversation_history,
                **kwargs
            )
            
            # Process based on model type
            if self._is_qwen2:
                # Qwen2-VL processing
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": formatted_prompt}
                        ]
                    }
                ]
                
                # Prepare inputs
                text_inputs = self.processor(
                    messages,
                    return_tensors="pt",
                    **kwargs
                )
                
                # Move to device
                if self._device.startswith("cuda"):
                    text_inputs = {k: v.to(self._device) for k, v in text_inputs.items()}
                
                result = {
                    'input_ids': text_inputs.get('input_ids'),
                    'attention_mask': text_inputs.get('attention_mask'),
                    'pixel_values': text_inputs.get('pixel_values'),
                    'original_image': pil_image,
                    'image_size': pil_image.size,
                    'prompt': formatted_prompt,
                    'original_prompt': prompt,
                    'model_inputs': text_inputs,
                }
                
            else:
                # Original Qwen-VL processing
                # Convert image to base64 for Qwen-VL format
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                # Create query with image
                query = self.tokenizer.from_list_format([
                    {'image': f'data:image/png;base64,{img_base64}'},
                    {'text': formatted_prompt},
                ])
                
                # Tokenize
                inputs = self.tokenizer(query, return_tensors='pt')
                
                # Move to device
                if self._device.startswith("cuda"):
                    inputs = {k: v.to(self._device) for k, v in inputs.items()}
                
                result = {
                    'input_ids': inputs.get('input_ids'),
                    'attention_mask': inputs.get('attention_mask'),
                    'original_image': pil_image,
                    'image_size': pil_image.size,
                    'prompt': formatted_prompt,
                    'original_prompt': prompt,
                    'model_inputs': inputs,
                    'image_base64': img_base64,
                }
            
            # Add conversation history if provided
            if conversation_history:
                result['conversation_history'] = conversation_history.copy()
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process image for Qwen-VL: {e}")
            raise RuntimeError(f"Failed to process image for Qwen-VL: {e}")
    
    def _format_prompt(
        self,
        image: Optional[Image.Image] = None,
        prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """
        Format prompt for Qwen-VL model.
        
        Args:
            image: PIL Image (optional, for size info)
            prompt: User prompt
            conversation_history: Previous conversation turns
            **kwargs: Additional formatting arguments
            
        Returns:
            Formatted prompt string
        """
        # Start with conversation history if provided
        formatted = ""
        
        if conversation_history:
            for turn in conversation_history[-self._max_history_length:]:
                user_msg = turn.get('user', '')
                assistant_msg = turn.get('assistant', '')
                
                if user_msg:
                    formatted += f"Human: {user_msg}\n"
                if assistant_msg:
                    formatted += f"Assistant: {assistant_msg}\n"
        
        # Add current prompt
        if prompt:
            # Check if prompt matches a template
            template_found = False
            for template_key, template in self.PROMPT_TEMPLATES.items():
                if template_key in prompt.lower() or prompt.lower() in template_key:
                    # Use the template
                    if '{' in template:
                        # Format template with any provided kwargs
                        try:
                            formatted_prompt = template.format(**kwargs)
                        except KeyError:
                            formatted_prompt = prompt
                    else:
                        formatted_prompt = template
                    template_found = True
                    break
            
            if not template_found:
                formatted_prompt = prompt
            
            formatted += f"Human: {formatted_prompt}\nAssistant:"
        else:
            # Use default captioning prompt
            formatted += f"Human: {self.PROMPT_TEMPLATES['captioning']}\nAssistant:"
        
        return formatted.strip()
    
    def generate_text(
        self,
        image_input: Dict[str, Any],
        config: Optional[ImageCaptionConfig] = None,
        **kwargs
    ) -> ImageTextResult:
        """
        Generate text response from Qwen-VL.
        
        Args:
            image_input: Processed image and prompt input
            config: Configuration for text generation
            **kwargs: Additional generation arguments
            
        Returns:
            ImageTextResult containing generated response and metadata
            
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
            
            # Extract model inputs
            input_ids = image_input.get('input_ids')
            attention_mask = image_input.get('attention_mask')
            
            if input_ids is None:
                raise ValueError("Processed input missing required fields")
            
            # Prepare generation parameters
            gen_params = self._default_generation_params.copy()
            
            # Update with config
            if config.generation_config:
                gen_params.update({
                    "max_new_tokens": config.generation_config.max_new_tokens,
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
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_params
                )
            
            # Decode generated tokens
            if self._is_qwen2:
                generated_text = self.processor.decode(
                    outputs[0],
                    skip_special_tokens=True
                )
            else:
                generated_text = self.tokenizer.decode(
                    outputs.cpu()[0],
                    skip_special_tokens=True
                )
            
            # Extract only the assistant's response
            # Remove the prompt and any special formatting
            prompt_text = image_input.get('prompt', '')
            if prompt_text and prompt_text in generated_text:
                # Extract text after the prompt
                response = generated_text.split(prompt_text)[-1]
            else:
                # Try to extract after the last "Assistant:" marker
                parts = generated_text.split('Assistant:')
                response = parts[-1].strip() if len(parts) > 1 else generated_text
            
            # Clean up the response
            response = self._clean_response(response)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Track performance
            self._processing_times.append(processing_time_ms)
            self._total_images_processed += 1
            
            # Update conversation history
            original_prompt = image_input.get('original_prompt')
            if original_prompt and response:
                self._update_conversation_history(original_prompt, response)
            
            # Prepare result
            result = ImageTextResult(
                text=response.strip(),
                model_name=self.model_name,
                model_version=self.get_version(),
                input_image_info={
                    "original_size": image_input.get('image_size'),
                    "processed_size": self.image_size,
                    "prompt": original_prompt,
                    "conversation_history_length": len(self._conversation_history),
                },
                task_type=config.task_type,
                processing_time_ms=processing_time_ms,
                metadata={
                    "generation_params": gen_params,
                    "is_qwen2": self._is_qwen2,
                    "device": self._device,
                    "prompt": original_prompt,
                    "conversation_history": self._conversation_history.copy(),
                    "model_info": self.MODEL_INFO.get(self.model_name, {}),
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate text with Qwen-VL: {e}")
            raise RuntimeError(f"Failed to generate text with Qwen-VL: {e}")
    
    def _clean_response(self, response: str) -> str:
        """
        Clean up Qwen-VL response.
        
        Args:
            response: Raw generated response
            
        Returns:
            Cleaned response
        """
        # Remove special tokens
        for token in self.SPECIAL_TOKENS.values():
            response = response.replace(token, "")
        
        # Remove any image references
        response = re.sub(r'!\[.*?\]\(.*?\)', '', response)  # Markdown images
        response = re.sub(r'<img.*?>', '', response)  # HTML images
        
        # Remove duplicate whitespace
        response = re.sub(r'\s+', ' ', response).strip()
        
        # Remove trailing markers
        response = response.rstrip('Human:').rstrip('Assistant:').strip()
        
        return response
    
    def _update_conversation_history(self, user_message: str, assistant_response: str) -> None:
        """
        Update conversation history with new turn.
        
        Args:
            user_message: User's message
            assistant_response: Assistant's response
        """
        self._conversation_history.append({
            "user": user_message,
            "assistant": assistant_response,
            "timestamp": time.time(),
        })
        
        # Limit history length
        if len(self._conversation_history) > self._max_history_length:
            self._conversation_history = self._conversation_history[-self._max_history_length:]
    
    def chat_with_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        message: str,
        clear_history: bool = False,
        **kwargs
    ) -> ImageTextResult:
        """
        Have a conversation about an image.
        
        Args:
            image: Input image
            message: User message about the image
            clear_history: Whether to clear conversation history
            **kwargs: Additional arguments
            
        Returns:
            ImageTextResult containing response and metadata
        """
        if clear_history:
            self.clear_conversation_history()
        
        # Update config
        config_dict = self.config.dict()
        config_dict['task_type'] = ImageTaskType.MULTIMODAL_CHAT
        config_dict.update(kwargs)
        config = ImageCaptionConfig(**config_dict)
        
        # Process image with message
        image_input = self.process_image(
            image,
            prompt=message,
            conversation_history=self._conversation_history,
            **kwargs
        )
        
        # Generate response
        return self.generate_text(image_input, config)
    
    def analyze_document(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        document_type: Optional[str] = None,
        extract_structure: bool = True,
        extract_text: bool = True,
        extract_entities: bool = False,
        **kwargs
    ) -> ImageTextResult:
        """
        Analyze a document image.
        
        Args:
            image: Document image
            document_type: Type of document (receipt, invoice, form, etc.)
            extract_structure: Extract document structure
            extract_text: Extract text content
            extract_entities: Extract named entities
            **kwargs: Additional arguments
            
        Returns:
            ImageTextResult containing document analysis
        """
        # Build analysis prompt
        prompt_parts = ["Analyze this document."]
        
        if document_type:
            prompt_parts.append(f"It appears to be a {document_type}.")
        
        if extract_structure:
            prompt_parts.append("Describe the structure and layout.")
        
        if extract_text:
            prompt_parts.append("Extract and summarize the text content.")
        
        if extract_entities:
            prompt_parts.append("Identify any important entities like names, dates, amounts, etc.")
        
        prompt = " ".join(prompt_parts)
        
        # Update generation parameters for factual output
        kwargs['temperature'] = kwargs.get('temperature', 0.1)
        kwargs['do_sample'] = kwargs.get('do_sample', False)
        kwargs['max_new_tokens'] = kwargs.get('max_new_tokens', 1024)
        
        return self.chat_with_image(image, prompt, **kwargs)
    
    def describe_with_bbox(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        bbox: Tuple[float, float, float, float],  # (x1, y1, x2, y2) normalized 0-1
        description_type: str = "detailed",
        **kwargs
    ) -> ImageTextResult:
        """
        Describe a specific region of an image defined by bounding box.
        
        Args:
            image: Input image
            bbox: Bounding box coordinates (x1, y1, x2, y2) normalized 0-1
            description_type: Type of description ("detailed", "brief", "attributes")
            **kwargs: Additional arguments
            
        Returns:
            ImageTextResult containing region description
        """
        # Format bbox for Qwen-VL
        x1, y1, x2, y2 = bbox
        bbox_str = f"({x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f})"
        
        # Build prompt with bbox
        if description_type == "detailed":
            prompt = f"Describe in detail the region at {bbox_str} in this image."
        elif description_type == "brief":
            prompt = f"Briefly describe the region at {bbox_str} in this image."
        elif description_type == "attributes":
            prompt = f"Describe the attributes of the object at {bbox_str} in this image."
        else:
            prompt = f"Describe the region at {bbox_str} in this image."
        
        return self.chat_with_image(image, prompt, **kwargs)
    
    def compare_images(
        self,
        image1: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        image2: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        comparison_type: str = "similarities_differences",
        **kwargs
    ) -> ImageTextResult:
        """
        Compare two images (requires multiple image support).
        
        Note: This is a placeholder implementation. Full multi-image
        support requires more complex prompt formatting.
        
        Args:
            image1: First image
            image2: Second image
            comparison_type: Type of comparison
            **kwargs: Additional arguments
            
        Returns:
            ImageTextResult containing comparison
        """
        comparison_prompts = {
            "similarities_differences": "Compare these two images. Describe their similarities and differences.",
            "chronological": "These images show a sequence. Describe what changed between them.",
            "quality": "Compare the quality of these two images.",
            "content": "Compare the content of these two images.",
        }
        
        prompt = comparison_prompts.get(
            comparison_type.lower(),
            comparison_prompts["similarities_differences"]
        )
        
        # For now, we'll just analyze the first image
        # Full multi-image support requires model-specific implementations
        logger.warning(
            "Multi-image comparison requires specific model support. "
            "Analyzing only the first image."
        )
        
        return self.chat_with_image(image1, prompt, **kwargs)
    
    def extract_text(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        language: Optional[str] = None,
        preserve_layout: bool = False,
        **kwargs
    ) -> ImageTextResult:
        """
        Extract text from image (OCR).
        
        Args:
            image: Input image with text
            language: Language of the text (optional)
            preserve_layout: Attempt to preserve text layout
            **kwargs: Additional arguments
            
        Returns:
            ImageTextResult containing extracted text
        """
        prompt = "Extract all text from this image"
        
        if language:
            prompt += f" (language: {language})"
        
        if preserve_layout:
            prompt += ". Preserve the layout and formatting as much as possible."
        else:
            prompt += "."
        
        # Use lower temperature for more accurate OCR
        kwargs['temperature'] = kwargs.get('temperature', 0.01)
        kwargs['do_sample'] = kwargs.get('do_sample', False)
        
        return self.chat_with_image(image, prompt, **kwargs)
    
    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self._conversation_history.clear()
        logger.info("Cleared Qwen-VL conversation history")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the current conversation history.
        
        Returns:
            List of conversation turns
        """
        return self._conversation_history.copy()
    
    def get_model_capabilities(self) -> Dict[str, Any]:
        """
        Get capabilities of the Qwen-VL model.
        
        Returns:
            Dictionary with model capabilities
        """
        model_info = self.MODEL_INFO.get(self.model_name, {}).copy()
        
        capabilities = {
            "model": self.model_name,
            "vision_model": model_info.get("vision_model", "ViT-g/14"),
            "language_model": model_info.get("language_model", "Qwen-7B"),
            "tasks": model_info.get("tasks", []),
            "max_image_size": self.image_size,
            "max_tokens": model_info.get("max_tokens", 2048),
            "is_qwen2": self._is_qwen2,
            "supports_bbox": True,
            "supports_multi_image": "qwen2" in self.model_name.lower(),
            "supports_document_understanding": "document_understanding" in model_info.get("tasks", []),
            "max_resolution": model_info.get("max_resolution", (448, 448)),
            "aspect_ratios": model_info.get("aspect_ratios", ["1:1", "4:3", "3:4", "16:9", "9:16"]),
        }
        
        return capabilities
    
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
                    return f"qwen-vl-{config.model_type}"
            
            # Extract from model name
            model_lower = self.model_name.lower()
            
            if "qwen2" in model_lower:
                if "72b" in model_lower:
                    return "qwen2-vl-72b"
                elif "7b" in model_lower:
                    return "qwen2-vl-7b"
                elif "2b" in model_lower:
                    return "qwen2-vl-2b"
                else:
                    return "qwen2-vl"
            elif "plus" in model_lower:
                return "qwen-vl-plus"
            else:
                return "qwen-vl"
            
        except:
            return "unknown"
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the loaded Qwen-VL model.
        
        Returns:
            Dictionary containing model information
        """
        model_info = self.MODEL_INFO.get(self.model_name, {}).copy()
        
        info = {
            "model_name": self.model_name,
            "version": self.get_version(),
            "is_loaded": self._is_loaded,
            "device": self._device,
            "is_qwen2": self._is_qwen2,
            "image_size": self.image_size,
            "conversation_history_length": len(self._conversation_history),
            "total_images_processed": self._total_images_processed,
            "performance_stats": {
                "average_processing_time_ms": self.get_average_processing_time(),
                "min_processing_time_ms": self.get_min_processing_time(),
                "max_processing_time_ms": self.get_max_processing_time(),
            },
            "capabilities": self.get_model_capabilities(),
            "model_details": model_info,
        }
        
        return info
    
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
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._is_loaded = False
        logger.info(f"Unloaded Qwen-VL model {self.model_name}")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.unload()
        except:
            pass
    
    def get_average_processing_time(self) -> float:
        """
        Get average processing time.
        
        Returns:
            Average processing time in milliseconds
        """
        if not self._processing_times:
            return 0.0
        return sum(self._processing_times) / len(self._processing_times)
    
    def get_min_processing_time(self) -> float:
        """
        Get minimum processing time.
        
        Returns:
            Minimum processing time in milliseconds
        """
        if not self._processing_times:
            return 0.0
        return min(self._processing_times)
    
    def get_max_processing_time(self) -> float:
        """
        Get maximum processing time.
        
        Returns:
            Maximum processing time in milliseconds
        """
        if not self._processing_times:
            return 0.0
        return max(self._processing_times)
    
    def save_conversation_history(self, filepath: Union[str, Path]) -> None:
        """
        Save conversation history to file.
        
        Args:
            filepath: Path to save file
        """
        filepath = Path(filepath)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self._conversation_history, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved conversation history to {filepath}")
    
    def load_conversation_history(self, filepath: Union[str, Path]) -> None:
        """
        Load conversation history from file.
        
        Args:
            filepath: Path to load file from
        """
        filepath = Path(filepath)
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                self._conversation_history = json.load(f)
            logger.info(f"Loaded conversation history from {filepath}")
        else:
            logger.warning(f"Conversation history file not found: {filepath}")
    
    def visualize_attention(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        prompt: Optional[str] = None,
        return_type: str = "heatmap",
        **kwargs
    ) -> Optional[np.ndarray]:
        """
        Visualize attention maps (if supported by model).
        
        Note: This is a placeholder implementation.
        Actual attention visualization requires model-specific hooks.
        
        Args:
            image: Input image
            prompt: Optional prompt
            return_type: Type of visualization ("heatmap", "overlay", "raw")
            **kwargs: Additional arguments
            
        Returns:
            Attention visualization or None if not supported
        """
        logger.warning("Attention visualization not fully implemented for Qwen-VL")
        return None
    
    def batch_process(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes]],
        prompts: Optional[List[str]] = None,
        batch_size: int = 4,
        **kwargs
    ) -> List[ImageTextResult]:
        """
        Process multiple images in batch.
        
        Args:
            images: List of input images
            prompts: Optional list of prompts (one per image)
            batch_size: Batch size for processing
            **kwargs: Additional arguments
            
        Returns:
            List of ImageTextResult objects
        """
        if prompts is None:
            prompts = [None] * len(images)
        
        if len(images) != len(prompts):
            raise ValueError("Number of images must match number of prompts")
        
        results = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_prompts = prompts[i:i+batch_size]
            
            for image, prompt in zip(batch_images, batch_prompts):
                try:
                    result = self.chat_with_image(image, prompt or "Describe this image", **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process image {i}: {e}")
                    # Add placeholder result
                    results.append(ImageTextResult(
                        text=f"Error processing image: {str(e)}",
                        model_name=self.model_name,
                        processing_time_ms=0,
                        metadata={"error": str(e)}
                    ))
            
            # Small delay between batches
            if i + batch_size < len(images):
                time.sleep(0.1)
        
        return results