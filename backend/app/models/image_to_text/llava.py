"""
LLaVA (Large Language and Vision Assistant) model implementation.
LLaVA models are powerful vision-language models for detailed image understanding,
visual instruction following, and complex reasoning.
"""
import logging
from time import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

from anyio import Path
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    LlavaConfig,
)

from .base import (
    BaseImageToTextModel, ImageTextResult, ImageCaptionConfig,
    ImageTaskType, ImageFormat, TextGenerationConfig
)
from ...base import ModelDevice

logger = logging.getLogger(__name__)


class LLaVAModel(BaseImageToTextModel):
    """
    LLaVA (Large Language and Vision Assistant) model wrapper.
    
    LLaVA models combine vision encoders (CLIP) with large language models (LLaMA/Vicuna)
    to enable sophisticated visual understanding and reasoning.
    """
    
    # LLaVA model variants and their specifications
    MODEL_INFO = {
        # LLaVA 1.5 models (7B parameters)
        "llava-hf/llava-1.5-7b-hf": {
            "description": "LLaVA 1.5 7B model with strong visual instruction following",
            "max_tokens": 2048,
            "image_size": 336,
            "vision_model": "CLIP-ViT-L/14",
            "language_model": "Vicuna-7B-v1.5",
            "parameters": 7_000_000_000,
            "performance": "balanced",
            "memory_mb": 14000,
            "tasks": ["visual_instruction", "vqa", "detailed_description", "complex_reasoning"],
        },
        "llava-hf/llava-1.5-7b-hf": {
            "description": "LLaVA 1.5 7B base model",
            "max_tokens": 2048,
            "image_size": 336,
            "vision_model": "CLIP-ViT-L/14",
            "language_model": "Vicuna-7B-v1.5",
            "parameters": 7_000_000_000,
            "performance": "balanced",
            "memory_mb": 14000,
            "tasks": ["visual_instruction", "vqa", "detailed_description"],
        },
        
        # LLaVA 1.5 models (13B parameters)
        "llava-hf/llava-1.5-13b-hf": {
            "description": "LLaVA 1.5 13B model with enhanced capabilities",
            "max_tokens": 2048,
            "image_size": 336,
            "vision_model": "CLIP-ViT-L/14",
            "language_model": "Vicuna-13B-v1.5",
            "parameters": 13_000_000_000,
            "performance": "accurate",
            "memory_mb": 26000,
            "tasks": ["visual_instruction", "vqa", "complex_reasoning", "detailed_analysis"],
        },
        
        # LLaVA-NeXT models (improved versions)
        "llava-hf/llava-v1.6-7b-hf": {
            "description": "LLaVA v1.6 7B with improved multimodal understanding",
            "max_tokens": 4096,
            "image_size": 384,
            "vision_model": "CLIP-ViT-L/14-336",
            "language_model": "Mistral-7B",
            "parameters": 7_000_000_000,
            "performance": "excellent",
            "memory_mb": 14000,
            "tasks": ["visual_instruction", "vqa", "complex_reasoning", "multimodal_chat"],
        },
        "llava-hf/llava-v1.6-13b-hf": {
            "description": "LLaVA v1.6 13B with state-of-the-art performance",
            "max_tokens": 4096,
            "image_size": 384,
            "vision_model": "CLIP-ViT-L/14-336",
            "language_model": "Vicuna-13B",
            "parameters": 13_000_000_000,
            "performance": "state_of_the_art",
            "memory_mb": 26000,
            "tasks": ["visual_instruction", "vqa", "complex_reasoning", "multimodal_chat", "detailed_analysis"],
        },
        
        # Specialized models
        "llava-hf/llava-med-7b": {
            "description": "LLaVA-Med 7B for biomedical visual question answering",
            "max_tokens": 2048,
            "image_size": 224,
            "vision_model": "PubMedCLIP",
            "language_model": "Vicuna-7B",
            "parameters": 7_000_000_000,
            "performance": "accurate",
            "memory_mb": 14000,
            "tasks": ["medical_vqa", "biomedical_analysis"],
            "domain": "medical",
        },
    }
    
    # Pre-defined conversation templates for different tasks
    CONVERSATION_TEMPLATES = {
        "default": {
            "system": "A chat between a curious user and an artificial intelligence assistant. "
                     "The assistant gives helpful, detailed, and polite answers to the user's questions.",
            "user": "{user_input}",
            "assistant": "{assistant_response}",
        },
        "detailed_description": {
            "system": "You are a helpful assistant that provides detailed, accurate descriptions of images.",
            "user": "Please provide a detailed description of this image.",
            "assistant": "{assistant_response}",
        },
        "visual_instruction": {
            "system": "You are a helpful assistant that follows visual instructions accurately.",
            "user": "{instruction}",
            "assistant": "{assistant_response}",
        },
        "complex_reasoning": {
            "system": "You are a helpful assistant that performs complex reasoning about images. "
                     "Think step by step and explain your reasoning process.",
            "user": "{question}",
            "assistant": "{assistant_response}",
        },
        "multimodal_chat": {
            "system": "You are a helpful, respectful, and honest assistant. "
                     "You engage in natural conversations about images and provide useful information.",
            "user": "{user_message}",
            "assistant": "{assistant_response}",
        },
    }
    
    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        config: Optional[ImageCaptionConfig] = None,
        conversation_template: str = "default",
        model_kwargs: Optional[Dict[str, Any]] = None,
        processor_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize LLaVA model.
        
        Args:
            model_name: Name of the LLaVA model
            config: Configuration for image understanding
            conversation_template: Template for conversation formatting
            model_kwargs: Additional arguments for model initialization
            processor_kwargs: Additional arguments for processor initialization
            **kwargs: Additional arguments passed to configuration
        """
        # Update config with LLaVA-specific defaults
        if config is None:
            config = ImageCaptionConfig()
        
        # LLaVA models work with specific image sizes
        model_info = self.MODEL_INFO.get(model_name, {})
        config.image_size = (model_info.get("image_size", 336), model_info.get("image_size", 336))
        config.task_type = ImageTaskType.VISUAL_INSTRUCTION
        
        # Increase max tokens for detailed responses
        if config.generation_config.max_new_tokens < 512:
            config.generation_config.max_new_tokens = 1024
        
        super().__init__(model_name, config, **kwargs)
        
        # LLaVA-specific initialization
        self.model_kwargs = model_kwargs or {}
        self.processor_kwargs = processor_kwargs or {}
        self.conversation_template = conversation_template
        
        # Chat history for multi-turn conversations
        self._chat_history: List[Dict[str, str]] = []
        self._max_history_length = 10
        
        # Generation parameters optimized for conversational responses
        self._default_generation_params = {
            "max_new_tokens": self.config.generation_config.max_new_tokens,
            "temperature": 0.2,  # Lower temperature for more coherent responses
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "num_beams": 1,  # Typically use sampling for conversational models
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
        }
        
        logger.info(f"Initialized LLaVA model: {model_name} with template: {conversation_template}")
    
    def load(self) -> None:
        """
        Load the LLaVA model and processor.
        
        Raises:
            RuntimeError: If model loading fails
        """
        if self._is_loaded:
            logger.warning(f"Model {self.model_name} is already loaded")
            return
        
        try:
            logger.info(f"Loading LLaVA model: {self.model_name}")
            
            # Determine device with memory optimization
            if torch.cuda.is_available():
                self._device = "cuda"
                
                # Check available GPU memory
                try:
                    free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                    free_memory_mb = free_memory / (1024 * 1024)
                    
                    model_info = self.MODEL_INFO.get(self.model_name, {})
                    required_memory_mb = model_info.get("memory_mb", 14000)
                    
                    if free_memory_mb < required_memory_mb * 1.2:  # 20% buffer
                        logger.warning(
                            f"Insufficient GPU memory: {free_memory_mb:.0f}MB available, "
                            f"{required_memory_mb:.0f}MB recommended. "
                            "Consider using CPU or a smaller model."
                        )
                        
                        # Try to load with memory optimization
                        self.model_kwargs["low_cpu_mem_usage"] = True
                        self.model_kwargs["torch_dtype"] = torch.float16
                
                except Exception as e:
                    logger.warning(f"Could not check GPU memory: {e}")
            else:
                self._device = "cpu"
                logger.warning(
                    "LLaVA models are large and run slowly on CPU. "
                    "Consider using GPU for better performance."
                )
            
            # Load processor
            logger.info(f"Loading LLaVA processor for {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                **self.processor_kwargs
            )
            
            # Configure model loading for memory efficiency
            load_kwargs = self.model_kwargs.copy()
            
            if self._device == "cuda":
                # Use bfloat16 if available for better performance
                if torch.cuda.is_bf16_supported():
                    load_kwargs["torch_dtype"] = torch.bfloat16
                else:
                    load_kwargs["torch_dtype"] = torch.float16
                
                load_kwargs["device_map"] = "auto"
                load_kwargs["low_cpu_mem_usage"] = True
            
            # Load model
            logger.info(f"Loading LLaVA model for {self.model_name}")
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_name,
                **load_kwargs
            )
            
            # If not using device_map="auto", move to device
            if "device_map" not in load_kwargs:
                self.model.to(self._device)
            
            self.model.eval()  # Set to evaluation mode
            
            self._is_loaded = True
            
            # Update image size from processor
            if hasattr(self.processor, 'image_processor'):
                image_processor = self.processor.image_processor
                if hasattr(image_processor, 'size'):
                    size = image_processor.size
                    if isinstance(size, dict):
                        size = size.get('shortest_edge', 336)
                    elif isinstance(size, (list, tuple)):
                        size = min(size)
                    self.image_size = (size, size)
                    logger.info(f"Updated image size from processor: {self.image_size}")
            
            logger.info(f"Successfully loaded LLaVA model {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load LLaVA model {self.model_name}: {e}")
            raise RuntimeError(f"Failed to load LLaVA model {self.model_name}: {e}")
    
    def process_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process an image and prompt for LLaVA model input.
        
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
            
            # Format prompt with conversation history
            formatted_prompt = self._format_prompt(
                prompt=prompt,
                conversation_history=conversation_history,
                **kwargs
            )
            
            # Process with LLaVA processor
            inputs = self.processor(
                text=formatted_prompt,
                images=pil_image,
                return_tensors="pt",
                padding=True,
                **kwargs
            )
            
            # Move inputs to device
            if self._device.startswith("cuda"):
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            # Prepare result
            result = {
                'input_ids': inputs.get('input_ids'),
                'attention_mask': inputs.get('attention_mask'),
                'pixel_values': inputs.get('pixel_values'),
                'image_patches': inputs.get('image_patches'),
                'image_patches_indices': inputs.get('image_patches_indices'),
                'original_image': pil_image,
                'image_size': pil_image.size,
                'prompt': formatted_prompt,
                'original_prompt': prompt,
                'model_inputs': inputs,
            }
            
            # Add conversation history if provided
            if conversation_history:
                result['conversation_history'] = conversation_history.copy()
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process image for LLaVA: {e}")
            raise RuntimeError(f"Failed to process image for LLaVA: {e}")
    
    def _format_prompt(
        self,
        prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """
        Format prompt for LLaVA model.
        
        Args:
            prompt: User prompt
            conversation_history: Previous conversation turns
            **kwargs: Additional formatting arguments
            
        Returns:
            Formatted prompt string
        """
        # Get conversation template
        template = self.CONVERSATION_TEMPLATES.get(
            self.conversation_template,
            self.CONVERSATION_TEMPLATES["default"]
        )
        
        # Start with system message
        formatted = f"<s>[INST] <<SYS>>\n{template['system']}\n<</SYS>>\n\n"
        
        # Add conversation history if provided
        if conversation_history:
            for turn in conversation_history[-self._max_history_length:]:
                user_msg = turn.get('user', '')
                assistant_msg = turn.get('assistant', '')
                
                if user_msg:
                    formatted += f"{user_msg} [/INST] "
                if assistant_msg:
                    formatted += f"{assistant_msg} </s><s>[INST] "
        
        # Add current user prompt
        if prompt:
            # Use template's user format if prompt matches template structure
            if '{user_input}' in template['user']:
                user_prompt = template['user'].format(user_input=prompt)
            elif '{instruction}' in template['user']:
                user_prompt = template['user'].format(instruction=prompt)
            elif '{question}' in template['user']:
                user_prompt = template['user'].format(question=prompt)
            elif '{user_message}' in template['user']:
                user_prompt = template['user'].format(user_message=prompt)
            else:
                user_prompt = prompt
            
            formatted += f"{user_prompt} [/INST] "
        else:
            # Use default user prompt from template
            default_user = template['user'].replace('{user_input}', '')\
                                         .replace('{instruction}', '')\
                                         .replace('{question}', '')\
                                         .replace('{user_message}', '')
            formatted += f"{default_user.strip()} [/INST] "
        
        return formatted
    
    def generate_text(
        self,
        image_input: Dict[str, Any],
        config: Optional[ImageCaptionConfig] = None,
        **kwargs
    ) -> ImageTextResult:
        """
        Generate text response from LLaVA.
        
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
            import time
            start_time = time.time()
            
            # Extract model inputs
            input_ids = image_input.get('input_ids')
            attention_mask = image_input.get('attention_mask')
            pixel_values = image_input.get('pixel_values')
            
            if input_ids is None or pixel_values is None:
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
                    pixel_values=pixel_values,
                    **gen_params
                )
            
            # Decode generated tokens
            generated_text = self.processor.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Extract only the assistant's response
            # Remove the prompt and any remaining special tokens
            prompt_text = image_input.get('prompt', '')
            if prompt_text and prompt_text in generated_text:
                # Extract text after the prompt
                response = generated_text.split(prompt_text)[-1]
            else:
                # Try to extract after the last [/INST] marker
                parts = generated_text.split('[/INST]')
                response = parts[-1].strip() if len(parts) > 1 else generated_text
            
            # Clean up the response
            response = self._clean_response(response)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Track performance
            self._processing_times.append(processing_time_ms)
            self._total_images_processed += 1
            
            # Update chat history
            original_prompt = image_input.get('original_prompt')
            if original_prompt and response:
                self._update_chat_history(original_prompt, response)
            
            # Prepare result
            result = ImageTextResult(
                text=response.strip(),
                model_name=self.model_name,
                model_version=self.get_version(),
                input_image_info={
                    "original_size": image_input.get('image_size'),
                    "processed_size": self.image_size,
                    "prompt": original_prompt,
                    "conversation_history_length": len(self._chat_history),
                },
                task_type=config.task_type,
                processing_time_ms=processing_time_ms,
                metadata={
                    "generation_params": gen_params,
                    "conversation_template": self.conversation_template,
                    "device": self._device,
                    "prompt": original_prompt,
                    "chat_history": self._chat_history.copy(),
                    "model_info": self.MODEL_INFO.get(self.model_name, {}),
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate text with LLaVA: {e}")
            raise RuntimeError(f"Failed to generate text with LLaVA: {e}")
    
    def _clean_response(self, response: str) -> str:
        """
        Clean up LLaVA response.
        
        Args:
            response: Raw generated response
            
        Returns:
            Cleaned response
        """
        import re
        
        # Remove special tokens
        special_tokens = ["<s>", "</s>", "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"]
        for token in special_tokens:
            response = response.replace(token, "")
        
        # Remove any remaining XML/HTML tags
        response = re.sub(r'<[^>]+>', '', response)
        
        # Remove duplicate whitespace
        response = re.sub(r'\s+', ' ', response).strip()
        
        # Remove trailing punctuation repeats
        response = re.sub(r'([.!?])\1+', r'\1', response)
        
        return response
    
    def _update_chat_history(self, user_message: str, assistant_response: str) -> None:
        """
        Update chat history with new conversation turn.
        
        Args:
            user_message: User's message
            assistant_response: Assistant's response
        """
        self._chat_history.append({
            "user": user_message,
            "assistant": assistant_response,
            "timestamp": time.time() if hasattr(time, 'time') else None,
        })
        
        # Limit history length
        if len(self._chat_history) > self._max_history_length:
            self._chat_history = self._chat_history[-self._max_history_length:]
    
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
            self.clear_chat_history()
        
        # Update config
        config_dict = self.config.dict()
        config_dict['task_type'] = ImageTaskType.MULTIMODAL_CHAT
        config_dict.update(kwargs)
        config = ImageCaptionConfig(**config_dict)
        
        # Process image with message
        image_input = self.process_image(
            image,
            prompt=message,
            conversation_history=self._chat_history,
            **kwargs
        )
        
        # Generate response
        return self.generate_text(image_input, config)
    
    def describe_image_detailed(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        include_objects: bool = True,
        include_attributes: bool = True,
        include_context: bool = True,
        include_relations: bool = True,
        **kwargs
    ) -> ImageTextResult:
        """
        Generate a detailed description of an image.
        
        Args:
            image: Input image
            include_objects: Include object descriptions
            include_attributes: Include object attributes
            include_context: Include scene context
            include_relations: Include spatial relations
            **kwargs: Additional arguments
            
        Returns:
            ImageTextResult containing detailed description
        """
        # Build detailed description prompt
        prompt_parts = []
        
        if include_context:
            prompt_parts.append("describe the overall scene and setting")
        
        if include_objects:
            prompt_parts.append("list and describe all the main objects")
        
        if include_attributes:
            prompt_parts.append("include their colors, sizes, shapes, and materials")
        
        if include_relations:
            prompt_parts.append("describe their spatial relationships and positions")
        
        prompt = f"Please provide a comprehensive description of this image. {', '.join(prompt_parts)}."
        
        # Update generation parameters for longer output
        kwargs['max_new_tokens'] = kwargs.get('max_new_tokens', 512)
        kwargs['temperature'] = kwargs.get('temperature', 0.3)
        
        return self.chat_with_image(image, prompt, **kwargs)
    
    def analyze_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        analysis_type: str = "comprehensive",
        **kwargs
    ) -> ImageTextResult:
        """
        Perform detailed analysis of an image.
        
        Args:
            image: Input image
            analysis_type: Type of analysis ("comprehensive", "technical", "artistic", "emotional")
            **kwargs: Additional arguments
            
        Returns:
            ImageTextResult containing analysis
        """
        analysis_prompts = {
            "comprehensive": (
                "Analyze this image comprehensively. "
                "Describe what you see, interpret the meaning or purpose, "
                "identify any text or symbols, and provide context or implications."
            ),
            "technical": (
                "Provide a technical analysis of this image. "
                "Describe the composition, lighting, perspective, focal points, "
                "color palette, and any technical aspects of the image creation."
            ),
            "artistic": (
                "Provide an artistic analysis of this image. "
                "Discuss the style, mood, symbolism, artistic techniques used, "
                "and the overall aesthetic impact."
            ),
            "emotional": (
                "Analyze the emotional content of this image. "
                "Describe the mood, feelings evoked, emotional cues, "
                "and the psychological impact of the visual elements."
            ),
            "practical": (
                "Provide a practical analysis of this image. "
                "Identify any useful information, instructions, warnings, "
                "or practical implications shown in the image."
            ),
        }
        
        prompt = analysis_prompts.get(
            analysis_type.lower(),
            analysis_prompts["comprehensive"]
        )
        
        # Update generation parameters for analytical output
        kwargs['temperature'] = kwargs.get('temperature', 0.1)
        kwargs['do_sample'] = kwargs.get('do_sample', False)
        kwargs['num_beams'] = kwargs.get('num_beams', 3)
        
        return self.chat_with_image(image, prompt, **kwargs)
    
    def answer_with_reasoning(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        question: str,
        show_steps: bool = True,
        **kwargs
    ) -> ImageTextResult:
        """
        Answer a question about an image with step-by-step reasoning.
        
        Args:
            image: Input image
            question: Question about the image
            show_steps: Whether to show reasoning steps
            **kwargs: Additional arguments
            
        Returns:
            ImageTextResult containing answer and reasoning
        """
        if show_steps:
            prompt = (
                f"{question} "
                "Think step by step and explain your reasoning process before giving the final answer."
            )
        else:
            prompt = question
        
        # Use complex reasoning template
        self.conversation_template = "complex_reasoning"
        
        result = self.chat_with_image(image, prompt, **kwargs)
        
        # Restore default template
        self.conversation_template = "default"
        
        return result
    
    def compare_images(
        self,
        image1: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        image2: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        comparison_type: str = "similarities_differences",
        **kwargs
    ) -> ImageTextResult:
        """
        Compare two images.
        
        Args:
            image1: First image
            image2: Second image
            comparison_type: Type of comparison
            **kwargs: Additional arguments
            
        Returns:
            ImageTextResult containing comparison analysis
        """
        comparison_prompts = {
            "similarities_differences": (
                "Compare these two images. "
                "Describe their similarities and differences in content, style, and meaning."
            ),
            "chronological": (
                "Compare these two images assuming they show a sequence. "
                "Describe what changed between them and what might have happened."
            ),
            "quality": (
                "Compare the quality of these two images. "
                "Discuss composition, lighting, focus, and overall visual appeal."
            ),
            "content": (
                "Compare the content of these two images. "
                "Describe what each shows and how they relate to each other."
            ),
        }
        
        prompt = comparison_prompts.get(
            comparison_type.lower(),
            comparison_prompts["similarities_differences"]
        )
        
        # For image comparison, we need to process both images
        # This is a simplified approach - in practice, you might need
        # to concatenate images or process them separately
        
        logger.warning(
            "Image comparison in LLaVA typically requires processing multiple images. "
            "This implementation processes only the first image. "
            "Consider concatenating images or using a different approach for true comparison."
        )
        
        return self.chat_with_image(image1, prompt, **kwargs)
    
    def clear_chat_history(self) -> None:
        """Clear the conversation history."""
        self._chat_history.clear()
        logger.info("Cleared LLaVA chat history")
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """
        Get the current conversation history.
        
        Returns:
            List of conversation turns
        """
        return self._chat_history.copy()
    
    def get_conversation_templates(self) -> Dict[str, Dict[str, str]]:
        """
        Get available conversation templates.
        
        Returns:
            Dictionary of template names to template definitions
        """
        return self.CONVERSATION_TEMPLATES.copy()
    
    def set_conversation_template(self, template_name: str) -> None:
        """
        Set the conversation template to use.
        
        Args:
            template_name: Name of the template to use
            
        Raises:
            ValueError: If template name is not found
        """
        if template_name not in self.CONVERSATION_TEMPLATES:
            raise ValueError(
                f"Unknown conversation template: {template_name}. "
                f"Available templates: {list(self.CONVERSATION_TEMPLATES.keys())}"
            )
        
        self.conversation_template = template_name
        logger.info(f"Set LLaVA conversation template to: {template_name}")
    
    def get_model_capabilities(self) -> Dict[str, Any]:
        """
        Get capabilities of the LLaVA model.
        
        Returns:
            Dictionary with model capabilities
        """
        model_info = self.MODEL_INFO.get(self.model_name, {}).copy()
        
        capabilities = {
            "model": self.model_name,
            "vision_model": model_info.get("vision_model", "CLIP-ViT-L/14"),
            "language_model": model_info.get("language_model", "Vicuna-7B"),
            "tasks": model_info.get("tasks", []),
            "max_image_size": self.image_size,
            "max_tokens": model_info.get("max_tokens", 2048),
            "supports_multi_turn_conversation": True,
            "supports_complex_reasoning": "complex_reasoning" in model_info.get("tasks", []),
            "supports_detailed_analysis": "detailed_analysis" in model_info.get("tasks", []),
            "domain": model_info.get("domain", "general"),
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
                    return f"llava-{config.model_type}"
            
            # Extract from model name
            if "1.6" in self.model_name:
                return "1.6"
            elif "1.5" in self.model_name:
                return "1.5"
            elif "med" in self.model_name.lower():
                return "med"
            
            return "1.0"
            
        except:
            return "unknown"
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the loaded LLaVA model.
        
        Returns:
            Dictionary with model information
        """
        info = super().get_model_info()
        
        # Add LLaVA-specific information
        llava_info = self.MODEL_INFO.get(self.model_name, {}).copy()
        
        info.update({
            "conversation_template": self.conversation_template,
            "chat_history_length": len(self._chat_history),
            "model_capabilities": self.get_model_capabilities(),
            "model_specs": llava_info,
            "available_templates": list(self.CONVERSATION_TEMPLATES.keys()),
        })
        
        return info
    
    def optimize_for_inference(
        self,
        use_4bit_quantization: bool = True,
        use_8bit_quantization: bool = False,
        use_flash_attention: bool = True,
        **kwargs
    ) -> None:
        """
        Optimize LLaVA model for inference performance.
        
        Args:
            use_4bit_quantization: Use 4-bit quantization (recommended for memory savings)
            use_8bit_quantization: Use 8-bit quantization
            use_flash_attention: Use Flash Attention for faster inference
            **kwargs: Additional optimization arguments
        """
        if not self._is_loaded:
            raise ValueError(f"Model {self.model_name} is not loaded. Call load() first.")
        
        logger.info(f"Optimizing LLaVA model {self.model_name} for inference")
        
        try:
            # Apply quantization
            if use_4bit_quantization:
                try:
                    from transformers import BitsAndBytesConfig
                    import bitsandbytes as bnb
                    
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    
                    # Reload model with 4-bit quantization
                    self.model = LlavaForConditionalGeneration.from_pretrained(
                        self.model_name,
                        quantization_config=bnb_config,
                        device_map="auto",
                        **self.model_kwargs
                    )
                    
                    logger.info("Applied 4-bit quantization")
                    
                except ImportError:
                    logger.warning(
                        "bitsandbytes not installed. 4-bit quantization skipped. "
                        "Install with: pip install bitsandbytes"
                    )
                except Exception as e:
                    logger.warning(f"Failed to apply 4-bit quantization: {e}")
            
            elif use_8bit_quantization:
                try:
                    # Load with 8-bit quantization
                    self.model = LlavaForConditionalGeneration.from_pretrained(
                        self.model_name,
                        load_in_8bit=True,
                        device_map="auto",
                        **self.model_kwargs
                    )
                    
                    logger.info("Applied 8-bit quantization")
                    
                except Exception as e:
                    logger.warning(f"Failed to apply 8-bit quantization: {e}")
            
            # Apply Flash Attention
            if use_flash_attention:
                try:
                    # Check if flash attention is available
                    self.model = self.model.to_bettertransformer()
                    logger.info("Applied Flash Attention optimization")
                except Exception as e:
                    logger.warning(f"Failed to apply Flash Attention: {e}")
        
        except Exception as e:
            logger.error(f"Failed to optimize model: {e}")
            raise RuntimeError(f"Failed to optimize model: {e}")
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """
        List all available LLaVA models.
        
        Returns:
            Dictionary of model names to model information
        """
        return cls.MODEL_INFO.copy()
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific LLaVA model.
        
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
        template = self.conversation_template
        history_len = len(self._chat_history)
        return (
            f"LLaVAModel(model_name={self.model_name}, "
            f"template={template}, history={history_len}, loaded={loaded}, device={device})"
        )


# Register with factory
try:
    from .base import ImageToTextModelFactory
    ImageToTextModelFactory.register_model('llava', LLaVAModel)
    logger.info("Registered LLaVAModel with ImageToTextModelFactory")
except ImportError:
    logger.warning("Could not register LLaVAModel with factory")


__all__ = [
    'LLaVAModel',
    'MODEL_INFO',
    'CONVERSATION_TEMPLATES',
]