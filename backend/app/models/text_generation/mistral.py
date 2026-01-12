"""
Mistral model implementation for text generation.

Mistral models are efficient open-source LLMs with strong performance
and innovative architectures like sliding window attention and Mixture of Experts.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import (
    BaseTextGenerationModel, TextGenerationResult, TextGenerationConfig,
    TextTaskType, TextGenerationParameters, ModelDevice
)

logger = logging.getLogger(__name__)


class MistralModel(BaseTextGenerationModel):
    """
    Mistral model wrapper for WorldBrief360.
    
    Mistral models are known for their efficiency and performance:
    - Mistral 7B: Outperforms larger models with 7B parameters
    - Mixtral 8x7B: Mixture of Experts with 47B total, 13B active parameters
    - Sliding Window Attention: Efficient long context processing
    - Strong multilingual capabilities
    
    Key features:
    - Efficient inference
    - Good reasoning capabilities
    - Strong code generation
    - Multilingual support
    """
    
    # Mistral model variants and their specifications
    MODEL_INFO = {
        # Mistral 7B models
        "mistralai/Mistral-7B-v0.1": {
            "description": "Mistral 7B base model",
            "max_tokens": 8192,
            "architecture": "decoder-only",
            "parameters": 7_000_000_000,
            "performance": "excellent",
            "memory_mb": 14000,
            "tasks": ["text_completion", "summarization", "translation", "question_answering", "coding"],
            "languages": ["en", "fr", "es", "de", "it"],
            "context_window": 8192,
            "sliding_window": 4096,
        },
        "mistralai/Mistral-7B-Instruct-v0.1": {
            "description": "Mistral 7B instruct model",
            "max_tokens": 8192,
            "architecture": "decoder-only",
            "parameters": 7_000_000_000,
            "performance": "excellent",
            "memory_mb": 14000,
            "tasks": ["conversation", "instruction_following", "question_answering", "summarization", "coding"],
            "languages": ["en", "fr", "es", "de", "it"],
            "context_window": 8192,
            "sliding_window": 4096,
        },
        "mistralai/Mistral-7B-Instruct-v0.2": {
            "description": "Mistral 7B instruct v0.2 (improved)",
            "max_tokens": 32768,
            "architecture": "decoder-only",
            "parameters": 7_000_000_000,
            "performance": "excellent",
            "memory_mb": 14000,
            "tasks": ["conversation", "instruction_following", "question_answering", "summarization", "coding"],
            "languages": ["en", "fr", "es", "de", "it", "zh", "ja", "ko"],
            "context_window": 32768,
            "sliding_window": 4096,
        },
        
        # Mixtral 8x7B models (Mixture of Experts)
        "mistralai/Mixtral-8x7B-v0.1": {
            "description": "Mixtral 8x7B base model (47B total, 13B active)",
            "max_tokens": 32768,
            "architecture": "MoE",
            "parameters": 47_000_000_000,
            "active_parameters": 13_000_000_000,
            "performance": "state_of_the_art",
            "memory_mb": 26000,
            "tasks": ["text_completion", "summarization", "translation", "question_answering", "coding", "reasoning"],
            "languages": ["en", "fr", "es", "de", "it", "zh", "ja", "ko"],
            "context_window": 32768,
            "sliding_window": 4096,
            "num_experts": 8,
            "experts_per_token": 2,
        },
        "mistralai/Mixtral-8x7B-Instruct-v0.1": {
            "description": "Mixtral 8x7B instruct model (47B total, 13B active)",
            "max_tokens": 32768,
            "architecture": "MoE",
            "parameters": 47_000_000_000,
            "active_parameters": 13_000_000_000,
            "performance": "state_of_the_art",
            "memory_mb": 26000,
            "tasks": ["conversation", "instruction_following", "question_answering", "summarization", "coding", "reasoning"],
            "languages": ["en", "fr", "es", "de", "it", "zh", "ja", "ko"],
            "context_window": 32768,
            "sliding_window": 4096,
            "num_experts": 8,
            "experts_per_token": 2,
        },
        
        # Mistral Small (newer 22B model)
        "mistralai/Mistral-Small-22B-v0.1": {
            "description": "Mistral Small 22B model",
            "max_tokens": 65536,
            "architecture": "decoder-only",
            "parameters": 22_000_000_000,
            "performance": "excellent",
            "memory_mb": 44000,
            "tasks": ["text_completion", "summarization", "translation", "question_answering", "coding", "reasoning"],
            "languages": ["en", "fr", "es", "de", "it", "zh", "ja", "ko"],
            "context_window": 65536,
            "sliding_window": 8192,
        },
        "mistralai/Mistral-Small-22B-Instruct-v0.1": {
            "description": "Mistral Small 22B instruct model",
            "max_tokens": 65536,
            "architecture": "decoder-only",
            "parameters": 22_000_000_000,
            "performance": "excellent",
            "memory_mb": 44000,
            "tasks": ["conversation", "instruction_following", "question_answering", "summarization", "coding", "reasoning"],
            "languages": ["en", "fr", "es", "de", "it", "zh", "ja", "ko"],
            "context_window": 65536,
            "sliding_window": 8192,
        },
        
        # Mistral Large (72B model)
        "mistralai/Mistral-Large-72B-v0.1": {
            "description": "Mistral Large 72B model",
            "max_tokens": 131072,
            "architecture": "decoder-only",
            "parameters": 72_000_000_000,
            "performance": "state_of_the_art",
            "memory_mb": 144000,
            "tasks": ["text_completion", "summarization", "translation", "question_answering", "coding", "reasoning", "complex_analysis"],
            "languages": ["en", "fr", "es", "de", "it", "zh", "ja", "ko", "ru", "ar"],
            "context_window": 131072,
            "sliding_window": 16384,
        },
        "mistralai/Mistral-Large-72B-Instruct-v0.1": {
            "description": "Mistral Large 72B instruct model",
            "max_tokens": 131072,
            "architecture": "decoder-only",
            "parameters": 72_000_000_000,
            "performance": "state_of_the_art",
            "memory_mb": 144000,
            "tasks": ["conversation", "instruction_following", "question_answering", "summarization", "coding", "reasoning", "complex_analysis"],
            "languages": ["en", "fr", "es", "de", "it", "zh", "ja", "ko", "ru", "ar"],
            "context_window": 131072,
            "sliding_window": 16384,
        },
        
        # Code-specific Mistral models
        "codestral-22b-v0.1": {
            "description": "Codestral 22B model for code generation",
            "max_tokens": 32768,
            "architecture": "decoder-only",
            "parameters": 22_000_000_000,
            "performance": "excellent",
            "memory_mb": 44000,
            "tasks": ["code_generation", "code_explanation", "code_debugging", "code_completion"],
            "languages": ["code", "en"],
            "context_window": 32768,
            "sliding_window": 4096,
            "supported_languages": ["python", "javascript", "java", "cpp", "go", "rust", "sql", "bash"],
        },
    }
    
    # Special tokens for Mistral
    SPECIAL_TOKENS = {
        "bos": "<s>",
        "eos": "</s>",
        "system": "[INST] ",
        "user": "",
        "assistant": "",
        "end": " [/INST]",
    }
    
    # Task-specific prompt templates
    PROMPT_TEMPLATES = {
        "conversation": "[INST] {system_prompt}\n\n{user_prompt} [/INST]",
        "code_generation": "[INST] You are an expert programming assistant. Write efficient, well-commented code to solve this problem:\n\n{problem}\n\nProvide the complete solution in {language}. [/INST]",
        "code_explanation": "[INST] Explain this code clearly and thoroughly:\n```\n{code}\n```\n\nExplain what it does, how it works, and any potential issues. [/INST]",
        "summarization": "[INST] Summarize this text concisely:\n\n{text}\n\nProvide a clear and concise summary. [/INST]",
        "translation": "[INST] Translate this text accurately from {source_language} to {target_language}:\n\n{text} [/INST]",
        "question_answering": "[INST] Answer this question accurately and thoroughly:\n\n{question}\n\nProvide a detailed and well-structured answer. [/INST]",
        "creative_writing": "[INST] Write a {genre} about {topic} with these requirements:\n{requirements}\n\nMake it engaging and well-structured. [/INST]",
        "mathematical_reasoning": "[INST] Solve this mathematical problem step by step:\n\n{problem}\n\nShow your reasoning clearly. [/INST]",
        "text_completion": "[INST] Complete this text naturally:\n\n{text} [/INST]",
    }
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        config: Optional[TextGenerationConfig] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize Mistral model.
        
        Args:
            model_name: Name of the Mistral model
            config: Configuration for text generation
            model_kwargs: Additional arguments for model initialization
            tokenizer_kwargs: Additional arguments for tokenizer initialization
            **kwargs: Additional arguments
        """
        # Update config with Mistral-specific defaults
        if config is None:
            config = TextGenerationConfig()
        
        # Get model info
        model_info = self.MODEL_INFO.get(model_name, {})
        context_window = model_info.get("context_window", 8192)
        
        # Adjust max tokens based on model capabilities
        if config.generation_params.max_new_tokens > context_window:
            config.generation_params.max_new_tokens = context_window
        
        # Set task type based on model name
        if "codestral" in model_name.lower() or "code" in model_name.lower():
            config.task_type = TextTaskType.CODE_GENERATION
        elif "instruct" in model_name.lower():
            config.task_type = TextTaskType.CONVERSATION
        
        super().__init__(model_name, config, **kwargs)
        
        # Mistral-specific initialization
        self.model_kwargs = model_kwargs or {}
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        
        # Check model type
        self._is_mixtral = "mixtral" in model_name.lower()
        self._is_codestral = "codestral" in model_name.lower()
        self._is_small = "small" in model_name.lower()
        self._is_large = "large" in model_name.lower()
        self._is_instruct = "instruct" in model_name.lower()
        
        # Conversation history for instruct models
        self._conversation_history: List[Dict[str, str]] = []
        self._max_history_length = 10
        
        # System prompt
        self._system_prompt = self._get_default_system_prompt()
        
        # Generation parameters optimized for Mistral
        self._default_generation_params = {
            "max_new_tokens": config.generation_params.max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.0,
            "do_sample": True,
            "num_beams": 1,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "early_stopping": False,
        }
        
        # Adjust for code generation
        if self._is_codestral:
            self._default_generation_params.update({
                "temperature": 0.2,
                "top_p": 0.95,
                "repetition_penalty": 1.05,
            })
        
        # Adjust for large models
        if self._is_large:
            self._default_generation_params.update({
                "temperature": 0.8,
                "top_p": 0.95,
            })
        
        logger.info(f"Initialized Mistral model: {model_name} "
                   f"(Mixtral: {self._is_mixtral}, Codestral: {self._is_codestral}, "
                   f"Small: {self._is_small}, Large: {self._is_large}, Instruct: {self._is_instruct})")
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt based on model type."""
        if self._is_codestral:
            return "You are Codestral, a state-of-the-art AI model specialized in code generation and programming assistance. Provide accurate, efficient, and well-commented code solutions."
        elif self._is_instruct:
            return "You are a helpful AI assistant."
        else:
            return ""
    
    def load(self) -> None:
        """
        Load the Mistral model and tokenizer.
        
        Raises:
            RuntimeError: If model loading fails
        """
        if self._is_loaded:
            logger.warning(f"Model {self.model_name} is already loaded")
            return
        
        try:
            logger.info(f"Loading Mistral model: {self.model_name}")
            
            # Determine device with memory considerations
            if torch.cuda.is_available():
                self._device = "cuda"
                
                # Check memory requirements
                model_info = self.MODEL_INFO.get(self.model_name, {})
                required_memory_mb = model_info.get("memory_mb", 14000)
                
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
                        
                        # For large models, use device_map
                        if required_memory_mb > 30000:
                            self.model_kwargs["device_map"] = "auto"
                
                except Exception as e:
                    logger.warning(f"Could not check GPU memory: {e}")
            else:
                self._device = "cpu"
                logger.warning(
                    "Mistral models are large and run slowly on CPU. "
                    "Consider using GPU for reasonable performance."
                )
            
            # Load tokenizer
            logger.info(f"Loading tokenizer for {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                **self.tokenizer_kwargs
            )
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            logger.info(f"Loading Mistral model for {self.model_name}")
            
            # Set default model kwargs
            self.model_kwargs.setdefault("torch_dtype", torch.float16 if self._device == "cuda" else torch.float32)
            
            # For Mixtral models, use specific optimizations
            if self._is_mixtral:
                self.model_kwargs.setdefault("torch_dtype", torch.float16)
                self.model_kwargs.setdefault("device_map", "auto")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **self.model_kwargs
            )
            
            # Move to device if not using device_map
            if "device_map" not in self.model_kwargs:
                self.model.to(self._device)
            
            self.model.eval()  # Set to evaluation mode
            
            self._is_loaded = True
            
            # Get model info
            model_info = self.MODEL_INFO.get(self.model_name, {})
            self.context_window = model_info.get("context_window", 8192)
            self.sliding_window = model_info.get("sliding_window", 4096)
            
            logger.info(f"Successfully loaded Mistral model {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load Mistral model {self.model_name}: {e}")
            raise RuntimeError(f"Failed to load Mistral model {self.model_name}: {e}")
    
    def process_text(
        self,
        text: str,
        prompt_template: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process text for Mistral model input.
        
        Args:
            text: Input text or prompt
            prompt_template: Optional template to format the prompt
            conversation_history: Optional previous conversation turns
            system_prompt: Optional system prompt
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
            # Format prompt
            formatted_prompt = self._format_prompt(
                text=text,
                prompt_template=prompt_template,
                conversation_history=conversation_history,
                system_prompt=system_prompt,
                **kwargs
            )
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.context_window - self.config.generation_params.max_new_tokens,
                **kwargs
            )
            
            # Move to device
            if self._device.startswith("cuda"):
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            # Prepare result
            result = {
                'input_ids': inputs.get('input_ids'),
                'attention_mask': inputs.get('attention_mask'),
                'prompt': formatted_prompt,
                'original_text': text,
                'prompt_length': inputs['input_ids'].shape[1],
                'model_inputs': inputs,
            }
            
            # Add conversation history if provided
            if conversation_history:
                result['conversation_history'] = conversation_history.copy()
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process text for Mistral: {e}")
            raise RuntimeError(f"Failed to process text for Mistral: {e}")
    
    def _format_prompt(
        self,
        text: str,
        prompt_template: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Format prompt for Mistral model.
        
        Args:
            text: User text
            prompt_template: Optional template name or custom template
            conversation_history: Previous conversation turns
            system_prompt: System prompt to use
            **kwargs: Additional formatting arguments
            
        Returns:
            Formatted prompt string
        """
        # For non-instruct models, use simple formatting
        if not self._is_instruct:
            if prompt_template and prompt_template in self.PROMPT_TEMPLATES:
                template = self.PROMPT_TEMPLATES[prompt_template]
                try:
                    return template.format(text=text, **kwargs)
                except KeyError:
                    logger.warning(f"Failed to format template {prompt_template}, using text directly")
            return text
        
        # For instruct models with conversation history
        if conversation_history:
            return self._format_chat_prompt(
                text=text,
                conversation_history=conversation_history,
                system_prompt=system_prompt,
                **kwargs
            )
        
        # For single-turn instruct prompts
        system_prompt = system_prompt or self._system_prompt
        
        if prompt_template:
            # Check if it's a predefined template
            if prompt_template in self.PROMPT_TEMPLATES:
                template = self.PROMPT_TEMPLATES[prompt_template]
                
                # Format with kwargs
                format_kwargs = {"text": text, "system_prompt": system_prompt, **kwargs}
                try:
                    return template.format(**format_kwargs)
                except KeyError:
                    # If template formatting fails, use default
                    logger.warning(f"Failed to format template {prompt_template}, using default")
            
            # Use custom template
            else:
                try:
                    return prompt_template.format(text=text, system_prompt=system_prompt, **kwargs)
                except:
                    logger.warning(f"Failed to format custom template, using default format")
        
        # Default format for instruct models
        if system_prompt:
            return f"{self.SPECIAL_TOKENS['system']}{system_prompt}\n\n{text}{self.SPECIAL_TOKENS['end']}"
        else:
            return f"{self.SPECIAL_TOKENS['system']}{text}{self.SPECIAL_TOKENS['end']}"
    
    def _format_chat_prompt(
        self,
        text: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Format chat prompt for Mistral instruct models.
        
        Args:
            text: Current user message
            conversation_history: Previous conversation turns
            system_prompt: System prompt
            **kwargs: Additional arguments
            
        Returns:
            Formatted chat prompt
        """
        system_prompt = system_prompt or self._system_prompt
        
        messages = []
        
        # Add system prompt only for first message
        if not conversation_history or len(conversation_history) == 0:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        if conversation_history:
            for turn in conversation_history[-self._max_history_length:]:
                user_msg = turn.get('user', '')
                assistant_msg = turn.get('assistant', '')
                
                if user_msg:
                    messages.append({"role": "user", "content": user_msg})
                if assistant_msg:
                    messages.append({"role": "assistant", "content": assistant_msg})
        
        # Add current message
        messages.append({"role": "user", "content": text})
        
        # Format messages using tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"Failed to apply chat template: {e}")
        
        # Fallback: manual formatting
        formatted = []
        for i, message in enumerate(messages):
            role = message["role"]
            content = message["content"]
            
            if role == "system" and i == 0:
                formatted.append(f"{self.SPECIAL_TOKENS['system']}{content}\n\n")
            elif role == "user":
                if i == len(messages) - 1:
                    # Last message (current)
                    formatted.append(f"{content}{self.SPECIAL_TOKENS['end']}")
                else:
                    formatted.append(f"{content}")
            elif role == "assistant":
                formatted.append(f"{content}{self.SPECIAL_TOKENS['eos']}")
        
        return "".join(formatted)
    
    def generate_text(
        self,
        text_input: Dict[str, Any],
        config: Optional[TextGenerationConfig] = None,
        **kwargs
    ) -> TextGenerationResult:
        """
        Generate text response from Mistral.
        
        Args:
            text_input: Processed text input
            config: Configuration for text generation
            **kwargs: Additional generation arguments
            
        Returns:
            TextGenerationResult containing generated response and metadata
            
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
            input_ids = text_input.get('input_ids')
            attention_mask = text_input.get('attention_mask')
            
            if input_ids is None:
                raise ValueError("Processed input missing required fields")
            
            # Prepare generation parameters
            gen_params = self._default_generation_params.copy()
            
            # Update with config
            if config.generation_params:
                gen_params.update({
                    "max_new_tokens": config.generation_params.max_new_tokens,
                    "temperature": config.generation_params.temperature,
                    "top_p": config.generation_params.top_p,
                    "top_k": config.generation_params.top_k,
                    "repetition_penalty": config.generation_params.repetition_penalty,
                    "do_sample": config.generation_params.do_sample,
                    "num_beams": config.generation_params.num_beams,
                    "length_penalty": config.generation_params.length_penalty,
                    "no_repeat_ngram_size": config.generation_params.no_repeat_ngram_size,
                    "early_stopping": config.generation_params.early_stopping,
                })
            
            # Update with kwargs
            gen_params.update(kwargs)
            
            # Set seed if provided
            if config.generation_params.seed is not None:
                torch.manual_seed(config.generation_params.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(config.generation_params.seed)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_params
                )
            
            # Decode generated tokens
            generated_tokens = outputs[0][input_ids.shape[1]:]  # Remove input tokens
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Clean up the response
            generated_text = self._clean_response(generated_text)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Track performance
            self._processing_times.append(processing_time_ms)
            self._total_texts_processed += 1
            
            # Update conversation history for instruct models
            original_text = text_input.get('original_text')
            if self._is_instruct and original_text and generated_text:
                self._update_conversation_history(original_text, generated_text)
            
            # Prepare result
            result = TextGenerationResult(
                text=generated_text.strip(),
                model_name=self.model_name,
                model_version=self.get_version(),
                task_type=config.task_type,
                processing_time_ms=processing_time_ms,
                tokens_generated=len(generated_tokens),
                tokens_processed=input_ids.shape[1],
                input_info={
                    "prompt_length": input_ids.shape[1],
                    "original_text": original_text,
                    "conversation_history_length": len(self._conversation_history),
                },
                metadata={
                    "generation_params": gen_params,
                    "is_mixtral": self._is_mixtral,
                    "is_codestral": self._is_codestral,
                    "is_small": self._is_small,
                    "is_large": self._is_large,
                    "is_instruct": self._is_instruct,
                    "device": self._device,
                    "conversation_history": self._conversation_history.copy(),
                    "model_info": self.MODEL_INFO.get(self.model_name, {}),
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate text with Mistral: {e}")
            raise RuntimeError(f"Failed to generate text with Mistral: {e}")
    
    def _clean_response(self, response: str) -> str:
        """
        Clean up Mistral response.
        
        Args:
            response: Raw generated response
            
        Returns:
            Cleaned response
        """
        import re
        
        # Remove special tokens
        for token in self.SPECIAL_TOKENS.values():
            response = response.replace(token, "")
        
        # Remove end of text markers
        response = response.replace("<|endoftext|>", "")
        response = response.replace("</s>", "")
        
        # Remove any formatting artifacts
        response = re.sub(r'\[INST\].*?\[/INST\]', '', response)
        
        # Remove trailing whitespace and newlines
        response = response.strip()
        
        # Remove any code block markers if they're incomplete
        response = re.sub(r'```[^`]*$', '', response)  # Remove trailing incomplete code blocks
        
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
    
    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        clear_history: bool = False,
        **kwargs
    ) -> TextGenerationResult:
        """
        Have a conversation with the Mistral instruct model.
        
        Args:
            message: User message
            system_prompt: Optional system prompt
            clear_history: Whether to clear conversation history
            **kwargs: Additional arguments
            
        Returns:
            TextGenerationResult containing response and metadata
        """
        if not self._is_instruct:
            logger.warning(f"Model {self.model_name} is not an instruct model. Using anyway.")
        
        if clear_history:
            self.clear_conversation_history()
        
        # Update config for conversation
        config_dict = self.config.dict()
        config_dict['task_type'] = TextTaskType.CONVERSATION
        config_dict.update(kwargs)
        config = TextGenerationConfig(**config_dict)
        
        # Process message with conversation history
        text_input = self.process_text(
            message,
            conversation_history=self._conversation_history,
            system_prompt=system_prompt,
            **kwargs
        )
        
        # Generate response
        return self.generate_text(text_input, config)
    
    def generate_code(
        self,
        problem: str,
        language: str = "python",
        include_comments: bool = True,
        include_tests: bool = False,
        **kwargs
    ) -> TextGenerationResult:
        """
        Generate code for a given problem.
        
        Args:
            problem: Problem description or requirements
            language: Programming language
            include_comments: Whether to include comments
            include_tests: Whether to include test cases
            **kwargs: Additional arguments
            
        Returns:
            TextGenerationResult containing generated code
        """
        if not self._is_codestral:
            logger.warning(f"Model {self.model_name} is not specialized for code generation.")
        
        # Build code generation prompt
        prompt_parts = [f"Write {language} code to solve: {problem}"]
        
        if include_comments:
            prompt_parts.append("Include detailed comments.")
        
        if include_tests:
            prompt_parts.append("Include test cases.")
        
        prompt = " ".join(prompt_parts)
        
        # Update generation parameters for code
        kwargs['temperature'] = kwargs.get('temperature', 0.2)
        kwargs['do_sample'] = kwargs.get('do_sample', True)
        
        return self.generate(prompt, task_type=TextTaskType.CODE_GENERATION, **kwargs)
    
    def summarize_long_text(
        self,
        text: str,
        max_summary_length: int = 500,
        **kwargs
    ) -> TextGenerationResult:
        """
        Summarize long text using Mistral's extended context.
        
        Args:
            text: Long text to summarize
            max_summary_length: Maximum length of summary in tokens
            **kwargs: Additional arguments
            
        Returns:
            TextGenerationResult containing summary
        """
        # Build summarization prompt
        prompt = f"Summarize this text in about {max_summary_length} words:\n\n{text}"
        
        # Update generation parameters
        kwargs['max_new_tokens'] = min(kwargs.get('max_new_tokens', max_summary_length), 2000)
        kwargs['temperature'] = kwargs.get('temperature', 0.3)
        
        return self.generate(prompt, task_type=TextTaskType.SUMMARIZATION, **kwargs)
    
    def translate_text(
        self,
        text: str,
        source_language: str = "auto",
        target_language: str = "English",
        **kwargs
    ) -> TextGenerationResult:
        """
        Translate text between languages.
        
        Args:
            text: Text to translate
            source_language: Source language (or "auto" for detection)
            target_language: Target language
            **kwargs: Additional arguments
            
        Returns:
            TextGenerationResult containing translation
        """
        # Build translation prompt
        if source_language.lower() == "auto":
            prompt = f"Translate this text to {target_language}:\n\n{text}"
        else:
            prompt = f"Translate this text from {source_language} to {target_language}:\n\n{text}"
        
        # Update generation parameters for translation
        kwargs['temperature'] = kwargs.get('temperature', 0.3)
        kwargs['do_sample'] = kwargs.get('do_sample', False)  # More deterministic for translation
        
        return self.generate(prompt, task_type=TextTaskType.TRANSLATION, **kwargs)
    
    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self._conversation_history.clear()
        logger.info("Cleared Mistral conversation history")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the current conversation history.
        
        Returns:
            List of conversation turns
        """
        return self._conversation_history.copy()
    
    def set_system_prompt(self, system_prompt: str) -> None:
        """
        Set the system prompt.
        
        Args:
            system_prompt: New system prompt
        """
        self._system_prompt = system_prompt
        logger.info("Updated Mistral system prompt")
    
    def get_system_prompt(self) -> str:
        """
        Get the current system prompt.
        
        Returns:
            Current system prompt
        """
        return self._system_prompt
    
    def unload(self) -> None:
        """Unload the model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._is_loaded = False
        logger.info(f"Unloaded Mistral model {self.model_name}")
    
    def get_model_capabilities(self) -> Dict[str, Any]:
        """
        Get capabilities of the Mistral model.
        
        Returns:
            Dictionary with model capabilities
        """
        model_info = self.MODEL_INFO.get(self.model_name, {}).copy()
        
        capabilities = {
            "model": self.model_name,
            "architecture": model_info.get("architecture", "decoder-only"),
            "tasks": model_info.get("tasks", []),
            "context_window": model_info.get("context_window", 8192),
            "sliding_window": model_info.get("sliding_window", 4096),
            "max_tokens": model_info.get("max_tokens", 8192),
            "is_mixtral": self._is_mixtral,
            "is_codestral": self._is_codestral,
            "is_small": self._is_small,
            "is_large": self._is_large,
            "is_instruct": self._is_instruct,
            "supports_code": self._is_codestral or "code" in model_info.get("tasks", []),
            "supports_long_context": model_info.get("context_window", 8192) > 16000,
            "multilingual": True,  # Mistral models have strong multilingual support
            "supported_languages": model_info.get("languages", ["en"]),
            "model_parameters": model_info.get("parameters", 0),
            "active_parameters": model_info.get("active_parameters", model_info.get("parameters", 0)),
            "memory_requirements_mb": model_info.get("memory_mb", 14000),
            "num_experts": model_info.get("num_experts", 1),
            "experts_per_token": model_info.get("experts_per_token", 1),
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
            # Extract from model name
            model_lower = self.model_name.lower()
            
            if "mixtral" in model_lower:
                return "mixtral-8x7b"
            elif "codestral" in model_lower:
                return "codestral-22b"
            elif "small" in model_lower:
                return "mistral-small-22b"
            elif "large" in model_lower:
                return "mistral-large-72b"
            elif "v0.2" in model_lower:
                return "mistral-7b-v0.2"
            elif "v0.1" in model_lower:
                return "mistral-7b-v0.1"
            else:
                return "mistral"
            
        except:
            return "unknown"
    
    def list_prompt_templates(self) -> List[str]:
        """
        List available prompt templates.
        
        Returns:
            List of template names
        """
        return list(self.PROMPT_TEMPLATES.keys())
    
    def add_prompt_template(self, name: str, template: str) -> None:
        """
        Add a custom prompt template.
        
        Args:
            name: Name of the template
            template: Template string
        """
        self.PROMPT_TEMPLATES[name] = template
        logger.info(f"Added prompt template: {name}")
    
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