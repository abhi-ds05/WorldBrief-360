"""
Llama model implementation for text generation.

Llama models are powerful open-source LLMs from Meta, with strong performance
across a wide range of tasks including conversation, reasoning, and coding.
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


class LlamaModel(BaseTextGenerationModel):
    """
    Llama model wrapper for WorldBrief360.
    
    Llama models are state-of-the-art open-source LLMs with:
    - Strong general language understanding
    - Good reasoning capabilities
    - Support for long contexts (up to 32K tokens)
    - Code generation capabilities
    - Multilingual support
    
    Model variants:
    - Llama 2: 7B, 13B, 70B parameters
    - Llama 3: 8B, 70B, 405B parameters (coming soon)
    - Code Llama: Specialized for code generation
    - Llama Guard: For safety and moderation
    """
    
    # Llama model variants and their specifications
    MODEL_INFO = {
        # Llama 2 models
        "meta-llama/Llama-2-7b-hf": {
            "description": "Llama 2 7B base model",
            "max_tokens": 4096,
            "architecture": "decoder-only",
            "parameters": 7_000_000_000,
            "performance": "good",
            "memory_mb": 14000,
            "tasks": ["text_completion", "summarization", "translation", "question_answering"],
            "languages": ["en"],
            "context_window": 4096,
            "requires_auth": True,
        },
        "meta-llama/Llama-2-13b-hf": {
            "description": "Llama 2 13B base model",
            "max_tokens": 4096,
            "architecture": "decoder-only",
            "parameters": 13_000_000_000,
            "performance": "very_good",
            "memory_mb": 26000,
            "tasks": ["text_completion", "summarization", "translation", "question_answering", "reasoning"],
            "languages": ["en"],
            "context_window": 4096,
            "requires_auth": True,
        },
        "meta-llama/Llama-2-70b-hf": {
            "description": "Llama 2 70B base model",
            "max_tokens": 4096,
            "architecture": "decoder-only",
            "parameters": 70_000_000_000,
            "performance": "excellent",
            "memory_mb": 140000,
            "tasks": ["text_completion", "summarization", "translation", "question_answering", "reasoning", "coding"],
            "languages": ["en"],
            "context_window": 4096,
            "requires_auth": True,
        },
        
        # Llama 2 Chat models
        "meta-llama/Llama-2-7b-chat-hf": {
            "description": "Llama 2 7B chat model",
            "max_tokens": 4096,
            "architecture": "decoder-only",
            "parameters": 7_000_000_000,
            "performance": "good",
            "memory_mb": 14000,
            "tasks": ["conversation", "instruction_following", "question_answering", "summarization"],
            "languages": ["en"],
            "context_window": 4096,
            "requires_auth": True,
        },
        "meta-llama/Llama-2-13b-chat-hf": {
            "description": "Llama 2 13B chat model",
            "max_tokens": 4096,
            "architecture": "decoder-only",
            "parameters": 13_000_000_000,
            "performance": "very_good",
            "memory_mb": 26000,
            "tasks": ["conversation", "instruction_following", "question_answering", "summarization", "reasoning"],
            "languages": ["en"],
            "context_window": 4096,
            "requires_auth": True,
        },
        "meta-llama/Llama-2-70b-chat-hf": {
            "description": "Llama 2 70B chat model",
            "max_tokens": 4096,
            "architecture": "decoder-only",
            "parameters": 70_000_000_000,
            "performance": "excellent",
            "memory_mb": 140000,
            "tasks": ["conversation", "instruction_following", "question_answering", "summarization", "reasoning", "coding"],
            "languages": ["en"],
            "context_window": 4096,
            "requires_auth": True,
        },
        
        # Code Llama models
        "codellama/CodeLlama-7b-hf": {
            "description": "Code Llama 7B base model for code generation",
            "max_tokens": 16384,
            "architecture": "decoder-only",
            "parameters": 7_000_000_000,
            "performance": "good",
            "memory_mb": 14000,
            "tasks": ["code_generation", "code_explanation", "code_debugging", "text_completion"],
            "languages": ["code", "en"],
            "context_window": 16384,
            "requires_auth": False,
        },
        "codellama/CodeLlama-13b-hf": {
            "description": "Code Llama 13B base model for code generation",
            "max_tokens": 16384,
            "architecture": "decoder-only",
            "parameters": 13_000_000_000,
            "performance": "very_good",
            "memory_mb": 26000,
            "tasks": ["code_generation", "code_explanation", "code_debugging", "text_completion"],
            "languages": ["code", "en"],
            "context_window": 16384,
            "requires_auth": False,
        },
        "codellama/CodeLlama-34b-hf": {
            "description": "Code Llama 34B base model for code generation",
            "max_tokens": 16384,
            "architecture": "decoder-only",
            "parameters": 34_000_000_000,
            "performance": "excellent",
            "memory_mb": 68000,
            "tasks": ["code_generation", "code_explanation", "code_debugging", "text_completion"],
            "languages": ["code", "en"],
            "context_window": 16384,
            "requires_auth": False,
        },
        
        # Code Llama Instruct models
        "codellama/CodeLlama-7b-Instruct-hf": {
            "description": "Code Llama 7B instruct model for code tasks",
            "max_tokens": 16384,
            "architecture": "decoder-only",
            "parameters": 7_000_000_000,
            "performance": "good",
            "memory_mb": 14000,
            "tasks": ["code_generation", "code_explanation", "code_debugging", "instruction_following"],
            "languages": ["code", "en"],
            "context_window": 16384,
            "requires_auth": False,
        },
        "codellama/CodeLlama-13b-Instruct-hf": {
            "description": "Code Llama 13B instruct model for code tasks",
            "max_tokens": 16384,
            "architecture": "decoder-only",
            "parameters": 13_000_000_000,
            "performance": "very_good",
            "memory_mb": 26000,
            "tasks": ["code_generation", "code_explanation", "code_debugging", "instruction_following"],
            "languages": ["code", "en"],
            "context_window": 16384,
            "requires_auth": False,
        },
        "codellama/CodeLlama-34b-Instruct-hf": {
            "description": "Code Llama 34B instruct model for code tasks",
            "max_tokens": 16384,
            "architecture": "decoder-only",
            "parameters": 34_000_000_000,
            "performance": "excellent",
            "memory_mb": 68000,
            "tasks": ["code_generation", "code_explanation", "code_debugging", "instruction_following"],
            "languages": ["code", "en"],
            "context_window": 16384,
            "requires_auth": False,
        },
        
        # Llama 2 Long models (extended context)
        "togethercomputer/LLaMA-2-7B-32K": {
            "description": "Llama 2 7B with 32K context window",
            "max_tokens": 32768,
            "architecture": "decoder-only",
            "parameters": 7_000_000_000,
            "performance": "good",
            "memory_mb": 28000,
            "tasks": ["text_completion", "summarization", "document_analysis"],
            "languages": ["en"],
            "context_window": 32768,
            "requires_auth": False,
        },
        "togethercomputer/LLaMA-2-13B-32K": {
            "description": "Llama 2 13B with 32K context window",
            "max_tokens": 32768,
            "architecture": "decoder-only",
            "parameters": 13_000_000_000,
            "performance": "very_good",
            "memory_mb": 52000,
            "tasks": ["text_completion", "summarization", "document_analysis", "reasoning"],
            "languages": ["en"],
            "context_window": 32768,
            "requires_auth": False,
        },
        
        # Llama 3 models (when available)
        "meta-llama/Meta-Llama-3-8B": {
            "description": "Llama 3 8B base model",
            "max_tokens": 8192,
            "architecture": "decoder-only",
            "parameters": 8_000_000_000,
            "performance": "very_good",
            "memory_mb": 16000,
            "tasks": ["text_completion", "summarization", "translation", "question_answering", "reasoning"],
            "languages": ["en"],
            "context_window": 8192,
            "requires_auth": True,
        },
        "meta-llama/Meta-Llama-3-70B": {
            "description": "Llama 3 70B base model",
            "max_tokens": 8192,
            "architecture": "decoder-only",
            "parameters": 70_000_000_000,
            "performance": "excellent",
            "memory_mb": 140000,
            "tasks": ["text_completion", "summarization", "translation", "question_answering", "reasoning", "coding"],
            "languages": ["en"],
            "context_window": 8192,
            "requires_auth": True,
        },
    }
    
    # Special tokens for Llama
    SPECIAL_TOKENS = {
        "bos": "<s>",
        "eos": "</s>",
        "system": "[INST] <<SYS>>\n",
        "system_end": "\n<</SYS>>\n\n",
        "user": "",
        "user_end": " [/INST]",
        "assistant": "",
        "assistant_end": "</s>",
    }
    
    # Task-specific prompt templates
    PROMPT_TEMPLATES = {
        "conversation": "[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]",
        "code_generation": "[INST] <<SYS>>\nYou are a helpful coding assistant. Write efficient, well-commented code.\n<</SYS>>\n\nWrite code to solve: {problem}\n\nProvide the complete solution in {language}. [/INST]",
        "code_explanation": "[INST] <<SYS>>\nYou are a helpful coding assistant. Explain code clearly and thoroughly.\n<</SYS>>\n\nExplain this code:\n```\n{code}\n```\n\nExplain what it does, how it works, and any potential issues. [/INST]",
        "summarization": "[INST] <<SYS>>\nYou are a helpful summarization assistant. Provide concise, accurate summaries.\n<</SYS>>\n\nSummarize this text:\n\n{text}\n\nProvide a concise summary. [/INST]",
        "translation": "[INST] <<SYS>>\nYou are a helpful translation assistant. Provide accurate translations.\n<</SYS>>\n\nTranslate this text from {source_language} to {target_language}:\n\n{text} [/INST]",
        "question_answering": "[INST] <<SYS>>\nYou are a helpful question-answering assistant. Provide accurate, detailed answers.\n<</SYS>>\n\nAnswer this question:\n\n{question}\n\nProvide a detailed and accurate answer. [/INST]",
        "creative_writing": "[INST] <<SYS>>\nYou are a creative writing assistant. Write engaging, well-structured content.\n<</SYS>>\n\nWrite a {genre} about {topic} with these requirements:\n{requirements} [/INST]",
        "mathematical_reasoning": "[INST] <<SYS>>\nYou are a mathematical reasoning assistant. Show step-by-step solutions.\n<</SYS>>\n\nSolve this mathematical problem:\n\n{problem}\n\nShow your reasoning step by step. [/INST]",
        "text_completion": "[INST] <<SYS>>\nYou are a helpful text completion assistant.\n<</SYS>>\n\nComplete this text:\n\n{text} [/INST]",
    }
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        config: Optional[TextGenerationConfig] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize Llama model.
        
        Args:
            model_name: Name of the Llama model
            config: Configuration for text generation
            model_kwargs: Additional arguments for model initialization
            tokenizer_kwargs: Additional arguments for tokenizer initialization
            **kwargs: Additional arguments
        """
        # Update config with Llama-specific defaults
        if config is None:
            config = TextGenerationConfig()
        
        # Get model info
        model_info = self.MODEL_INFO.get(model_name, {})
        context_window = model_info.get("context_window", 4096)
        requires_auth = model_info.get("requires_auth", True)
        
        # Check authentication requirement
        if requires_auth:
            self._check_auth_requirements()
        
        # Adjust max tokens based on model capabilities
        if config.generation_params.max_new_tokens > context_window:
            config.generation_params.max_new_tokens = context_window
        
        # Set task type based on model name
        if "code" in model_name.lower() or "codellama" in model_name.lower():
            config.task_type = TextTaskType.CODE_GENERATION
        elif "chat" in model_name.lower() or "instruct" in model_name.lower():
            config.task_type = TextTaskType.CONVERSATION
        
        super().__init__(model_name, config, **kwargs)
        
        # Llama-specific initialization
        self.model_kwargs = model_kwargs or {}
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        
        # Check model type
        self._is_llama2 = "llama-2" in model_name.lower()
        self._is_llama3 = "llama-3" in model_name.lower() or "meta-llama-3" in model_name.lower()
        self._is_codelama = "code" in model_name.lower() or "codellama" in model_name.lower()
        self._is_chat = "chat" in model_name.lower() or "instruct" in model_name.lower()
        self._is_long = "32k" in model_name.lower() or "long" in model_name.lower()
        
        # Conversation history for chat models
        self._conversation_history: List[Dict[str, str]] = []
        self._max_history_length = 10
        
        # System prompt
        self._system_prompt = self._get_default_system_prompt()
        
        # Generation parameters optimized for Llama
        self._default_generation_params = {
            "max_new_tokens": config.generation_params.max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "num_beams": 1,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
        }
        
        # Adjust for code generation
        if self._is_codelama:
            self._default_generation_params.update({
                "temperature": 0.2,
                "top_p": 0.95,
                "repetition_penalty": 1.05,
            })
        
        # Adjust for long context models
        if self._is_long:
            self._default_generation_params.update({
                "max_new_tokens": min(config.generation_params.max_new_tokens, 8192),
            })
        
        logger.info(f"Initialized Llama model: {model_name} "
                   f"(Llama2: {self._is_llama2}, Llama3: {self._is_llama3}, "
                   f"CodeLlama: {self._is_codelama}, Chat: {self._is_chat}, Long: {self._is_long})")
    
    def _check_auth_requirements(self) -> None:
        """Check if authentication requirements are met for Llama models."""
        try:
            from huggingface_hub import login
            # Check if we have a token or can access gated models
            import os
            if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
                logger.warning(
                    "Llama models from Meta require authentication. "
                    "Set HF_TOKEN environment variable or use huggingface_hub.login()"
                )
        except ImportError:
            pass
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt based on model type."""
        if self._is_codelama:
            return "You are a helpful coding assistant. Write efficient, well-commented code. Explain your solutions clearly."
        elif self._is_chat:
            return "You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible."
        else:
            return "You are a helpful AI assistant."
    
    def load(self) -> None:
        """
        Load the Llama model and tokenizer.
        
        Raises:
            RuntimeError: If model loading fails
        """
        if self._is_loaded:
            logger.warning(f"Model {self.model_name} is already loaded")
            return
        
        try:
            logger.info(f"Loading Llama model: {self.model_name}")
            
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
                    "Llama models are large and run slowly on CPU. "
                    "Consider using GPU for reasonable performance."
                )
            
            # Load tokenizer
            logger.info(f"Loading tokenizer for {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                **self.tokenizer_kwargs
            )
            
            # Set padding token if not set (important for batch processing)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            logger.info(f"Loading Llama model for {self.model_name}")
            
            # Set default model kwargs
            self.model_kwargs.setdefault("torch_dtype", torch.float16 if self._device == "cuda" else torch.float32)
            
            # For Llama 3 models, use specific settings
            if self._is_llama3:
                self.model_kwargs.setdefault("torch_dtype", torch.bfloat16)
            
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
            self.context_window = model_info.get("context_window", 4096)
            
            logger.info(f"Successfully loaded Llama model {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load Llama model {self.model_name}: {e}")
            raise RuntimeError(f"Failed to load Llama model {self.model_name}: {e}")
    
    def process_text(
        self,
        text: str,
        prompt_template: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process text for Llama model input.
        
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
            logger.error(f"Failed to process text for Llama: {e}")
            raise RuntimeError(f"Failed to process text for Llama: {e}")
    
    def _format_prompt(
        self,
        text: str,
        prompt_template: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Format prompt for Llama model.
        
        Args:
            text: User text
            prompt_template: Optional template name or custom template
            conversation_history: Previous conversation turns
            system_prompt: System prompt to use
            **kwargs: Additional formatting arguments
            
        Returns:
            Formatted prompt string
        """
        system_prompt = system_prompt or self._system_prompt
        
        # Handle chat models with conversation format
        if self._is_chat and conversation_history:
            return self._format_chat_prompt(
                text=text,
                conversation_history=conversation_history,
                system_prompt=system_prompt,
                **kwargs
            )
        
        # Handle template-based formatting
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
                    logger.warning(f"Failed to format custom template, using text directly")
        
        # Default: use the standard Llama 2 chat format
        if self._is_chat:
            return f"{self.SPECIAL_TOKENS['system']}{system_prompt}{self.SPECIAL_TOKENS['system_end']}{text}{self.SPECIAL_TOKENS['user_end']}"
        
        # For base models, just use the text
        return text
    
    def _format_chat_prompt(
        self,
        text: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Format chat prompt for Llama chat models.
        
        Args:
            text: Current user message
            conversation_history: Previous conversation turns
            system_prompt: System prompt
            **kwargs: Additional arguments
            
        Returns:
            Formatted chat prompt
        """
        system_prompt = system_prompt or self._system_prompt
        
        formatted = []
        
        # Add system prompt only for first message
        if not conversation_history or len(conversation_history) == 0:
            formatted.append(f"{self.SPECIAL_TOKENS['system']}{system_prompt}{self.SPECIAL_TOKENS['system_end']}")
        
        # Add conversation history
        if conversation_history:
            for turn in conversation_history[-self._max_history_length:]:
                user_msg = turn.get('user', '')
                assistant_msg = turn.get('assistant', '')
                
                if user_msg:
                    formatted.append(f"{user_msg}{self.SPECIAL_TOKENS['user_end']}")
                if assistant_msg:
                    formatted.append(f"{assistant_msg}{self.SPECIAL_TOKENS['assistant_end']}")
        
        # Add current message
        formatted.append(f"{text}{self.SPECIAL_TOKENS['user_end']}")
        
        return "".join(formatted)
    
    def generate_text(
        self,
        text_input: Dict[str, Any],
        config: Optional[TextGenerationConfig] = None,
        **kwargs
    ) -> TextGenerationResult:
        """
        Generate text response from Llama.
        
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
            
            # Update conversation history for chat models
            original_text = text_input.get('original_text')
            if self._is_chat and original_text and generated_text:
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
                    "is_llama2": self._is_llama2,
                    "is_llama3": self._is_llama3,
                    "is_codelama": self._is_codelama,
                    "is_chat": self._is_chat,
                    "is_long": self._is_long,
                    "device": self._device,
                    "conversation_history": self._conversation_history.copy(),
                    "model_info": self.MODEL_INFO.get(self.model_name, {}),
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate text with Llama: {e}")
            raise RuntimeError(f"Failed to generate text with Llama: {e}")
    
    def _clean_response(self, response: str) -> str:
        """
        Clean up Llama response.
        
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
        
        # Remove any Llama-specific formatting artifacts
        response = re.sub(r'\[INST\].*?\[/INST\]', '', response)
        response = re.sub(r'<<SYS>>.*?<</SYS>>', '', response, flags=re.DOTALL)
        
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
        Have a conversation with the Llama chat model.
        
        Args:
            message: User message
            system_prompt: Optional system prompt
            clear_history: Whether to clear conversation history
            **kwargs: Additional arguments
            
        Returns:
            TextGenerationResult containing response and metadata
        """
        if not self._is_chat:
            logger.warning(f"Model {self.model_name} is not a chat model. Using anyway.")
        
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
        if not self._is_codelama:
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
    
    def summarize_long_document(
        self,
        document: str,
        max_summary_length: int = 500,
        **kwargs
    ) -> TextGenerationResult:
        """
        Summarize a long document using Llama's long context capabilities.
        
        Args:
            document: Long document text
            max_summary_length: Maximum length of summary in tokens
            **kwargs: Additional arguments
            
        Returns:
            TextGenerationResult containing summary
        """
        if not self._is_long:
            logger.warning(f"Model {self.model_name} does not have extended context window.")
        
        # Build summarization prompt
        prompt = f"Summarize this document in about {max_summary_length} words:\n\n{document}"
        
        # Update generation parameters
        kwargs['max_new_tokens'] = min(kwargs.get('max_new_tokens', max_summary_length), 2000)
        kwargs['temperature'] = kwargs.get('temperature', 0.3)
        
        return self.generate(prompt, task_type=TextTaskType.SUMMARIZATION, **kwargs)
    
    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self._conversation_history.clear()
        logger.info("Cleared Llama conversation history")
    
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
        logger.info("Updated Llama system prompt")
    
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
        logger.info(f"Unloaded Llama model {self.model_name}")
    
    def get_model_capabilities(self) -> Dict[str, Any]:
        """
        Get capabilities of the Llama model.
        
        Returns:
            Dictionary with model capabilities
        """
        model_info = self.MODEL_INFO.get(self.model_name, {}).copy()
        
        capabilities = {
            "model": self.model_name,
            "architecture": model_info.get("architecture", "decoder-only"),
            "tasks": model_info.get("tasks", []),
            "context_window": model_info.get("context_window", 4096),
            "max_tokens": model_info.get("max_tokens", 4096),
            "is_llama2": self._is_llama2,
            "is_llama3": self._is_llama3,
            "is_codelama": self._is_codelama,
            "is_chat": self._is_chat,
            "is_long": self._is_long,
            "supports_code": self._is_codelama or "code" in model_info.get("tasks", []),
            "supports_long_context": self._is_long or model_info.get("context_window", 4096) > 8000,
            "requires_auth": model_info.get("requires_auth", True),
            "multilingual": False,  # Llama models are primarily English
            "supported_languages": model_info.get("languages", ["en"]),
            "model_parameters": model_info.get("parameters", 0),
            "memory_requirements_mb": model_info.get("memory_mb", 14000),
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
            
            if "llama-3" in model_lower or "meta-llama-3" in model_lower:
                if "70b" in model_lower:
                    return "llama-3-70b"
                elif "8b" in model_lower:
                    return "llama-3-8b"
                else:
                    return "llama-3"
            elif "llama-2" in model_lower:
                if "70b" in model_lower:
                    return "llama-2-70b"
                elif "13b" in model_lower:
                    return "llama-2-13b"
                elif "7b" in model_lower:
                    return "llama-2-7b"
                else:
                    return "llama-2"
            elif "codellama" in model_lower:
                if "34b" in model_lower:
                    return "codellama-34b"
                elif "13b" in model_lower:
                    return "codellama-13b"
                elif "7b" in model_lower:
                    return "codellama-7b"
                else:
                    return "codellama"
            else:
                return "llama"
            
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