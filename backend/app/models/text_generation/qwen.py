"""
Qwen model implementation for text generation.

Qwen models are powerful multilingual LLMs with strong reasoning capabilities,
support for long contexts, and excellent performance across various tasks.
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


class QwenModel(BaseTextGenerationModel):
    """
    Qwen model wrapper for WorldBrief360.
    
    Qwen models are state-of-the-art multilingual LLMs with:
    - Strong multilingual capabilities (Chinese, English, etc.)
    - Excellent reasoning and coding abilities
    - Support for long context windows (up to 32K tokens)
    - Efficient inference
    - Strong mathematical reasoning
    
    Model variants:
    - Qwen2.5: Latest generation with improved capabilities
    - Qwen2.5-Coder: Specialized for code generation
    - Qwen2.5-Math: Specialized for mathematical reasoning
    - Qwen2: Previous generation with strong performance
    - Qwen1.5: Widely used open-source models
    """
    
    # Qwen model variants and their specifications
    MODEL_INFO = {
        # Qwen2.5 models (latest)
        "Qwen/Qwen2.5-0.5B-Instruct": {
            "description": "Qwen2.5 0.5B instruct model - very efficient",
            "max_tokens": 32768,
            "architecture": "decoder-only",
            "parameters": 500_000_000,
            "performance": "efficient",
            "memory_mb": 1000,
            "tasks": ["conversation", "text_completion", "summarization", "question_answering"],
            "languages": ["en", "zh", "multi"],
            "context_window": 32768,
            "requires_auth": False,
        },
        "Qwen/Qwen2.5-1.5B-Instruct": {
            "description": "Qwen2.5 1.5B instruct model - good balance",
            "max_tokens": 32768,
            "architecture": "decoder-only",
            "parameters": 1_500_000_000,
            "performance": "good",
            "memory_mb": 3000,
            "tasks": ["conversation", "text_completion", "summarization", "question_answering", "translation"],
            "languages": ["en", "zh", "multi"],
            "context_window": 32768,
            "requires_auth": False,
        },
        "Qwen/Qwen2.5-3B-Instruct": {
            "description": "Qwen2.5 3B instruct model - strong performance",
            "max_tokens": 32768,
            "architecture": "decoder-only",
            "parameters": 3_000_000_000,
            "performance": "very_good",
            "memory_mb": 6000,
            "tasks": ["conversation", "text_completion", "summarization", "question_answering", "translation", "coding"],
            "languages": ["en", "zh", "multi"],
            "context_window": 32768,
            "requires_auth": False,
        },
        "Qwen/Qwen2.5-7B-Instruct": {
            "description": "Qwen2.5 7B instruct model - excellent performance",
            "max_tokens": 32768,
            "architecture": "decoder-only",
            "parameters": 7_000_000_000,
            "performance": "excellent",
            "memory_mb": 14000,
            "tasks": ["conversation", "text_completion", "summarization", "question_answering", "translation", "coding", "reasoning"],
            "languages": ["en", "zh", "multi"],
            "context_window": 32768,
            "requires_auth": False,
        },
        "Qwen/Qwen2.5-14B-Instruct": {
            "description": "Qwen2.5 14B instruct model - powerful",
            "max_tokens": 32768,
            "architecture": "decoder-only",
            "parameters": 14_000_000_000,
            "performance": "excellent",
            "memory_mb": 28000,
            "tasks": ["conversation", "text_completion", "summarization", "question_answering", "translation", "coding", "reasoning", "mathematical"],
            "languages": ["en", "zh", "multi"],
            "context_window": 32768,
            "requires_auth": False,
        },
        "Qwen/Qwen2.5-32B-Instruct": {
            "description": "Qwen2.5 32B instruct model - very powerful",
            "max_tokens": 32768,
            "architecture": "decoder-only",
            "parameters": 32_000_000_000,
            "performance": "state_of_the_art",
            "memory_mb": 64000,
            "tasks": ["conversation", "text_completion", "summarization", "question_answering", "translation", "coding", "reasoning", "mathematical"],
            "languages": ["en", "zh", "multi"],
            "context_window": 32768,
            "requires_auth": False,
        },
        "Qwen/Qwen2.5-72B-Instruct": {
            "description": "Qwen2.5 72B instruct model - most powerful",
            "max_tokens": 32768,
            "architecture": "decoder-only",
            "parameters": 72_000_000_000,
            "performance": "state_of_the_art",
            "memory_mb": 144000,
            "tasks": ["conversation", "text_completion", "summarization", "question_answering", "translation", "coding", "reasoning", "mathematical", "complex_analysis"],
            "languages": ["en", "zh", "multi"],
            "context_window": 32768,
            "requires_auth": False,
        },
        
        # Qwen2.5-Coder models
        "Qwen/Qwen2.5-Coder-1.5B-Instruct": {
            "description": "Qwen2.5-Coder 1.5B for code generation",
            "max_tokens": 32768,
            "architecture": "decoder-only",
            "parameters": 1_500_000_000,
            "performance": "good",
            "memory_mb": 3000,
            "tasks": ["code_generation", "code_explanation", "code_debugging", "code_completion"],
            "languages": ["code", "en"],
            "context_window": 32768,
            "requires_auth": False,
        },
        "Qwen/Qwen2.5-Coder-7B-Instruct": {
            "description": "Qwen2.5-Coder 7B for code generation",
            "max_tokens": 32768,
            "architecture": "decoder-only",
            "parameters": 7_000_000_000,
            "performance": "excellent",
            "memory_mb": 14000,
            "tasks": ["code_generation", "code_explanation", "code_debugging", "code_completion"],
            "languages": ["code", "en"],
            "context_window": 32768,
            "requires_auth": False,
        },
        "Qwen/Qwen2.5-Coder-32B-Instruct": {
            "description": "Qwen2.5-Coder 32B for code generation",
            "max_tokens": 32768,
            "architecture": "decoder-only",
            "parameters": 32_000_000_000,
            "performance": "state_of_the_art",
            "memory_mb": 64000,
            "tasks": ["code_generation", "code_explanation", "code_debugging", "code_completion"],
            "languages": ["code", "en"],
            "context_window": 32768,
            "requires_auth": False,
        },
        
        # Qwen2.5-Math models
        "Qwen/Qwen2.5-Math-1.5B-Instruct": {
            "description": "Qwen2.5-Math 1.5B for mathematical reasoning",
            "max_tokens": 32768,
            "architecture": "decoder-only",
            "parameters": 1_500_000_000,
            "performance": "good",
            "memory_mb": 3000,
            "tasks": ["mathematical_reasoning", "problem_solving", "calculations"],
            "languages": ["en", "math"],
            "context_window": 32768,
            "requires_auth": False,
        },
        "Qwen/Qwen2.5-Math-7B-Instruct": {
            "description": "Qwen2.5-Math 7B for mathematical reasoning",
            "max_tokens": 32768,
            "architecture": "decoder-only",
            "parameters": 7_000_000_000,
            "performance": "excellent",
            "memory_mb": 14000,
            "tasks": ["mathematical_reasoning", "problem_solving", "calculations", "proof_generation"],
            "languages": ["en", "math"],
            "context_window": 32768,
            "requires_auth": False,
        },
        
        # Qwen2 models (previous generation)
        "Qwen/Qwen2-7B-Instruct": {
            "description": "Qwen2 7B instruct model",
            "max_tokens": 32768,
            "architecture": "decoder-only",
            "parameters": 7_000_000_000,
            "performance": "excellent",
            "memory_mb": 14000,
            "tasks": ["conversation", "text_completion", "summarization", "question_answering", "translation", "coding"],
            "languages": ["en", "zh", "multi"],
            "context_window": 32768,
            "requires_auth": False,
        },
        "Qwen/Qwen2-72B-Instruct": {
            "description": "Qwen2 72B instruct model",
            "max_tokens": 32768,
            "architecture": "decoder-only",
            "parameters": 72_000_000_000,
            "performance": "state_of_the_art",
            "memory_mb": 144000,
            "tasks": ["conversation", "text_completion", "summarization", "question_answering", "translation", "coding", "reasoning"],
            "languages": ["en", "zh", "multi"],
            "context_window": 32768,
            "requires_auth": False,
        },
        
        # Qwen1.5 models (widely used)
        "Qwen/Qwen1.5-0.5B-Chat": {
            "description": "Qwen1.5 0.5B chat model",
            "max_tokens": 32768,
            "architecture": "decoder-only",
            "parameters": 500_000_000,
            "performance": "efficient",
            "memory_mb": 1000,
            "tasks": ["conversation", "text_completion", "summarization"],
            "languages": ["en", "zh"],
            "context_window": 32768,
            "requires_auth": False,
        },
        "Qwen/Qwen1.5-7B-Chat": {
            "description": "Qwen1.5 7B chat model",
            "max_tokens": 32768,
            "architecture": "decoder-only",
            "parameters": 7_000_000_000,
            "performance": "excellent",
            "memory_mb": 14000,
            "tasks": ["conversation", "text_completion", "summarization", "question_answering", "translation"],
            "languages": ["en", "zh", "multi"],
            "context_window": 32768,
            "requires_auth": False,
        },
        "Qwen/Qwen1.5-14B-Chat": {
            "description": "Qwen1.5 14B chat model",
            "max_tokens": 32768,
            "architecture": "decoder-only",
            "parameters": 14_000_000_000,
            "performance": "excellent",
            "memory_mb": 28000,
            "tasks": ["conversation", "text_completion", "summarization", "question_answering", "translation", "coding"],
            "languages": ["en", "zh", "multi"],
            "context_window": 32768,
            "requires_auth": False,
        },
        "Qwen/Qwen1.5-32B-Chat": {
            "description": "Qwen1.5 32B chat model",
            "max_tokens": 32768,
            "architecture": "decoder-only",
            "parameters": 32_000_000_000,
            "performance": "state_of_the_art",
            "memory_mb": 64000,
            "tasks": ["conversation", "text_completion", "summarization", "question_answering", "translation", "coding", "reasoning"],
            "languages": ["en", "zh", "multi"],
            "context_window": 32768,
            "requires_auth": False,
        },
        "Qwen/Qwen1.5-72B-Chat": {
            "description": "Qwen1.5 72B chat model",
            "max_tokens": 32768,
            "architecture": "decoder-only",
            "parameters": 72_000_000_000,
            "performance": "state_of_the_art",
            "memory_mb": 144000,
            "tasks": ["conversation", "text_completion", "summarization", "question_answering", "translation", "coding", "reasoning"],
            "languages": ["en", "zh", "multi"],
            "context_window": 32768,
            "requires_auth": False,
        },
    }
    
    # Special tokens for Qwen
    SPECIAL_TOKENS = {
        "system": "<|im_start|>system\n",
        "user": "<|im_start|>user\n",
        "assistant": "<|im_start|>assistant\n",
        "end": "<|im_end|>\n",
    }
    
    # Task-specific prompt templates
    PROMPT_TEMPLATES = {
        "conversation": "{system_prompt}{conversation_history}{user_prompt}",
        "code_generation": "<|im_start|>system\nYou are an expert programming assistant. Write efficient, well-commented code.\n<|im_end|>\n<|im_start|>user\nWrite code to solve: {problem}\n\nProvide the complete solution in {language}.<|im_end|>\n<|im_start|>assistant\n",
        "code_explanation": "<|im_start|>system\nYou are an expert programming assistant. Explain code clearly and thoroughly.\n<|im_end|>\n<|im_start|>user\nExplain this code:\n```\n{code}\n```\n\nExplain what it does, how it works, and any potential issues.<|im_end|>\n<|im_start|>assistant\n",
        "summarization": "<|im_start|>system\nYou are a helpful summarization assistant. Provide concise, accurate summaries.\n<|im_end|>\n<|im_start|>user\nSummarize this text:\n\n{text}\n\nProvide a concise summary.<|im_end|>\n<|im_start|>assistant\n",
        "translation": "<|im_start|>system\nYou are a helpful translation assistant. Provide accurate translations.\n<|im_end|>\n<|im_start|>user\nTranslate this text from {source_language} to {target_language}:\n\n{text}<|im_end|>\n<|im_start|>assistant\n",
        "question_answering": "<|im_start|>system\nYou are a helpful question-answering assistant. Provide accurate, detailed answers.\n<|im_end|>\n<|im_start|>user\nAnswer this question:\n\n{question}\n\nProvide a detailed and accurate answer.<|im_end|>\n<|im_start|>assistant\n",
        "creative_writing": "<|im_start|>system\nYou are a creative writing assistant. Write engaging, well-structured content.\n<|im_end|>\n<|im_start|>user\nWrite a {genre} about {topic} with these requirements:\n{requirements}<|im_end|>\n<|im_start|>assistant\n",
        "mathematical_reasoning": "<|im_start|>system\nYou are a mathematical reasoning assistant. Show step-by-step solutions.\n<|im_end|>\n<|im_start|>user\nSolve this mathematical problem:\n\n{problem}\n\nShow your reasoning step by step.<|im_end|>\n<|im_start|>assistant\n",
        "text_completion": "<|im_start|>system\nYou are a helpful text completion assistant.\n<|im_end|>\n<|im_start|>user\nComplete this text:\n\n{text}<|im_end|>\n<|im_start|>assistant\n",
    }
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        config: Optional[TextGenerationConfig] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize Qwen model.
        
        Args:
            model_name: Name of the Qwen model
            config: Configuration for text generation
            model_kwargs: Additional arguments for model initialization
            tokenizer_kwargs: Additional arguments for tokenizer initialization
            **kwargs: Additional arguments
        """
        # Update config with Qwen-specific defaults
        if config is None:
            config = TextGenerationConfig()
        
        # Get model info
        model_info = self.MODEL_INFO.get(model_name, {})
        context_window = model_info.get("context_window", 32768)
        
        # Adjust max tokens based on model capabilities
        if config.generation_params.max_new_tokens > context_window:
            config.generation_params.max_new_tokens = context_window
        
        # Set task type based on model name
        if "coder" in model_name.lower():
            config.task_type = TextTaskType.CODE_GENERATION
        elif "math" in model_name.lower():
            config.task_type = TextTaskType.MATHEMATICAL_REASONING
        elif "instruct" in model_name.lower() or "chat" in model_name.lower():
            config.task_type = TextTaskType.CONVERSATION
        
        super().__init__(model_name, config, **kwargs)
        
        # Qwen-specific initialization
        self.model_kwargs = model_kwargs or {}
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        
        # Check model type
        self._is_qwen25 = "2.5" in model_name
        self._is_qwen2 = "2" in model_name and "2.5" not in model_name
        self._is_qwen15 = "1.5" in model_name
        self._is_coder = "coder" in model_name.lower()
        self._is_math = "math" in model_name.lower()
        self._is_instruct = "instruct" in model_name.lower() or "chat" in model_name.lower()
        
        # Conversation history for instruct/chat models
        self._conversation_history: List[Dict[str, str]] = []
        self._max_history_length = 20  # Qwen supports longer context
        
        # System prompt
        self._system_prompt = self._get_default_system_prompt()
        
        # Generation parameters optimized for Qwen
        self._default_generation_params = {
            "max_new_tokens": config.generation_params.max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "num_beams": 1,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "early_stopping": False,
        }
        
        # Adjust for code generation
        if self._is_coder:
            self._default_generation_params.update({
                "temperature": 0.2,
                "top_p": 0.95,
                "repetition_penalty": 1.05,
            })
        
        # Adjust for mathematical reasoning
        if self._is_math:
            self._default_generation_params.update({
                "temperature": 0.1,
                "top_p": 0.9,
                "do_sample": False,
            })
        
        logger.info(f"Initialized Qwen model: {model_name} "
                   f"(Qwen2.5: {self._is_qwen25}, Qwen2: {self._is_qwen2}, Qwen1.5: {self._is_qwen15}, "
                   f"Coder: {self._is_coder}, Math: {self._is_math}, Instruct: {self._is_instruct})")
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt based on model type."""
        if self._is_coder:
            return "You are Qwen-Coder, a helpful AI assistant specialized in programming and code generation. Provide accurate, efficient, and well-commented code solutions."
        elif self._is_math:
            return "You are Qwen-Math, a helpful AI assistant specialized in mathematical reasoning. Provide step-by-step solutions with clear explanations and accurate calculations."
        elif self._is_instruct:
            return "You are Qwen, a helpful AI assistant."
        else:
            return "You are a helpful AI assistant."
    
    def load(self) -> None:
        """
        Load the Qwen model and tokenizer.
        
        Raises:
            RuntimeError: If model loading fails
        """
        if self._is_loaded:
            logger.warning(f"Model {self.model_name} is already loaded")
            return
        
        try:
            logger.info(f"Loading Qwen model: {self.model_name}")
            
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
                    "Qwen models are large and run slowly on CPU. "
                    "Consider using GPU for reasonable performance."
                )
            
            # Load tokenizer
            logger.info(f"Loading tokenizer for {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                **self.tokenizer_kwargs
            )
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            logger.info(f"Loading Qwen model for {self.model_name}")
            
            # Set default model kwargs
            self.model_kwargs.setdefault("trust_remote_code", True)
            self.model_kwargs.setdefault("torch_dtype", torch.float16 if self._device == "cuda" else torch.float32)
            
            # For Qwen2.5 models, use bfloat16 if available
            if self._is_qwen25 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.model_kwargs["torch_dtype"] = torch.bfloat16
            
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
            self.context_window = model_info.get("context_window", 32768)
            
            logger.info(f"Successfully loaded Qwen model {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen model {self.model_name}: {e}")
            raise RuntimeError(f"Failed to load Qwen model {self.model_name}: {e}")
    
    def process_text(
        self,
        text: str,
        prompt_template: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process text for Qwen model input.
        
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
            logger.error(f"Failed to process text for Qwen: {e}")
            raise RuntimeError(f"Failed to process text for Qwen: {e}")
    
    def _format_prompt(
        self,
        text: str,
        prompt_template: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Format prompt for Qwen model.
        
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
        
        # Handle instruct models with conversation format
        if self._is_instruct and conversation_history:
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
                    logger.warning(f"Failed to format custom template, using default format")
        
        # Default format for instruct models
        if self._is_instruct:
            return f"{self.SPECIAL_TOKENS['system']}{system_prompt}{self.SPECIAL_TOKENS['end']}" \
                   f"{self.SPECIAL_TOKENS['user']}{text}{self.SPECIAL_TOKENS['end']}" \
                   f"{self.SPECIAL_TOKENS['assistant']}"
        
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
        Format chat prompt for Qwen instruct models.
        
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
        
        # Add system prompt
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
        
        # Format messages using Qwen's chat template
        try:
            # Try to use tokenizer's chat template if available
            if hasattr(self.tokenizer, 'apply_chat_template'):
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}")
        
        # Fallback: manual formatting
        formatted = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted.append(f"{self.SPECIAL_TOKENS['system']}{content}{self.SPECIAL_TOKENS['end']}")
            elif role == "user":
                formatted.append(f"{self.SPECIAL_TOKENS['user']}{content}{self.SPECIAL_TOKENS['end']}")
            elif role == "assistant":
                formatted.append(f"{self.SPECIAL_TOKENS['assistant']}{content}{self.SPECIAL_TOKENS['end']}")
        
        # Add assistant token for completion
        formatted.append(f"{self.SPECIAL_TOKENS['assistant']}")
        
        return "".join(formatted)
    
    def generate_text(
        self,
        text_input: Dict[str, Any],
        config: Optional[TextGenerationConfig] = None,
        **kwargs
    ) -> TextGenerationResult:
        """
        Generate text response from Qwen.
        
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
                    "is_qwen25": self._is_qwen25,
                    "is_qwen2": self._is_qwen2,
                    "is_qwen15": self._is_qwen15,
                    "is_coder": self._is_coder,
                    "is_math": self._is_math,
                    "is_instruct": self._is_instruct,
                    "device": self._device,
                    "conversation_history": self._conversation_history.copy(),
                    "model_info": self.MODEL_INFO.get(self.model_name, {}),
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate text with Qwen: {e}")
            raise RuntimeError(f"Failed to generate text with Qwen: {e}")
    
    def _clean_response(self, response: str) -> str:
        """
        Clean up Qwen response.
        
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
        response = response.replace("<|im_end|>", "")
        response = response.replace("<|im_start|>", "")
        
        # Remove any formatting artifacts
        response = re.sub(r'<\|.*?\|>', '', response)
        
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
        Have a conversation with the Qwen instruct model.
        
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
        if not self._is_coder:
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
    
    def solve_math_problem(
        self,
        problem: str,
        show_steps: bool = True,
        **kwargs
    ) -> TextGenerationResult:
        """
        Solve a mathematical problem.
        
        Args:
            problem: Mathematical problem
            show_steps: Whether to show step-by-step reasoning
            **kwargs: Additional arguments
            
        Returns:
            TextGenerationResult containing solution
        """
        if not self._is_math:
            logger.warning(f"Model {self.model_name} is not specialized for mathematical reasoning.")
        
        # Build math problem prompt
        prompt = f"Solve: {problem}"
        if show_steps:
            prompt += "\n\nShow your step-by-step reasoning."
        
        # Update generation parameters for math
        kwargs['temperature'] = kwargs.get('temperature', 0.1)
        kwargs['do_sample'] = kwargs.get('do_sample', False)
        
        return self.generate(prompt, task_type=TextTaskType.MATHEMATICAL_REASONING, **kwargs)
    
    def translate_text(
        self,
        text: str,
        source_language: str = "auto",
        target_language: str = "English",
        **kwargs
    ) -> TextGenerationResult:
        """
        Translate text between languages using Qwen's multilingual capabilities.
        
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
        kwargs['do_sample'] = kwargs.get('do_sample', False)
        
        return self.generate(prompt, task_type=TextTaskType.TRANSLATION, **kwargs)
    
    def analyze_text(
        self,
        text: str,
        analysis_type: str = "sentiment",
        **kwargs
    ) -> TextGenerationResult:
        """
        Analyze text for various purposes.
        
        Args:
            text: Text to analyze
            analysis_type: Type of analysis (sentiment, entities, topics, etc.)
            **kwargs: Additional arguments
            
        Returns:
            TextGenerationResult containing analysis
        """
        analysis_prompts = {
            "sentiment": "Analyze the sentiment of this text:\n\n{text}\n\nProvide the sentiment analysis.",
            "entities": "Extract named entities from this text:\n\n{text}\n\nList all entities with their types.",
            "topics": "Extract the main topics from this text:\n\n{text}\n\nList the topics.",
            "summary": "Summarize this text:\n\n{text}\n\nProvide a concise summary.",
            "key_points": "Extract key points from this text:\n\n{text}\n\nList the key points.",
        }
        
        prompt_template = analysis_prompts.get(analysis_type, analysis_prompts["sentiment"])
        prompt = prompt_template.format(text=text)
        
        return self.generate(prompt, task_type=TextTaskType.QUESTION_ANSWERING, **kwargs)
    
    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self._conversation_history.clear()
        logger.info("Cleared Qwen conversation history")
    
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
        logger.info("Updated Qwen system prompt")
    
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
        logger.info(f"Unloaded Qwen model {self.model_name}")
    
    def get_model_capabilities(self) -> Dict[str, Any]:
        """
        Get capabilities of the Qwen model.
        
        Returns:
            Dictionary with model capabilities
        """
        model_info = self.MODEL_INFO.get(self.model_name, {}).copy()
        
        capabilities = {
            "model": self.model_name,
            "architecture": model_info.get("architecture", "decoder-only"),
            "tasks": model_info.get("tasks", []),
            "context_window": model_info.get("context_window", 32768),
            "max_tokens": model_info.get("max_tokens", 32768),
            "is_qwen25": self._is_qwen25,
            "is_qwen2": self._is_qwen2,
            "is_qwen15": self._is_qwen15,
            "is_coder": self._is_coder,
            "is_math": self._is_math,
            "is_instruct": self._is_instruct,
            "supports_code": self._is_coder or "code" in model_info.get("tasks", []),
            "supports_math": self._is_math or "math" in model_info.get("tasks", []),
            "multilingual": True,  # Qwen models have excellent multilingual support
            "supported_languages": model_info.get("languages", ["en", "zh", "multi"]),
            "model_parameters": model_info.get("parameters", 0),
            "memory_requirements_mb": model_info.get("memory_mb", 14000),
            "requires_auth": model_info.get("requires_auth", False),
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
            
            if "2.5" in model_lower:
                if "coder" in model_lower:
                    return "qwen2.5-coder"
                elif "math" in model_lower:
                    return "qwen2.5-math"
                elif "72b" in model_lower:
                    return "qwen2.5-72b"
                elif "32b" in model_lower:
                    return "qwen2.5-32b"
                elif "14b" in model_lower:
                    return "qwen2.5-14b"
                elif "7b" in model_lower:
                    return "qwen2.5-7b"
                elif "3b" in model_lower:
                    return "qwen2.5-3b"
                elif "1.5b" in model_lower:
                    return "qwen2.5-1.5b"
                elif "0.5b" in model_lower:
                    return "qwen2.5-0.5b"
                else:
                    return "qwen2.5"
            elif "2" in model_lower:
                if "72b" in model_lower:
                    return "qwen2-72b"
                elif "7b" in model_lower:
                    return "qwen2-7b"
                else:
                    return "qwen2"
            elif "1.5" in model_lower:
                if "72b" in model_lower:
                    return "qwen1.5-72b"
                elif "32b" in model_lower:
                    return "qwen1.5-32b"
                elif "14b" in model_lower:
                    return "qwen1.5-14b"
                elif "7b" in model_lower:
                    return "qwen1.5-7b"
                elif "0.5b" in model_lower:
                    return "qwen1.5-0.5b"
                else:
                    return "qwen1.5"
            else:
                return "qwen"
            
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