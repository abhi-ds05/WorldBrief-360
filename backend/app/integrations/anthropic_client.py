# backend/app/integrations/anthropic_client.py
"""
Anthropic Claude API integration for WorldBrief 360.
Provides access to Claude models for text generation, analysis, and reasoning.
"""

import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass
from enum import Enum

import anthropic
from pydantic import BaseModel, Field, validator
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from cachetools import TTLCache

from app.core.config import settings
from app.core.logging_config import logger
from app.services.utils.http_client import AsyncHTTPClient
from app.schemas.request.chat import ChatMessage


class ClaudeModel(Enum):
    """Available Claude models."""
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    CLAUDE_2_1 = "claude-2.1"
    CLAUDE_2_0 = "claude-2.0"
    CLAUDE_INSTANT_1_2 = "claude-instant-1.2"


class ClaudeRole(Enum):
    """Claude message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ClaudeMessage:
    """Claude API message structure."""
    role: ClaudeRole
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to API format."""
        return {
            "role": self.role.value,
            "content": self.content
        }


class ClaudeResponse(BaseModel):
    """Standardized Claude API response."""
    id: str
    model: str
    content: str
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: Dict[str, int]
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True


class AnthropicClient:
    """
    Client for interacting with Anthropic's Claude API.
    Supports synchronous and asynchronous operations with proper error handling.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        default_model: ClaudeModel = ClaudeModel.CLAUDE_3_SONNET
    ):
        """
        Initialize Anthropic client.
        
        Args:
            api_key: Anthropic API key (falls back to settings)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            default_model: Default model to use
        """
        self.api_key = api_key or settings.ANTHROPIC_API_KEY
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_model = default_model
        
        if not self.api_key:
            logger.warning("Anthropic API key not provided. Client will be disabled.")
            self.enabled = False
        else:
            self.enabled = True
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.async_client = anthropic.AsyncAnthropic(api_key=self.api_key)
            
        # Cache for model responses (TTL: 1 hour)
        self.cache = TTLCache(maxsize=1000, ttl=3600)
        self.http_client = AsyncHTTPClient()
        
    def is_available(self) -> bool:
        """Check if the client is properly configured."""
        return self.enabled and self.api_key is not None
    
    def _build_messages(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Convert internal chat messages to Claude format.
        
        Args:
            messages: List of ChatMessage objects
            system_prompt: Optional system prompt
            
        Returns:
            List of messages in Claude API format
        """
        claude_messages = []
        
        # Add system prompt if provided
        if system_prompt:
            claude_messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Convert chat messages
        for msg in messages:
            role = ClaudeRole.USER if msg.role == "user" else ClaudeRole.ASSISTANT
            claude_messages.append({
                "role": role.value,
                "content": msg.content
            })
            
        return claude_messages
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(
            (anthropic.APITimeoutError, anthropic.APIConnectionError)
        )
    )
    async def chat_completion(
        self,
        messages: List[ChatMessage],
        model: Optional[ClaudeModel] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        stream: bool = False,
        **kwargs
    ) -> Union[ClaudeResponse, AsyncGenerator[str, None]]:
        """
        Generate chat completion using Claude.
        
        Args:
            messages: List of chat messages
            model: Claude model to use (defaults to self.default_model)
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            stream: Whether to stream the response
            **kwargs: Additional parameters for Claude API
            
        Returns:
            ClaudeResponse or async generator for streaming
            
        Raises:
            ValueError: If messages are empty or client is not available
            anthropic.APIError: For API errors
        """
        if not self.is_available():
            raise ValueError("Anthropic client is not properly configured")
            
        if not messages:
            raise ValueError("Messages list cannot be empty")
            
        model_str = (model or self.default_model).value
        
        # Build cache key
        cache_key = self._generate_cache_key(
            messages=messages,
            model=model_str,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )
        
        # Check cache for non-streaming requests
        if not stream and cache_key in self.cache:
            logger.debug(f"Cache hit for Claude request: {cache_key[:50]}...")
            return self.cache[cache_key]
        
        # Prepare messages for Claude
        claude_messages = self._build_messages(messages, system_prompt)
        
        # Prepare API parameters
        params = {
            "model": model_str,
            "messages": claude_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            **kwargs
        }
        
        try:
            if stream:
                return self._stream_completion(params)
            else:
                return await self._complete_request(params, cache_key)
                
        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Claude completion: {str(e)}")
            raise
    
    async def _complete_request(
        self,
        params: Dict[str, Any],
        cache_key: str
    ) -> ClaudeResponse:
        """
        Make a non-streaming completion request.
        
        Args:
            params: API parameters
            cache_key: Cache key for response
            
        Returns:
            ClaudeResponse object
        """
        response = await self.async_client.messages.create(**params)
        
        # Create standardized response
        claude_response = ClaudeResponse(
            id=response.id,
            model=response.model,
            content=response.content[0].text,
            stop_reason=response.stop_reason,
            stop_sequence=response.stop_sequence,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
        )
        
        # Cache the response
        self.cache[cache_key] = claude_response
        
        return claude_response
    
    async def _stream_completion(
        self,
        params: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """
        Stream completion response.
        
        Args:
            params: API parameters
            
        Yields:
            Chunks of text as they become available
        """
        stream = await self.async_client.messages.create(
            **params,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.type == "content_block_delta":
                yield chunk.delta.text
            elif chunk.type == "error":
                logger.error(f"Stream error: {chunk.error}")
                raise anthropic.APIError(f"Stream error: {chunk.error}")
    
    async def analyze_text(
        self,
        text: str,
        analysis_type: str = "sentiment",
        model: Optional[ClaudeModel] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze text using Claude.
        
        Args:
            text: Text to analyze
            analysis_type: Type of analysis (sentiment, topics, summary, etc.)
            model: Claude model to use
            **kwargs: Additional parameters
            
        Returns:
            Analysis results
        """
        analysis_prompts = {
            "sentiment": "Analyze the sentiment of the following text. "
                        "Respond with a JSON object containing 'sentiment' (positive/negative/neutral), "
                        "'score' (0.0 to 1.0), and 'key_phrases' (list of key phrases).",
            "topics": "Extract main topics from the following text. "
                     "Respond with a JSON object containing 'topics' (list of topics with confidence scores).",
            "summary": "Provide a concise summary of the following text. "
                      "Respond with a JSON object containing 'summary' (string) and 'key_points' (list).",
            "fact_check": "Check the factual accuracy of the following text. "
                         "Respond with a JSON object containing 'verdict' (true/false/partially_true), "
                         "'confidence' (0.0 to 1.0), and 'evidence' (supporting evidence).",
        }
        
        if analysis_type not in analysis_prompts:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
        
        prompt = f"{analysis_prompts[analysis_type]}\n\nText:\n{text}"
        
        messages = [
            ChatMessage(role="user", content=prompt)
        ]
        
        response = await self.chat_completion(
            messages=messages,
            model=model,
            temperature=0.1,  # Lower temperature for more consistent analysis
            max_tokens=500,
            **kwargs
        )
        
        # Parse JSON response
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from Claude response: {response.content[:100]}...")
            return {"raw_response": response.content}
    
    async def compare_texts(
        self,
        text1: str,
        text2: str,
        comparison_type: str = "similarity",
        model: Optional[ClaudeModel] = None
    ) -> Dict[str, Any]:
        """
        Compare two texts using Claude.
        
        Args:
            text1: First text
            text2: Second text
            comparison_type: Type of comparison (similarity, contradiction, timeline, etc.)
            model: Claude model to use
            
        Returns:
            Comparison results
        """
        comparison_prompts = {
            "similarity": "Compare the similarity between these two texts. "
                         "Respond with a JSON object containing 'similarity_score' (0.0 to 1.0) "
                         "and 'key_differences' (list of differences).",
            "contradiction": "Check if these two texts contradict each other. "
                           "Respond with a JSON object containing 'contradiction_score' (0.0 to 1.0) "
                           "and 'contradictory_points' (list of contradictions).",
            "timeline": "Analyze the timeline relationship between these two texts. "
                       "Respond with a JSON object containing 'relationship' (before/after/simultaneous) "
                       "and 'confidence' (0.0 to 1.0).",
        }
        
        if comparison_type not in comparison_prompts:
            raise ValueError(f"Unsupported comparison type: {comparison_type}")
        
        prompt = f"{comparison_prompts[comparison_type]}\n\nText 1:\n{text1}\n\nText 2:\n{text2}"
        
        messages = [
            ChatMessage(role="user", content=prompt)
        ]
        
        response = await self.chat_completion(
            messages=messages,
            model=model,
            temperature=0.1,
            max_tokens=500
        )
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"raw_response": response.content}
    
    def _generate_cache_key(
        self,
        messages: List[ChatMessage],
        model: str,
        system_prompt: Optional[str],
        **kwargs
    ) -> str:
        """
        Generate cache key for request.
        
        Args:
            messages: List of messages
            model: Model name
            system_prompt: Optional system prompt
            **kwargs: Additional parameters
            
        Returns:
            Cache key string
        """
        import hashlib
        
        key_data = {
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "model": model,
            "system_prompt": system_prompt,
            **kwargs
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available Claude models.
        
        Returns:
            List of model information
        """
        # Note: Anthropic doesn't have a models endpoint like OpenAI
        # Return hardcoded list for now
        models = []
        
        for model in ClaudeModel:
            models.append({
                "id": model.value,
                "name": model.name.replace("_", " ").title(),
                "description": self._get_model_description(model),
                "max_tokens": self._get_model_max_tokens(model),
                "supports_vision": model.value.startswith("claude-3"),
                "context_window": self._get_context_window(model)
            })
        
        return models
    
    def _get_model_description(self, model: ClaudeModel) -> str:
        """Get description for a model."""
        descriptions = {
            ClaudeModel.CLAUDE_3_OPUS: "Most powerful model for highly complex tasks",
            ClaudeModel.CLAUDE_3_SONNET: "Balance of intelligence and speed for enterprise workloads",
            ClaudeModel.CLAUDE_3_HAIKU: "Fastest and most compact model for simple queries",
            ClaudeModel.CLAUDE_2_1: "Previous generation model with strong reasoning capabilities",
            ClaudeModel.CLAUDE_2_0: "Previous version of Claude 2",
            ClaudeModel.CLAUDE_INSTANT_1_2: "Fast, lightweight model for simple tasks"
        }
        return descriptions.get(model, "Claude model")
    
    def _get_model_max_tokens(self, model: ClaudeModel) -> int:
        """Get max tokens for a model."""
        max_tokens = {
            ClaudeModel.CLAUDE_3_OPUS: 4096,
            ClaudeModel.CLAUDE_3_SONNET: 4096,
            ClaudeModel.CLAUDE_3_HAIKU: 4096,
            ClaudeModel.CLAUDE_2_1: 4096,
            ClaudeModel.CLAUDE_2_0: 4096,
            ClaudeModel.CLAUDE_INSTANT_1_2: 4096,
        }
        return max_tokens.get(model, 4096)
    
    def _get_context_window(self, model: ClaudeModel) -> int:
        """Get context window size for a model."""
        context_windows = {
            ClaudeModel.CLAUDE_3_OPUS: 200000,
            ClaudeModel.CLAUDE_3_SONNET: 200000,
            ClaudeModel.CLAUDE_3_HAIKU: 200000,
            ClaudeModel.CLAUDE_2_1: 100000,
            ClaudeModel.CLAUDE_2_0: 100000,
            ClaudeModel.CLAUDE_INSTANT_1_2: 100000,
        }
        return context_windows.get(model, 100000)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Anthropic API.
        
        Returns:
            Health status information
        """
        if not self.is_available():
            return {
                "status": "disabled",
                "message": "Client not configured",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Simple test request
            test_messages = [
                ChatMessage(role="user", content="Hello, are you working?")
            ]
            
            start_time = datetime.now()
            response = await self.chat_completion(
                messages=test_messages,
                model=ClaudeModel.CLAUDE_3_HAIKU,
                max_tokens=10,
                temperature=0.1
            )
            end_time = datetime.now()
            
            latency = (end_time - start_time).total_seconds() * 1000  # ms
            
            return {
                "status": "healthy",
                "message": "API is responding",
                "latency_ms": round(latency, 2),
                "model": response.model,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }


# Factory function for dependency injection
def get_anthropic_client() -> AnthropicClient:
    """
    Factory function to create Anthropic client.
    
    Returns:
        Configured AnthropicClient instance
    """
    return AnthropicClient()