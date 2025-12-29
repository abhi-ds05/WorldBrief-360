# backend/app/integrations/openai_client.py
"""
OpenAI API integration for WorldBrief 360.
Provides access to GPT models, DALL-E, Whisper, and other OpenAI services.
"""

import asyncio
import hashlib
import json
import base64
import mimetypes
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import aiofiles

import aiohttp
from pydantic import BaseModel, Field, validator, HttpUrl
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from cachetools import TTLCache, LRUCache

from app.core.config import settings
from app.core.logging_config import logger
from app.services.utils.http_client import AsyncHTTPClient
from app.schemas.request.chat import ChatMessage


class OpenAIModel(Enum):
    """Available OpenAI models."""
    # GPT-4 Models
    GPT_4_TURBO = "gpt-4-turbo-preview"
    GPT_4 = "gpt-4"
    GPT_4_32K = "gpt-4-32k"
    GPT_4_VISION = "gpt-4-vision-preview"
    
    # GPT-3.5 Models
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_3_5_TURBO_16K = "gpt-3.5-turbo-16k"
    GPT_3_5_TURBO_INSTRUCT = "gpt-3.5-turbo-instruct"
    
    # Embedding Models
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    
    # Image Models
    DALL_E_3 = "dall-e-3"
    DALL_E_2 = "dall-e-2"
    
    # Audio Models
    WHISPER_1 = "whisper-1"
    TTS_1 = "tts-1"
    TTS_1_HD = "tts-1-hd"
    
    # Moderation Model
    TEXT_MODERATION_LATEST = "text-moderation-latest"
    TEXT_MODERATION_STABLE = "text-moderation-stable"


class OpenAIRole(Enum):
    """OpenAI message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"


class OpenAIFinishReason(Enum):
    """OpenAI finish reasons."""
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALLS = "tool_calls"


class OpenAIImageSize(Enum):
    """DALL-E image sizes."""
    SIZE_256x256 = "256x256"
    SIZE_512x512 = "512x512"
    SIZE_1024x1024 = "1024x1024"
    SIZE_1792x1024 = "1792x1024"
    SIZE_1024x1792 = "1024x1792"


class OpenAIImageStyle(Enum):
    """DALL-E image styles."""
    VIVID = "vivid"
    NATURAL = "natural"


class OpenAIImageQuality(Enum):
    """DALL-E image quality."""
    STANDARD = "standard"
    HD = "hd"


class OpenAIImageResponseFormat(Enum):
    """Image response formats."""
    URL = "url"
    B64_JSON = "b64_json"


class OpenAITTSVoice(Enum):
    """TTS voices."""
    ALLOY = "alloy"
    ECHO = "echo"
    FABLE = "fable"
    ONYX = "onyx"
    NOVA = "nova"
    SHIMMER = "shimmer"


class OpenAITTSSpeed(Enum):
    """TTS speed settings."""
    X_SLOW = 0.25
    SLOW = 0.5
    NORMAL = 1.0
    FAST = 1.5
    X_FAST = 2.0


class OpenAIImageContent(BaseModel):
    """Image content for multimodal models."""
    type: str = "image_url"
    image_url: Dict[str, str]
    
    @classmethod
    def from_url(cls, url: str, detail: str = "auto"):
        """Create image content from URL."""
        return cls(image_url={"url": url, "detail": detail})
    
    @classmethod
    def from_base64(cls, base64_image: str, detail: str = "auto"):
        """Create image content from base64."""
        url = f"data:image/jpeg;base64,{base64_image}"
        return cls(image_url={"url": url, "detail": detail})


class OpenAITextContent(BaseModel):
    """Text content for multimodal models."""
    type: str = "text"
    text: str


class OpenAIMessage(BaseModel):
    """OpenAI API message."""
    role: OpenAIRole
    content: Union[str, List[Union[OpenAITextContent, OpenAIImageContent]]]
    name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API format."""
        result = {
            "role": self.role.value,
            "content": self.content,
        }
        
        if self.name:
            result["name"] = self.name
        
        return result


class OpenAIToolCall(BaseModel):
    """OpenAI tool call."""
    id: str
    type: str = "function"
    function: Dict[str, Any]


class OpenAITool(BaseModel):
    """OpenAI tool definition."""
    type: str = "function"
    function: Dict[str, Any]


class OpenAIResponse(BaseModel):
    """Standardized OpenAI API response."""
    id: str
    model: str
    object: str
    created: datetime
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]
    system_fingerprint: Optional[str] = None
    
    def get_content(self) -> str:
        """Get text content from response."""
        if self.choices:
            choice = self.choices[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            if content:
                return content
            
            # Check for tool calls
            tool_calls = message.get("tool_calls", [])
            if tool_calls:
                return json.dumps(tool_calls, indent=2)
        
        return ""
    
    def get_tool_calls(self) -> List[OpenAIToolCall]:
        """Get tool calls from response."""
        tool_calls = []
        if self.choices:
            choice = self.choices[0]
            message = choice.get("message", {})
            raw_tool_calls = message.get("tool_calls", [])
            
            for tool_call in raw_tool_calls:
                tool_calls.append(OpenAIToolCall(**tool_call))
        
        return tool_calls


class OpenAIEmbedding(BaseModel):
    """OpenAI embedding response."""
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str
    usage: Dict[str, int]
    
    def get_embeddings(self) -> List[List[float]]:
        """Get embeddings as list of vectors."""
        embeddings = []
        for item in self.data:
            embedding = item.get("embedding", [])
            embeddings.append(embedding)
        return embeddings


class OpenAIImage(BaseModel):
    """OpenAI generated image."""
    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None
    
    def get_image_data(self) -> Optional[bytes]:
        """Get image data as bytes."""
        if self.b64_json:
            return base64.b64decode(self.b64_json)
        return None


class OpenAIImageResponse(BaseModel):
    """OpenAI image generation response."""
    created: datetime
    data: List[OpenAIImage]
    
    def get_images(self) -> List[OpenAIImage]:
        """Get list of generated images."""
        return self.data


class OpenAIAudioResponse(BaseModel):
    """OpenAI audio response."""
    text: str


class OpenAIModerationResponse(BaseModel):
    """OpenAI moderation response."""
    id: str
    model: str
    results: List[Dict[str, Any]]
    
    def is_flagged(self) -> bool:
        """Check if any content was flagged."""
        if self.results:
            return self.results[0].get("flagged", False)
        return False
    
    def get_categories(self) -> Dict[str, bool]:
        """Get moderation categories."""
        if self.results:
            return self.results[0].get("categories", {})
        return {}
    
    def get_category_scores(self) -> Dict[str, float]:
        """Get moderation category scores."""
        if self.results:
            return self.results[0].get("category_scores", {})
        return {}


class OpenAIFile(BaseModel):
    """OpenAI file."""
    id: str
    object: str = "file"
    bytes: int
    created_at: datetime
    filename: str
    purpose: str
    status: Optional[str] = None
    status_details: Optional[str] = None


class OpenAIUsage(BaseModel):
    """OpenAI usage statistics."""
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    total_cost: Optional[float] = None
    
    def calculate_cost(self, model: str) -> float:
        """Calculate cost based on model pricing."""
        # Prices per 1K tokens (as of 2024)
        prices = {
            OpenAIModel.GPT_4_TURBO.value: {"input": 0.01, "output": 0.03},
            OpenAIModel.GPT_4.value: {"input": 0.03, "output": 0.06},
            OpenAIModel.GPT_4_32K.value: {"input": 0.06, "output": 0.12},
            OpenAIModel.GPT_3_5_TURBO.value: {"input": 0.001, "output": 0.002},
            OpenAIModel.GPT_3_5_TURBO_16K.value: {"input": 0.003, "output": 0.004},
            OpenAIModel.TEXT_EMBEDDING_ADA_002.value: {"input": 0.0001, "output": 0},
        }
        
        model_prices = prices.get(model, {"input": 0.0, "output": 0.0})
        cost = (self.prompt_tokens / 1000 * model_prices["input"] +
                self.completion_tokens / 1000 * model_prices["output"])
        
        self.total_cost = cost
        return cost


class OpenAIClient:
    """
    Client for OpenAI API.
    Provides access to GPT, DALL-E, Whisper, and other OpenAI services.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 120,
        max_retries: int = 3,
        cache_ttl: int = 3600
    ):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            organization: OpenAI organization ID
            base_url: API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            cache_ttl: Cache TTL in seconds
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.organization = organization or settings.OPENAI_ORGANIZATION
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        
        self.http_client = AsyncHTTPClient(
            timeout=timeout,
            retries=max_retries
        )
        
        # Caches
        self.model_cache = TTLCache(maxsize=100, ttl=cache_ttl * 24)  # 24 hours
        self.completion_cache = TTLCache(maxsize=1000, ttl=cache_ttl // 2)  # 30 minutes
        self.embedding_cache = TTLCache(maxsize=5000, ttl=cache_ttl)  # 1 hour
        
        # Headers
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        if self.organization:
            self.headers["OpenAI-Organization"] = self.organization
        
        # Rate limiting tracking
        self.rate_limit_remaining = None
        self.rate_limit_reset = None
        
        logger.info(f"OpenAI client initialized with key: {'Yes' if self.api_key else 'No'}")
    
    def is_available(self) -> bool:
        """Check if client is properly configured."""
        return self.api_key is not None
    
    def _update_rate_limits(self, response_headers):
        """Update rate limit information from response headers."""
        if 'x-ratelimit-remaining-requests' in response_headers:
            self.rate_limit_remaining = int(response_headers['x-ratelimit-remaining-requests'])
        
        if 'x-ratelimit-reset-requests' in response_headers:
            reset_str = response_headers['x-ratelimit-reset-requests']
            self.rate_limit_reset = datetime.fromisoformat(reset_str.replace('Z', '+00:00'))
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError,))
    )
    async def get_models(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Get list of available OpenAI models.
        
        Args:
            use_cache: Whether to use cache
            
        Returns:
            List of model information
        """
        cache_key = "models"
        
        if use_cache and cache_key in self.model_cache:
            return self.model_cache[cache_key]
        
        try:
            url = f"{self.base_url}/models"
            
            async with self.http_client.session.get(url, headers=self.headers) as response:
                self._update_rate_limits(response.headers)
                
                if response.status == 200:
                    data = await response.json()
                    models = data.get("data", [])
                    
                    # Filter and format models
                    formatted_models = []
                    for model in models:
                        model_id = model.get("id", "")
                        
                        # Skip deprecated models
                        if "deprecated" in model_id.lower():
                            continue
                        
                        formatted_models.append({
                            "id": model_id,
                            "object": model.get("object", ""),
                            "created": model.get("created", 0),
                            "owned_by": model.get("owned_by", ""),
                            "permission": model.get("permission", []),
                        })
                    
                    # Cache the results
                    self.model_cache[cache_key] = formatted_models
                    
                    return formatted_models
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get models: {response.status} - {error_text}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting OpenAI models: {str(e)}")
            return []
    
    async def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model information or None if not found
        """
        try:
            url = f"{self.base_url}/models/{model_id}"
            
            async with self.http_client.session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting model info for {model_id}: {str(e)}")
            return None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError,))
    )
    async def chat_completion(
        self,
        messages: List[ChatMessage],
        model: Union[str, OpenAIModel] = OpenAIModel.GPT_3_5_TURBO,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stream: bool = False,
        tools: Optional[List[OpenAITool]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        user: Optional[str] = None,
        **kwargs
    ) -> Union[OpenAIResponse, AsyncGenerator[str, None]]:
        """
        Generate chat completion using OpenAI GPT models.
        
        Args:
            messages: List of chat messages
            model: Model to use
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            frequency_penalty: Frequency penalty (-2.0 to 2.0)
            presence_penalty: Presence penalty (-2.0 to 2.0)
            stream: Whether to stream the response
            tools: List of tools available to the model
            tool_choice: Tool choice configuration
            response_format: Response format specification
            seed: Random seed for reproducibility
            stop: Stop sequences
            user: User identifier for abuse monitoring
            **kwargs: Additional parameters
            
        Returns:
            OpenAIResponse or async generator for streaming
            
        Raises:
            ValueError: If messages are empty or client is not available
        """
        if not self.is_available():
            raise ValueError("OpenAI client is not properly configured")
        
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        model_str = model.value if isinstance(model, OpenAIModel) else model
        
        # Build cache key for non-streaming requests
        cache_key = None
        if not stream:
            cache_key = self._generate_completion_cache_key(
                messages=messages,
                model=model_str,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                **kwargs
            )
            
            if cache_key in self.completion_cache:
                logger.debug(f"Cache hit for OpenAI completion: {cache_key[:50]}...")
                return self.completion_cache[cache_key]
        
        # Prepare messages for OpenAI
        openai_messages = []
        for msg in messages:
            role = OpenAIRole.USER if msg.role == "user" else OpenAIRole.ASSISTANT
            openai_messages.append(OpenAIMessage(role=role, content=msg.content))
        
        # Prepare API parameters
        params = {
            "model": model_str,
            "messages": [m.to_dict() for m in openai_messages],
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        if stream:
            params["stream"] = True
        
        if tools:
            params["tools"] = [t.dict() for t in tools]
        
        if tool_choice:
            params["tool_choice"] = tool_choice
        
        if response_format:
            params["response_format"] = response_format
        
        if seed is not None:
            params["seed"] = seed
        
        if stop:
            params["stop"] = stop
        
        if user:
            params["user"] = user
        
        params.update(kwargs)
        
        try:
            if stream:
                return self._stream_completion(params)
            else:
                return await self._complete_request(params, cache_key)
                
        except aiohttp.ClientError as e:
            logger.error(f"OpenAI API connection error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in OpenAI completion: {str(e)}")
            raise
    
    async def _complete_request(
        self,
        params: Dict[str, Any],
        cache_key: Optional[str] = None
    ) -> OpenAIResponse:
        """
        Make a non-streaming completion request.
        
        Args:
            params: API parameters
            cache_key: Cache key for response
            
        Returns:
            OpenAIResponse object
        """
        url = f"{self.base_url}/chat/completions"
        
        async with self.http_client.session.post(
            url, headers=self.headers, json=params
        ) as response:
            self._update_rate_limits(response.headers)
            
            if response.status == 200:
                data = await response.json()
                openai_response = OpenAIResponse(**data)
                
                # Cache the response
                if cache_key:
                    self.completion_cache[cache_key] = openai_response
                
                # Log usage
                self._log_usage(openai_response)
                
                return openai_response
            else:
                error_text = await response.text()
                logger.error(f"OpenAI completion failed: {response.status} - {error_text}")
                raise Exception(f"OpenAI API error: {error_text}")
    
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
        url = f"{self.base_url}/chat/completions"
        
        async with self.http_client.session.post(
            url, headers=self.headers, json=params
        ) as response:
            self._update_rate_limits(response.headers)
            
            if response.status == 200:
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        
                        if data == '[DONE]':
                            break
                        
                        try:
                            chunk_data = json.loads(data)
                            choices = chunk_data.get('choices', [])
                            if choices:
                                delta = choices[0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
            else:
                error_text = await response.text()
                logger.error(f"OpenAI streaming failed: {response.status} - {error_text}")
                raise Exception(f"OpenAI API error: {error_text}")
    
    def _generate_completion_cache_key(
        self,
        messages: List[ChatMessage],
        model: str,
        **kwargs
    ) -> str:
        """Generate cache key for completion request."""
        key_data = {
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "model": model,
            **kwargs
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def create_embeddings(
        self,
        input_text: Union[str, List[str]],
        model: Union[str, OpenAIModel] = OpenAIModel.TEXT_EMBEDDING_ADA_002,
        dimensions: Optional[int] = None,
        user: Optional[str] = None,
        **kwargs
    ) -> OpenAIEmbedding:
        """
        Create embeddings using OpenAI embedding models.
        
        Args:
            input_text: Text or list of texts to embed
            model: Embedding model to use
            dimensions: Number of dimensions for the output
            user: User identifier for abuse monitoring
            **kwargs: Additional parameters
            
        Returns:
            OpenAIEmbedding response
        """
        if not self.is_available():
            raise ValueError("OpenAI client is not properly configured")
        
        model_str = model.value if isinstance(model, OpenAIModel) else model
        
        # Generate cache key
        cache_key = self._generate_embedding_cache_key(
            input_text=input_text,
            model=model_str,
            dimensions=dimensions,
            **kwargs
        )
        
        if cache_key in self.embedding_cache:
            logger.debug(f"Cache hit for OpenAI embeddings: {cache_key[:50]}...")
            return self.embedding_cache[cache_key]
        
        # Prepare parameters
        params = {
            "model": model_str,
            "input": input_text,
        }
        
        if dimensions:
            params["dimensions"] = dimensions
        
        if user:
            params["user"] = user
        
        params.update(kwargs)
        
        url = f"{self.base_url}/embeddings"
        
        try:
            async with self.http_client.session.post(
                url, headers=self.headers, json=params
            ) as response:
                self._update_rate_limits(response.headers)
                
                if response.status == 200:
                    data = await response.json()
                    embedding_response = OpenAIEmbedding(**data)
                    
                    # Cache the response
                    self.embedding_cache[cache_key] = embedding_response
                    
                    return embedding_response
                else:
                    error_text = await response.text()
                    logger.error(f"OpenAI embeddings failed: {response.status} - {error_text}")
                    raise Exception(f"OpenAI API error: {error_text}")
                    
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise
    
    def _generate_embedding_cache_key(
        self,
        input_text: Union[str, List[str]],
        model: str,
        **kwargs
    ) -> str:
        """Generate cache key for embedding request."""
        if isinstance(input_text, list):
            input_key = "|".join(sorted(input_text))
        else:
            input_key = input_text
        
        key_data = {
            "input": input_key,
            "model": model,
            **kwargs
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def generate_image(
        self,
        prompt: str,
        model: Union[str, OpenAIModel] = OpenAIModel.DALL_E_3,
        n: int = 1,
        size: Union[str, OpenAIImageSize] = OpenAIImageSize.SIZE_1024x1024,
        quality: Union[str, OpenAIImageQuality] = OpenAIImageQuality.STANDARD,
        style: Union[str, OpenAIImageStyle] = OpenAIImageStyle.VIVID,
        response_format: Union[str, OpenAIImageResponseFormat] = OpenAIImageResponseFormat.URL,
        user: Optional[str] = None,
        **kwargs
    ) -> OpenAIImageResponse:
        """
        Generate image using DALL-E.
        
        Args:
            prompt: Text prompt for image generation
            model: DALL-E model to use
            n: Number of images to generate (1 for DALL-E 3)
            size: Image size
            quality: Image quality
            style: Image style
            response_format: Response format (url or b64_json)
            user: User identifier for abuse monitoring
            **kwargs: Additional parameters
            
        Returns:
            OpenAIImageResponse with generated images
        """
        if not self.is_available():
            raise ValueError("OpenAI client is not properly configured")
        
        model_str = model.value if isinstance(model, OpenAIModel) else model
        size_str = size.value if isinstance(size, OpenAIImageSize) else size
        quality_str = quality.value if isinstance(quality, OpenAIImageQuality) else quality
        style_str = style.value if isinstance(style, OpenAIImageStyle) else style
        format_str = response_format.value if isinstance(response_format, OpenAIImageResponseFormat) else response_format
        
        # DALL-E 3 restrictions
        if model_str == OpenAIModel.DALL_E_3.value:
            n = 1  # DALL-E 3 only generates 1 image at a time
            if size_str not in [OpenAIImageSize.SIZE_1024x1024.value,
                               OpenAIImageSize.SIZE_1792x1024.value,
                               OpenAIImageSize.SIZE_1024x1792.value]:
                size_str = OpenAIImageSize.SIZE_1024x1024.value
        
        params = {
            "model": model_str,
            "prompt": prompt,
            "n": n,
            "size": size_str,
            "quality": quality_str,
            "style": style_str,
            "response_format": format_str,
        }
        
        if user:
            params["user"] = user
        
        params.update(kwargs)
        
        url = f"{self.base_url}/images/generations"
        
        try:
            async with self.http_client.session.post(
                url, headers=self.headers, json=params
            ) as response:
                self._update_rate_limits(response.headers)
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Parse images
                    images = []
                    for image_data in data.get("data", []):
                        images.append(OpenAIImage(**image_data))
                    
                    return OpenAIImageResponse(
                        created=datetime.fromtimestamp(data.get("created", 0)),
                        data=images
                    )
                else:
                    error_text = await response.text()
                    logger.error(f"DALL-E image generation failed: {response.status} - {error_text}")
                    raise Exception(f"OpenAI API error: {error_text}")
                    
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def transcribe_audio(
        self,
        audio_file: Union[str, bytes],
        model: Union[str, OpenAIModel] = OpenAIModel.WHISPER_1,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0.0,
        language: Optional[str] = None,
        **kwargs
    ) -> OpenAIAudioResponse:
        """
        Transcribe audio using Whisper.
        
        Args:
            audio_file: Path to audio file or audio bytes
            model: Whisper model to use
            prompt: Optional prompt to guide transcription
            response_format: Output format (json, text, srt, verbose_json, vtt)
            temperature: Sampling temperature
            language: Language code (e.g., 'en', 'es', 'fr')
            **kwargs: Additional parameters
            
        Returns:
            OpenAIAudioResponse with transcribed text
        """
        if not self.is_available():
            raise ValueError("OpenAI client is not properly configured")
        
        model_str = model.value if isinstance(model, OpenAIModel) else model
        
        # Prepare form data
        form_data = aiohttp.FormData()
        
        # Add file
        if isinstance(audio_file, str):
            # Read from file
            async with aiofiles.open(audio_file, 'rb') as f:
                audio_bytes = await f.read()
            filename = Path(audio_file).name
        else:
            audio_bytes = audio_file
            filename = "audio.mp3"
        
        # Determine content type
        content_type = mimetypes.guess_type(filename)[0] or 'audio/mpeg'
        
        form_data.add_field('file', audio_bytes, filename=filename, content_type=content_type)
        form_data.add_field('model', model_str)
        form_data.add_field('response_format', response_format)
        form_data.add_field('temperature', str(temperature))
        
        if prompt:
            form_data.add_field('prompt', prompt)
        
        if language:
            form_data.add_field('language', language)
        
        for key, value in kwargs.items():
            if value is not None:
                form_data.add_field(key, str(value))
        
        # Update headers for multipart form
        headers = self.headers.copy()
        headers.pop('Content-Type', None)  # Let aiohttp set content type
        
        url = f"{self.base_url}/audio/transcriptions"
        
        try:
            async with self.http_client.session.post(
                url, headers=headers, data=form_data
            ) as response:
                self._update_rate_limits(response.headers)
                
                if response.status == 200:
                    if response_format == 'json':
                        data = await response.json()
                        return OpenAIAudioResponse(text=data.get("text", ""))
                    else:
                        text = await response.text()
                        return OpenAIAudioResponse(text=text)
                else:
                    error_text = await response.text()
                    logger.error(f"Whisper transcription failed: {response.status} - {error_text}")
                    raise Exception(f"OpenAI API error: {error_text}")
                    
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def text_to_speech(
        self,
        input_text: str,
        voice: Union[str, OpenAITTSVoice] = OpenAITTSVoice.ALLOY,
        model: Union[str, OpenAIModel] = OpenAIModel.TTS_1,
        speed: Union[float, OpenAITTSSpeed] = 1.0,
        response_format: str = "mp3",
        **kwargs
    ) -> bytes:
        """
        Convert text to speech using OpenAI TTS.
        
        Args:
            input_text: Text to convert to speech
            voice: Voice to use
            model: TTS model to use
            speed: Speed multiplier (0.25 to 4.0)
            response_format: Audio format (mp3, opus, aac, flac, wav, pcm)
            **kwargs: Additional parameters
            
        Returns:
            Audio bytes
        """
        if not self.is_available():
            raise ValueError("OpenAI client is not properly configured")
        
        model_str = model.value if isinstance(model, OpenAIModel) else model
        voice_str = voice.value if isinstance(voice, OpenAITTSVoice) else voice
        speed_val = speed.value if isinstance(speed, OpenAITTSSpeed) else speed
        
        # Validate speed
        if not 0.25 <= speed_val <= 4.0:
            raise ValueError("Speed must be between 0.25 and 4.0")
        
        params = {
            "model": model_str,
            "input": input_text,
            "voice": voice_str,
            "speed": speed_val,
            "response_format": response_format,
        }
        
        params.update(kwargs)
        
        url = f"{self.base_url}/audio/speech"
        
        try:
            async with self.http_client.session.post(
                url, headers=self.headers, json=params
            ) as response:
                self._update_rate_limits(response.headers)
                
                if response.status == 200:
                    return await response.read()
                else:
                    error_text = await response.text()
                    logger.error(f"TTS failed: {response.status} - {error_text}")
                    raise Exception(f"OpenAI API error: {error_text}")
                    
        except Exception as e:
            logger.error(f"Error converting text to speech: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def moderate_content(
        self,
        input_text: Union[str, List[str]],
        model: Union[str, OpenAIModel] = OpenAIModel.TEXT_MODERATION_LATEST,
        **kwargs
    ) -> OpenAIModerationResponse:
        """
        Moderate content using OpenAI moderation API.
        
        Args:
            input_text: Text or list of texts to moderate
            model: Moderation model to use
            **kwargs: Additional parameters
            
        Returns:
            OpenAIModerationResponse with moderation results
        """
        if not self.is_available():
            raise ValueError("OpenAI client is not properly configured")
        
        model_str = model.value if isinstance(model, OpenAIModel) else model
        
        params = {
            "model": model_str,
            "input": input_text,
        }
        
        params.update(kwargs)
        
        url = f"{self.base_url}/moderations"
        
        try:
            async with self.http_client.session.post(
                url, headers=self.headers, json=params
            ) as response:
                self._update_rate_limits(response.headers)
                
                if response.status == 200:
                    data = await response.json()
                    return OpenAIModerationResponse(**data)
                else:
                    error_text = await response.text()
                    logger.error(f"Moderation failed: {response.status} - {error_text}")
                    raise Exception(f"OpenAI API error: {error_text}")
                    
        except Exception as e:
            logger.error(f"Error moderating content: {str(e)}")
            raise
    
    async def analyze_text(
        self,
        text: str,
        analysis_type: str = "sentiment",
        model: Union[str, OpenAIModel] = OpenAIModel.GPT_3_5_TURBO,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze text using OpenAI models.
        
        Args:
            text: Text to analyze
            analysis_type: Type of analysis (sentiment, topics, summary, etc.)
            model: Model to use
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
            "language": "Detect the language of the following text. "
                       "Respond with a JSON object containing 'language' (ISO code) and 'confidence' (0.0 to 1.0).",
            "entities": "Extract named entities from the following text. "
                       "Respond with a JSON object containing 'entities' (list of entities with types).",
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
            temperature=0.1,  # Lower temperature for consistent analysis
            max_tokens=500,
            **kwargs
        )
        
        # Parse JSON response
        try:
            content = response.get_content()
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from OpenAI response: {content[:100]}...")
            return {"raw_response": content}
    
    async def compare_texts(
        self,
        text1: str,
        text2: str,
        comparison_type: str = "similarity",
        model: Union[str, OpenAIModel] = OpenAIModel.GPT_3_5_TURBO
    ) -> Dict[str, Any]:
        """
        Compare two texts using OpenAI.
        
        Args:
            text1: First text
            text2: Second text
            comparison_type: Type of comparison (similarity, contradiction, timeline)
            model: Model to use
            
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
            content = response.get_content()
            return json.loads(content)
        except json.JSONDecodeError:
            return {"raw_response": content}
    
    def _log_usage(self, response: OpenAIResponse):
        """Log OpenAI usage for monitoring."""
        usage = response.usage
        
        log_data = {
            "model": response.model,
            "total_tokens": usage.get("total_tokens", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "cost": OpenAIUsage(**usage).calculate_cost(response.model),
            "timestamp": datetime.now().isoformat(),
        }
        
        logger.info(f"OpenAI usage: {json.dumps(log_data)}")
    
    async def get_usage_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get usage statistics (requires organization API key).
        
        Args:
            start_date: Start date for usage query
            end_date: End date for usage query
            
        Returns:
            Usage statistics
        """
        if not self.is_available():
            raise ValueError("OpenAI client is not properly configured")
        
        # Note: This endpoint requires special permissions
        # This is a placeholder implementation
        
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        
        if not end_date:
            end_date = datetime.now()
        
        try:
            # Format dates for OpenAI API
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            params = {
                "start_date": start_str,
                "end_date": end_str,
            }
            
            url = f"{self.base_url}/usage"
            
            async with self.http_client.session.get(
                url, headers=self.headers, params=params
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get usage stats: {response.status} - {error_text}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error getting usage stats: {str(e)}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on OpenAI API.
        
        Returns:
            Health status information
        """
        if not self.is_available():
            return {
                "status": "disabled",
                "message": "Client not configured (no API key)",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Test with a simple models request
            start_time = datetime.now()
            
            async with self.http_client.session.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=10
            ) as response:
                end_time = datetime.now()
                
                latency = (end_time - start_time).total_seconds() * 1000
                
                if response.status == 200:
                    # Check rate limits
                    rate_limit_info = {}
                    if self.rate_limit_remaining is not None:
                        rate_limit_info["remaining"] = self.rate_limit_remaining
                    if self.rate_limit_reset is not None:
                        rate_limit_info["reset"] = self.rate_limit_reset.isoformat()
                    
                    return {
                        "status": "healthy",
                        "message": "API is responding",
                        "latency_ms": round(latency, 2),
                        "rate_limits": rate_limit_info,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "message": f"API returned status {response.status}",
                        "timestamp": datetime.now().isoformat()
                    }
                    
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }


# Factory function for dependency injection
def get_openai_client() -> OpenAIClient:
    """
    Factory function to create OpenAI client.
    
    Returns:
        Configured OpenAIClient instance
    """
    return OpenAIClient()