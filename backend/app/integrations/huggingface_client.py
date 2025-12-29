# backend/app/integrations/huggingface_client.py
"""
Hugging Face integration for WorldBrief 360.
Provides access to thousands of models for text, image, audio, and multimodal tasks.
"""

import asyncio
import json
import base64
import io
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, BinaryIO, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import aiohttp
from pydantic import BaseModel, Field, validator, HttpUrl
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from cachetools import TTLCache
import numpy as np
from PIL import Image

from app.core.config import settings
from app.core.logging_config import logger
from app.services.utils.http_client import AsyncHTTPClient
from app.models.text_generation.base import BaseLLM, GenerationConfig


class HFModelType(Enum):
    """Hugging Face model types."""
    TEXT_GENERATION = "text-generation"
    TEXT_CLASSIFICATION = "text-classification"
    TEXT_EMBEDDING = "sentence-similarity"
    IMAGE_CLASSIFICATION = "image-classification"
    OBJECT_DETECTION = "object-detection"
    IMAGE_SEGMENTATION = "image-segmentation"
    IMAGE_TO_TEXT = "image-to-text"
    TEXT_TO_IMAGE = "text-to-image"
    AUDIO_CLASSIFICATION = "audio-classification"
    AUTOMATIC_SPEECH_RECOGNITION = "automatic-speech-recognition"
    TEXT_TO_SPEECH = "text-to-speech"
    ZERO_SHOT_CLASSIFICATION = "zero-shot-classification"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question-answering"
    TABLE_QUESTION_ANSWERING = "table-question-answering"
    FILL_MASK = "fill-mask"
    TOKEN_CLASSIFICATION = "token-classification"
    FEATURE_EXTRACTION = "feature-extraction"
    CONVERSATIONAL = "conversational"
    VISUAL_QUESTION_ANSWERING = "visual-question-answering"


class HFModelFramework(Enum):
    """Model frameworks."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    JAX = "jax"
    SAFETENSORS = "safetensors"


class HFModelTask(BaseModel):
    """Model task information."""
    task: HFModelType
    description: str
    supported_formats: List[str] = Field(default_factory=list)
    input_types: List[str] = Field(default_factory=list)
    output_types: List[str] = Field(default_factory=list)


class HFModelInfo(BaseModel):
    """Hugging Face model information."""
    model_id: str
    model_name: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    tasks: List[HFModelTask] = Field(default_factory=list)
    framework: Optional[HFModelFramework] = None
    license: Optional[str] = None
    downloads: int = 0
    likes: int = 0
    last_modified: Optional[datetime] = None
    created_at: Optional[datetime] = None
    library_name: Optional[str] = None
    pipeline_tag: Optional[str] = None
    model_size: Optional[int] = None  # in bytes
    config: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True


class HFInferenceRequest(BaseModel):
    """Hugging Face inference request."""
    model_id: str
    inputs: Union[str, List[str], Dict[str, Any], bytes]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    options: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('inputs')
    def validate_inputs(cls, v):
        if v is None or (isinstance(v, (str, list, dict)) and not v):
            raise ValueError('Inputs cannot be empty')
        return v


class HFInferenceResponse(BaseModel):
    """Hugging Face inference response."""
    request_id: str
    model_id: str
    outputs: Union[str, List[str], Dict[str, Any], List[Dict[str, Any]], bytes]
    inference_time: float  # in seconds
    cached: bool = False
    metrics: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True


class HFTextGenerationResponse(HFInferenceResponse):
    """Specialized response for text generation."""
    generated_text: str
    generated_texts: List[str] = Field(default_factory=list)
    finish_reason: Optional[str] = None
    token_counts: Optional[Dict[str, int]] = None


class HFEmbeddingResponse(HFInferenceResponse):
    """Specialized response for embeddings."""
    embeddings: List[List[float]]
    embedding_dim: int
    normalized: bool = False


class HFImageGenerationResponse(HFInferenceResponse):
    """Specialized response for image generation."""
    images: List[bytes]  # List of image bytes
    image_formats: List[str]  # e.g., ["png", "jpeg"]
    dimensions: List[Tuple[int, int]]  # List of (width, height) tuples


class HuggingFaceClient:
    """
    Client for Hugging Face Inference API and model hub.
    Supports both Inference API and self-hosted models.
    """
    
    def __init__(
        self,
        api_token: Optional[str] = None,
        inference_api_url: str = "https://api-inference.huggingface.co",
        models_api_url: str = "https://huggingface.co/api",
        timeout: int = 120,
        max_retries: int = 3,
        cache_ttl: int = 3600
    ):
        """
        Initialize Hugging Face client.
        
        Args:
            api_token: Hugging Face API token
            inference_api_url: Inference API base URL
            models_api_url: Models API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            cache_ttl: Cache TTL in seconds
        """
        self.api_token = api_token or settings.HUGGINGFACE_API_TOKEN
        self.inference_api_url = inference_api_url
        self.models_api_url = models_api_url
        self.timeout = timeout
        self.max_retries = max_retries
        
        self.http_client = AsyncHTTPClient(
            timeout=timeout,
            retries=max_retries
        )
        
        # Cache for model info and inference results
        self.model_cache = TTLCache(maxsize=100, ttl=cache_ttl)
        self.inference_cache = TTLCache(maxsize=1000, ttl=300)  # 5 minutes
        
        # Model registry
        self._preferred_models = self._load_preferred_models()
        
        # Headers for authenticated requests
        self.headers = {}
        if self.api_token:
            self.headers["Authorization"] = f"Bearer {self.api_token}"
        
        logger.info(f"HuggingFace client initialized with token: {'Yes' if self.api_token else 'No'}")
    
    def _load_preferred_models(self) -> Dict[HFModelType, List[str]]:
        """Load preferred models for each task."""
        return {
            HFModelType.TEXT_GENERATION: [
                "mistralai/Mistral-7B-Instruct-v0.2",
                "google/flan-t5-xxl",
                "meta-llama/Llama-2-7b-chat-hf",
                "tiiuae/falcon-7b-instruct",
                "microsoft/phi-2",
            ],
            HFModelType.TEXT_EMBEDDING: [
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/all-MiniLM-L6-v2",
                "BAAI/bge-large-en",
                "intfloat/e5-large-v2",
            ],
            HFModelType.IMAGE_TO_TEXT: [
                "Salesforce/blip-image-captioning-base",
                "nlpconnect/vit-gpt2-image-captioning",
                "microsoft/git-base",
            ],
            HFModelType.TEXT_TO_IMAGE: [
                "stabilityai/stable-diffusion-2-1",
                "runwayml/stable-diffusion-v1-5",
                "CompVis/stable-diffusion-v1-4",
                "prompthero/openjourney",
            ],
            HFModelType.TEXT_CLASSIFICATION: [
                "distilbert-base-uncased-finetuned-sst-2-english",
                "nlptown/bert-base-multilingual-uncased-sentiment",
            ],
            HFModelType.SUMMARIZATION: [
                "facebook/bart-large-cnn",
                "google/pegasus-xsum",
                "philschmid/bart-large-cnn-samsum",
            ],
            HFModelType.TRANSLATION: [
                "Helsinki-NLP/opus-mt-en-es",
                "Helsinki-NLP/opus-mt-mul-en",
                "facebook/m2m100_418M",
            ],
        }
    
    def is_available(self) -> bool:
        """Check if client is properly configured."""
        return self.api_token is not None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError,))
    )
    async def get_model_info(self, model_id: str, use_cache: bool = True) -> HFModelInfo:
        """
        Get detailed information about a model.
        
        Args:
            model_id: Model identifier (e.g., "gpt2")
            use_cache: Whether to use cache
            
        Returns:
            HFModelInfo object
        """
        cache_key = f"model_info:{model_id}"
        
        if use_cache and cache_key in self.model_cache:
            return self.model_cache[cache_key]
        
        try:
            # Try to get model info from Hugging Face API
            url = f"{self.models_api_url}/models/{model_id}"
            
            async with self.http_client.session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Parse model info
                    model_info = self._parse_model_info(model_id, data)
                    self.model_cache[cache_key] = model_info
                    
                    return model_info
                else:
                    logger.warning(f"Failed to get model info for {model_id}: {response.status}")
                    # Return basic info
                    return HFModelInfo(
                        model_id=model_id,
                        model_name=model_id.split("/")[-1],
                        description="Model information unavailable"
                    )
                    
        except Exception as e:
            logger.error(f"Error getting model info for {model_id}: {str(e)}")
            raise
    
    def _parse_model_info(self, model_id: str, data: Dict[str, Any]) -> HFModelInfo:
        """Parse raw API response into HFModelInfo."""
        # Extract tags
        tags = data.get("tags", [])
        
        # Determine tasks from tags
        tasks = []
        for tag in tags:
            if tag in [t.value for t in HFModelType]:
                task_type = HFModelType(tag)
                tasks.append(HFModelTask(
                    task=task_type,
                    description=self._get_task_description(task_type)
                ))
        
        # Parse dates
        last_modified = None
        created_at = None
        
        if data.get("lastModified"):
            try:
                last_modified = datetime.fromisoformat(data["lastModified"].replace("Z", "+00:00"))
            except:
                pass
        
        if data.get("createdAt"):
            try:
                created_at = datetime.fromisoformat(data["createdAt"].replace("Z", "+00:00"))
            except:
                pass
        
        return HFModelInfo(
            model_id=model_id,
            model_name=data.get("id", model_id).split("/")[-1],
            description=data.get("description", ""),
            tags=tags,
            tasks=tasks,
            library_name=data.get("library_name"),
            pipeline_tag=data.get("pipeline_tag"),
            downloads=data.get("downloads", 0),
            likes=data.get("likes", 0),
            last_modified=last_modified,
            created_at=created_at,
            config=data.get("config", {})
        )
    
    def _get_task_description(self, task: HFModelType) -> str:
        """Get description for a task."""
        descriptions = {
            HFModelType.TEXT_GENERATION: "Generate text",
            HFModelType.TEXT_CLASSIFICATION: "Classify text",
            HFModelType.TEXT_EMBEDDING: "Create text embeddings",
            HFModelType.IMAGE_CLASSIFICATION: "Classify images",
            HFModelType.IMAGE_TO_TEXT: "Generate text from images",
            HFModelType.TEXT_TO_IMAGE: "Generate images from text",
            HFModelType.SUMMARIZATION: "Summarize text",
            HFModelType.TRANSLATION: "Translate text",
            HFModelType.QUESTION_ANSWERING: "Answer questions",
        }
        return descriptions.get(task, task.value.replace("-", " ").title())
    
    async def search_models(
        self,
        query: str,
        task: Optional[HFModelType] = None,
        limit: int = 20,
        sort: str = "downloads",
        direction: str = "desc"
    ) -> List[HFModelInfo]:
        """
        Search for models on Hugging Face.
        
        Args:
            query: Search query
            task: Filter by task
            limit: Maximum number of results
            sort: Sort field (downloads, likes, lastModified)
            direction: Sort direction (asc, desc)
            
        Returns:
            List of model information
        """
        try:
            params = {
                "search": query,
                "limit": limit,
                "sort": sort,
                "direction": direction,
            }
            
            if task:
                params["pipeline_tag"] = task.value
            
            url = f"{self.models_api_url}/models"
            
            async with self.http_client.session.get(url, params=params, headers=self.headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    models = []
                    for item in data:
                        model_id = item.get("id")
                        if model_id:
                            model_info = self._parse_model_info(model_id, item)
                            models.append(model_info)
                    
                    return models
                else:
                    logger.error(f"Failed to search models: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error searching models: {str(e)}")
            return []
    
    async def get_recommended_models(
        self,
        task: HFModelType,
        min_downloads: int = 10000,
        limit: int = 10
    ) -> List[HFModelInfo]:
        """
        Get recommended models for a task.
        
        Args:
            task: Model task
            min_downloads: Minimum downloads filter
            limit: Maximum number of results
            
        Returns:
            List of recommended models
        """
        # First check preferred models
        preferred = self._preferred_models.get(task, [])
        models = []
        
        for model_id in preferred[:limit]:
            try:
                model_info = await self.get_model_info(model_id)
                if model_info.downloads >= min_downloads:
                    models.append(model_info)
            except:
                continue
        
        # If not enough preferred models, search for more
        if len(models) < limit:
            search_results = await self.search_models(
                query=task.value,
                task=task,
                limit=limit * 2
            )
            
            # Filter by downloads and add to results
            for model in search_results:
                if model.downloads >= min_downloads and model.model_id not in [m.model_id for m in models]:
                    models.append(model)
                    if len(models) >= limit:
                        break
        
        return models[:limit]
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError,))
    )
    async def inference(
        self,
        request: HFInferenceRequest,
        use_cache: bool = True,
        wait_for_model: bool = True
    ) -> HFInferenceResponse:
        """
        Run inference using Hugging Face Inference API.
        
        Args:
            request: Inference request
            use_cache: Whether to use inference cache
            wait_for_model: Wait for model to load if not available
            
        Returns:
            Inference response
        """
        cache_key = None
        if use_cache:
            cache_key = self._generate_cache_key(request)
            if cache_key in self.inference_cache:
                cached_response = self.inference_cache[cache_key]
                cached_response.cached = True
                return cached_response
        
        try:
            url = f"{self.inference_api_url}/models/{request.model_id}"
            
            # Prepare request data
            data = {"inputs": request.inputs}
            if request.parameters:
                data["parameters"] = request.parameters
            if request.options:
                data["options"] = request.options
            if wait_for_model:
                data["options"] = data.get("options", {})
                data["options"]["wait_for_model"] = True
            
            # Make inference request
            start_time = datetime.now()
            
            async with self.http_client.session.post(
                url,
                headers=self.headers,
                json=data
            ) as response:
                inference_time = (datetime.now() - start_time).total_seconds()
                
                if response.status == 200:
                    response_data = await response.json()
                    
                    # Create response based on model type
                    model_info = await self.get_model_info(request.model_id, use_cache=False)
                    model_type = self._detect_model_type(model_info)
                    
                    inference_response = self._create_inference_response(
                        request=request,
                        response_data=response_data,
                        inference_time=inference_time,
                        model_type=model_type
                    )
                    
                    # Cache the response
                    if use_cache and cache_key:
                        self.inference_cache[cache_key] = inference_response
                    
                    return inference_response
                    
                elif response.status == 503:
                    # Model is loading
                    error_text = await response.text()
                    logger.warning(f"Model {request.model_id} is loading: {error_text}")
                    
                    if wait_for_model:
                        # Wait and retry
                        await asyncio.sleep(5)
                        return await self.inference(request, use_cache=False, wait_for_model=True)
                    else:
                        raise Exception(f"Model {request.model_id} is not ready")
                        
                else:
                    error_text = await response.text()
                    logger.error(f"Inference failed for {request.model_id}: {response.status} - {error_text}")
                    raise Exception(f"Inference failed: {error_text}")
                    
        except Exception as e:
            logger.error(f"Error during inference for {request.model_id}: {str(e)}")
            raise
    
    def _detect_model_type(self, model_info: HFModelInfo) -> Optional[HFModelType]:
        """Detect model type from model info."""
        if model_info.tasks:
            for task in model_info.tasks:
                return task.task
        
        if model_info.pipeline_tag:
            try:
                return HFModelType(model_info.pipeline_tag)
            except ValueError:
                pass
        
        return None
    
    def _create_inference_response(
        self,
        request: HFInferenceRequest,
        response_data: Any,
        inference_time: float,
        model_type: Optional[HFModelType] = None
    ) -> HFInferenceResponse:
        """Create appropriate inference response based on model type."""
        request_id = f"{request.model_id}:{hash(str(request.inputs))}"
        
        if model_type == HFModelType.TEXT_GENERATION:
            if isinstance(response_data, list) and len(response_data) > 0:
                first_result = response_data[0]
                generated_text = first_result.get("generated_text", "")
                
                return HFTextGenerationResponse(
                    request_id=request_id,
                    model_id=request.model_id,
                    outputs=response_data,
                    generated_text=generated_text,
                    generated_texts=[r.get("generated_text", "") for r in response_data],
                    inference_time=inference_time,
                    metrics={"response_count": len(response_data)}
                )
        
        elif model_type == HFModelType.TEXT_EMBEDDING:
            embeddings = response_data
            if isinstance(embeddings, list):
                if len(embeddings) > 0 and isinstance(embeddings[0], list):
                    embedding_dim = len(embeddings[0])
                else:
                    embedding_dim = len(embeddings)
            else:
                embeddings = [embeddings]
                embedding_dim = len(embeddings[0]) if isinstance(embeddings[0], list) else 1
            
            return HFEmbeddingResponse(
                request_id=request_id,
                model_id=request.model_id,
                outputs=response_data,
                embeddings=embeddings,
                embedding_dim=embedding_dim,
                inference_time=inference_time,
                metrics={"embedding_dim": embedding_dim}
            )
        
        elif model_type == HFModelType.TEXT_TO_IMAGE:
            # Assuming response is image bytes
            images = [response_data] if isinstance(response_data, bytes) else response_data
            return HFImageGenerationResponse(
                request_id=request_id,
                model_id=request.model_id,
                outputs=response_data,
                images=images,
                image_formats=["png"] * len(images),
                dimensions=[(512, 512)] * len(images),
                inference_time=inference_time,
                metrics={"image_count": len(images)}
            )
        
        # Generic response for other model types
        return HFInferenceResponse(
            request_id=request_id,
            model_id=request.model_id,
            outputs=response_data,
            inference_time=inference_time,
            metrics={"response_type": type(response_data).__name__}
        )
    
    def _generate_cache_key(self, request: HFInferenceRequest) -> str:
        """Generate cache key for inference request."""
        import hashlib
        
        key_data = {
            "model_id": request.model_id,
            "inputs": request.inputs,
            "parameters": request.parameters,
            "options": request.options,
        }
        
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    # Specialized methods for common tasks
    
    async def generate_text(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        max_length: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        num_return_sequences: int = 1,
        **kwargs
    ) -> HFTextGenerationResponse:
        """
        Generate text using a language model.
        
        Args:
            prompt: Input prompt
            model_id: Model ID (uses default if None)
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            repetition_penalty: Penalty for repetition
            num_return_sequences: Number of sequences to generate
            **kwargs: Additional parameters
            
        Returns:
            Text generation response
        """
        if model_id is None:
            # Use default text generation model
            preferred = self._preferred_models[HFModelType.TEXT_GENERATION]
            model_id = preferred[0]
        
        parameters = {
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "num_return_sequences": num_return_sequences,
            "do_sample": temperature > 0,
            **kwargs
        }
        
        request = HFInferenceRequest(
            model_id=model_id,
            inputs=prompt,
            parameters=parameters
        )
        
        response = await self.inference(request)
        
        if isinstance(response, HFTextGenerationResponse):
            return response
        else:
            # Convert generic response to text generation response
            return HFTextGenerationResponse(
                request_id=response.request_id,
                model_id=response.model_id,
                outputs=response.outputs,
                generated_text=str(response.outputs),
                generated_texts=[str(response.outputs)],
                inference_time=response.inference_time,
                cached=response.cached,
                metrics=response.metrics,
                warnings=response.warnings,
                created_at=response.created_at
            )
    
    async def create_embeddings(
        self,
        texts: Union[str, List[str]],
        model_id: Optional[str] = None,
        normalize: bool = True,
        **kwargs
    ) -> HFEmbeddingResponse:
        """
        Create text embeddings.
        
        Args:
            texts: Text or list of texts
            model_id: Model ID (uses default if None)
            normalize: Whether to normalize embeddings
            **kwargs: Additional parameters
            
        Returns:
            Embedding response
        """
        if model_id is None:
            # Use default embedding model
            preferred = self._preferred_models[HFModelType.TEXT_EMBEDDING]
            model_id = preferred[0]
        
        if isinstance(texts, str):
            texts = [texts]
        
        parameters = {
            "normalize": normalize,
            **kwargs
        }
        
        request = HFInferenceRequest(
            model_id=model_id,
            inputs=texts,
            parameters=parameters
        )
        
        response = await self.inference(request)
        
        if isinstance(response, HFEmbeddingResponse):
            return response
        else:
            # Convert generic response to embedding response
            embeddings = response.outputs
            if isinstance(embeddings, list):
                embedding_dim = len(embeddings[0]) if embeddings else 0
            else:
                embeddings = [embeddings]
                embedding_dim = len(embeddings[0]) if isinstance(embeddings[0], list) else 1
            
            return HFEmbeddingResponse(
                request_id=response.request_id,
                model_id=response.model_id,
                outputs=response.outputs,
                embeddings=embeddings,
                embedding_dim=embedding_dim,
                normalized=normalize,
                inference_time=response.inference_time,
                cached=response.cached,
                metrics=response.metrics,
                warnings=response.warnings,
                created_at=response.created_at
            )
    
    async def generate_image(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        num_images: int = 1,
        width: int = 512,
        height: int = 512,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        **kwargs
    ) -> HFImageGenerationResponse:
        """
        Generate image from text prompt.
        
        Args:
            prompt: Text prompt
            model_id: Model ID (uses default if None)
            negative_prompt: Negative prompt
            num_images: Number of images to generate
            width: Image width
            height: Image height
            guidance_scale: Guidance scale
            num_inference_steps: Number of inference steps
            seed: Random seed
            **kwargs: Additional parameters
            
        Returns:
            Image generation response
        """
        if model_id is None:
            # Use default image generation model
            preferred = self._preferred_models[HFModelType.TEXT_TO_IMAGE]
            model_id = preferred[0]
        
        inputs = {"inputs": prompt}
        if negative_prompt:
            inputs["negative_prompt"] = negative_prompt
        
        parameters = {
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "num_images_per_prompt": num_images,
            **kwargs
        }
        
        if seed is not None:
            parameters["seed"] = seed
        
        request = HFInferenceRequest(
            model_id=model_id,
            inputs=inputs,
            parameters=parameters
        )
        
        response = await self.inference(request)
        
        if isinstance(response, HFImageGenerationResponse):
            return response
        else:
            # Convert generic response to image generation response
            images = [response.outputs] if isinstance(response.outputs, bytes) else response.outputs
            
            return HFImageGenerationResponse(
                request_id=response.request_id,
                model_id=response.model_id,
                outputs=response.outputs,
                images=images,
                image_formats=["png"] * len(images),
                dimensions=[(width, height)] * len(images),
                inference_time=response.inference_time,
                cached=response.cached,
                metrics=response.metrics,
                warnings=response.warnings,
                created_at=response.created_at
            )
    
    async def caption_image(
        self,
        image_bytes: bytes,
        model_id: Optional[str] = None,
        **kwargs
    ) -> HFInferenceResponse:
        """
        Generate caption for an image.
        
        Args:
            image_bytes: Image bytes
            model_id: Model ID (uses default if None)
            **kwargs: Additional parameters
            
        Returns:
            Inference response with caption
        """
        if model_id is None:
            # Use default image captioning model
            preferred = self._preferred_models[HFModelType.IMAGE_TO_TEXT]
            model_id = preferred[0]
        
        # Encode image as base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        request = HFInferenceRequest(
            model_id=model_id,
            inputs=image_b64,
            parameters=kwargs
        )
        
        return await self.inference(request)
    
    async def classify_text(
        self,
        text: str,
        model_id: Optional[str] = None,
        candidate_labels: Optional[List[str]] = None,
        **kwargs
    ) -> HFInferenceResponse:
        """
        Classify text.
        
        Args:
            text: Text to classify
            model_id: Model ID (uses default if None)
            candidate_labels: Candidate labels for zero-shot classification
            **kwargs: Additional parameters
            
        Returns:
            Inference response with classification
        """
        if model_id is None:
            if candidate_labels:
                # Use zero-shot classification
                model_type = HFModelType.ZERO_SHOT_CLASSIFICATION
            else:
                # Use regular text classification
                model_type = HFModelType.TEXT_CLASSIFICATION
            
            preferred = self._preferred_models.get(model_type, [])
            model_id = preferred[0] if preferred else "distilbert-base-uncased-finetuned-sst-2-english"
        
        inputs = text
        if candidate_labels and model_id.endswith("zero-shot-classification"):
            inputs = {
                "sequences": text,
                "candidate_labels": candidate_labels
            }
        
        request = HFInferenceRequest(
            model_id=model_id,
            inputs=inputs,
            parameters=kwargs
        )
        
        return await self.inference(request)
    
    async def summarize_text(
        self,
        text: str,
        model_id: Optional[str] = None,
        max_length: int = 150,
        min_length: int = 30,
        **kwargs
    ) -> HFInferenceResponse:
        """
        Summarize text.
        
        Args:
            text: Text to summarize
            model_id: Model ID (uses default if None)
            max_length: Maximum summary length
            min_length: Minimum summary length
            **kwargs: Additional parameters
            
        Returns:
            Inference response with summary
        """
        if model_id is None:
            preferred = self._preferred_models[HFModelType.SUMMARIZATION]
            model_id = preferred[0]
        
        parameters = {
            "max_length": max_length,
            "min_length": min_length,
            **kwargs
        }
        
        request = HFInferenceRequest(
            model_id=model_id,
            inputs=text,
            parameters=parameters
        )
        
        return await self.inference(request)
    
    async def translate_text(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None,
        model_id: Optional[str] = None,
        **kwargs
    ) -> HFInferenceResponse:
        """
        Translate text.
        
        Args:
            text: Text to translate
            target_language: Target language code (e.g., "es", "fr")
            source_language: Source language code (optional)
            model_id: Model ID (uses default if None)
            **kwargs: Additional parameters
            
        Returns:
            Inference response with translation
        """
        if model_id is None:
            # Try to find appropriate translation model
            if source_language and target_language:
                model_query = f"opus-mt-{source_language}-{target_language}"
                models = await self.search_models(model_query, task=HFModelType.TRANSLATION, limit=5)
                if models:
                    model_id = models[0].model_id
            
            if model_id is None:
                preferred = self._preferred_models[HFModelType.TRANSLATION]
                model_id = preferred[0]
        
        # Format for translation models
        if source_language and target_language:
            inputs = f"{source_language}: {text}"
            parameters = {"target_lang": target_language, **kwargs}
        else:
            inputs = text
            parameters = kwargs
        
        request = HFInferenceRequest(
            model_id=model_id,
            inputs=inputs,
            parameters=parameters
        )
        
        return await self.inference(request)
    
    async def batch_inference(
        self,
        requests: List[HFInferenceRequest],
        max_concurrent: int = 5
    ) -> List[HFInferenceResponse]:
        """
        Run multiple inference requests concurrently.
        
        Args:
            requests: List of inference requests
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of inference responses
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_with_semaphore(request):
            async with semaphore:
                return await self.inference(request, wait_for_model=False)
        
        tasks = [run_with_semaphore(request) for request in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Batch inference failed for request {i}: {str(response)}")
                processed_responses.append(HFInferenceResponse(
                    request_id=f"error:{i}",
                    model_id=requests[i].model_id,
                    outputs=None,
                    inference_time=0,
                    errors=[str(response)]
                ))
            else:
                processed_responses.append(response)
        
        return processed_responses
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Hugging Face API.
        
        Returns:
            Health status information
        """
        if not self.is_available():
            return {
                "status": "disabled",
                "message": "Client not configured (no API token)",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Test with a simple model info request
            start_time = datetime.now()
            
            async with self.http_client.session.get(
                f"{self.models_api_url}/models/gpt2",
                headers=self.headers
            ) as response:
                end_time = datetime.now()
                
                latency = (end_time - start_time).total_seconds() * 1000
                
                if response.status == 200:
                    return {
                        "status": "healthy",
                        "message": "API is responding",
                        "latency_ms": round(latency, 2),
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
def get_huggingface_client() -> HuggingFaceClient:
    """
    Factory function to create Hugging Face client.
    
    Returns:
        Configured HuggingFaceClient instance
    """
    return HuggingFaceClient()