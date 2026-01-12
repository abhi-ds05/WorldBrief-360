"""
Model Manager for centralized ML model lifecycle management.
Handles loading, unloading, versioning, and distribution of models.
"""
import asyncio
import gc
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union,
    Iterator, Generic, cast
)

import torch
import yaml
from pydantic import BaseModel, Field, validator

from .base import ModelType, ModelFramework, ModelDevice, ModelConfig, ModelMetadata
from .model_cache import ModelCache, get_global_cache

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='BaseModel')

# Type aliases
ModelId = str
ModelVersion = str
ModelPath = Union[str, Path]


class ModelStatus(Enum):
    """Model lifecycle status."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    ERROR = "error"
    DEPRECATED = "deprecated"


class ModelPriority(Enum):
    """Priority for model loading/unloading."""
    HIGH = 100
    MEDIUM = 50
    LOW = 10
    BACKGROUND = 1


class LoadStrategy(Enum):
    """Strategies for model loading."""
    EAGER = "eager"  # Load immediately
    LAZY = "lazy"    # Load on first use
    PRELOAD = "preload"  # Preload in background
    STREAM = "stream"    # Stream from remote


class DeviceAllocation(Enum):
    """Device allocation strategies."""
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    AFFINITY = "affinity"  # Stick models to specific devices
    AUTO = "auto"


@dataclass
class ModelInfo:
    """Information about a managed model."""
    model_id: ModelId
    model_type: ModelType
    model_name: str
    version: ModelVersion
    status: ModelStatus
    framework: ModelFramework
    device: ModelDevice
    loaded_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    load_time_seconds: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    call_count: int = 0
    error_message: Optional[str] = None
    priority: ModelPriority = ModelPriority.MEDIUM
    config: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['model_type'] = self.model_type.value
        result['status'] = self.status.value
        result['framework'] = self.framework.value
        result['device'] = self.device.value
        result['priority'] = self.priority.value
        
        # Convert datetime to string
        for field in ['loaded_at', 'last_used']:
            if result[field]:
                result[field] = result[field].isoformat()
        
        return result


@dataclass
class ModelRequest:
    """Request for model loading/usage."""
    model_id: ModelId
    model_type: ModelType
    config: Dict[str, Any]
    priority: ModelPriority = ModelPriority.MEDIUM
    timeout_seconds: Optional[float] = 30.0
    load_if_missing: bool = True
    callback: Optional[Callable[[Any], None]] = None
    user_data: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for a model."""
    model_id: ModelId
    version: ModelVersion
    inference_time_avg_ms: float = 0.0
    inference_time_p95_ms: float = 0.0
    inference_time_p99_ms: float = 0.0
    throughput_rps: float = 0.0  # Requests per second
    memory_usage_mb: float = 0.0
    error_rate: float = 0.0
    call_count: int = 0
    success_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update(self, inference_time_ms: float, success: bool = True) -> None:
        """Update metrics with new inference result."""
        self.call_count += 1
        if success:
            self.success_count += 1
        
        # Update average inference time (simple moving average)
        alpha = 0.1  # Smoothing factor
        self.inference_time_avg_ms = (
            alpha * inference_time_ms + 
            (1 - alpha) * self.inference_time_avg_ms
        )
        
        # Update throughput (simplified)
        window_seconds = 60
        self.throughput_rps = self.success_count / window_seconds
        
        # Update error rate
        self.error_rate = (self.call_count - self.success_count) / max(self.call_count, 1)
        
        self.last_updated = datetime.now()


class ModelManagerConfig(BaseModel):
    """Configuration for ModelManager."""
    # Cache settings
    cache_enabled: bool = True
    cache_max_size_mb: int = 2048  # 2GB
    cache_max_items: int = 20
    cache_strategy: str = "lru"
    
    # Loading settings
    max_concurrent_loads: int = 3
    load_timeout_seconds: float = 300.0
    load_retry_attempts: int = 3
    load_retry_delay_seconds: float = 2.0
    
    # Memory management
    max_total_memory_mb: int = 8192  # 8GB
    max_models_in_memory: int = 10
    auto_unload_enabled: bool = True
    auto_unload_idle_minutes: int = 30
    memory_pressure_threshold: float = 0.85  # 85% memory usage
    
    # Device management
    device_allocation_strategy: str = "auto"
    gpu_memory_reserved_mb: int = 512  # Reserve 512MB for system
    cpu_fallback_enabled: bool = True
    
    # Performance monitoring
    metrics_collection_enabled: bool = True
    metrics_retention_days: int = 30
    health_check_interval_seconds: int = 60
    
    # Model registry
    model_registry_url: Optional[str] = None
    local_model_dir: Path = Path("./models")
    allow_downloads: bool = True
    download_timeout_seconds: int = 600
    
    # Logging and debugging
    log_level: str = "INFO"
    enable_debug_logging: bool = False
    profile_inference: bool = False
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True
        json_encoders = {Path: str}
    
    @validator('local_model_dir')
    def validate_local_model_dir(cls, v):
        """Ensure local model directory exists."""
        v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v


class ModelLoadError(Exception):
    """Exception raised when model loading fails."""
    pass


class ModelNotFoundError(Exception):
    """Exception raised when model is not found."""
    pass


class ModelUnavailableError(Exception):
    """Exception raised when model is unavailable."""
    pass


class ModelManager:
    """
    Central manager for ML model lifecycle.
    
    Features:
    - Model loading/unloading with priority
    - Model caching with various strategies
    - Device management and load balancing
    - Performance monitoring and metrics
    - Automatic cleanup and memory management
    - Model registry integration
    """
    
    def __init__(self, config: Optional[Union[Dict[str, Any], ModelManagerConfig]] = None):
        """
        Initialize the ModelManager.
        
        Args:
            config: Configuration dictionary or ModelManagerConfig instance
        """
        # Parse configuration
        if config is None:
            config = ModelManagerConfig()
        elif isinstance(config, dict):
            config = ModelManagerConfig(**config)
        
        self.config = config
        
        # Initialize components
        self._models: Dict[ModelId, Any] = {}  # Loaded models
        self._model_info: Dict[ModelId, ModelInfo] = {}
        self._performance_metrics: Dict[ModelId, PerformanceMetrics] = {}
        
        # Initialize cache
        self._cache = None
        if self.config.cache_enabled:
            self._cache = get_global_cache(
                max_size_mb=self.config.cache_max_size_mb,
                max_items=self.config.cache_max_items,
                strategy=self.config.cache_strategy
            )
        
        # Thread pool for async operations
        self._thread_pool = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_loads,
            thread_name_prefix="ModelLoader"
        )
        
        # Locks and synchronization
        self._models_lock = threading.RLock()
        self._loading_locks: Dict[ModelId, threading.Lock] = {}
        self._loading_events: Dict[ModelId, threading.Event] = {}
        self._model_queues: Dict[ModelPriority, List[ModelRequest]] = {
            priority: [] for priority in ModelPriority
        }
        
        # Device management
        self._available_devices = self._discover_devices()
        self._device_usage: Dict[str, List[ModelId]] = {device: [] for device in self._available_devices}
        
        # Background tasks
        self._scheduler_thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._should_stop = threading.Event()
        
        # Statistics
        self._stats = {
            'total_models_loaded': 0,
            'total_models_unloaded': 0,
            'total_cache_hits': 0,
            'total_cache_misses': 0,
            'total_load_errors': 0,
            'start_time': datetime.now()
        }
        
        # Start background threads
        self._start_background_threads()
        
        logger.info(f"ModelManager initialized with config: {config.dict()}")
    
    def _discover_devices(self) -> List[str]:
        """Discover available compute devices."""
        devices = []
        
        # Check for CUDA
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                memory_mb = torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)
                devices.append(f"cuda:{i}")
                logger.info(f"Found GPU {i}: {device_name} ({memory_mb:.0f}MB)")
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append("mps")
            logger.info("Found Apple Silicon GPU (MPS)")
        
        # Always add CPU
        devices.append("cpu")
        
        logger.info(f"Available devices: {devices}")
        return devices
    
    def _start_background_threads(self) -> None:
        """Start background maintenance threads."""
        # Scheduler thread for processing model queues
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="ModelManager-Scheduler"
        )
        self._scheduler_thread.start()
        
        # Monitor thread for health checks and cleanup
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="ModelManager-Monitor"
        )
        self._monitor_thread.start()
    
    def _scheduler_loop(self) -> None:
        """Process model loading queues."""
        while not self._should_stop.is_set():
            try:
                self._process_model_queues()
                time.sleep(0.1)  # Small sleep to prevent CPU spinning
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(1)
    
    def _monitor_loop(self) -> None:
        """Monitor model health and perform cleanup."""
        while not self._should_stop.is_set():
            try:
                self._perform_health_checks()
                self._check_memory_usage()
                self._cleanup_idle_models()
                time.sleep(self.config.health_check_interval_seconds)
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(5)
    
    def _process_model_queues(self) -> None:
        """Process model loading requests from queues."""
        # Process in priority order: HIGH -> MEDIUM -> LOW -> BACKGROUND
        for priority in [ModelPriority.HIGH, ModelPriority.MEDIUM, 
                        ModelPriority.LOW, ModelPriority.BACKGROUND]:
            queue = self._model_queues[priority]
            
            if not queue:
                continue
            
            # Process requests (limit concurrent loads)
            current_loads = self._count_loading_models()
            if current_loads >= self.config.max_concurrent_loads:
                break
            
            # Get next request
            request = queue.pop(0)
            
            # Submit to thread pool for async loading
            future = self._thread_pool.submit(
                self._load_model_internal,
                request
            )
            
            # Add callback if provided
            if request.callback:
                future.add_done_callback(
                    lambda f: request.callback(f.result())  # type: ignore
                )
    
    def _count_loading_models(self) -> int:
        """Count models currently being loaded."""
        with self._models_lock:
            return sum(
                1 for info in self._model_info.values()
                if info.status == ModelStatus.LOADING
            )
    
    def _load_model_internal(self, request: ModelRequest) -> Any:
        """
        Internal method to load a model.
        
        Args:
            request: Model loading request
            
        Returns:
            Loaded model
            
        Raises:
            ModelLoadError: If loading fails
        """
        model_id = request.model_id
        
        # Check if already loading or loaded
        with self._models_lock:
            if model_id in self._model_info:
                info = self._model_info[model_id]
                if info.status == ModelStatus.LOADED:
                    return self._models[model_id]
                elif info.status == ModelStatus.LOADING:
                    # Wait for loading to complete
                    event = self._loading_events.get(model_id)
                    if event:
                        event.wait(request.timeout_seconds)
                    
                    if model_id in self._models:
                        return self._models[model_id]
        
        # Create loading lock and event
        loading_lock = self._loading_locks.setdefault(model_id, threading.Lock())
        loading_event = self._loading_events.setdefault(model_id, threading.Event())
        
        with loading_lock:
            # Double-check after acquiring lock
            with self._models_lock:
                if model_id in self._models:
                    loading_event.set()
                    return self._models[model_id]
            
            # Update model info
            with self._models_lock:
                self._model_info[model_id] = ModelInfo(
                    model_id=model_id,
                    model_type=request.model_type,
                    model_name=self._extract_model_name(model_id),
                    version="1.0.0",  # Default, should be extracted from config
                    status=ModelStatus.LOADING,
                    framework=ModelFramework.TRANSFORMERS,  # Default
                    device=ModelDevice.AUTO,
                    config=request.config
                )
            
            try:
                logger.info(f"Loading model: {model_id}")
                start_time = time.time()
                
                # Check cache first
                model = None
                if self._cache:
                    model = self._cache.get(request.config)
                
                # Load model if not in cache
                if model is None:
                    model = self._load_model_from_source(request)
                    
                    # Cache the model
                    if self._cache and model:
                        self._cache.put(request.config, model)
                        self._stats['total_cache_misses'] += 1
                    else:
                        self._stats['total_cache_hits'] += 1
                
                # Select device
                device = self._select_device(request)
                
                # Move model to device if it's a PyTorch model
                if hasattr(model, 'to') and device != 'cpu':
                    try:
                        model = model.to(device)
                    except Exception as e:
                        logger.warning(f"Failed to move model to {device}: {e}")
                        if self.config.cpu_fallback_enabled:
                            device = 'cpu'
                            model = model.to('cpu')
                
                # Update model info
                load_time = time.time() - start_time
                with self._models_lock:
                    self._models[model_id] = model
                    self._model_info[model_id] = ModelInfo(
                        model_id=model_id,
                        model_type=request.model_type,
                        model_name=self._extract_model_name(model_id),
                        version="1.0.0",
                        status=ModelStatus.LOADED,
                        framework=ModelFramework.TRANSFORMERS,
                        device=ModelDevice(device),
                        loaded_at=datetime.now(),
                        load_time_seconds=load_time,
                        memory_usage_mb=self._estimate_model_memory(model),
                        config=request.config
                    )
                    
                    # Update device usage
                    self._device_usage[device].append(model_id)
                
                logger.info(f"Model loaded: {model_id} in {load_time:.2f}s on {device}")
                self._stats['total_models_loaded'] += 1
                
                # Set event to notify waiters
                loading_event.set()
                
                return model
                
            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {e}")
                
                # Update model info with error
                with self._models_lock:
                    self._model_info[model_id] = ModelInfo(
                        model_id=model_id,
                        model_type=request.model_type,
                        model_name=self._extract_model_name(model_id),
                        version="1.0.0",
                        status=ModelStatus.ERROR,
                        framework=ModelFramework.TRANSFORMERS,
                        device=ModelDevice.AUTO,
                        error_message=str(e),
                        config=request.config
                    )
                
                self._stats['total_load_errors'] += 1
                loading_event.set()  # Still set event to unblock waiters
                raise ModelLoadError(f"Failed to load model {model_id}: {e}")
    
    def _load_model_from_source(self, request: ModelRequest) -> Any:
        """
        Load model from appropriate source.
        
        Args:
            request: Model loading request
            
        Returns:
            Loaded model
        """
        config = request.config
        model_type = request.model_type
        
        # Determine source
        source = config.get('source', 'huggingface')
        
        try:
            if source == 'huggingface':
                return self._load_from_huggingface(config)
            elif source == 'local':
                return self._load_from_local(config)
            elif source == 'openai':
                return self._load_from_openai(config)
            elif source == 'anthropic':
                return self._load_from_anthropic(config)
            elif source == 'custom':
                return self._load_custom_model(config)
            else:
                raise ValueError(f"Unknown model source: {source}")
                
        except Exception as e:
            logger.error(f"Error loading from {source}: {e}")
            raise
    
    def _load_from_huggingface(self, config: Dict[str, Any]) -> Any:
        """Load model from Hugging Face."""
        from transformers import (
            AutoModelForCausalLM, AutoModelForSeq2SeqLM,
            AutoModelForSequenceClassification, AutoTokenizer,
            AutoProcessor, AutoImageProcessor, pipeline
        )
        
        model_name = config.get('model_name')
        if not model_name:
            raise ValueError("model_name is required for Hugging Face models")
        
        # Model loading parameters
        kwargs = {
            'pretrained_model_name_or_path': model_name,
            'cache_dir': str(self.config.local_model_dir),
            'trust_remote_code': config.get('trust_remote_code', True),
            'device_map': config.get('device_map', 'auto'),
            'torch_dtype': self._parse_dtype(config.get('torch_dtype')),
            'low_cpu_mem_usage': config.get('low_cpu_mem_usage', True),
        }
        
        # Remove None values
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        # Load based on task
        task = config.get('task', 'text-generation')
        
        if task == 'text-generation':
            model = AutoModelForCausalLM.from_pretrained(**kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return {'model': model, 'tokenizer': tokenizer}
        
        elif task == 'text-classification':
            model = AutoModelForSequenceClassification.from_pretrained(**kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return {'model': model, 'tokenizer': tokenizer}
        
        elif task == 'summarization':
            model = AutoModelForSeq2SeqLM.from_pretrained(**kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return {'model': model, 'tokenizer': tokenizer}
        
        elif task == 'image-to-text':
            from transformers import VisionEncoderDecoderModel
            model = VisionEncoderDecoderModel.from_pretrained(**kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            processor = AutoImageProcessor.from_pretrained(model_name)
            return {'model': model, 'tokenizer': tokenizer, 'processor': processor}
        
        else:
            # Use pipeline for other tasks
            return pipeline(
                task,
                model=model_name,
                device_map=kwargs.get('device_map'),
                torch_dtype=kwargs.get('torch_dtype')
            )
    
    def _load_from_local(self, config: Dict[str, Any]) -> Any:
        """Load model from local filesystem."""
        model_path = Path(config.get('model_path'))
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        # Check for various model formats
        if model_path.is_file():
            # Single file model (e.g .pth, .pt, .bin)
            if model_path.suffix in ['.pth', '.pt', '.bin']:
                return torch.load(model_path, map_location='cpu')
            elif model_path.suffix == '.onnx':
                import onnxruntime as ort
                return ort.InferenceSession(str(model_path))
            else:
                raise ValueError(f"Unsupported model format: {model_path.suffix}")
        else:
            # Directory (likely Hugging Face format)
            from transformers import AutoModel, AutoTokenizer
            model = AutoModel.from_pretrained(str(model_path))
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            return {'model': model, 'tokenizer': tokenizer}
    
    def _load_from_openai(self, config: Dict[str, Any]) -> Any:
        """Load OpenAI model (API client)."""
        api_key = config.get('api_key') or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found")
        
        # Return a client configuration
        return {
            'api_key': api_key,
            'model': config.get('model_name', 'gpt-4'),
            'base_url': config.get('base_url', 'https://api.openai.com/v1'),
            'max_retries': config.get('max_retries', 3)
        }
    
    def _load_from_anthropic(self, config: Dict[str, Any]) -> Any:
        """Load Anthropic model (API client)."""
        api_key = config.get('api_key') or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("Anthropic API key not found")
        
        # Return a client configuration
        return {
            'api_key': api_key,
            'model': config.get('model_name', 'claude-3-opus-20240229'),
            'base_url': config.get('base_url', 'https://api.anthropic.com'),
            'max_retries': config.get('max_retries', 3)
        }
    
    def _load_custom_model(self, config: Dict[str, Any]) -> Any:
        """Load custom model using provided loader function."""
        loader_func = config.get('loader_function')
        if not loader_func or not callable(loader_func):
            raise ValueError("loader_function must be a callable for custom models")
        
        loader_args = config.get('loader_args', {})
        return loader_func(**loader_args)
    
    def _parse_dtype(self, dtype_str: Optional[str]) -> Optional[torch.dtype]:
        """Parse torch dtype from string."""
        if dtype_str is None:
            return None
        
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'int8': torch.int8,
            'int16': torch.int16,
            'int32': torch.int32,
            'int64': torch.int64,
        }
        
        return dtype_map.get(dtype_str.lower())
    
    def _select_device(self, request: ModelRequest) -> str:
        """
        Select appropriate device for model.
        
        Args:
            request: Model loading request
            
        Returns:
            Device string
        """
        config = request.config
        
        # Check if device is specified in config
        device = config.get('device')
        if device:
            if device in self._available_devices:
                return device
            else:
                logger.warning(f"Requested device {device} not available, using auto selection")
        
        # Auto selection based on strategy
        strategy = self.config.device_allocation_strategy
        
        if strategy == DeviceAllocation.ROUND_ROBIN.value:
            return self._select_device_round_robin()
        elif strategy == DeviceAllocation.LOAD_BALANCED.value:
            return self._select_device_load_balanced()
        elif strategy == DeviceAllocation.AFFINITY.value:
            return self._select_device_affinity(request.model_id)
        else:  # AUTO
            return self._select_device_auto(request)
    
    def _select_device_round_robin(self) -> str:
        """Round-robin device selection."""
        # Find device with fewest models
        min_models = float('inf')
        selected_device = 'cpu'
        
        for device, models in self._device_usage.items():
            if len(models) < min_models and device in self._available_devices:
                min_models = len(models)
                selected_device = device
        
        return selected_device
    
    def _select_device_load_balanced(self) -> str:
        """Load-balanced device selection."""
        # Simple implementation: choose device with most free memory
        if 'cuda:0' in self._available_devices:
            # For GPUs, check memory
            try:
                import torch.cuda as cuda
                free_memory = cuda.mem_get_info()[0] / (1024 * 1024)  # MB
                
                # Subtract reserved memory
                free_memory -= self.config.gpu_memory_reserved_mb
                
                if free_memory > 500:  # At least 500MB free
                    return 'cuda:0'
            except:
                pass
        
        return 'cpu'
    
    def _select_device_affinity(self, model_id: str) -> str:
        """Affinity-based device selection."""
        # Check if model was previously loaded on a device
        with self._models_lock:
            for device, models in self._device_usage.items():
                if model_id in models:
                    return device
        
        # Fall back to round-robin
        return self._select_device_round_robin()
    
    def _select_device_auto(self, request: ModelRequest) -> str:
        """Automatic device selection based on model type and size."""
        model_type = request.model_type
        
        # Large language models on GPU if available
        if model_type == ModelType.TEXT_GENERATION:
            if any('cuda' in d for d in self._available_devices):
                return 'cuda:0'
        
        # Image models on GPU
        elif model_type in [ModelType.TEXT_TO_IMAGE, ModelType.IMAGE_TO_TEXT]:
            if any('cuda' in d for d in self._available_devices):
                return 'cuda:0'
        
        # Embedding models can be on CPU
        elif model_type == ModelType.EMBEDDING:
            return 'cpu'
        
        return 'cpu'
    
    def _estimate_model_memory(self, model: Any) -> float:
        """Estimate model memory usage in MB."""
        try:
            # For PyTorch models
            if hasattr(model, 'parameters'):
                param_size = sum(
                    p.numel() * p.element_size()
                    for p in model.parameters()
                )
                buffer_size = sum(
                    b.numel() * b.element_size()
                    for b in model.buffers()
                )
                total_bytes = param_size + buffer_size
                return total_bytes / (1024 * 1024)
            
            # For dictionaries containing models
            elif isinstance(model, dict):
                total_mb = 0.0
                for value in model.values():
                    if hasattr(value, 'parameters'):
                        total_mb += self._estimate_model_memory(value)
                return total_mb
            
            else:
                # Rough estimate
                return 100.0  # 100MB default
            
        except Exception as e:
            logger.warning(f"Could not estimate model memory: {e}")
            return 100.0
    
    def _extract_model_name(self, model_id: str) -> str:
        """Extract human-readable model name from ID."""
        # Extract from Hugging Face format
        if '/' in model_id:
            return model_id.split('/')[-1]
        
        # Extract from path
        if os.path.sep in model_id:
            return os.path.basename(model_id)
        
        return model_id
    
    def load_model(
        self,
        model_id: str,
        model_type: Union[str, ModelType],
        config: Dict[str, Any],
        priority: Union[str, ModelPriority] = ModelPriority.MEDIUM,
        timeout_seconds: Optional[float] = None,
        async_load: bool = False
    ) -> Any:
        """
        Load a model.
        
        Args:
            model_id: Unique identifier for the model
            model_type: Type of model (text_generation, embedding, etc.)
            config: Model configuration
            priority: Loading priority
            timeout_seconds: Timeout for loading
            async_load: Whether to load asynchronously
            
        Returns:
            Loaded model
            
        Raises:
            ModelLoadError: If loading fails
            TimeoutError: If loading times out
        """
        # Parse parameters
        if isinstance(model_type, str):
            model_type = ModelType(model_type)
        
        if isinstance(priority, str):
            priority = ModelPriority[priority.upper()]
        
        if timeout_seconds is None:
            timeout_seconds = self.config.load_timeout_seconds
        
        # Check if already loaded
        with self._models_lock:
            if model_id in self._models:
                info = self._model_info.get(model_id)
                if info and info.status == ModelStatus.LOADED:
                    info.last_used = datetime.now()
                    return self._models[model_id]
        
        # Create request
        request = ModelRequest(
            model_id=model_id,
            model_type=model_type,
            config=config,
            priority=priority,
            timeout_seconds=timeout_seconds
        )
        
        if async_load:
            # Queue for background loading
            self._model_queues[priority].append(request)
            
            # Return a future
            future = self._thread_pool.submit(
                self._load_model_internal,
                request
            )
            return future
        
        else:
            # Load synchronously
            return self._load_model_internal(request)
    
    async def load_model_async(
        self,
        model_id: str,
        model_type: Union[str, ModelType],
        config: Dict[str, Any],
        priority: Union[str, ModelPriority] = ModelPriority.MEDIUM,
        timeout_seconds: Optional[float] = None
    ) -> Any:
        """
        Load a model asynchronously.
        
        Args:
            model_id: Unique identifier for the model
            model_type: Type of model
            config: Model configuration
            priority: Loading priority
            timeout_seconds: Timeout for loading
            
        Returns:
            Loaded model
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.load_model(
                model_id, model_type, config, priority, timeout_seconds
            )
        )
    
    def get_model(self, model_id: str, update_last_used: bool = True) -> Optional[Any]:
        """
        Get a loaded model.
        
        Args:
            model_id: Model identifier
            update_last_used: Whether to update last used timestamp
            
        Returns:
            The model or None if not loaded
        """
        with self._models_lock:
            if model_id in self._models:
                model = self._models[model_id]
                
                if update_last_used:
                    info = self._model_info.get(model_id)
                    if info:
                        info.last_used = datetime.now()
                        info.call_count += 1
                
                return model
        
        return None
    
    def unload_model(
        self,
        model_id: str,
        force: bool = False,
        remove_from_cache: bool = False
    ) -> bool:
        """
        Unload a model from memory.
        
        Args:
            model_id: Model identifier
            force: Force unload even if in use
            remove_from_cache: Also remove from cache
            
        Returns:
            True if unloaded, False otherwise
        """
        with self._models_lock:
            if model_id not in self._models:
                return False
            
            # Check if model is in use (simplified check)
            if not force:
                # Add more sophisticated in-use tracking if needed
                pass
            
            # Get model and info
            model = self._models.pop(model_id)
            info = self._model_info.get(model_id)
            
            if info:
                # Update device usage
                for device, models in self._device_usage.items():
                    if model_id in models:
                        models.remove(model_id)
                        break
                
                # Update status
                info.status = ModelStatus.UNLOADED
                info.loaded_at = None
            
            # Clean up model from memory
            try:
                if hasattr(model, 'cpu'):
                    model.cpu()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                del model
                gc.collect()
                
            except Exception as e:
                logger.warning(f"Error cleaning up model {model_id}: {e}")
            
            # Remove from cache if requested
            if remove_from_cache and self._cache:
                self._cache.remove(self._model_info[model_id].config)
            
            logger.info(f"Unloaded model: {model_id}")
            self._stats['total_models_unloaded'] += 1
            
            return True
    
    def unload_all_models(self, force: bool = False) -> int:
        """
        Unload all models.
        
        Args:
            force: Force unload even if in use
            
        Returns:
            Number of models unloaded
        """
        unloaded_count = 0
        
        with self._models_lock:
            model_ids = list(self._models.keys())
            
            for model_id in model_ids:
                if self.unload_model(model_id, force):
                    unloaded_count += 1
        
        return unloaded_count
    
    def _perform_health_checks(self) -> None:
        """Perform health checks on loaded models."""
        with self._models_lock:
            for model_id, model in list(self._models.items()):
                try:
                    # Simple health check: try to access model
                    if hasattr(model, 'eval'):
                        model.eval()  # Just accessing, not actually running
                    
                    # Check if model is on correct device
                    info = self._model_info.get(model_id)
                    if info and info.device:
                        if hasattr(model, 'device'):
                            current_device = str(model.device)
                            expected_device = info.device.value
                            
                            if current_device != expected_device:
                                logger.warning(
                                    f"Model {model_id} on wrong device: "
                                    f"{current_device} != {expected_device}"
                                )
                    
                except Exception as e:
                    logger.error(f"Health check failed for model {model_id}: {e}")
                    
                    # Mark as error
                    if model_id in self._model_info:
                        self._model_info[model_id].status = ModelStatus.ERROR
                        self._model_info[model_id].error_message = str(e)
    
    def _check_memory_usage(self) -> None:
        """Check memory usage and unload models if necessary."""
        if not self.config.auto_unload_enabled:
            return
        
        # Calculate total memory usage
        total_memory_mb = sum(
            info.memory_usage_mb or 0
            for info in self._model_info.values()
            if info.status == ModelStatus.LOADED
        )
        
        memory_ratio = total_memory_mb / self.config.max_total_memory_mb
        
        if memory_ratio > self.config.memory_pressure_threshold:
            logger.warning(
                f"Memory pressure: {total_memory_mb:.0f}MB / "
                f"{self.config.max_total_memory_mb:.0f}MB ({memory_ratio:.1%})"
            )
            
            # Unload low-priority, idle models
            self._unload_low_priority_models()
    
    def _cleanup_idle_models(self) -> None:
        """Unload models that have been idle for too long."""
        if not self.config.auto_unload_enabled:
            return
        
        idle_threshold = timedelta(minutes=self.config.auto_unload_idle_minutes)
        now = datetime.now()
        
        with self._models_lock:
            for model_id, info in list(self._model_info.items()):
                if (info.status == ModelStatus.LOADED and 
                    info.last_used and 
                    (now - info.last_used) > idle_threshold and
                    info.priority == ModelPriority.LOW):
                    
                    logger.info(
                        f"Unloading idle model: {model_id} "
                        f"(idle for {(now - info.last_used).total_seconds() / 60:.0f} min)"
                    )
                    self.unload_model(model_id)
    
    def _unload_low_priority_models(self) -> None:
        """Unload low priority models to free memory."""
        # Sort models by priority and last used
        models_to_unload = []
        
        with self._models_lock:
            for model_id, info in self._model_info.items():
                if info.status == ModelStatus.LOADED:
                    models_to_unload.append((
                        model_id,
                        info.priority.value,
                        info.last_used or datetime.min
                    ))
        
        # Sort by priority (ascending) and last used (ascending)
        models_to_unload.sort(key=lambda x: (x[1], x[2]))
        
        # Unload until memory pressure is reduced
        target_ratio = self.config.memory_pressure_threshold * 0.8  # Target 80% of threshold
        
        current_memory_mb = sum(
            info.memory_usage_mb or 0
            for info in self._model_info.values()
            if info.status == ModelStatus.LOADED
        )
        
        for model_id, priority, _ in models_to_unload:
            if priority <= ModelPriority.LOW.value:
                info = self._model_info.get(model_id)
                if info and info.memory_usage_mb:
                    self.unload_model(model_id)
                    current_memory_mb -= info.memory_usage_mb
                    
                    # Check if we've reduced enough
                    memory_ratio = current_memory_mb / self.config.max_total_memory_mb
                    if memory_ratio < target_ratio:
                        break
    
    def list_models(self, status: Optional[ModelStatus] = None) -> List[ModelInfo]:
        """
        List all models or models with specific status.
        
        Args:
            status: Filter by status
            
        Returns:
            List of model information
        """
        with self._models_lock:
            if status:
                return [
                    info for info in self._model_info.values()
                    if info.status == status
                ]
            else:
                return list(self._model_info.values())
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """
        Get information about a specific model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model information or None
        """
        with self._models_lock:
            return self._model_info.get(model_id)
    
    def update_model_config(
        self,
        model_id: str,
        config_updates: Dict[str, Any]
    ) -> bool:
        """
        Update model configuration.
        
        Args:
            model_id: Model identifier
            config_updates: Configuration updates
            
        Returns:
            True if updated, False otherwise
        """
        with self._models_lock:
            if model_id not in self._model_info:
                return False
            
            info = self._model_info[model_id]
            if info.config:
                info.config.update(config_updates)
            else:
                info.config = config_updates
            
            return True
    
    def get_performance_metrics(
        self,
        model_id: str
    ) -> Optional[PerformanceMetrics]:
        """
        Get performance metrics for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Performance metrics or None
        """
        return self._performance_metrics.get(model_id)
    
    def record_inference(
        self,
        model_id: str,
        inference_time_ms: float,
        success: bool = True
    ) -> None:
        """
        Record inference performance metrics.
        
        Args:
            model_id: Model identifier
            inference_time_ms: Inference time in milliseconds
            success: Whether inference was successful
        """
        if model_id not in self._performance_metrics:
            self._performance_metrics[model_id] = PerformanceMetrics(
                model_id=model_id,
                version="1.0.0"  # Should get from model info
            )
        
        metrics = self._performance_metrics[model_id]
        metrics.update(inference_time_ms, success)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        with self._models_lock:
            loaded_models = sum(
                1 for info in self._model_info.values()
                if info.status == ModelStatus.LOADED
            )
            
            loading_models = sum(
                1 for info in self._model_info.values()
                if info.status == ModelStatus.LOADING
            )
            
            total_memory_mb = sum(
                info.memory_usage_mb or 0
                for info in self._model_info.values()
                if info.status == ModelStatus.LOADED
            )
            
            uptime = datetime.now() - self._stats['start_time']
            
            stats = self._stats.copy()
            stats.update({
                'loaded_models': loaded_models,
                'loading_models': loading_models,
                'total_managed_models': len(self._model_info),
                'total_memory_usage_mb': total_memory_mb,
                'uptime_seconds': uptime.total_seconds(),
                'available_devices': self._available_devices,
                'device_usage': self._device_usage.copy(),
            })
            
            if self._cache:
                cache_stats = self._cache.stats()
                stats['cache'] = cache_stats
            
            return stats
    
    def shutdown(self) -> None:
        """Shutdown the model manager and cleanup resources."""
        logger.info("Shutting down ModelManager")
        
        # Stop background threads
        self._should_stop.set()
        
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        # Shutdown thread pool
        self._thread_pool.shutdown(wait=True)
        
        # Unload all models
        self.unload_all_models(force=True)
        
        # Clear cache
        if self._cache:
            self._cache.clear()
        
        logger.info("ModelManager shutdown complete")


# Singleton instance for global access
_global_manager: Optional[ModelManager] = None


def get_model_manager(
    config: Optional[Union[Dict[str, Any], ModelManagerConfig]] = None
) -> ModelManager:
    """
    Get or create global model manager instance.
    
    Args:
        config: Configuration for the model manager
        
    Returns:
        Global ModelManager instance
    """
    global _global_manager
    
    if _global_manager is None:
        _global_manager = ModelManager(config)
    
    return _global_manager


def shutdown_model_manager() -> None:
    """Shutdown the global model manager."""
    global _global_manager
    
    if _global_manager:
        _global_manager.shutdown()
        _global_manager = None