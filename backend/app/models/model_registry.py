"""
Model Registry for versioning, metadata management, and model lifecycle tracking.
Acts as a central catalog for all ML models in the system.
"""
import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import uuid
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Union, Callable,
    Iterator, ClassVar, BinaryIO
)

import yaml
from pydantic import (
    BaseModel, Field, validator, root_validator,
    HttpUrl, IPvAnyAddress
)

from .base import ModelType, ModelFramework, ModelDevice # type: ignore

logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Model lifecycle stages."""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"
    DEPRECATED = "Deprecated"


class ModelAccessLevel(Enum):
    """Model access levels."""
    PUBLIC = "public"
    PRIVATE = "private"
    INTERNAL = "internal"
    RESTRICTED = "restricted"


class ModelSourceType(Enum):
    """Model source types."""
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    GIT = "git"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"


class VersioningScheme(Enum):
    """Versioning schemes for models."""
    SEMVER = "semver"  # Semantic Versioning (major.minor.patch)
    TIMESTAMP = "timestamp"  # YYYYMMDD.HHMMSS
    INCREMENTAL = "incremental"  # Simple integer increment
    GIT_HASH = "git_hash"  # Git commit hash


class ArtifactType(Enum):
    """Types of model artifacts."""
    MODEL_WEIGHTS = "model_weights"
    CONFIG = "config"
    TOKENIZER = "tokenizer"
    PROCESSOR = "processor"
    VOCABULARY = "vocabulary"
    CHECKPOINT = "checkpoint"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    METADATA = "metadata"
    EVALUATION = "evaluation"
    TRAINING_LOG = "training_log"
    CUSTOM = "custom"


class DependencyType(Enum):
    """Types of model dependencies."""
    FRAMEWORK = "framework"
    LIBRARY = "library"
    SYSTEM = "system"
    HARDWARE = "hardware"


@dataclass
class ModelDependency:
    """Dependency for a model."""
    name: str
    version: str
    dependency_type: DependencyType
    required: bool = True
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['dependency_type'] = self.dependency_type.value
        return result


@dataclass
class ModelArtifact:
    """Artifact associated with a model version."""
    artifact_id: str
    model_id: str
    version: str
    artifact_type: ArtifactType
    path: str
    size_bytes: int
    checksum: str
    checksum_algorithm: str = "sha256"
    created_at: datetime = field(default_factory=datetime.now)
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['artifact_type'] = self.artifact_type.value
        result['created_at'] = self.created_at.isoformat()
        return result


@dataclass
class ModelVersion:
    """Version of a model in the registry."""
    model_id: str
    version: str
    version_scheme: VersioningScheme
    stage: ModelStage = ModelStage.NONE
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Model metadata
    model_type: Optional[ModelType] = None
    framework: Optional[ModelFramework] = None
    framework_version: Optional[str] = None
    parameters: Optional[int] = None  # Number of parameters
    size_bytes: Optional[int] = None
    
    # Performance metrics
    evaluation_metrics: Dict[str, float] = field(default_factory=dict)
    training_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Source information
    source_type: Optional[ModelSourceType] = None
    source_uri: Optional[str] = None
    source_commit: Optional[str] = None
    
    # Dependencies
    dependencies: List[ModelDependency] = field(default_factory=list)
    
    # References
    parent_version: Optional[str] = None  # Previous version
    base_model: Optional[str] = None  # Base model if fine-tuned
    
    # Artifacts
    artifacts: List[ModelArtifact] = field(default_factory=list)
    
    # Runtime info
    compatible_devices: List[ModelDevice] = field(default_factory=list)
    memory_requirements_mb: Optional[float] = None
    inference_time_ms: Optional[float] = None
    
    # Custom metadata
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        
        # Convert enums
        if self.model_type:
            result['model_type'] = self.model_type.value
        if self.framework:
            result['framework'] = self.framework.value
        if self.source_type:
            result['source_type'] = self.source_type.value
        
        # Convert lists of enums
        result['compatible_devices'] = [d.value for d in self.compatible_devices]
        result['stage'] = self.stage.value
        result['version_scheme'] = self.version_scheme.value
        
        # Convert dependencies and artifacts
        result['dependencies'] = [d.to_dict() for d in self.dependencies]
        result['artifacts'] = [a.to_dict() for a in self.artifacts]
        
        # Convert timestamps
        result['created_at'] = self.created_at.isoformat()
        result['updated_at'] = self.updated_at.isoformat()
        
        return result


@dataclass
class RegisteredModel:
    """A model registered in the registry."""
    model_id: str
    name: str
    description: Optional[str] = None
    owner: Optional[str] = None
    team: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Model characteristics
    primary_type: Optional[ModelType] = None
    supported_frameworks: List[ModelFramework] = field(default_factory=list)
    supported_tasks: List[str] = field(default_factory=list)
    
    # Access control
    access_level: ModelAccessLevel = ModelAccessLevel.PRIVATE
    allowed_users: List[str] = field(default_factory=list)
    allowed_teams: List[str] = field(default_factory=list)
    
    # Versioning
    latest_version: Optional[str] = None
    latest_production_version: Optional[str] = None
    versions: List[ModelVersion] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    documentation_url: Optional[str] = None
    license: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        
        # Convert enums
        if self.primary_type:
            result['primary_type'] = self.primary_type.value
        result['supported_frameworks'] = [f.value for f in self.supported_frameworks]
        result['access_level'] = self.access_level.value
        
        # Convert versions
        result['versions'] = [v.to_dict() for v in self.versions]
        
        # Convert timestamps
        result['created_at'] = self.created_at.isoformat()
        result['updated_at'] = self.updated_at.isoformat()
        
        return result


class ModelRegistryConfig(BaseModel):
    """Configuration for the model registry."""
    # Storage settings
    storage_backend: str = "sqlite"  # "sqlite", "postgresql", "mongodb", "filesystem"
    storage_uri: Optional[str] = None  # Connection string or path
    local_storage_path: Path = Path("./model_registry")
    
    # Registry settings
    default_namespace: str = "default"
    enable_namespaces: bool = True
    auto_versioning: bool = True
    default_version_scheme: str = "semver"
    
    # Model validation
    validate_checksums: bool = True
    allowed_checksum_algorithms: List[str] = Field(default_factory=lambda: ["sha256", "md5"])
    max_model_size_mb: int = 10240  # 10GB
    
    # Security
    enable_access_control: bool = True
    require_authentication: bool = False
    default_access_level: str = "private"
    
    # Performance
    cache_enabled: bool = True
    cache_max_size_mb: int = 512
    cache_ttl_seconds: int = 300
    
    # Backup and retention
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    retention_days: int = 365
    auto_cleanup: bool = True
    
    # Integration
    webhook_urls: List[str] = Field(default_factory=list)
    enable_webhooks: bool = False
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True
        json_encoders = {Path: str}
    
    @validator('local_storage_path')
    def validate_local_storage_path(cls, v):
        """Ensure local storage directory exists."""
        v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator('storage_uri')
    def validate_storage_uri(cls, v, values):
        """Set default storage URI based on backend."""
        if v is None:
            backend = values.get('storage_backend', 'sqlite')
            if backend == 'sqlite':
                storage_path = values.get('local_storage_path', Path('./model_registry'))
                v = f"sqlite:///{storage_path}/registry.db"
            elif backend == 'filesystem':
                storage_path = values.get('local_storage_path', Path('./model_registry'))
                v = f"file://{storage_path}"
        
        return v


class RegistryError(Exception):
    """Base exception for registry errors."""
    pass


class ModelNotFoundError(RegistryError):
    """Exception raised when model is not found."""
    pass


class VersionNotFoundError(RegistryError):
    """Exception raised when model version is not found."""
    pass


class DuplicateModelError(RegistryError):
    """Exception raised when model already exists."""
    pass


class InvalidModelError(RegistryError):
    """Exception raised when model is invalid."""
    pass


class AccessDeniedError(RegistryError):
    """Exception raised when access is denied."""
    pass


class StorageBackend(ABC):
    """Abstract base class for registry storage backends."""
    
    @abstractmethod
    def save_model(self, model: RegisteredModel) -> None:
        """Save a model to storage."""
        pass
    
    @abstractmethod
    def load_model(self, model_id: str) -> Optional[RegisteredModel]:
        """Load a model from storage."""
        pass
    
    @abstractmethod
    def delete_model(self, model_id: str) -> bool:
        """Delete a model from storage."""
        pass
    
    @abstractmethod
    def list_models(
        self,
        namespace: Optional[str] = None,
        tags: Optional[List[str]] = None,
        model_type: Optional[ModelType] = None
    ) -> List[RegisteredModel]:
        """List models with optional filtering."""
        pass
    
    @abstractmethod
    def search_models(
        self,
        query: str,
        namespace: Optional[str] = None
    ) -> List[RegisteredModel]:
        """Search models by name, description, or tags."""
        pass
    
    @abstractmethod
    def save_artifact(
        self,
        artifact: ModelArtifact,
        data: Optional[BinaryIO] = None
    ) -> str:
        """Save an artifact to storage."""
        pass
    
    @abstractmethod
    def load_artifact(
        self,
        artifact_id: str
    ) -> Optional[Tuple[ModelArtifact, Optional[BinaryIO]]]:
        """Load an artifact from storage."""
        pass
    
    @abstractmethod
    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete an artifact from storage."""
        pass
    
    @abstractmethod
    def get_model_versions(
        self,
        model_id: str,
        stage: Optional[ModelStage] = None
    ) -> List[ModelVersion]:
        """Get versions of a model."""
        pass
    
    @abstractmethod
    def get_version(
        self,
        model_id: str,
        version: str
    ) -> Optional[ModelVersion]:
        """Get a specific version of a model."""
        pass


class FileSystemStorage(StorageBackend):
    """File system storage backend for the registry."""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.models_path = self.base_path / "models"
        self.artifacts_path = self.base_path / "artifacts"
        self.metadata_path = self.base_path / "metadata"
        
        for path in [self.models_path, self.artifacts_path, self.metadata_path]:
            path.mkdir(exist_ok=True)
        
        self._lock = Lock()
        logger.info(f"FileSystemStorage initialized at {base_path}")
    
    def _get_model_path(self, model_id: str) -> Path:
        """Get path for model metadata."""
        safe_id = self._sanitize_filename(model_id)
        return self.models_path / f"{safe_id}.json"
    
    def _get_artifact_path(self, artifact_id: str) -> Path:
        """Get path for artifact data."""
        # Use first 2 chars as subdirectory to avoid too many files in one directory
        subdir = artifact_id[:2]
        artifact_dir = self.artifacts_path / subdir
        artifact_dir.mkdir(exist_ok=True)
        return artifact_dir / f"{artifact_id}.bin"
    
    def _get_metadata_path(self, artifact_id: str) -> Path:
        """Get path for artifact metadata."""
        subdir = artifact_id[:2]
        metadata_dir = self.metadata_path / subdir
        metadata_dir.mkdir(exist_ok=True)
        return metadata_dir / f"{artifact_id}.json"
    
    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Sanitize filename to be safe for filesystem."""
        # Replace unsafe characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Limit length
        if len(filename) > 255:
            filename = filename[:255]
        return filename
    
    def save_model(self, model: RegisteredModel) -> None:
        """Save a model to storage."""
        model_path = self._get_model_path(model.model_id)
        
        with self._lock:
            try:
                with open(model_path, 'w', encoding='utf-8') as f:
                    json.dump(model.to_dict(), f, indent=2, ensure_ascii=False)
            except Exception as e:
                raise RegistryError(f"Failed to save model {model.model_id}: {e}")
    
    def load_model(self, model_id: str) -> Optional[RegisteredModel]:
        """Load a model from storage."""
        model_path = self._get_model_path(model_id)
        
        if not model_path.exists():
            return None
        
        with self._lock:
            try:
                with open(model_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert back to RegisteredModel
                return self._dict_to_model(data)
                
            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {e}")
                return None
    
    def _dict_to_model(self, data: Dict[str, Any]) -> RegisteredModel:
        """Convert dictionary to RegisteredModel."""
        # Convert versions
        versions = []
        for v_data in data.get('versions', []):
            version = self._dict_to_version(v_data)
            versions.append(version)
        
        # Create model
        model = RegisteredModel(
            model_id=data['model_id'],
            name=data['name'],
            description=data.get('description'),
            owner=data.get('owner'),
            team=data.get('team'),
            tags=data.get('tags', []),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            primary_type=ModelType(data['primary_type']) if data.get('primary_type') else None,
            supported_frameworks=[ModelFramework(f) for f in data.get('supported_frameworks', [])],
            supported_tasks=data.get('supported_tasks', []),
            access_level=ModelAccessLevel(data.get('access_level', 'private')),
            allowed_users=data.get('allowed_users', []),
            allowed_teams=data.get('allowed_teams', []),
            latest_version=data.get('latest_version'),
            latest_production_version=data.get('latest_production_version'),
            versions=versions,
            metadata=data.get('metadata', {}),
            documentation_url=data.get('documentation_url'),
            license=data.get('license')
        )
        
        return model
    
    def _dict_to_version(self, data: Dict[str, Any]) -> ModelVersion:
        """Convert dictionary to ModelVersion."""
        # Convert dependencies
        dependencies = []
        for d_data in data.get('dependencies', []):
            dependency = ModelDependency(
                name=d_data['name'],
                version=d_data['version'],
                dependency_type=DependencyType(d_data['dependency_type']),
                required=d_data.get('required', True),
                description=d_data.get('description')
            )
            dependencies.append(dependency)
        
        # Convert artifacts
        artifacts = []
        for a_data in data.get('artifacts', []):
            artifact = ModelArtifact(
                artifact_id=a_data['artifact_id'],
                model_id=a_data['model_id'],
                version=a_data['version'],
                artifact_type=ArtifactType(a_data['artifact_type']),
                path=a_data['path'],
                size_bytes=a_data['size_bytes'],
                checksum=a_data['checksum'],
                checksum_algorithm=a_data.get('checksum_algorithm', 'sha256'),
                created_at=datetime.fromisoformat(a_data['created_at']),
                description=a_data.get('description'),
                metadata=a_data.get('metadata', {})
            )
            artifacts.append(artifact)
        
        # Create version
        version = ModelVersion(
            model_id=data['model_id'],
            version=data['version'],
            version_scheme=VersioningScheme(data['version_scheme']),
            stage=ModelStage(data.get('stage', 'None')),
            description=data.get('description'),
            tags=data.get('tags', []),
            created_at=datetime.fromisoformat(data['created_at']),
            created_by=data.get('created_by'),
            updated_at=datetime.fromisoformat(data['updated_at']),
            model_type=ModelType(data['model_type']) if data.get('model_type') else None,
            framework=ModelFramework(data['framework']) if data.get('framework') else None,
            framework_version=data.get('framework_version'),
            parameters=data.get('parameters'),
            size_bytes=data.get('size_bytes'),
            evaluation_metrics=data.get('evaluation_metrics', {}),
            training_metrics=data.get('training_metrics', {}),
            source_type=ModelSourceType(data['source_type']) if data.get('source_type') else None,
            source_uri=data.get('source_uri'),
            source_commit=data.get('source_commit'),
            dependencies=dependencies,
            parent_version=data.get('parent_version'),
            base_model=data.get('base_model'),
            artifacts=artifacts,
            compatible_devices=[ModelDevice(d) for d in data.get('compatible_devices', [])],
            memory_requirements_mb=data.get('memory_requirements_mb'),
            inference_time_ms=data.get('inference_time_ms'),
            custom_metadata=data.get('custom_metadata', {})
        )
        
        return version
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model from storage."""
        model_path = self._get_model_path(model_id)
        
        with self._lock:
            if not model_path.exists():
                return False
            
            try:
                # Load model first to get artifact IDs
                model = self.load_model(model_id)
                if model:
                    # Delete all artifacts
                    for version in model.versions:
                        for artifact in version.artifacts:
                            self.delete_artifact(artifact.artifact_id)
                
                # Delete model file
                model_path.unlink()
                return True
                
            except Exception as e:
                logger.error(f"Failed to delete model {model_id}: {e}")
                return False
    
    def list_models(
        self,
        namespace: Optional[str] = None,
        tags: Optional[List[str]] = None,
        model_type: Optional[ModelType] = None
    ) -> List[RegisteredModel]:
        """List models with optional filtering."""
        models = []
        
        with self._lock:
            for model_file in self.models_path.glob("*.json"):
                try:
                    model = self.load_model(model_file.stem)
                    if model:
                        # Apply filters
                        if tags and not all(tag in model.tags for tag in tags):
                            continue
                        if model_type and model.primary_type != model_type:
                            continue
                        
                        models.append(model)
                except Exception as e:
                    logger.error(f"Failed to load model from {model_file}: {e}")
        
        return models
    
    def search_models(
        self,
        query: str,
        namespace: Optional[str] = None
    ) -> List[RegisteredModel]:
        """Search models by name, description, or tags."""
        query_lower = query.lower()
        results = []
        
        with self._lock:
            for model_file in self.models_path.glob("*.json"):
                try:
                    model = self.load_model(model_file.stem)
                    if not model:
                        continue
                    
                    # Search in name, description, and tags
                    if (query_lower in model.name.lower() or
                        (model.description and query_lower in model.description.lower()) or
                        any(query_lower in tag.lower() for tag in model.tags)):
                        results.append(model)
                        
                except Exception as e:
                    logger.error(f"Failed to search model from {model_file}: {e}")
        
        return results
    
    def save_artifact(
        self,
        artifact: ModelArtifact,
        data: Optional[BinaryIO] = None
    ) -> str:
        """Save an artifact to storage."""
        artifact_path = self._get_artifact_path(artifact.artifact_id)
        metadata_path = self._get_metadata_path(artifact.artifact_id)
        
        with self._lock:
            try:
                # Save metadata
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(artifact.to_dict(), f, indent=2, ensure_ascii=False)
                
                # Save data if provided
                if data:
                    with open(artifact_path, 'wb') as f:
                        shutil.copyfileobj(data, f)
                
                return str(artifact_path)
                
            except Exception as e:
                raise RegistryError(f"Failed to save artifact {artifact.artifact_id}: {e}")
    
    def load_artifact(
        self,
        artifact_id: str
    ) -> Optional[Tuple[ModelArtifact, Optional[BinaryIO]]]:
        """Load an artifact from storage."""
        metadata_path = self._get_metadata_path(artifact_id)
        artifact_path = self._get_artifact_path(artifact_id)
        
        if not metadata_path.exists():
            return None
        
        with self._lock:
            try:
                # Load metadata
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert to ModelArtifact
                artifact = ModelArtifact(
                    artifact_id=data['artifact_id'],
                    model_id=data['model_id'],
                    version=data['version'],
                    artifact_type=ArtifactType(data['artifact_type']),
                    path=data['path'],
                    size_bytes=data['size_bytes'],
                    checksum=data['checksum'],
                    checksum_algorithm=data.get('checksum_algorithm', 'sha256'),
                    created_at=datetime.fromisoformat(data['created_at']),
                    description=data.get('description'),
                    metadata=data.get('metadata', {})
                )
                
                # Load data if file exists
                data_file = None
                if artifact_path.exists():
                    data_file = open(artifact_path, 'rb')
                
                return (artifact, data_file)
                
            except Exception as e:
                logger.error(f"Failed to load artifact {artifact_id}: {e}")
                return None
    
    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete an artifact from storage."""
        metadata_path = self._get_metadata_path(artifact_id)
        artifact_path = self._get_artifact_path(artifact_id)
        
        with self._lock:
            try:
                # Delete metadata
                if metadata_path.exists():
                    metadata_path.unlink()
                
                # Delete data
                if artifact_path.exists():
                    artifact_path.unlink()
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to delete artifact {artifact_id}: {e}")
                return False
    
    def get_model_versions(
        self,
        model_id: str,
        stage: Optional[ModelStage] = None
    ) -> List[ModelVersion]:
        """Get versions of a model."""
        model = self.load_model(model_id)
        if not model:
            return []
        
        if stage:
            return [v for v in model.versions if v.stage == stage]
        else:
            return model.versions
    
    def get_version(
        self,
        model_id: str,
        version: str
    ) -> Optional[ModelVersion]:
        """Get a specific version of a model."""
        model = self.load_model(model_id)
        if not model:
            return None
        
        for v in model.versions:
            if v.version == version:
                return v
        
        return None


class ModelRegistry:
    """
    Central registry for ML model management.
    
    Features:
    - Model versioning with multiple schemes
    - Metadata management and search
    - Artifact storage with checksum validation
    - Access control and namespacing
    - Model lineage and dependency tracking
    - Webhook integration for events
    """
    
    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], ModelRegistryConfig]] = None
    ):
        """
        Initialize the model registry.
        
        Args:
            config: Configuration dictionary or ModelRegistryConfig instance
        """
        # Parse configuration
        if config is None:
            config = ModelRegistryConfig()
        elif isinstance(config, dict):
            config = ModelRegistryConfig(**config)
        
        self.config = config
        
        # Initialize storage backend
        self._storage = self._create_storage()
        
        # Cache for frequently accessed models
        self._cache: Dict[str, RegisteredModel] = {}
        self._cache_lock = Lock()
        
        # Statistics
        self._stats = {
            'total_models': 0,
            'total_versions': 0,
            'total_artifacts': 0,
            'total_artifact_size_bytes': 0,
            'start_time': datetime.now()
        }
        
        # Update stats from existing data
        self._update_stats_from_storage()
        
        logger.info(f"ModelRegistry initialized with config: {config.dict()}")
    
    def _create_storage(self) -> StorageBackend:
        """Create appropriate storage backend."""
        backend = self.config.storage_backend.lower()
        
        if backend == "filesystem":
            return FileSystemStorage(self.config.local_storage_path)
        elif backend == "sqlite":
            from .storage.sqlite_storage import SQLiteStorage # type: ignore
            return SQLiteStorage(self.config.storage_uri)
        elif backend == "postgresql":
            from .storage.postgres_storage import PostgreSQLStorage # type: ignore
            return PostgreSQLStorage(self.config.storage_uri)
        elif backend == "mongodb":
            from .storage.mongodb_storage import MongoDBStorage # type: ignore
            return MongoDBStorage(self.config.storage_uri)
        else:
            logger.warning(f"Unknown storage backend: {backend}, using filesystem")
            return FileSystemStorage(self.config.local_storage_path)
    
    def _update_stats_from_storage(self) -> None:
        """Update statistics from existing storage data."""
        try:
            models = self._storage.list_models()
            
            total_versions = 0
            total_artifacts = 0
            total_artifact_size = 0
            
            for model in models:
                total_versions += len(model.versions)
                for version in model.versions:
                    total_artifacts += len(version.artifacts)
                    for artifact in version.artifacts:
                        total_artifact_size += artifact.size_bytes
            
            self._stats.update({
                'total_models': len(models),
                'total_versions': total_versions,
                'total_artifacts': total_artifacts,
                'total_artifact_size_bytes': total_artifact_size
            })
            
        except Exception as e:
            logger.error(f"Failed to update stats from storage: {e}")
    
    def _generate_model_id(self, name: str, namespace: Optional[str] = None) -> str:
        """Generate a unique model ID."""
        if namespace and self.config.enable_namespaces:
            model_id = f"{namespace}.{name}"
        else:
            model_id = name
        
        # Make URL-safe
        model_id = re.sub(r'[^a-zA-Z0-9_.-]', '_', model_id)
        return model_id
    
    def _generate_artifact_id(self, model_id: str, version: str, artifact_type: ArtifactType) -> str:
        """Generate a unique artifact ID."""
        base = f"{model_id}_{version}_{artifact_type.value}"
        return hashlib.sha256(base.encode()).hexdigest()[:32]
    
    def _calculate_checksum(self, file_path: Union[str, Path], algorithm: str = "sha256") -> str:
        """Calculate checksum of a file."""
        hash_func = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    def _validate_model_name(self, name: str) -> bool:
        """Validate model name."""
        if not name or len(name) > 100:
            return False
        
        # Allow alphanumeric, dots, dashes, underscores
        if not re.match(r'^[a-zA-Z0-9_.-]+$', name):
            return False
        
        return True
    
    def _validate_version(self, version: str, scheme: VersioningScheme) -> bool:
        """Validate version string against scheme."""
        try:
            if scheme == VersioningScheme.SEMVER:
                # Semantic versioning: major.minor.patch
                pattern = r'^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$'
                return bool(re.match(pattern, version))
            elif scheme == VersioningScheme.TIMESTAMP:
                # Timestamp: YYYYMMDD.HHMMSS
                pattern = r'^\d{8}\.\d{6}$'
                return bool(re.match(pattern, version))
            elif scheme == VersioningScheme.INCREMENTAL:
                # Incremental: integer
                return version.isdigit()
            elif scheme == VersioningScheme.GIT_HASH:
                # Git hash: 40-character hex or 7-character short hash
                pattern = r'^[a-fA-F0-9]{7,40}$'
                return bool(re.match(pattern, version))
            else:
                return True  # Custom scheme
        except:
            return False
    
    def _generate_next_version(
        self,
        model_id: str,
        scheme: VersioningScheme
    ) -> str:
        """Generate next version based on scheme and existing versions."""
        try:
            versions = self._storage.get_model_versions(model_id)
            
            if scheme == VersioningScheme.SEMVER:
                # Find highest semantic version
                max_version = "0.0.0"
                for v in versions:
                    if self._validate_version(v.version, scheme):
                        # Simple comparison (for production, use proper semver parsing)
                        if v.version > max_version:
                            max_version = v.version
                
                # Increment patch version
                parts = max_version.split('.')
                if len(parts) == 3:
                    patch = int(parts[2]) + 1
                    return f"{parts[0]}.{parts[1]}.{patch}"
                else:
                    return "1.0.0"
            
            elif scheme == VersioningScheme.TIMESTAMP:
                # Current timestamp
                now = datetime.now()
                return now.strftime("%Y%m%d.%H%M%S")
            
            elif scheme == VersioningScheme.INCREMENTAL:
                # Find highest integer version
                max_version = 0
                for v in versions:
                    if v.version.isdigit():
                        version_int = int(v.version)
                        if version_int > max_version:
                            max_version = version_int
                
                return str(max_version + 1)
            
            elif scheme == VersioningScheme.GIT_HASH:
                # This would typically come from Git, not generated here
                raise ValueError("Git hash versions must be provided, not generated")
            
            else:
                # Default to timestamp
                now = datetime.now()
                return now.strftime("%Y%m%d.%H%M%S")
                
        except Exception as e:
            logger.error(f"Failed to generate next version: {e}")
            # Fallback to timestamp
            now = datetime.now()
            return now.strftime("%Y%m%d.%H%M%S")
    
    def _check_access(
        self,
        model: RegisteredModel,
        user: Optional[str] = None,
        team: Optional[str] = None
    ) -> bool:
        """Check if user/team has access to model."""
        if not self.config.enable_access_control:
            return True
        
        # Public models are accessible to everyone
        if model.access_level == ModelAccessLevel.PUBLIC:
            return True
        
        # Check allowed users and teams
        if user and user in model.allowed_users:
            return True
        
        if team and team in model.allowed_teams:
            return True
        
        # Check if user is owner
        if user and user == model.owner:
            return True
        
        # Check if user's team matches model's team
        if team and team == model.team:
            return True
        
        # Internal models are accessible to authenticated users
        if (model.access_level == ModelAccessLevel.INTERNAL and 
            user and self.config.require_authentication):
            return True
        
        return False
    
    def register_model(
        self,
        name: str,
        model_type: Union[str, ModelType],
        namespace: Optional[str] = None,
        description: Optional[str] = None,
        owner: Optional[str] = None,
        team: Optional[str] = None,
        tags: Optional[List[str]] = None,
        access_level: Union[str, ModelAccessLevel] = ModelAccessLevel.PRIVATE,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RegisteredModel:
        """
        Register a new model in the registry.
        
        Args:
            name: Model name
            model_type: Type of model
            namespace: Optional namespace
            description: Model description
            owner: Model owner
            team: Owner team
            tags: Model tags
            access_level: Access level
            metadata: Additional metadata
            
        Returns:
            Registered model
            
        Raises:
            DuplicateModelError: If model already exists
            InvalidModelError: If model name is invalid
        """
        # Validate inputs
        if not self._validate_model_name(name):
            raise InvalidModelError(f"Invalid model name: {name}")
        
        if isinstance(model_type, str):
            model_type = ModelType(model_type)
        
        if isinstance(access_level, str):
            access_level = ModelAccessLevel(access_level)
        
        # Generate model ID
        model_id = self._generate_model_id(name, namespace)
        
        # Check if model already exists
        existing_model = self._storage.load_model(model_id)
        if existing_model:
            raise DuplicateModelError(f"Model already exists: {model_id}")
        
        # Create model
        model = RegisteredModel(
            model_id=model_id,
            name=name,
            description=description,
            owner=owner,
            team=team,
            tags=tags or [],
            primary_type=model_type,
            access_level=access_level,
            allowed_users=[owner] if owner else [],
            allowed_teams=[team] if team else [],
            metadata=metadata or {}
        )
        
        # Save to storage
        try:
            self._storage.save_model(model)
            
            # Update cache
            with self._cache_lock:
                self._cache[model_id] = model
            
            # Update stats
            self._stats['total_models'] += 1
            
            # Trigger webhook
            self._trigger_webhook("model_registered", {
                'model_id': model_id,
                'name': name,
                'owner': owner,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Registered model: {model_id}")
            return model
            
        except Exception as e:
            raise RegistryError(f"Failed to register model: {e}")
    
    def get_model(
        self,
        model_id: str,
        user: Optional[str] = None,
        team: Optional[str] = None
    ) -> Optional[RegisteredModel]:
        """
        Get a model from the registry.
        
        Args:
            model_id: Model identifier
            user: Optional user for access control
            team: Optional team for access control
            
        Returns:
            Registered model or None if not found/access denied
        """
        # Check cache first
        with self._cache_lock:
            if model_id in self._cache:
                model = self._cache[model_id]
                if self._check_access(model, user, team):
                    return model
                else:
                    raise AccessDeniedError(f"Access denied to model: {model_id}")
        
        # Load from storage
        model = self._storage.load_model(model_id)
        if not model:
            return None
        
        # Check access
        if not self._check_access(model, user, team):
            raise AccessDeniedError(f"Access denied to model: {model_id}")
        
        # Update cache
        with self._cache_lock:
            self._cache[model_id] = model
        
        return model
    
    def update_model(
        self,
        model_id: str,
        updates: Dict[str, Any],
        user: Optional[str] = None,
        team: Optional[str] = None
    ) -> Optional[RegisteredModel]:
        """
        Update model metadata.
        
        Args:
            model_id: Model identifier
            updates: Dictionary of updates
            user: Optional user for access control
            team: Optional team for access control
            
        Returns:
            Updated model or None if not found
        """
        # Get existing model
        model = self.get_model(model_id, user, team)
        if not model:
            return None
        
        # Check if user is owner or has permission
        if user and user != model.owner:
            # In a real implementation, you might check for admin rights
            raise AccessDeniedError(f"Only the owner can update model: {model_id}")
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(model, key):
                setattr(model, key, value)
        
        model.updated_at = datetime.now()
        
        # Save updated model
        try:
            self._storage.save_model(model)
            
            # Update cache
            with self._cache_lock:
                self._cache[model_id] = model
            
            # Trigger webhook
            self._trigger_webhook("model_updated", {
                'model_id': model_id,
                'updates': updates,
                'user': user,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Updated model: {model_id}")
            return model
            
        except Exception as e:
            raise RegistryError(f"Failed to update model: {e}")
    
    def delete_model(
        self,
        model_id: str,
        user: Optional[str] = None,
        team: Optional[str] = None
    ) -> bool:
        """
        Delete a model from the registry.
        
        Args:
            model_id: Model identifier
            user: Optional user for access control
            team: Optional team for access control
            
        Returns:
            True if deleted, False otherwise
        """
        # Get model first to check access
        model = self.get_model(model_id, user, team)
        if not model:
            return False
        
        # Check if user is owner or has permission
        if user and user != model.owner:
            raise AccessDeniedError(f"Only the owner can delete model: {model_id}")
        
        # Delete from storage
        try:
            success = self._storage.delete_model(model_id)
            
            if success:
                # Remove from cache
                with self._cache_lock:
                    self._cache.pop(model_id, None)
                
                # Update stats
                self._stats['total_models'] = max(0, self._stats['total_models'] - 1)
                
                # Trigger webhook
                self._trigger_webhook("model_deleted", {
                    'model_id': model_id,
                    'user': user,
                    'timestamp': datetime.now().isoformat()
                })
                
                logger.info(f"Deleted model: {model_id}")
            
            return success
            
        except Exception as e:
            raise RegistryError(f"Failed to delete model: {e}")
    
    def list_models(
        self,
        namespace: Optional[str] = None,
        tags: Optional[List[str]] = None,
        model_type: Optional[ModelType] = None,
        user: Optional[str] = None,
        team: Optional[str] = None
    ) -> List[RegisteredModel]:
        """
        List models with optional filtering.
        
        Args:
            namespace: Filter by namespace
            tags: Filter by tags
            model_type: Filter by model type
            user: Optional user for access control
            team: Optional team for access control
            
        Returns:
            List of models
        """
        # Get all models from storage
        models = self._storage.list_models(namespace, tags, model_type)
        
        # Apply access control
        if self.config.enable_access_control:
            filtered_models = []
            for model in models:
                if self._check_access(model, user, team):
                    filtered_models.append(model)
            return filtered_models
        else:
            return models
    
    def search_models(
        self,
        query: str,
        namespace: Optional[str] = None,
        user: Optional[str] = None,
        team: Optional[str] = None
    ) -> List[RegisteredModel]:
        """
        Search models by name, description, or tags.
        
        Args:
            query: Search query
            namespace: Optional namespace filter
            user: Optional user for access control
            team: Optional team for access control
            
        Returns:
            List of matching models
        """
        # Search in storage
        models = self._storage.search_models(query, namespace)
        
        # Apply access control
        if self.config.enable_access_control:
            filtered_models = []
            for model in models:
                if self._check_access(model, user, team):
                    filtered_models.append(model)
            return filtered_models
        else:
            return models
    
    def register_version(
        self,
        model_id: str,
        version: Optional[str] = None,
        version_scheme: Union[str, VersioningScheme] = VersioningScheme.SEMVER,
        stage: Union[str, ModelStage] = ModelStage.NONE,
        description: Optional[str] = None,
        created_by: Optional[str] = None,
        model_type: Optional[Union[str, ModelType]] = None,
        framework: Optional[Union[str, ModelFramework]] = None,
        framework_version: Optional[str] = None,
        parameters: Optional[int] = None,
        source_type: Optional[Union[str, ModelSourceType]] = None,
        source_uri: Optional[str] = None,
        source_commit: Optional[str] = None,
        parent_version: Optional[str] = None,
        base_model: Optional[str] = None,
        evaluation_metrics: Optional[Dict[str, float]] = None,
        training_metrics: Optional[Dict[str, float]] = None,
        dependencies: Optional[List[ModelDependency]] = None,
        compatible_devices: Optional[List[Union[str, ModelDevice]]] = None,
        memory_requirements_mb: Optional[float] = None,
        inference_time_ms: Optional[float] = None,
        tags: Optional[List[str]] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
        user: Optional[str] = None,
        team: Optional[str] = None
    ) -> ModelVersion:
        """
        Register a new version of a model.
        
        Args:
            model_id: Model identifier
            version: Version string (auto-generated if None)
            version_scheme: Versioning scheme
            stage: Model stage
            description: Version description
            created_by: User who created this version
            model_type: Type of model
            framework: Model framework
            framework_version: Framework version
            parameters: Number of parameters
            source_type: Source type
            source_uri: Source URI
            source_commit: Source commit hash
            parent_version: Parent version
            base_model: Base model if fine-tuned
            evaluation_metrics: Evaluation metrics
            training_metrics: Training metrics
            dependencies: Model dependencies
            compatible_devices: Compatible devices
            memory_requirements_mb: Memory requirements in MB
            inference_time_ms: Average inference time in ms
            tags: Version tags
            custom_metadata: Custom metadata
            user: Optional user for access control
            team: Optional team for access control
            
        Returns:
            Registered model version
            
        Raises:
            ModelNotFoundError: If model doesn't exist
            InvalidModelError: If version is invalid
        """
        # Get model first to check access
        model = self.get_model(model_id, user, team)
        if not model:
            raise ModelNotFoundError(f"Model not found: {model_id}")
        
        # Parse enum parameters
        if isinstance(version_scheme, str):
            version_scheme = VersioningScheme(version_scheme)
        
        if isinstance(stage, str):
            stage = ModelStage(stage)
        
        if isinstance(model_type, str):
            model_type = ModelType(model_type)
        
        if isinstance(framework, str):
            framework = ModelFramework(framework)
        
        if isinstance(source_type, str):
            source_type = ModelSourceType(source_type)
        
        # Convert compatible devices
        device_list = []
        if compatible_devices:
            for device in compatible_devices:
                if isinstance(device, str):
                    device_list.append(ModelDevice(device))
                else:
                    device_list.append(device)
        
        # Generate version if not provided
        if version is None:
            if self.config.auto_versioning:
                version = self._generate_next_version(model_id, version_scheme)
            else:
                raise InvalidModelError("Version is required when auto_versioning is disabled")
        
        # Validate version
        if not self._validate_version(version, version_scheme):
            raise InvalidModelError(f"Invalid version '{version}' for scheme {version_scheme.value}")
        
        # Check if version already exists
        existing_version = self._storage.get_version(model_id, version)
        if existing_version:
            raise DuplicateModelError(f"Version already exists: {model_id}:{version}")
        
        # Create model version
        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            version_scheme=version_scheme,
            stage=stage,
            description=description,
            tags=tags or [],
            created_by=created_by,
            model_type=model_type or model.primary_type,
            framework=framework,
            framework_version=framework_version,
            parameters=parameters,
            source_type=source_type,
            source_uri=source_uri,
            source_commit=source_commit,
            parent_version=parent_version,
            base_model=base_model,
            evaluation_metrics=evaluation_metrics or {},
            training_metrics=training_metrics or {},
            dependencies=dependencies or [],
            compatible_devices=device_list,
            memory_requirements_mb=memory_requirements_mb,
            inference_time_ms=inference_time_ms,
            custom_metadata=custom_metadata or {}
        )
        
        # Add version to model
        model.versions.append(model_version)
        model.updated_at = datetime.now()
        
        # Update latest version
        if not model.latest_version or model_version.created_at > self._storage.get_version(model_id, model.latest_version).created_at:
            model.latest_version = version
        
        # Update latest production version
        if stage == ModelStage.PRODUCTION:
            model.latest_production_version = version
        
        # Save updated model
        try:
            self._storage.save_model(model)
            
            # Update cache
            with self._cache_lock:
                self._cache[model_id] = model
            
            # Update stats
            self._stats['total_versions'] += 1
            
            # Trigger webhook
            self._trigger_webhook("version_registered", {
                'model_id': model_id,
                'version': version,
                'stage': stage.value,
                'created_by': created_by,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Registered version: {model_id}:{version}")
            return model_version
            
        except Exception as e:
            raise RegistryError(f"Failed to register version: {e}")
    
    def add_artifact(
        self,
        model_id: str,
        version: str,
        artifact_type: Union[str, ArtifactType],
        file_path: Union[str, Path],
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user: Optional[str] = None,
        team: Optional[str] = None
    ) -> ModelArtifact:
        """
        Add an artifact to a model version.
        
        Args:
            model_id: Model identifier
            version: Version string
            artifact_type: Type of artifact
            file_path: Path to artifact file
            description: Artifact description
            metadata: Additional metadata
            user: Optional user for access control
            team: Optional team for access control
            
        Returns:
            Registered artifact
            
        Raises:
            ModelNotFoundError: If model/version doesn't exist
            InvalidModelError: If file is invalid
        """
        # Get model first to check access
        model = self.get_model(model_id, user, team)
        if not model:
            raise ModelNotFoundError(f"Model not found: {model_id}")
        
        # Find version
        model_version = None
        for v in model.versions:
            if v.version == version:
                model_version = v
                break
        
        if not model_version:
            raise VersionNotFoundError(f"Version not found: {model_id}:{version}")
        
        # Parse artifact type
        if isinstance(artifact_type, str):
            artifact_type = ArtifactType(artifact_type)
        
        # Validate file
        file_path = Path(file_path)
        if not file_path.exists():
            raise InvalidModelError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise InvalidModelError(f"Not a file: {file_path}")
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > self.config.max_model_size_mb * 1024 * 1024:
            raise InvalidModelError(
                f"File too large: {file_size / (1024*1024):.1f}MB > "
                f"{self.config.max_model_size_mb}MB limit"
            )
        
        # Calculate checksum
        if self.config.validate_checksums:
            checksum = self._calculate_checksum(file_path)
            checksum_algorithm = "sha256"
        else:
            checksum = ""
            checksum_algorithm = ""
        
        # Generate artifact ID
        artifact_id = self._generate_artifact_id(model_id, version, artifact_type)
        
        # Create artifact
        artifact = ModelArtifact(
            artifact_id=artifact_id,
            model_id=model_id,
            version=version,
            artifact_type=artifact_type,
            path=str(file_path),
            size_bytes=file_size,
            checksum=checksum,
            checksum_algorithm=checksum_algorithm,
            description=description,
            metadata=metadata or {}
        )
        
        # Add artifact to version
        model_version.artifacts.append(artifact)
        model_version.updated_at = datetime.now()
        
        # Update version size
        model_version.size_bytes = sum(a.size_bytes for a in model_version.artifacts)
        
        # Save artifact data
        try:
            with open(file_path, 'rb') as f:
                storage_path = self._storage.save_artifact(artifact, f)
                artifact.path = storage_path  # Update path to storage location
            
            # Save updated model
            self._storage.save_model(model)
            
            # Update cache
            with self._cache_lock:
                self._cache[model_id] = model
            
            # Update stats
            self._stats['total_artifacts'] += 1
            self._stats['total_artifact_size_bytes'] += file_size
            
            # Trigger webhook
            self._trigger_webhook("artifact_added", {
                'model_id': model_id,
                'version': version,
                'artifact_id': artifact_id,
                'artifact_type': artifact_type.value,
                'size_bytes': file_size,
                'user': user,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Added artifact: {artifact_id} for {model_id}:{version}")
            return artifact
            
        except Exception as e:
            raise RegistryError(f"Failed to add artifact: {e}")
    
    def get_artifact(
        self,
        artifact_id: str,
        user: Optional[str] = None,
        team: Optional[str] = None
    ) -> Optional[Tuple[ModelArtifact, Optional[BinaryIO]]]:
        """
        Get an artifact from storage.
        
        Args:
            artifact_id: Artifact identifier
            user: Optional user for access control
            team: Optional team for access control
            
        Returns:
            Tuple of (artifact, file_handle) or None if not found
        """
        # Load artifact from storage
        result = self._storage.load_artifact(artifact_id)
        if not result:
            return None
        
        artifact, file_handle = result
        
        # Check access to the model
        try:
            model = self.get_model(artifact.model_id, user, team)
            if not model:
                return None
        except AccessDeniedError:
            return None
        
        return (artifact, file_handle)
    
    def transition_stage(
        self,
        model_id: str,
        version: str,
        new_stage: Union[str, ModelStage],
        user: Optional[str] = None,
        team: Optional[str] = None
    ) -> bool:
        """
        Transition a model version to a new stage.
        
        Args:
            model_id: Model identifier
            version: Version string
            new_stage: New stage
            user: Optional user for access control
            team: Optional team for access control
            
        Returns:
            True if transitioned, False otherwise
        """
        # Get model first to check access
        model = self.get_model(model_id, user, team)
        if not model:
            return False
        
        # Parse stage
        if isinstance(new_stage, str):
            new_stage = ModelStage(new_stage)
        
        # Find version
        for model_version in model.versions:
            if model_version.version == version:
                # Update stage
                old_stage = model_version.stage
                model_version.stage = new_stage
                model_version.updated_at = datetime.now()
                
                # Update model's latest production version if needed
                if new_stage == ModelStage.PRODUCTION:
                    model.latest_production_version = version
                
                # Save updated model
                try:
                    self._storage.save_model(model)
                    
                    # Update cache
                    with self._cache_lock:
                        self._cache[model_id] = model
                    
                    # Trigger webhook
                    self._trigger_webhook("stage_transitioned", {
                        'model_id': model_id,
                        'version': version,
                        'old_stage': old_stage.value,
                        'new_stage': new_stage.value,
                        'user': user,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    logger.info(f"Transitioned {model_id}:{version} from {old_stage.value} to {new_stage.value}")
                    return True
                    
                except Exception as e:
                    logger.error(f"Failed to transition stage: {e}")
                    return False
        
        return False
    
    def get_version(
        self,
        model_id: str,
        version: str,
        user: Optional[str] = None,
        team: Optional[str] = None
    ) -> Optional[ModelVersion]:
        """
        Get a specific version of a model.
        
        Args:
            model_id: Model identifier
            version: Version string
            user: Optional user for access control
            team: Optional team for access control
            
        Returns:
            Model version or None if not found
        """
        # Get model first to check access
        model = self.get_model(model_id, user, team)
        if not model:
            return None
        
        # Find version
        for model_version in model.versions:
            if model_version.version == version:
                return model_version
        
        return None
    
    def get_model_versions(
        self,
        model_id: str,
        stage: Optional[Union[str, ModelStage]] = None,
        user: Optional[str] = None,
        team: Optional[str] = None
    ) -> List[ModelVersion]:
        """
        Get versions of a model.
        
        Args:
            model_id: Model identifier
            stage: Optional stage filter
            user: Optional user for access control
            team: Optional team for access control
            
        Returns:
            List of model versions
        """
        # Get model first to check access
        model = self.get_model(model_id, user, team)
        if not model:
            return []
        
        # Parse stage
        stage_enum = None
        if stage is not None:
            if isinstance(stage, str):
                stage_enum = ModelStage(stage)
            else:
                stage_enum = stage
        
        # Filter by stage if specified
        if stage_enum:
            return [v for v in model.versions if v.stage == stage_enum]
        else:
            return model.versions
    
    def _trigger_webhook(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Trigger webhook for registry events."""
        if not self.config.enable_webhooks or not self.config.webhook_urls:
            return
        
        # This would be implemented with async HTTP requests in production
        # For now, just log the event
        logger.info(f"Webhook event: {event_type} - {payload}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        uptime = datetime.now() - self._stats['start_time']
        
        stats = self._stats.copy()
        stats.update({
            'uptime_seconds': uptime.total_seconds(),
            'average_versions_per_model': (
                self._stats['total_versions'] / max(self._stats['total_models'], 1)
            ),
            'average_artifacts_per_version': (
                self._stats['total_artifacts'] / max(self._stats['total_versions'], 1)
            ),
            'total_artifact_size_gb': self._stats['total_artifact_size_bytes'] / (1024**3),
            'config': self.config.dict()
        })
        
        return stats
    
    def export_model(
        self,
        model_id: str,
        version: str,
        export_format: str = "directory",
        export_path: Optional[Path] = None,
        user: Optional[str] = None,
        team: Optional[str] = None
    ) -> Path:
        """
        Export a model version to a specific format.
        
        Args:
            model_id: Model identifier
            version: Version string
            export_format: Export format ("directory", "tar", "zip")
            export_path: Optional export path
            user: Optional user for access control
            team: Optional team for access control
            
        Returns:
            Path to exported model
            
        Raises:
            ModelNotFoundError: If model/version doesn't exist
        """
        # Get model version
        model_version = self.get_version(model_id, version, user, team)
        if not model_version:
            raise ModelNotFoundError(f"Model version not found: {model_id}:{version}")
        
        # Create export directory
        if export_path is None:
            export_dir = Path(tempfile.mkdtemp(prefix=f"export_{model_id}_{version}_"))
        else:
            export_dir = Path(export_path)
            export_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata file
        metadata = {
            'model_id': model_id,
            'version': version,
            'exported_at': datetime.now().isoformat(),
            'model_metadata': model_version.to_dict()
        }
        
        metadata_path = export_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Export artifacts
        artifacts_dir = export_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        for artifact in model_version.artifacts:
            result = self.get_artifact(artifact.artifact_id, user, team)
            if result:
                artifact_obj, file_handle = result
                if file_handle:
                    artifact_path = artifacts_dir / f"{artifact_obj.artifact_id}.bin"
                    with open(artifact_path, 'wb') as f:
                        shutil.copyfileobj(file_handle, f)
                    file_handle.close()
        
        # Package if requested
        if export_format == "tar":
            import tarfile
            tar_path = export_dir.parent / f"{model_id}_{version}.tar"
            with tarfile.open(tar_path, 'w') as tar:
                tar.add(export_dir, arcname=f"{model_id}_{version}")
            shutil.rmtree(export_dir)
            return tar_path
        
        elif export_format == "zip":
            import zipfile
            zip_path = export_dir.parent / f"{model_id}_{version}.zip"
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for root, dirs, files in os.walk(export_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, export_dir)
                        zipf.write(file_path, arcname=f"{model_id}_{version}/{arcname}")
            shutil.rmtree(export_dir)
            return zip_path
        
        else:  # directory format
            return export_dir
    
    def cleanup_old_versions(
        self,
        retention_days: Optional[int] = None,
        keep_min_versions: int = 5
    ) -> int:
        """
        Cleanup old model versions.
        
        Args:
            retention_days: Days to retain versions (default from config)
            keep_min_versions: Minimum number of versions to keep per model
            
        Returns:
            Number of versions cleaned up
        """
        if retention_days is None:
            retention_days = self.config.retention_days
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        cleaned_count = 0
        
        # Get all models
        models = self._storage.list_models()
        
        for model in models:
            # Sort versions by creation date (oldest first)
            old_versions = sorted(
                [v for v in model.versions if v.created_at < cutoff_date],
                key=lambda x: x.created_at
            )
            
            # Keep at least keep_min_versions
            if len(old_versions) > keep_min_versions:
                versions_to_remove = old_versions[:-keep_min_versions]
            else:
                versions_to_remove = []
            
            # Remove old versions
            for version in versions_to_remove:
                try:
                    # Remove artifacts
                    for artifact in version.artifacts:
                        self._storage.delete_artifact(artifact.artifact_id)
                    
                    # Remove version from model
                    model.versions = [v for v in model.versions if v.version != version.version]
                    
                    # Update latest version pointers
                    if model.latest_version == version.version:
                        if model.versions:
                            latest = max(model.versions, key=lambda x: x.created_at)
                            model.latest_version = latest.version
                        else:
                            model.latest_version = None
                    
                    if model.latest_production_version == version.version:
                        production_versions = [v for v in model.versions if v.stage == ModelStage.PRODUCTION]
                        if production_versions:
                            latest_prod = max(production_versions, key=lambda x: x.created_at)
                            model.latest_production_version = latest_prod.version
                        else:
                            model.latest_production_version = None
                    
                    cleaned_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to cleanup version {model.model_id}:{version.version}: {e}")
            
            # Save updated model if changes were made
            if versions_to_remove:
                try:
                    self._storage.save_model(model)
                    with self._cache_lock:
                        self._cache.pop(model.model_id, None)
                except Exception as e:
                    logger.error(f"Failed to save model after cleanup: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old model versions")
        
        return cleaned_count
    
    def shutdown(self) -> None:
        """Shutdown the model registry."""
        logger.info("Shutting down ModelRegistry")
        
        # Clear cache
        with self._cache_lock:
            self._cache.clear()
        
        logger.info("ModelRegistry shutdown complete")


# Singleton instance for global access
_global_registry: Optional[ModelRegistry] = None


def get_model_registry(
    config: Optional[Union[Dict[str, Any], ModelRegistryConfig]] = None
) -> ModelRegistry:
    """
    Get or create global model registry instance.
    
    Args:
        config: Configuration for the model registry
        
    Returns:
        Global ModelRegistry instance
    """
    global _global_registry
    
    if _global_registry is None:
        _global_registry = ModelRegistry(config)
    
    return _global_registry


def shutdown_model_registry() -> None:
    """Shutdown the global model registry."""
    global _global_registry
    
    if _global_registry:
        _global_registry.shutdown()
        _global_registry = None