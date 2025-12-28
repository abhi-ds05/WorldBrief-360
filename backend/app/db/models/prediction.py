"""
prediction.py - Prediction and Analytics Model

This module defines models for machine learning predictions, analytics,
forecasting, and risk assessment across the platform. This includes:
- Incident predictions (likelihood, severity, location)
- Risk assessments and threat modeling
- Trend analysis and forecasting
- Anomaly detection
- Pattern recognition
- Model performance tracking
- Feature engineering pipelines

Key Features:
- Multi-model support (regression, classification, clustering)
- Real-time and batch predictions
- Model versioning and A/B testing
- Feature store integration
- Prediction explainability (SHAP, LIME)
- Confidence scoring and uncertainty quantification
- Performance monitoring and drift detection
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, TYPE_CHECKING, Union
from enum import Enum
import numpy as np
from sqlalchemy import (
    Column, String, Text, ForeignKey, Integer, DateTime,BigInteger, 
    Boolean, Enum as SQLEnum, JSON, Float, CheckConstraint,
    Index, Table, UniqueConstraint, LargeBinary, ARRAY
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY as PG_ARRAY
from sqlalchemy.sql import func, expression
from sqlalchemy.ext.hybrid import hybrid_property

from db.base import Base
from models.mixins import TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from models.user import User
    from models.incident import Incident
    from models.article import Article
    from models.dataset import Dataset


class ModelType(Enum):
    """Types of machine learning models."""
    # Classification models
    BINARY_CLASSIFICATION = "binary_classification"      # Two-class classification
    MULTICLASS_CLASSIFICATION = "multiclass_classification"  # Multi-class classification
    MULTILABEL_CLASSIFICATION = "multilabel_classification"  # Multi-label classification
    
    # Regression models
    LINEAR_REGRESSION = "linear_regression"              # Linear regression
    LOGISTIC_REGRESSION = "logistic_regression"          # Logistic regression
    RIDGE_REGRESSION = "ridge_regression"                # Ridge regression
    LASSO_REGRESSION = "lasso_regression"                # Lasso regression
    
    # Time series models
    ARIMA = "arima"                                      # ARIMA models
    SARIMA = "sarima"                                    # Seasonal ARIMA
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"      # Exponential smoothing
    PROPHET = "prophet"                                  # Facebook Prophet
    LSTM = "lstm"                                        # LSTM networks
    
    # Clustering models
    KMEANS = "kmeans"                                    # K-means clustering
    DBSCAN = "dbscan"                                    # DBSCAN clustering
    HIERARCHICAL_CLUSTERING = "hierarchical_clustering"  # Hierarchical clustering
    GAUSSIAN_MIXTURE = "gaussian_mixture"                # Gaussian mixture models
    
    # Anomaly detection
    ISOLATION_FOREST = "isolation_forest"                # Isolation Forest
    ONE_CLASS_SVM = "one_class_svm"                      # One-class SVM
    AUTOENCODER = "autoencoder"                          # Autoencoder-based
    
    # Ensemble models
    RANDOM_FOREST = "random_forest"                      # Random Forest
    GRADIENT_BOOSTING = "gradient_boosting"              # Gradient Boosting
    XGBOOST = "xgboost"                                  # XGBoost
    LIGHTGBM = "lightgbm"                                # LightGBM
    CATBOOST = "catboost"                                # CatBoost
    
    # Deep learning models
    CNN = "cnn"                                          # Convolutional Neural Network
    RNN = "rnn"                                          # Recurrent Neural Network
    TRANSFORMER = "transformer"                          # Transformer models
    GAN = "gan"                                          # Generative Adversarial Network
    
    # Other models
    BAYESIAN = "bayesian"                                # Bayesian models
    DECISION_TREE = "decision_tree"                      # Decision Tree
    SVM = "svm"                                          # Support Vector Machine
    KNN = "knn"                                          # K-Nearest Neighbors
    NAIVE_BAYES = "naive_bayes"                          # Naive Bayes
    CUSTOM = "custom"                                    # Custom/unknown model


class ModelStatus(Enum):
    """Status of machine learning models."""
    DEVELOPING = "developing"            # Under development
    TRAINING = "training"                # Currently training
    VALIDATING = "validating"            # Validation phase
    TESTING = "testing"                  # Testing phase
    ACTIVE = "active"                    # Active and deployed
    STAGING = "staging"                  # Staging for deployment
    INACTIVE = "inactive"                # Inactive but saved
    ARCHIVED = "archived"                # Archived
    FAILED = "failed"                    # Failed training/deployment
    RETIRED = "retired"                  # Retired from service


class PredictionType(Enum):
    """Types of predictions."""
    INCIDENT_LIKELIHOOD = "incident_likelihood"          # Likelihood of incident
    INCIDENT_SEVERITY = "incident_severity"              # Severity prediction
    INCIDENT_LOCATION = "incident_location"              # Location prediction
    INCIDENT_TYPE = "incident_type"                      # Type prediction
    RISK_SCORE = "risk_score"                            # Risk score calculation
    TREND_FORECAST = "trend_forecast"                    # Trend forecasting
    ANOMALY_DETECTION = "anomaly_detection"              # Anomaly detection
    PATTERN_RECOGNITION = "pattern_recognition"          # Pattern recognition
    CLUSTER_ASSIGNMENT = "cluster_assignment"            # Cluster assignment
    SENTIMENT_ANALYSIS = "sentiment_analysis"            # Sentiment analysis
    TOPIC_MODELING = "topic_modeling"                    # Topic modeling
    ENTITY_EXTRACTION = "entity_extraction"              # Entity extraction
    IMAGE_CLASSIFICATION = "image_classification"        # Image classification
    TEXT_CLASSIFICATION = "text_classification"          # Text classification
    TIME_SERIES_FORECAST = "time_series_forecast"        # Time series forecast
    RECOMMENDATION = "recommendation"                    # Recommendation
    OTHER = "other"                                      # Other prediction types


class PredictionStatus(Enum):
    """Status of predictions."""
    PENDING = "pending"                  # Prediction pending
    PROCESSING = "processing"            # Being processed
    COMPLETED = "completed"              # Successfully completed
    FAILED = "failed"                    # Failed
    CANCELLED = "cancelled"              # Cancelled
    EXPIRED = "expired"                  # Expired
    VALIDATING = "validating"            # Being validated
    CONFIRMED = "confirmed"              # Confirmed by human
    REJECTED = "rejected"                # Rejected by human
    OUTDATED = "outdated"                # Outdated prediction


class ModelFramework(Enum):
    """Machine learning frameworks."""
    SCIKIT_LEARN = "scikit_learn"        # Scikit-learn
    TENSORFLOW = "tensorflow"            # TensorFlow
    PYTORCH = "pytorch"                  # PyTorch
    KERAS = "keras"                      # Keras
    XGBOOST_LIB = "xgboost_lib"          # XGBoost
    LIGHTGBM_LIB = "lightgbm_lib"        # LightGBM
    CATBOOST_LIB = "catboost_lib"        # CatBoost
    PROPHET_LIB = "prophet_lib"          # Prophet
    SPARK_ML = "spark_ml"                # Spark ML
    H2O = "h2o"                          # H2O.ai
    CUSTOM = "custom"                    # Custom framework


class FeatureType(Enum):
    """Types of features in feature engineering."""
    NUMERICAL = "numerical"              # Numerical features
    CATEGORICAL = "categorical"          # Categorical features
    TEXT = "text"                        # Text features
    DATETIME = "datetime"                # DateTime features
    GEOGRAPHICAL = "geographical"        # Geographical features
    IMAGE = "image"                      # Image features
    AUDIO = "audio"                      # Audio features
    VIDEO = "video"                      # Video features
    EMBEDDING = "embedding"              # Embedding vectors
    DERIVED = "derived"                  # Derived features
    AGGREGATED = "aggregated"            # Aggregated features


class Model(Base, UUIDMixin, TimestampMixin):
    """
    Machine learning model metadata and configuration.
    
    This model stores metadata about trained machine learning models,
    including configuration, performance metrics, and deployment status.
    
    Attributes:
        id: Primary key UUID
        name: Model name
        version: Model version
        model_type: Type of ML model
        model_framework: Framework used
        status: Current model status
        description: Model description
        target_variable: Variable being predicted
        feature_names: List of feature names
        hyperparameters: Model hyperparameters
        training_config: Training configuration
        performance_metrics: Performance metrics
        training_data_info: Information about training data
        validation_data_info: Information about validation data
        test_data_info: Information about test data
        model_size_bytes: Size of model file in bytes
        model_format: Format of saved model
        model_path: Path to model file
        model_hash: Hash of model file
        created_by: User who created the model
        last_trained_at: When model was last trained
        deployment_config: Deployment configuration
        is_production: Whether model is in production
        parent_model_id: Parent model for versioning
        tags: Categorization tags
        metadata: Additional metadata
    """
    
    __tablename__ = "ml_models"
    
    # Basic information
    name = Column(String(200), nullable=False, index=True)
    version = Column(String(50), nullable=False, default="1.0.0")
    model_type = Column(SQLEnum(ModelType), nullable=False, index=True)
    model_framework = Column(SQLEnum(ModelFramework), nullable=True, index=True)
    status = Column(SQLEnum(ModelStatus), default=ModelStatus.DEVELOPING, nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Prediction target
    target_variable = Column(String(200), nullable=True)
    feature_names = Column(PG_ARRAY(String), default=[], nullable=False)
    
    # Model configuration
    hyperparameters = Column(JSONB, default=dict, nullable=False)
    training_config = Column(JSONB, default=dict, nullable=False)
    performance_metrics = Column(JSONB, default=dict, nullable=False)
    
    # Data information
    training_data_info = Column(JSONB, default=dict, nullable=False)
    validation_data_info = Column(JSONB, default=dict, nullable=False)
    test_data_info = Column(JSONB, default=dict, nullable=False)
    
    # Model storage
    model_size_bytes = Column(BigInteger, nullable=True)
    model_format = Column(String(50), nullable=True)
    model_path = Column(String(2000), nullable=True)
    model_hash = Column(String(128), nullable=True, index=True)
    
    # Model artifacts (serialized model)
    model_artifact = Column(LargeBinary, nullable=True)
    
    # Ownership and tracking
    created_by = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    last_trained_at = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # Deployment
    deployment_config = Column(JSONB, default=dict, nullable=False)
    is_production = Column(Boolean, default=False, nullable=False, index=True)
    deployed_at = Column(DateTime(timezone=True), nullable=True)
    retired_at = Column(DateTime(timezone=True), nullable=True)
    
    # Versioning
    parent_model_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("ml_models.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    
    # Metadata
    tags = Column(PG_ARRAY(String), default=[], nullable=False, index=True)
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    creator = relationship("User", foreign_keys=[created_by])
    parent_model = relationship("Model", remote_side=[id], backref="child_models")
    predictions = relationship("Prediction", back_populates="model")
    model_versions = relationship("ModelVersion", back_populates="model")
    feature_sets = relationship("FeatureSet", secondary="model_features", back_populates="models")
    
    # Check constraints
    __table_args__ = (
        UniqueConstraint('name', 'version', name='uq_model_name_version'),
        CheckConstraint('model_size_bytes IS NULL OR model_size_bytes > 0', name='check_model_size_positive'),
        CheckConstraint('version ~* \'^\\d+\\.\\d+\\.\\d+$\'', name='check_version_format'),
        Index('ix_models_production_status', 'is_production', 'status'),
        Index('ix_models_type_framework', 'model_type', 'model_framework'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Model(id={self.id}, name={self.name}, version={self.version}, type={self.model_type.value})>"
    
    @property
    def accuracy(self) -> Optional[float]:
        """Get model accuracy if available."""
        return self.performance_metrics.get('accuracy')
    
    @property
    def precision(self) -> Optional[float]:
        """Get model precision if available."""
        return self.performance_metrics.get('precision')
    
    @property
    def recall(self) -> Optional[float]:
        """Get model recall if available."""
        return self.performance_metrics.get('recall')
    
    @property
    def f1_score(self) -> Optional[float]:
        """Get model F1 score if available."""
        return self.performance_metrics.get('f1_score')
    
    @property
    def mse(self) -> Optional[float]:
        """Get model mean squared error if available."""
        return self.performance_metrics.get('mse')
    
    @property
    def mae(self) -> Optional[float]:
        """Get model mean absolute error if available."""
        return self.performance_metrics.get('mae')
    
    @property
    def r2_score(self) -> Optional[float]:
        """Get model R² score if available."""
        return self.performance_metrics.get('r2_score')
    
    @property
    def auc_roc(self) -> Optional[float]:
        """Get model AUC-ROC score if available."""
        return self.performance_metrics.get('auc_roc')
    
    @property
    def is_active(self) -> bool:
        """Check if model is active."""
        return self.status == ModelStatus.ACTIVE
    
    @property
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self.last_trained_at is not None
    
    @property
    def training_duration_seconds(self) -> Optional[float]:
        """Get training duration in seconds."""
        if self.last_trained_at and self.created_at:
            return (self.last_trained_at - self.created_at).total_seconds()
        return None
    
    @property
    def num_features(self) -> int:
        """Get number of features."""
        return len(self.feature_names) if self.feature_names else 0
    
    def deploy_to_production(self) -> None:
        """Deploy model to production."""
        self.is_production = True
        self.status = ModelStatus.ACTIVE
        self.deployed_at = datetime.utcnow()
    
    def retire_from_production(self) -> None:
        """Retire model from production."""
        self.is_production = False
        self.status = ModelStatus.RETIRED
        self.retired_at = datetime.utcnow()
    
    def update_performance(self, metrics: Dict[str, Any]) -> None:
        """Update performance metrics."""
        self.performance_metrics.update(metrics)
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the model."""
        tag_lower = tag.strip().lower()
        if tag_lower and tag_lower not in self.tags:
            self.tags = self.tags + [tag_lower]
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the model."""
        tag_lower = tag.strip().lower()
        if tag_lower in self.tags:
            self.tags = [t for t in self.tags if t != tag_lower]
    
    def to_dict(self, include_artifacts: bool = False) -> Dict[str, Any]:
        """
        Convert model to dictionary.
        
        Args:
            include_artifacts: Whether to include model artifacts
            
        Returns:
            Dictionary representation
        """
        result = {
            "id": str(self.id),
            "name": self.name,
            "version": self.version,
            "model_type": self.model_type.value,
            "model_framework": self.model_framework.value if self.model_framework else None,
            "status": self.status.value,
            "description": self.description,
            "target_variable": self.target_variable,
            "feature_names": self.feature_names,
            "num_features": self.num_features,
            "hyperparameters": self.hyperparameters,
            "training_config": self.training_config,
            "performance_metrics": self.performance_metrics,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "mse": self.mse,
            "mae": self.mae,
            "r2_score": self.r2_score,
            "auc_roc": self.auc_roc,
            "training_data_info": self.training_data_info,
            "validation_data_info": self.validation_data_info,
            "test_data_info": self.test_data_info,
            "model_size_bytes": self.model_size_bytes,
            "model_format": self.model_format,
            "model_path": self.model_path,
            "model_hash": self.model_hash,
            "created_by": str(self.created_by) if self.created_by else None,
            "last_trained_at": self.last_trained_at.isoformat() if self.last_trained_at else None,
            "deployment_config": self.deployment_config,
            "is_production": self.is_production,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "retired_at": self.retired_at.isoformat() if self.retired_at else None,
            "parent_model_id": str(self.parent_model_id) if self.parent_model_id else None,
            "tags": self.tags,
            "metadata": self.metadata,
            "is_active": self.is_active,
            "is_trained": self.is_trained,
            "training_duration_seconds": round(self.training_duration_seconds, 2) if self.training_duration_seconds else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_artifacts and self.model_artifact:
            result["model_artifact"] = self.model_artifact.hex()
        
        if self.creator:
            result["creator"] = {
                "id": str(self.creator.id),
                "username": self.creator.username,
                "email": getattr(self.creator, 'email', None)
            }
        
        return result
    
    @classmethod
    def create(
        cls,
        name: str,
        model_type: ModelType,
        created_by: Optional[uuid.UUID] = None,
        version: str = "1.0.0",
        model_framework: Optional[ModelFramework] = None,
        description: Optional[str] = None,
        target_variable: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        parent_model_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> 'Model':
        """
        Factory method to create a new model.
        
        Args:
            name: Model name
            model_type: Type of ML model
            created_by: User who created the model
            version: Model version
            model_framework: Framework used
            description: Model description
            target_variable: Variable being predicted
            feature_names: List of feature names
            hyperparameters: Model hyperparameters
            training_config: Training configuration
            parent_model_id: Parent model for versioning
            tags: Categorization tags
            metadata: Additional metadata
            **kwargs: Additional arguments
            
        Returns:
            A new Model instance
        """
        model = cls(
            name=name.strip(),
            model_type=model_type,
            version=version,
            model_framework=model_framework,
            description=description,
            target_variable=target_variable,
            feature_names=feature_names or [],
            hyperparameters=hyperparameters or {},
            training_config=training_config or {},
            performance_metrics={},
            training_data_info={},
            validation_data_info={},
            test_data_info={},
            deployment_config={},
            created_by=created_by,
            parent_model_id=parent_model_id,
            tags=tags or [],
            metadata=metadata or {},
            status=ModelStatus.DEVELOPING,
            **kwargs
        )
        
        return model


class Prediction(Base, UUIDMixin, TimestampMixin):
    """
    Prediction instance model.
    
    This model stores individual predictions made by machine learning models,
    including input features, output results, and confidence scores.
    
    Attributes:
        id: Primary key UUID
        model_id: Model used for prediction
        prediction_type: Type of prediction
        status: Prediction status
        input_features: Input features for prediction
        output_result: Prediction output
        confidence_score: Confidence score (0-1)
        confidence_interval: Confidence interval
        explanation: Explanation of prediction
        metadata: Additional metadata
        requested_by: User who requested prediction
        processing_time_ms: Time taken to process
        related_incident_id: Related incident
        related_article_id: Related article
        dataset_id: Dataset used
        expires_at: When prediction expires
        validated_by: User who validated
        validated_at: When validated
        validation_status: Validation status
        feedback: User feedback on prediction
        tags: Categorization tags
    """
    
    __tablename__ = "predictions"
    
    # Model reference
    model_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("ml_models.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # Prediction type and status
    prediction_type = Column(SQLEnum(PredictionType), nullable=False, index=True)
    status = Column(SQLEnum(PredictionStatus), default=PredictionStatus.PENDING, nullable=False, index=True)
    
    # Input and output
    input_features = Column(JSONB, default=dict, nullable=False)
    output_result = Column(JSONB, default=dict, nullable=False)
    
    # Confidence and uncertainty
    confidence_score = Column(Float, nullable=True, index=True)
    confidence_interval_lower = Column(Float, nullable=True)
    confidence_interval_upper = Column(Float, nullable=True)
    
    # Explainability
    explanation = Column(JSONB, nullable=True)
    feature_importance = Column(JSONB, nullable=True)
    shap_values = Column(JSONB, nullable=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Request tracking
    requested_by = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    processing_time_ms = Column(Integer, nullable=True)
    
    # Related entities
    related_incident_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("incidents.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    related_article_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("articles.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    dataset_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("datasets.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    
    # Expiration
    expires_at = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # Validation
    validated_by = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    validated_at = Column(DateTime(timezone=True), nullable=True)
    validation_status = Column(String(50), nullable=True, index=True)
    validation_notes = Column(Text, nullable=True)
    
    # Feedback
    feedback = Column(JSONB, nullable=True)
    feedback_score = Column(Integer, nullable=True, index=True)
    feedback_notes = Column(Text, nullable=True)
    
    # Categorization
    tags = Column(PG_ARRAY(String), default=[], nullable=False, index=True)
    
    # Relationships
    model = relationship("Model", back_populates="predictions")
    requester = relationship("User", foreign_keys=[requested_by])
    validator = relationship("User", foreign_keys=[validated_by])
    related_incident = relationship("Incident")
    related_article = relationship("Article")
    dataset = relationship("Dataset")
    
    # Check constraints
    __table_args__ = (
        CheckConstraint(
            'confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1)',
            name='check_confidence_score_range'
        ),
        CheckConstraint(
            'feedback_score IS NULL OR (feedback_score >= 0 AND feedback_score <= 5)',
            name='check_feedback_score_range'
        ),
        CheckConstraint(
            'processing_time_ms IS NULL OR processing_time_ms >= 0',
            name='check_processing_time_non_negative'
        ),
        Index('ix_predictions_model_status', 'model_id', 'status'),
        Index('ix_predictions_type_confidence', 'prediction_type', 'confidence_score'),
        Index('ix_predictions_created_expires', 'created_at', 'expires_at'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Prediction(id={self.id}, type={self.prediction_type.value}, model={self.model_id})>"
    
    @property
    def predicted_value(self) -> Any:
        """Get the predicted value."""
        return self.output_result.get('prediction')
    
    @property
    def predicted_probabilities(self) -> Optional[Dict[str, float]]:
        """Get predicted probabilities for classification."""
        return self.output_result.get('probabilities')
    
    @property
    def predicted_class(self) -> Optional[str]:
        """Get predicted class for classification."""
        return self.output_result.get('predicted_class')
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if prediction has high confidence."""
        return self.confidence_score is not None and self.confidence_score >= 0.8
    
    @property
    def is_medium_confidence(self) -> bool:
        """Check if prediction has medium confidence."""
        return self.confidence_score is not None and 0.5 <= self.confidence_score < 0.8
    
    @property
    def is_low_confidence(self) -> bool:
        """Check if prediction has low confidence."""
        return self.confidence_score is not None and self.confidence_score < 0.5
    
    @property
    def is_completed(self) -> bool:
        """Check if prediction is completed."""
        return self.status == PredictionStatus.COMPLETED
    
    @property
    def is_failed(self) -> bool:
        """Check if prediction failed."""
        return self.status == PredictionStatus.FAILED
    
    @property
    def is_expired(self) -> bool:
        """Check if prediction has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    @property
    def is_validated(self) -> bool:
        """Check if prediction has been validated."""
        return self.validated_at is not None and self.validation_status is not None
    
    @property
    def confidence_interval_width(self) -> Optional[float]:
        """Get width of confidence interval."""
        if self.confidence_interval_lower is not None and self.confidence_interval_upper is not None:
            return self.confidence_interval_upper - self.confidence_interval_lower
        return None
    
    @property
    def feature_names(self) -> List[str]:
        """Get feature names from input features."""
        return list(self.input_features.keys()) if self.input_features else []
    
    @property
    def top_features(self, n: int = 5) -> Optional[List[Dict[str, Any]]]:
        """Get top N important features."""
        if not self.feature_importance:
            return None
        
        # Sort features by importance
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:n]
        
        return [
            {
                "feature": feature,
                "importance": importance,
                "value": self.input_features.get(feature)
            }
            for feature, importance in sorted_features
        ]
    
    def mark_as_completed(
        self,
        output_result: Dict[str, Any],
        confidence_score: Optional[float] = None,
        processing_time_ms: Optional[int] = None,
        explanation: Optional[Dict[str, Any]] = None,
        feature_importance: Optional[Dict[str, float]] = None
    ) -> None:
        """Mark prediction as completed with results."""
        self.status = PredictionStatus.COMPLETED
        self.output_result = output_result
        self.confidence_score = confidence_score
        self.processing_time_ms = processing_time_ms
        self.explanation = explanation
        self.feature_importance = feature_importance
        
        # Set default expiration if not set
        if self.expires_at is None:
            self.expires_at = datetime.utcnow() + timedelta(days=7)
    
    def mark_as_failed(self, error_message: str, error_code: Optional[str] = None) -> None:
        """Mark prediction as failed."""
        self.status = PredictionStatus.FAILED
        self.metadata["error"] = error_message
        if error_code:
            self.metadata["error_code"] = error_code
    
    def validate(
        self,
        validated_by: uuid.UUID,
        validation_status: str,
        validation_notes: Optional[str] = None,
        feedback_score: Optional[int] = None,
        feedback_notes: Optional[str] = None
    ) -> None:
        """Validate prediction with user feedback."""
        self.validated_by = validated_by
        self.validated_at = datetime.utcnow()
        self.validation_status = validation_status
        self.validation_notes = validation_notes
        self.feedback_score = feedback_score
        self.feedback_notes = feedback_notes
        
        # Update feedback in metadata
        self.feedback = {
            "validated_by": str(validated_by),
            "validated_at": self.validated_at.isoformat(),
            "validation_status": validation_status,
            "validation_notes": validation_notes,
            "feedback_score": feedback_score,
            "feedback_notes": feedback_notes
        }
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the prediction."""
        tag_lower = tag.strip().lower()
        if tag_lower and tag_lower not in self.tags:
            self.tags = self.tags + [tag_lower]
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the prediction."""
        tag_lower = tag.strip().lower()
        if tag_lower in self.tags:
            self.tags = [t for t in self.tags if t != tag_lower]
    
    def to_dict(self, include_features: bool = True, include_explanation: bool = False) -> Dict[str, Any]:
        """
        Convert prediction to dictionary.
        
        Args:
            include_features: Whether to include input features
            include_explanation: Whether to include explanation
            
        Returns:
            Dictionary representation
        """
        result = {
            "id": str(self.id),
            "model_id": str(self.model_id),
            "prediction_type": self.prediction_type.value,
            "status": self.status.value,
            "predicted_value": self.predicted_value,
            "predicted_class": self.predicted_class,
            "predicted_probabilities": self.predicted_probabilities,
            "confidence_score": self.confidence_score,
            "confidence_interval_lower": self.confidence_interval_lower,
            "confidence_interval_upper": self.confidence_interval_upper,
            "confidence_interval_width": self.confidence_interval_width,
            "is_high_confidence": self.is_high_confidence,
            "is_medium_confidence": self.is_medium_confidence,
            "is_low_confidence": self.is_low_confidence,
            "is_completed": self.is_completed,
            "is_failed": self.is_failed,
            "is_expired": self.is_expired,
            "is_validated": self.is_validated,
            "processing_time_ms": self.processing_time_ms,
            "requested_by": str(self.requested_by) if self.requested_by else None,
            "related_incident_id": str(self.related_incident_id) if self.related_incident_id else None,
            "related_article_id": str(self.related_article_id) if self.related_article_id else None,
            "dataset_id": str(self.dataset_id) if self.dataset_id else None,
            "validated_by": str(self.validated_by) if self.validated_by else None,
            "validated_at": self.validated_at.isoformat() if self.validated_at else None,
            "validation_status": self.validation_status,
            "validation_notes": self.validation_notes,
            "feedback_score": self.feedback_score,
            "feedback_notes": self.feedback_notes,
            "tags": self.tags,
            "metadata": self.metadata,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_features:
            result["input_features"] = self.input_features
            result["feature_names"] = self.feature_names
        
        if include_explanation:
            result["explanation"] = self.explanation
            result["feature_importance"] = self.feature_importance
            result["shap_values"] = self.shap_values
            result["top_features"] = self.top_features(5) if self.top_features else None
        
        if self.model:
            result["model"] = {
                "id": str(self.model.id),
                "name": self.model.name,
                "version": self.model.version,
                "model_type": self.model.model_type.value
            }
        
        if self.requester:
            result["requester"] = {
                "id": str(self.requester.id),
                "username": self.requester.username
            }
        
        if self.validator:
            result["validator"] = {
                "id": str(self.validator.id),
                "username": self.validator.username
            }
        
        return result
    
    @classmethod
    def create_incident_likelihood(
        cls,
        model_id: uuid.UUID,
        input_features: Dict[str, Any],
        requested_by: Optional[uuid.UUID] = None,
        related_incident_id: Optional[uuid.UUID] = None,
        confidence_threshold: float = 0.7,
        **kwargs
    ) -> 'Prediction':
        """
        Create incident likelihood prediction.
        
        Args:
            model_id: Model ID
            input_features: Input features
            requested_by: User who requested
            related_incident_id: Related incident ID
            confidence_threshold: Confidence threshold
            **kwargs: Additional arguments
            
        Returns:
            Prediction instance
        """
        return cls.create(
            model_id=model_id,
            prediction_type=PredictionType.INCIDENT_LIKELIHOOD,
            input_features=input_features,
            requested_by=requested_by,
            related_incident_id=related_incident_id,
            metadata={
                "confidence_threshold": confidence_threshold,
                "prediction_category": "incident_risk"
            },
            tags=["incident", "likelihood", "risk"],
            **kwargs
        )
    
    @classmethod
    def create_risk_score(
        cls,
        model_id: uuid.UUID,
        input_features: Dict[str, Any],
        requested_by: Optional[uuid.UUID] = None,
        location: Optional[str] = None,
        **kwargs
    ) -> 'Prediction':
        """
        Create risk score prediction.
        
        Args:
            model_id: Model ID
            input_features: Input features
            requested_by: User who requested
            location: Location for risk assessment
            **kwargs: Additional arguments
            
        Returns:
            Prediction instance
        """
        metadata = kwargs.get('metadata', {})
        metadata.update({
            "location": location,
            "prediction_category": "risk_assessment"
        })
        
        return cls.create(
            model_id=model_id,
            prediction_type=PredictionType.RISK_SCORE,
            input_features=input_features,
            requested_by=requested_by,
            metadata=metadata,
            tags=["risk", "assessment", "score"],
            **kwargs
        )
    
    @classmethod
    def create_anomaly_detection(
        cls,
        model_id: uuid.UUID,
        input_features: Dict[str, Any],
        requested_by: Optional[uuid.UUID] = None,
        dataset_id: Optional[uuid.UUID] = None,
        anomaly_threshold: float = 0.9,
        **kwargs
    ) -> 'Prediction':
        """
        Create anomaly detection prediction.
        
        Args:
            model_id: Model ID
            input_features: Input features
            requested_by: User who requested
            dataset_id: Dataset ID
            anomaly_threshold: Anomaly threshold
            **kwargs: Additional arguments
            
        Returns:
            Prediction instance
        """
        return cls.create(
            model_id=model_id,
            prediction_type=PredictionType.ANOMALY_DETECTION,
            input_features=input_features,
            requested_by=requested_by,
            dataset_id=dataset_id,
            metadata={
                "anomaly_threshold": anomaly_threshold,
                "prediction_category": "anomaly"
            },
            tags=["anomaly", "detection", "outlier"],
            **kwargs
        )
    
    @classmethod
    def create(
        cls,
        model_id: uuid.UUID,
        prediction_type: PredictionType,
        input_features: Dict[str, Any],
        requested_by: Optional[uuid.UUID] = None,
        related_incident_id: Optional[uuid.UUID] = None,
        related_article_id: Optional[uuid.UUID] = None,
        dataset_id: Optional[uuid.UUID] = None,
        confidence_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> 'Prediction':
        """
        Factory method to create a new prediction.
        
        Args:
            model_id: Model ID
            prediction_type: Type of prediction
            input_features: Input features
            requested_by: User who requested prediction
            related_incident_id: Related incident ID
            related_article_id: Related article ID
            dataset_id: Dataset ID
            confidence_score: Initial confidence score
            metadata: Additional metadata
            tags: Categorization tags
            **kwargs: Additional arguments
            
        Returns:
            A new Prediction instance
        """
        prediction = cls(
            model_id=model_id,
            prediction_type=prediction_type,
            input_features=input_features,
            output_result={},
            confidence_score=confidence_score,
            requested_by=requested_by,
            related_incident_id=related_incident_id,
            related_article_id=related_article_id,
            dataset_id=dataset_id,
            metadata=metadata or {},
            tags=tags or [],
            status=PredictionStatus.PENDING,
            **kwargs
        )
        
        return prediction


class FeatureSet(Base, UUIDMixin, TimestampMixin):
    """
    Feature set model for feature engineering and management.
    
    This model stores feature definitions, transformations, and metadata
    for machine learning feature engineering pipelines.
    
    Attributes:
        id: Primary key UUID
        name: Feature set name
        version: Feature set version
        description: Feature set description
        feature_definitions: Feature definitions
        feature_schema: Schema for features
        transformations: Feature transformations
        data_source: Source of data
        sample_data: Sample feature data
        statistics: Feature statistics
        created_by: User who created feature set
        is_active: Whether feature set is active
        tags: Categorization tags
        metadata: Additional metadata
    """
    
    __tablename__ = "feature_sets"
    
    # Basic information
    name = Column(String(200), nullable=False, index=True)
    version = Column(String(50), nullable=False, default="1.0.0")
    description = Column(Text, nullable=True)
    
    # Feature definitions
    feature_definitions = Column(JSONB, default=dict, nullable=False)
    feature_schema = Column(JSONB, default=dict, nullable=False)
    
    # Transformations
    transformations = Column(JSONB, default=dict, nullable=False)
    
    # Data source
    data_source = Column(JSONB, default=dict, nullable=False)
    sample_data = Column(JSONB, nullable=True)
    
    # Statistics
    statistics = Column(JSONB, default=dict, nullable=False)
    
    # Ownership and status
    created_by = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    
    # Metadata
    tags = Column(PG_ARRAY(String), default=[], nullable=False, index=True)
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    creator = relationship("User", foreign_keys=[created_by])
    models = relationship("Model", secondary="model_features", back_populates="feature_sets")
    
    # Check constraints
    __table_args__ = (
        UniqueConstraint('name', 'version', name='uq_feature_set_name_version'),
        CheckConstraint('version ~* \'^\\d+\\.\\d+\\.\\d+$\'', name='check_version_format'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<FeatureSet(id={self.id}, name={self.name}, version={self.version})>"
    
    @property
    def num_features(self) -> int:
        """Get number of features."""
        return len(self.feature_definitions) if self.feature_definitions else 0
    
    @property
    def feature_names(self) -> List[str]:
        """Get feature names."""
        return list(self.feature_definitions.keys()) if self.feature_definitions else []
    
    @property
    def feature_types(self) -> Dict[str, str]:
        """Get feature types."""
        return {
            name: definition.get('type', 'unknown')
            for name, definition in self.feature_definitions.items()
        }
    
    @property
    def categorical_features(self) -> List[str]:
        """Get categorical feature names."""
        return [
            name for name, definition in self.feature_definitions.items()
            if definition.get('type') == FeatureType.CATEGORICAL.value
        ]
    
    @property
    def numerical_features(self) -> List[str]:
        """Get numerical feature names."""
        return [
            name for name, definition in self.feature_definitions.items()
            if definition.get('type') == FeatureType.NUMERICAL.value
        ]
    
    @property
    def text_features(self) -> List[str]:
        """Get text feature names."""
        return [
            name for name, definition in self.feature_definitions.items()
            if definition.get('type') == FeatureType.TEXT.value
        ]
    
    def get_feature_definition(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """Get definition for a specific feature."""
        return self.feature_definitions.get(feature_name)
    
    def add_feature(
        self,
        name: str,
        feature_type: FeatureType,
        description: Optional[str] = None,
        data_type: Optional[str] = None,
        allowed_values: Optional[List[Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        default_value: Optional[Any] = None
    ) -> None:
        """Add a feature definition."""
        feature_def = {
            "type": feature_type.value,
            "description": description,
            "data_type": data_type,
            "allowed_values": allowed_values,
            "constraints": constraints,
            "default_value": default_value,
            "created_at": datetime.utcnow().isoformat()
        }
        
        self.feature_definitions[name] = feature_def
    
    def remove_feature(self, feature_name: str) -> None:
        """Remove a feature definition."""
        if feature_name in self.feature_definitions:
            del self.feature_definitions[feature_name]
    
    def update_statistics(self, statistics: Dict[str, Any]) -> None:
        """Update feature statistics."""
        self.statistics.update(statistics)
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the feature set."""
        tag_lower = tag.strip().lower()
        if tag_lower and tag_lower not in self.tags:
            self.tags = self.tags + [tag_lower]
    
    def to_dict(self, include_samples: bool = True) -> Dict[str, Any]:
        """Convert feature set to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "num_features": self.num_features,
            "feature_names": self.feature_names,
            "feature_types": self.feature_types,
            "categorical_features": self.categorical_features,
            "numerical_features": self.numerical_features,
            "text_features": self.text_features,
            "feature_definitions": self.feature_definitions,
            "feature_schema": self.feature_schema,
            "transformations": self.transformations,
            "data_source": self.data_source,
            "sample_data": self.sample_data if include_samples else None,
            "statistics": self.statistics,
            "created_by": str(self.created_by) if self.created_by else None,
            "is_active": self.is_active,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


# Association table for model-feature relationships
model_features = Table(
    'model_features',
    Base.metadata,
    Column('model_id', UUID(as_uuid=True), ForeignKey('ml_models.id'), primary_key=True),
    Column('feature_set_id', UUID(as_uuid=True), ForeignKey('feature_sets.id'), primary_key=True),
    Column('created_at', DateTime(timezone=True), server_default=func.now()),
    Index('ix_model_features_model', 'model_id'),
    Index('ix_model_features_feature_set', 'feature_set_id')
)


class ModelVersion(Base, UUIDMixin, TimestampMixin):
    """
    Model versioning and A/B testing.
    
    This model tracks different versions of models for versioning,
    A/B testing, and gradual rollouts.
    """
    
    __tablename__ = "model_versions"
    
    model_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("ml_models.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    version = Column(String(50), nullable=False)
    weight = Column(Float, default=1.0, nullable=False)  # For A/B testing
    
    # Performance tracking
    performance_metrics = Column(JSONB, default=dict, nullable=False)
    prediction_count = Column(Integer, default=0, nullable=False)
    success_count = Column(Integer, default=0, nullable=False)
    failure_count = Column(Integer, default=0, nullable=False)
    
    # Deployment
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    activated_at = Column(DateTime(timezone=True), nullable=True)
    deactivated_at = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    release_notes = Column(Text, nullable=True)
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    model = relationship("Model", back_populates="model_versions")
    
    # Check constraints
    __table_args__ = (
        UniqueConstraint('model_id', 'version', name='uq_model_version'),
        CheckConstraint('weight >= 0 AND weight <= 1', name='check_weight_range'),
        CheckConstraint('prediction_count >= 0', name='check_prediction_count_non_negative'),
        CheckConstraint('success_count >= 0', name='check_success_count_non_negative'),
        CheckConstraint('failure_count >= 0', name='check_failure_count_non_negative'),
    )
    
    @property
    def success_rate(self) -> float:
        """Get success rate of predictions."""
        if self.prediction_count == 0:
            return 0.0
        return self.success_count / self.prediction_count
    
    @property
    def failure_rate(self) -> float:
        """Get failure rate of predictions."""
        if self.prediction_count == 0:
            return 0.0
        return self.failure_count / self.prediction_count
    
    def record_prediction(self, success: bool = True) -> None:
        """Record a prediction for this model version."""
        self.prediction_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
    
    def activate(self) -> None:
        """Activate this model version."""
        self.is_active = True
        self.activated_at = datetime.utcnow()
    
    def deactivate(self) -> None:
        """Deactivate this model version."""
        self.is_active = False
        self.deactivated_at = datetime.utcnow()


class PredictionBatch(Base, UUIDMixin, TimestampMixin):
    """
    Batch prediction job model.
    
    This model tracks batch prediction jobs for processing
    multiple predictions at once.
    """
    
    __tablename__ = "prediction_batches"
    
    name = Column(String(200), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Batch configuration
    model_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("ml_models.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    dataset_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("datasets.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    
    # Status tracking
    status = Column(String(50), default="pending", nullable=False, index=True)
    total_items = Column(Integer, nullable=True)
    processed_items = Column(Integer, default=0, nullable=False)
    failed_items = Column(Integer, default=0, nullable=False)
    
    # Results
    result_path = Column(String(2000), nullable=True)
    result_format = Column(String(50), nullable=True)
    
    # Processing info
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    
    # Metadata
    created_by = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    tags = Column(PG_ARRAY(String), default=[], nullable=False, index=True)
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    model = relationship("Model")
    dataset = relationship("Dataset")
    creator = relationship("User", foreign_keys=[created_by])
    
    # Check constraints
    __table_args__ = (
        CheckConstraint('total_items IS NULL OR total_items >= 0', name='check_total_items_non_negative'),
        CheckConstraint('processed_items >= 0', name='check_processed_items_non_negative'),
        CheckConstraint('failed_items >= 0', name='check_failed_items_non_negative'),
        CheckConstraint('processing_time_ms IS NULL OR processing_time_ms >= 0', name='check_processing_time_non_negative'),
    )
    
    @property
    def progress_percentage(self) -> float:
        """Get progress percentage."""
        if not self.total_items:
            return 0.0
        return (self.processed_items / self.total_items) * 100
    
    @property
    def success_rate(self) -> float:
        """Get success rate."""
        if self.processed_items == 0:
            return 0.0
        return ((self.processed_items - self.failed_items) / self.processed_items) * 100
    
    @property
    def is_completed(self) -> bool:
        """Check if batch is completed."""
        return self.status == "completed"
    
    @property
    def is_failed(self) -> bool:
        """Check if batch failed."""
        return self.status == "failed"
    
    @property
    def is_processing(self) -> bool:
        """Check if batch is processing."""
        return self.status == "processing"


class ModelMonitoring(Base, UUIDMixin, TimestampMixin):
    """
    Model monitoring and drift detection.
    
    This model tracks model performance over time and detects
    concept drift, data drift, and performance degradation.
    """
    
    __tablename__ = "model_monitoring"
    
    model_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("ml_models.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # Drift detection
    data_drift_score = Column(Float, nullable=True)
    concept_drift_score = Column(Float, nullable=True)
    performance_drift_score = Column(Float, nullable=True)
    
    # Statistical tests
    statistical_tests = Column(JSONB, default=dict, nullable=False)
    
    # Alert thresholds
    alert_thresholds = Column(JSONB, default=dict, nullable=False)
    
    # Monitoring period
    monitoring_window_start = Column(DateTime(timezone=True), nullable=False)
    monitoring_window_end = Column(DateTime(timezone=True), nullable=False)
    
    # Alerts
    alerts_triggered = Column(Integer, default=0, nullable=False)
    last_alert_at = Column(DateTime(timezone=True), nullable=True)
    
    # Status
    status = Column(String(50), default="monitoring", nullable=False, index=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    model = relationship("Model")
    
    # Check constraints
    __table_args__ = (
        CheckConstraint('data_drift_score IS NULL OR (data_drift_score >= 0 AND data_drift_score <= 1)', name='check_data_drift_score_range'),
        CheckConstraint('concept_drift_score IS NULL OR (concept_drift_score >= 0 AND concept_drift_score <= 1)', name='check_concept_drift_score_range'),
        CheckConstraint('performance_drift_score IS NULL OR (performance_drift_score >= 0 AND performance_drift_score <= 1)', name='check_performance_drift_score_range'),
        CheckConstraint('alerts_triggered >= 0', name='check_alerts_triggered_non_negative'),
    )
    
    @property
    def has_drift(self) -> bool:
        """Check if any drift detected."""
        thresholds = self.alert_thresholds
        data_threshold = thresholds.get('data_drift_threshold', 0.3)
        concept_threshold = thresholds.get('concept_drift_threshold', 0.3)
        performance_threshold = thresholds.get('performance_drift_threshold', 0.3)
        
        return (
            (self.data_drift_score is not None and self.data_drift_score > data_threshold) or
            (self.concept_drift_score is not None and self.concept_drift_score > concept_threshold) or
            (self.performance_drift_score is not None and self.performance_drift_score > performance_threshold)
        )
    
    @property
    def monitoring_duration_days(self) -> float:
        """Get monitoring duration in days."""
        delta = self.monitoring_window_end - self.monitoring_window_start
        return delta.total_seconds() / (24 * 3600)


# Helper functions
def calculate_confidence_interval(
    mean: float,
    std_dev: float,
    n: int,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Calculate confidence interval.
    
    Args:
        mean: Mean value
        std_dev: Standard deviation
        n: Sample size
        confidence_level: Confidence level (default: 0.95)
        
    Returns:
        Dictionary with lower and upper bounds
    """
    from scipy import stats
    
    # Calculate z-score for confidence level
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    # Calculate margin of error
    margin_of_error = z_score * (std_dev / (n ** 0.5))
    
    return {
        "lower": mean - margin_of_error,
        "upper": mean + margin_of_error,
        "margin_of_error": margin_of_error
    }


def validate_features_against_schema(
    features: Dict[str, Any],
    feature_schema: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate features against schema.
    
    Args:
        features: Input features
        feature_schema: Feature schema
        
    Returns:
        Validation result with errors if any
    """
    errors = []
    
    for feature_name, feature_def in feature_schema.items():
        if feature_name not in features:
            if feature_def.get('required', True):
                errors.append(f"Missing required feature: {feature_name}")
            continue
        
        value = features[feature_name]
        data_type = feature_def.get('data_type')
        
        # Type validation
        if data_type == 'integer' and not isinstance(value, int):
            errors.append(f"Feature {feature_name} should be integer, got {type(value)}")
        elif data_type == 'float' and not isinstance(value, (int, float)):
            errors.append(f"Feature {feature_name} should be float, got {type(value)}")
        elif data_type == 'string' and not isinstance(value, str):
            errors.append(f"Feature {feature_name} should be string, got {type(value)}")
        elif data_type == 'boolean' and not isinstance(value, bool):
            errors.append(f"Feature {feature_name} should be boolean, got {type(value)}")
        
        # Value constraints
        allowed_values = feature_def.get('allowed_values')
        if allowed_values and value not in allowed_values:
            errors.append(f"Feature {feature_name} value {value} not in allowed values: {allowed_values}")
        
        # Range constraints
        min_value = feature_def.get('min_value')
        max_value = feature_def.get('max_value')
        if min_value is not None and value < min_value:
            errors.append(f"Feature {feature_name} value {value} below minimum {min_value}")
        if max_value is not None and value > max_value:
            errors.append(f"Feature {feature_name} value {value} above maximum {max_value}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "num_features_provided": len(features),
        "num_features_expected": len(feature_schema)
    }