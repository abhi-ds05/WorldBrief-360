"""
Feature evaluation context.
Provides structured context for feature flag evaluation including user, system, and request information.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from enum import Enum
import uuid
import json


class ContextSource(str, Enum):
    """Sources of context information."""
    REQUEST = "request"        # From HTTP request
    USER = "user"             # From user profile
    SYSTEM = "system"         # From system state
    SESSION = "session"       # From user session
    GEO = "geo"               # From geolocation
    DEVICE = "device"         # From device info


class UserTier(str, Enum):
    """User subscription tiers."""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class DeviceType(str, Enum):
    """Device types."""
    DESKTOP = "desktop"
    MOBILE = "mobile"
    TABLET = "tablet"
    TV = "tv"
    BOT = "bot"


@dataclass
class GeoContext:
    """Geographic context information."""
    
    country: Optional[str] = None
    region: Optional[str] = None
    city: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timezone: Optional[str] = None
    continent: Optional[str] = None
    locale: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GeoContext':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DeviceContext:
    """Device and browser context."""
    
    device_type: Optional[DeviceType] = None
    browser: Optional[str] = None
    browser_version: Optional[str] = None
    os: Optional[str] = None
    os_version: Optional[str] = None
    screen_width: Optional[int] = None
    screen_height: Optional[int] = None
    is_mobile: Optional[bool] = None
    is_tablet: Optional[bool] = None
    is_desktop: Optional[bool] = None
    user_agent: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        if self.device_type:
            result["device_type"] = self.device_type.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeviceContext':
        """Create from dictionary."""
        if "device_type" in data and data["device_type"]:
            data["device_type"] = DeviceType(data["device_type"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SessionContext:
    """User session information."""
    
    session_id: Optional[str] = None
    session_start: Optional[datetime] = None
    session_duration: Optional[int] = None  # in seconds
    page_views: int = 0
    last_activity: Optional[datetime] = None
    is_new_session: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        # Convert datetime to ISO string
        for field in ["session_start", "last_activity"]:
            if value := getattr(self, field):
                result[field] = value.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionContext':
        """Create from dictionary."""
        # Convert ISO strings back to datetime
        for field in ["session_start", "last_activity"]:
            if field in data and data[field]:
                data[field] = datetime.fromisoformat(data[field].replace('Z', '+00:00'))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class UserContext:
    """User-specific context information."""
    
    # Basic user info
    user_id: Optional[str] = None
    username: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    tier: Optional[UserTier] = None
    
    # User attributes
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    is_verified: bool = False
    is_active: bool = True
    is_staff: bool = False
    is_superuser: bool = False
    
    # User engagement metrics
    engagement_score: float = 0.0
    days_active: int = 0
    days_since_registration: Optional[int] = None
    
    # Usage metrics
    incidents_reported: int = 0
    incidents_verified: int = 0
    briefings_viewed: int = 0
    briefings_generated: int = 0
    chat_sessions: int = 0
    coins_earned: int = 0
    coins_spent: int = 0
    
    # User preferences
    language: str = "en"
    theme: str = "light"
    notification_enabled: bool = True
    
    # Segments (calculated)
    segments: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        # Handle enums
        if self.tier:
            result["tier"] = self.tier.value
        
        # Convert datetime to ISO string
        for field in ["created_at", "last_login"]:
            if value := getattr(self, field):
                result[field] = value.isoformat()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserContext':
        """Create from dictionary."""
        # Convert ISO strings back to datetime
        for field in ["created_at", "last_login"]:
            if field in data and data[field]:
                data[field] = datetime.fromisoformat(data[field].replace('Z', '+00:00'))
        
        # Convert enums
        if "tier" in data and data["tier"]:
            data["tier"] = UserTier(data["tier"])
        
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def get_days_since_registration(self) -> Optional[int]:
        """Calculate days since registration."""
        if not self.created_at:
            return None
        
        delta = datetime.utcnow() - self.created_at
        return delta.days
    
    def calculate_engagement_score(self) -> float:
        """Calculate engagement score based on various factors."""
        score = 0.0
        
        # Base score for being active
        if self.is_active:
            score += 10.0
        
        # Days active bonus
        if self.days_active > 30:
            score += min(self.days_active / 10, 50)  # Max 50 points
        
        # Content creation
        score += self.incidents_reported * 5
        score += self.incidents_verified * 10
        
        # Consumption
        score += min(self.briefings_viewed / 10, 20)  # Max 20 points
        score += min(self.chat_sessions / 5, 15)      # Max 15 points
        
        # Financial engagement
        score += min((self.coins_earned + self.coins_spent) / 100, 25)
        
        # Verification status
        if self.is_verified:
            score += 15
        
        # Staff/superuser bonus
        if self.is_staff:
            score += 20
        if self.is_superuser:
            score += 30
        
        return min(score, 100.0)  # Cap at 100


@dataclass
class SystemContext:
    """System and environmental context."""
    
    # Environment info
    environment: str = "production"
    deployment_id: Optional[str] = None
    version: Optional[str] = None
    region: Optional[str] = None
    availability_zone: Optional[str] = None
    
    # System metrics
    system_load: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    request_rate: float = 0.0  # requests per second
    
    # Feature-specific metrics
    active_users: int = 0
    concurrent_sessions: int = 0
    error_rate: float = 0.0
    response_time_p95: float = 0.0
    
    # Time-based context
    current_time: datetime = field(default_factory=datetime.utcnow)
    is_business_hours: bool = False
    day_of_week: int = 0  # 0 = Monday
    hour_of_day: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["current_time"] = self.current_time.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemContext':
        """Create from dictionary."""
        if "current_time" in data and data["current_time"]:
            data["current_time"] = datetime.fromisoformat(data["current_time"].replace('Z', '+00:00'))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def update_time_context(self):
        """Update time-based context fields."""
        now = datetime.utcnow()
        self.current_time = now
        self.day_of_week = now.weekday()
        self.hour_of_day = now.hour
        
        # Business hours: 9 AM to 5 PM UTC, Monday to Friday
        self.is_business_hours = (
            self.day_of_week < 5 and  # Monday to Friday
            9 <= self.hour_of_day < 17
        )


@dataclass
class RequestContext:
    """HTTP request context."""
    
    request_id: Optional[str] = None
    method: Optional[str] = None
    path: Optional[str] = None
    endpoint: Optional[str] = None
    query_params: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    
    # Client info
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    referer: Optional[str] = None
    
    # Request timing
    request_start: Optional[datetime] = None
    latency: Optional[float] = None  # in milliseconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        if self.request_start:
            result["request_start"] = self.request_start.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RequestContext':
        """Create from dictionary."""
        if "request_start" in data and data["request_start"]:
            data["request_start"] = datetime.fromisoformat(data["request_start"].replace('Z', '+00:00'))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def generate_request_id(self):
        """Generate a unique request ID."""
        self.request_id = f"req_{uuid.uuid4().hex[:16]}"
    
    def calculate_latency(self):
        """Calculate request latency if start time is set."""
        if self.request_start:
            self.latency = (datetime.utcnow() - self.request_start).total_seconds() * 1000


@dataclass
class FeatureContext:
    """
    Complete feature evaluation context.
    Aggregates all context sources for feature flag evaluation.
    """
    
    # Context sources
    user: Optional[UserContext] = None
    system: Optional[SystemContext] = None
    request: Optional[RequestContext] = None
    geo: Optional[GeoContext] = None
    device: Optional[DeviceContext] = None
    session: Optional[SessionContext] = None
    
    # Custom context (for extensibility)
    custom: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: ContextSource = ContextSource.REQUEST
    evaluation_id: Optional[str] = None
    
    def __post_init__(self):
        """Initialize context."""
        if not self.evaluation_id:
            self.evaluation_id = f"eval_{uuid.uuid4().hex[:16]}"
        
        # Update system time context if available
        if self.system:
            self.system.update_time_context()
        
        # Update user engagement score if user context exists
        if self.user:
            self.user.engagement_score = self.user.calculate_engagement_score()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Flatten context to a single dictionary for feature evaluation.
        
        Returns:
            Flattened dictionary with all context information
        """
        result = {}
        
        # Add metadata
        result["timestamp"] = self.timestamp.isoformat()
        result["source"] = self.source.value
        result["evaluation_id"] = self.evaluation_id
        
        # Flatten user context
        if self.user:
            user_dict = self.user.to_dict()
            # Prefix user fields to avoid collisions
            for key, value in user_dict.items():
                result[f"user_{key}"] = value
        
        # Flatten system context
        if self.system:
            system_dict = self.system.to_dict()
            for key, value in system_dict.items():
                result[f"system_{key}"] = value
        
        # Flatten request context
        if self.request:
            request_dict = self.request.to_dict()
            for key, value in request_dict.items():
                result[f"request_{key}"] = value
        
        # Flatten geo context
        if self.geo:
            geo_dict = self.geo.to_dict()
            for key, value in geo_dict.items():
                result[f"geo_{key}"] = value
        
        # Flatten device context
        if self.device:
            device_dict = self.device.to_dict()
            for key, value in device_dict.items():
                result[f"device_{key}"] = value
        
        # Flatten session context
        if self.session:
            session_dict = self.session.to_dict()
            for key, value in session_dict.items():
                result[f"session_{key}"] = value
        
        # Add custom context
        for key, value in self.custom.items():
            result[f"custom_{key}"] = value
        
        return result
    
    def to_json(self) -> str:
        """Serialize context to JSON."""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureContext':
        """Create context from flattened dictionary."""
        # Separate fields by prefix
        user_data = {}
        system_data = {}
        request_data = {}
        geo_data = {}
        device_data = {}
        session_data = {}
        custom_data = {}
        metadata = {}
        
        for key, value in data.items():
            if key.startswith("user_"):
                user_data[key[5:]] = value
            elif key.startswith("system_"):
                system_data[key[7:]] = value
            elif key.startswith("request_"):
                request_data[key[8:]] = value
            elif key.startswith("geo_"):
                geo_data[key[4:]] = value
            elif key.startswith("device_"):
                device_data[key[7:]] = value
            elif key.startswith("session_"):
                session_data[key[8:]] = value
            elif key.startswith("custom_"):
                custom_data[key[7:]] = value
            elif key in ["timestamp", "source", "evaluation_id"]:
                metadata[key] = value
        
        # Create context objects
        user = UserContext.from_dict(user_data) if user_data else None
        system = SystemContext.from_dict(system_data) if system_data else None
        request = RequestContext.from_dict(request_data) if request_data else None
        geo = GeoContext.from_dict(geo_data) if geo_data else None
        device = DeviceContext.from_dict(device_data) if device_data else None
        session = SessionContext.from_dict(session_data) if session_data else None
        
        # Handle timestamp
        if "timestamp" in metadata and metadata["timestamp"]:
            metadata["timestamp"] = datetime.fromisoformat(metadata["timestamp"].replace('Z', '+00:00'))
        
        # Handle source enum
        if "source" in metadata and metadata["source"]:
            metadata["source"] = ContextSource(metadata["source"])
        
        return cls(
            user=user,
            system=system,
            request=request,
            geo=geo,
            device=device,
            session=session,
            custom=custom_data,
            **{k: v for k, v in metadata.items() if k in ["timestamp", "source", "evaluation_id"]}
        )
    
    def merge(self, other: 'FeatureContext') -> 'FeatureContext':
        """Merge another context into this one."""
        # Create new context with combined data
        # For simplicity, non-None values from other context override this one
        return FeatureContext(
            user=other.user or self.user,
            system=other.system or self.system,
            request=other.request or self.request,
            geo=other.geo or self.geo,
            device=other.device or self.device,
            session=other.session or self.session,
            custom={**self.custom, **other.custom},
            timestamp=other.timestamp or self.timestamp,
            source=other.source or self.source,
            evaluation_id=other.evaluation_id or self.evaluation_id,
        )
    
    def get_user_id(self) -> Optional[str]:
        """Get user ID from context."""
        if self.user and self.user.user_id:
            return self.user.user_id
        return None
    
    def get_session_id(self) -> Optional[str]:
        """Get session ID from context."""
        if self.session and self.session.session_id:
            return self.session.session_id
        return None
    
    def get_ip_address(self) -> Optional[str]:
        """Get IP address from context."""
        if self.request and self.request.ip_address:
            return self.request.ip_address
        return None
    
    def get_segments(self) -> List[str]:
        """Get user segments from context."""
        if self.user and self.user.segments:
            return self.user.segments
        return []


# Factory functions for common context creation patterns
def create_request_context(
    request_id: Optional[str] = None,
    method: Optional[str] = None,
    path: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
) -> RequestContext:
    """Create request context from HTTP request."""
    ctx = RequestContext(
        request_id=request_id,
        method=method,
        path=path,
        ip_address=ip_address,
        user_agent=user_agent,
        headers=headers or {},
    )
    if not ctx.request_id:
        ctx.generate_request_id()
    ctx.request_start = datetime.utcnow()
    return ctx


def create_user_context(
    user_id: str,
    username: Optional[str] = None,
    role: str = "user",
    tier: UserTier = UserTier.FREE,
    language: str = "en",
) -> UserContext:
    """Create basic user context."""
    return UserContext(
        user_id=user_id,
        username=username,
        role=role,
        tier=tier,
        language=language,
        created_at=datetime.utcnow(),
        last_login=datetime.utcnow(),
    )


def create_feature_context(
    user: Optional[UserContext] = None,
    request: Optional[RequestContext] = None,
    source: ContextSource = ContextSource.REQUEST,
) -> FeatureContext:
    """Create a complete feature context."""
    return FeatureContext(
        user=user,
        request=request,
        system=SystemContext(),
        source=source,
    )


# Context builder for chaining
class ContextBuilder:
    """Builder pattern for creating feature contexts."""
    
    def __init__(self):
        self._context = FeatureContext()
    
    def with_user(self, user: UserContext) -> 'ContextBuilder':
        """Add user context."""
        self._context.user = user
        return self
    
    def with_request(self, request: RequestContext) -> 'ContextBuilder':
        """Add request context."""
        self._context.request = request
        return self
    
    def with_system(self, system: SystemContext) -> 'ContextBuilder':
        """Add system context."""
        self._context.system = system
        return self
    
    def with_geo(self, geo: GeoContext) -> 'ContextBuilder':
        """Add geographic context."""
        self._context.geo = geo
        return self
    
    def with_device(self, device: DeviceContext) -> 'ContextBuilder':
        """Add device context."""
        self._context.device = device
        return self
    
    def with_session(self, session: SessionContext) -> 'ContextBuilder':
        """Add session context."""
        self._context.session = session
        return self
    
    def with_custom(self, key: str, value: Any) -> 'ContextBuilder':
        """Add custom context value."""
        self._context.custom[key] = value
        return self
    
    def build(self) -> FeatureContext:
        """Build the feature context."""
        return self._context