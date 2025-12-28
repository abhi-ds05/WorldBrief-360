"""
Feature flag definitions for WorldBrief360.
Contains all feature flags used across the application.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json


class FlagType(str, Enum):
    """Types of feature flags."""
    BOOLEAN = "boolean"           # Simple on/off flag
    PERCENTAGE = "percentage"     # Rollout percentage
    MULTIVARIATE = "multivariate" # Multiple variants
    TARGETED = "targeted"        # Targeted to specific users/segments
    TIME_BASED = "time_based"    # Time-based activation


class VariantType(str, Enum):
    """Types of variant values."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    JSON = "json"
    CONFIG = "config"


@dataclass
class FeatureFlag:
    """
    Definition of a feature flag.
    
    Attributes:
        name: Unique identifier for the flag
        description: Human-readable description
        flag_type: Type of flag (boolean, percentage, etc.)
        enabled: Global enable/disable
        variants: Available variants for multivariate flags
        default_variant: Default variant if flag is disabled
        rollout_percentage: Percentage of users (0.0 to 1.0)
        target_users: Specific user IDs to target
        target_segments: User segments to target
        start_time: When flag becomes active
        end_time: When flag expires
        environments: Environments where flag is available
        metadata: Additional flag metadata
    """
    
    name: str
    description: str
    flag_type: FlagType = FlagType.BOOLEAN
    enabled: bool = False
    variants: Dict[str, Any] = field(default_factory=dict)
    default_variant: Any = None
    rollout_percentage: float = 0.0  # 0.0 to 1.0
    target_users: List[str] = field(default_factory=list)
    target_segments: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    environments: List[str] = field(default_factory=lambda: ["development", "staging", "production"])
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate flag configuration."""
        if not 0.0 <= self.rollout_percentage <= 1.0:
            raise ValueError(f"Rollout percentage must be between 0.0 and 1.0, got {self.rollout_percentage}")
        
        if self.flag_type == FlagType.MULTIVARIATE and not self.variants:
            raise ValueError("Multivariate flags must have at least one variant")
        
        if self.start_time and self.end_time and self.start_time >= self.end_time:
            raise ValueError("Start time must be before end time")
    
    def is_enabled_for(
        self, 
        user_id: Optional[str] = None, 
        context: Optional[Dict[str, Any]] = None,
        environment: str = "production"
    ) -> bool:
        """
        Check if feature is enabled for specific user/context.
        
        Args:
            user_id: Optional user identifier
            context: Additional evaluation context
            environment: Current environment
        
        Returns:
            True if feature is enabled, False otherwise
        """
        # Environment check
        if environment not in self.environments:
            return False
        
        # Global enabled check
        if not self.enabled:
            return False
        
        # Time-based checks
        now = datetime.utcnow()
        if self.start_time and now < self.start_time:
            return False
        if self.end_time and now > self.end_time:
            return False
        
        # Targeted users check (explicit allowlist)
        if self.target_users and user_id:
            return user_id in self.target_users
        
        # Segment-based targeting
        if self.target_segments and context and "segments" in context:
            user_segments = context.get("segments", [])
            if any(segment in user_segments for segment in self.target_segments):
                return True
        
        # Percentage-based rollout
        if self.rollout_percentage < 1.0 and user_id:
            return self._get_user_percentage(user_id) < self.rollout_percentage
        
        # If no targeting rules, flag is globally enabled
        return True
    
    def get_variant(
        self, 
        user_id: Optional[str] = None, 
        context: Optional[Dict[str, Any]] = None,
        environment: str = "production"
    ) -> Any:
        """
        Get variant value for user.
        
        Args:
            user_id: Optional user identifier
            context: Additional evaluation context
            environment: Current environment
        
        Returns:
            Variant value or default_variant if flag is disabled
        """
        if not self.is_enabled_for(user_id, context, environment):
            return self.default_variant
        
        if self.flag_type == FlagType.BOOLEAN:
            return True
        
        if self.flag_type == FlagType.PERCENTAGE:
            return self.rollout_percentage
        
        if self.flag_type == FlagType.MULTIVARIATE and self.variants:
            if not self.variants:
                return self.default_variant
            
            # Assign consistent variant based on user_id or context
            variant_key = self._get_variant_key(user_id, context)
            variant_names = list(self.variants.keys())
            variant_idx = variant_key % len(variant_names)
            variant_name = variant_names[variant_idx]
            return self.variants[variant_name]
        
        return self.default_variant
    
    def _get_user_percentage(self, user_id: str) -> float:
        """Get stable percentage for user (0.0 to 1.0)."""
        hash_val = int(hashlib.md5(f"{self.name}:{user_id}".encode()).hexdigest()[:8], 16)
        return (hash_val % 10000) / 10000.0
    
    def _get_variant_key(self, user_id: Optional[str], context: Optional[Dict[str, Any]]) -> int:
        """Generate consistent key for variant selection."""
        if user_id:
            key_str = f"{self.name}:variant:{user_id}"
        elif context and "session_id" in context:
            key_str = f"{self.name}:variant:{context['session_id']}"
        else:
            key_str = f"{self.name}:variant:anonymous"
        
        hash_val = int(hashlib.md5(key_str.encode()).hexdigest()[:8], 16)
        return hash_val
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert flag to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "flag_type": self.flag_type.value,
            "enabled": self.enabled,
            "variants": self.variants,
            "default_variant": self.default_variant,
            "rollout_percentage": self.rollout_percentage,
            "target_users": self.target_users,
            "target_segments": self.target_segments,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "environments": self.environments,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureFlag':
        """Create flag from dictionary."""
        # Handle datetime fields
        start_time = None
        if data.get("start_time"):
            start_time = datetime.fromisoformat(data["start_time"].replace('Z', '+00:00'))
        
        end_time = None
        if data.get("end_time"):
            end_time = datetime.fromisoformat(data["end_time"].replace('Z', '+00:00'))
        
        # Handle flag_type enum
        flag_type = FlagType(data.get("flag_type", "boolean"))
        
        return cls(
            name=data["name"],
            description=data["description"],
            flag_type=flag_type,
            enabled=data.get("enabled", False),
            variants=data.get("variants", {}),
            default_variant=data.get("default_variant"),
            rollout_percentage=data.get("rollout_percentage", 0.0),
            target_users=data.get("target_users", []),
            target_segments=data.get("target_segments", []),
            start_time=start_time,
            end_time=end_time,
            environments=data.get("environments", ["development", "staging", "production"]),
            metadata=data.get("metadata", {}),
        )


class FeatureFlags:
    """
    Collection of all feature flags for WorldBrief360.
    This class serves as a registry and documentation of all available flags.
    """
    
    # ==================== AI & MODEL FEATURES ====================
    
    @classmethod
    def ENHANCED_SUMMARIZATION(cls) -> FeatureFlag:
        """Use advanced summarization models with better coherence."""
        return FeatureFlag(
            name="enhanced_summarization",
            description="Use advanced summarization models with better coherence",
            flag_type=FlagType.PERCENTAGE,
            enabled=True,
            rollout_percentage=0.5,  # 50% rollout
            variants={"model": "mistral-large", "temperature": 0.3},
            environments=["development", "staging", "production"],
        )
    
    @classmethod
    def MULTIMODAL_CHAT(cls) -> FeatureFlag:
        """Enable image upload and analysis in chat."""
        return FeatureFlag(
            name="multimodal_chat",
            description="Enable image upload and analysis in chat",
            flag_type=FlagType.BOOLEAN,
            enabled=False,  # Still in development
            environments=["development", "staging"],
        )
    
    @classmethod
    def REAL_TIME_TRANSLATION(cls) -> FeatureFlag:
        """Real-time translation of news content."""
        return FeatureFlag(
            name="real_time_translation",
            description="Real-time translation of news content",
            flag_type=FlagType.MULTIVARIATE,
            enabled=True,
            variants={
                "google": {"provider": "google", "quality": "high"},
                "deepl": {"provider": "deepl", "quality": "premium"},
                "local": {"provider": "nllb", "quality": "medium"},
            },
            default_variant={"provider": "google", "quality": "high"},
            environments=["development", "staging", "production"],
        )
    
    # ==================== USER EXPERIENCE FEATURES ====================
    
    @classmethod
    def DARK_MODE(cls) -> FeatureFlag:
        """Enable dark mode interface."""
        return FeatureFlag(
            name="dark_mode",
            description="Enable dark mode interface",
            flag_type=FlagType.BOOLEAN,
            enabled=True,
            environments=["development", "staging", "production"],
        )
    
    @classmethod
    def AUDIO_BRIEFINGS(cls) -> FeatureFlag:
        """Enable audio playback of briefings."""
        return FeatureFlag(
            name="audio_briefings",
            description="Enable audio playback of briefings",
            flag_type=FlagType.PERCENTAGE,
            enabled=True,
            rollout_percentage=0.3,  # 30% rollout
            environments=["development", "staging", "production"],
        )
    
    @classmethod
    def PERSONALIZED_FEED(cls) -> FeatureFlag:
        """Personalized news feed based on user interests."""
        return FeatureFlag(
            name="personalized_feed",
            description="Personalized news feed based on user interests",
            flag_type=FlagType.MULTIVARIATE,
            enabled=True,
            variants={
                "basic": {"algorithm": "content_based", "refresh_rate": "hourly"},
                "advanced": {"algorithm": "collaborative", "refresh_rate": "realtime"},
                "experimental": {"algorithm": "hybrid_ai", "refresh_rate": "continuous"},
            },
            default_variant={"algorithm": "content_based", "refresh_rate": "hourly"},
            environments=["development", "staging", "production"],
        )
    
    # ==================== COMMUNITY FEATURES ====================
    
    @classmethod
    def COMMUNITY_VERIFICATION(cls) -> FeatureFlag:
        """Enable community-based incident verification."""
        return FeatureFlag(
            name="community_verification",
            description="Enable community-based incident verification",
            flag_type=FlagType.TARGETED,
            enabled=True,
            target_users=[],  # Add specific user IDs for beta
            target_segments=["power_users", "moderators"],
            environments=["development", "staging", "production"],
        )
    
    @classmethod
    def GAMIFICATION(cls) -> FeatureFlag:
        """Enable gamification features (badges, leaderboards)."""
        return FeatureFlag(
            name="gamification",
            description="Enable gamification features (badges, leaderboards)",
            flag_type=FlagType.BOOLEAN,
            enabled=False,  # In development
            environments=["development", "staging"],
        )
    
    # ==================== PERFORMANCE & INFRASTRUCTURE ====================
    
    @classmethod
    def CACHING_STRATEGY(cls) -> FeatureFlag:
        """Experimental caching strategy for improved performance."""
        return FeatureFlag(
            name="caching_strategy",
            description="Experimental caching strategy for improved performance",
            flag_type=FlagType.MULTIVARIATE,
            enabled=True,
            variants={
                "redis": {"backend": "redis", "ttl": 300},
                "memcached": {"backend": "memcached", "ttl": 600},
                "hybrid": {"backend": "redis+memcached", "ttl": 900},
            },
            default_variant={"backend": "redis", "ttl": 300},
            environments=["staging", "production"],
        )
    
    @classmethod
    def CDN_ENABLED(cls) -> FeatureFlag:
        """Serve static assets through CDN."""
        return FeatureFlag(
            name="cdn_enabled",
            description="Serve static assets through CDN",
            flag_type=FlagType.BOOLEAN,
            enabled=True,
            environments=["production"],
        )
    
    # ==================== ADMIN & MODERATION ====================
    
    @classmethod
    def ADVANCED_MODERATION(cls) -> FeatureFlag:
        """AI-powered content moderation."""
        return FeatureFlag(
            name="advanced_moderation",
            description="AI-powered content moderation",
            flag_type=FlagType.BOOLEAN,
            enabled=True,
            environments=["development", "staging", "production"],
        )
    
    @classmethod
    def ANALYTICS_DASHBOARD(cls) -> FeatureFlag:
        """New analytics dashboard for admins."""
        return FeatureFlag(
            name="analytics_dashboard",
            description="New analytics dashboard for admins",
            flag_type=FlagType.TARGETED,
            enabled=True,
            target_segments=["admins", "analysts"],
            environments=["development", "staging", "production"],
        )
    
    # ==================== BETA / EXPERIMENTAL ====================
    
    @classmethod
    def VOICE_COMMANDS(cls) -> FeatureFlag:
        """Voice command interface (experimental)."""
        return FeatureFlag(
            name="voice_commands",
            description="Voice command interface (experimental)",
            flag_type=FlagType.BOOLEAN,
            enabled=False,
            environments=["development"],
        )
    
    @classmethod
    def AR_VISUALIZATIONS(cls) -> FeatureFlag:
        """AR visualizations for incident data."""
        return FeatureFlag(
            name="ar_visualizations",
            description="AR visualizations for incident data",
            flag_type=FlagType.BOOLEAN,
            enabled=False,
            environments=["development"],
        )
    
    @classmethod
    def get_all_flags(cls) -> Dict[str, FeatureFlag]:
        """Get all defined feature flags."""
        return {
            "enhanced_summarization": cls.ENHANCED_SUMMARIZATION(),
            "multimodal_chat": cls.MULTIMODAL_CHAT(),
            "real_time_translation": cls.REAL_TIME_TRANSLATION(),
            "dark_mode": cls.DARK_MODE(),
            "audio_briefings": cls.AUDIO_BRIEFINGS(),
            "personalized_feed": cls.PERSONALIZED_FEED(),
            "community_verification": cls.COMMUNITY_VERIFICATION(),
            "gamification": cls.GAMIFICATION(),
            "caching_strategy": cls.CACHING_STRATEGY(),
            "cdn_enabled": cls.CDN_ENABLED(),
            "advanced_moderation": cls.ADVANCED_MODERATION(),
            "analytics_dashboard": cls.ANALYTICS_DASHBOARD(),
            "voice_commands": cls.VOICE_COMMANDS(),
            "ar_visualizations": cls.AR_VISUALIZATIONS(),
        }
    
    @classmethod
    def get_flag(cls, name: str) -> Optional[FeatureFlag]:
        """Get specific flag by name."""
        flags = cls.get_all_flags()
        return flags.get(name)
    
    @classmethod
    def get_active_flags(cls, environment: str = "production") -> Dict[str, FeatureFlag]:
        """Get flags active in the specified environment."""
        all_flags = cls.get_all_flags()
        return {
            name: flag for name, flag in all_flags.items()
            if environment in flag.environments
        }


# Helper functions
def create_flag_from_dict(data: dict) -> FeatureFlag:
    """Create a feature flag from dictionary data."""
    return FeatureFlag.from_dict(data)


def hash_user_id(user_id: str, salt: str = "") -> int:
    """Create a stable hash for user ID."""
    key = f"{salt}:{user_id}" if salt else user_id
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


def should_include_user(user_id: str, percentage: float, salt: str = "") -> bool:
    """Determine if user should be included in percentage rollout."""
    if percentage >= 1.0:
        return True
    if percentage <= 0.0:
        return False
    
    hash_val = hash_user_id(user_id, salt)
    return (hash_val % 10000) / 10000.0 < percentage