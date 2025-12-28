"""
user_preferences.py - User Preferences and Settings Model

This module defines models for detailed user preferences, settings, and personalization.
This includes:
- Notification preferences and delivery channels
- Privacy settings and data sharing preferences
- Display and UI preferences
- Content filtering and personalization
- Communication preferences
- Accessibility settings
- Security preferences
- Data export and deletion preferences

Key Features:
- Granular preference control
- Multi-channel notification preferences
- Privacy and data control
- UI customization
- Content personalization
- Accessibility support
- Preference inheritance and defaults
- Preference versioning and history
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from enum import Enum
from sqlalchemy import (
    Column, String, Text, ForeignKey, Integer, DateTime, 
    Boolean, Enum as SQLEnum, JSON, Float, CheckConstraint,
    Index, Table, UniqueConstraint, LargeBinary
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.sql import func, expression
from sqlalchemy.ext.hybrid import hybrid_property

from db.base import Base
from models.mixins import TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from models.user import User
    from models.notification import NotificationChannel, NotificationType


class PreferenceCategory(Enum):
    """Categories of user preferences."""
    NOTIFICATIONS = "notifications"          # Notification preferences
    PRIVACY = "privacy"                      # Privacy settings
    DISPLAY = "display"                      # Display and UI preferences
    CONTENT = "content"                      # Content preferences
    COMMUNICATION = "communication"          # Communication preferences
    ACCESSIBILITY = "accessibility"          # Accessibility settings
    SECURITY = "security"                    # Security preferences
    DATA = "data"                            # Data management preferences
    SYSTEM = "system"                        # System preferences
    OTHER = "other"                          # Other preferences


class PrivacyLevel(Enum):
    """Privacy levels for user data."""
    PUBLIC = "public"                        # Visible to everyone
    REGISTERED = "registered"                # Visible to registered users
    FRIENDS = "friends"                      # Visible to friends/connections
    ORGANIZATION = "organization"            # Visible to organization members
    PRIVATE = "private"                      # Visible only to user
    SYSTEM = "system"                        # Visible only to system


class ThemeMode(Enum):
    """UI theme modes."""
    LIGHT = "light"                          # Light theme
    DARK = "dark"                            # Dark theme
    AUTO = "auto"                            # Automatic (based on system)
    HIGH_CONTRAST = "high_contrast"          # High contrast theme


class FontSize(Enum):
    """Font size options."""
    SMALL = "small"                          # Small font size
    MEDIUM = "medium"                        # Medium font size
    LARGE = "large"                          # Large font size
    EXTRA_LARGE = "extra_large"              # Extra large font size


class ContentRating(Enum):
    """Content rating filters."""
    GENERAL = "general"                      # General audience
    TEEN = "teen"                            # Teen audience
    MATURE = "mature"                        # Mature audience
    ADULT = "adult"                          # Adult audience
    UNRATED = "unrated"                      # No rating


class EmailFrequency(Enum):
    """Email frequency options."""
    IMMEDIATE = "immediate"                  # Immediate notifications
    HOURLY = "hourly"                        # Hourly digest
    DAILY = "daily"                          # Daily digest
    WEEKLY = "weekly"                        # Weekly digest
    MONTHLY = "monthly"                      # Monthly digest
    NEVER = "never"                          # Never send emails


class UserPreferences(Base, UUIDMixin, TimestampMixin):
    """
    User preferences model for managing all user settings.
    
    This model consolidates all user preferences in a structured way,
    with categories and versioning support.
    
    Attributes:
        id: Primary key UUID
        user_id: User ID
        version: Preferences version
        is_active: Whether these preferences are active
        notification_preferences: Notification settings
        privacy_preferences: Privacy settings
        display_preferences: Display/UI settings
        content_preferences: Content settings
        communication_preferences: Communication settings
        accessibility_preferences: Accessibility settings
        security_preferences: Security settings
        data_preferences: Data management settings
        system_preferences: System settings
        custom_preferences: Custom/user-defined preferences
        inherited_from: Source of inherited preferences
        metadata: Additional metadata
    """
    
    __tablename__ = "user_preferences"
    
    # User reference
    user_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        unique=True,  # One active preference set per user
        index=True
    )
    
    # Versioning
    version = Column(String(50), default="1.0.0", nullable=False)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    
    # Preference categories (JSONB for flexibility)
    notification_preferences = Column(JSONB, default=dict, nullable=False)
    privacy_preferences = Column(JSONB, default=dict, nullable=False)
    display_preferences = Column(JSONB, default=dict, nullable=False)
    content_preferences = Column(JSONB, default=dict, nullable=False)
    communication_preferences = Column(JSONB, default=dict, nullable=False)
    accessibility_preferences = Column(JSONB, default=dict, nullable=False)
    security_preferences = Column(JSONB, default=dict, nullable=False)
    data_preferences = Column(JSONB, default=dict, nullable=False)
    system_preferences = Column(JSONB, default=dict, nullable=False)
    custom_preferences = Column(JSONB, default=dict, nullable=False)
    
    # Inheritance and metadata
    inherited_from = Column(JSONB, nullable=True)  # Source of inherited preferences
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    user = relationship("User", backref="user_preferences")
    preference_history = relationship("PreferenceHistory", back_populates="user_preferences", cascade="all, delete-orphan")
    
    # Check constraints
    __table_args__ = (
        CheckConstraint(
            'version ~* \'^\\d+\\.\\d+\\.\\d+$\'',
            name='check_version_format'
        ),
        Index('ix_user_preferences_user_active', 'user_id', 'is_active'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<UserPreferences(id={self.id}, user={self.user_id}, version={self.version})>"
    
    @property
    def all_preferences(self) -> Dict[str, Any]:
        """Get all preferences as a single dictionary."""
        return {
            "notifications": self.notification_preferences,
            "privacy": self.privacy_preferences,
            "display": self.display_preferences,
            "content": self.content_preferences,
            "communication": self.communication_preferences,
            "accessibility": self.accessibility_preferences,
            "security": self.security_preferences,
            "data": self.data_preferences,
            "system": self.system_preferences,
            "custom": self.custom_preferences
        }
    
    @property
    def notification_channels(self) -> Dict[str, bool]:
        """Get enabled notification channels."""
        return self.notification_preferences.get("channels", {})
    
    @property
    def privacy_settings(self) -> Dict[str, PrivacyLevel]:
        """Get privacy settings."""
        return self.privacy_preferences.get("settings", {})
    
    @property
    def theme_settings(self) -> Dict[str, Any]:
        """Get theme settings."""
        return self.display_preferences.get("theme", {})
    
    @property
    def content_filters(self) -> Dict[str, Any]:
        """Get content filters."""
        return self.content_preferences.get("filters", {})
    
    @property
    def email_preferences(self) -> Dict[str, Any]:
        """Get email communication preferences."""
        return self.communication_preferences.get("email", {})
    
    @property
    def accessibility_settings(self) -> Dict[str, Any]:
        """Get accessibility settings."""
        return self.accessibility_preferences.get("settings", {})
    
    @property
    def security_settings(self) -> Dict[str, Any]:
        """Get security settings."""
        return self.security_preferences.get("settings", {})
    
    @property
    def data_management_settings(self) -> Dict[str, Any]:
        """Get data management settings."""
        return self.data_preferences.get("management", {})
    
    def get_preference(self, category: PreferenceCategory, key: str, default: Any = None) -> Any:
        """Get a specific preference value."""
        category_map = {
            PreferenceCategory.NOTIFICATIONS: self.notification_preferences,
            PreferenceCategory.PRIVACY: self.privacy_preferences,
            PreferenceCategory.DISPLAY: self.display_preferences,
            PreferenceCategory.CONTENT: self.content_preferences,
            PreferenceCategory.COMMUNICATION: self.communication_preferences,
            PreferenceCategory.ACCESSIBILITY: self.accessibility_preferences,
            PreferenceCategory.SECURITY: self.security_preferences,
            PreferenceCategory.DATA: self.data_preferences,
            PreferenceCategory.SYSTEM: self.system_preferences,
            PreferenceCategory.OTHER: self.custom_preferences
        }
        
        preferences = category_map.get(category)
        if not preferences:
            return default
        
        # Support dot notation for nested keys
        keys = key.split('.')
        current = preferences
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def set_preference(self, category: PreferenceCategory, key: str, value: Any) -> None:
        """Set a specific preference value."""
        category_map = {
            PreferenceCategory.NOTIFICATIONS: self.notification_preferences,
            PreferenceCategory.PRIVACY: self.privacy_preferences,
            PreferenceCategory.DISPLAY: self.display_preferences,
            PreferenceCategory.CONTENT: self.content_preferences,
            PreferenceCategory.COMMUNICATION: self.communication_preferences,
            PreferenceCategory.ACCESSIBILITY: self.accessibility_preferences,
            PreferenceCategory.SECURITY: self.security_preferences,
            PreferenceCategory.DATA: self.data_preferences,
            PreferenceCategory.SYSTEM: self.system_preferences,
            PreferenceCategory.OTHER: self.custom_preferences
        }
        
        preferences = category_map.get(category)
        if not preferences:
            raise ValueError(f"Invalid preference category: {category}")
        
        # Support dot notation for nested keys
        keys = key.split('.')
        current = preferences
        
        for i, k in enumerate(keys[:-1]):
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def enable_notification_channel(self, channel: str, notification_type: Optional[str] = None) -> None:
        """Enable a notification channel."""
        channels = self.notification_preferences.setdefault("channels", {})
        
        if notification_type:
            # Enable for specific notification type
            type_channels = channels.setdefault(notification_type, {})
            type_channels[channel] = True
        else:
            # Enable for all notifications
            channels[channel] = True
    
    def disable_notification_channel(self, channel: str, notification_type: Optional[str] = None) -> None:
        """Disable a notification channel."""
        channels = self.notification_preferences.setdefault("channels", {})
        
        if notification_type:
            # Disable for specific notification type
            if notification_type in channels and isinstance(channels[notification_type], dict):
                channels[notification_type][channel] = False
        else:
            # Disable for all notifications
            channels[channel] = False
    
    def is_notification_enabled(self, channel: str, notification_type: Optional[str] = None) -> bool:
        """Check if a notification channel is enabled."""
        channels = self.notification_preferences.get("channels", {})
        
        if notification_type:
            # Check for specific notification type
            if notification_type in channels and isinstance(channels[notification_type], dict):
                return channels[notification_type].get(channel, False)
            return False
        else:
            # Check global setting
            return channels.get(channel, False)
    
    def set_privacy_level(self, data_type: str, level: PrivacyLevel) -> None:
        """Set privacy level for a data type."""
        settings = self.privacy_preferences.setdefault("settings", {})
        settings[data_type] = level.value
    
    def get_privacy_level(self, data_type: str) -> PrivacyLevel:
        """Get privacy level for a data type."""
        settings = self.privacy_preferences.get("settings", {})
        level_str = settings.get(data_type, PrivacyLevel.PRIVATE.value)
        return PrivacyLevel(level_str)
    
    def set_theme(self, mode: ThemeMode, custom_colors: Optional[Dict[str, str]] = None) -> None:
        """Set UI theme."""
        theme = self.display_preferences.setdefault("theme", {})
        theme["mode"] = mode.value
        
        if custom_colors:
            theme["custom_colors"] = custom_colors
    
    def get_theme_mode(self) -> ThemeMode:
        """Get current theme mode."""
        theme = self.display_preferences.get("theme", {})
        mode_str = theme.get("mode", ThemeMode.AUTO.value)
        return ThemeMode(mode_str)
    
    def add_content_filter(self, filter_type: str, value: Any) -> None:
        """Add a content filter."""
        filters = self.content_preferences.setdefault("filters", {})
        filter_list = filters.setdefault(filter_type, [])
        
        if value not in filter_list:
            filter_list.append(value)
    
    def remove_content_filter(self, filter_type: str, value: Any) -> None:
        """Remove a content filter."""
        filters = self.content_preferences.get("filters", {})
        if filter_type in filters and isinstance(filters[filter_type], list):
            filter_list = filters[filter_type]
            if value in filter_list:
                filter_list.remove(value)
    
    def set_email_frequency(self, email_type: str, frequency: EmailFrequency) -> None:
        """Set email frequency for a specific email type."""
        email_prefs = self.communication_preferences.setdefault("email", {})
        email_prefs[email_type] = frequency.value
    
    def get_email_frequency(self, email_type: str) -> EmailFrequency:
        """Get email frequency for a specific email type."""
        email_prefs = self.communication_preferences.get("email", {})
        freq_str = email_prefs.get(email_type, EmailFrequency.DAILY.value)
        return EmailFrequency(freq_str)
    
    def enable_accessibility_feature(self, feature: str, settings: Optional[Dict[str, Any]] = None) -> None:
        """Enable an accessibility feature."""
        accessibility = self.accessibility_preferences.setdefault("settings", {})
        feature_settings = accessibility.setdefault(feature, {})
        feature_settings["enabled"] = True
        
        if settings:
            feature_settings.update(settings)
    
    def disable_accessibility_feature(self, feature: str) -> None:
        """Disable an accessibility feature."""
        accessibility = self.accessibility_preferences.setdefault("settings", {})
        if feature in accessibility:
            accessibility[feature]["enabled"] = False
    
    def is_accessibility_feature_enabled(self, feature: str) -> bool:
        """Check if an accessibility feature is enabled."""
        accessibility = self.accessibility_preferences.get("settings", {})
        if feature in accessibility:
            return accessibility[feature].get("enabled", False)
        return False
    
    def set_security_preference(self, setting: str, value: Any) -> None:
        """Set a security preference."""
        security = self.security_preferences.setdefault("settings", {})
        security[setting] = value
    
    def get_security_preference(self, setting: str, default: Any = None) -> Any:
        """Get a security preference."""
        security = self.security_preferences.get("settings", {})
        return security.get(setting, default)
    
    def set_data_preference(self, setting: str, value: Any) -> None:
        """Set a data management preference."""
        data = self.data_preferences.setdefault("management", {})
        data[setting] = value
    
    def get_data_preference(self, setting: str, default: Any = None) -> Any:
        """Get a data management preference."""
        data = self.data_preferences.get("management", {})
        return data.get(setting, default)
    
    def create_snapshot(self, description: Optional[str] = None) -> 'PreferenceHistory':
        """Create a snapshot of current preferences."""
        from models.user_preferences import PreferenceHistory
        
        snapshot = PreferenceHistory(
            user_preferences_id=self.id,
            preferences_snapshot=self.all_preferences,
            version=self.version,
            description=description or f"Snapshot at {datetime.utcnow().isoformat()}"
        )
        return snapshot
    
    def reset_to_defaults(self, category: Optional[PreferenceCategory] = None) -> None:
        """Reset preferences to defaults."""
        defaults = get_default_preferences()
        
        if category:
            # Reset specific category
            category_map = {
                PreferenceCategory.NOTIFICATIONS: ("notification_preferences", defaults["notifications"]),
                PreferenceCategory.PRIVACY: ("privacy_preferences", defaults["privacy"]),
                PreferenceCategory.DISPLAY: ("display_preferences", defaults["display"]),
                PreferenceCategory.CONTENT: ("content_preferences", defaults["content"]),
                PreferenceCategory.COMMUNICATION: ("communication_preferences", defaults["communication"]),
                PreferenceCategory.ACCESSIBILITY: ("accessibility_preferences", defaults["accessibility"]),
                PreferenceCategory.SECURITY: ("security_preferences", defaults["security"]),
                PreferenceCategory.DATA: ("data_preferences", defaults["data"]),
                PreferenceCategory.SYSTEM: ("system_preferences", defaults["system"]),
                PreferenceCategory.OTHER: ("custom_preferences", defaults["custom"])
            }
            
            attr_name, default_value = category_map.get(category, (None, None))
            if attr_name:
                setattr(self, attr_name, default_value)
        else:
            # Reset all preferences
            self.notification_preferences = defaults["notifications"]
            self.privacy_preferences = defaults["privacy"]
            self.display_preferences = defaults["display"]
            self.content_preferences = defaults["content"]
            self.communication_preferences = defaults["communication"]
            self.accessibility_preferences = defaults["accessibility"]
            self.security_preferences = defaults["security"]
            self.data_preferences = defaults["data"]
            self.system_preferences = defaults["system"]
            self.custom_preferences = defaults["custom"]
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert user preferences to dictionary."""
        result = {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "version": self.version,
            "is_active": self.is_active,
            "notification_preferences": self.notification_preferences,
            "privacy_preferences": self.privacy_preferences,
            "display_preferences": self.display_preferences,
            "content_preferences": self.content_preferences,
            "communication_preferences": self.communication_preferences,
            "accessibility_preferences": self.accessibility_preferences,
            "security_preferences": self.security_preferences,
            "data_preferences": self.data_preferences,
            "system_preferences": self.system_preferences,
            "custom_preferences": self.custom_preferences,
            "inherited_from": self.inherited_from,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        if not include_sensitive:
            # Remove sensitive information
            sensitive_keys = ["password", "secret", "token", "key"]
            for category in result:
                if isinstance(result[category], dict):
                    result[category] = self._filter_sensitive_data(result[category], sensitive_keys)
        
        return result
    
    def _filter_sensitive_data(self, data: Dict[str, Any], sensitive_keys: List[str]) -> Dict[str, Any]:
        """Filter sensitive data from dictionary."""
        filtered = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                filtered[key] = "[FILTERED]"
            elif isinstance(value, dict):
                filtered[key] = self._filter_sensitive_data(value, sensitive_keys)
            elif isinstance(value, list):
                filtered[key] = [
                    self._filter_sensitive_data(item, sensitive_keys) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                filtered[key] = value
        return filtered
    
    @classmethod
    def create_for_user(
        cls,
        user_id: uuid.UUID,
        inherit_from: Optional['UserPreferences'] = None,
        **kwargs
    ) -> 'UserPreferences':
        """
        Create preferences for a user.
        
        Args:
            user_id: User ID
            inherit_from: Preferences to inherit from
            **kwargs: Additional preferences
            
        Returns:
            UserPreferences instance
        """
        defaults = get_default_preferences()
        
        if inherit_from:
            # Inherit preferences from another user
            preferences = cls(
                user_id=user_id,
                notification_preferences=inherit_from.notification_preferences,
                privacy_preferences=inherit_from.privacy_preferences,
                display_preferences=inherit_from.display_preferences,
                content_preferences=inherit_from.content_preferences,
                communication_preferences=inherit_from.communication_preferences,
                accessibility_preferences=inherit_from.accessibility_preferences,
                security_preferences=inherit_from.security_preferences,
                data_preferences=inherit_from.data_preferences,
                system_preferences=inherit_from.system_preferences,
                custom_preferences=inherit_from.custom_preferences,
                inherited_from={
                    "source_user_id": str(inherit_from.user_id),
                    "inherited_at": datetime.utcnow().isoformat()
                },
                **kwargs
            )
        else:
            # Use defaults
            preferences = cls(
                user_id=user_id,
                notification_preferences=defaults["notifications"],
                privacy_preferences=defaults["privacy"],
                display_preferences=defaults["display"],
                content_preferences=defaults["content"],
                communication_preferences=defaults["communication"],
                accessibility_preferences=defaults["accessibility"],
                security_preferences=defaults["security"],
                data_preferences=defaults["data"],
                system_preferences=defaults["system"],
                custom_preferences=defaults["custom"],
                **kwargs
            )
        
        return preferences


class PreferenceHistory(Base, UUIDMixin, TimestampMixin):
    """
    Preference history model for tracking changes to user preferences.
    
    This model stores historical snapshots of user preferences for
    auditing, rollback, and analysis purposes.
    
    Attributes:
        id: Primary key UUID
        user_preferences_id: User preferences ID
        preferences_snapshot: Snapshot of all preferences
        version: Preferences version at snapshot
        description: Snapshot description
        change_summary: Summary of changes
        metadata: Additional metadata
    """
    
    __tablename__ = "preference_history"
    
    # Reference to user preferences
    user_preferences_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("user_preferences.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # Snapshot data
    preferences_snapshot = Column(JSONB, nullable=False)
    version = Column(String(50), nullable=False)
    description = Column(Text, nullable=True)
    change_summary = Column(JSONB, nullable=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    user_preferences = relationship("UserPreferences", back_populates="preference_history")
    
    # Check constraints
    __table_args__ = (
        CheckConstraint(
            'version ~* \'^\\d+\\.\\d+\\.\\d+$\'',
            name='check_version_format'
        ),
        Index('ix_preference_history_preferences_date', 'user_preferences_id', 'created_at'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<PreferenceHistory(id={self.id}, preferences={self.user_preferences_id}, version={self.version})>"
    
    @property
    def snapshot_age_days(self) -> float:
        """Get snapshot age in days."""
        delta = datetime.utcnow() - self.created_at
        return delta.total_seconds() / (24 * 3600)
    
    def get_preference_value(self, category: str, key: str, default: Any = None) -> Any:
        """Get a preference value from snapshot."""
        category_data = self.preferences_snapshot.get(category, {})
        
        # Support dot notation for nested keys
        keys = key.split('.')
        current = category_data
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def compare_with_current(self, current_preferences: UserPreferences) -> Dict[str, Any]:
        """Compare snapshot with current preferences."""
        current_all = current_preferences.all_preferences
        changes = {}
        
        for category in current_all:
            if category in self.preferences_snapshot:
                category_changes = self._compare_dicts(
                    self.preferences_snapshot[category],
                    current_all[category]
                )
                if category_changes:
                    changes[category] = category_changes
        
        return changes
    
    def _compare_dicts(self, old: Dict[str, Any], new: Dict[str, Any], path: str = "") -> Dict[str, Any]:
        """Recursively compare two dictionaries."""
        changes = {}
        all_keys = set(old.keys()) | set(new.keys())
        
        for key in all_keys:
            current_path = f"{path}.{key}" if path else key
            old_val = old.get(key)
            new_val = new.get(key)
            
            if isinstance(old_val, dict) and isinstance(new_val, dict):
                nested_changes = self._compare_dicts(old_val, new_val, current_path)
                if nested_changes:
                    changes[key] = nested_changes
            elif old_val != new_val:
                changes[key] = {
                    "old": old_val,
                    "new": new_val,
                    "changed": True
                }
        
        return changes
    
    def restore_to_user(self, user_preferences: UserPreferences) -> Dict[str, Any]:
        """Restore this snapshot to user preferences."""
        # Store current preferences for reference
        current_snapshot = user_preferences.all_preferences
        
        # Restore from snapshot
        user_preferences.notification_preferences = self.preferences_snapshot.get("notifications", {})
        user_preferences.privacy_preferences = self.preferences_snapshot.get("privacy", {})
        user_preferences.display_preferences = self.preferences_snapshot.get("display", {})
        user_preferences.content_preferences = self.preferences_snapshot.get("content", {})
        user_preferences.communication_preferences = self.preferences_snapshot.get("communication", {})
        user_preferences.accessibility_preferences = self.preferences_snapshot.get("accessibility", {})
        user_preferences.security_preferences = self.preferences_snapshot.get("security", {})
        user_preferences.data_preferences = self.preferences_snapshot.get("data", {})
        user_preferences.system_preferences = self.preferences_snapshot.get("system", {})
        user_preferences.custom_preferences = self.preferences_snapshot.get("custom", {})
        
        # Create restoration metadata
        restoration_info = {
            "restored_from_snapshot": str(self.id),
            "restored_at": datetime.utcnow().isoformat(),
            "previous_version": user_preferences.version,
            "changes": self.compare_with_current(user_preferences)
        }
        
        user_preferences.version = self.version
        user_preferences.metadata["restorations"] = user_preferences.metadata.get("restorations", []) + [restoration_info]
        
        return restoration_info
    
    def to_dict(self, include_snapshot: bool = True) -> Dict[str, Any]:
        """Convert preference history to dictionary."""
        result = {
            "id": str(self.id),
            "user_preferences_id": str(self.user_preferences_id),
            "version": self.version,
            "description": self.description,
            "change_summary": self.change_summary,
            "snapshot_age_days": self.snapshot_age_days,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_snapshot:
            result["preferences_snapshot"] = self.preferences_snapshot
        
        return result


class UserPreferenceOverride(Base, UUIDMixin, TimestampMixin):
    """
    User preference override model for temporary or context-specific preferences.
    
    This model allows users to override their default preferences
    for specific contexts, organizations, or time periods.
    
    Attributes:
        id: Primary key UUID
        user_id: User ID
        context_type: Type of context
        context_id: Context identifier
        override_preferences: Override preferences
        expires_at: When override expires
        is_active: Whether override is active
        priority: Override priority
        metadata: Additional metadata
    """
    
    __tablename__ = "user_preference_overrides"
    
    # User and context
    user_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    context_type = Column(String(100), nullable=False, index=True)  # organization, project, device, etc.
    context_id = Column(String(200), nullable=False, index=True)
    
    # Override preferences
    override_preferences = Column(JSONB, default=dict, nullable=False)
    
    # Override settings
    expires_at = Column(DateTime(timezone=True), nullable=True, index=True)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    priority = Column(Integer, default=0, nullable=False)  # Higher priority overrides lower
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Check constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'context_type', 'context_id', name='uq_user_context_override'),
        CheckConstraint('priority >= 0', name='check_priority_non_negative'),
        Index('ix_preference_overrides_active_expiry', 'is_active', 'expires_at'),
        Index('ix_preference_overrides_user_priority', 'user_id', 'priority'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<UserPreferenceOverride(id={self.id}, user={self.user_id}, context={self.context_type}:{self.context_id})>"
    
    @property
    def is_expired(self) -> bool:
        """Check if override is expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    @property
    def is_valid(self) -> bool:
        """Check if override is valid (active and not expired)."""
        return self.is_active and not self.is_expired
    
    @property
    def seconds_until_expiry(self) -> Optional[float]:
        """Get seconds until override expires."""
        if not self.expires_at:
            return None
        remaining = self.expires_at - datetime.utcnow()
        return max(0, remaining.total_seconds())
    
    @property
    def composite_context_id(self) -> str:
        """Get composite context identifier."""
        return f"{self.context_type}:{self.context_id}"
    
    def get_effective_preferences(self, base_preferences: UserPreferences) -> Dict[str, Any]:
        """Get effective preferences by applying override to base."""
        base_all = base_preferences.all_preferences.copy()
        
        # Deep merge override preferences
        return self._deep_merge(base_all, self.override_preferences)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries (override wins)."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def expire(self) -> None:
        """Expire the override."""
        self.is_active = False
    
    def extend(self, additional_days: int = 7) -> None:
        """Extend override expiration."""
        if not self.expires_at:
            self.expires_at = datetime.utcnow() + timedelta(days=additional_days)
        else:
            self.expires_at = self.expires_at + timedelta(days=additional_days)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert preference override to dictionary."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "context_type": self.context_type,
            "context_id": self.context_id,
            "composite_context_id": self.composite_context_id,
            "override_preferences": self.override_preferences,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_active": self.is_active,
            "is_expired": self.is_expired,
            "is_valid": self.is_valid,
            "seconds_until_expiry": self.seconds_until_expiry,
            "priority": self.priority,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class PreferenceTemplate(Base, UUIDMixin, TimestampMixin):
    """
    Preference template model for predefined preference sets.
    
    This model stores templates for common preference configurations
    that users can apply or organizations can enforce.
    
    Attributes:
        id: Primary key UUID
        name: Template name
        description: Template description
        template_type: Type of template
        preferences: Template preferences
        is_active: Whether template is active
        is_system: Whether template is system template
        tags: Categorization tags
        metadata: Additional metadata
    """
    
    __tablename__ = "preference_templates"
    
    # Template information
    name = Column(String(200), nullable=False, index=True)
    description = Column(Text, nullable=True)
    template_type = Column(String(100), nullable=False, index=True)  # organization, role, project, etc.
    
    # Template preferences
    preferences = Column(JSONB, default=dict, nullable=False)
    
    # Template status
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    is_system = Column(Boolean, default=False, nullable=False, index=True)
    
    # Categorization
    tags = Column(ARRAY(String), default=[], nullable=False, index=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Check constraints
    __table_args__ = (
        UniqueConstraint('name', 'template_type', name='uq_template_name_type'),
        Index('ix_preference_templates_type_active', 'template_type', 'is_active'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<PreferenceTemplate(id={self.id}, name={self.name}, type={self.template_type})>"
    
    @property
    def template_categories(self) -> List[str]:
        """Get categories covered by this template."""
        return list(self.preferences.keys())
    
    def apply_to_user(self, user_preferences: UserPreferences, merge_strategy: str = "override") -> None:
        """
        Apply template to user preferences.
        
        Args:
            user_preferences: User preferences to apply to
            merge_strategy: How to merge (override, merge, additive)
        """
        if merge_strategy == "override":
            # Override existing preferences
            for category, settings in self.preferences.items():
                if category == "notifications":
                    user_preferences.notification_preferences = settings
                elif category == "privacy":
                    user_preferences.privacy_preferences = settings
                elif category == "display":
                    user_preferences.display_preferences = settings
                elif category == "content":
                    user_preferences.content_preferences = settings
                elif category == "communication":
                    user_preferences.communication_preferences = settings
                elif category == "accessibility":
                    user_preferences.accessibility_preferences = settings
                elif category == "security":
                    user_preferences.security_preferences = settings
                elif category == "data":
                    user_preferences.data_preferences = settings
                elif category == "system":
                    user_preferences.system_preferences = settings
                elif category == "custom":
                    user_preferences.custom_preferences = settings
        
        elif merge_strategy == "merge":
            # Merge with existing preferences (template wins on conflict)
            for category, settings in self.preferences.items():
                current = getattr(user_preferences, f"{category}_preferences", {})
                merged = self._deep_merge(current, settings)
                setattr(user_preferences, f"{category}_preferences", merged)
        
        elif merge_strategy == "additive":
            # Only add template settings that don't exist
            for category, settings in self.preferences.items():
                current = getattr(user_preferences, f"{category}_preferences", {})
                additive = self._additive_merge(current, settings)
                setattr(user_preferences, f"{category}_preferences", additive)
        
        # Record template application
        user_preferences.metadata.setdefault("applied_templates", []).append({
            "template_id": str(self.id),
            "template_name": self.name,
            "applied_at": datetime.utcnow().isoformat(),
            "merge_strategy": merge_strategy
        })
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries (override wins)."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _additive_merge(self, base: Dict[str, Any], addition: Dict[str, Any]) -> Dict[str, Any]:
        """Additive merge (only add missing keys)."""
        result = base.copy()
        
        for key, value in addition.items():
            if key not in result:
                result[key] = value
            elif isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._additive_merge(result[key], value)
        
        return result
    
    def to_dict(self, include_preferences: bool = True) -> Dict[str, Any]:
        """Convert preference template to dictionary."""
        result = {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "template_type": self.template_type,
            "is_active": self.is_active,
            "is_system": self.is_system,
            "template_categories": self.template_categories,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_preferences:
            result["preferences"] = self.preferences
        
        return result


# Helper functions
def get_default_preferences() -> Dict[str, Dict[str, Any]]:
    """Get default preferences for all categories."""
    return {
        "notifications": {
            "enabled": True,
            "channels": {
                "in_app": True,
                "email": True,
                "push": True,
                "sms": False,
                "webhook": False
            },
            "quiet_hours": {
                "enabled": False,
                "start": "22:00",
                "end": "07:00",
                "timezone": "UTC"
            },
            "digest_frequency": "daily",
            "mute_categories": [],
            "urgent_notifications": {
                "always_show": True,
                "override_quiet_hours": True
            }
        },
        "privacy": {
            "settings": {
                "profile": PrivacyLevel.PUBLIC.value,
                "email": PrivacyLevel.PRIVATE.value,
                "phone": PrivacyLevel.PRIVATE.value,
                "location": PrivacyLevel.PRIVATE.value,
                "activity": PrivacyLevel.REGISTERED.value,
                "connections": PrivacyLevel.FRIENDS.value,
                "content": PrivacyLevel.PUBLIC.value
            },
            "data_sharing": {
                "analytics": True,
                "personalization": True,
                "third_parties": False,
                "research": False
            },
            "visibility": {
                "online_status": True,
                "last_seen": True,
                "read_receipts": True
            }
        },
        "display": {
            "theme": {
                "mode": ThemeMode.AUTO.value,
                "custom_colors": {},
                "font_size": FontSize.MEDIUM.value,
                "font_family": "system-ui"
            },
            "layout": {
                "density": "comfortable",
                "sidebar_position": "left",
                "language": "en-US",
                "timezone": "UTC",
                "date_format": "YYYY-MM-DD",
                "time_format": "24h"
            },
            "animations": {
                "enabled": True,
                "reduce_motion": False
            }
        },
        "content": {
            "filters": {
                "content_rating": [ContentRating.GENERAL.value, ContentRating.TEEN.value],
                "languages": ["en"],
                "topics": [],
                "sources": [],
                "exclude_keywords": []
            },
            "personalization": {
                "recommendations": True,
                "trending": True,
                "popular": True,
                "following": True
            },
            "display": {
                "auto_play_videos": False,
                "show_images": True,
                "show_videos": True,
                "show_nsfw": False,
                "spoiler_warnings": True
            }
        },
        "communication": {
            "email": {
                "notifications": EmailFrequency.DAILY.value,
                "marketing": EmailFrequency.WEEKLY.value,
                "newsletter": EmailFrequency.WEEKLY.value,
                "updates": EmailFrequency.IMMEDIATE.value
            },
            "messaging": {
                "allow_direct_messages": True,
                "allow_group_messages": True,
                "message_notifications": True,
                "read_receipts": True
            },
            "comments": {
                "notify_on_mentions": True,
                "notify_on_replies": True,
                "allow_comments": True,
                "moderate_comments": False
            }
        },
        "accessibility": {
            "settings": {
                "screen_reader": {
                    "enabled": False,
                    "verbosity": "normal"
                },
                "high_contrast": {
                    "enabled": False,
                    "mode": "normal"
                },
                "keyboard_navigation": {
                    "enabled": True,
                    "shortcuts": True
                },
                "text_to_speech": {
                    "enabled": False,
                    "rate": "normal"
                },
                "color_blindness": {
                    "enabled": False,
                    "type": "protanopia"
                }
            }
        },
        "security": {
            "settings": {
                "two_factor_auth": False,
                "login_alerts": True,
                "device_management": True,
                "session_timeout": 120,  # minutes
                "password_requirements": {
                    "min_length": 8,
                    "require_uppercase": True,
                    "require_lowercase": True,
                    "require_numbers": True,
                    "require_special": True
                }
            },
            "privacy": {
                "data_export": True,
                "data_deletion": True,
                "cookie_consent": "essential"
            }
        },
        "data": {
            "management": {
                "auto_delete_old_data": False,
                "data_retention_days": 365,
                "export_format": "json",
                "backup_frequency": "monthly"
            },
            "storage": {
                "max_storage_mb": 1024,
                "auto_cleanup": True,
                "cloud_sync": False
            }
        },
        "system": {
            "performance": {
                "cache_enabled": True,
                "prefetch_data": True,
                "background_sync": True
            },
            "updates": {
                "auto_check": True,
                "auto_download": False,
                "notification_channel": "in_app"
            },
            "diagnostics": {
                "error_reporting": True,
                "usage_statistics": True,
                "crash_reports": True
            }
        },
        "custom": {}
    }


def merge_preferences(base: Dict[str, Any], override: Dict[str, Any], strategy: str = "override") -> Dict[str, Any]:
    """
    Merge two preference dictionaries.
    
    Args:
        base: Base preferences
        override: Override preferences
        strategy: Merge strategy (override, merge, additive)
        
    Returns:
        Merged preferences
    """
    if strategy == "override":
        return override.copy()
    elif strategy == "merge":
        return _deep_merge_dicts(base, override)
    elif strategy == "additive":
        return _additive_merge_dicts(base, override)
    else:
        raise ValueError(f"Unknown merge strategy: {strategy}")


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries (override wins)."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def _additive_merge_dicts(base: Dict[str, Any], addition: Dict[str, Any]) -> Dict[str, Any]:
    """Additive merge (only add missing keys)."""
    result = base.copy()
    
    for key, value in addition.items():
        if key not in result:
            result[key] = value
        elif isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _additive_merge_dicts(result[key], value)
    
    return result


def validate_preferences(preferences: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate preference structure and values.
    
    Args:
        preferences: Preferences to validate
        
    Returns:
        Validation result with errors and warnings
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "normalized": {}
    }
    
    defaults = get_default_preferences()
    
    for category, default_settings in defaults.items():
        if category not in preferences:
            result["warnings"].append(f"Missing category: {category}")
            result["normalized"][category] = default_settings
            continue
        
        user_settings = preferences[category]
        
        if not isinstance(user_settings, dict):
            result["errors"].append(f"Category {category} must be a dictionary")
            result["valid"] = False
            continue
        
        # Validate structure recursively
        validation = _validate_preference_structure(user_settings, default_settings, category)
        
        if validation["errors"]:
            result["errors"].extend(validation["errors"])
            result["valid"] = False
        
        if validation["warnings"]:
            result["warnings"].extend(validation["warnings"])
        
        result["normalized"][category] = validation["normalized"]
    
    return result


def _validate_preference_structure(
    user_settings: Dict[str, Any],
    default_settings: Dict[str, Any],
    path: str = ""
) -> Dict[str, Any]:
    """Recursively validate preference structure."""
    result = {
        "errors": [],
        "warnings": [],
        "normalized": default_settings.copy()
    }
    
    for key, default_value in default_settings.items():
        current_path = f"{path}.{key}" if path else key
        
        if key in user_settings:
            user_value = user_settings[key]
            
            if isinstance(default_value, dict) and isinstance(user_value, dict):
                # Recursively validate nested dictionary
                nested = _validate_preference_structure(user_value, default_value, current_path)
                result["errors"].extend(nested["errors"])
                result["warnings"].extend(nested["warnings"])
                result["normalized"][key] = nested["normalized"]
            else:
                # Validate leaf value
                if type(user_value) != type(default_value) and not (
                    isinstance(user_value, (int, float)) and isinstance(default_value, (int, float))
                ):
                    result["warnings"].append(
                        f"Type mismatch at {current_path}: "
                        f"expected {type(default_value).__name__}, got {type(user_value).__name__}"
                    )
                result["normalized"][key] = user_value
    
    return result


def get_effective_preferences(
    user_id: uuid.UUID,
    context_type: Optional[str] = None,
    context_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get effective preferences for a user in a specific context.
    
    This function would typically query the database and apply
    all relevant overrides and templates.
    
    Args:
        user_id: User ID
        context_type: Context type
        context_id: Context ID
        
    Returns:
        Effective preferences
    """
    # This is a placeholder for the actual implementation
    # In production, this would query the database and apply logic
    
    # Get base preferences
    # Apply organization templates
    # Apply role-based templates
    # Apply context overrides
    # Return merged preferences
    
    return get_default_preferences()