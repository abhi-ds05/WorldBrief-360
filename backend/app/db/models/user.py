"""
user.py - User and Authentication Model

This module defines models for user management, authentication, and profiles.
This includes:
- User accounts and authentication
- User profiles and preferences
- Authentication methods (password, OAuth, MFA)
- Role and permission management
- User sessions and activity tracking
- Password reset and account recovery
- User preferences and settings

Key Features:
- Multi-factor authentication
- OAuth2 and social login integration
- Role-based access control (RBAC)
- Session management with JWT
- User activity tracking
- Password policy enforcement
- Account verification and recovery
- GDPR compliance tools
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from enum import Enum
import secrets
import hashlib
from sqlalchemy import (
    Column, String, Text, ForeignKey, Integer, DateTime, 
    Boolean, Enum as SQLEnum, JSON, Float, CheckConstraint,
    Index, Table, UniqueConstraint, LargeBinary, event
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.sql import func, expression
from sqlalchemy.ext.hybrid import hybrid_property

from db.base import Base
from models.mixins import TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from models.organization import Organization, OrganizationMember
    from models.subscription import Subscription
    from models.notification import Notification
    from models.incident import Incident
    from models.article import Article
    from models.comment import Comment
    from models.topic import UserTopicInterest


class UserStatus(Enum):
    """User account status."""
    ACTIVE = "active"                # Active account
    INACTIVE = "inactive"            # Inactive account
    SUSPENDED = "suspended"          # Suspended account
    BANNED = "banned"                # Banned account
    PENDING = "pending"              # Pending activation/verification
    ARCHIVED = "archived"            # Archived account
    DELETED = "deleted"              # Soft-deleted account


class UserRole(Enum):
    """User system roles."""
    SUPER_ADMIN = "super_admin"      # System super administrator
    ADMIN = "admin"                  # System administrator
    MODERATOR = "moderator"          # Content moderator
    EDITOR = "editor"                # Content editor
    VERIFIER = "verifier"            # Incident verifier
    ANALYST = "analyst"              # Data analyst
    RESEARCHER = "researcher"        # Researcher
    REPORTER = "reporter"            # Incident reporter
    VIEWER = "viewer"                # Read-only viewer
    USER = "user"                    # Regular user
    GUEST = "guest"                  # Guest user


class AuthProvider(Enum):
    """Authentication providers."""
    LOCAL = "local"                  # Local authentication
    GOOGLE = "google"                # Google OAuth
    FACEBOOK = "facebook"            # Facebook OAuth
    GITHUB = "github"                # GitHub OAuth
    TWITTER = "twitter"              # Twitter OAuth
    MICROSOFT = "microsoft"          # Microsoft OAuth
    LINKEDIN = "linkedin"            # LinkedIn OAuth
    APPLE = "apple"                  # Apple Sign-In
    SAML = "saml"                    # SAML SSO
    LDAP = "ldap"                    # LDAP authentication


class MfaMethod(Enum):
    """Multi-factor authentication methods."""
    TOTP = "totp"                    # Time-based OTP (Google Authenticator)
    SMS = "sms"                      # SMS verification
    EMAIL = "email"                  # Email verification
    BACKUP_CODE = "backup_code"      # Backup codes
    SECURITY_KEY = "security_key"    # Security key (WebAuthn)
    PUSH = "push"                    # Push notification


class Gender(Enum):
    """Gender options."""
    MALE = "male"                    # Male
    FEMALE = "female"                # Female
    NON_BINARY = "non_binary"        # Non-binary
    OTHER = "other"                  # Other
    PREFER_NOT_TO_SAY = "prefer_not_to_say"  # Prefer not to say


class User(Base, UUIDMixin, TimestampMixin):
    """
    User model for account management and authentication.
    
    This model represents user accounts in the system, including
    authentication, profiles, and account settings.
    
    Attributes:
        id: Primary key UUID
        username: Unique username
        email: Unique email address
        email_verified: Whether email is verified
        phone_number: Phone number
        phone_verified: Whether phone is verified
        password_hash: Hashed password
        password_salt: Password salt
        password_changed_at: When password was last changed
        status: User account status
        role: User system role
        auth_provider: Primary authentication provider
        mfa_enabled: Whether MFA is enabled
        mfa_method: Primary MFA method
        last_login_at: Last login timestamp
        last_login_ip: Last login IP address
        login_count: Total login count
        failed_login_attempts: Failed login attempts
        locked_until: Account locked until timestamp
        timezone: User timezone
        locale: User locale/language
        profile: User profile information
        preferences: User preferences
        settings: User settings
        metadata: Additional metadata
        tags: Categorization tags
        deleted_at: When user was soft deleted
    """
    
    __tablename__ = "users"
    
    # Authentication
    username = Column(String(100), nullable=False, unique=True, index=True)
    email = Column(String(255), nullable=False, unique=True, index=True)
    email_verified = Column(Boolean, default=False, nullable=False, index=True)
    phone_number = Column(String(50), nullable=True, unique=True, index=True)
    phone_verified = Column(Boolean, default=False, nullable=False)
    
    # Password (for local authentication)
    password_hash = Column(String(255), nullable=True)
    password_salt = Column(String(255), nullable=True)
    password_changed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Account status
    status = Column(SQLEnum(UserStatus), default=UserStatus.PENDING, nullable=False, index=True)
    role = Column(SQLEnum(UserRole), default=UserRole.USER, nullable=False, index=True)
    
    # Authentication provider
    auth_provider = Column(SQLEnum(AuthProvider), default=AuthProvider.LOCAL, nullable=False, index=True)
    auth_provider_id = Column(String(255), nullable=True, index=True)  # External provider ID
    
    # Multi-factor authentication
    mfa_enabled = Column(Boolean, default=False, nullable=False, index=True)
    mfa_method = Column(SQLEnum(MfaMethod), nullable=True)
    mfa_secret = Column(String(255), nullable=True)  # Encrypted MFA secret
    
    # Login tracking
    last_login_at = Column(DateTime(timezone=True), nullable=True, index=True)
    last_login_ip = Column(String(50), nullable=True)
    login_count = Column(Integer, default=0, nullable=False)
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # Localization
    timezone = Column(String(100), default="UTC", nullable=False)
    locale = Column(String(10), default="en-US", nullable=False)
    
    # User data
    profile = Column(JSONB, default=dict, nullable=False)
    preferences = Column(JSONB, default=dict, nullable=False)
    settings = Column(JSONB, default=dict, nullable=False)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    tags = Column(ARRAY(String), default=[], nullable=False, index=True)
    
    # Deletion (soft delete)
    deleted_at = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # Relationships
    organizations = relationship("OrganizationMember", back_populates="user", cascade="all, delete-orphan")
    subscriptions = relationship("Subscription", foreign_keys="Subscription.user_id", back_populates="user")
    notifications = relationship("Notification", foreign_keys="Notification.user_id", back_populates="user")
    incidents = relationship("Incident", back_populates="reporter")
    articles = relationship("Article", back_populates="author")
    comments = relationship("Comment", back_populates="author")
    user_topic_interests = relationship("UserTopicInterest", back_populates="user")
    
    # Authentication relationships
    auth_tokens = relationship("AuthToken", back_populates="user", cascade="all, delete-orphan")
    oauth_accounts = relationship("OAuthAccount", back_populates="user", cascade="all, delete-orphan")
    mfa_devices = relationship("MfaDevice", back_populates="user", cascade="all, delete-orphan")
    backup_codes = relationship("BackupCode", back_populates="user", cascade="all, delete-orphan")
    password_resets = relationship("PasswordReset", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    login_history = relationship("LoginHistory", back_populates="user", cascade="all, delete-orphan")
    
    # Audit relationships
    audit_logs = relationship("AuditLog", foreign_keys="AuditLog.user_id", back_populates="user")
    created_transactions = relationship("Transaction", foreign_keys="Transaction.initiated_by", back_populates="initiator")
    
    # Check constraints
    __table_args__ = (
        CheckConstraint('login_count >= 0', name='check_login_count_non_negative'),
        CheckConstraint('failed_login_attempts >= 0', name='check_failed_login_attempts_non_negative'),
        CheckConstraint(
            'email ~* \'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}$\'',
            name='check_email_format'
        ),
        CheckConstraint(
            'username ~* \'^[a-zA-Z0-9_]{3,100}$\'',
            name='check_username_format'
        ),
        Index('ix_users_status_role', 'status', 'role'),
        Index('ix_users_auth_provider', 'auth_provider', 'auth_provider_id'),
        Index('ix_users_mfa_status', 'mfa_enabled', 'status'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<User(id={self.id}, username={self.username}, email={self.email}, role={self.role.value})>"
    
    @property
    def is_active(self) -> bool:
        """Check if user account is active."""
        return self.status == UserStatus.ACTIVE
    
    @property
    def is_suspended(self) -> bool:
        """Check if user account is suspended."""
        return self.status == UserStatus.SUSPENDED
    
    @property
    def is_banned(self) -> bool:
        """Check if user account is banned."""
        return self.status == UserStatus.BANNED
    
    @property
    def is_pending(self) -> bool:
        """Check if user account is pending."""
        return self.status == UserStatus.PENDING
    
    @property
    def is_deleted(self) -> bool:
        """Check if user account is deleted."""
        return self.deleted_at is not None or self.status == UserStatus.DELETED
    
    @property
    def is_locked(self) -> bool:
        """Check if user account is locked."""
        if not self.locked_until:
            return False
        return datetime.utcnow() < self.locked_until
    
    @property
    def is_admin(self) -> bool:
        """Check if user is an administrator."""
        return self.role in [UserRole.SUPER_ADMIN, UserRole.ADMIN]
    
    @property
    def is_moderator(self) -> bool:
        """Check if user is a moderator."""
        return self.role in [UserRole.SUPER_ADMIN, UserRole.ADMIN, UserRole.MODERATOR]
    
    @property
    def is_verified(self) -> bool:
        """Check if user is verified (email or phone)."""
        return self.email_verified or self.phone_verified
    
    @property
    def full_name(self) -> Optional[str]:
        """Get user's full name from profile."""
        profile = self.profile or {}
        first_name = profile.get('first_name', '').strip()
        last_name = profile.get('last_name', '').strip()
        
        if first_name and last_name:
            return f"{first_name} {last_name}"
        elif first_name:
            return first_name
        elif last_name:
            return last_name
        return None
    
    @property
    def display_name(self) -> str:
        """Get display name (full name or username)."""
        return self.full_name or self.username
    
    @property
    def avatar_url(self) -> Optional[str]:
        """Get avatar URL from profile."""
        profile = self.profile or {}
        return profile.get('avatar_url')
    
    @property
    def age(self) -> Optional[int]:
        """Calculate age from birthdate."""
        profile = self.profile or {}
        birthdate_str = profile.get('birthdate')
        
        if not birthdate_str:
            return None
        
        try:
            birthdate = datetime.fromisoformat(birthdate_str.replace('Z', '+00:00'))
            today = datetime.utcnow()
            age = today.year - birthdate.year
            
            # Adjust if birthday hasn't occurred yet this year
            if (today.month, today.day) < (birthdate.month, birthdate.day):
                age -= 1
            
            return max(0, age)
        except (ValueError, TypeError):
            return None
    
    @property
    def days_since_last_login(self) -> Optional[int]:
        """Get days since last login."""
        if not self.last_login_at:
            return None
        delta = datetime.utcnow() - self.last_login_at
        return delta.days
    
    @property
    def account_age_days(self) -> int:
        """Get account age in days."""
        delta = datetime.utcnow() - self.created_at
        return delta.days
    
    @property
    def password_age_days(self) -> Optional[int]:
        """Get password age in days."""
        if not self.password_changed_at:
            return None
        delta = datetime.utcnow() - self.password_changed_at
        return delta.days
    
    @property
    def requires_password_change(self) -> bool:
        """Check if password change is required."""
        if not self.password_age_days:
            return False
        
        # Check password policy
        max_password_age = self.settings.get('security', {}).get('max_password_age_days', 90)
        return self.password_age_days > max_password_age
    
    @hybrid_property
    def is_authenticated(self) -> bool:
        """Check if user is authenticated."""
        return self.is_active and not self.is_locked and not self.is_deleted
    
    @validates('email')
    def validate_email(self, key: str, email: str) -> str:
        """Validate email address."""
        email = email.strip().lower()
        if not email:
            raise ValueError("Email cannot be empty")
        
        # Basic email validation
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            raise ValueError("Invalid email format")
        
        return email
    
    @validates('username')
    def validate_username(self, key: str, username: str) -> str:
        """Validate username."""
        username = username.strip()
        if not username:
            raise ValueError("Username cannot be empty")
        
        if len(username) < 3:
            raise ValueError("Username must be at least 3 characters")
        
        if len(username) > 100:
            raise ValueError("Username cannot exceed 100 characters")
        
        # Alphanumeric and underscores only
        import re
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            raise ValueError("Username can only contain letters, numbers, and underscores")
        
        return username
    
    @validates('phone_number')
    def validate_phone_number(self, key: str, phone_number: Optional[str]) -> Optional[str]:
        """Validate phone number."""
        if not phone_number:
            return None
        
        phone_number = phone_number.strip()
        
        # Basic phone validation - can be enhanced with library like phonenumbers
        import re
        phone_pattern = r'^[\+\d\s\-\(\)]+$'
        if not re.match(phone_pattern, phone_number):
            raise ValueError("Invalid phone number format")
        
        return phone_number
    
    def set_password(self, password: str) -> None:
        """Set user password with hashing."""
        if not password:
            raise ValueError("Password cannot be empty")
        
        # Enforce password policy
        self._validate_password_policy(password)
        
        # Generate salt and hash
        salt = secrets.token_hex(32)
        password_hash = self._hash_password(password, salt)
        
        self.password_hash = password_hash
        self.password_salt = salt
        self.password_changed_at = datetime.utcnow()
        self.failed_login_attempts = 0  # Reset failed attempts on password change
    
    def verify_password(self, password: str) -> bool:
        """Verify password against stored hash."""
        if not self.password_hash or not self.password_salt:
            return False
        
        # Check if account is locked
        if self.is_locked:
            return False
        
        # Hash provided password with stored salt
        test_hash = self._hash_password(password, self.password_salt)
        
        # Compare hashes
        if secrets.compare_digest(test_hash, self.password_hash):
            # Reset failed login attempts on successful login
            self.failed_login_attempts = 0
            return True
        else:
            # Increment failed login attempts
            self.failed_login_attempts += 1
            
            # Lock account after too many failed attempts
            max_attempts = self.settings.get('security', {}).get('max_login_attempts', 5)
            if self.failed_login_attempts >= max_attempts:
                self.lock_account(minutes=15)
            
            return False
    
    def _hash_password(self, password: str, salt: str) -> str:
        """Hash password with salt."""
        # Use PBKDF2 with SHA-256
        iterations = 100000
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            iterations
        )
        return f"pbkdf2_sha256${iterations}${salt}${key.hex()}"
    
    def _validate_password_policy(self, password: str) -> None:
        """Validate password against security policy."""
        policy = self.settings.get('security', {}).get('password_policy', {})
        
        min_length = policy.get('min_length', 8)
        if len(password) < min_length:
            raise ValueError(f"Password must be at least {min_length} characters")
        
        if policy.get('require_uppercase', True):
            if not any(c.isupper() for c in password):
                raise ValueError("Password must contain at least one uppercase letter")
        
        if policy.get('require_lowercase', True):
            if not any(c.islower() for c in password):
                raise ValueError("Password must contain at least one lowercase letter")
        
        if policy.get('require_numbers', True):
            if not any(c.isdigit() for c in password):
                raise ValueError("Password must contain at least one number")
        
        if policy.get('require_special', True):
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?~"
            if not any(c in special_chars for c in password):
                raise ValueError(f"Password must contain at least one special character ({special_chars})")
    
    def lock_account(self, minutes: int = 15) -> None:
        """Lock user account for specified minutes."""
        self.locked_until = datetime.utcnow() + timedelta(minutes=minutes)
    
    def unlock_account(self) -> None:
        """Unlock user account."""
        self.locked_until = None
        self.failed_login_attempts = 0
    
    def record_login(self, ip_address: Optional[str] = None) -> None:
        """Record user login."""
        self.last_login_at = datetime.utcnow()
        self.last_login_ip = ip_address
        self.login_count += 1
        self.failed_login_attempts = 0  # Reset on successful login
        
        # Unlock account if it was locked
        if self.is_locked:
            self.unlock_account()
    
    def enable_mfa(self, method: MfaMethod, secret: Optional[str] = None) -> None:
        """Enable multi-factor authentication."""
        self.mfa_enabled = True
        self.mfa_method = method
        if secret:
            self.mfa_secret = secret  # Should be encrypted in production
    
    def disable_mfa(self) -> None:
        """Disable multi-factor authentication."""
        self.mfa_enabled = False
        self.mfa_method = None
        self.mfa_secret = None
    
    def verify_email(self) -> None:
        """Verify user email address."""
        self.email_verified = True
        if self.status == UserStatus.PENDING:
            self.status = UserStatus.ACTIVE
    
    def verify_phone(self) -> None:
        """Verify user phone number."""
        self.phone_verified = True
    
    def update_profile(self, profile_data: Dict[str, Any]) -> None:
        """Update user profile."""
        current_profile = self.profile.copy()
        current_profile.update(profile_data)
        self.profile = current_profile
    
    def update_preferences(self, preferences_data: Dict[str, Any]) -> None:
        """Update user preferences."""
        current_preferences = self.preferences.copy()
        current_preferences.update(preferences_data)
        self.preferences = current_preferences
    
    def update_settings(self, settings_data: Dict[str, Any]) -> None:
        """Update user settings."""
        current_settings = self.settings.copy()
        current_settings.update(settings_data)
        self.settings = current_settings
    
    def soft_delete(self) -> None:
        """Soft delete user account."""
        self.status = UserStatus.DELETED
        self.deleted_at = datetime.utcnow()
        
        # Anonymize sensitive data
        self.email = f"deleted_{self.id}@deleted.local"
        self.username = f"deleted_{self.id}"
        self.phone_number = None
        self.password_hash = None
        self.password_salt = None
        self.mfa_secret = None
    
    def restore(self) -> None:
        """Restore soft-deleted user account."""
        self.status = UserStatus.ACTIVE
        self.deleted_at = None
    
    def to_dict(
        self,
        include_sensitive: bool = False,
        include_relationships: bool = False
    ) -> Dict[str, Any]:
        """Convert user to dictionary."""
        result = {
            "id": str(self.id),
            "username": self.username,
            "email": self.email,
            "email_verified": self.email_verified,
            "phone_number": self.phone_number,
            "phone_verified": self.phone_verified,
            "status": self.status.value,
            "role": self.role.value,
            "auth_provider": self.auth_provider.value,
            "mfa_enabled": self.mfa_enabled,
            "mfa_method": self.mfa_method.value if self.mfa_method else None,
            "is_active": self.is_active,
            "is_suspended": self.is_suspended,
            "is_banned": self.is_banned,
            "is_pending": self.is_pending,
            "is_deleted": self.is_deleted,
            "is_locked": self.is_locked,
            "is_admin": self.is_admin,
            "is_moderator": self.is_moderator,
            "is_verified": self.is_verified,
            "is_authenticated": self.is_authenticated,
            "full_name": self.full_name,
            "display_name": self.display_name,
            "avatar_url": self.avatar_url,
            "age": self.age,
            "timezone": self.timezone,
            "locale": self.locale,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
            "last_login_ip": self.last_login_ip,
            "days_since_last_login": self.days_since_last_login,
            "login_count": self.login_count,
            "failed_login_attempts": self.failed_login_attempts,
            "locked_until": self.locked_until.isoformat() if self.locked_until else None,
            "account_age_days": self.account_age_days,
            "password_age_days": self.password_age_days,
            "requires_password_change": self.requires_password_change,
            "profile": self.profile,
            "preferences": self.preferences,
            "settings": self.settings,
            "tags": self.tags,
            "metadata": self.metadata,
            "deleted_at": self.deleted_at.isoformat() if self.deleted_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_sensitive:
            result.update({
                "auth_provider_id": self.auth_provider_id,
                "password_changed_at": self.password_changed_at.isoformat() if self.password_changed_at else None,
            })
        
        if include_relationships:
            # Add counts for relationships
            result.update({
                "organization_count": len(self.organizations),
                "subscription_count": len(self.subscriptions),
                "notification_count": len(self.notifications),
                "incident_count": len(self.incidents),
                "article_count": len(self.articles),
                "comment_count": len(self.comments)
            })
        
        return result
    
    @classmethod
    def create(
        cls,
        username: str,
        email: str,
        auth_provider: AuthProvider = AuthProvider.LOCAL,
        password: Optional[str] = None,
        role: UserRole = UserRole.USER,
        status: UserStatus = UserStatus.PENDING,
        auth_provider_id: Optional[str] = None,
        profile: Optional[Dict[str, Any]] = None,
        preferences: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None,
        timezone: str = "UTC",
        locale: str = "en-US",
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> 'User':
        """
        Factory method to create a new user.
        
        Args:
            username: Username
            email: Email address
            auth_provider: Authentication provider
            password: Password (for local auth)
            role: User role
            status: User status
            auth_provider_id: External provider ID
            profile: User profile
            preferences: User preferences
            settings: User settings
            timezone: Timezone
            locale: Locale
            metadata: Additional metadata
            tags: Categorization tags
            **kwargs: Additional arguments
            
        Returns:
            A new User instance
        """
        user = cls(
            username=username,
            email=email,
            auth_provider=auth_provider,
            auth_provider_id=auth_provider_id,
            role=role,
            status=status,
            timezone=timezone,
            locale=locale,
            profile=profile or {},
            preferences=preferences or {},
            settings=settings or cls._get_default_settings(),
            metadata=metadata or {},
            tags=tags or [],
            **kwargs
        )
        
        # Set password if provided
        if password and auth_provider == AuthProvider.LOCAL:
            user.set_password(password)
        
        return user
    
    @staticmethod
    def _get_default_settings() -> Dict[str, Any]:
        """Get default user settings."""
        return {
            "security": {
                "password_policy": {
                    "min_length": 8,
                    "require_uppercase": True,
                    "require_lowercase": True,
                    "require_numbers": True,
                    "require_special": True,
                    "max_password_age_days": 90
                },
                "max_login_attempts": 5,
                "session_timeout_minutes": 120,
                "require_mfa": False
            },
            "notifications": {
                "email": True,
                "push": True,
                "in_app": True,
                "digest_frequency": "daily"
            },
            "privacy": {
                "profile_visibility": "public",
                "activity_visibility": "friends",
                "data_sharing": "essential_only"
            }
        }
    
    @classmethod
    def create_admin(
        cls,
        username: str,
        email: str,
        password: str,
        **kwargs
    ) -> 'User':
        """Create an admin user."""
        return cls.create(
            username=username,
            email=email,
            password=password,
            role=UserRole.ADMIN,
            status=UserStatus.ACTIVE,
            email_verified=True,
            tags=["admin", "staff"],
            **kwargs
        )
    
    @classmethod
    def create_with_oauth(
        cls,
        username: str,
        email: str,
        auth_provider: AuthProvider,
        auth_provider_id: str,
        **kwargs
    ) -> 'User':
        """Create a user with OAuth authentication."""
        return cls.create(
            username=username,
            email=email,
            auth_provider=auth_provider,
            auth_provider_id=auth_provider_id,
            email_verified=True,  # OAuth providers typically verify email
            status=UserStatus.ACTIVE,
            **kwargs
        )


class AuthToken(Base, UUIDMixin, TimestampMixin):
    """
    Authentication token model for API and session tokens.
    
    This model stores authentication tokens (JWT, API keys, etc.)
    for user authentication and authorization.
    
    Attributes:
        id: Primary key UUID
        user_id: User ID
        token_type: Type of token
        token: Token value
        refresh_token: Refresh token
        expires_at: Token expiration
        revoked_at: When token was revoked
        last_used_at: When token was last used
        user_agent: User agent for token
        ip_address: IP address for token
        scopes: Token scopes/permissions
        metadata: Additional metadata
    """
    
    __tablename__ = "auth_tokens"
    
    # User
    user_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # Token details
    token_type = Column(String(50), nullable=False, index=True)  # jwt, api_key, refresh, etc.
    token = Column(String(500), nullable=False, unique=True, index=True)
    refresh_token = Column(String(500), nullable=True, unique=True, index=True)
    
    # Token status
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True, index=True)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    
    # Context
    user_agent = Column(Text, nullable=True)
    ip_address = Column(String(50), nullable=True)
    
    # Permissions
    scopes = Column(ARRAY(String), default=[], nullable=False)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="auth_tokens")
    
    # Check constraints
    __table_args__ = (
        Index('ix_auth_tokens_user_type', 'user_id', 'token_type'),
        Index('ix_auth_tokens_expiry', 'expires_at', 'revoked_at'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<AuthToken(id={self.id}, user={self.user_id}, type={self.token_type})>"
    
    @property
    def is_expired(self) -> bool:
        """Check if token is expired."""
        return datetime.utcnow() > self.expires_at
    
    @property
    def is_revoked(self) -> bool:
        """Check if token is revoked."""
        return self.revoked_at is not None
    
    @property
    def is_valid(self) -> bool:
        """Check if token is valid (not expired and not revoked)."""
        return not self.is_expired and not self.is_revoked
    
    @property
    def seconds_until_expiry(self) -> float:
        """Get seconds until token expiry."""
        remaining = self.expires_at - datetime.utcnow()
        return max(0, remaining.total_seconds())
    
    @property
    def token_age_seconds(self) -> float:
        """Get token age in seconds."""
        age = datetime.utcnow() - self.created_at
        return age.total_seconds()
    
    def revoke(self) -> None:
        """Revoke the token."""
        self.revoked_at = datetime.utcnow()
    
    def renew(self, new_expires_at: datetime) -> None:
        """Renew the token with new expiration."""
        if self.is_revoked:
            raise ValueError("Cannot renew a revoked token")
        
        self.expires_at = new_expires_at
    
    def record_usage(self, ip_address: Optional[str] = None) -> None:
        """Record token usage."""
        self.last_used_at = datetime.utcnow()
        if ip_address:
            self.ip_address = ip_address
    
    def has_scope(self, scope: str) -> bool:
        """Check if token has specific scope."""
        return scope in self.scopes or "all" in self.scopes
    
    def to_dict(self, include_token: bool = False) -> Dict[str, Any]:
        """Convert auth token to dictionary."""
        result = {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "token_type": self.token_type,
            "expires_at": self.expires_at.isoformat(),
            "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "user_agent": self.user_agent,
            "ip_address": self.ip_address,
            "scopes": self.scopes,
            "is_expired": self.is_expired,
            "is_revoked": self.is_revoked,
            "is_valid": self.is_valid,
            "seconds_until_expiry": self.seconds_until_expiry,
            "token_age_seconds": self.token_age_seconds,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_token:
            result["token"] = self.token
            result["refresh_token"] = self.refresh_token
        
        return result


class OAuthAccount(Base, UUIDMixin, TimestampMixin):
    """
    OAuth account model for social login integration.
    
    This model stores OAuth account connections for users.
    
    Attributes:
        id: Primary key UUID
        user_id: User ID
        provider: OAuth provider
        provider_id: Provider's user ID
        provider_email: Email from provider
        access_token: OAuth access token
        refresh_token: OAuth refresh token
        token_expires_at: Token expiration
        scopes: Granted scopes
        profile_data: User profile data from provider
        metadata: Additional metadata
    """
    
    __tablename__ = "oauth_accounts"
    
    # User and provider
    user_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    provider = Column(SQLEnum(AuthProvider), nullable=False, index=True)
    provider_id = Column(String(255), nullable=False, index=True)
    provider_email = Column(String(255), nullable=True)
    
    # OAuth tokens
    access_token = Column(String(2000), nullable=True)  # Encrypted in production
    refresh_token = Column(String(2000), nullable=True)  # Encrypted in production
    token_expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # OAuth data
    scopes = Column(ARRAY(String), default=[], nullable=False)
    profile_data = Column(JSONB, default=dict, nullable=False)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="oauth_accounts")
    
    # Check constraints
    __table_args__ = (
        UniqueConstraint('provider', 'provider_id', name='uq_oauth_provider_account'),
        UniqueConstraint('user_id', 'provider', name='uq_user_oauth_provider'),
        Index('ix_oauth_accounts_provider_email', 'provider', 'provider_email'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<OAuthAccount(id={self.id}, user={self.user_id}, provider={self.provider.value})>"
    
    @property
    def is_token_expired(self) -> bool:
        """Check if OAuth token is expired."""
        if not self.token_expires_at:
            return False
        return datetime.utcnow() > self.token_expires_at
    
    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return self.provider.value.title()
    
    def update_tokens(
        self,
        access_token: str,
        refresh_token: Optional[str] = None,
        expires_in: Optional[int] = None
    ) -> None:
        """Update OAuth tokens."""
        self.access_token = access_token
        if refresh_token:
            self.refresh_token = refresh_token
        if expires_in:
            self.token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
    
    def to_dict(self, include_tokens: bool = False) -> Dict[str, Any]:
        """Convert OAuth account to dictionary."""
        result = {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "provider": self.provider.value,
            "provider_name": self.provider_name,
            "provider_id": self.provider_id,
            "provider_email": self.provider_email,
            "token_expires_at": self.token_expires_at.isoformat() if self.token_expires_at else None,
            "is_token_expired": self.is_token_expired,
            "scopes": self.scopes,
            "profile_data": self.profile_data,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_tokens:
            result["access_token"] = self.access_token
            result["refresh_token"] = self.refresh_token
        
        return result


class MfaDevice(Base, UUIDMixin, TimestampMixin):
    """
    MFA device model for multi-factor authentication.
    
    This model stores MFA devices and configurations for users.
    
    Attributes:
        id: Primary key UUID
        user_id: User ID
        device_type: Type of MFA device
        device_name: Device name/identifier
        secret: Encrypted MFA secret
        backup_codes: Backup recovery codes
        last_used_at: When device was last used
        is_verified: Whether device is verified
        metadata: Additional metadata
    """
    
    __tablename__ = "mfa_devices"
    
    # User and device
    user_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    device_type = Column(SQLEnum(MfaMethod), nullable=False, index=True)
    device_name = Column(String(200), nullable=False)
    
    # MFA configuration
    secret = Column(String(500), nullable=True)  # Encrypted secret (TOTP, etc.)
    backup_codes = Column(ARRAY(String), default=[], nullable=False)
    
    # Status
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    is_verified = Column(Boolean, default=False, nullable=False, index=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="mfa_devices")
    
    # Check constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'device_name', name='uq_user_mfa_device'),
        Index('ix_mfa_devices_type_user', 'device_type', 'user_id'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<MfaDevice(id={self.id}, user={self.user_id}, type={self.device_type.value}, name={self.device_name})>"
    
    @property
    def is_totp_device(self) -> bool:
        """Check if device is TOTP (Google Authenticator)."""
        return self.device_type == MfaMethod.TOTP
    
    @property
    def is_sms_device(self) -> bool:
        """Check if device is SMS-based."""
        return self.device_type == MfaMethod.SMS
    
    @property
    def days_since_last_use(self) -> Optional[int]:
        """Get days since last use."""
        if not self.last_used_at:
            return None
        delta = datetime.utcnow() - self.last_used_at
        return delta.days
    
    def record_usage(self) -> None:
        """Record device usage."""
        self.last_used_at = datetime.utcnow()
    
    def verify(self) -> None:
        """Verify the MFA device."""
        self.is_verified = True
    
    def generate_backup_codes(self, count: int = 10) -> List[str]:
        """Generate backup recovery codes."""
        import secrets
        
        codes = []
        for _ in range(count):
            # Generate 8-character alphanumeric code
            code = secrets.token_urlsafe(8)[:8].upper()
            codes.append(code)
        
        self.backup_codes = codes
        return codes
    
    def verify_backup_code(self, code: str) -> bool:
        """Verify a backup code."""
        if code in self.backup_codes:
            # Remove used code
            self.backup_codes = [c for c in self.backup_codes if c != code]
            return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert MFA device to dictionary."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "device_type": self.device_type.value,
            "device_name": self.device_name,
            "is_totp_device": self.is_totp_device,
            "is_sms_device": self.is_sms_device,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "days_since_last_use": self.days_since_last_use,
            "is_verified": self.is_verified,
            "has_backup_codes": len(self.backup_codes) > 0,
            "backup_codes_count": len(self.backup_codes),
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class BackupCode(Base, UUIDMixin, TimestampMixin):
    """
    Backup code model for MFA recovery.
    
    This model stores individual backup codes for MFA recovery.
    
    Attributes:
        id: Primary key UUID
        user_id: User ID
        code: Backup code (hashed)
        used_at: When code was used
        expires_at: When code expires
        metadata: Additional metadata
    """
    
    __tablename__ = "backup_codes"
    
    # User
    user_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # Code
    code = Column(String(255), nullable=False, unique=True, index=True)  # Hashed code
    used_at = Column(DateTime(timezone=True), nullable=True, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="backup_codes")
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<BackupCode(id={self.id}, user={self.user_id})>"
    
    @property
    def is_used(self) -> bool:
        """Check if code has been used."""
        return self.used_at is not None
    
    @property
    def is_expired(self) -> bool:
        """Check if code is expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    @property
    def is_valid(self) -> bool:
        """Check if code is valid (not used and not expired)."""
        return not self.is_used and not self.is_expired
    
    def mark_used(self) -> None:
        """Mark code as used."""
        self.used_at = datetime.utcnow()


class PasswordReset(Base, UUIDMixin, TimestampMixin):
    """
    Password reset model for account recovery.
    
    This model stores password reset requests and tokens.
    
    Attributes:
        id: Primary key UUID
        user_id: User ID
        token: Reset token
        expires_at: When token expires
        used_at: When token was used
        ip_address: Request IP address
        user_agent: Request user agent
        metadata: Additional metadata
    """
    
    __tablename__ = "password_resets"
    
    # User
    user_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # Token
    token = Column(String(255), nullable=False, unique=True, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    used_at = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # Context
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(Text, nullable=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="password_resets")
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<PasswordReset(id={self.id}, user={self.user_id})>"
    
    @property
    def is_expired(self) -> bool:
        """Check if token is expired."""
        return datetime.utcnow() > self.expires_at
    
    @property
    def is_used(self) -> bool:
        """Check if token has been used."""
        return self.used_at is not None
    
    @property
    def is_valid(self) -> bool:
        """Check if token is valid (not used and not expired)."""
        return not self.is_used and not self.is_expired
    
    @property
    def seconds_until_expiry(self) -> float:
        """Get seconds until token expiry."""
        remaining = self.expires_at - datetime.utcnow()
        return max(0, remaining.total_seconds())
    
    def mark_used(self) -> None:
        """Mark token as used."""
        self.used_at = datetime.utcnow()


class UserSession(Base, UUIDMixin, TimestampMixin):
    """
    User session model for tracking active sessions.
    
    This model stores user sessions for session management
    and security monitoring.
    
    Attributes:
        id: Primary key UUID
        user_id: User ID
        session_id: Session identifier
        ip_address: Session IP address
        user_agent: Session user agent
        device_info: Device information
        location_info: Geographic location
        last_activity_at: Last activity timestamp
        expires_at: Session expiration
        terminated_at: When session was terminated
        metadata: Additional metadata
    """
    
    __tablename__ = "user_sessions"
    
    # User
    user_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # Session details
    session_id = Column(String(255), nullable=False, unique=True, index=True)
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(Text, nullable=True)
    device_info = Column(JSONB, nullable=True)
    location_info = Column(JSONB, nullable=True)
    
    # Session status
    last_activity_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    terminated_at = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    
    # Check constraints
    __table_args__ = (
        CheckConstraint('expires_at > created_at', name='check_session_expiry_valid'),
        Index('ix_user_sessions_activity', 'user_id', 'last_activity_at'),
        Index('ix_user_sessions_expiry', 'expires_at', 'terminated_at'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<UserSession(id={self.id}, user={self.user_id}, session={self.session_id})>"
    
    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return (
            not self.terminated_at and
            not self.is_expired and
            datetime.utcnow() <= self.expires_at
        )
    
    @property
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.utcnow() > self.expires_at
    
    @property
    def is_terminated(self) -> bool:
        """Check if session was terminated."""
        return self.terminated_at is not None
    
    @property
    def duration_seconds(self) -> float:
        """Get session duration in seconds."""
        end_time = self.terminated_at or datetime.utcnow()
        duration = end_time - self.created_at
        return duration.total_seconds()
    
    @property
    def inactivity_seconds(self) -> float:
        """Get seconds since last activity."""
        inactivity = datetime.utcnow() - self.last_activity_at
        return inactivity.total_seconds()
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity_at = datetime.utcnow()
    
    def terminate(self) -> None:
        """Terminate the session."""
        self.terminated_at = datetime.utcnow()
    
    def renew(self, additional_seconds: int = 3600) -> None:
        """Renew session expiration."""
        if self.is_terminated:
            raise ValueError("Cannot renew a terminated session")
        
        self.expires_at = datetime.utcnow() + timedelta(seconds=additional_seconds)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user session to dictionary."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "device_info": self.device_info,
            "location_info": self.location_info,
            "last_activity_at": self.last_activity_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "terminated_at": self.terminated_at.isoformat() if self.terminated_at else None,
            "is_active": self.is_active,
            "is_expired": self.is_expired,
            "is_terminated": self.is_terminated,
            "duration_seconds": self.duration_seconds,
            "inactivity_seconds": self.inactivity_seconds,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class LoginHistory(Base, UUIDMixin, TimestampMixin):
    """
    Login history model for tracking authentication events.
    
    This model stores detailed login history for security monitoring
    and audit purposes.
    
    Attributes:
        id: Primary key UUID
        user_id: User ID
        auth_method: Authentication method used
        ip_address: Login IP address
        user_agent: Login user agent
        success: Whether login was successful
        failure_reason: Reason for login failure
        location_info: Geographic location
        device_info: Device information
        metadata: Additional metadata
    """
    
    __tablename__ = "login_history"
    
    # User
    user_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # Login details
    auth_method = Column(String(100), nullable=False, index=True)
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(Text, nullable=True)
    success = Column(Boolean, default=False, nullable=False, index=True)
    failure_reason = Column(String(200), nullable=True)
    
    # Additional info
    location_info = Column(JSONB, nullable=True)
    device_info = Column(JSONB, nullable=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="login_history")
    
    # Check constraints
    __table_args__ = (
        Index('ix_login_history_user_date', 'user_id', 'created_at'),
        Index('ix_login_history_success_date', 'success', 'created_at'),
        Index('ix_login_history_ip_date', 'ip_address', 'created_at'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        status = "success" if self.success else f"failed: {self.failure_reason}"
        return f"<LoginHistory(id={self.id}, user={self.user_id}, auth={self.auth_method}, status={status})>"
    
    @property
    def is_failed(self) -> bool:
        """Check if login failed."""
        return not self.success
    
    @classmethod
    def record_login(
        cls,
        user_id: uuid.UUID,
        auth_method: str,
        success: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        failure_reason: Optional[str] = None,
        location_info: Optional[Dict[str, Any]] = None,
        device_info: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'LoginHistory':
        """Record a login attempt."""
        return cls(
            user_id=user_id,
            auth_method=auth_method,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent,
            failure_reason=failure_reason,
            location_info=location_info,
            device_info=device_info,
            metadata=metadata or {}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert login history to dictionary."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "auth_method": self.auth_method,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "success": self.success,
            "is_failed": self.is_failed,
            "failure_reason": self.failure_reason,
            "location_info": self.location_info,
            "device_info": self.device_info,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


# Event listeners for automatic updates
@event.listens_for(User, 'before_update')
def update_user_timestamp(mapper, connection, target):
    """Update user updated_at timestamp."""
    target.updated_at = datetime.utcnow()


# Helper functions
def generate_secure_token(length: int = 32) -> str:
    """Generate a secure random token."""
    return secrets.token_urlsafe(length)


def generate_otp_code(length: int = 6) -> str:
    """Generate a numeric OTP code."""
    import random
    digits = "0123456789"
    return ''.join(random.choice(digits) for _ in range(length))


def hash_backup_code(code: str) -> str:
    """Hash a backup code for secure storage."""
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256(f"{code}{salt}".encode()).hexdigest()
    return f"sha256${salt}${hashed}"


def verify_backup_code_hash(code: str, hashed_code: str) -> bool:
    """Verify a backup code against its hash."""
    try:
        algorithm, salt, stored_hash = hashed_code.split('$')
        if algorithm != 'sha256':
            return False
        
        test_hash = hashlib.sha256(f"{code}{salt}".encode()).hexdigest()
        return secrets.compare_digest(test_hash, stored_hash)
    except (ValueError, AttributeError):
        return False


def validate_password_strength(password: str, user_settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Validate password strength against policy."""
    results = {
        "valid": True,
        "errors": [],
        "score": 0,
        "strength": "weak"
    }
    
    # Default policy
    policy = {
        "min_length": 8,
        "require_uppercase": True,
        "require_lowercase": True,
        "require_numbers": True,
        "require_special": True
    }
    
    # Override with user settings if provided
    if user_settings and 'password_policy' in user_settings.get('security', {}):
        policy.update(user_settings['security']['password_policy'])
    
    # Check length
    if len(password) < policy['min_length']:
        results["valid"] = False
        results["errors"].append(f"Password must be at least {policy['min_length']} characters")
    
    # Check uppercase
    if policy['require_uppercase'] and not any(c.isupper() for c in password):
        results["valid"] = False
        results["errors"].append("Password must contain at least one uppercase letter")
    
    # Check lowercase
    if policy['require_lowercase'] and not any(c.islower() for c in password):
        results["valid"] = False
        results["errors"].append("Password must contain at least one lowercase letter")
    
    # Check numbers
    if policy['require_numbers'] and not any(c.isdigit() for c in password):
        results["valid"] = False
        results["errors"].append("Password must contain at least one number")
    
    # Check special characters
    if policy['require_special']:
        special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?~"
        if not any(c in special_chars for c in password):
            results["valid"] = False
            results["errors"].append(f"Password must contain at least one special character")
    
    # Calculate score (0-100)
    score = 0
    
    # Length contributes up to 30 points
    length_score = min(30, len(password) * 2)
    score += length_score
    
    # Character variety contributes up to 40 points
    variety_score = 0
    if any(c.isupper() for c in password):
        variety_score += 10
    if any(c.islower() for c in password):
        variety_score += 10
    if any(c.isdigit() for c in password):
        variety_score += 10
    if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?~" for c in password):
        variety_score += 10
    score += variety_score
    
    # Entropy/randomness contributes up to 30 points
    # Simple check for repeated patterns
    import re
    repeated_patterns = re.findall(r'(.)\1{2,}', password)
    if not repeated_patterns:
        score += 30
    
    results["score"] = min(100, score)
    
    # Determine strength
    if score >= 80:
        results["strength"] = "strong"
    elif score >= 60:
        results["strength"] = "good"
    elif score >= 40:
        results["strength"] = "fair"
    else:
        results["strength"] = "weak"
    
    return results