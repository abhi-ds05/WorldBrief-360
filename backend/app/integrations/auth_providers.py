# backend/app/integrations/auth_providers.py
"""
OAuth2 authentication providers for WorldBrief 360.
Supports Google, GitHub, and other OAuth providers for user authentication.
"""

import asyncio
import json
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlencode, urljoin

import aiohttp
from pydantic import BaseModel, Field, validator, HttpUrl
from authlib.integrations.starlette_client import OAuth
from authlib.jose import jwt
from authlib.oidc.core import UserInfo
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging_config import logger
from app.core.security import create_access_token, verify_password_hash
from app.db.models.user import User
from app.schemas.request.auth import OAuthLoginRequest
from app.schemas.response.auth import OAuthTokenResponse, UserProfile


class OAuthProviderType(Enum):
    """Supported OAuth providers."""
    GOOGLE = "google"
    GITHUB = "github"
    FACEBOOK = "facebook"
    TWITTER = "twitter"
    MICROSOFT = "microsoft"
    LINKEDIN = "linkedin"
    APPLE = "apple"


@dataclass
class OAuthProviderConfig:
    """Configuration for an OAuth provider."""
    name: str
    client_id: str
    client_secret: str
    authorize_url: str
    access_token_url: str
    api_base_url: str
    userinfo_endpoint: str
    scope: List[str]
    jwks_uri: Optional[str] = None
    issuer: Optional[str] = None
    enabled: bool = True


class OAuthUserInfo(BaseModel):
    """Standardized OAuth user information."""
    provider: OAuthProviderType
    provider_user_id: str
    email: str
    email_verified: bool = False
    name: Optional[str] = None
    given_name: Optional[str] = None
    family_name: Optional[str] = None
    picture: Optional[str] = None
    locale: Optional[str] = None
    timezone: Optional[str] = None
    raw_data: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('email')
    def validate_email(cls, v):
        if not v or '@' not in v:
            raise ValueError('Invalid email address')
        return v.lower()


class OAuthState(BaseModel):
    """OAuth state for CSRF protection."""
    state: str
    redirect_uri: str
    provider: OAuthProviderType
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: datetime = Field(default_factory=lambda: datetime.now() + timedelta(minutes=10))
    
    def is_expired(self) -> bool:
        """Check if state has expired."""
        return datetime.now() > self.expires_at


class BaseOAuthProvider:
    """Base class for OAuth providers."""
    
    def __init__(
        self,
        provider_type: OAuthProviderType,
        config: OAuthProviderConfig,
        session_storage: Optional[Dict[str, Any]] = None
    ):
        self.provider_type = provider_type
        self.config = config
        self.session_storage = session_storage or {}
        self.oauth_client = None
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize OAuth client."""
        if not self.config.enabled:
            logger.warning(f"OAuth provider {self.provider_type.value} is disabled")
            return
            
        self.oauth_client = OAuth().register(
            name=self.provider_type.value,
            client_id=self.config.client_id,
            client_secret=self.config.client_secret,
            authorize_url=self.config.authorize_url,
            access_token_url=self.config.access_token_url,
            api_base_url=self.config.api_base_url,
            userinfo_endpoint=self.config.userinfo_endpoint,
            jwks_uri=self.config.jwks_uri,
            issuer=self.config.issuer,
            client_kwargs={
                'scope': ' '.join(self.config.scope),
                'token_endpoint_auth_method': 'client_secret_post',
            }
        )
    
    def is_enabled(self) -> bool:
        """Check if provider is enabled."""
        return self.config.enabled and self.oauth_client is not None
    
    def generate_authorization_url(
        self,
        redirect_uri: str,
        state: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, str]:
        """
        Generate OAuth authorization URL.
        
        Args:
            redirect_uri: Callback URL
            state: Optional state parameter (auto-generated if None)
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (authorization_url, state)
        """
        if not self.is_enabled():
            raise ValueError(f"OAuth provider {self.provider_type.value} is disabled")
            
        state = state or secrets.token_urlsafe(32)
        
        params = {
            'redirect_uri': redirect_uri,
            'state': state,
            **kwargs
        }
        
        # Store state in session storage
        self.session_storage[f"oauth_state:{state}"] = OAuthState(
            state=state,
            redirect_uri=redirect_uri,
            provider=self.provider_type
        )
        
        authorization_url = self.oauth_client.authorize_redirect(**params).url
        return authorization_url, state
    
    async def validate_state(self, state: str) -> bool:
        """
        Validate OAuth state.
        
        Args:
            state: State parameter from callback
            
        Returns:
            True if state is valid, False otherwise
        """
        state_key = f"oauth_state:{state}"
        
        if state_key not in self.session_storage:
            logger.warning(f"Invalid state: {state}")
            return False
            
        oauth_state = self.session_storage[state_key]
        
        if oauth_state.is_expired():
            logger.warning(f"Expired state: {state}")
            del self.session_storage[state_key]
            return False
            
        # Clean up used state
        del self.session_storage[state_key]
        return True
    
    async def exchange_code_for_token(
        self,
        code: str,
        redirect_uri: str
    ) -> Dict[str, Any]:
        """
        Exchange authorization code for access token.
        
        Args:
            code: Authorization code
            redirect_uri: Callback URL
            
        Returns:
            Token response
        """
        if not self.is_enabled():
            raise ValueError(f"OAuth provider {self.provider_type.value} is disabled")
            
        try:
            token = await self.oauth_client.authorize_access_token(
                code=code,
                redirect_uri=redirect_uri
            )
            return token
            
        except Exception as e:
            logger.error(f"Failed to exchange code for token: {str(e)}")
            raise
    
    async def get_user_info(self, token: Dict[str, Any]) -> OAuthUserInfo:
        """
        Get user information from OAuth provider.
        
        Args:
            token: Access token
            
        Returns:
            OAuthUserInfo object
        """
        if not self.is_enabled():
            raise ValueError(f"OAuth provider {self.provider_type.value} is disabled")
            
        try:
            userinfo = await self.oauth_client.userinfo(token=token)
            return self._normalize_user_info(userinfo)
            
        except Exception as e:
            logger.error(f"Failed to get user info: {str(e)}")
            raise
    
    def _normalize_user_info(self, userinfo: UserInfo) -> OAuthUserInfo:
        """
        Normalize user info from provider-specific format.
        
        Args:
            userinfo: Raw user info from provider
            
        Returns:
            Normalized OAuthUserInfo
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    async def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """
        Refresh access token.
        
        Args:
            refresh_token: Refresh token
            
        Returns:
            New token response
        """
        if not self.is_enabled():
            raise ValueError(f"OAuth provider {self.provider_type.value} is disabled")
            
        try:
            # Note: Not all providers support token refresh
            # This is a placeholder implementation
            raise NotImplementedError(
                f"Token refresh not implemented for {self.provider_type.value}"
            )
            
        except Exception as e:
            logger.error(f"Failed to refresh token: {str(e)}")
            raise


class GoogleOAuth(BaseOAuthProvider):
    """Google OAuth provider."""
    
    def __init__(self, config: Optional[OAuthProviderConfig] = None):
        if config is None:
            config = OAuthProviderConfig(
                name="google",
                client_id=settings.GOOGLE_OAUTH_CLIENT_ID,
                client_secret=settings.GOOGLE_OAUTH_CLIENT_SECRET,
                authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
                access_token_url="https://oauth2.googleapis.com/token",
                api_base_url="https://www.googleapis.com/",
                userinfo_endpoint="https://openidconnect.googleapis.com/v1/userinfo",
                jwks_uri="https://www.googleapis.com/oauth2/v3/certs",
                issuer="https://accounts.google.com",
                scope=[
                    "openid",
                    "email",
                    "profile",
                    "https://www.googleapis.com/auth/user.birthday.read",
                    "https://www.googleapis.com/auth/user.gender.read"
                ],
                enabled=bool(settings.GOOGLE_OAUTH_CLIENT_ID)
            )
        
        super().__init__(OAuthProviderType.GOOGLE, config)
    
    def _normalize_user_info(self, userinfo: UserInfo) -> OAuthUserInfo:
        """Normalize Google user info."""
        return OAuthUserInfo(
            provider=self.provider_type,
            provider_user_id=userinfo.get("sub"),
            email=userinfo.get("email"),
            email_verified=userinfo.get("email_verified", False),
            name=userinfo.get("name"),
            given_name=userinfo.get("given_name"),
            family_name=userinfo.get("family_name"),
            picture=userinfo.get("picture"),
            locale=userinfo.get("locale"),
            raw_data=dict(userinfo)
        )
    
    async def validate_id_token(self, id_token: str) -> Dict[str, Any]:
        """
        Validate Google ID token.
        
        Args:
            id_token: JWT ID token
            
        Returns:
            Decoded token claims
        """
        try:
            claims = jwt.decode(
                id_token,
                key=self.config.jwks_uri,
                claims_options={
                    'iss': {'value': self.config.issuer},
                    'aud': {'value': self.config.client_id},
                }
            )
            claims.validate()
            return claims
            
        except Exception as e:
            logger.error(f"Failed to validate Google ID token: {str(e)}")
            raise


class GitHubOAuth(BaseOAuthProvider):
    """GitHub OAuth provider."""
    
    def __init__(self, config: Optional[OAuthProviderConfig] = None):
        if config is None:
            config = OAuthProviderConfig(
                name="github",
                client_id=settings.GITHUB_OAUTH_CLIENT_ID,
                client_secret=settings.GITHUB_OAUTH_CLIENT_SECRET,
                authorize_url="https://github.com/login/oauth/authorize",
                access_token_url="https://github.com/login/oauth/access_token",
                api_base_url="https://api.github.com/",
                userinfo_endpoint="https://api.github.com/user",
                scope=["user:email", "read:user"],
                enabled=bool(settings.GITHUB_OAUTH_CLIENT_ID)
            )
        
        super().__init__(OAuthProviderType.GITHUB, config)
    
    def _normalize_user_info(self, userinfo: UserInfo) -> OAuthUserInfo:
        """Normalize GitHub user info."""
        email = userinfo.get("email")
        
        # GitHub doesn't always return email in userinfo
        # May need to fetch it separately
        if not email and userinfo.get("login"):
            # Try to construct a GitHub no-reply email
            email = f"{userinfo.get('login')}@users.noreply.github.com"
        
        return OAuthUserInfo(
            provider=self.provider_type,
            provider_user_id=str(userinfo.get("id")),
            email=email or "",
            email_verified=False,  # GitHub doesn't provide email verification status
            name=userinfo.get("name"),
            given_name=userinfo.get("given_name"),
            family_name=userinfo.get("family_name"),
            picture=userinfo.get("avatar_url"),
            locale=userinfo.get("locale"),
            raw_data=dict(userinfo)
        )
    
    async def get_user_emails(self, token: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get user emails from GitHub.
        
        Args:
            token: Access token
            
        Returns:
            List of email addresses
        """
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"token {token['access_token']}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            async with session.get(
                "https://api.github.com/user/emails",
                headers=headers
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to fetch GitHub emails: {response.status}")
                    return []


class FacebookOAuth(BaseOAuthProvider):
    """Facebook OAuth provider."""
    
    def __init__(self, config: Optional[OAuthProviderConfig] = None):
        if config is None:
            config = OAuthProviderConfig(
                name="facebook",
                client_id=settings.FACEBOOK_OAUTH_CLIENT_ID,
                client_secret=settings.FACEBOOK_OAUTH_CLIENT_SECRET,
                authorize_url="https://www.facebook.com/v12.0/dialog/oauth",
                access_token_url="https://graph.facebook.com/v12.0/oauth/access_token",
                api_base_url="https://graph.facebook.com/",
                userinfo_endpoint="https://graph.facebook.com/me",
                scope=["email", "public_profile"],
                enabled=bool(settings.FACEBOOK_OAUTH_CLIENT_ID)
            )
        
        super().__init__(OAuthProviderType.FACEBOOK, config)
    
    def _normalize_user_info(self, userinfo: UserInfo) -> OAuthUserInfo:
        """Normalize Facebook user info."""
        return OAuthUserInfo(
            provider=self.provider_type,
            provider_user_id=userinfo.get("id"),
            email=userinfo.get("email", ""),
            email_verified=False,  # Facebook doesn't provide verification status
            name=userinfo.get("name"),
            given_name=userinfo.get("first_name"),
            family_name=userinfo.get("last_name"),
            picture=self._get_profile_picture(userinfo.get("id")),
            locale=userinfo.get("locale"),
            raw_data=dict(userinfo)
        )
    
    def _get_profile_picture(self, user_id: str) -> Optional[str]:
        """Get Facebook profile picture URL."""
        if not user_id:
            return None
        return f"https://graph.facebook.com/{user_id}/picture?type=large"


class OAuthProviderManager:
    """Manager for all OAuth providers."""
    
    def __init__(self):
        self.providers: Dict[OAuthProviderType, BaseOAuthProvider] = {}
        self.session_storage = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all configured OAuth providers."""
        providers = [
            (OAuthProviderType.GOOGLE, GoogleOAuth),
            (OAuthProviderType.GITHUB, GitHubOAuth),
            (OAuthProviderType.FACEBOOK, FacebookOAuth),
        ]
        
        for provider_type, provider_class in providers:
            try:
                provider = provider_class()
                if provider.is_enabled():
                    self.providers[provider_type] = provider
                    logger.info(f"Initialized OAuth provider: {provider_type.value}")
                else:
                    logger.info(f"Skipping disabled OAuth provider: {provider_type.value}")
            except Exception as e:
                logger.error(f"Failed to initialize {provider_type.value}: {str(e)}")
    
    def get_provider(self, provider_type: Union[str, OAuthProviderType]) -> BaseOAuthProvider:
        """
        Get OAuth provider by type.
        
        Args:
            provider_type: Provider type string or enum
            
        Returns:
            OAuth provider instance
            
        Raises:
            ValueError: If provider not found or disabled
        """
        if isinstance(provider_type, str):
            try:
                provider_type = OAuthProviderType(provider_type.lower())
            except ValueError:
                raise ValueError(f"Unknown OAuth provider: {provider_type}")
        
        provider = self.providers.get(provider_type)
        
        if not provider:
            raise ValueError(f"OAuth provider {provider_type.value} not available")
        
        return provider
    
    def get_available_providers(self) -> List[Dict[str, Any]]:
        """
        Get list of available OAuth providers.
        
        Returns:
            List of provider information
        """
        available = []
        
        for provider_type, provider in self.providers.items():
            if provider.is_enabled():
                available.append({
                    "provider": provider_type.value,
                    "name": provider_type.value.title(),
                    "enabled": True,
                    "scope": provider.config.scope
                })
        
        return available
    
    async def handle_oauth_callback(
        self,
        provider_type: str,
        code: str,
        state: str,
        redirect_uri: str
    ) -> Tuple[OAuthUserInfo, OAuthTokenResponse]:
        """
        Handle OAuth callback and return user info and tokens.
        
        Args:
            provider_type: OAuth provider type
            code: Authorization code
            state: State parameter
            redirect_uri: Callback URL
            
        Returns:
            Tuple of (user_info, token_response)
        """
        # Get provider
        provider = self.get_provider(provider_type)
        
        # Validate state
        if not await provider.validate_state(state):
            raise ValueError("Invalid or expired OAuth state")
        
        # Exchange code for token
        token = await provider.exchange_code_for_token(code, redirect_uri)
        
        # Get user info
        user_info = await provider.get_user_info(token)
        
        # Create token response
        token_response = OAuthTokenResponse(
            access_token=token.get("access_token"),
            refresh_token=token.get("refresh_token"),
            id_token=token.get("id_token"),
            token_type=token.get("token_type", "Bearer"),
            expires_in=token.get("expires_in", 3600),
            scope=token.get("scope", ""),
            provider=provider_type
        )
        
        return user_info, token_response
    
    async def link_existing_account(
        self,
        user: User,
        provider_type: str,
        user_info: OAuthUserInfo
    ) -> User:
        """
        Link OAuth account to existing user.
        
        Args:
            user: Existing user
            provider_type: OAuth provider type
            user_info: OAuth user info
            
        Returns:
            Updated user
        """
        # Check if OAuth account is already linked
        existing_link = await self._get_oauth_link(user, provider_type)
        
        if existing_link:
            # Update existing link
            existing_link.provider_user_id = user_info.provider_user_id
            existing_link.email = user_info.email
            existing_link.raw_data = user_info.raw_data
            existing_link.updated_at = datetime.now()
        else:
            # Create new link
            from app.db.models.user import OAuthAccount
            oauth_account = OAuthAccount(
                user_id=user.id,
                provider=provider_type,
                provider_user_id=user_info.provider_user_id,
                email=user_info.email,
                raw_data=user_info.raw_data
            )
            user.oauth_accounts.append(oauth_account)
        
        return user
    
    async def create_user_from_oauth(
        self,
        user_info: OAuthUserInfo,
        session
    ) -> User:
        """
        Create new user from OAuth info.
        
        Args:
            user_info: OAuth user info
            session: Database session
            
        Returns:
            Newly created user
        """
        from app.db.models.user import User, OAuthAccount
        
        # Generate username from email or name
        username = self._generate_username(user_info)
        
        # Create user
        user = User(
            email=user_info.email,
            username=username,
            full_name=user_info.name or "",
            is_active=True,
            email_verified=user_info.email_verified,
            avatar_url=user_info.picture
        )
        
        # Create OAuth account link
        oauth_account = OAuthAccount(
            provider=user_info.provider.value,
            provider_user_id=user_info.provider_user_id,
            email=user_info.email,
            raw_data=user_info.raw_data
        )
        
        user.oauth_accounts.append(oauth_account)
        session.add(user)
        
        return user
    
    def _generate_username(self, user_info: OAuthUserInfo) -> str:
        """Generate unique username from OAuth info."""
        base_name = ""
        
        if user_info.email:
            base_name = user_info.email.split('@')[0]
        elif user_info.name:
            base_name = user_info.name.lower().replace(' ', '_')
        else:
            base_name = f"user_{user_info.provider_user_id[:8]}"
        
        # Clean username
        import re
        base_name = re.sub(r'[^a-z0-9_]', '', base_name.lower())
        
        # Add random suffix if needed
        if len(base_name) < 3:
            base_name = f"user_{secrets.token_hex(3)}"
        
        return base_name
    
    async def _get_oauth_link(
        self,
        user: User,
        provider_type: str
    ) -> Optional[Any]:
        """Get OAuth account link for user."""
        from app.db.models.user import OAuthAccount
        
        for account in user.oauth_accounts:
            if account.provider == provider_type:
                return account
        return None


class OAuthService:
    """Service for OAuth authentication operations."""
    
    def __init__(self, provider_manager: Optional[OAuthProviderManager] = None):
        self.provider_manager = provider_manager or OAuthProviderManager()
        self.db_session = None
    
    async def get_authorization_url(
        self,
        provider_type: str,
        redirect_uri: str,
        **kwargs
    ) -> Dict[str, str]:
        """
        Get OAuth authorization URL.
        
        Args:
            provider_type: OAuth provider type
            redirect_uri: Callback URL
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with authorization_url and state
        """
        provider = self.provider_manager.get_provider(provider_type)
        auth_url, state = provider.generate_authorization_url(
            redirect_uri=redirect_uri,
            **kwargs
        )
        
        return {
            "authorization_url": auth_url,
            "state": state,
            "provider": provider_type
        }
    
    async def authenticate(
        self,
        provider_type: str,
        code: str,
        state: str,
        redirect_uri: str,
        session
    ) -> Tuple[User, OAuthTokenResponse]:
        """
        Authenticate user via OAuth.
        
        Args:
            provider_type: OAuth provider type
            code: Authorization code
            state: State parameter
            redirect_uri: Callback URL
            session: Database session
            
        Returns:
            Tuple of (user, token_response)
        """
        # Handle OAuth callback
        user_info, token_response = await self.provider_manager.handle_oauth_callback(
            provider_type=provider_type,
            code=code,
            state=state,
            redirect_uri=redirect_uri
        )
        
        # Check if user exists by email
        from app.db.models.user import User
        user = await session.query(User).filter(
            User.email == user_info.email
        ).first()
        
        if user:
            # Link OAuth account to existing user
            user = await self.provider_manager.link_existing_account(
                user=user,
                provider_type=provider_type,
                user_info=user_info
            )
        else:
            # Create new user from OAuth
            user = await self.provider_manager.create_user_from_oauth(
                user_info=user_info,
                session=session
            )
        
        # Generate JWT token for our app
        access_token = create_access_token(
            data={"sub": str(user.id), "email": user.email}
        )
        
        # Add our JWT to the response
        token_response.app_access_token = access_token
        token_response.user_id = str(user.id)
        
        await session.commit()
        
        return user, token_response
    
    async def unlink_account(
        self,
        user: User,
        provider_type: str,
        session
    ) -> bool:
        """
        Unlink OAuth account from user.
        
        Args:
            user: User object
            provider_type: OAuth provider type
            session: Database session
            
        Returns:
            True if successful, False otherwise
        """
        from app.db.models.user import OAuthAccount
        
        oauth_account = await session.query(OAuthAccount).filter(
            OAuthAccount.user_id == user.id,
            OAuthAccount.provider == provider_type
        ).first()
        
        if oauth_account:
            await session.delete(oauth_account)
            await session.commit()
            return True
        
        return False
    
    async def get_linked_accounts(self, user: User) -> List[Dict[str, Any]]:
        """
        Get list of OAuth accounts linked to user.
        
        Args:
            user: User object
            
        Returns:
            List of linked account information
        """
        linked_accounts = []
        
        for account in user.oauth_accounts:
            linked_accounts.append({
                "provider": account.provider,
                "email": account.email,
                "created_at": account.created_at,
                "updated_at": account.updated_at
            })
        
        return linked_accounts


# Factory function for dependency injection
def get_oauth_service() -> OAuthService:
    """
    Factory function to create OAuth service.
    
    Returns:
        Configured OAuthService instance
    """
    return OAuthService()