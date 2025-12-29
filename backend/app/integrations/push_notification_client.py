# backend/app/integrations/push_notification_client.py
"""
Push notification client for sending mobile/web push notifications.
Supports Firebase Cloud Messaging (FCM), Apple Push Notification Service (APNS),
and Web Push Protocol.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from uuid import uuid4

import aiohttp
from pydantic import BaseModel, Field, validator

from app.core.config import settings
from app.core.logging_config import get_logger
from app.cache.redis_client import get_redis_client
from app.events.event_bus import EventBus, EventType

logger = get_logger(__name__)


# ============ Data Models ============

class PushNotificationType(str, Enum):
    """Types of push notifications."""
    BREAKING_NEWS = "breaking_news"
    INCIDENT_VERIFIED = "incident_verified"
    INCIDENT_NEARBY = "incident_nearby"
    BRIEFING_READY = "briefing_ready"
    VERIFICATION_REQUEST = "verification_request"
    COIN_REWARDED = "coin_rewarded"
    NEW_MESSAGE = "new_message"
    SYSTEM_ALERT = "system_alert"
    TOPIC_UPDATE = "topic_update"
    USER_MENTION = "user_mention"


class PushPlatform(str, Enum):
    """Supported push notification platforms."""
    FCM_ANDROID = "fcm_android"
    FCM_IOS = "fcm_ios"
    APNS = "apns"
    WEB_PUSH = "web_push"
    HUAWEI = "huawei_push"


class PushPriority(str, Enum):
    """Notification priority levels."""
    NORMAL = "normal"
    HIGH = "high"
    MAXIMUM = "maximum"


class PushNotification(BaseModel):
    """Push notification data model."""
    notification_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    title: str
    body: str
    data: Dict[str, Any] = Field(default_factory=dict)
    platform: PushPlatform
    device_token: str
    priority: PushPriority = PushPriority.NORMAL
    collapse_key: Optional[str] = None
    time_to_live: int = Field(default=86400, ge=0, le=2419200)  # seconds (28 days max)
    badge_count: Optional[int] = None
    sound: Optional[str] = None
    image_url: Optional[str] = None
    deep_link: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    notification_type: PushNotificationType = PushNotificationType.SYSTEM_ALERT
    
    @validator('title')
    def validate_title_length(cls, v):
        if len(v) > 100:
            raise ValueError('Title must be 100 characters or less')
        return v
    
    @validator('body')
    def validate_body_length(cls, v):
        if len(v) > 240:
            raise ValueError('Body must be 240 characters or less')
        return v


class BatchPushRequest(BaseModel):
    """Batch push notification request."""
    notifications: List[PushNotification]
    dry_run: bool = False


class PushResponse(BaseModel):
    """Push notification response."""
    notification_id: str
    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None
    platform_response: Optional[Dict[str, Any]] = None


class DeviceToken(BaseModel):
    """User device token registration."""
    user_id: str
    device_token: str
    platform: PushPlatform
    device_id: Optional[str] = None
    app_version: Optional[str] = None
    os_version: Optional[str] = None
    language: str = "en"
    timezone: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True


# ============ Abstract Base Classes ============

class BasePushProvider(ABC):
    """Abstract base class for push notification providers."""
    
    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        self.session: Optional[aiohttp.ClientSession] = None
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider (create session, load config)."""
        pass
    
    @abstractmethod
    async def send_single(self, notification: PushNotification) -> PushResponse:
        """Send a single push notification."""
        pass
    
    @abstractmethod
    async def send_batch(self, notifications: List[PushNotification]) -> List[PushResponse]:
        """Send multiple push notifications."""
        pass
    
    @abstractmethod
    async def validate_token(self, device_token: str) -> bool:
        """Validate a device token."""
        pass
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.session:
            await self.session.close()
    
    def _create_headers(self) -> Dict[str, str]:
        """Create common HTTP headers."""
        return {
            "Content-Type": "application/json",
            "User-Agent": f"WorldBrief360/1.0 ({self.provider_name})"
        }


# ============ Concrete Providers ============

class FCMProvider(BasePushProvider):
    """Firebase Cloud Messaging provider."""
    
    def __init__(self):
        super().__init__("fcm")
        self.project_id = settings.FCM_PROJECT_ID
        self.api_key = settings.FCM_API_KEY
        self.fcm_url = f"https://fcm.googleapis.com/v1/projects/{self.project_id}/messages:send"
        
    async def initialize(self) -> None:
        """Initialize FCM provider."""
        if not self.api_key:
            raise ValueError("FCM_API_KEY not configured")
        
        headers = self._create_headers()
        headers["Authorization"] = f"Bearer {self.api_key}"
        
        self.session = aiohttp.ClientSession(
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=30)
        )
    
    async def send_single(self, notification: PushNotification) -> PushResponse:
        """Send FCM notification."""
        try:
            payload = self._build_fcm_payload(notification)
            
            async with self.session.post(self.fcm_url, json=payload) as response:
                response_data = await response.json()
                
                if response.status == 200:
                    return PushResponse(
                        notification_id=notification.notification_id,
                        success=True,
                        message_id=response_data.get("name", "").split("/")[-1],
                        platform_response=response_data
                    )
                else:
                    error_msg = response_data.get("error", {}).get("message", "Unknown FCM error")
                    logger.error(f"FCM error: {error_msg}")
                    return PushResponse(
                        notification_id=notification.notification_id,
                        success=False,
                        error=error_msg,
                        platform_response=response_data
                    )
                    
        except Exception as e:
            logger.error(f"FCM send failed: {str(e)}", exc_info=True)
            return PushResponse(
                notification_id=notification.notification_id,
                success=False,
                error=str(e)
            )
    
    def _build_fcm_payload(self, notification: PushNotification) -> Dict[str, Any]:
        """Build FCM-specific payload."""
        # Platform-specific configurations
        android_config = {}
        apns_config = {}
        webpush_config = {}
        
        if notification.platform in [PushPlatform.FCM_ANDROID, PushPlatform.FCM_IOS]:
            # Android config
            android_config = {
                "priority": "HIGH" if notification.priority == PushPriority.HIGH else "NORMAL",
                "ttl": f"{notification.time_to_live}s",
                "collapse_key": notification.collapse_key or f"collapse_{notification.notification_type}"
            }
            
            # iOS config
            apns_config = {
                "headers": {
                    "apns-priority": "10" if notification.priority == PushPriority.HIGH else "5",
                    "apns-collapse-id": notification.collapse_key or notification.notification_id
                },
                "payload": {
                    "aps": {
                        "alert": {
                            "title": notification.title,
                            "body": notification.body
                        },
                        "badge": notification.badge_count,
                        "sound": notification.sound or "default",
                        "content-available": 1
                    }
                }
            }
        
        # Common message structure
        message = {
            "token": notification.device_token,
            "notification": {
                "title": notification.title,
                "body": notification.body,
                "image": notification.image_url
            },
            "data": notification.data,
            "android": android_config if android_config else None,
            "apns": apns_config if apns_config else None,
            "webpush": webpush_config if webpush_config else None
        }
        
        # Add deep link
        if notification.deep_link:
            message["fcm_options"] = {"link": notification.deep_link}
        
        return {"message": message}
    
    async def send_batch(self, notifications: List[PushNotification]) -> List[PushResponse]:
        """Send batch notifications via FCM batch API."""
        # FCM doesn't have a true batch API, so we send individually but concurrently
        tasks = [self.send_single(notification) for notification in notifications]
        return await asyncio.gather(*tasks)
    
    async def validate_token(self, device_token: str) -> bool:
        """Validate FCM token."""
        try:
            # Send a test notification to validate token
            test_notification = PushNotification(
                user_id="system",
                title="Test",
                body="Token validation",
                device_token=device_token,
                platform=PushPlatform.FCM_ANDROID,
                notification_type=PushNotificationType.SYSTEM_ALERT,
                priority=PushPriority.NORMAL
            )
            
            response = await self.send_single(test_notification)
            if not response.success:
                # Check for specific invalid token errors
                error_lower = (response.error or "").lower()
                invalid_keywords = ["invalid", "not registered", "unregistered", "bad"]
                if any(keyword in error_lower for keyword in invalid_keywords):
                    return False
            return response.success
            
        except Exception as e:
            logger.error(f"Token validation failed: {str(e)}")
            return False


class APNSProvider(BasePushProvider):
    """Apple Push Notification Service provider (direct, not via FCM)."""
    
    def __init__(self):
        super().__init__("apns")
        self.bundle_id = settings.APNS_BUNDLE_ID
        self.team_id = settings.APNS_TEAM_ID
        self.key_id = settings.APNS_KEY_ID
        self.private_key = settings.APNS_PRIVATE_KEY
        
    async def initialize(self) -> None:
        """Initialize APNS provider with JWT authentication."""
        # Note: In production, you'd want to cache the JWT token
        # and refresh it when needed (token expires after 1 hour)
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
    
    async def send_single(self, notification: PushNotification) -> PushResponse:
        """Send APNS notification."""
        try:
            # Determine environment (development/production)
            environment = "development" if settings.DEBUG else "production"
            base_url = f"https://api.sandbox.push.apple.com" if environment == "development" else "https://api.push.apple.com"
            
            # Construct APNS endpoint URL
            endpoint = f"{base_url}/3/device/{notification.device_token}"
            
            # Create JWT token for authentication
            jwt_token = self._generate_apns_jwt()
            
            # Build APNS headers
            headers = {
                "authorization": f"bearer {jwt_token}",
                "apns-topic": self.bundle_id,
                "apns-priority": "10" if notification.priority == PushPriority.HIGH else "5",
                "apns-push-type": "alert",
            }
            
            if notification.collapse_key:
                headers["apns-collapse-id"] = notification.collapse_key
            
            # Build APNS payload
            payload = {
                "aps": {
                    "alert": {
                        "title": notification.title,
                        "body": notification.body
                    },
                    "badge": notification.badge_count,
                    "sound": notification.sound or "default",
                    "content-available": 1
                },
                "data": notification.data
            }
            
            # Add deep link if provided
            if notification.deep_link:
                payload["deep_link"] = notification.deep_link
            
            async with self.session.post(endpoint, headers=headers, json=payload) as response:
                response_text = await response.text()
                
                if response.status == 200:
                    return PushResponse(
                        notification_id=notification.notification_id,
                        success=True,
                        message_id=response.headers.get("apns-id"),
                        platform_response={"status": "success"}
                    )
                else:
                    error_msg = self._parse_apns_error(response.status, response_text)
                    logger.error(f"APNS error ({response.status}): {error_msg}")
                    return PushResponse(
                        notification_id=notification.notification_id,
                        success=False,
                        error=error_msg,
                        platform_response={"status": response.status, "body": response_text}
                    )
                    
        except Exception as e:
            logger.error(f"APNS send failed: {str(e)}", exc_info=True)
            return PushResponse(
                notification_id=notification.notification_id,
                success=False,
                error=str(e)
            )
    
    def _generate_apns_jwt(self) -> str:
        """Generate JWT token for APNS authentication."""
        # Implementation depends on your JWT library
        # This is a simplified example
        import jwt
        import time
        
        now = int(time.time())
        token = jwt.encode(
            {
                "iss": self.team_id,
                "iat": now
            },
            self.private_key,
            algorithm="ES256",
            headers={
                "kid": self.key_id
            }
        )
        return token
    
    def _parse_apns_error(self, status_code: int, response_text: str) -> str:
        """Parse APNS error response."""
        error_mapping = {
            400: "Bad request",
            403: "Authentication error or wrong certificate",
            405: "Wrong HTTP method",
            410: "Device token is no longer active",
            413: "Notification payload too large",
            429: "Too many requests",
            500: "Internal server error",
            503: "Service unavailable"
        }
        
        base_error = error_mapping.get(status_code, f"HTTP {status_code}")
        try:
            error_data = json.loads(response_text)
            reason = error_data.get("reason", "unknown")
            return f"{base_error}: {reason}"
        except:
            return base_error
    
    async def send_batch(self, notifications: List[PushNotification]) -> List[PushResponse]:
        """APNS doesn't support batch, send sequentially."""
        results = []
        for notification in notifications:
            result = await self.send_single(notification)
            results.append(result)
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.1)
        return results
    
    async def validate_token(self, device_token: str) -> bool:
        """Validate APNS token by sending a silent notification."""
        try:
            test_notification = PushNotification(
                user_id="system",
                title="",
                body="",
                device_token=device_token,
                platform=PushPlatform.APNS,
                notification_type=PushNotificationType.SYSTEM_ALERT,
                priority=PushPriority.NORMAL
            )
            
            response = await self.send_single(test_notification)
            return response.success
            
        except Exception as e:
            logger.error(f"APNS token validation failed: {str(e)}")
            return False


class WebPushProvider(BasePushProvider):
    """Web Push Protocol provider."""
    
    def __init__(self):
        super().__init__("web_push")
        self.vapid_public_key = settings.VAPID_PUBLIC_KEY
        self.vapid_private_key = settings.VAPID_PRIVATE_KEY
        self.vapid_subject = settings.VAPID_SUBJECT
        
    async def initialize(self) -> None:
        """Initialize Web Push provider."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
    
    async def send_single(self, notification: PushNotification) -> PushResponse:
        """Send web push notification."""
        try:
            # Parse the device token (which is actually a subscription object)
            subscription = json.loads(notification.device_token)
            
            # Import webpush library dynamically
            from pywebpush import webpush, WebPushException
            
            # Prepare web push data
            webpush_data = {
                "title": notification.title,
                "body": notification.body,
                "icon": notification.image_url or "/icons/icon-192x192.png",
                "badge": "/icons/icon-192x192.png",
                "timestamp": datetime.utcnow().isoformat(),
                "data": notification.data,
                "actions": [
                    {
                        "action": "view",
                        "title": "View Details"
                    }
                ]
            }
            
            if notification.deep_link:
                webpush_data["data"]["url"] = notification.deep_link
            
            # Send web push
            response = webpush(
                subscription_info=subscription,
                data=json.dumps(webpush_data),
                vapid_private_key=self.vapid_private_key,
                vapid_claims={
                    "sub": self.vapid_subject,
                    "exp": int(datetime.utcnow().timestamp()) + 12 * 3600  # 12 hours
                }
            )
            
            return PushResponse(
                notification_id=notification.notification_id,
                success=True,
                message_id=str(response.status_code),
                platform_response={"status": response.status_code}
            )
            
        except Exception as e:
            logger.error(f"Web push failed: {str(e)}", exc_info=True)
            error_msg = str(e)
            
            # Check for subscription expiration
            if "410" in error_msg or "expired" in error_msg.lower():
                error_msg = "Push subscription expired"
            
            return PushResponse(
                notification_id=notification.notification_id,
                success=False,
                error=error_msg
            )
    
    async def send_batch(self, notifications: List[PushNotification]) -> List[PushResponse]:
        """Send batch web push notifications."""
        tasks = [self.send_single(notification) for notification in notifications]
        return await asyncio.gather(*tasks)
    
    async def validate_token(self, device_token: str) -> bool:
        """Validate web push subscription."""
        try:
            subscription = json.loads(device_token)
            required_keys = {"endpoint", "keys", "auth", "p256dh"}
            return all(key in subscription for key in required_keys)
        except:
            return False


# ============ Main Push Notification Client ============

class PushNotificationClient:
    """
    Main push notification client that orchestrates multiple providers.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        self.event_bus = event_bus or EventBus()
        self.providers: Dict[PushPlatform, BasePushProvider] = {}
        self.redis = get_redis_client()
        self.initialized = False
        
    async def initialize(self) -> None:
        """Initialize all push notification providers."""
        if self.initialized:
            return
        
        logger.info("Initializing push notification client...")
        
        # Initialize providers based on configuration
        providers_to_init = []
        
        if settings.FCM_API_KEY:
            fcm_provider = FCMProvider()
            await fcm_provider.initialize()
            self.providers[PushPlatform.FCM_ANDROID] = fcm_provider
            self.providers[PushPlatform.FCM_IOS] = fcm_provider
            providers_to_init.append("FCM")
        
        if settings.APNS_PRIVATE_KEY:
            apns_provider = APNSProvider()
            await apns_provider.initialize()
            self.providers[PushPlatform.APNS] = apns_provider
            providers_to_init.append("APNS")
        
        if settings.VAPID_PRIVATE_KEY:
            web_push_provider = WebPushProvider()
            await web_push_provider.initialize()
            self.providers[PushPlatform.WEB_PUSH] = web_push_provider
            providers_to_init.append("Web Push")
        
        logger.info(f"Initialized push providers: {', '.join(providers_to_init)}")
        self.initialized = True
        
        # Subscribe to events
        await self._subscribe_to_events()
    
    async def _subscribe_to_events(self) -> None:
        """Subscribe to relevant system events."""
        if self.event_bus:
            self.event_bus.subscribe(EventType.INCIDENT_VERIFIED, self._handle_incident_verified)
            self.event_bus.subscribe(EventType.BREAKING_NEWS, self._handle_breaking_news)
            self.event_bus.subscribe(EventType.COIN_REWARDED, self._handle_coin_rewarded)
    
    async def send_notification(self, notification: PushNotification) -> PushResponse:
        """
        Send a single push notification.
        
        Args:
            notification: Push notification to send
            
        Returns:
            PushResponse with result
        """
        if not self.initialized:
            await self.initialize()
        
        # Get appropriate provider
        provider = self.providers.get(notification.platform)
        if not provider:
            return PushResponse(
                notification_id=notification.notification_id,
                success=False,
                error=f"No provider configured for platform: {notification.platform}"
            )
        
        # Send notification
        response = await provider.send_single(notification)
        
        # Log and emit event
        await self._log_notification(notification, response)
        
        # Update user's notification settings if token is invalid
        if not response.success and self._is_invalid_token_error(response.error):
            await self._handle_invalid_token(notification.user_id, notification.device_token)
        
        return response
    
    async def send_batch(self, request: BatchPushRequest) -> List[PushResponse]:
        """
        Send multiple push notifications in batch.
        
        Args:
            request: Batch push request
            
        Returns:
            List of push responses
        """
        if not self.initialized:
            await self.initialize()
        
        # Group notifications by platform
        notifications_by_platform: Dict[PushPlatform, List[PushNotification]] = {}
        for notification in request.notifications:
            platform_notifications = notifications_by_platform.get(notification.platform, [])
            platform_notifications.append(notification)
            notifications_by_platform[notification.platform] = platform_notifications
        
        # Send batches per platform
        all_responses = []
        for platform, notifications in notifications_by_platform.items():
            provider = self.providers.get(platform)
            if not provider:
                # Create error responses for unsupported platforms
                error_responses = [
                    PushResponse(
                        notification_id=n.notification_id,
                        success=False,
                        error=f"No provider for platform: {platform}"
                    )
                    for n in notifications
                ]
                all_responses.extend(error_responses)
                continue
            
            # Send batch
            responses = await provider.send_batch(notifications)
            all_responses.extend(responses)
            
            # Log and handle invalid tokens
            for notification, response in zip(notifications, responses):
                await self._log_notification(notification, response)
                
                if not response.success and self._is_invalid_token_error(response.error):
                    await self._handle_invalid_token(notification.user_id, notification.device_token)
        
        return all_responses
    
    async def register_device_token(self, device_token: DeviceToken) -> bool:
        """
        Register a new device token for a user.
        
        Args:
            device_token: Device token information
            
        Returns:
            True if registration successful
        """
        try:
            # Validate token with provider
            provider = self.providers.get(device_token.platform)
            if not provider:
                logger.error(f"No provider for platform: {device_token.platform}")
                return False
            
            is_valid = await provider.validate_token(device_token.device_token)
            if not is_valid:
                logger.warning(f"Invalid device token for user {device_token.user_id}")
                return False
            
            # Store token in Redis with expiration (90 days)
            redis_key = f"user:{device_token.user_id}:push_tokens"
            token_data = device_token.dict()
            token_data["registered_at"] = datetime.utcnow().isoformat()
            
            await self.redis.hset(
                redis_key,
                device_token.device_token,
                json.dumps(token_data)
            )
            
            # Set expiration on the hash
            await self.redis.expire(redis_key, 90 * 24 * 3600)  # 90 days
            
            logger.info(f"Registered device token for user {device_token.user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Device token registration failed: {str(e)}", exc_info=True)
            return False
    
    async def unregister_device_token(self, user_id: str, device_token: str) -> bool:
        """
        Unregister a device token.
        
        Args:
            user_id: User ID
            device_token: Device token to remove
            
        Returns:
            True if removal successful
        """
        try:
            redis_key = f"user:{user_id}:push_tokens"
            removed = await self.redis.hdel(redis_key, device_token)
            return removed > 0
        except Exception as e:
            logger.error(f"Device token unregistration failed: {str(e)}")
            return False
    
    async def get_user_tokens(self, user_id: str) -> List[DeviceToken]:
        """
        Get all registered device tokens for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of device tokens
        """
        try:
            redis_key = f"user:{user_id}:push_tokens"
            token_data = await self.redis.hgetall(redis_key)
            
            tokens = []
            for token_json in token_data.values():
                try:
                    token_dict = json.loads(token_json)
                    token_dict["device_token"] = token_dict.get("device_token", "")
                    tokens.append(DeviceToken(**token_dict))
                except Exception as e:
                    logger.warning(f"Failed to parse device token: {str(e)}")
            
            return tokens
        except Exception as e:
            logger.error(f"Failed to get user tokens: {str(e)}")
            return []
    
    async def send_to_user(
        self,
        user_id: str,
        title: str,
        body: str,
        notification_type: PushNotificationType,
        data: Optional[Dict[str, Any]] = None,
        priority: PushPriority = PushPriority.NORMAL
    ) -> List[PushResponse]:
        """
        Send push notification to all devices of a user.
        
        Args:
            user_id: User ID
            title: Notification title
            body: Notification body
            notification_type: Type of notification
            data: Additional data payload
            priority: Notification priority
            
        Returns:
            List of push responses
        """
        # Get user's device tokens
        device_tokens = await self.get_user_tokens(user_id)
        if not device_tokens:
            logger.warning(f"No device tokens found for user {user_id}")
            return []
        
        # Check user notification preferences
        # (You would query user preferences from database)
        
        # Create notifications for each device
        notifications = []
        for device_token in device_tokens:
            if not device_token.is_active:
                continue
            
            notification = PushNotification(
                user_id=user_id,
                title=title,
                body=body,
                data=data or {},
                platform=device_token.platform,
                device_token=device_token.device_token,
                priority=priority,
                notification_type=notification_type,
                collapse_key=f"collapse_{notification_type}",
                deep_link=data.get("deep_link") if data else None
            )
            notifications.append(notification)
        
        # Send batch
        if notifications:
            batch_request = BatchPushRequest(notifications=notifications)
            return await self.send_batch(batch_request)
        
        return []
    
    async def _log_notification(self, notification: PushNotification, response: PushResponse) -> None:
        """Log notification sending attempt."""
        log_data = {
            "notification_id": notification.notification_id,
            "user_id": notification.user_id,
            "platform": notification.platform,
            "type": notification.notification_type,
            "success": response.success,
            "error": response.error,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if response.success:
            logger.info(f"Push notification sent: {log_data}")
        else:
            logger.error(f"Push notification failed: {log_data}")
        
        # Store in Redis for analytics (keep for 30 days)
        analytics_key = f"push_analytics:{datetime.utcnow().strftime('%Y-%m-%d')}"
        await self.redis.lpush(analytics_key, json.dumps(log_data))
        await self.redis.expire(analytics_key, 30 * 24 * 3600)  # 30 days
    
    def _is_invalid_token_error(self, error: Optional[str]) -> bool:
        """Check if error indicates an invalid device token."""
        if not error:
            return False
        
        error_lower = error.lower()
        invalid_indicators = [
            "invalid", "not registered", "unregistered", "bad", 
            "expired", "410", "gone", "inactive"
        ]
        
        return any(indicator in error_lower for indicator in invalid_indicators)
    
    async def _handle_invalid_token(self, user_id: str, device_token: str) -> None:
        """Handle invalid device token by removing it."""
        logger.warning(f"Removing invalid device token for user {user_id}")
        await self.unregister_device_token(user_id, device_token)
    
    # ============ Event Handlers ============
    
    async def _handle_incident_verified(self, event_data: Dict[str, Any]) -> None:
        """Handle incident verified event."""
        incident_id = event_data.get("incident_id")
        user_id = event_data.get("user_id")
        
        if not incident_id or not user_id:
            return
        
        # Send notification to the user who reported the incident
        await self.send_to_user(
            user_id=user_id,
            title="Incident Verified!",
            body="Your reported incident has been verified by the community. You've earned 10 coins!",
            notification_type=PushNotificationType.INCIDENT_VERIFIED,
            data={
                "incident_id": incident_id,
                "coins_awarded": 10,
                "deep_link": f"/incidents/{incident_id}"
            },
            priority=PushPriority.HIGH
        )
    
    async def _handle_breaking_news(self, event_data: Dict[str, Any]) -> None:
        """Handle breaking news event."""
        # Get users subscribed to the topic/region
        # This is a simplified example
        topic_id = event_data.get("topic_id")
        
        if not topic_id:
            return
        
        # In reality, you'd query users subscribed to this topic
        # For now, we'll send to all users (with proper segmentation in production)
        
        logger.info(f"Breaking news for topic {topic_id}, would send to subscribed users")
    
    async def _handle_coin_rewarded(self, event_data: Dict[str, Any]) -> None:
        """Handle coin rewarded event."""
        user_id = event_data.get("user_id")
        amount = event_data.get("amount", 0)
        reason = event_data.get("reason", "")
        
        if not user_id or amount <= 0:
            return
        
        await self.send_to_user(
            user_id=user_id,
            title="🎉 Coins Awarded!",
            body=f"You've earned {amount} coins for {reason}",
            notification_type=PushNotificationType.COIN_REWARDED,
            data={
                "amount": amount,
                "reason": reason,
                "deep_link": "/wallet"
            },
            priority=PushPriority.NORMAL
        )
    
    async def cleanup(self) -> None:
        """Clean up all providers."""
        for provider in self.providers.values():
            await provider.cleanup()
        self.initialized = False
        logger.info("Push notification client cleaned up")


# ============ Factory Function ============

_push_client: Optional[PushNotificationClient] = None

async def get_push_notification_client() -> PushNotificationClient:
    """
    Get or create a push notification client singleton.
    
    Returns:
        PushNotificationClient instance
    """
    global _push_client
    
    if _push_client is None:
        _push_client = PushNotificationClient()
        await _push_client.initialize()
    
    return _push_client


# ============ Utility Functions ============

async def send_breaking_news(
    title: str,
    body: str,
    topic_id: Optional[str] = None,
    region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Utility function to send breaking news notifications.
    
    Args:
        title: News title
        body: News body
        topic_id: Optional topic ID for targeting
        region: Optional region for targeting
        
    Returns:
        Statistics about the send operation
    """
    client = await get_push_notification_client()
    
    # This would query users from database based on topic/region subscriptions
    # For now, returns placeholder
    logger.info(f"Would send breaking news: {title}")
    
    return {
        "sent": 0,
        "targeted_topic": topic_id,
        "targeted_region": region,
        "message": "Breaking news queued for sending"
    }


async def send_incident_alert(
    incident_id: str,
    title: str,
    body: str,
    latitude: float,
    longitude: float,
    radius_km: int = 50
) -> Dict[str, Any]:
    """
    Send incident alert to users within a radius.
    
    Args:
        incident_id: Incident ID
        title: Alert title
        body: Alert body
        latitude: Incident latitude
        longitude: Incident longitude
        radius_km: Alert radius in kilometers
        
    Returns:
        Statistics about the send operation
    """
    client = await get_push_notification_client()
    
    # This would query users in the radius from database
    # For now, returns placeholder
    logger.info(f"Would send incident alert to users within {radius_km}km of ({latitude}, {longitude})")
    
    return {
        "incident_id": incident_id,
        "radius_km": radius_km,
        "message": "Incident alert queued for sending"
    }