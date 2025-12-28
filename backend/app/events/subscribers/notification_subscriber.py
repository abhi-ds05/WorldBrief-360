"""
Notification Subscriber for handling all notification events in the system.
Listens for events that require user notifications and dispatches them through
appropriate channels (push, email, SMS, in-app, webhook).
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
import json
import uuid
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict

from sqlalchemy import select, update, and_, or_, desc, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.core.logging_config import logger
from app.db.session import AsyncSessionLocal
from app.db.models.user import User
from app.db.models.notification import (
    Notification, 
    NotificationType, 
    NotificationChannel, 
    NotificationPriority,
    NotificationStatus,
    UserNotificationPreference
)
from app.db.models.notification_template import NotificationTemplate
from app.events.event_bus import EventBus
from app.events.event_types import EventType
from app.services.utils.caching import cache
from app.core.config import settings
from app.integrations.email_client import EmailClient
from app.integrations.sms_client import SMSClient
from app.integrations.push_notification_client import PushNotificationClient
from app.services.utils.background_tasks import execute_background_task


class NotificationCategory(str, Enum):
    """Categories for organizing notifications."""
    SYSTEM = "system"               # System updates, maintenance
    SECURITY = "security"           # Security alerts, login attempts
    ACTIVITY = "activity"           # User activity, mentions, replies
    CONTENT = "content"             # Content updates, new posts
    COMMUNITY = "community"         # Community events, moderation
    REWARDS = "rewards"             # Coin rewards, transactions
    INCIDENT = "incident"           # Incident reports, verifications
    BRIEFING = "briefing"           # Briefing updates, ready notifications
    CHAT = "chat"                   # Chat messages, mentions
    MARKETING = "marketing"         # Promotional, newsletters (opt-in)


@dataclass
class NotificationRequest:
    """Data structure for notification requests."""
    user_id: int
    notification_type: NotificationType
    category: NotificationCategory
    title: str
    message: str
    priority: NotificationPriority = NotificationPriority.MEDIUM
    data: Dict[str, Any] = None
    channels: List[NotificationChannel] = None
    template_id: Optional[str] = None
    template_variables: Dict[str, Any] = None
    scheduled_for: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    source_event: Optional[str] = None
    source_event_id: Optional[str] = None
    
    def __post_init__(self):
        """Set default values."""
        if self.data is None:
            self.data = {}
        if self.channels is None:
            self.channels = [NotificationChannel.IN_APP]
        if self.template_variables is None:
            self.template_variables = {}
        
        # Add metadata
        if 'metadata' not in self.data:
            self.data['metadata'] = {}
        self.data['metadata'].update({
            'source_event': self.source_event,
            'source_event_id': self.source_event_id,
            'created_at': datetime.utcnow().isoformat()
        })


@dataclass
class NotificationDispatchResult:
    """Result of notification dispatch."""
    notification_id: str
    user_id: int
    status: NotificationStatus
    sent_channels: List[NotificationChannel]
    failed_channels: List[NotificationChannel]
    channel_results: Dict[NotificationChannel, Dict[str, Any]]
    error_messages: List[str]


class NotificationSubscriber:
    """
    Subscriber that listens for notification events and dispatches notifications
    through appropriate channels based on user preferences.
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._subscriptions = []
        
        # Initialize clients
        self.email_client = EmailClient()
        self.sms_client = SMSClient()
        self.push_client = PushNotificationClient()
        
        # Rate limiting
        self.user_notification_count = defaultdict(int)
        self.user_notification_reset = defaultdict(datetime)
        
        # Template cache
        self.template_cache = {}
        
        # WebSocket connections (for real-time notifications)
        self.websocket_connections = {}
        
    async def initialize(self):
        """Subscribe to notification events."""
        # User-related events
        self._subscriptions.extend([
            await self.event_bus.subscribe(EventType.USER_CREATED, self.handle_user_created),
            await self.event_bus.subscribe(EventType.USER_LOGIN, self.handle_user_login),
            await self.event_bus.subscribe(EventType.USER_LOGOUT, self.handle_user_logout),
            await self.event_bus.subscribe(EventType.PASSWORD_CHANGED, self.handle_password_changed),
            await self.event_bus.subscribe(EventType.EMAIL_VERIFIED, self.handle_email_verified),
            await self.event_bus.subscribe(EventType.TWO_FACTOR_ENABLED, self.handle_2fa_enabled),
        ])
        
        # Content and activity events
        self._subscriptions.extend([
            await self.event_bus.subscribe(EventType.INCIDENT_REPORTED, self.handle_incident_reported),
            await self.event_bus.subscribe(EventType.INCIDENT_VERIFIED, self.handle_incident_verified),
            await self.event_bus.subscribe(EventType.INCIDENT_UPDATED, self.handle_incident_updated),
            await self.event_bus.subscribe(EventType.COMMENT_CREATED, self.handle_comment_created),
            await self.event_bus.subscribe(EventType.COMMENT_REPLY, self.handle_comment_reply),
            await self.event_bus.subscribe(EventType.COMMENT_MENTION, self.handle_comment_mention),
            await self.event_bus.subscribe(EventType.BRIEFING_GENERATED, self.handle_briefing_generated),
            await self.event_bus.subscribe(EventType.BRIEFING_READY, self.handle_briefing_ready),
        ])
        
        # Chat and messaging events
        self._subscriptions.extend([
            await self.event_bus.subscribe(EventType.CHAT_MESSAGE_SENT, self.handle_chat_message),
            await self.event_bus.subscribe(EventType.CHAT_MENTION, self.handle_chat_mention),
            await self.event_bus.subscribe(EventType.CHAT_INVITE, self.handle_chat_invite),
        ])
        
        # Community and moderation events
        self._subscriptions.extend([
            await self.event_bus.subscribe(EventType.USER_REPORTED, self.handle_user_reported),
            await self.event_bus.subscribe(EventType.CONTENT_REPORTED, self.handle_content_reported),
            await self.event_bus.subscribe(EventType.MODERATION_ACTION_TAKEN, self.handle_moderation_action),
            await self.event_bus.subscribe(EventType.APPEAL_SUBMITTED, self.handle_appeal_submitted),
            await self.event_bus.subscribe(EventType.APPEAL_REVIEWED, self.handle_appeal_reviewed),
        ])
        
        # Rewards and wallet events
        self._subscriptions.extend([
            await self.event_bus.subscribe(EventType.COIN_REWARDED, self.handle_coin_rewarded),
            await self.event_bus.subscribe(EventType.TRANSACTION_COMPLETED, self.handle_transaction_completed),
            await self.event_bus.subscribe(EventType.WITHDRAWAL_REQUESTED, self.handle_withdrawal_requested),
            await self.event_bus.subscribe(EventType.WITHDRAWAL_COMPLETED, self.handle_withdrawal_completed),
        ])
        
        # System events
        self._subscriptions.extend([
            await self.event_bus.subscribe(EventType.SYSTEM_ALERT, self.handle_system_alert),
            await self.event_bus.subscribe(EventType.SYSTEM_MAINTENANCE, self.handle_system_maintenance),
            await self.event_bus.subscribe(EventType.NEW_FEATURE, self.handle_new_feature),
        ])
        
        # Direct notification requests
        self._subscriptions.extend([
            await self.event_bus.subscribe(EventType.NOTIFICATION_SEND, self.handle_notification_send),
            await self.event_bus.subscribe(EventType.NOTIFICATION_BROADCAST, self.handle_notification_broadcast),
        ])
        
        # Start background tasks
        asyncio.create_task(self._cleanup_expired_notifications())
        asyncio.create_task(self._process_notification_queue())
        
        logger.info("NotificationSubscriber initialized")
    
    async def cleanup(self):
        """Cleanup subscriptions and connections."""
        for subscription in self._subscriptions:
            await self.event_bus.unsubscribe(subscription)
        self._subscriptions.clear()
        
        # Close all WebSocket connections
        for connection in self.websocket_connections.values():
            await connection.close()
        self.websocket_connections.clear()
        
        logger.info("NotificationSubscriber cleaned up")
    
    def register_websocket_connection(self, user_id: int, websocket):
        """
        Register a WebSocket connection for real-time notifications.
        
        Args:
            user_id: User ID
            websocket: WebSocket connection
        """
        if user_id not in self.websocket_connections:
            self.websocket_connections[user_id] = []
        self.websocket_connections[user_id].append(websocket)
        logger.debug(f"WebSocket registered for user {user_id}")
    
    def unregister_websocket_connection(self, user_id: int, websocket):
        """
        Unregister a WebSocket connection.
        
        Args:
            user_id: User ID
            websocket: WebSocket connection
        """
        if user_id in self.websocket_connections:
            try:
                self.websocket_connections[user_id].remove(websocket)
                if not self.websocket_connections[user_id]:
                    del self.websocket_connections[user_id]
                logger.debug(f"WebSocket unregistered for user {user_id}")
            except ValueError:
                pass
    
    async def _get_notification_template(
        self, 
        template_id: str
    ) -> Optional[NotificationTemplate]:
        """
        Get notification template from cache or database.
        
        Args:
            template_id: Template identifier
            
        Returns:
            NotificationTemplate or None
        """
        # Check cache first
        if template_id in self.template_cache:
            template, cached_at = self.template_cache[template_id]
            # Check if cache is stale (older than 5 minutes)
            if datetime.utcnow() - cached_at < timedelta(minutes=5):
                return template
        
        # Fetch from database
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(NotificationTemplate)
                    .where(NotificationTemplate.template_id == template_id)
                    .where(NotificationTemplate.is_active == True)
                )
                template = result.scalar_one_or_none()
                
                if template:
                    # Update cache
                    self.template_cache[template_id] = (template, datetime.utcnow())
                    return template
                
        except Exception as e:
            logger.error(f"Error fetching notification template {template_id}: {e}")
        
        return None
    
    async def _render_template(
        self,
        template: NotificationTemplate,
        variables: Dict[str, Any],
        user: Optional[User] = None
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Render notification template with variables.
        
        Args:
            template: Notification template
            variables: Template variables
            user: User object (for user-specific variables)
            
        Returns:
            Tuple of (title, message, data)
        """
        try:
            # Add user-specific variables if available
            all_variables = variables.copy()
            if user:
                all_variables.update({
                    'user_id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'full_name': user.full_name or user.username,
                })
            
            # Add system variables
            all_variables.update({
                'current_date': datetime.utcnow().strftime('%Y-%m-%d'),
                'current_time': datetime.utcnow().strftime('%H:%M:%S'),
                'app_name': settings.APP_NAME,
                'app_url': settings.FRONTEND_URL,
            })
            
            # Simple template rendering (replace {variable} with values)
            title = template.title_template
            message = template.message_template
            
            for key, value in all_variables.items():
                placeholder = f"{{{key}}}"
                title = title.replace(placeholder, str(value))
                message = message.replace(placeholder, str(value))
            
            # Parse data template
            data = {}
            if template.data_template:
                try:
                    data = json.loads(template.data_template)
                    # Replace variables in nested data
                    data = self._replace_variables_in_dict(data, all_variables)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in data template for {template.template_id}")
                    data = {}
            
            return title, message, data
            
        except Exception as e:
            logger.error(f"Error rendering template {template.template_id}: {e}")
            # Fallback to template defaults
            return template.default_title, template.default_message, {}
    
    def _replace_variables_in_dict(self, data: Dict, variables: Dict[str, Any]) -> Dict:
        """Recursively replace variables in dictionary."""
        if isinstance(data, dict):
            return {k: self._replace_variables_in_dict(v, variables) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._replace_variables_in_dict(item, variables) for item in data]
        elif isinstance(data, str):
            for key, value in variables.items():
                placeholder = f"{{{key}}}"
                if placeholder in data:
                    data = data.replace(placeholder, str(value))
            return data
        else:
            return data
    
    async def _get_user_notification_preferences(
        self, 
        user_id: int,
        notification_type: NotificationType,
        category: NotificationCategory
    ) -> Dict[NotificationChannel, bool]:
        """
        Get user notification preferences for specific type and category.
        
        Args:
            user_id: User ID
            notification_type: Type of notification
            category: Notification category
            
        Returns:
            Dictionary mapping channels to enabled status
        """
        try:
            async with AsyncSessionLocal() as session:
                # Get user preferences
                result = await session.execute(
                    select(UserNotificationPreference)
                    .where(UserNotificationPreference.user_id == user_id)
                    .where(
                        or_(
                            UserNotificationPreference.notification_type == notification_type,
                            UserNotificationPreference.notification_type == NotificationType.ALL
                        )
                    )
                    .where(
                        or_(
                            UserNotificationPreference.category == category,
                            UserNotificationPreference.category == None
                        )
                    )
                )
                preferences = result.scalars().all()
                
                # Default preferences
                default_channels = {
                    NotificationChannel.IN_APP: True,
                    NotificationChannel.EMAIL: False,
                    NotificationChannel.PUSH: False,
                    NotificationChannel.SMS: False,
                    NotificationChannel.WEBHOOK: False,
                }
                
                # Apply user preferences
                for pref in preferences:
                    if pref.channel in default_channels:
                        default_channels[pref.channel] = pref.is_enabled
                
                return default_channels
                
        except Exception as e:
            logger.error(f"Error fetching notification preferences for user {user_id}: {e}")
            # Return default preferences on error
            return {
                NotificationChannel.IN_APP: True,
                NotificationChannel.EMAIL: False,
                NotificationChannel.PUSH: False,
                NotificationChannel.SMS: False,
                NotificationChannel.WEBHOOK: False,
            }
    
    async def _check_rate_limit(self, user_id: int) -> bool:
        """
        Check if user has exceeded notification rate limit.
        
        Args:
            user_id: User ID
            
        Returns:
            True if allowed, False if rate limited
        """
        current_time = datetime.utcnow()
        
        # Reset counter if new hour
        if (
            user_id not in self.user_notification_reset or
            current_time - self.user_notification_reset[user_id] >= timedelta(hours=1)
        ):
            self.user_notification_count[user_id] = 0
            self.user_notification_reset[user_id] = current_time
        
        # Check limit (100 notifications per hour)
        if self.user_notification_count[user_id] >= 100:
            logger.warning(f"User {user_id} exceeded notification rate limit")
            return False
        
        self.user_notification_count[user_id] += 1
        return True
    
    async def _create_notification_record(
        self,
        request: NotificationRequest
    ) -> Optional[Notification]:
        """
        Create notification record in database.
        
        Args:
            request: Notification request
            
        Returns:
            Notification record or None
        """
        try:
            async with AsyncSessionLocal() as session:
                # Get user
                result = await session.execute(
                    select(User).where(User.id == request.user_id)
                )
                user = result.scalar_one_or_none()
                
                if not user:
                    logger.error(f"User {request.user_id} not found for notification")
                    return None
                
                # Check rate limit
                if not await self._check_rate_limit(request.user_id):
                    logger.warning(f"Rate limited notification for user {request.user_id}")
                    return None
                
                # Generate notification ID
                notification_id = str(uuid.uuid4())
                
                # Apply template if specified
                title = request.title
                message = request.message
                data = request.data or {}
                
                if request.template_id:
                    template = await self._get_notification_template(request.template_id)
                    if template:
                        rendered_title, rendered_message, template_data = await self._render_template(
                            template, request.template_variables, user
                        )
                        title = rendered_title
                        message = rendered_message
                        data.update(template_data)
                
                # Create notification
                notification = Notification(
                    notification_id=notification_id,
                    user_id=request.user_id,
                    notification_type=request.notification_type,
                    category=request.category.value,
                    title=title,
                    message=message,
                    data=data,
                    priority=request.priority,
                    status=NotificationStatus.PENDING,
                    scheduled_for=request.scheduled_for,
                    expires_at=request.expires_at,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                
                session.add(notification)
                await session.commit()
                await session.refresh(notification)
                
                logger.info(f"Created notification {notification_id} for user {request.user_id}")
                return notification
                
        except Exception as e:
            logger.error(f"Failed to create notification record: {e}", exc_info=True)
            return None
    
    async def _dispatch_notification(
        self,
        notification: Notification,
        channels: List[NotificationChannel]
    ) -> NotificationDispatchResult:
        """
        Dispatch notification through specified channels.
        
        Args:
            notification: Notification record
            channels: Channels to use
            
        Returns:
            Dispatch result
        """
        sent_channels = []
        failed_channels = []
        channel_results = {}
        error_messages = []
        
        # Get user preferences
        preferences = await self._get_user_notification_preferences(
            notification.user_id,
            notification.notification_type,
            NotificationCategory(notification.category)
        )
        
        # Filter channels based on preferences
        allowed_channels = [ch for ch in channels if preferences.get(ch, False)]
        
        for channel in allowed_channels:
            try:
                result = await self._dispatch_to_channel(notification, channel)
                channel_results[channel] = result
                
                if result.get('success', False):
                    sent_channels.append(channel)
                    logger.debug(f"Notification {notification.notification_id} sent via {channel}")
                else:
                    failed_channels.append(channel)
                    error_msg = result.get('error', 'Unknown error')
                    error_messages.append(f"{channel}: {error_msg}")
                    logger.warning(f"Failed to send notification via {channel}: {error_msg}")
                    
            except Exception as e:
                failed_channels.append(channel)
                error_messages.append(f"{channel}: {str(e)}")
                logger.error(f"Error dispatching notification via {channel}: {e}", exc_info=True)
        
        # Update notification status
        status = NotificationStatus.SENT if sent_channels else NotificationStatus.FAILED
        
        try:
            async with AsyncSessionLocal() as session:
                db_notification = await session.get(Notification, notification.id)
                if db_notification:
                    db_notification.status = status
                    db_notification.sent_at = datetime.utcnow() if sent_channels else None
                    db_notification.channels_used = json.dumps([ch.value for ch in sent_channels])
                    db_notification.updated_at = datetime.utcnow()
                    await session.commit()
        except Exception as e:
            logger.error(f"Failed to update notification status: {e}")
        
        return NotificationDispatchResult(
            notification_id=notification.notification_id,
            user_id=notification.user_id,
            status=status,
            sent_channels=sent_channels,
            failed_channels=failed_channels,
            channel_results=channel_results,
            error_messages=error_messages
        )
    
    async def _dispatch_to_channel(
        self,
        notification: Notification,
        channel: NotificationChannel
    ) -> Dict[str, Any]:
        """
        Dispatch notification to specific channel.
        
        Args:
            notification: Notification record
            channel: Channel to use
            
        Returns:
            Dictionary with result information
        """
        user_id = notification.user_id
        
        if channel == NotificationChannel.IN_APP:
            return await self._send_in_app_notification(notification, user_id)
        elif channel == NotificationChannel.EMAIL:
            return await self._send_email_notification(notification, user_id)
        elif channel == NotificationChannel.PUSH:
            return await self._send_push_notification(notification, user_id)
        elif channel == NotificationChannel.SMS:
            return await self._send_sms_notification(notification, user_id)
        elif channel == NotificationChannel.WEBHOOK:
            return await self._send_webhook_notification(notification, user_id)
        else:
            return {'success': False, 'error': f'Unknown channel: {channel}'}
    
    async def _send_in_app_notification(
        self,
        notification: Notification,
        user_id: int
    ) -> Dict[str, Any]:
        """Send in-app notification."""
        try:
            # Send via WebSocket if available
            if user_id in self.websocket_connections:
                notification_data = {
                    'id': notification.notification_id,
                    'type': notification.notification_type.value,
                    'category': notification.category,
                    'title': notification.title,
                    'message': notification.message,
                    'data': notification.data,
                    'priority': notification.priority.value,
                    'created_at': notification.created_at.isoformat(),
                    'read': False
                }
                
                for websocket in self.websocket_connections[user_id]:
                    try:
                        await websocket.send_json({
                            'type': 'notification',
                            'data': notification_data
                        })
                    except Exception as e:
                        logger.error(f"Error sending WebSocket notification: {e}")
            
            return {'success': True, 'channel': 'in_app'}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'channel': 'in_app'}
    
    async def _send_email_notification(
        self,
        notification: Notification,
        user_id: int
    ) -> Dict[str, Any]:
        """Send email notification."""
        try:
            # Get user email
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(User.email).where(User.id == user_id)
                )
                user_email = result.scalar_one_or_none()
                
                if not user_email:
                    return {'success': False, 'error': 'User email not found', 'channel': 'email'}
                
                # Send email
                email_data = {
                    'to': user_email,
                    'subject': notification.title,
                    'body': notification.message,
                    'html_body': self._create_email_html(notification),
                    'category': notification.category,
                    'metadata': notification.data.get('metadata', {})
                }
                
                success = await self.email_client.send_email(**email_data)
                
                if success:
                    return {'success': True, 'channel': 'email', 'recipient': user_email}
                else:
                    return {'success': False, 'error': 'Email client failed', 'channel': 'email'}
                
        except Exception as e:
            return {'success': False, 'error': str(e), 'channel': 'email'}
    
    def _create_email_html(self, notification: Notification) -> str:
        """Create HTML email from notification."""
        # Simple HTML template
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{notification.title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #4f46e5; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 30px; background-color: #f9fafb; }}
                .footer {{ padding: 20px; text-align: center; color: #6b7280; font-size: 12px; }}
                .button {{ display: inline-block; padding: 10px 20px; background-color: #4f46e5; color: white; text-decoration: none; border-radius: 5px; }}
                .priority-high {{ border-left: 4px solid #dc2626; }}
                .priority-medium {{ border-left: 4px solid #f59e0b; }}
                .priority-low {{ border-left: 4px solid #10b981; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{settings.APP_NAME}</h1>
                </div>
                <div class="content {f'priority-{notification.priority.value}'}">
                    <h2>{notification.title}</h2>
                    <p>{notification.message}</p>
                    {self._get_email_action_button(notification)}
                </div>
                <div class="footer">
                    <p>© {datetime.now().year} {settings.APP_NAME}. All rights reserved.</p>
                    <p><a href="{settings.FRONTEND_URL}/notifications">Manage notifications</a></p>
                </div>
            </div>
        </body>
        </html>
        """
        return html
    
    def _get_email_action_button(self, notification: Notification) -> str:
        """Get action button HTML for email."""
        action_url = notification.data.get('action_url')
        action_text = notification.data.get('action_text', 'View Details')
        
        if action_url:
            return f'<p><a href="{action_url}" class="button">{action_text}</a></p>'
        return ''
    
    async def _send_push_notification(
        self,
        notification: Notification,
        user_id: int
    ) -> Dict[str, Any]:
        """Send push notification."""
        try:
            # Get user's push tokens
            # This would come from your database or cache
            push_tokens = []  # Replace with actual token retrieval
            
            if not push_tokens:
                return {'success': False, 'error': 'No push tokens found', 'channel': 'push'}
            
            # Send push notification
            push_data = {
                'title': notification.title,
                'body': notification.message,
                'data': notification.data,
                'priority': notification.priority.value,
                'tokens': push_tokens
            }
            
            success = await self.push_client.send_notification(**push_data)
            
            if success:
                return {'success': True, 'channel': 'push', 'tokens_sent': len(push_tokens)}
            else:
                return {'success': False, 'error': 'Push client failed', 'channel': 'push'}
                
        except Exception as e:
            return {'success': False, 'error': str(e), 'channel': 'push'}
    
    async def _send_sms_notification(
        self,
        notification: Notification,
        user_id: int
    ) -> Dict[str, Any]:
        """Send SMS notification."""
        try:
            # Get user's phone number
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(User.phone_number).where(User.id == user_id)
                )
                phone_number = result.scalar_one_or_none()
                
                if not phone_number:
                    return {'success': False, 'error': 'User phone number not found', 'channel': 'sms'}
                
                # Send SMS (truncate message if needed)
                message = notification.message[:160]  # SMS character limit
                success = await self.sms_client.send_sms(
                    to=phone_number,
                    message=message
                )
                
                if success:
                    return {'success': True, 'channel': 'sms', 'recipient': phone_number}
                else:
                    return {'success': False, 'error': 'SMS client failed', 'channel': 'sms'}
                
        except Exception as e:
            return {'success': False, 'error': str(e), 'channel': 'sms'}
    
    async def _send_webhook_notification(
        self,
        notification: Notification,
        user_id: int
    ) -> Dict[str, Any]:
        """Send webhook notification."""
        try:
            # Get user's webhook URLs
            # This would come from your database
            webhook_urls = []  # Replace with actual webhook retrieval
            
            if not webhook_urls:
                return {'success': False, 'error': 'No webhook URLs found', 'channel': 'webhook'}
            
            # Prepare webhook payload
            payload = {
                'notification_id': notification.notification_id,
                'user_id': user_id,
                'type': notification.notification_type.value,
                'category': notification.category,
                'title': notification.title,
                'message': notification.message,
                'data': notification.data,
                'priority': notification.priority.value,
                'timestamp': notification.created_at.isoformat()
            }
            
            # Send to all webhooks
            # In practice, you'd make HTTP requests to each webhook URL
            # For now, just log and return success
            logger.info(f"Webhook notification would be sent to {len(webhook_urls)} endpoints")
            
            return {'success': True, 'channel': 'webhook', 'endpoints': len(webhook_urls)}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'channel': 'webhook'}
    
    async def send_notification(self, request: NotificationRequest) -> NotificationDispatchResult:
        """
        Send a notification.
        
        Args:
            request: Notification request
            
        Returns:
            Dispatch result
        """
        try:
            # Create notification record
            notification = await self._create_notification_record(request)
            if not notification:
                return NotificationDispatchResult(
                    notification_id='',
                    user_id=request.user_id,
                    status=NotificationStatus.FAILED,
                    sent_channels=[],
                    failed_channels=[],
                    channel_results={},
                    error_messages=['Failed to create notification record']
                )
            
            # Dispatch notification
            result = await self._dispatch_notification(notification, request.channels)
            
            # Log result
            if result.status == NotificationStatus.SENT:
                logger.info(f"Notification {notification.notification_id} sent via {len(result.sent_channels)} channels")
            else:
                logger.warning(f"Notification {notification.notification_id} failed: {result.error_messages}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}", exc_info=True)
            return NotificationDispatchResult(
                notification_id='',
                user_id=request.user_id,
                status=NotificationStatus.FAILED,
                sent_channels=[],
                failed_channels=[],
                channel_results={},
                error_messages=[str(e)]
            )
    
    async def send_bulk_notifications(
        self,
        requests: List[NotificationRequest]
    ) -> List[NotificationDispatchResult]:
        """
        Send multiple notifications.
        
        Args:
            requests: List of notification requests
            
        Returns:
            List of dispatch results
        """
        results = []
        for request in requests:
            result = await self.send_notification(request)
            results.append(result)
        return results
    
    async def broadcast_notification(
        self,
        user_ids: List[int],
        request: NotificationRequest
    ) -> Dict[int, NotificationDispatchResult]:
        """
        Send notification to multiple users.
        
        Args:
            user_ids: List of user IDs
            request: Base notification request (user_id will be replaced)
            
        Returns:
            Dictionary mapping user_id to result
        """
        results = {}
        for user_id in user_ids:
            user_request = NotificationRequest(
                user_id=user_id,
                notification_type=request.notification_type,
                category=request.category,
                title=request.title,
                message=request.message,
                priority=request.priority,
                data=request.data,
                channels=request.channels,
                template_id=request.template_id,
                template_variables=request.template_variables,
                scheduled_for=request.scheduled_for,
                expires_at=request.expires_at,
                source_event=request.source_event,
                source_event_id=request.source_event_id
            )
            
            result = await self.send_notification(user_request)
            results[user_id] = result
        
        return results
    
    # Event Handlers
    
    async def handle_user_created(self, event):
        """Handle user creation."""
        request = NotificationRequest(
            user_id=event.user_id,
            notification_type=NotificationType.WELCOME,
            category=NotificationCategory.SYSTEM,
            title="Welcome to WorldBrief360!",
            message="Your account has been successfully created. Get started by exploring topics or reporting incidents.",
            priority=NotificationPriority.MEDIUM,
            data={
                'action_url': f'{settings.FRONTEND_URL}/dashboard',
                'action_text': 'Go to Dashboard'
            },
            channels=[NotificationChannel.IN_APP, NotificationChannel.EMAIL],
            source_event=EventType.USER_CREATED,
            source_event_id=str(event.user_id)
        )
        
        await self.send_notification(request)
    
    async def handle_user_login(self, event):
        """Handle user login."""
        # Only notify for suspicious logins
        is_suspicious = event.data.get('is_suspicious', False)
        new_device = event.data.get('new_device', False)
        
        if is_suspicious or new_device:
            request = NotificationRequest(
                user_id=event.user_id,
                notification_type=NotificationType.SECURITY_ALERT,
                category=NotificationCategory.SECURITY,
                title="New Login Detected",
                message=f"Your account was accessed from {event.data.get('ip_address', 'unknown location')}.",
                priority=NotificationPriority.HIGH,
                data={
                    'ip_address': event.data.get('ip_address'),
                    'device': event.data.get('user_agent'),
                    'action_url': f'{settings.FRONTEND_URL}/security',
                    'action_text': 'Review Security'
                },
                channels=[NotificationChannel.IN_APP, NotificationChannel.EMAIL],
                source_event=EventType.USER_LOGIN,
                source_event_id=str(event.user_id)
            )
            
            await self.send_notification(request)
    
    async def handle_user_logout(self, event):
        """Handle user logout."""
        # Usually no notification needed for logout
        pass
    
    async def handle_password_changed(self, event):
        """Handle password change."""
        request = NotificationRequest(
            user_id=event.user_id,
            notification_type=NotificationType.SECURITY_ALERT,
            category=NotificationCategory.SECURITY,
            title="Password Changed",
            message="Your password has been successfully changed.",
            priority=NotificationPriority.MEDIUM,
            channels=[NotificationChannel.IN_APP, NotificationChannel.EMAIL],
            source_event=EventType.PASSWORD_CHANGED,
            source_event_id=str(event.user_id)
        )
        
        await self.send_notification(request)
    
    async def handle_email_verified(self, event):
        """Handle email verification."""
        request = NotificationRequest(
            user_id=event.user_id,
            notification_type=NotificationType.EMAIL_VERIFIED,
            category=NotificationCategory.SYSTEM,
            title="Email Verified",
            message="Your email address has been successfully verified.",
            priority=NotificationPriority.LOW,
            channels=[NotificationChannel.IN_APP],
            source_event=EventType.EMAIL_VERIFIED,
            source_event_id=str(event.user_id)
        )
        
        await self.send_notification(request)
    
    async def handle_2fa_enabled(self, event):
        """Handle 2FA enable/disable."""
        action = "enabled" if event.data.get('enabled', True) else "disabled"
        
        request = NotificationRequest(
            user_id=event.user_id,
            notification_type=NotificationType.SECURITY_ALERT,
            category=NotificationCategory.SECURITY,
            title=f"2FA {action.capitalize()}",
            message=f"Two-factor authentication has been {action} for your account.",
            priority=NotificationPriority.MEDIUM,
            channels=[NotificationChannel.IN_APP, NotificationChannel.EMAIL],
            source_event=EventType.TWO_FACTOR_ENABLED,
            source_event_id=str(event.user_id)
        )
        
        await self.send_notification(request)
    
    async def handle_incident_reported(self, event):
        """Handle incident report."""
        incident_id = event.data.get('incident_id')
        incident_title = event.data.get('title', 'New Incident')
        
        # Notify the reporter
        reporter_request = NotificationRequest(
            user_id=event.user_id,
            notification_type=NotificationType.INCIDENT_REPORTED,
            category=NotificationCategory.INCIDENT,
            title="Incident Reported",
            message=f"Your incident '{incident_title}' has been submitted for verification.",
            priority=NotificationPriority.MEDIUM,
            data={
                'incident_id': incident_id,
                'action_url': f'{settings.FRONTEND_URL}/incidents/{incident_id}',
                'action_text': 'View Incident'
            },
            channels=[NotificationChannel.IN_APP],
            source_event=EventType.INCIDENT_REPORTED,
            source_event_id=incident_id
        )
        
        await self.send_notification(reporter_request)
    
    async def handle_incident_verified(self, event):
        """Handle incident verification."""
        incident_id = event.data.get('incident_id')
        status = event.data.get('status', 'verified')
        reporter_id = event.data.get('reporter_id')
        
        if reporter_id:
            status_text = "verified" if status == "verified" else "rejected"
            
            request = NotificationRequest(
                user_id=reporter_id,
                notification_type=NotificationType.INCIDENT_VERIFIED,
                category=NotificationCategory.INCIDENT,
                title=f"Incident {status_text.capitalize()}",
                message=f"Your incident report has been {status_text} by the community.",
                priority=NotificationPriority.MEDIUM,
                data={
                    'incident_id': incident_id,
                    'status': status,
                    'action_url': f'{settings.FRONTEND_URL}/incidents/{incident_id}',
                    'action_text': 'View Incident'
                },
                channels=[NotificationChannel.IN_APP],
                source_event=EventType.INCIDENT_VERIFIED,
                source_event_id=incident_id
            )
            
            await self.send_notification(request)
    
    async def handle_incident_updated(self, event):
        """Handle incident update."""
        incident_id = event.data.get('incident_id')
        updater_id = event.user_id
        
        # Get incident followers and notify them
        # This would query your database for users following this incident
        follower_ids = []  # Replace with actual follower retrieval
        
        if follower_ids and updater_id in follower_ids:
            follower_ids.remove(updater_id)  # Don't notify the updater
        
        if follower_ids:
            request = NotificationRequest(
                user_id=0,  # Will be replaced in broadcast
                notification_type=NotificationType.INCIDENT_UPDATED,
                category=NotificationCategory.INCIDENT,
                title="Incident Updated",
                message="An incident you're following has been updated.",
                priority=NotificationPriority.LOW,
                data={
                    'incident_id': incident_id,
                    'action_url': f'{settings.FRONTEND_URL}/incidents/{incident_id}',
                    'action_text': 'View Updates'
                },
                channels=[NotificationChannel.IN_APP],
                source_event=EventType.INCIDENT_UPDATED,
                source_event_id=incident_id
            )
            
            await self.broadcast_notification(follower_ids, request)
    
    async def handle_comment_created(self, event):
        """Handle comment creation."""
        # Notify content owner if comment is on their content
        content_owner_id = event.data.get('content_owner_id')
        content_type = event.data.get('content_type')
        content_id = event.data.get('content_id')
        
        if content_owner_id and content_owner_id != event.user_id:
            request = NotificationRequest(
                user_id=content_owner_id,
                notification_type=NotificationType.COMMENT_RECEIVED,
                category=NotificationCategory.ACTIVITY,
                title="New Comment",
                message=f"Someone commented on your {content_type}.",
                priority=NotificationPriority.LOW,
                data={
                    'content_type': content_type,
                    'content_id': content_id,
                    'comment_id': event.data.get('comment_id'),
                    'action_url': f'{settings.FRONTEND_URL}/{content_type}/{content_id}',
                    'action_text': 'View Comment'
                },
                channels=[NotificationChannel.IN_APP],
                source_event=EventType.COMMENT_CREATED,
                source_event_id=event.data.get('comment_id')
            )
            
            await self.send_notification(request)
    
    async def handle_comment_reply(self, event):
        """Handle comment reply."""
        parent_comment_author_id = event.data.get('parent_comment_author_id')
        
        if parent_comment_author_id and parent_comment_author_id != event.user_id:
            request = NotificationRequest(
                user_id=parent_comment_author_id,
                notification_type=NotificationType.COMMENT_REPLY,
                category=NotificationCategory.ACTIVITY,
                title="Reply to Your Comment",
                message="Someone replied to your comment.",
                priority=NotificationPriority.LOW,
                data={
                    'comment_id': event.data.get('comment_id'),
                    'parent_comment_id': event.data.get('parent_comment_id'),
                    'action_url': f'{settings.FRONTEND_URL}/comments/{event.data.get("parent_comment_id")}',
                    'action_text': 'View Reply'
                },
                channels=[NotificationChannel.IN_APP],
                source_event=EventType.COMMENT_REPLY,
                source_event_id=event.data.get('comment_id')
            )
            
            await self.send_notification(request)
    
    async def handle_comment_mention(self, event):
        """Handle comment mention."""
        mentioned_user_ids = event.data.get('mentioned_user_ids', [])
        
        if mentioned_user_ids:
            request = NotificationRequest(
                user_id=0,  # Will be replaced in broadcast
                notification_type=NotificationType.MENTION,
                category=NotificationCategory.ACTIVITY,
                title="You Were Mentioned",
                message="Someone mentioned you in a comment.",
                priority=NotificationPriority.MEDIUM,
                data={
                    'comment_id': event.data.get('comment_id'),
                    'content_type': event.data.get('content_type'),
                    'content_id': event.data.get('content_id'),
                    'action_url': f'{settings.FRONTEND_URL}/comments/{event.data.get("comment_id")}',
                    'action_text': 'View Mention'
                },
                channels=[NotificationChannel.IN_APP],
                source_event=EventType.COMMENT_MENTION,
                source_event_id=event.data.get('comment_id')
            )
            
            await self.broadcast_notification(mentioned_user_ids, request)
    
    async def handle_briefing_generated(self, event):
        """Handle briefing generation."""
        # Notify when briefing generation starts
        pass
    
    async def handle_briefing_ready(self, event):
        """Handle briefing ready."""
        briefing_id = event.data.get('briefing_id')
        user_id = event.user_id
        
        request = NotificationRequest(
            user_id=user_id,
            notification_type=NotificationType.BRIEFING_READY,
            category=NotificationCategory.BRIEFING,
            title="Briefing Ready",
            message="Your personalized briefing is ready to view.",
            priority=NotificationPriority.MEDIUM,
            data={
                'briefing_id': briefing_id,
                'action_url': f'{settings.FRONTEND_URL}/briefings/{briefing_id}',
                'action_text': 'View Briefing'
            },
            channels=[NotificationChannel.IN_APP, NotificationChannel.EMAIL],
            source_event=EventType.BRIEFING_READY,
            source_event_id=briefing_id
        )
        
        await self.send_notification(request)
    
    async def handle_chat_message(self, event):
        """Handle chat message."""
        chat_id = event.data.get('chat_id')
        recipient_ids = event.data.get('recipient_ids', [])
        
        if recipient_ids:
            request = NotificationRequest(
                user_id=0,  # Will be replaced in broadcast
                notification_type=NotificationType.CHAT_MESSAGE,
                category=NotificationCategory.CHAT,
                title="New Chat Message",
                message="You have a new message in chat.",
                priority=NotificationPriority.MEDIUM,
                data={
                    'chat_id': chat_id,
                    'sender_id': event.user_id,
                    'action_url': f'{settings.FRONTEND_URL}/chat/{chat_id}',
                    'action_text': 'Open Chat'
                },
                channels=[NotificationChannel.IN_APP, NotificationChannel.PUSH],
                source_event=EventType.CHAT_MESSAGE_SENT,
                source_event_id=chat_id
            )
            
            await self.broadcast_notification(recipient_ids, request)
    
    async def handle_chat_mention(self, event):
        """Handle chat mention."""
        mentioned_user_ids = event.data.get('mentioned_user_ids', [])
        chat_id = event.data.get('chat_id')
        
        if mentioned_user_ids:
            request = NotificationRequest(
                user_id=0,  # Will be replaced in broadcast
                notification_type=NotificationType.MENTION,
                category=NotificationCategory.CHAT,
                title="Chat Mention",
                message="You were mentioned in a chat.",
                priority=NotificationPriority.HIGH,
                data={
                    'chat_id': chat_id,
                    'sender_id': event.user_id,
                    'action_url': f'{settings.FRONTEND_URL}/chat/{chat_id}',
                    'action_text': 'View Chat'
                },
                channels=[NotificationChannel.IN_APP, NotificationChannel.PUSH],
                source_event=EventType.CHAT_MENTION,
                source_event_id=chat_id
            )
            
            await self.broadcast_notification(mentioned_user_ids, request)
    
    async def handle_chat_invite(self, event):
        """Handle chat invite."""
        invitee_ids = event.data.get('invitee_ids', [])
        chat_id = event.data.get('chat_id')
        
        if invitee_ids:
            request = NotificationRequest(
                user_id=0,  # Will be replaced in broadcast
                notification_type=NotificationType.CHAT_INVITE,
                category=NotificationCategory.CHAT,
                title="Chat Invitation",
                message="You've been invited to join a chat.",
                priority=NotificationPriority.MEDIUM,
                data={
                    'chat_id': chat_id,
                    'inviter_id': event.user_id,
                    'action_url': f'{settings.FRONTEND_URL}/chat/invites/{chat_id}',
                    'action_text': 'View Invitation'
                },
                channels=[NotificationChannel.IN_APP, NotificationChannel.EMAIL],
                source_event=EventType.CHAT_INVITE,
                source_event_id=chat_id
            )
            
            await self.broadcast_notification(invitee_ids, request)
    
    async def handle_user_reported(self, event):
        """Handle user report."""
        reported_user_id = event.data.get('reported_user_id')
        
        # Notify moderators
        # This would query for moderator users
        moderator_ids = []  # Replace with actual moderator retrieval
        
        if moderator_ids:
            request = NotificationRequest(
                user_id=0,  # Will be replaced in broadcast
                notification_type=NotificationType.MODERATION_ALERT,
                category=NotificationCategory.COMMUNITY,
                title="User Reported",
                message="A user has been reported for review.",
                priority=NotificationPriority.HIGH,
                data={
                    'reported_user_id': reported_user_id,
                    'reporter_id': event.user_id,
                    'reason': event.data.get('reason'),
                    'action_url': f'{settings.FRONTEND_URL}/admin/users/{reported_user_id}',
                    'action_text': 'Review Report'
                },
                channels=[NotificationChannel.IN_APP],
                source_event=EventType.USER_REPORTED,
                source_event_id=str(reported_user_id)
            )
            
            await self.broadcast_notification(moderator_ids, request)
    
    async def handle_content_reported(self, event):
        """Handle content report."""
        content_id = event.data.get('content_id')
        content_type = event.data.get('content_type')
        
        # Notify moderators
        moderator_ids = []  # Replace with actual moderator retrieval
        
        if moderator_ids:
            request = NotificationRequest(
                user_id=0,  # Will be replaced in broadcast
                notification_type=NotificationType.MODERATION_ALERT,
                category=NotificationCategory.COMMUNITY,
                title="Content Reported",
                message=f"A {content_type} has been reported for review.",
                priority=NotificationPriority.HIGH,
                data={
                    'content_type': content_type,
                    'content_id': content_id,
                    'reporter_id': event.user_id,
                    'reason': event.data.get('reason'),
                    'action_url': f'{settings.FRONTEND_URL}/admin/moderation/{content_type}/{content_id}',
                    'action_text': 'Review Report'
                },
                channels=[NotificationChannel.IN_APP],
                source_event=EventType.CONTENT_REPORTED,
                source_event_id=content_id
            )
            
            await self.broadcast_notification(moderator_ids, request)
    
    async def handle_moderation_action(self, event):
        """Handle moderation action."""
        user_id = event.data.get('user_id')
        action = event.data.get('action')
        reason = event.data.get('reason')
        
        if user_id:
            action_map = {
                'warn': 'warning',
                'suspend': 'suspension',
                'ban': 'ban',
                'remove_content': 'content removal'
            }
            
            action_text = action_map.get(action, action)
            
            request = NotificationRequest(
                user_id=user_id,
                notification_type=NotificationType.MODERATION_ACTION,
                category=NotificationCategory.COMMUNITY,
                title=f"Moderation Action: {action_text.title()}",
                message=f"Action taken: {reason}",
                priority=NotificationPriority.HIGH,
                data={
                    'action': action,
                    'reason': reason,
                    'action_url': f'{settings.FRONTEND_URL}/account/moderation',
                    'action_text': 'View Details'
                },
                channels=[NotificationChannel.IN_APP, NotificationChannel.EMAIL],
                source_event=EventType.MODERATION_ACTION_TAKEN,
                source_event_id=str(user_id)
            )
            
            await self.send_notification(request)
    
    async def handle_appeal_submitted(self, event):
        """Handle appeal submission."""
        appeal_id = event.data.get('appeal_id')
        
        # Notify moderators
        moderator_ids = []  # Replace with actual moderator retrieval
        
        if moderator_ids:
            request = NotificationRequest(
                user_id=0,  # Will be replaced in broadcast
                notification_type=NotificationType.MODERATION_ALERT,
                category=NotificationCategory.COMMUNITY,
                title="New Appeal Submitted",
                message="A user has submitted an appeal for review.",
                priority=NotificationPriority.MEDIUM,
                data={
                    'appeal_id': appeal_id,
                    'user_id': event.user_id,
                    'action_url': f'{settings.FRONTEND_URL}/admin/moderation/appeals/{appeal_id}',
                    'action_text': 'Review Appeal'
                },
                channels=[NotificationChannel.IN_APP],
                source_event=EventType.APPEAL_SUBMITTED,
                source_event_id=appeal_id
            )
            
            await self.broadcast_notification(moderator_ids, request)
    
    async def handle_appeal_reviewed(self, event):
        """Handle appeal review."""
        appeal_id = event.data.get('appeal_id')
        user_id = event.data.get('user_id')
        approved = event.data.get('approved', False)
        
        if user_id:
            status = "approved" if approved else "rejected"
            
            request = NotificationRequest(
                user_id=user_id,
                notification_type=NotificationType.APPEAL_REVIEWED,
                category=NotificationCategory.COMMUNITY,
                title=f"Appeal {status.title()}",
                message=f"Your appeal has been {status}.",
                priority=NotificationPriority.MEDIUM,
                data={
                    'appeal_id': appeal_id,
                    'status': status,
                    'reviewer_notes': event.data.get('reviewer_notes'),
                    'action_url': f'{settings.FRONTEND_URL}/account/moderation',
                    'action_text': 'View Details'
                },
                channels=[NotificationChannel.IN_APP, NotificationChannel.EMAIL],
                source_event=EventType.APPEAL_REVIEWED,
                source_event_id=appeal_id
            )
            
            await self.send_notification(request)
    
    async def handle_coin_rewarded(self, event):
        """Handle coin reward."""
        amount = event.data.get('amount', 0)
        reason = event.data.get('reason', '')
        
        request = NotificationRequest(
            user_id=event.user_id,
            notification_type=NotificationType.REWARD_RECEIVED,
            category=NotificationCategory.REWARDS,
            title="Coins Received!",
            message=f"You received {amount} coins for {reason}.",
            priority=NotificationPriority.MEDIUM,
            data={
                'amount': amount,
                'reason': reason,
                'transaction_id': event.data.get('transaction_id'),
                'action_url': f'{settings.FRONTEND_URL}/wallet',
                'action_text': 'View Wallet'
            },
            channels=[NotificationChannel.IN_APP],
            source_event=EventType.COIN_REWARDED,
            source_event_id=event.data.get('transaction_id')
        )
        
        await self.send_notification(request)
    
    async def handle_transaction_completed(self, event):
        """Handle transaction completion."""
        transaction_type = event.data.get('type', '')
        amount = event.data.get('amount', 0)
        
        type_map = {
            'purchase': 'Purchase',
            'withdrawal': 'Withdrawal',
            'transfer': 'Transfer',
            'refund': 'Refund'
        }
        
        type_text = type_map.get(transaction_type, transaction_type)
        
        request = NotificationRequest(
            user_id=event.user_id,
            notification_type=NotificationType.TRANSACTION_COMPLETED,
            category=NotificationCategory.REWARDS,
            title=f"{type_text} Completed",
            message=f"Your {type_text.lower()} of {amount} coins has been completed.",
            priority=NotificationPriority.MEDIUM,
            data={
                'type': transaction_type,
                'amount': amount,
                'transaction_id': event.data.get('transaction_id'),
                'action_url': f'{settings.FRONTEND_URL}/wallet/transactions',
                'action_text': 'View Transactions'
            },
            channels=[NotificationChannel.IN_APP],
            source_event=EventType.TRANSACTION_COMPLETED,
            source_event_id=event.data.get('transaction_id')
        )
        
        await self.send_notification(request)
    
    async def handle_withdrawal_requested(self, event):
        """Handle withdrawal request."""
        amount = event.data.get('amount', 0)
        
        # Notify user
        user_request = NotificationRequest(
            user_id=event.user_id,
            notification_type=NotificationType.WITHDRAWAL_REQUESTED,
            category=NotificationCategory.REWARDS,
            title="Withdrawal Requested",
            message=f"Your withdrawal request for {amount} coins has been received and is being processed.",
            priority=NotificationPriority.MEDIUM,
            data={
                'amount': amount,
                'request_id': event.data.get('request_id'),
                'action_url': f'{settings.FRONTEND_URL}/wallet/withdrawals',
                'action_text': 'Track Withdrawal'
            },
            channels=[NotificationChannel.IN_APP, NotificationChannel.EMAIL],
            source_event=EventType.WITHDRAWAL_REQUESTED,
            source_event_id=event.data.get('request_id')
        )
        
        await self.send_notification(user_request)
        
        # Notify admins for approval
        admin_ids = []  # Replace with actual admin retrieval
        
        if admin_ids:
            admin_request = NotificationRequest(
                user_id=0,  # Will be replaced in broadcast
                notification_type=NotificationType.WITHDRAWAL_APPROVAL,
                category=NotificationCategory.REWARDS,
                title="Withdrawal Requires Approval",
                message=f"User {event.user_id} requested a withdrawal of {amount} coins.",
                priority=NotificationPriority.HIGH,
                data={
                    'user_id': event.user_id,
                    'amount': amount,
                    'request_id': event.data.get('request_id'),
                    'action_url': f'{settings.FRONTEND_URL}/admin/withdrawals/{event.data.get("request_id")}',
                    'action_text': 'Review Withdrawal'
                },
                channels=[NotificationChannel.IN_APP],
                source_event=EventType.WITHDRAWAL_REQUESTED,
                source_event_id=event.data.get('request_id')
            )
            
            await self.broadcast_notification(admin_ids, admin_request)
    
    async def handle_withdrawal_completed(self, event):
        """Handle withdrawal completion."""
        amount = event.data.get('amount', 0)
        status = event.data.get('status', 'completed')
        
        status_text = "completed" if status == "completed" else "failed"
        
        request = NotificationRequest(
            user_id=event.user_id,
            notification_type=NotificationType.WITHDRAWAL_COMPLETED,
            category=NotificationCategory.REWARDS,
            title=f"Withdrawal {status_text.title()}",
            message=f"Your withdrawal of {amount} coins has been {status_text}.",
            priority=NotificationPriority.MEDIUM,
            data={
                'amount': amount,
                'status': status,
                'request_id': event.data.get('request_id'),
                'action_url': f'{settings.FRONTEND_URL}/wallet/withdrawals',
                'action_text': 'View Details'
            },
            channels=[NotificationChannel.IN_APP, NotificationChannel.EMAIL],
            source_event=EventType.WITHDRAWAL_COMPLETED,
            source_event_id=event.data.get('request_id')
        )
        
        await self.send_notification(request)
    
    async def handle_system_alert(self, event):
        """Handle system alert."""
        alert_type = event.data.get('alert_type', '')
        message = event.data.get('message', '')
        
        # Notify all admins
        admin_ids = []  # Replace with actual admin retrieval
        
        if admin_ids:
            request = NotificationRequest(
                user_id=0,  # Will be replaced in broadcast
                notification_type=NotificationType.SYSTEM_ALERT,
                category=NotificationCategory.SYSTEM,
                title=f"System Alert: {alert_type}",
                message=message,
                priority=NotificationPriority.CRITICAL,
                data=event.data,
                channels=[NotificationChannel.IN_APP, NotificationChannel.EMAIL, NotificationChannel.PUSH],
                source_event=EventType.SYSTEM_ALERT,
                source_event_id=event.data.get('alert_id', '')
            )
            
            await self.broadcast_notification(admin_ids, request)
    
    async def handle_system_maintenance(self, event):
        """Handle system maintenance."""
        schedule = event.data.get('schedule', {})
        
        # Notify all users about upcoming maintenance
        # In practice, you'd broadcast to all users or users with upcoming sessions
        user_ids = []  # Replace with actual user retrieval or broadcast logic
        
        if user_ids:
            request = NotificationRequest(
                user_id=0,  # Will be replaced in broadcast
                notification_type=NotificationType.SYSTEM_MAINTENANCE,
                category=NotificationCategory.SYSTEM,
                title="Scheduled Maintenance",
                message=f"System maintenance scheduled for {schedule.get('start_time')}. Expect downtime.",
                priority=NotificationPriority.HIGH,
                data=schedule,
                channels=[NotificationChannel.IN_APP, NotificationChannel.EMAIL],
                source_event=EventType.SYSTEM_MAINTENANCE,
                source_event_id=schedule.get('maintenance_id', '')
            )
            
            await self.broadcast_notification(user_ids, request)
    
    async def handle_new_feature(self, event):
        """Handle new feature announcement."""
        feature_name = event.data.get('feature_name', '')
        description = event.data.get('description', '')
        
        # Notify all users or opted-in users
        user_ids = []  # Replace with actual user retrieval
        
        if user_ids:
            request = NotificationRequest(
                user_id=0,  # Will be replaced in broadcast
                notification_type=NotificationType.NEW_FEATURE,
                category=NotificationCategory.MARKETING,
                title=f"New Feature: {feature_name}",
                message=description,
                priority=NotificationPriority.MEDIUM,
                data=event.data,
                channels=[NotificationChannel.IN_APP],
                source_event=EventType.NEW_FEATURE,
                source_event_id=event.data.get('feature_id', '')
            )
            
            await self.broadcast_notification(user_ids, request)
    
    async def handle_notification_send(self, event):
        """Handle direct notification send request."""
        request = NotificationRequest(
            user_id=event.data.get('user_id'),
            notification_type=NotificationType(event.data.get('type', NotificationType.GENERIC)),
            category=NotificationCategory(event.data.get('category', NotificationCategory.SYSTEM)),
            title=event.data.get('title', ''),
            message=event.data.get('message', ''),
            priority=NotificationPriority(event.data.get('priority', NotificationPriority.MEDIUM)),
            data=event.data.get('data', {}),
            channels=[NotificationChannel(ch) for ch in event.data.get('channels', [NotificationChannel.IN_APP])],
            source_event=EventType.NOTIFICATION_SEND,
            source_event_id=event.data.get('request_id', '')
        )
        
        await self.send_notification(request)
    
    async def handle_notification_broadcast(self, event):
        """Handle notification broadcast request."""
        user_ids = event.data.get('user_ids', [])
        
        if not user_ids:
            logger.warning("Broadcast notification requested with no user IDs")
            return
        
        request = NotificationRequest(
            user_id=0,  # Will be replaced in broadcast
            notification_type=NotificationType(event.data.get('type', NotificationType.GENERIC)),
            category=NotificationCategory(event.data.get('category', NotificationCategory.SYSTEM)),
            title=event.data.get('title', ''),
            message=event.data.get('message', ''),
            priority=NotificationPriority(event.data.get('priority', NotificationPriority.MEDIUM)),
            data=event.data.get('data', {}),
            channels=[NotificationChannel(ch) for ch in event.data.get('channels', [NotificationChannel.IN_APP])],
            source_event=EventType.NOTIFICATION_BROADCAST,
            source_event_id=event.data.get('request_id', '')
        )
        
        await self.broadcast_notification(user_ids, request)
    
    # Background tasks
    
    async def _cleanup_expired_notifications(self):
        """Clean up expired notifications."""
        while True:
            try:
                async with AsyncSessionLocal() as session:
                    # Mark expired notifications as expired
                    expired_cutoff = datetime.utcnow()
                    
                    result = await session.execute(
                        update(Notification)
                        .where(
                            and_(
                                Notification.expires_at < expired_cutoff,
                                Notification.status.notin_([NotificationStatus.EXPIRED, NotificationStatus.READ])
                            )
                        )
                        .values(
                            status=NotificationStatus.EXPIRED,
                            updated_at=datetime.utcnow()
                        )
                    )
                    
                    expired_count = result.rowcount
                    await session.commit()
                    
                    if expired_count > 0:
                        logger.info(f"Marked {expired_count} notifications as expired")
                    
                    # Delete old notifications (older than 90 days)
                    delete_cutoff = datetime.utcnow() - timedelta(days=90)
                    
                    result = await session.execute(
                        select(Notification).where(
                            and_(
                                Notification.created_at < delete_cutoff,
                                or_(
                                    Notification.status == NotificationStatus.EXPIRED,
                                    Notification.status == NotificationStatus.READ
                                )
                            )
                        )
                    )
                    
                    old_notifications = result.scalars().all()
                    
                    for notification in old_notifications:
                        await session.delete(notification)
                    
                    deleted_count = len(old_notifications)
                    await session.commit()
                    
                    if deleted_count > 0:
                        logger.info(f"Deleted {deleted_count} old notifications")
                
            except Exception as e:
                logger.error(f"Error cleaning up notifications: {e}", exc_info=True)
            
            # Run every hour
            await asyncio.sleep(3600)
    
    async def _process_notification_queue(self):
        """Process scheduled notifications."""
        while True:
            try:
                async with AsyncSessionLocal() as session:
                    # Get scheduled notifications that are ready
                    current_time = datetime.utcnow()
                    
                    result = await session.execute(
                        select(Notification)
                        .where(
                            and_(
                                Notification.status == NotificationStatus.PENDING,
                                Notification.scheduled_for <= current_time,
                                or_(
                                    Notification.expires_at == None,
                                    Notification.expires_at > current_time
                                )
                            )
                        )
                        .order_by(Notification.priority.desc(), Notification.created_at.asc())
                        .limit(100)
                    )
                    
                    scheduled_notifications = result.scalars().all()
                    
                    for notification in scheduled_notifications:
                        try:
                            # Convert to NotificationRequest
                            request = NotificationRequest(
                                user_id=notification.user_id,
                                notification_type=notification.notification_type,
                                category=NotificationCategory(notification.category),
                                title=notification.title,
                                message=notification.message,
                                priority=notification.priority,
                                data=notification.data,
                                channels=[NotificationChannel.IN_APP],  # Default channel
                                source_event='scheduled',
                                source_event_id=notification.notification_id
                            )
                            
                            # Dispatch notification
                            await self._dispatch_notification(notification, [NotificationChannel.IN_APP])
                            
                            logger.debug(f"Processed scheduled notification {notification.notification_id}")
                            
                        except Exception as e:
                            logger.error(f"Error processing scheduled notification {notification.notification_id}: {e}")
                            # Mark as failed
                            notification.status = NotificationStatus.FAILED
                            notification.updated_at = datetime.utcnow()
                    
                    await session.commit()
                    
                    if scheduled_notifications:
                        logger.info(f"Processed {len(scheduled_notifications)} scheduled notifications")
                
            except Exception as e:
                logger.error(f"Error processing notification queue: {e}", exc_info=True)
            
            # Run every minute
            await asyncio.sleep(60)
    
    # Public API methods
    
    async def get_user_notifications(
        self,
        user_id: int,
        unread_only: bool = False,
        limit: int = 50,
        offset: int = 0
    ) -> List[Notification]:
        """Get notifications for a user."""
        try:
            async with AsyncSessionLocal() as session:
                query = select(Notification).where(Notification.user_id == user_id)
                
                if unread_only:
                    query = query.where(Notification.status == NotificationStatus.SENT)
                
                query = query.order_by(desc(Notification.created_at)).limit(limit).offset(offset)
                
                result = await session.execute(query)
                return result.scalars().all()
                
        except Exception as e:
            logger.error(f"Error getting user notifications: {e}")
            return []
    
    async def mark_as_read(self, notification_ids: List[str], user_id: int) -> bool:
        """Mark notifications as read."""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    update(Notification)
                    .where(
                        and_(
                            Notification.notification_id.in_(notification_ids),
                            Notification.user_id == user_id
                        )
                    )
                    .values(
                        status=NotificationStatus.READ,
                        read_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                )
                
                await session.commit()
                return result.rowcount > 0
                
        except Exception as e:
            logger.error(f"Error marking notifications as read: {e}")
            return False
    
    async def mark_all_as_read(self, user_id: int) -> bool:
        """Mark all user notifications as read."""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    update(Notification)
                    .where(
                        and_(
                            Notification.user_id == user_id,
                            Notification.status == NotificationStatus.SENT
                        )
                    )
                    .values(
                        status=NotificationStatus.READ,
                        read_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                )
                
                await session.commit()
                return result.rowcount > 0
                
        except Exception as e:
            logger.error(f"Error marking all notifications as read: {e}")
            return False
    
    async def get_notification_stats(self, user_id: int) -> Dict[str, Any]:
        """Get notification statistics for a user."""
        try:
            async with AsyncSessionLocal() as session:
                # Count unread notifications
                unread_result = await session.execute(
                    select(func.count(Notification.id))
                    .where(
                        and_(
                            Notification.user_id == user_id,
                            Notification.status == NotificationStatus.SENT
                        )
                    )
                )
                unread_count = unread_result.scalar() or 0
                
                # Count total notifications
                total_result = await session.execute(
                    select(func.count(Notification.id))
                    .where(Notification.user_id == user_id)
                )
                total_count = total_result.scalar() or 0
                
                # Get recent notification types
                recent_result = await session.execute(
                    select(Notification.notification_type, func.count(Notification.id))
                    .where(Notification.user_id == user_id)
                    .where(Notification.created_at >= datetime.utcnow() - timedelta(days=7))
                    .group_by(Notification.notification_type)
                    .order_by(func.count(Notification.id).desc())
                    .limit(5)
                )
                recent_types = dict(recent_result.all())
                
                return {
                    'unread_count': unread_count,
                    'total_count': total_count,
                    'read_count': total_count - unread_count,
                    'recent_types': recent_types,
                    'last_updated': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting notification stats: {e}")
            return {
                'unread_count': 0,
                'total_count': 0,
                'read_count': 0,
                'recent_types': {},
                'error': str(e)
            }


# Factory function
async def create_notification_subscriber(event_bus: EventBus) -> NotificationSubscriber:
    """Create and initialize a notification subscriber."""
    subscriber = NotificationSubscriber(event_bus)
    await subscriber.initialize()
    return subscriber