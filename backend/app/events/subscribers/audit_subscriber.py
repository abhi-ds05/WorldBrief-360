"""
Audit event subscriber for tracking user activities and maintaining audit trails.
Listens for user-related events and logs them to the audit log table.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import json
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging_config import logger
from app.db.session import AsyncSessionLocal
from app.db.models.user import User
from app.db.models.audit_log import AuditLog
from app.events.event_bus import EventBus
from app.events.event_types import EventType, UserEvent, AuthEvent, ContentEvent


class AuditSubscriber:
    """
    Subscriber that listens for audit-related events and logs them to the database.
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._subscriptions = []
        
    async def initialize(self):
        """Subscribe to relevant audit events."""
        # Subscribe to user events
        self._subscriptions.extend([
            await self.event_bus.subscribe(EventType.USER_CREATED, self.handle_user_created),
            await self.event_bus.subscribe(EventType.USER_UPDATED, self.handle_user_updated),
            await self.event_bus.subscribe(EventType.USER_DELETED, self.handle_user_deleted),
            await self.event_bus.subscribe(EventType.USER_LOGIN, self.handle_user_login),
            await self.event_bus.subscribe(EventType.USER_LOGOUT, self.handle_user_logout),
            await self.event_bus.subscribe(EventType.PASSWORD_CHANGED, self.handle_password_changed),
        ])
        
        # Subscribe to content events
        self._subscriptions.extend([
            await self.event_bus.subscribe(EventType.INCIDENT_REPORTED, self.handle_incident_reported),
            await self.event_bus.subscribe(EventType.INCIDENT_UPDATED, self.handle_incident_updated),
            await self.event_bus.subscribe(EventType.INCIDENT_VERIFIED, self.handle_incident_verified),
            await self.event_bus.subscribe(EventType.BRIEFING_GENERATED, self.handle_briefing_generated),
            await self.event_bus.subscribe(EventType.CHAT_SESSION_STARTED, self.handle_chat_session),
        ])
        
        # Subscribe to system events
        self._subscriptions.extend([
            await self.event_bus.subscribe(EventType.ADMIN_ACTION, self.handle_admin_action),
            await self.event_bus.subscribe(EventType.SYSTEM_ALERT, self.handle_system_alert),
        ])
        
        logger.info("AuditSubscriber initialized and subscribed to events")
    
    async def cleanup(self):
        """Unsubscribe from all events."""
        for subscription in self._subscriptions:
            await self.event_bus.unsubscribe(subscription)
        self._subscriptions.clear()
        logger.info("AuditSubscriber cleaned up")
    
    async def _log_audit_event(
        self,
        user_id: Optional[int],
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """
        Log an audit event to the database.
        
        Args:
            user_id: ID of the user performing the action (None for system actions)
            action: The action performed (e.g., 'LOGIN', 'CREATE', 'UPDATE')
            resource_type: Type of resource affected (e.g., 'USER', 'INCIDENT', 'BRIEFING')
            resource_id: ID of the affected resource
            details: Additional details about the event
            ip_address: IP address of the request
            user_agent: User agent string
        """
        try:
            async with AsyncSessionLocal() as session:
                audit_log = AuditLog(
                    user_id=user_id,
                    action=action,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    details=json.dumps(details) if details else None,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    timestamp=datetime.utcnow()
                )
                session.add(audit_log)
                await session.commit()
                
                logger.debug(f"Audit event logged: {action} on {resource_type} by user {user_id}")
                
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}", exc_info=True)
    
    # User Event Handlers
    async def handle_user_created(self, event: UserEvent):
        """Handle user creation events."""
        await self._log_audit_event(
            user_id=event.user_id,
            action="CREATE",
            resource_type="USER",
            resource_id=str(event.user_id),
            details={
                "email": event.data.get("email"),
                "username": event.data.get("username"),
                "created_at": event.timestamp.isoformat()
            },
            ip_address=event.metadata.get("ip_address"),
            user_agent=event.metadata.get("user_agent")
        )
    
    async def handle_user_updated(self, event: UserEvent):
        """Handle user update events."""
        await self._log_audit_event(
            user_id=event.user_id,
            action="UPDATE",
            resource_type="USER",
            resource_id=str(event.user_id),
            details={
                "updated_fields": event.data.get("updated_fields", {}),
                "updated_at": event.timestamp.isoformat()
            },
            ip_address=event.metadata.get("ip_address"),
            user_agent=event.metadata.get("user_agent")
        )
    
    async def handle_user_deleted(self, event: UserEvent):
        """Handle user deletion events."""
        await self._log_audit_event(
            user_id=event.user_id,
            action="DELETE",
            resource_type="USER",
            resource_id=str(event.user_id),
            details={
                "deleted_at": event.timestamp.isoformat()
            },
            ip_address=event.metadata.get("ip_address"),
            user_agent=event.metadata.get("user_agent")
        )
    
    async def handle_user_login(self, event: AuthEvent):
        """Handle user login events."""
        await self._log_audit_event(
            user_id=event.user_id,
            action="LOGIN",
            resource_type="AUTH",
            resource_id=None,
            details={
                "login_method": event.data.get("method", "password"),
                "success": event.data.get("success", True)
            },
            ip_address=event.metadata.get("ip_address"),
            user_agent=event.metadata.get("user_agent")
        )
    
    async def handle_user_logout(self, event: AuthEvent):
        """Handle user logout events."""
        await self._log_audit_event(
            user_id=event.user_id,
            action="LOGOUT",
            resource_type="AUTH",
            resource_id=None,
            details={
                "logout_time": event.timestamp.isoformat()
            },
            ip_address=event.metadata.get("ip_address"),
            user_agent=event.metadata.get("user_agent")
        )
    
    async def handle_password_changed(self, event: AuthEvent):
        """Handle password change events."""
        await self._log_audit_event(
            user_id=event.user_id,
            action="PASSWORD_CHANGE",
            resource_type="AUTH",
            resource_id=None,
            details={
                "changed_at": event.timestamp.isoformat()
            },
            ip_address=event.metadata.get("ip_address"),
            user_agent=event.metadata.get("user_agent")
        )
    
    # Content Event Handlers
    async def handle_incident_reported(self, event: ContentEvent):
        """Handle incident reporting events."""
        await self._log_audit_event(
            user_id=event.user_id,
            action="REPORT",
            resource_type="INCIDENT",
            resource_id=str(event.data.get("incident_id")),
            details={
                "incident_type": event.data.get("incident_type"),
                "location": event.data.get("location"),
                "severity": event.data.get("severity"),
                "reported_at": event.timestamp.isoformat()
            },
            ip_address=event.metadata.get("ip_address"),
            user_agent=event.metadata.get("user_agent")
        )
    
    async def handle_incident_updated(self, event: ContentEvent):
        """Handle incident update events."""
        await self._log_audit_event(
            user_id=event.user_id,
            action="UPDATE",
            resource_type="INCIDENT",
            resource_id=str(event.data.get("incident_id")),
            details={
                "updated_fields": event.data.get("updated_fields", {}),
                "updated_at": event.timestamp.isoformat()
            },
            ip_address=event.metadata.get("ip_address"),
            user_agent=event.metadata.get("user_agent")
        )
    
    async def handle_incident_verified(self, event: ContentEvent):
        """Handle incident verification events."""
        await self._log_audit_event(
            user_id=event.user_id,
            action="VERIFY",
            resource_type="INCIDENT",
            resource_id=str(event.data.get("incident_id")),
            details={
                "verification_status": event.data.get("status"),
                "confidence_score": event.data.get("confidence_score"),
                "verified_by": event.user_id,
                "verified_at": event.timestamp.isoformat()
            },
            ip_address=event.metadata.get("ip_address"),
            user_agent=event.metadata.get("user_agent")
        )
    
    async def handle_briefing_generated(self, event: ContentEvent):
        """Handle briefing generation events."""
        await self._log_audit_event(
            user_id=event.user_id,
            action="GENERATE",
            resource_type="BRIEFING",
            resource_id=str(event.data.get("briefing_id")),
            details={
                "topic": event.data.get("topic"),
                "level": event.data.get("level"),
                "duration_seconds": event.data.get("duration_seconds"),
                "generated_at": event.timestamp.isoformat()
            },
            ip_address=event.metadata.get("ip_address"),
            user_agent=event.metadata.get("user_agent")
        )
    
    async def handle_chat_session(self, event: ContentEvent):
        """Handle chat session events."""
        await self._log_audit_event(
            user_id=event.user_id,
            action="CHAT_START",
            resource_type="CHAT",
            resource_id=str(event.data.get("session_id")),
            details={
                "initial_query": event.data.get("query"),
                "model_used": event.data.get("model"),
                "session_started": event.timestamp.isoformat()
            },
            ip_address=event.metadata.get("ip_address"),
            user_agent=event.metadata.get("user_agent")
        )
    
    # System Event Handlers
    async def handle_admin_action(self, event):
        """Handle admin action events."""
        await self._log_audit_event(
            user_id=event.user_id,
            action=event.data.get("action", "ADMIN_ACTION"),
            resource_type="SYSTEM",
            resource_id=event.data.get("resource_id"),
            details={
                "admin_action": event.data.get("action_details", {}),
                "performed_at": event.timestamp.isoformat()
            },
            ip_address=event.metadata.get("ip_address"),
            user_agent=event.metadata.get("user_agent")
        )
    
    async def handle_system_alert(self, event):
        """Handle system alert events."""
        await self._log_audit_event(
            user_id=None,  # System event, no user involved
            action="SYSTEM_ALERT",
            resource_type="SYSTEM",
            resource_id=None,
            details={
                "alert_type": event.data.get("alert_type"),
                "severity": event.data.get("severity"),
                "message": event.data.get("message"),
                "triggered_at": event.timestamp.isoformat()
            },
            ip_address=None,
            user_agent=None
        )
    
    # Query methods for generating reports (from your original code)
    async def get_user_activity_report(
        self, 
        user_id: int, 
        days: int = 30
    ) -> List[AuditLog]:
        """
        Generate user activity report for the specified user.
        
        Args:
            user_id: ID of the user
            days: Number of days to look back
            
        Returns:
            List of audit log entries
        """
        try:
            async with AsyncSessionLocal() as db:
                # Query user activities
                activities = await db.execute(
                    select(AuditLog)
                    .where(AuditLog.user_id == user_id)
                    .where(AuditLog.timestamp >= datetime.utcnow() - timedelta(days=days))
                    .order_by(desc(AuditLog.timestamp))
                )
                activities = activities.scalars().all()
                
                logger.info(f"Generated activity report for user {user_id} ({len(activities)} activities)")
                return activities
                
        except Exception as e:
            logger.error(f"Error generating user activity report: {e}", exc_info=True)
            raise
    
    async def get_system_audit_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        action_filter: Optional[str] = None,
        resource_type_filter: Optional[str] = None
    ) -> List[AuditLog]:
        """
        Generate system-wide audit report with optional filters.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            action_filter: Filter by specific action
            resource_type_filter: Filter by resource type
            
        Returns:
            List of audit log entries
        """
        try:
            async with AsyncSessionLocal() as db:
                query = select(AuditLog)
                
                # Apply date filters
                if start_date:
                    query = query.where(AuditLog.timestamp >= start_date)
                if end_date:
                    query = query.where(AuditLog.timestamp <= end_date)
                
                # Apply action filter
                if action_filter:
                    query = query.where(AuditLog.action == action_filter)
                
                # Apply resource type filter
                if resource_type_filter:
                    query = query.where(AuditLog.resource_type == resource_type_filter)
                
                # Order by timestamp
                query = query.order_by(desc(AuditLog.timestamp))
                
                result = await db.execute(query)
                logs = result.scalars().all()
                
                logger.info(f"Generated system audit report with {len(logs)} entries")
                return logs
                
        except Exception as e:
            logger.error(f"Error generating system audit report: {e}", exc_info=True)
            raise


# Factory function to create and initialize the subscriber
async def create_audit_subscriber(event_bus: EventBus) -> AuditSubscriber:
    """Create and initialize an audit subscriber."""
    subscriber = AuditSubscriber(event_bus)
    await subscriber.initialize()
    return subscriber