"""
Moderation Subscriber for monitoring and moderating user-generated content.
Listens for content creation/update events and applies moderation rules, filters,
and automated actions based on configured policies.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import json
import re
from enum import Enum
from dataclasses import dataclass, asdict

from sqlalchemy import select, update, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.core.logging_config import logger
from app.db.session import AsyncSessionLocal
from app.db.models.user import User
from app.db.models.incident import Incident, IncidentStatus
from app.db.models.comment import Comment
from app.db.models.briefing import Briefing
from app.db.models.audit_log import AuditLog
from app.db.models.moderation_log import ModerationLog, ModerationAction, ContentType
from app.events.event_bus import EventBus
from app.events.event_types import EventType
from app.services.community.moderation_service import ModerationService
from app.services.processing.moderation_filters import ModerationFilters
from app.services.utils.caching import cache
from app.core.config import settings


class ModerationSeverity(str, Enum):
    """Severity levels for moderation violations."""
    LOW = "low"           # Minor violations (warning)
    MEDIUM = "medium"     # Moderate violations (content removal)
    HIGH = "high"         # Serious violations (temporary ban)
    CRITICAL = "critical" # Severe violations (permanent ban)


@dataclass
class ModerationRule:
    """Moderation rule definition."""
    name: str
    pattern: str
    severity: ModerationSeverity
    action: ModerationAction
    content_types: List[ContentType]
    enabled: bool = True
    regex_flags: int = re.IGNORECASE
    metadata: Dict[str, Any] = None
    
    def matches(self, text: str) -> bool:
        """Check if text matches this rule."""
        try:
            return bool(re.search(self.pattern, text, flags=self.regex_flags))
        except re.error:
            logger.error(f"Invalid regex pattern in rule {self.name}: {self.pattern}")
            return False


class ModerationSubscriber:
    """
    Subscriber that listens for content events and applies moderation rules.
    Handles automated moderation, user behavior analysis, and enforcement actions.
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.moderation_service = ModerationService()
        self.moderation_filters = ModerationFilters()
        self._subscriptions = []
        
        # Load moderation rules
        self.rules = self._load_moderation_rules()
        
        # User reputation tracking
        self.user_violations_cache = {}  # user_id -> violation_count
        
    def _load_moderation_rules(self) -> List[ModerationRule]:
        """Load moderation rules from configuration."""
        rules = [
            # Hate speech and discrimination
            ModerationRule(
                name="hate_speech_racial",
                pattern=r"\b(nigger|chink|spic|kike|wetback|gook)\b",
                severity=ModerationSeverity.CRITICAL,
                action=ModerationAction.REMOVE_CONTENT_AND_SUSPEND,
                content_types=[ContentType.INCIDENT, ContentType.COMMENT, ContentType.CHAT]
            ),
            ModerationRule(
                name="hate_speech_generic",
                pattern=r"\b(kill all|exterminate|die|fag|tranny)\b.*?\b(people|persons|humans|community|group)\b",
                severity=ModerationSeverity.HIGH,
                action=ModerationAction.REMOVE_CONTENT_AND_WARN,
                content_types=[ContentType.INCIDENT, ContentType.COMMENT, ContentType.CHAT]
            ),
            
            # Threats and violence
            ModerationRule(
                name="violent_threats",
                pattern=r"\b(i will|i'm going to|gonna)\b.*?\b(kill|murder|hurt|harm|attack|beat|stab|shoot)\b.*?\b(you|them|us|someone|people)\b",
                severity=ModerationSeverity.CRITICAL,
                action=ModerationAction.REMOVE_CONTENT_AND_SUSPEND,
                content_types=[ContentType.INCIDENT, ContentType.COMMENT, ContentType.CHAT]
            ),
            ModerationRule(
                name="incitement_violence",
                pattern=r"\b(let's|we should|someone should)\b.*?\b(attack|burn|destroy|riot|loot)\b",
                severity=ModerationSeverity.HIGH,
                action=ModerationAction.REMOVE_CONTENT_AND_WARN,
                content_types=[ContentType.INCIDENT, ContentType.COMMENT]
            ),
            
            # Harassment and bullying
            ModerationRule(
                name="personal_harassment",
                pattern=r"\b(you are|you're)\b.*?\b(stupid|idiot|moron|retard|ugly|fat|worthless|failure)\b",
                severity=ModerationSeverity.MEDIUM,
                action=ModerationAction.FLAG_FOR_REVIEW,
                content_types=[ContentType.COMMENT, ContentType.CHAT]
            ),
            
            # Spam and scams
            ModerationRule(
                name="financial_scams",
                pattern=r"\b(send money|wire transfer|western union|moneygram|bitcoin|ethereum|crypto)\b.*?\b(urgent|emergency|help|assistance|prize|lottery|won)\b",
                severity=ModerationSeverity.HIGH,
                action=ModerationAction.REMOVE_CONTENT,
                content_types=[ContentType.INCIDENT, ContentType.COMMENT]
            ),
            ModerationRule(
                name="spam_links",
                pattern=r"(?i)\b(bit\.ly|tinyurl|goo\.gl|t\.co|ow\.ly|is\.gd|buff\.ly|adf\.ly|shorte\.st)\b",
                severity=ModerationSeverity.MEDIUM,
                action=ModerationAction.REMOVE_CONTENT,
                content_types=[ContentType.INCIDENT, ContentType.COMMENT, ContentType.CHAT]
            ),
            
            # Misinformation patterns
            ModerationRule(
                name="misinformation_keywords",
                pattern=r"\b(fake news|hoax|conspiracy|deep state|false flag|government coverup|not real)\b",
                severity=ModerationSeverity.MEDIUM,
                action=ModerationAction.FLAG_FOR_REVIEW,
                content_types=[ContentType.INCIDENT, ContentType.COMMENT],
                metadata={"requires_context": True}
            ),
            
            # Personal information
            ModerationRule(
                name="personal_info",
                pattern=r"\b(\d{3}[-.]?\d{3}[-.]?\d{4}|\(\d{3}\)\s*\d{3}[-.]?\d{4}|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b|\b\d{5}(?:[-\s]\d{4})?\b)",
                severity=ModerationSeverity.MEDIUM,
                action=ModerationAction.REDACT_AND_WARN,
                content_types=[ContentType.INCIDENT, ContentType.COMMENT, ContentType.CHAT]
            ),
            
            # Automated bot patterns
            ModerationRule(
                name="bot_patterns",
                pattern=r"(?i)\b(check out|visit|click here|subscribe|follow|like|share|retweet|promo|discount|offer|limited time)\b.*?(http|www|\.com|\.org)",
                severity=ModerationSeverity.MEDIUM,
                action=ModerationAction.REMOVE_CONTENT,
                content_types=[ContentType.INCIDENT, ContentType.COMMENT]
            ),
        ]
        
        # Add custom rules from settings if available
        if hasattr(settings, 'MODERATION_RULES'):
            try:
                custom_rules = json.loads(settings.MODERATION_RULES)
                for rule_data in custom_rules:
                    rules.append(ModerationRule(**rule_data))
            except Exception as e:
                logger.error(f"Failed to load custom moderation rules: {e}")
        
        logger.info(f"Loaded {len(rules)} moderation rules")
        return rules
    
    async def initialize(self):
        """Subscribe to moderation-related events."""
        # Content creation events
        self._subscriptions.extend([
            await self.event_bus.subscribe(EventType.INCIDENT_REPORTED, self.handle_incident_reported),
            await self.event_bus.subscribe(EventType.INCIDENT_UPDATED, self.handle_incident_updated),
            await self.event_bus.subscribe(EventType.COMMENT_CREATED, self.handle_comment_created),
            await self.event_bus.subscribe(EventType.COMMENT_UPDATED, self.handle_comment_updated),
            await self.event_bus.subscribe(EventType.CHAT_MESSAGE_SENT, self.handle_chat_message),
        ])
        
        # User behavior events
        self._subscriptions.extend([
            await self.event_bus.subscribe(EventType.USER_CREATED, self.handle_user_created),
            await self.event_bus.subscribe(EventType.USER_REPORTED, self.handle_user_reported),
            await self.event_bus.subscribe(EventType.CONTENT_REPORTED, self.handle_content_reported),
        ])
        
        # System moderation events
        self._subscriptions.extend([
            await self.event_bus.subscribe(EventType.MODERATION_ACTION_REQUIRED, self.handle_moderation_action),
            await self.event_bus.subscribe(EventType.AUTO_MODERATION_CHECK, self.handle_auto_moderation_check),
        ])
        
        # Start periodic moderation tasks
        asyncio.create_task(self._periodic_moderation_scan())
        
        logger.info("ModerationSubscriber initialized")
    
    async def cleanup(self):
        """Cleanup subscriptions."""
        for subscription in self._subscriptions:
            await self.event_bus.unsubscribe(subscription)
        self._subscriptions.clear()
        logger.info("ModerationSubscriber cleaned up")
    
    async def _log_moderation_action(
        self,
        action: ModerationAction,
        content_type: ContentType,
        content_id: str,
        user_id: Optional[int],
        reason: str,
        severity: ModerationSeverity,
        matched_rules: List[str] = None,
        details: Dict[str, Any] = None
    ):
        """Log a moderation action to the database."""
        try:
            async with AsyncSessionLocal() as session:
                log_entry = ModerationLog(
                    action=action,
                    content_type=content_type,
                    content_id=content_id,
                    user_id=user_id,
                    moderator_id=None,  # Automated action
                    reason=reason,
                    severity=severity,
                    matched_rules=json.dumps(matched_rules) if matched_rules else None,
                    details=json.dumps(details) if details else None,
                    created_at=datetime.utcnow()
                )
                
                session.add(log_entry)
                await session.commit()
                
                # Also log to audit trail
                await self._log_audit_event(
                    user_id=user_id,
                    action=f"MODERATION_{action.value}",
                    resource_type=content_type.value,
                    resource_id=content_id,
                    details={
                        "reason": reason,
                        "severity": severity.value,
                        "matched_rules": matched_rules,
                        "moderator": "automated_system"
                    }
                )
                
                logger.info(f"Moderation action logged: {action} on {content_type}:{content_id}")
                
        except Exception as e:
            logger.error(f"Failed to log moderation action: {e}", exc_info=True)
    
    async def _log_audit_event(self, **kwargs):
        """Log to audit trail."""
        try:
            # Emit event that audit subscriber will pick up
            await self.event_bus.emit(EventType.SYSTEM_ALERT, {
                "alert_type": "moderation_action",
                "data": kwargs
            })
        except Exception as e:
            logger.error(f"Failed to emit audit event: {e}")
    
    async def _check_content(
        self,
        text: str,
        content_type: ContentType,
        user_id: Optional[int] = None
    ) -> Tuple[List[ModerationRule], ModerationSeverity]:
        """
        Check content against moderation rules.
        
        Args:
            text: Content text to check
            content_type: Type of content
            user_id: ID of user who created the content
            
        Returns:
            Tuple of (matched_rules, highest_severity)
        """
        matched_rules = []
        highest_severity = ModerationSeverity.LOW
        
        # Skip moderation for trusted users (admins, moderators, high-reputation users)
        if user_id and await self._is_trusted_user(user_id):
            return matched_rules, highest_severity
        
        # Check each rule
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            if content_type not in rule.content_types:
                continue
            
            if rule.matches(text):
                matched_rules.append(rule.name)
                
                # Update highest severity
                severity_order = {
                    ModerationSeverity.LOW: 0,
                    ModerationSeverity.MEDIUM: 1,
                    ModerationSeverity.HIGH: 2,
                    ModerationSeverity.CRITICAL: 3
                }
                
                if severity_order[rule.severity] > severity_order[highest_severity]:
                    highest_severity = rule.severity
        
        return matched_rules, highest_severity
    
    async def _is_trusted_user(self, user_id: int) -> bool:
        """Check if user is trusted (admin, moderator, or high reputation)."""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(User).where(User.id == user_id)
                )
                user = result.scalar_one_or_none()
                
                if not user:
                    return False
                
                # Check if user is admin or moderator
                if user.is_admin or user.is_moderator:
                    return True
                
                # Check reputation score
                if hasattr(user, 'reputation_score') and user.reputation_score >= 100:
                    return True
                
                # Check account age (older than 30 days)
                if user.created_at and (datetime.utcnow() - user.created_at).days > 30:
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Error checking trusted user status: {e}")
            return False
    
    async def _apply_moderation_action(
        self,
        action: ModerationAction,
        content_type: ContentType,
        content_id: str,
        user_id: Optional[int],
        reason: str,
        severity: ModerationSeverity,
        matched_rules: List[str]
    ):
        """
        Apply moderation action to content.
        
        Args:
            action: Action to take
            content_type: Type of content
            content_id: ID of content
            user_id: ID of user who created content
            reason: Reason for action
            severity: Severity level
            matched_rules: Rules that were matched
        """
        try:
            async with AsyncSessionLocal() as session:
                # Apply action based on content type
                if content_type == ContentType.INCIDENT:
                    await self._moderate_incident(session, content_id, action, reason)
                elif content_type == ContentType.COMMENT:
                    await self._moderate_comment(session, content_id, action, reason)
                elif content_type == ContentType.CHAT:
                    await self._moderate_chat_message(session, content_id, action, reason)
                elif content_type == ContentType.BRIEFING:
                    await self._moderate_briefing(session, content_id, action, reason)
                
                # Apply user-level actions if needed
                if user_id and action in [
                    ModerationAction.WARN_USER,
                    ModerationAction.SUSPEND_USER,
                    ModerationAction.BAN_USER,
                    ModerationAction.REMOVE_CONTENT_AND_WARN,
                    ModerationAction.REMOVE_CONTENT_AND_SUSPEND,
                    ModerationAction.REDACT_AND_WARN
                ]:
                    await self._apply_user_action(session, user_id, action, reason, severity)
                
                await session.commit()
                
                # Log the action
                await self._log_moderation_action(
                    action=action,
                    content_type=content_type,
                    content_id=content_id,
                    user_id=user_id,
                    reason=reason,
                    severity=severity,
                    matched_rules=matched_rules,
                    details={"applied_at": datetime.utcnow().isoformat()}
                )
                
                # Emit notification event
                await self._notify_moderation_action(
                    action, content_type, content_id, user_id, reason
                )
                
        except Exception as e:
            logger.error(f"Failed to apply moderation action: {e}", exc_info=True)
    
    async def _moderate_incident(
        self,
        session: AsyncSession,
        incident_id: str,
        action: ModerationAction,
        reason: str
    ):
        """Apply moderation action to incident."""
        incident = await session.get(Incident, incident_id)
        if not incident:
            logger.warning(f"Incident {incident_id} not found for moderation")
            return
        
        if action in [
            ModerationAction.REMOVE_CONTENT,
            ModerationAction.REMOVE_CONTENT_AND_WARN,
            ModerationAction.REMOVE_CONTENT_AND_SUSPEND
        ]:
            incident.status = IncidentStatus.REMOVED
            incident.moderation_notes = f"Automated removal: {reason}"
        elif action == ModerationAction.FLAG_FOR_REVIEW:
            incident.needs_moderation_review = True
            incident.moderation_notes = f"Flagged for review: {reason}"
        elif action == ModerationAction.REDACT_AND_WARN:
            # Redact sensitive information
            if incident.description:
                incident.description = self._redact_personal_info(incident.description)
            incident.moderation_notes = f"Redacted: {reason}"
    
    async def _moderate_comment(
        self,
        session: AsyncSession,
        comment_id: str,
        action: ModerationAction,
        reason: str
    ):
        """Apply moderation action to comment."""
        from app.db.models.comment import Comment
        
        comment = await session.get(Comment, comment_id)
        if not comment:
            logger.warning(f"Comment {comment_id} not found for moderation")
            return
        
        if action in [
            ModerationAction.REMOVE_CONTENT,
            ModerationAction.REMOVE_CONTENT_AND_WARN,
            ModerationAction.REMOVE_CONTENT_AND_SUSPEND
        ]:
            comment.is_removed = True
            comment.removal_reason = f"Automated removal: {reason}"
        elif action == ModerationAction.FLAG_FOR_REVIEW:
            comment.needs_moderation_review = True
        elif action == ModerationAction.REDACT_AND_WARN:
            # Redact sensitive information
            if comment.content:
                comment.content = self._redact_personal_info(comment.content)
    
    async def _moderate_chat_message(
        self,
        session: AsyncSession,
        message_id: str,
        action: ModerationAction,
        reason: str
    ):
        """Apply moderation action to chat message."""
        # This would interact with your chat message model
        # Adjust based on your actual chat message model structure
        logger.info(f"Moderating chat message {message_id}: {action} - {reason}")
    
    async def _moderate_briefing(
        self,
        session: AsyncSession,
        briefing_id: str,
        action: ModerationAction,
        reason: str
    ):
        """Apply moderation action to briefing."""
        briefing = await session.get(Briefing, briefing_id)
        if not briefing:
            logger.warning(f"Briefing {briefing_id} not found for moderation")
            return
        
        if action in [
            ModerationAction.REMOVE_CONTENT,
            ModerationAction.REMOVE_CONTENT_AND_WARN,
            ModerationAction.REMOVE_CONTENT_AND_SUSPEND
        ]:
            briefing.is_removed = True
            briefing.removal_reason = f"Automated removal: {reason}"
        elif action == ModerationAction.FLAG_FOR_REVIEW:
            briefing.needs_moderation_review = True
    
    async def _apply_user_action(
        self,
        session: AsyncSession,
        user_id: int,
        action: ModerationAction,
        reason: str,
        severity: ModerationSeverity
    ):
        """Apply user-level moderation action."""
        user = await session.get(User, user_id)
        if not user:
            logger.warning(f"User {user_id} not found for moderation action")
            return
        
        # Track violation
        violation_key = f"user_violations:{user_id}"
        current_violations = self.user_violations_cache.get(user_id, 0)
        current_violations += 1
        self.user_violations_cache[user_id] = current_violations
        
        # Update user's violation count in database
        if hasattr(user, 'moderation_violation_count'):
            user.moderation_violation_count = current_violations
        
        # Apply action based on severity and violation count
        if action in [ModerationAction.WARN_USER, ModerationAction.REDACT_AND_WARN, ModerationAction.REMOVE_CONTENT_AND_WARN]:
            user.last_warning_at = datetime.utcnow()
            user.warning_count = getattr(user, 'warning_count', 0) + 1
            
            # Send warning notification
            await self._send_user_warning(user, reason, severity)
            
        elif action in [ModerationAction.SUSPEND_USER, ModerationAction.REMOVE_CONTENT_AND_SUSPEND]:
            # Calculate suspension duration based on severity and violation count
            if severity == ModerationSeverity.HIGH:
                suspension_days = min(7 * current_violations, 30)  # Up to 30 days
            else:
                suspension_days = min(3 * current_violations, 14)  # Up to 14 days
            
            user.is_suspended = True
            user.suspended_until = datetime.utcnow() + timedelta(days=suspension_days)
            user.suspension_reason = reason
            
            # Send suspension notification
            await self._send_user_suspension(user, reason, suspension_days)
            
        elif action == ModerationAction.BAN_USER:
            user.is_banned = True
            user.banned_at = datetime.utcnow()
            user.ban_reason = reason
            
            # Send ban notification
            await self._send_user_ban(user, reason)
    
    def _redact_personal_info(self, text: str) -> str:
        """Redact personal information from text."""
        # Redact email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL REDACTED]', text)
        
        # Redact phone numbers
        text = re.sub(r'\b(\d{3}[-.]?\d{3}[-.]?\d{4}|\(\d{3}\)\s*\d{3}[-.]?\d{4})\b', '[PHONE REDACTED]', text)
        
        # Redact social security numbers (US)
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN REDACTED]', text)
        
        # Redact credit card numbers
        text = re.sub(r'\b(?:\d[ -]*?){13,16}\b', '[CARD REDACTED]', text)
        
        return text
    
    async def _send_user_warning(self, user: User, reason: str, severity: ModerationSeverity):
        """Send warning notification to user."""
        try:
            await self.event_bus.emit(EventType.NOTIFICATION_SEND, {
                "user_id": user.id,
                "type": "moderation_warning",
                "title": "Content Moderation Warning",
                "message": f"Your content was moderated: {reason}",
                "severity": severity.value,
                "data": {
                    "warning_reason": reason,
                    "violation_severity": severity.value,
                    "timestamp": datetime.utcnow().isoformat()
                }
            })
        except Exception as e:
            logger.error(f"Failed to send warning notification: {e}")
    
    async def _send_user_suspension(self, user: User, reason: str, days: int):
        """Send suspension notification to user."""
        try:
            await self.event_bus.emit(EventType.NOTIFICATION_SEND, {
                "user_id": user.id,
                "type": "account_suspension",
                "title": "Account Suspended",
                "message": f"Your account has been suspended for {days} days: {reason}",
                "severity": "high",
                "data": {
                    "suspension_reason": reason,
                    "suspension_days": days,
                    "suspended_until": user.suspended_until.isoformat() if user.suspended_until else None,
                    "timestamp": datetime.utcnow().isoformat()
                }
            })
        except Exception as e:
            logger.error(f"Failed to send suspension notification: {e}")
    
    async def _send_user_ban(self, user: User, reason: str):
        """Send ban notification to user."""
        try:
            await self.event_bus.emit(EventType.NOTIFICATION_SEND, {
                "user_id": user.id,
                "type": "account_banned",
                "title": "Account Banned",
                "message": f"Your account has been permanently banned: {reason}",
                "severity": "critical",
                "data": {
                    "ban_reason": reason,
                    "banned_at": user.banned_at.isoformat() if user.banned_at else None,
                    "timestamp": datetime.utcnow().isoformat()
                }
            })
        except Exception as e:
            logger.error(f"Failed to send ban notification: {e}")
    
    async def _notify_moderation_action(
        self,
        action: ModerationAction,
        content_type: ContentType,
        content_id: str,
        user_id: Optional[int],
        reason: str
    ):
        """Notify moderators about automated moderation action."""
        try:
            await self.event_bus.emit(EventType.MODERATION_ACTION_TAKEN, {
                "action": action.value,
                "content_type": content_type.value,
                "content_id": content_id,
                "user_id": user_id,
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat(),
                "automated": True
            })
        except Exception as e:
            logger.error(f"Failed to notify moderation action: {e}")
    
    # Event Handlers
    
    async def handle_incident_reported(self, event):
        """Handle newly reported incidents."""
        incident_id = event.data.get("incident_id")
        description = event.data.get("description", "")
        user_id = event.user_id
        
        # Check content
        matched_rules, severity = await self._check_content(
            description,
            ContentType.INCIDENT,
            user_id
        )
        
        if matched_rules:
            # Determine action based on severity
            if severity == ModerationSeverity.CRITICAL:
                action = ModerationAction.REMOVE_CONTENT_AND_SUSPEND
            elif severity == ModerationSeverity.HIGH:
                action = ModerationAction.REMOVE_CONTENT_AND_WARN
            elif severity == ModerationSeverity.MEDIUM:
                action = ModerationAction.FLAG_FOR_REVIEW
            else:
                action = ModerationAction.WARN_USER
            
            reason = f"Matched rules: {', '.join(matched_rules)}"
            
            await self._apply_moderation_action(
                action=action,
                content_type=ContentType.INCIDENT,
                content_id=incident_id,
                user_id=user_id,
                reason=reason,
                severity=severity,
                matched_rules=matched_rules
            )
    
    async def handle_incident_updated(self, event):
        """Handle updated incidents."""
        incident_id = event.data.get("incident_id")
        description = event.data.get("description", "")
        user_id = event.user_id
        
        # Check updated content
        matched_rules, severity = await self._check_content(
            description,
            ContentType.INCIDENT,
            user_id
        )
        
        if matched_rules:
            action = ModerationAction.FLAG_FOR_REVIEW if severity == ModerationSeverity.MEDIUM else ModerationAction.WARN_USER
            reason = f"Update matched rules: {', '.join(matched_rules)}"
            
            await self._apply_moderation_action(
                action=action,
                content_type=ContentType.INCIDENT,
                content_id=incident_id,
                user_id=user_id,
                reason=reason,
                severity=severity,
                matched_rules=matched_rules
            )
    
    async def handle_comment_created(self, event):
        """Handle new comments."""
        comment_id = event.data.get("comment_id")
        content = event.data.get("content", "")
        user_id = event.user_id
        
        matched_rules, severity = await self._check_content(
            content,
            ContentType.COMMENT,
            user_id
        )
        
        if matched_rules:
            if severity == ModerationSeverity.CRITICAL:
                action = ModerationAction.REMOVE_CONTENT_AND_SUSPEND
            elif severity == ModerationSeverity.HIGH:
                action = ModerationAction.REMOVE_CONTENT_AND_WARN
            elif severity == ModerationSeverity.MEDIUM:
                action = ModerationAction.FLAG_FOR_REVIEW
            else:
                action = ModerationAction.WARN_USER
            
            reason = f"Comment matched rules: {', '.join(matched_rules)}"
            
            await self._apply_moderation_action(
                action=action,
                content_type=ContentType.COMMENT,
                content_id=comment_id,
                user_id=user_id,
                reason=reason,
                severity=severity,
                matched_rules=matched_rules
            )
    
    async def handle_comment_updated(self, event):
        """Handle updated comments."""
        await self.handle_comment_created(event)  # Same logic
    
    async def handle_chat_message(self, event):
        """Handle chat messages."""
        message_id = event.data.get("message_id")
        content = event.data.get("content", "")
        user_id = event.user_id
        
        matched_rules, severity = await self._check_content(
            content,
            ContentType.CHAT,
            user_id
        )
        
        if matched_rules:
            # For chat, we might want different actions
            if severity in [ModerationSeverity.CRITICAL, ModerationSeverity.HIGH]:
                action = ModerationAction.REMOVE_CONTENT_AND_WARN
            elif severity == ModerationSeverity.MEDIUM:
                action = ModerationAction.WARN_USER
            else:
                action = ModerationAction.NO_ACTION  # Just log it
            
            if action != ModerationAction.NO_ACTION:
                reason = f"Chat message matched rules: {', '.join(matched_rules)}"
                
                await self._apply_moderation_action(
                    action=action,
                    content_type=ContentType.CHAT,
                    content_id=message_id,
                    user_id=user_id,
                    reason=reason,
                    severity=severity,
                    matched_rules=matched_rules
                )
    
    async def handle_user_created(self, event):
        """Handle new user registration."""
        user_id = event.user_id
        
        # Check for suspicious patterns in registration data
        username = event.data.get("username", "")
        email = event.data.get("email", "")
        
        # Simple spam detection for usernames
        spam_patterns = [
            r"^\d+$",  # All numbers
            r"^[a-z0-9]{20,}$",  # Very long alphanumeric
            r".*(bot|spam|fake|test).*",  # Contains bot/spam keywords
        ]
        
        for pattern in spam_patterns:
            if re.match(pattern, username, re.IGNORECASE):
                logger.warning(f"Suspicious username detected for user {user_id}: {username}")
                
                # Flag user for manual review
                async with AsyncSessionLocal() as session:
                    user = await session.get(User, user_id)
                    if user:
                        user.needs_moderation_review = True
                        await session.commit()
                
                break
    
    async def handle_user_reported(self, event):
        """Handle user reports."""
        reported_user_id = event.data.get("reported_user_id")
        reason = event.data.get("reason", "")
        reporter_id = event.user_id
        
        # Check report reason for patterns
        matched_rules, severity = await self._check_content(
            reason,
            ContentType.COMMENT,  # Using comment type for text checking
            reporter_id
        )
        
        # If the report itself contains violations, moderate the reporter
        if matched_rules:
            action = ModerationAction.WARN_USER
            reason_text = f"Report contained violations: {', '.join(matched_rules)}"
            
            await self._apply_moderation_action(
                action=action,
                content_type=ContentType.USER,  # User-level action
                content_id=str(reporter_id),
                user_id=reporter_id,
                reason=reason_text,
                severity=severity,
                matched_rules=matched_rules
            )
        
        # Analyze the reported user's behavior
        violation_count = self.user_violations_cache.get(reported_user_id, 0)
        
        # If user has multiple violations, flag for review
        if violation_count >= 3:
            async with AsyncSessionLocal() as session:
                user = await session.get(User, reported_user_id)
                if user:
                    user.needs_moderation_review = True
                    await session.commit()
    
    async def handle_content_reported(self, event):
        """Handle content reports."""
        content_type = event.data.get("content_type")
        content_id = event.data.get("content_id")
        reason = event.data.get("reason", "")
        reporter_id = event.user_id
        
        # Convert string to ContentType enum
        try:
            content_type_enum = ContentType(content_type)
        except ValueError:
            logger.warning(f"Invalid content type in report: {content_type}")
            return
        
        # Check report reason
        matched_rules, severity = await self._check_content(
            reason,
            ContentType.COMMENT,
            reporter_id
        )
        
        # If report contains violations, warn reporter
        if matched_rules:
            action = ModerationAction.WARN_USER
            reason_text = f"Report contained violations: {', '.join(matched_rules)}"
            
            await self._apply_moderation_action(
                action=action,
                content_type=ContentType.USER,
                content_id=str(reporter_id),
                user_id=reporter_id,
                reason=reason_text,
                severity=severity,
                matched_rules=matched_rules
            )
        
        # Flag reported content for manual review
        try:
            async with AsyncSessionLocal() as session:
                if content_type_enum == ContentType.INCIDENT:
                    incident = await session.get(Incident, content_id)
                    if incident:
                        incident.needs_moderation_review = True
                elif content_type_enum == ContentType.COMMENT:
                    comment = await session.get(Comment, content_id)
                    if comment:
                        comment.needs_moderation_review = True
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to flag content for review: {e}")
    
    async def handle_moderation_action(self, event):
        """Handle manual moderation action requests."""
        # This would be triggered when a moderator manually requests an action
        # For now, just log it
        logger.info(f"Manual moderation action requested: {event.data}")
    
    async def handle_auto_moderation_check(self, event):
        """Handle automated moderation check requests."""
        content_type = event.data.get("content_type")
        content_id = event.data.get("content_id")
        
        try:
            content_type_enum = ContentType(content_type)
        except ValueError:
            logger.warning(f"Invalid content type in auto check: {content_type}")
            return
        
        # Fetch content and check it
        async with AsyncSessionLocal() as session:
            if content_type_enum == ContentType.INCIDENT:
                incident = await session.get(Incident, content_id)
                if incident and incident.description:
                    matched_rules, severity = await self._check_content(
                        incident.description,
                        ContentType.INCIDENT,
                        incident.user_id
                    )
                    
                    if matched_rules:
                        # Apply appropriate action
                        if severity in [ModerationSeverity.CRITICAL, ModerationSeverity.HIGH]:
                            action = ModerationAction.REMOVE_CONTENT_AND_WARN
                        else:
                            action = ModerationAction.FLAG_FOR_REVIEW
                        
                        await self._apply_moderation_action(
                            action=action,
                            content_type=ContentType.INCIDENT,
                            content_id=content_id,
                            user_id=incident.user_id,
                            reason=f"Auto-check matched rules: {', '.join(matched_rules)}",
                            severity=severity,
                            matched_rules=matched_rules
                        )
    
    async def _periodic_moderation_scan(self):
        """Periodically scan for content needing moderation."""
        while True:
            try:
                logger.info("Starting periodic moderation scan")
                
                # Scan recent incidents
                await self._scan_recent_incidents()
                
                # Scan recent comments
                await self._scan_recent_comments()
                
                # Clean up old cache entries
                await self._cleanup_violation_cache()
                
                logger.info("Periodic moderation scan completed")
                
            except Exception as e:
                logger.error(f"Error in periodic moderation scan: {e}", exc_info=True)
            
            # Run every 30 minutes
            await asyncio.sleep(1800)
    
    async def _scan_recent_incidents(self):
        """Scan recent incidents for moderation."""
        try:
            async with AsyncSessionLocal() as session:
                # Get incidents from last 24 hours that haven't been moderated
                cutoff = datetime.utcnow() - timedelta(hours=24)
                
                result = await session.execute(
                    select(Incident)
                    .where(Incident.created_at >= cutoff)
                    .where(Incident.needs_moderation_review == False)
                    .where(Incident.status != IncidentStatus.REMOVED)
                    .limit(100)
                )
                
                incidents = result.scalars().all()
                
                for incident in incidents:
                    if incident.description:
                        matched_rules, severity = await self._check_content(
                            incident.description,
                            ContentType.INCIDENT,
                            incident.user_id
                        )
                        
                        if matched_rules and severity in [ModerationSeverity.CRITICAL, ModerationSeverity.HIGH]:
                            # Auto-remove critical/high severity content
                            incident.status = IncidentStatus.REMOVED
                            incident.moderation_notes = f"Auto-removed: Matched rules {', '.join(matched_rules)}"
                            
                            # Log the action
                            await self._log_moderation_action(
                                action=ModerationAction.REMOVE_CONTENT,
                                content_type=ContentType.INCIDENT,
                                content_id=str(incident.id),
                                user_id=incident.user_id,
                                reason=f"Periodic scan matched rules: {', '.join(matched_rules)}",
                                severity=severity,
                                matched_rules=matched_rules
                            )
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Error scanning recent incidents: {e}")
    
    async def _scan_recent_comments(self):
        """Scan recent comments for moderation."""
        try:
            async with AsyncSessionLocal() as session:
                from app.db.models.comment import Comment
                
                cutoff = datetime.utcnow() - timedelta(hours=24)
                
                result = await session.execute(
                    select(Comment)
                    .where(Comment.created_at >= cutoff)
                    .where(Comment.is_removed == False)
                    .where(Comment.needs_moderation_review == False)
                    .limit(200)
                )
                
                comments = result.scalars().all()
                
                for comment in comments:
                    if comment.content:
                        matched_rules, severity = await self._check_content(
                            comment.content,
                            ContentType.COMMENT,
                            comment.user_id
                        )
                        
                        if matched_rules and severity == ModerationSeverity.CRITICAL:
                            # Auto-remove critical content
                            comment.is_removed = True
                            comment.removal_reason = f"Auto-removed: Matched rules {', '.join(matched_rules)}"
                            
                            # Log the action
                            await self._log_moderation_action(
                                action=ModerationAction.REMOVE_CONTENT,
                                content_type=ContentType.COMMENT,
                                content_id=str(comment.id),
                                user_id=comment.user_id,
                                reason=f"Periodic scan matched rules: {', '.join(matched_rules)}",
                                severity=severity,
                                matched_rules=matched_rules
                            )
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Error scanning recent comments: {e}")
    
    async def _cleanup_violation_cache(self):
        """Clean up old entries from violation cache."""
        # Keep only entries from last 30 days in memory
        # In production, you'd use Redis or database for this
        cutoff = datetime.utcnow() - timedelta(days=30)
        
        # This is a simple in-memory implementation
        # In production, use proper caching with TTL
        logger.debug(f"Violation cache size before cleanup: {len(self.user_violations_cache)}")
        
        # For now, just log - implement proper cleanup with Redis/database
        pass
    
    # Public API methods for manual moderation
    
    async def check_text(self, text: str, content_type: ContentType, user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Check text against moderation rules.
        
        Args:
            text: Text to check
            content_type: Type of content
            user_id: Optional user ID
            
        Returns:
            Dictionary with check results
        """
        matched_rules, severity = await self._check_content(text, content_type, user_id)
        
        return {
            "has_violations": len(matched_rules) > 0,
            "matched_rules": matched_rules,
            "severity": severity.value,
            "recommended_action": self._get_recommended_action(severity, len(matched_rules)),
            "safe_text": self._redact_personal_info(text) if matched_rules else text
        }
    
    def _get_recommended_action(self, severity: ModerationSeverity, rule_count: int) -> str:
        """Get recommended action based on severity and rule count."""
        if severity == ModerationSeverity.CRITICAL:
            return "remove_and_suspend"
        elif severity == ModerationSeverity.HIGH:
            return "remove_and_warn" if rule_count > 1 else "remove_content"
        elif severity == ModerationSeverity.MEDIUM:
            return "flag_for_review"
        elif severity == ModerationSeverity.LOW:
            return "warn_user" if rule_count > 2 else "no_action"
        return "no_action"
    
    async def get_user_moderation_history(self, user_id: int) -> Dict[str, Any]:
        """
        Get moderation history for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            User moderation history
        """
        try:
            async with AsyncSessionLocal() as session:
                # Get user
                user_result = await session.execute(
                    select(User).where(User.id == user_id)
                )
                user = user_result.scalar_one_or_none()
                
                if not user:
                    return {"error": "User not found"}
                
                # Get moderation logs for user
                logs_result = await session.execute(
                    select(ModerationLog)
                    .where(ModerationLog.user_id == user_id)
                    .order_by(desc(ModerationLog.created_at))
                    .limit(50)
                )
                logs = logs_result.scalars().all()
                
                # Get violation count
                violation_count = self.user_violations_cache.get(user_id, 0)
                if hasattr(user, 'moderation_violation_count'):
                    violation_count = user.moderation_violation_count or violation_count
                
                return {
                    "user_id": user_id,
                    "username": user.username,
                    "email": user.email,
                    "is_banned": user.is_banned if hasattr(user, 'is_banned') else False,
                    "is_suspended": user.is_suspended if hasattr(user, 'is_suspended') else False,
                    "suspended_until": user.suspended_until.isoformat() if hasattr(user, 'suspended_until') and user.suspended_until else None,
                    "warning_count": user.warning_count if hasattr(user, 'warning_count') else 0,
                    "violation_count": violation_count,
                    "moderation_logs": [
                        {
                            "action": log.action.value,
                            "content_type": log.content_type.value,
                            "content_id": log.content_id,
                            "reason": log.reason,
                            "severity": log.severity,
                            "created_at": log.created_at.isoformat(),
                            "matched_rules": json.loads(log.matched_rules) if log.matched_rules else []
                        }
                        for log in logs
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error getting user moderation history: {e}")
            return {"error": str(e)}


# Factory function
async def create_moderation_subscriber(event_bus: EventBus) -> ModerationSubscriber:
    """Create and initialize a moderation subscriber."""
    subscriber = ModerationSubscriber(event_bus)
    await subscriber.initialize()
    return subscriber