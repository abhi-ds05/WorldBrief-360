"""
Alert Management System

This module provides alerting capabilities for monitoring anomalies,
threshold violations, and system issues. It supports:

- Configurable alert rules and thresholds
- Multiple notification channels (email, Slack, PagerDuty, etc.)
- Alert deduplication and suppression
- Alert escalation policies
- Alert history and persistence
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
from threading import Lock, Timer
from collections import defaultdict, deque

import redis
import pydantic
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Severity levels for alerts."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    
    @classmethod
    def from_string(cls, severity: str) -> 'AlertSeverity':
        """Convert string to AlertSeverity enum."""
        severity_lower = severity.lower()
        for sev in cls:
            if sev.value == severity_lower:
                return sev
        raise ValueError(f"Invalid severity: {severity}")


class AlertStatus(Enum):
    """Status of an alert."""
    FIRING = "firing"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"


@dataclass
class AlertLabels:
    """Labels for identifying and grouping alerts."""
    alertname: str
    severity: AlertSeverity
    service: str = "worldbrief-360"
    component: Optional[str] = None
    instance: Optional[str] = None
    environment: Optional[str] = None
    custom_labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, str]:
        """Convert labels to dictionary."""
        result = {
            "alertname": self.alertname,
            "severity": self.severity.value,
            "service": self.service,
        }
        
        if self.component:
            result["component"] = self.component
        if self.instance:
            result["instance"] = self.instance
        if self.environment:
            result["environment"] = self.environment
        
        result.update(self.custom_labels)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'AlertLabels':
        """Create AlertLabels from dictionary."""
        kwargs = {
            "alertname": data["alertname"],
            "severity": AlertSeverity.from_string(data["severity"]),
            "service": data.get("service", "worldbrief-360"),
            "component": data.get("component"),
            "instance": data.get("instance"),
            "environment": data.get("environment"),
            "custom_labels": {k: v for k, v in data.items() 
                            if k not in ["alertname", "severity", "service", 
                                       "component", "instance", "environment"]}
        }
        return cls(**kwargs)


@dataclass
class AlertAnnotation:
    """Additional information about an alert."""
    summary: str
    description: Optional[str] = None
    runbook_url: Optional[str] = None
    dashboard_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, str]:
        """Convert annotations to dictionary."""
        result = {"summary": self.summary}
        if self.description:
            result["description"] = self.description
        if self.runbook_url:
            result["runbook_url"] = self.runbook_url
        if self.dashboard_url:
            result["dashboard_url"] = self.dashboard_url
        return result


class Alert(BaseModel):
    """Alert model representing a single alert instance."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    labels: AlertLabels
    annotations: AlertAnnotation
    status: AlertStatus = AlertStatus.FIRING
    starts_at: datetime = Field(default_factory=datetime.utcnow)
    ends_at: Optional[datetime] = None
    generator_url: Optional[str] = None
    fingerprint: Optional[str] = None
    silenced: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    last_notified_at: Optional[datetime] = None
    notification_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            AlertSeverity: lambda sev: sev.value,
            AlertStatus: lambda status: status.value,
        }
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Get alert duration."""
        if self.ends_at:
            return self.ends_at - self.starts_at
        return datetime.utcnow() - self.starts_at
    
    @property
    def is_active(self) -> bool:
        """Check if alert is active."""
        return self.status == AlertStatus.FIRING and not self.silenced
    
    def acknowledge(self, user: str) -> None:
        """Acknowledge the alert."""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_by = user
        self.acknowledged_at = datetime.utcnow()
    
    def resolve(self) -> None:
        """Resolve the alert."""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.utcnow()
        self.ends_at = datetime.utcnow()
    
    def suppress(self) -> None:
        """Suppress the alert."""
        self.silenced = True
    
    def unsuppress(self) -> None:
        """Unsuppress the alert."""
        self.silenced = False
    
    def to_prometheus_format(self) -> Dict[str, Any]:
        """Convert to Prometheus alert format."""
        return {
            "labels": self.labels.to_dict(),
            "annotations": self.annotations.to_dict(),
            "state": self.status.value.upper(),
            "activeAt": self.starts_at.isoformat(),
            "value": "1" if self.is_active else "0",
        }


class AlertRule(BaseModel):
    """Rule for generating alerts based on conditions."""
    name: str
    description: str
    severity: AlertSeverity
    condition: Callable[[Dict[str, Any]], bool]
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)
    for_duration: timedelta = Field(default=timedelta(seconds=0))
    check_interval: timedelta = Field(default=timedelta(seconds=30))
    enabled: bool = True
    suppress_duplicates_for: timedelta = Field(default=timedelta(minutes=5))
    
    class Config:
        arbitrary_types_allowed = True
    
    def evaluate(self, data: Dict[str, Any]) -> Optional[Alert]:
        """Evaluate rule against data."""
        if not self.enabled:
            return None
        
        try:
            if self.condition(data):
                labels = AlertLabels.from_dict({
                    "alertname": self.name,
                    "severity": self.severity.value,
                    **self.labels
                })
                
                annotations = AlertAnnotation(
                    summary=self.annotations.get("summary", self.description),
                    description=self.annotations.get("description"),
                    runbook_url=self.annotations.get("runbook_url"),
                    dashboard_url=self.annotations.get("dashboard_url")
                )
                
                return Alert(labels=labels, annotations=annotations)
        except Exception as e:
            logger.error(f"Error evaluating alert rule '{self.name}': {e}")
        
        return None


class NotificationChannel(BaseModel):
    """Configuration for a notification channel."""
    name: str
    channel_type: str  # "email", "slack", "pagerduty", "webhook", "sms"
    config: Dict[str, Any]
    enabled: bool = True
    severity_filter: List[AlertSeverity] = Field(default_factory=list)
    rate_limit: Optional[int] = None  # Max notifications per minute
    last_notification_time: Optional[datetime] = None
    notification_count: int = 0
    
    def should_notify(self, alert: Alert) -> bool:
        """Check if this channel should notify for the given alert."""
        if not self.enabled:
            return False
        
        if self.severity_filter and alert.labels.severity not in self.severity_filter:
            return False
        
        # Check rate limiting
        if self.rate_limit and self.last_notification_time:
            time_since_last = datetime.utcnow() - self.last_notification_time
            if time_since_last.total_seconds() < 60 / self.rate_limit:
                return False
        
        return True
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send notification for an alert."""
        try:
            if self.channel_type == "slack":
                await self._send_slack_notification(alert)
            elif self.channel_type == "email":
                await self._send_email_notification(alert)
            elif self.channel_type == "pagerduty":
                await self._send_pagerduty_notification(alert)
            elif self.channel_type == "webhook":
                await self._send_webhook_notification(alert)
            elif self.channel_type == "sms":
                await self._send_sms_notification(alert)
            else:
                logger.warning(f"Unknown notification channel type: {self.channel_type}")
                return False
            
            self.last_notification_time = datetime.utcnow()
            self.notification_count += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to send notification via {self.name}: {e}")
            return False
    
    async def _send_slack_notification(self, alert: Alert) -> None:
        """Send notification to Slack."""
        import aiohttp
        
        webhook_url = self.config.get("webhook_url")
        if not webhook_url:
            raise ValueError("Slack webhook URL not configured")
        
        severity_color = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ffcc00",
            AlertSeverity.ERROR: "#ff9900",
            AlertSeverity.CRITICAL: "#ff0000",
        }
        
        message = {
            "attachments": [{
                "color": severity_color.get(alert.labels.severity, "#36a64f"),
                "title": f"[{alert.labels.severity.value.upper()}] {alert.annotations.summary}",
                "text": alert.annotations.description or "",
                "fields": [
                    {"title": "Service", "value": alert.labels.service, "short": True},
                    {"title": "Component", "value": alert.labels.component or "N/A", "short": True},
                    {"title": "Alert Name", "value": alert.labels.alertname, "short": True},
                    {"title": "Environment", "value": alert.labels.environment or "N/A", "short": True},
                    {"title": "Status", "value": alert.status.value.upper(), "short": True},
                    {"title": "Started", "value": alert.starts_at.strftime("%Y-%m-%d %H:%M:%S UTC"), "short": True},
                ],
                "footer": "WorldBrief 360 Monitoring",
                "ts": int(time.time()),
            }]
        }
        
        if alert.annotations.runbook_url:
            message["attachments"][0]["actions"] = [{
                "type": "button",
                "text": "View Runbook",
                "url": alert.annotations.runbook_url
            }]
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=message) as response:
                response.raise_for_status()
    
    async def _send_email_notification(self, alert: Alert) -> None:
        """Send notification via email."""
        # Implementation depends on your email service
        # Example using SMTP
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        smtp_config = self.config
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[{alert.labels.severity.value.upper()}] {alert.annotations.summary}"
        msg["From"] = smtp_config.get("from_email")
        msg["To"] = ", ".join(smtp_config.get("to_emails", []))
        
        # Create HTML email
        html = f"""
        <html>
        <body>
            <h2>Alert: {alert.annotations.summary}</h2>
            <p><strong>Severity:</strong> {alert.labels.severity.value.upper()}</p>
            <p><strong>Service:</strong> {alert.labels.service}</p>
            <p><strong>Component:</strong> {alert.labels.component or 'N/A'}</p>
            <p><strong>Description:</strong> {alert.annotations.description or 'No description'}</p>
            <p><strong>Started:</strong> {alert.starts_at}</p>
            <p><strong>Status:</strong> {alert.status.value.upper()}</p>
        """
        
        if alert.annotations.runbook_url:
            html += f'<p><a href="{alert.annotations.runbook_url}">View Runbook</a></p>'
        
        html += """
        </body>
        </html>
        """
        
        msg.attach(MIMEText(html, "html"))
        
        # Send email
        with smtplib.SMTP(smtp_config.get("smtp_host"), smtp_config.get("smtp_port", 587)) as server:
            server.starttls()
            server.login(smtp_config.get("smtp_username"), smtp_config.get("smtp_password"))
            server.send_message(msg)
    
    async def _send_pagerduty_notification(self, alert: Alert) -> None:
        """Send notification to PagerDuty."""
        import aiohttp
        
        api_key = self.config.get("api_key")
        service_id = self.config.get("service_id")
        
        if not api_key or not service_id:
            raise ValueError("PagerDuty API key or service ID not configured")
        
        severity_map = {
            AlertSeverity.INFO: "info",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.ERROR: "error",
            AlertSeverity.CRITICAL: "critical",
        }
        
        payload = {
            "routing_key": api_key,
            "event_action": "trigger",
            "dedup_key": alert.id,
            "payload": {
                "summary": alert.annotations.summary,
                "source": alert.labels.service,
                "severity": severity_map.get(alert.labels.severity, "info"),
                "component": alert.labels.component,
                "custom_details": {
                    "alert_id": alert.id,
                    "description": alert.annotations.description,
                    "environment": alert.labels.environment,
                    "status": alert.status.value,
                },
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                response.raise_for_status()
    
    async def _send_webhook_notification(self, alert: Alert) -> None:
        """Send notification to webhook."""
        import aiohttp
        
        webhook_url = self.config.get("url")
        if not webhook_url:
            raise ValueError("Webhook URL not configured")
        
        payload = {
            "alert": alert.dict(),
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "alert_notification",
        }
        
        headers = self.config.get("headers", {})
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload, headers=headers) as response:
                response.raise_for_status()
    
    async def _send_sms_notification(self, alert: Alert) -> None:
        """Send notification via SMS."""
        # Implementation depends on your SMS provider
        # Example using Twilio
        from twilio.rest import Client
        
        config = self.config
        client = Client(config.get("account_sid"), config.get("auth_token"))
        
        message = f"[{alert.labels.severity.value.upper()}] {alert.annotations.summary}"
        if len(message) > 160:  # SMS character limit
            message = message[:157] + "..."
        
        for phone_number in config.get("phone_numbers", []):
            client.messages.create(
                body=message,
                from_=config.get("from_number"),
                to=phone_number
            )


class AlertManager:
    """Main alert manager that handles alert rules, notifications, and state."""
    
    def __init__(
        self,
        service_name: str = "worldbrief-360",
        redis_client: Optional[redis.Redis] = None,
        alert_ttl: timedelta = timedelta(days=7),
        **kwargs
    ):
        """
        Initialize the AlertManager.
        
        Args:
            service_name: Name of the service
            redis_client: Redis client for alert persistence
            alert_ttl: Time-to-live for alerts in storage
        """
        self.service_name = service_name
        self.redis_client = redis_client
        self.alert_ttl = alert_ttl
        
        # Storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)  # In-memory history
        self.alert_rules: Dict[str, AlertRule] = {}
        self.notification_channels: Dict[str, NotificationChannel] = {}
        
        # State
        self._lock = Lock()
        self._running = False
        self._evaluation_tasks: List[asyncio.Task] = []
        
        # Statistics
        self.stats = {
            "alerts_triggered": 0,
            "alerts_resolved": 0,
            "notifications_sent": 0,
            "evaluation_errors": 0,
        }
        
        # Load default rules
        self._load_default_rules()
        
        logger.info(f"AlertManager initialized for service: {service_name}")
    
    def _load_default_rules(self) -> None:
        """Load default alert rules."""
        # High error rate rule
        self.add_rule(AlertRule(
            name="high_error_rate",
            description="Error rate is above threshold",
            severity=AlertSeverity.ERROR,
            condition=lambda data: data.get("error_rate", 0) > 0.05,  # 5% error rate
            labels={"component": "api"},
            annotations={
                "summary": "High error rate detected",
                "description": "Error rate has exceeded 5% threshold",
                "runbook_url": "https://docs.worldbrief360.com/runbooks/high-error-rate"
            },
            for_duration=timedelta(minutes=5)
        ))
        
        # High latency rule
        self.add_rule(AlertRule(
            name="high_latency",
            description="API latency is above threshold",
            severity=AlertSeverity.WARNING,
            condition=lambda data: data.get("p95_latency_ms", 0) > 1000,  # 1 second
            labels={"component": "api"},
            annotations={
                "summary": "High latency detected",
                "description": "95th percentile latency has exceeded 1000ms",
                "runbook_url": "https://docs.worldbrief360.com/runbooks/high-latency"
            },
            for_duration=timedelta(minutes=2)
        ))
        
        # Service down rule
        self.add_rule(AlertRule(
            name="service_down",
            description="Service is not responding",
            severity=AlertSeverity.CRITICAL,
            condition=lambda data: data.get("health_status") == "down",
            labels={"component": "service"},
            annotations={
                "summary": "Service is down",
                "description": "Service is not responding to health checks",
                "runbook_url": "https://docs.worldbrief360.com/runbooks/service-down"
            }
        ))
        
        # Memory usage rule
        self.add_rule(AlertRule(
            name="high_memory_usage",
            description="Memory usage is above threshold",
            severity=AlertSeverity.WARNING,
            condition=lambda data: data.get("memory_usage_percent", 0) > 90,
            labels={"component": "infrastructure"},
            annotations={
                "summary": "High memory usage",
                "description": "Memory usage has exceeded 90%",
                "runbook_url": "https://docs.worldbrief360.com/runbooks/high-memory"
            },
            for_duration=timedelta(minutes=5)
        ))
        
        # CPU usage rule
        self.add_rule(AlertRule(
            name="high_cpu_usage",
            description="CPU usage is above threshold",
            severity=AlertSeverity.WARNING,
            condition=lambda data: data.get("cpu_usage_percent", 0) > 85,
            labels={"component": "infrastructure"},
            annotations={
                "summary": "High CPU usage",
                "description": "CPU usage has exceeded 85%",
                "runbook_url": "https://docs.worldbrief360.com/runbooks/high-cpu"
            },
            for_duration=timedelta(minutes=5)
        ))
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        with self._lock:
            self.alert_rules[rule.name] = rule
            logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove an alert rule."""
        with self._lock:
            if rule_name in self.alert_rules:
                del self.alert_rules[rule_name]
                logger.info(f"Removed alert rule: {rule_name}")
                return True
            return False
    
    def add_notification_channel(self, channel: NotificationChannel) -> None:
        """Add a notification channel."""
        with self._lock:
            self.notification_channels[channel.name] = channel
            logger.info(f"Added notification channel: {channel.name}")
    
    def remove_notification_channel(self, channel_name: str) -> bool:
        """Remove a notification channel."""
        with self._lock:
            if channel_name in self.notification_channels:
                del self.notification_channels[channel_name]
                logger.info(f"Removed notification channel: {channel_name}")
                return True
            return False
    
    def trigger_alert(self, alert: Alert) -> None:
        """
        Trigger a new alert.
        
        Args:
            alert: The alert to trigger
        """
        with self._lock:
            alert_id = alert.id
            
            # Check for duplicate alerts (same fingerprint)
            if alert.fingerprint and alert.fingerprint in self.active_alerts:
                existing_alert = self.active_alerts[alert.fingerprint]
                if existing_alert.is_active:
                    # Update existing alert
                    existing_alert.last_notified_at = datetime.utcnow()
                    existing_alert.notification_count += 1
                    logger.debug(f"Updated existing alert: {alert_id}")
                    return
            
            # Store alert
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Persist to Redis if available
            if self.redis_client:
                try:
                    alert_key = f"alert:{alert_id}"
                    self.redis_client.setex(
                        alert_key,
                        int(self.alert_ttl.total_seconds()),
                        alert.json()
                    )
                    
                    # Also store in active alerts set
                    self.redis_client.sadd("alerts:active", alert_id)
                except Exception as e:
                    logger.error(f"Failed to persist alert to Redis: {e}")
            
            self.stats["alerts_triggered"] += 1
            logger.info(f"Triggered alert: {alert.labels.alertname} (ID: {alert_id})")
        
        # Send notifications asynchronously
        asyncio.create_task(self._send_notifications(alert))
    
    async def _send_notifications(self, alert: Alert) -> None:
        """Send notifications for an alert."""
        notification_tasks = []
        
        for channel_name, channel in self.notification_channels.items():
            if channel.should_notify(alert):
                task = asyncio.create_task(
                    channel.send_notification(alert),
                    name=f"notification_{channel_name}_{alert.id}"
                )
                notification_tasks.append(task)
        
        if notification_tasks:
            results = await asyncio.gather(*notification_tasks, return_exceptions=True)
            
            success_count = sum(1 for r in results if r is True)
            self.stats["notifications_sent"] += success_count
            
            alert.last_notified_at = datetime.utcnow()
            alert.notification_count += 1
            
            logger.info(f"Sent {success_count}/{len(notification_tasks)} notifications for alert: {alert.id}")
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: ID of the alert to resolve
            
        Returns:
            bool: True if alert was resolved, False otherwise
        """
        with self._lock:
            if alert_id not in self.active_alerts:
                return False
            
            alert = self.active_alerts[alert_id]
            alert.resolve()
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            # Update Redis
            if self.redis_client:
                try:
                    self.redis_client.srem("alerts:active", alert_id)
                    alert_key = f"alert:{alert_id}"
                    self.redis_client.setex(
                        alert_key,
                        int(self.alert_ttl.total_seconds()),
                        alert.json()
                    )
                except Exception as e:
                    logger.error(f"Failed to update alert in Redis: {e}")
            
            self.stats["alerts_resolved"] += 1
            logger.info(f"Resolved alert: {alert_id}")
            
            # Send resolution notifications
            asyncio.create_task(self._send_notifications(alert))
            
            return True
    
    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: ID of the alert to acknowledge
            user: User who acknowledged the alert
            
        Returns:
            bool: True if alert was acknowledged, False otherwise
        """
        with self._lock:
            if alert_id not in self.active_alerts:
                return False
            
            alert = self.active_alerts[alert_id]
            alert.acknowledge(user)
            
            logger.info(f"Alert {alert_id} acknowledged by {user}")
            return True
    
    def suppress_alert(self, alert_id: str) -> bool:
        """
        Suppress an alert (silence it).
        
        Args:
            alert_id: ID of the alert to suppress
            
        Returns:
            bool: True if alert was suppressed, False otherwise
        """
        with self._lock:
            if alert_id not in self.active_alerts:
                return False
            
            alert = self.active_alerts[alert_id]
            alert.suppress()
            
            logger.info(f"Suppressed alert: {alert_id}")
            return True
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self._lock:
            return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        with self._lock:
            return list(self.alert_history)[-limit:]
    
    async def evaluate_rules(self, metrics_data: Dict[str, Any]) -> None:
        """
        Evaluate all alert rules against metrics data.
        
        Args:
            metrics_data: Dictionary containing metrics values
        """
        evaluation_tasks = []
        
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            task = asyncio.create_task(
                self._evaluate_rule(rule, metrics_data),
                name=f"rule_evaluation_{rule_name}"
            )
            evaluation_tasks.append(task)
        
        if evaluation_tasks:
            await asyncio.gather(*evaluation_tasks, return_exceptions=True)
    
    async def _evaluate_rule(self, rule: AlertRule, metrics_data: Dict[str, Any]) -> None:
        """Evaluate a single rule."""
        try:
            alert = rule.evaluate(metrics_data)
            if alert:
                self.trigger_alert(alert)
        except Exception as e:
            self.stats["evaluation_errors"] += 1
            logger.error(f"Error evaluating rule {rule.name}: {e}")
    
    async def start_periodic_evaluation(self, interval_seconds: int = 30) -> None:
        """
        Start periodic evaluation of alert rules.
        
        Args:
            interval_seconds: Evaluation interval in seconds
        """
        self._running = True
        
        while self._running:
            try:
                # Get metrics data (you need to implement this)
                metrics_data = await self._collect_metrics_data()
                
                # Evaluate rules
                await self.evaluate_rules(metrics_data)
                
                # Clean up old alerts
                self._cleanup_old_alerts()
                
            except Exception as e:
                logger.error(f"Error in periodic evaluation: {e}")
            
            await asyncio.sleep(interval_seconds)
    
    async def _collect_metrics_data(self) -> Dict[str, Any]:
        """
        Collect metrics data for rule evaluation.
        
        This should be implemented to collect metrics from your monitoring system.
        """
        # Placeholder - implement based on your metrics collection
        return {
            "error_rate": 0.0,
            "p95_latency_ms": 0,
            "health_status": "healthy",
            "memory_usage_percent": 0.0,
            "cpu_usage_percent": 0.0,
        }
    
    def _cleanup_old_alerts(self) -> None:
        """Clean up old resolved alerts."""
        current_time = datetime.utcnow()
        alerts_to_remove = []
        
        with self._lock:
            for alert_id, alert in self.active_alerts.items():
                if alert.status == AlertStatus.RESOLVED:
                    if alert.resolved_at and (current_time - alert.resolved_at) > self.alert_ttl:
                        alerts_to_remove.append(alert_id)
            
            for alert_id in alerts_to_remove:
                del self.active_alerts[alert_id]
        
        if alerts_to_remove:
            logger.info(f"Cleaned up {len(alerts_to_remove)} old alerts")
    
    def stop(self) -> None:
        """Stop the alert manager."""
        self._running = False
        
        # Cancel all tasks
        for task in self._evaluation_tasks:
            task.cancel()
        
        logger.info("AlertManager stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get alert manager statistics."""
        with self._lock:
            return {
                **self.stats,
                "active_alerts_count": len(self.active_alerts),
                "rules_count": len(self.alert_rules),
                "channels_count": len(self.notification_channels),
                "history_size": len(self.alert_history),
            }


def setup_alerts(
    service_name: str = "worldbrief-360",
    redis_url: Optional[str] = None,
    **kwargs
) -> AlertManager:
    """
    Set up and configure the AlertManager.
    
    Args:
        service_name: Name of the service
        redis_url: Redis URL for alert persistence
        **kwargs: Additional configuration
        
    Returns:
        AlertManager: Configured alert manager instance
    """
    redis_client = None
    
    if redis_url:
        try:
            redis_client = redis.from_url(redis_url)
            # Test connection
            redis_client.ping()
            logger.info("Connected to Redis for alert persistence")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Alerts will not be persisted.")
    
    alert_manager = AlertManager(
        service_name=service_name,
        redis_client=redis_client,
        **kwargs
    )
    
    # Add default notification channels from config
    notification_configs = kwargs.get("notification_channels", [])
    for channel_config in notification_configs:
        try:
            channel = NotificationChannel(**channel_config)
            alert_manager.add_notification_channel(channel)
        except Exception as e:
            logger.error(f"Failed to add notification channel: {e}")
    
    return alert_manager


# Example usage
if __name__ == "__main__":
    import asyncio
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    async def main():
        # Setup alert manager
        alert_manager = setup_alerts(
            service_name="test-service",
            redis_url="redis://localhost:6379"
        )
        
        # Add a custom notification channel
        alert_manager.add_notification_channel(
            NotificationChannel(
                name="console",
                channel_type="webhook",
                config={"url": "http://localhost:8080/webhook"},
                severity_filter=[AlertSeverity.CRITICAL, AlertSeverity.ERROR]
            )
        )
        
        # Create and trigger a test alert
        test_alert = Alert(
            labels=AlertLabels(
                alertname="test_alert",
                severity=AlertSeverity.CRITICAL,
                component="testing",
                environment="development"
            ),
            annotations=AlertAnnotation(
                summary="Test Alert",
                description="This is a test alert",
                runbook_url="https://example.com/runbook"
            )
        )
        
        alert_manager.trigger_alert(test_alert)
        print(f"Triggered alert: {test_alert.id}")
        
        # Get stats
        print(f"Stats: {alert_manager.get_stats()}")
        
        # Start periodic evaluation
        evaluation_task = asyncio.create_task(
            alert_manager.start_periodic_evaluation(interval_seconds=10)
        )
        
        # Wait a bit
        await asyncio.sleep(30)
        
        # Stop
        alert_manager.stop()
        evaluation_task.cancel()
        
        print("Alert manager test completed")
    
    asyncio.run(main())