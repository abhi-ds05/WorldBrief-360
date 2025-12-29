# backend/app/integrations/email_client.py
"""
Email service integration for WorldBrief 360.
Supports multiple email providers: SendGrid, SMTP, AWS SES, etc.
"""

import asyncio
import json
import smtplib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.utils import formatdate, make_msgid
from pathlib import Path
import aiofiles

import aiohttp
from pydantic import BaseModel, Field, validator, EmailStr, HttpUrl
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from jinja2 import Environment, FileSystemLoader, select_autoescape
import premailer
import markdown2

from app.core.config import settings
from app.core.logging_config import logger
from app.services.utils.http_client import AsyncHTTPClient
from app.schemas.request.notifications import EmailRequest


class EmailProvider(Enum):
    """Supported email providers."""
    SMTP = "smtp"
    SENDGRID = "sendgrid"
    AWS_SES = "ses"
    MAILGUN = "mailgun"
    POSTMARK = "postmark"
    RESEND = "resend"
    LOG = "log"  # For development - logs instead of sending


class EmailPriority(Enum):
    """Email priority levels."""
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class EmailType(Enum):
    """Types of emails sent by the system."""
    WELCOME = "welcome"
    VERIFICATION = "verification"
    PASSWORD_RESET = "password_reset"
    NOTIFICATION = "notification"
    NEWSLETTER = "newsletter"
    INCIDENT_REPORT = "incident_report"
    VERIFICATION_UPDATE = "verification_update"
    WALLET_TRANSACTION = "wallet_transaction"
    ADMIN_ALERT = "admin_alert"
    SYSTEM = "system"


@dataclass
class EmailAttachment:
    """Email attachment."""
    filename: str
    content: bytes
    content_type: str
    content_id: Optional[str] = None  # For inline images


class EmailRecipient(BaseModel):
    """Email recipient information."""
    email: EmailStr
    name: Optional[str] = None
    
    def format(self) -> str:
        """Format for email header."""
        if self.name:
            return f'"{self.name}" <{self.email}>'
        return self.email


class EmailMessage(BaseModel):
    """Complete email message."""
    to: List[EmailRecipient]
    cc: List[EmailRecipient] = Field(default_factory=list)
    bcc: List[EmailRecipient] = Field(default_factory=list)
    from_email: EmailRecipient
    subject: str
    html_body: Optional[str] = None
    text_body: Optional[str] = None
    reply_to: Optional[EmailRecipient] = None
    priority: EmailPriority = EmailPriority.NORMAL
    category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    attachments: List[EmailAttachment] = Field(default_factory=list)
    template_id: Optional[str] = None
    template_data: Dict[str, Any] = Field(default_factory=dict)
    send_at: Optional[datetime] = None
    
    @validator('subject')
    def validate_subject(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Email subject cannot be empty')
        return v.strip()
    
    @validator('html_body', 'text_body', pre=True, always=True)
    def validate_body(cls, v, values, field):
        # At least one body must be provided
        if field.name == 'html_body' and not v and not values.get('text_body'):
            raise ValueError('Either html_body or text_body must be provided')
        return v


class EmailResponse(BaseModel):
    """Email sending response."""
    message_id: str
    provider: EmailProvider
    status: str
    recipient_count: int
    queued_at: datetime = Field(default_factory=datetime.now)
    provider_response: Optional[Dict[str, Any]] = None
    errors: List[str] = Field(default_factory=list)
    
    def is_success(self) -> bool:
        """Check if email was sent successfully."""
        return self.status == "sent" or self.status == "queued"


class BaseEmailProvider:
    """Base class for email providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.http_client = AsyncHTTPClient()
        
    async def send_email(self, message: EmailMessage) -> EmailResponse:
        """
        Send an email.
        
        Args:
            message: Email message to send
            
        Returns:
            EmailResponse with send status
        """
        raise NotImplementedError("Subclasses must implement send_email")
    
    async def validate_connection(self) -> bool:
        """
        Validate connection to email provider.
        
        Returns:
            True if connection is valid
        """
        raise NotImplementedError("Subclasses must implement validate_connection")
    
    async def get_quota_info(self) -> Dict[str, Any]:
        """
        Get quota/usage information.
        
        Returns:
            Quota information
        """
        return {"provider": self.name, "quota": "unlimited"}
    
    def __str__(self) -> str:
        return f"{self.name} Email Provider"


class SMTPProvider(BaseEmailProvider):
    """SMTP email provider."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {
                "host": settings.SMTP_HOST,
                "port": settings.SMTP_PORT,
                "username": settings.SMTP_USERNAME,
                "password": settings.SMTP_PASSWORD,
                "use_tls": settings.SMTP_USE_TLS,
                "use_ssl": settings.SMTP_USE_SSL,
                "timeout": 30,
            }
        super().__init__(config)
        self.connection = None
        
    async def _get_connection(self) -> smtplib.SMTP:
        """Get or create SMTP connection."""
        if self.connection is None or not self._is_connection_alive():
            self.connection = await self._create_connection()
        return self.connection
    
    async def _create_connection(self) -> smtplib.SMTP:
        """Create new SMTP connection."""
        try:
            if self.config.get("use_ssl"):
                smtp_class = smtplib.SMTP_SSL
            else:
                smtp_class = smtplib.SMTP
            
            # Run in thread pool since smtplib is synchronous
            loop = asyncio.get_event_loop()
            connection = await loop.run_in_executor(
                None,
                lambda: smtp_class(
                    host=self.config["host"],
                    port=self.config["port"],
                    timeout=self.config.get("timeout", 30)
                )
            )
            
            # Start TLS if required
            if self.config.get("use_tls") and not self.config.get("use_ssl"):
                await loop.run_in_executor(None, connection.starttls)
            
            # Login if credentials provided
            if self.config.get("username") and self.config.get("password"):
                await loop.run_in_executor(
                    None,
                    lambda: connection.login(
                        self.config["username"],
                        self.config["password"]
                    )
                )
            
            logger.info(f"Connected to SMTP server {self.config['host']}:{self.config['port']}")
            return connection
            
        except Exception as e:
            logger.error(f"Failed to connect to SMTP server: {str(e)}")
            raise
    
    def _is_connection_alive(self) -> bool:
        """Check if SMTP connection is still alive."""
        if self.connection is None:
            return False
        
        try:
            # Try a NOOP command
            self.connection.noop()
            return True
        except:
            return False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def send_email(self, message: EmailMessage) -> EmailResponse:
        """Send email via SMTP."""
        try:
            # Create MIME message
            mime_message = await self._create_mime_message(message)
            
            # Get connection
            connection = await self._get_connection()
            
            # Convert recipients to list of email addresses
            recipients = [r.email for r in message.to + message.cc + message.bcc]
            
            # Send email
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: connection.sendmail(
                    message.from_email.email,
                    recipients,
                    mime_message.as_string()
                )
            )
            
            # Generate message ID
            message_id = mime_message['Message-ID'] or make_msgid()
            
            return EmailResponse(
                message_id=message_id.strip("<>"),
                provider=EmailProvider.SMTP,
                status="sent",
                recipient_count=len(recipients),
                provider_response={"smtp_response": response}
            )
            
        except Exception as e:
            logger.error(f"Failed to send email via SMTP: {str(e)}")
            return EmailResponse(
                message_id="",
                provider=EmailProvider.SMTP,
                status="failed",
                recipient_count=0,
                errors=[str(e)]
            )
    
    async def _create_mime_message(self, message: EmailMessage) -> MIMEMultipart:
        """Create MIME message from EmailMessage."""
        # Create root message
        if message.attachments or message.html_body:
            mime_message = MIMEMultipart('mixed')
        else:
            mime_message = MIMEMultipart('alternative')
        
        # Set headers
        mime_message['From'] = message.from_email.format()
        mime_message['To'] = ', '.join([r.format() for r in message.to])
        mime_message['Subject'] = message.subject
        mime_message['Date'] = formatdate(localtime=True)
        mime_message['Message-ID'] = make_msgid()
        
        if message.cc:
            mime_message['Cc'] = ', '.join([r.format() for r in message.cc])
        if message.reply_to:
            mime_message['Reply-To'] = message.reply_to.format()
        
        # Set priority
        if message.priority == EmailPriority.HIGH:
            mime_message['X-Priority'] = '1'
            mime_message['X-MSMail-Priority'] = 'High'
        elif message.priority == EmailPriority.LOW:
            mime_message['X-Priority'] = '5'
            mime_message['X-MSMail-Priority'] = 'Low'
        
        # Add body
        if message.text_body:
            text_part = MIMEText(message.text_body, 'plain', 'utf-8')
            mime_message.attach(text_part)
        
        if message.html_body:
            # Inline CSS for better email client compatibility
            html_with_inline_css = premailer.transform(message.html_body)
            html_part = MIMEText(html_with_inline_css, 'html', 'utf-8')
            mime_message.attach(html_part)
        
        # Add attachments
        for attachment in message.attachments:
            if attachment.content_type.startswith('image/') and attachment.content_id:
                # Inline image
                image_part = MIMEImage(attachment.content)
                image_part.add_header('Content-ID', f'<{attachment.content_id}>')
                image_part.add_header('Content-Disposition', 'inline', filename=attachment.filename)
            else:
                # Regular attachment
                part = MIMEText(attachment.content.decode('utf-8', errors='ignore')) \
                    if attachment.content_type.startswith('text/') else \
                    MIMEImage(attachment.content) if attachment.content_type.startswith('image/') else \
                    MIMEText('', 'base64')
                
                part.add_header('Content-Type', attachment.content_type)
                part.add_header('Content-Disposition', 'attachment', filename=attachment.filename)
            
            mime_message.attach(part)
        
        return mime_message
    
    async def validate_connection(self) -> bool:
        """Validate SMTP connection."""
        try:
            connection = await self._get_connection()
            return self._is_connection_alive()
        except Exception as e:
            logger.error(f"SMTP connection validation failed: {str(e)}")
            return False
    
    async def __aenter__(self):
        await self._get_connection()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.connection.quit)
            self.connection = None


class SendGridProvider(BaseEmailProvider):
    """SendGrid email provider."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {
                "api_key": settings.SENDGRID_API_KEY,
                "from_email": settings.SENDGRID_FROM_EMAIL,
                "from_name": settings.SENDGRID_FROM_NAME,
                "sandbox_mode": settings.DEBUG,
            }
        super().__init__(config)
        self.base_url = "https://api.sendgrid.com/v3"
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError,))
    )
    async def send_email(self, message: EmailMessage) -> EmailResponse:
        """Send email via SendGrid API."""
        try:
            # Prepare SendGrid payload
            payload = self._prepare_sendgrid_payload(message)
            
            # Make API request
            headers = {
                "Authorization": f"Bearer {self.config['api_key']}",
                "Content-Type": "application/json",
            }
            
            async with self.http_client.session.post(
                f"{self.base_url}/mail/send",
                headers=headers,
                json=payload
            ) as response:
                response_data = await response.json()
                
                if response.status == 202:
                    # Email accepted for delivery
                    message_id = response.headers.get('X-Message-Id', '')
                    
                    return EmailResponse(
                        message_id=message_id,
                        provider=EmailProvider.SENDGRID,
                        status="queued",
                        recipient_count=len(message.to) + len(message.cc) + len(message.bcc),
                        provider_response=response_data
                    )
                else:
                    # SendGrid returned an error
                    errors = []
                    if isinstance(response_data, list):
                        for error in response_data:
                            if 'message' in error:
                                errors.append(error['message'])
                    
                    logger.error(f"SendGrid API error: {response.status} - {errors}")
                    
                    return EmailResponse(
                        message_id="",
                        provider=EmailProvider.SENDGRID,
                        status="failed",
                        recipient_count=0,
                        provider_response=response_data,
                        errors=errors
                    )
                    
        except Exception as e:
            logger.error(f"Failed to send email via SendGrid: {str(e)}")
            return EmailResponse(
                message_id="",
                provider=EmailProvider.SENDGRID,
                status="failed",
                recipient_count=0,
                errors=[str(e)]
            )
    
    def _prepare_sendgrid_payload(self, message: EmailMessage) -> Dict[str, Any]:
        """Prepare SendGrid API payload."""
        payload = {
            "personalizations": [{
                "to": [{"email": r.email, "name": r.name} for r in message.to],
                "cc": [{"email": r.email, "name": r.name} for r in message.cc] if message.cc else [],
                "bcc": [{"email": r.email, "name": r.name} for r in message.bcc] if message.bcc else [],
                "subject": message.subject,
            }],
            "from": {
                "email": message.from_email.email,
                "name": message.from_email.name or ""
            },
            "subject": message.subject,
        }
        
        # Add reply-to
        if message.reply_to:
            payload["reply_to"] = {
                "email": message.reply_to.email,
                "name": message.reply_to.name or ""
            }
        
        # Add content
        content = []
        if message.text_body:
            content.append({"type": "text/plain", "value": message.text_body})
        if message.html_body:
            content.append({"type": "text/html", "value": message.html_body})
        payload["content"] = content
        
        # Add custom args and categories
        if message.metadata:
            payload["custom_args"] = message.metadata
        
        if message.category:
            payload["category"] = message.category
        
        if message.tags:
            payload["categories"] = message.tags
        
        # Add attachments
        if message.attachments:
            payload["attachments"] = []
            for attachment in message.attachments:
                attachment_data = {
                    "content": attachment.content.decode('base64') if isinstance(attachment.content, bytes) else attachment.content,
                    "filename": attachment.filename,
                    "type": attachment.content_type,
                    "disposition": "inline" if attachment.content_id else "attachment"
                }
                if attachment.content_id:
                    attachment_data["content_id"] = attachment.content_id
                payload["attachments"].append(attachment_data)
        
        # Sandbox mode for testing
        if self.config.get("sandbox_mode"):
            payload["mail_settings"] = {"sandbox_mode": {"enable": True}}
        
        return payload
    
    async def validate_connection(self) -> bool:
        """Validate SendGrid API connection."""
        try:
            headers = {
                "Authorization": f"Bearer {self.config['api_key']}",
            }
            
            async with self.http_client.session.get(
                f"{self.base_url}/user/profile",
                headers=headers
            ) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"SendGrid connection validation failed: {str(e)}")
            return False
    
    async def get_quota_info(self) -> Dict[str, Any]:
        """Get SendGrid quota information."""
        try:
            headers = {
                "Authorization": f"Bearer {self.config['api_key']}",
            }
            
            async with self.http_client.session.get(
                f"{self.base_url}/user/credits",
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "provider": "sendgrid",
                        "remaining": data.get("remaining", 0),
                        "total": data.get("total", 0),
                        "used": data.get("used", 0),
                        "reset_date": data.get("reset_date")
                    }
                return {"provider": "sendgrid", "error": "Unable to fetch quota"}
                
        except Exception as e:
            logger.error(f"Failed to get SendGrid quota: {str(e)}")
            return {"provider": "sendgrid", "error": str(e)}


class LogProvider(BaseEmailProvider):
    """Log provider for development - logs emails instead of sending."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        super().__init__(config)
        
    async def send_email(self, message: EmailMessage) -> EmailResponse:
        """Log email instead of sending."""
        logger.info(
            f"[EMAIL LOGGED] To: {[r.email for r in message.to]}, "
            f"Subject: {message.subject}, "
            f"Body length: {len(message.html_body or message.text_body or '')}"
        )
        
        # Log email details for debugging
        email_details = {
            "to": [r.dict() for r in message.to],
            "subject": message.subject,
            "html_preview": (message.html_body or "")[:500] + "..." if message.html_body else None,
            "text_preview": (message.text_body or "")[:500] + "..." if message.text_body else None,
            "attachments": len(message.attachments),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.debug(f"Email details: {json.dumps(email_details, indent=2)}")
        
        return EmailResponse(
            message_id=f"log-{datetime.now().timestamp()}",
            provider=EmailProvider.LOG,
            status="logged",
            recipient_count=len(message.to),
            provider_response={"logged": True, "details": email_details}
        )
    
    async def validate_connection(self) -> bool:
        """Log provider is always available."""
        return True


class EmailTemplateManager:
    """Manager for email templates."""
    
    def __init__(self, template_dir: Optional[str] = None):
        self.template_dir = template_dir or "templates/emails"
        self.env = None
        self._initialize_jinja()
        
    def _initialize_jinja(self):
        """Initialize Jinja2 template environment."""
        template_path = Path(self.template_dir)
        
        if not template_path.exists():
            # Create default templates
            template_path.mkdir(parents=True, exist_ok=True)
            self._create_default_templates(template_path)
        
        self.env = Environment(
            loader=FileSystemLoader(template_path),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters
        self.env.filters['markdown'] = self._markdown_filter
        self.env.filters['format_date'] = self._format_date_filter
    
    def _create_default_templates(self, template_path: Path):
        """Create default email templates."""
        default_templates = {
            "welcome.html": """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Welcome to WorldBrief 360</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: #4f46e5; color: white; padding: 20px; text-align: center; }
        .content { padding: 30px; background: #f9f9f9; }
        .button { display: inline-block; background: #4f46e5; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; }
        .footer { text-align: center; padding: 20px; color: #666; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Welcome to WorldBrief 360!</h1>
        </div>
        <div class="content">
            <h2>Hello {{ user.name }}!</h2>
            <p>Thank you for joining WorldBrief 360. We're excited to have you on board.</p>
            <p>With your account, you can:</p>
            <ul>
                <li>Get personalized news briefings</li>
                <li>Report and verify incidents</li>
                <li>Chat with our AI assistant</li>
                <li>Earn rewards for contributions</li>
            </ul>
            {% if verification_url %}
            <p style="text-align: center;">
                <a href="{{ verification_url }}" class="button">Verify Your Email</a>
            </p>
            {% endif %}
            <p>If you have any questions, feel free to reach out to our support team.</p>
            <p>Best regards,<br>The WorldBrief 360 Team</p>
        </div>
        <div class="footer">
            <p>© {{ year }} WorldBrief 360. All rights reserved.</p>
            <p><a href="{{ unsubscribe_url }}">Unsubscribe</a> | <a href="{{ preferences_url }}">Email Preferences</a></p>
        </div>
    </div>
</body>
</html>""",
            
            "password_reset.html": """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reset Your Password</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: #dc2626; color: white; padding: 20px; text-align: center; }
        .content { padding: 30px; background: #f9f9f9; }
        .button { display: inline-block; background: #dc2626; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; }
        .footer { text-align: center; padding: 20px; color: #666; font-size: 12px; }
        .warning { background: #fef3c7; border-left: 4px solid #d97706; padding: 10px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Password Reset Request</h1>
        </div>
        <div class="content">
            <h2>Hello {{ user.name }},</h2>
            <p>We received a request to reset your password for your WorldBrief 360 account.</p>
            
            <div class="warning">
                <p><strong>Important:</strong> This password reset link will expire in {{ expiry_hours }} hours.</p>
            </div>
            
            <p style="text-align: center;">
                <a href="{{ reset_url }}" class="button">Reset Password</a>
            </p>
            
            <p>If you didn't request a password reset, you can safely ignore this email. Your password will not be changed.</p>
            <p>For security reasons, this link can only be used once.</p>
            <p>Best regards,<br>The WorldBrief 360 Team</p>
        </div>
        <div class="footer">
            <p>© {{ year }} WorldBrief 360. All rights reserved.</p>
            <p>This is an automated message. Please do not reply to this email.</p>
        </div>
    </div>
</body>
</html>""",
            
            "incident_reported.html": """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Incident Reported</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: #059669; color: white; padding: 20px; text-align: center; }
        .content { padding: 30px; background: #f9f9f9; }
        .incident-details { background: white; border: 1px solid #ddd; padding: 20px; margin: 20px 0; }
        .button { display: inline-block; background: #059669; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; }
        .footer { text-align: center; padding: 20px; color: #666; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Incident Reported Successfully</h1>
        </div>
        <div class="content">
            <h2>Thank you for your report, {{ user.name }}!</h2>
            <p>Your incident report has been submitted successfully and is now being reviewed.</p>
            
            <div class="incident-details">
                <h3>Incident Details:</h3>
                <p><strong>Title:</strong> {{ incident.title }}</p>
                <p><strong>Location:</strong> {{ incident.location }}</p>
                <p><strong>Category:</strong> {{ incident.category }}</p>
                <p><strong>Reported At:</strong> {{ incident.reported_at }}</p>
                <p><strong>Reference ID:</strong> {{ incident.reference_id }}</p>
            </div>
            
            <p>What happens next:</p>
            <ol>
                <li>Our verification team will review your report</li>
                <li>Other users may contribute additional information</li>
                <li>You'll earn {{ reward_amount }} coins for your contribution</li>
                <li>You'll be notified when the incident is verified</li>
            </ol>
            
            <p style="text-align: center;">
                <a href="{{ incident_url }}" class="button">View Incident</a>
            </p>
            
            <p>Thank you for helping keep our community informed!</p>
            <p>Best regards,<br>The WorldBrief 360 Team</p>
        </div>
        <div class="footer">
            <p>© {{ year }} WorldBrief 360. All rights reserved.</p>
        </div>
    </div>
</body>
</html>"""
        }
        
        for filename, content in default_templates.items():
            template_file = template_path / filename
            if not template_file.exists():
                template_file.write_text(content)
    
    def _markdown_filter(self, text: str) -> str:
        """Convert markdown to HTML."""
        if not text:
            return ""
        return markdown2.markdown(text, extras=["fenced-code-blocks", "tables"])
    
    def _format_date_filter(self, date_value, format_str="%B %d, %Y %I:%M %p"):
        """Format date for display."""
        if isinstance(date_value, str):
            from dateutil.parser import parse
            date_value = parse(date_value)
        return date_value.strftime(format_str)
    
    def render_template(
        self,
        template_name: str,
        context: Dict[str, Any],
        generate_text: bool = True
    ) -> Tuple[str, Optional[str]]:
        """
        Render email template.
        
        Args:
            template_name: Template filename
            context: Template context variables
            generate_text: Whether to generate plain text version
            
        Returns:
            Tuple of (html_content, text_content)
        """
        # Add default context variables
        default_context = {
            "year": datetime.now().year,
            "site_url": settings.SITE_URL,
            "support_email": settings.SUPPORT_EMAIL,
            "current_date": datetime.now(),
        }
        context = {**default_context, **context}
        
        # Render HTML template
        template = self.env.get_template(template_name)
        html_content = template.render(**context)
        
        # Generate plain text version if requested
        text_content = None
        if generate_text:
            # Simple HTML to text conversion
            import re
            text_content = re.sub(r'<[^>]+>', '', html_content)
            text_content = re.sub(r'\n\s*\n', '\n\n', text_content).strip()
        
        return html_content, text_content
    
    def get_available_templates(self) -> List[str]:
        """Get list of available templates."""
        if not self.env:
            return []
        
        return self.env.list_templates()


class EmailClient:
    """
    Main email client for WorldBrief 360.
    Supports multiple providers with fallback logic.
    """
    
    def __init__(
        self,
        default_provider: Optional[EmailProvider] = None,
        template_manager: Optional[EmailTemplateManager] = None
    ):
        self.default_provider = default_provider or EmailProvider.SENDGRID
        self.template_manager = template_manager or EmailTemplateManager()
        self.providers: Dict[EmailProvider, BaseEmailProvider] = {}
        self._initialize_providers()
        
    def _initialize_providers(self):
        """Initialize email providers."""
        # Determine which providers to initialize
        provider_configs = []
        
        if settings.SENDGRID_API_KEY:
            provider_configs.append((EmailProvider.SENDGRID, SendGridProvider))
        
        if settings.SMTP_HOST:
            provider_configs.append((EmailProvider.SMTP, SMTPProvider))
        
        if settings.AWS_SES_ACCESS_KEY:
            # Would add SES provider here
            pass
        
        # Always add log provider in debug mode
        if settings.DEBUG:
            provider_configs.append((EmailProvider.LOG, LogProvider))
        
        # Initialize providers
        for provider_type, provider_class in provider_configs:
            try:
                provider = provider_class()
                if asyncio.run(provider.validate_connection()):
                    self.providers[provider_type] = provider
                    logger.info(f"Initialized email provider: {provider_type.value}")
                else:
                    logger.warning(f"Failed to initialize email provider: {provider_type.value}")
            except Exception as e:
                logger.error(f"Error initializing {provider_type.value}: {str(e)}")
        
        if not self.providers:
            logger.warning("No email providers initialized!")
    
    def get_provider(self, provider_type: Optional[EmailProvider] = None) -> BaseEmailProvider:
        """
        Get email provider.
        
        Args:
            provider_type: Provider type (uses default if None)
            
        Returns:
            Email provider instance
            
        Raises:
            ValueError: If provider not available
        """
        provider_type = provider_type or self.default_provider
        
        if provider_type not in self.providers:
            # Try to get any available provider
            if self.providers:
                return next(iter(self.providers.values()))
            raise ValueError(f"No email providers available. Requested: {provider_type.value}")
        
        return self.providers[provider_type]
    
    def get_available_providers(self) -> List[Dict[str, Any]]:
        """Get list of available email providers."""
        available = []
        
        for provider_type, provider in self.providers.items():
            available.append({
                "provider": provider_type.value,
                "name": provider_type.name,
                "enabled": True,
                "description": str(provider)
            })
        
        return available
    
    async def send_email(
        self,
        message: EmailMessage,
        provider_type: Optional[EmailProvider] = None
    ) -> EmailResponse:
        """
        Send email using specified or default provider.
        
        Args:
            message: Email message
            provider_type: Optional provider type
            
        Returns:
            Email response
        """
        try:
            provider = self.get_provider(provider_type)
            response = await provider.send_email(message)
            
            # Log email sending
            self._log_email_sent(message, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            
            return EmailResponse(
                message_id="",
                provider=provider_type or self.default_provider,
                status="failed",
                recipient_count=0,
                errors=[str(e)]
            )
    
    async def send_templated_email(
        self,
        template_name: str,
        to: List[EmailRecipient],
        context: Dict[str, Any],
        subject: Optional[str] = None,
        from_email: Optional[EmailRecipient] = None,
        category: Optional[str] = None,
        **kwargs
    ) -> EmailResponse:
        """
        Send email using template.
        
        Args:
            template_name: Template filename
            to: Recipients
            context: Template context
            subject: Email subject (can be in template context)
            from_email: Sender (defaults to configured from email)
            category: Email category
            **kwargs: Additional EmailMessage parameters
            
        Returns:
            Email response
        """
        # Render template
        html_body, text_body = self.template_manager.render_template(
            template_name,
            context
        )
        
        # Get subject from context or use default
        if not subject:
            subject = context.get('subject', f"Message from {settings.SITE_NAME}")
        
        # Set default from email
        if not from_email:
            from_email = EmailRecipient(
                email=settings.DEFAULT_FROM_EMAIL,
                name=settings.DEFAULT_FROM_NAME
            )
        
        # Create email message
        message = EmailMessage(
            to=to,
            from_email=from_email,
            subject=subject,
            html_body=html_body,
            text_body=text_body,
            category=category,
            **kwargs
        )
        
        return await self.send_email(message)
    
    async def send_batch_emails(
        self,
        messages: List[EmailMessage],
        provider_type: Optional[EmailProvider] = None,
        max_concurrent: int = 10
    ) -> List[EmailResponse]:
        """
        Send multiple emails concurrently.
        
        Args:
            messages: List of email messages
            provider_type: Optional provider type
            max_concurrent: Maximum concurrent sends
            
        Returns:
            List of email responses
        """
        provider = self.get_provider(provider_type)
        
        # Use semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def send_with_semaphore(message):
            async with semaphore:
                return await provider.send_email(message)
        
        # Send all emails concurrently
        tasks = [send_with_semaphore(message) for message in messages]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Failed to send email {i}: {str(response)}")
                processed_responses.append(EmailResponse(
                    message_id="",
                    provider=provider_type or self.default_provider,
                    status="failed",
                    recipient_count=0,
                    errors=[str(response)]
                ))
            else:
                processed_responses.append(response)
        
        return processed_responses
    
    def _log_email_sent(self, message: EmailMessage, response: EmailResponse):
        """Log email sending for analytics."""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "provider": response.provider.value,
            "message_id": response.message_id,
            "status": response.status,
            "to": [r.email for r in message.to],
            "subject": message.subject,
            "category": message.category,
            "template_id": message.template_id,
            "recipient_count": response.recipient_count,
            "has_attachments": len(message.attachments) > 0,
        }
        
        logger.info(f"Email sent: {json.dumps(log_data)}")
        
        # Could also store in database for analytics
        # await self._store_email_analytics(log_data)
    
    async def validate_all_connections(self) -> Dict[str, bool]:
        """Validate connections for all providers."""
        results = {}
        
        for provider_type, provider in self.providers.items():
            try:
                results[provider_type.value] = await provider.validate_connection()
            except Exception as e:
                results[provider_type.value] = False
                logger.error(f"Connection validation failed for {provider_type.value}: {str(e)}")
        
        return results
    
    async def get_quota_info(self) -> Dict[str, Any]:
        """Get quota information for all providers."""
        quota_info = {}
        
        for provider_type, provider in self.providers.items():
            try:
                quota_info[provider_type.value] = await provider.get_quota_info()
            except Exception as e:
                quota_info[provider_type.value] = {"error": str(e)}
        
        return quota_info


# Factory function for dependency injection
def get_email_client() -> EmailClient:
    """
    Factory function to create email client.
    
    Returns:
        Configured EmailClient instance
    """
    return EmailClient()