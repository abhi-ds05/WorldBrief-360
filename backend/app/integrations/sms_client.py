# backend/app/integrations/sms_client.py
"""
SMS client for sending text messages via multiple providers.
Supports Twilio, Vonage (Nexmo), Amazon SNS, and Plivo with failover capability.
"""

import asyncio
import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
from uuid import uuid4

import aiohttp
from pydantic import BaseModel, Field, validator, root_validator
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.config import settings
from app.core.logging_config import get_logger
from app.cache.redis_client import get_redis_client
from app.utils.rate_limiter import RateLimiter

logger = get_logger(__name__)


# ============ Data Models ============

class SMSProvider(str, Enum):
    """Supported SMS providers."""
    TWILIO = "twilio"
    VONAGE = "vonage"
    AWS_SNS = "aws_sns"
    PLIVO = "plivo"
    TWILIO_VERIFY = "twilio_verify"  # For verification services
    CUSTOM = "custom"  # Custom gateway


class SMSType(str, Enum):
    """Types of SMS messages."""
    VERIFICATION = "verification"  # OTP/verification codes
    ALERT = "alert"  # Incident alerts
    NOTIFICATION = "notification"  # General notifications
    MARKETING = "marketing"  # Promotional messages
    TRANSACTIONAL = "transactional"  # Transaction confirmations
    SECURITY = "security"  # Security alerts


class SMSStatus(str, Enum):
    """SMS delivery status."""
    QUEUED = "queued"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    UNDELIVERED = "undelivered"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


class SMSMessage(BaseModel):
    """SMS message data model."""
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    to_phone: str  # E.164 format
    from_phone: Optional[str] = None  # Optional, will use default if not provided
    body: str
    sms_type: SMSType = SMSType.NOTIFICATION
    provider: Optional[SMSProvider] = None  # Auto-select if not specified
    country_code: Optional[str] = None  # Auto-detect from phone number
    template_id: Optional[str] = None  # For template-based messages
    template_variables: Dict[str, Any] = Field(default_factory=dict)
    media_urls: List[str] = Field(default_factory=list)  # For MMS
    scheduled_at: Optional[datetime] = None
    callback_url: Optional[str] = None  # For delivery status callbacks
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('to_phone')
    def validate_phone_number(cls, v):
        """Validate and normalize phone number."""
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', v)
        
        if not digits:
            raise ValueError('Invalid phone number: no digits found')
        
        # Basic validation - should be between 10-15 digits
        if len(digits) < 10 or len(digits) > 15:
            raise ValueError(f'Invalid phone number length: {len(digits)} digits')
        
        # Return in E.164 format (assuming + prefix if not present)
        if not v.startswith('+'):
            # This is a simplification - in production you'd use a library like phonenumbers
            return f"+{digits}"
        
        return v
    
    @validator('body')
    def validate_body_length(cls, v, values):
        """Validate SMS body length."""
        sms_type = values.get('sms_type', SMSType.NOTIFICATION)
        
        # Different limits for different SMS types
        limits = {
            SMSType.VERIFICATION: 160,  # Single segment for OTPs
            SMSType.SECURITY: 160,
            SMSType.TRANSACTIONAL: 160,
            SMSType.ALERT: 306,  # Two segments for alerts
            SMSType.NOTIFICATION: 459,  # Three segments max
            SMSType.MARKETING: 459,
        }
        
        limit = limits.get(sms_type, 160)
        if len(v) > limit:
            raise ValueError(f'SMS body too long: {len(v)} characters (max {limit})')
        
        return v
    
    @validator('media_urls')
    def validate_media_urls(cls, v):
        """Validate MMS media URLs."""
        if v and len(v) > 10:
            raise ValueError('Maximum 10 media URLs allowed for MMS')
        return v


class BatchSMSRequest(BaseModel):
    """Batch SMS request."""
    messages: List[SMSMessage]
    failover_enabled: bool = True  # Enable provider failover
    dry_run: bool = False


class SMSResponse(BaseModel):
    """SMS response model."""
    message_id: str
    success: bool
    provider: SMSProvider
    provider_message_id: Optional[str] = None
    cost: Optional[float] = None  # Cost in USD
    segments: int = 1  # Number of SMS segments
    remaining_balance: Optional[float] = None
    error: Optional[str] = None
    status: SMSStatus = SMSStatus.QUEUED
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DeliveryReport(BaseModel):
    """SMS delivery report."""
    message_id: str
    provider: SMSProvider
    provider_message_id: str
    status: SMSStatus
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    delivered_at: Optional[datetime] = None
    received_at: Optional[datetime] = None
    carrier: Optional[str] = None
    country: Optional[str] = None
    price: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ProviderConfig(BaseModel):
    """SMS provider configuration."""
    provider: SMSProvider
    enabled: bool = True
    priority: int = 1  # Lower number = higher priority
    weight: float = 1.0  # For weighted load balancing
    daily_limit: Optional[int] = None
    monthly_limit: Optional[int] = None
    supported_countries: List[str] = Field(default_factory=list)  # Empty = all countries
    excluded_countries: List[str] = Field(default_factory=list)
    cost_per_message: float = 0.01  # Default cost in USD
    supports_mms: bool = False
    supports_unicode: bool = True
    max_message_length: int = 160  # GSM-7 characters


# ============ Abstract Base Classes ============

class BaseSMSProvider(ABC):
    """Abstract base class for SMS providers."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.provider_name = config.provider.value
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = RateLimiter(
            max_requests=100,
            time_window=60  # 100 requests per minute
        )
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider."""
        pass
    
    @abstractmethod
    async def send_single(self, message: SMSMessage) -> SMSResponse:
        """Send a single SMS."""
        pass
    
    @abstractmethod
    async def send_batch(self, messages: List[SMSMessage]) -> List[SMSResponse]:
        """Send multiple SMS messages."""
        pass
    
    @abstractmethod
    async def get_balance(self) -> Optional[float]:
        """Get account balance."""
        pass
    
    @abstractmethod
    async def get_message_status(self, provider_message_id: str) -> DeliveryReport:
        """Get message delivery status."""
        pass
    
    def supports_country(self, country_code: str) -> bool:
        """Check if provider supports a country."""
        country_code = country_code.upper()
        
        # Check excluded countries first
        if country_code in self.config.excluded_countries:
            return False
        
        # Check supported countries
        if self.config.supported_countries:
            return country_code in self.config.supported_countries
        
        return True
    
    def calculate_segments(self, body: str) -> int:
        """Calculate number of SMS segments needed."""
        # GSM-7 vs Unicode handling
        if self.config.supports_unicode and self._contains_unicode(body):
            # Unicode uses 70 chars per segment
            chars_per_segment = 70
        else:
            # GSM-7 uses 160 chars per segment
            chars_per_segment = 160
        
        return max(1, (len(body) + chars_per_segment - 1) // chars_per_segment)
    
    def _contains_unicode(self, text: str) -> bool:
        """Check if text contains Unicode characters."""
        try:
            text.encode('ascii')
            return False
        except UnicodeEncodeError:
            return True
    
    def estimate_cost(self, message: SMSMessage) -> float:
        """Estimate cost for sending a message."""
        segments = self.calculate_segments(message.body)
        return segments * self.config.cost_per_message
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.session:
            await self.session.close()


# ============ Concrete Providers ============

class TwilioProvider(BaseSMSProvider):
    """Twilio SMS provider."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.account_sid = settings.TWILIO_ACCOUNT_SID
        self.auth_token = settings.TWILIO_AUTH_TOKEN
        self.from_number = settings.TWILIO_FROM_NUMBER
        self.api_base = "https://api.twilio.com/2010-04-01"
    
    async def initialize(self) -> None:
        """Initialize Twilio provider."""
        if not self.account_sid or not self.auth_token:
            raise ValueError("Twilio credentials not configured")
        
        # Create basic auth header
        auth = aiohttp.BasicAuth(self.account_sid, self.auth_token)
        
        self.session = aiohttp.ClientSession(
            auth=auth,
            timeout=aiohttp.ClientTimeout(total=30)
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def send_single(self, message: SMSMessage) -> SMSResponse:
        """Send SMS via Twilio."""
        try:
            # Apply rate limiting
            await self.rate_limiter.acquire()
            
            # Prepare request data
            data = {
                "To": message.to_phone,
                "From": message.from_phone or self.from_number,
                "Body": message.body,
                "StatusCallback": message.callback_url or f"{settings.API_BASE_URL}/webhooks/twilio/sms",
            }
            
            # Add media URLs for MMS
            if message.media_urls and self.config.supports_mms:
                for i, url in enumerate(message.media_urls[:10]):  # Twilio supports max 10 media
                    data[f"MediaUrl{i}"] = url
            
            # Send request
            url = f"{self.api_base}/Accounts/{self.account_sid}/Messages.json"
            
            async with self.session.post(url, data=data) as response:
                response_data = await response.json()
                
                if response.status == 201:
                    segments = int(response_data.get("num_segments", 1))
                    cost = float(response_data.get("price", "0")) or self.estimate_cost(message)
                    
                    return SMSResponse(
                        message_id=message.message_id,
                        success=True,
                        provider=SMSProvider.TWILIO,
                        provider_message_id=response_data.get("sid"),
                        cost=cost,
                        segments=segments,
                        remaining_balance=await self.get_balance(),
                        status=SMSStatus(response_data.get("status", "queued")),
                    )
                else:
                    error_msg = response_data.get("message", "Unknown Twilio error")
                    error_code = response_data.get("code")
                    
                    # Map common Twilio error codes
                    if error_code in [21614, 21408]:
                        error_msg = f"Invalid phone number: {error_msg}"
                    elif error_code == 21610:
                        error_msg = f"Daily limit exceeded: {error_msg}"
                    
                    logger.error(f"Twilio error {error_code}: {error_msg}")
                    
                    return SMSResponse(
                        message_id=message.message_id,
                        success=False,
                        provider=SMSProvider.TWILIO,
                        error=f"{error_code}: {error_msg}" if error_code else error_msg,
                        status=SMSStatus.FAILED
                    )
                    
        except Exception as e:
            logger.error(f"Twilio send failed: {str(e)}", exc_info=True)
            return SMSResponse(
                message_id=message.message_id,
                success=False,
                provider=SMSProvider.TWILIO,
                error=str(e),
                status=SMSStatus.FAILED
            )
    
    async def send_batch(self, messages: List[SMSMessage]) -> List[SMSResponse]:
        """Send batch SMS via Twilio (sequential due to API limitations)."""
        responses = []
        for message in messages:
            response = await self.send_single(message)
            responses.append(response)
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.1)
        return responses
    
    async def get_balance(self) -> Optional[float]:
        """Get Twilio account balance."""
        try:
            url = f"{self.api_base}/Accounts/{self.account_sid}/Balance.json"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data.get("balance", 0))
                return None
        except Exception as e:
            logger.error(f"Failed to get Twilio balance: {str(e)}")
            return None
    
    async def get_message_status(self, provider_message_id: str) -> DeliveryReport:
        """Get Twilio message status."""
        try:
            url = f"{self.api_base}/Accounts/{self.account_sid}/Messages/{provider_message_id}.json"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Map Twilio status to our status enum
                    status_mapping = {
                        "queued": SMSStatus.QUEUED,
                        "sent": SMSStatus.SENT,
                        "delivered": SMSStatus.DELIVERED,
                        "failed": SMSStatus.FAILED,
                        "undelivered": SMSStatus.UNDELIVERED,
                    }
                    
                    status = status_mapping.get(data.get("status"), SMSStatus.UNKNOWN)
                    
                    return DeliveryReport(
                        message_id="",  # Will be filled by caller
                        provider=SMSProvider.TWILIO,
                        provider_message_id=provider_message_id,
                        status=status,
                        error_code=data.get("error_code"),
                        error_message=data.get("error_message"),
                        delivered_at=self._parse_twilio_date(data.get("date_sent")),
                        carrier=data.get("carrier"),
                        country=data.get("country"),
                        price=float(data.get("price", 0)) if data.get("price") else None
                    )
                else:
                    return DeliveryReport(
                        message_id="",
                        provider=SMSProvider.TWILIO,
                        provider_message_id=provider_message_id,
                        status=SMSStatus.UNKNOWN,
                        error_message=f"HTTP {response.status}"
                    )
        except Exception as e:
            logger.error(f"Failed to get Twilio message status: {str(e)}")
            return DeliveryReport(
                message_id="",
                provider=SMSProvider.TWILIO,
                provider_message_id=provider_message_id,
                status=SMSStatus.UNKNOWN,
                error_message=str(e)
            )
    
    def _parse_twilio_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse Twilio date string."""
        if not date_str:
            return None
        try:
            # Twilio dates are in RFC 2822 format
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(date_str)
        except:
            return None


class VonageProvider(BaseSMSProvider):
    """Vonage (Nexmo) SMS provider."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.api_key = settings.VONAGE_API_KEY
        self.api_secret = settings.VONAGE_API_SECRET
        self.from_number = settings.VONAGE_FROM_NUMBER
        self.api_base = "https://rest.nexmo.com"
    
    async def initialize(self) -> None:
        """Initialize Vonage provider."""
        if not self.api_key or not self.api_secret:
            raise ValueError("Vonage credentials not configured")
        
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def send_single(self, message: SMSMessage) -> SMSResponse:
        """Send SMS via Vonage."""
        try:
            await self.rate_limiter.acquire()
            
            # Vonage API params
            params = {
                "api_key": self.api_key,
                "api_secret": self.api_secret,
                "to": message.to_phone.lstrip("+"),  # Vonage doesn't want +
                "from": message.from_phone or self.from_number,
                "text": message.body,
                "type": "unicode" if self._contains_unicode(message.body) else "text",
            }
            
            # Add callback URL if provided
            if message.callback_url:
                params["status-report-req"] = "1"
                params["callback"] = message.callback_url
            
            # Send SMS
            url = f"{self.api_base}/sms/json"
            
            async with self.session.post(url, data=params) as response:
                response_data = await response.json()
                
                if response.status == 200:
                    messages = response_data.get("messages", [])
                    if messages:
                        msg = messages[0]
                        if msg.get("status") == "0":
                            segments = int(msg.get("message-count", 1))
                            message_price = float(msg.get("message-price", "0")) or self.estimate_cost(message)
                            
                            return SMSResponse(
                                message_id=message.message_id,
                                success=True,
                                provider=SMSProvider.VONAGE,
                                provider_message_id=msg.get("message-id"),
                                cost=message_price,
                                segments=segments,
                                remaining_balance=await self.get_balance(),
                                status=SMSStatus.SENT,
                            )
                        else:
                            error_msg = msg.get("error-text", "Unknown Vonage error")
                            logger.error(f"Vonage error: {error_msg}")
                            
                            return SMSResponse(
                                message_id=message.message_id,
                                success=False,
                                provider=SMSProvider.VONAGE,
                                error=error_msg,
                                status=SMSStatus.FAILED
                            )
                    else:
                        return SMSResponse(
                            message_id=message.message_id,
                            success=False,
                            provider=SMSProvider.VONAGE,
                            error="No response messages from Vonage",
                            status=SMSStatus.FAILED
                        )
                else:
                    return SMSResponse(
                        message_id=message.message_id,
                        success=False,
                        provider=SMSProvider.VONAGE,
                        error=f"HTTP {response.status}",
                        status=SMSStatus.FAILED
                    )
                    
        except Exception as e:
            logger.error(f"Vonage send failed: {str(e)}", exc_info=True)
            return SMSResponse(
                message_id=message.message_id,
                success=False,
                provider=SMSProvider.VONAGE,
                error=str(e),
                status=SMSStatus.FAILED
            )
    
    async def send_batch(self, messages: List[SMSMessage]) -> List[SMSResponse]:
        """Vonage batch sending."""
        # Vonage doesn't have a true batch API, send sequentially
        responses = []
        for message in messages:
            response = await self.send_single(message)
            responses.append(response)
            await asyncio.sleep(0.1)
        return responses
    
    async def get_balance(self) -> Optional[float]:
        """Get Vonage account balance."""
        try:
            url = f"{self.api_base}/account/get-balance"
            params = {
                "api_key": self.api_key,
                "api_secret": self.api_secret
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data.get("value", 0))
                return None
        except Exception as e:
            logger.error(f"Failed to get Vonage balance: {str(e)}")
            return None
    
    async def get_message_status(self, provider_message_id: str) -> DeliveryReport:
        """Get Vonage message status via search API."""
        try:
            url = f"{self.api_base}/search/message"
            params = {
                "api_key": self.api_key,
                "api_secret": self.api_secret,
                "id": provider_message_id
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Map Vonage status
                    status_mapping = {
                        "DELIVRD": SMSStatus.DELIVERED,
                        "EXPIRED": SMSStatus.FAILED,
                        "DELETED": SMSStatus.FAILED,
                        "UNDELIV": SMSStatus.UNDELIVERED,
                        "ACCEPTD": SMSStatus.SENT,
                        "UNKNOWN": SMSStatus.UNKNOWN,
                    }
                    
                    status = status_mapping.get(data.get("status"), SMSStatus.UNKNOWN)
                    
                    return DeliveryReport(
                        message_id="",
                        provider=SMSProvider.VONAGE,
                        provider_message_id=provider_message_id,
                        status=status,
                        error_message=data.get("error-code-text"),
                        delivered_at=self._parse_iso_date(data.get("date-received")),
                        price=float(data.get("price", 0)) if data.get("price") else None
                    )
                else:
                    return DeliveryReport(
                        message_id="",
                        provider=SMSProvider.VONAGE,
                        provider_message_id=provider_message_id,
                        status=SMSStatus.UNKNOWN,
                        error_message=f"HTTP {response.status}"
                    )
        except Exception as e:
            logger.error(f"Failed to get Vonage message status: {str(e)}")
            return DeliveryReport(
                message_id="",
                provider=SMSProvider.VONAGE,
                provider_message_id=provider_message_id,
                status=SMSStatus.UNKNOWN,
                error_message=str(e)
            )
    
    def _parse_iso_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse ISO date string."""
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except:
            return None


class AWSSNSProvider(BaseSMSProvider):
    """AWS SNS SMS provider."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.aws_access_key = settings.AWS_ACCESS_KEY_ID
        self.aws_secret_key = settings.AWS_SECRET_ACCESS_KEY
        self.aws_region = settings.AWS_REGION or "us-east-1"
    
    async def initialize(self) -> None:
        """Initialize AWS SNS provider."""
        if not self.aws_access_key or not self.aws_secret_key:
            raise ValueError("AWS credentials not configured")
        
        # Note: For production, you'd want to use boto3 with async support
        # or aioboto3. This is a simplified implementation.
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
    
    async def send_single(self, message: SMSMessage) -> SMSResponse:
        """Send SMS via AWS SNS."""
        # Note: This is a placeholder. In production, use boto3/aioboto3.
        # AWS SNS requires signing requests with AWS Signature Version 4.
        logger.info(f"[AWS SNS] Would send SMS to {message.to_phone}")
        
        return SMSResponse(
            message_id=message.message_id,
            success=False,
            provider=SMSProvider.AWS_SNS,
            error="AWS SNS implementation requires boto3/aioboto3",
            status=SMSStatus.FAILED
        )
    
    # Other methods would be implemented similarly...


class PlivoProvider(BaseSMSProvider):
    """Plivo SMS provider."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.auth_id = settings.PLIVO_AUTH_ID
        self.auth_token = settings.PLIVO_AUTH_TOKEN
        self.from_number = settings.PLIVO_FROM_NUMBER
        self.api_base = "https://api.plivo.com/v1/Account"
    
    async def initialize(self) -> None:
        """Initialize Plivo provider."""
        if not self.auth_id or not self.auth_token:
            raise ValueError("Plivo credentials not configured")
        
        # Basic auth for Plivo
        auth = aiohttp.BasicAuth(self.auth_id, self.auth_token)
        
        self.session = aiohttp.ClientSession(
            auth=auth,
            timeout=aiohttp.ClientTimeout(total=30)
        )
    
    async def send_single(self, message: SMSMessage) -> SMSResponse:
        """Send SMS via Plivo."""
        try:
            await self.rate_limiter.acquire()
            
            # Plivo request data
            data = {
                "src": message.from_phone or self.from_number,
                "dst": message.to_phone,
                "text": message.body,
                "type": "sms",
            }
            
            # Add callback URL
            if message.callback_url:
                data["url"] = message.callback_url
                data["method"] = "POST"
            
            # Send request
            url = f"{self.api_base}/{self.auth_id}/Message/"
            
            async with self.session.post(url, json=data) as response:
                response_data = await response.json()
                
                if response.status in [200, 202]:
                    message_uuid = response_data.get("message_uuid")
                    if isinstance(message_uuid, list):
                        message_uuid = message_uuid[0] if message_uuid else None
                    
                    return SMSResponse(
                        message_id=message.message_id,
                        success=True,
                        provider=SMSProvider.PLIVO,
                        provider_message_id=message_uuid,
                        cost=self.estimate_cost(message),
                        segments=self.calculate_segments(message.body),
                        remaining_balance=await self.get_balance(),
                        status=SMSStatus.QUEUED,
                    )
                else:
                    error_msg = response_data.get("error", "Unknown Plivo error")
                    logger.error(f"Plivo error: {error_msg}")
                    
                    return SMSResponse(
                        message_id=message.message_id,
                        success=False,
                        provider=SMSProvider.PLIVO,
                        error=error_msg,
                        status=SMSStatus.FAILED
                    )
                    
        except Exception as e:
            logger.error(f"Plivo send failed: {str(e)}", exc_info=True)
            return SMSResponse(
                message_id=message.message_id,
                success=False,
                provider=SMSProvider.PLIVO,
                error=str(e),
                status=SMSStatus.FAILED
            )
    
    # Other methods similar to Twilio...


class TwilioVerifyProvider(BaseSMSProvider):
    """Twilio Verify service for OTP/verification."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.account_sid = settings.TWILIO_ACCOUNT_SID
        self.auth_token = settings.TWILIO_AUTH_TOKEN
        self.verify_service_sid = settings.TWILIO_VERIFY_SERVICE_SID
        self.api_base = "https://verify.twilio.com/v2/Services"
    
    async def initialize(self) -> None:
        """Initialize Twilio Verify."""
        if not all([self.account_sid, self.auth_token, self.verify_service_sid]):
            raise ValueError("Twilio Verify credentials not configured")
        
        auth = aiohttp.BasicAuth(self.account_sid, self.auth_token)
        
        self.session = aiohttp.ClientSession(
            auth=auth,
            timeout=aiohttp.ClientTimeout(total=30)
        )
    
    async def send_verification(self, to_phone: str, channel: str = "sms") -> Dict[str, Any]:
        """Send verification code."""
        try:
            url = f"{self.api_base}/{self.verify_service_sid}/Verifications"
            data = {
                "To": to_phone,
                "Channel": channel,
            }
            
            async with self.session.post(url, data=data) as response:
                response_data = await response.json()
                
                if response.status == 201:
                    return {
                        "success": True,
                        "verification_sid": response_data.get("sid"),
                        "status": response_data.get("status"),
                        "to": response_data.get("to"),
                        "channel": response_data.get("channel"),
                    }
                else:
                    return {
                        "success": False,
                        "error": response_data.get("message", "Unknown error"),
                        "code": response_data.get("code"),
                    }
                    
        except Exception as e:
            logger.error(f"Twilio Verify send failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def check_verification(self, to_phone: str, code: str) -> Dict[str, Any]:
        """Check verification code."""
        try:
            url = f"{self.api_base}/{self.verify_service_sid}/VerificationCheck"
            data = {
                "To": to_phone,
                "Code": code,
            }
            
            async with self.session.post(url, data=data) as response:
                response_data = await response.json()
                
                if response.status == 200:
                    return {
                        "success": True,
                        "valid": response_data.get("status") == "approved",
                        "status": response_data.get("status"),
                    }
                else:
                    return {
                        "success": False,
                        "error": response_data.get("message", "Unknown error"),
                    }
                    
        except Exception as e:
            logger.error(f"Twilio Verify check failed: {str(e)}")
            return {"success": False, "error": str(e)}


# ============ Main SMS Client ============

class SMSClient:
    """
    Main SMS client with multi-provider support and failover.
    """
    
    def __init__(self):
        self.providers: Dict[SMSProvider, BaseSMSProvider] = {}
        self.provider_configs: Dict[SMSProvider, ProviderConfig] = {}
        self.redis = get_redis_client()
        self.initialized = False
        self.sms_counter_key = "sms:daily_counter"
        
    async def initialize(self) -> None:
        """Initialize SMS client with configured providers."""
        if self.initialized:
            return
        
        logger.info("Initializing SMS client...")
        
        # Load provider configurations
        await self._load_provider_configs()
        
        # Initialize enabled providers
        for provider_name, config in self.provider_configs.items():
            if config.enabled:
                provider = self._create_provider(provider_name, config)
                if provider:
                    try:
                        await provider.initialize()
                        self.providers[provider_name] = provider
                        logger.info(f"Initialized SMS provider: {provider_name.value}")
                    except Exception as e:
                        logger.error(f"Failed to initialize {provider_name.value}: {str(e)}")
        
        if not self.providers:
            logger.warning("No SMS providers initialized!")
        
        self.initialized = True
    
    async def _load_provider_configs(self) -> None:
        """Load provider configurations from settings and database."""
        # Default configurations
        default_configs = {
            SMSProvider.TWILIO: ProviderConfig(
                provider=SMSProvider.TWILIO,
                enabled=bool(settings.TWILIO_ACCOUNT_SID),
                priority=1,
                weight=1.0,
                supports_mms=True,
                supports_unicode=True,
                cost_per_message=0.0075,  # Twilio approximate cost
            ),
            SMSProvider.VONAGE: ProviderConfig(
                provider=SMSProvider.VONAGE,
                enabled=bool(settings.VONAGE_API_KEY),
                priority=2,
                weight=1.0,
                supports_mms=False,  # Vonage MMS is separate API
                supports_unicode=True,
                cost_per_message=0.0055,  # Vonage approximate cost
            ),
            SMSProvider.AWS_SNS: ProviderConfig(
                provider=SMSProvider.AWS_SNS,
                enabled=bool(settings.AWS_ACCESS_KEY_ID),
                priority=3,
                weight=1.0,
                supports_mms=False,
                supports_unicode=True,
                cost_per_message=0.00645,  # AWS SNS approximate cost
            ),
            SMSProvider.PLIVO: ProviderConfig(
                provider=SMSProvider.PLIVO,
                enabled=bool(settings.PLIVO_AUTH_ID),
                priority=4,
                weight=1.0,
                supports_mms=True,
                supports_unicode=True,
                cost_per_message=0.004,  # Plivo approximate cost
            ),
            SMSProvider.TWILIO_VERIFY: ProviderConfig(
                provider=SMSProvider.TWILIO_VERIFY,
                enabled=bool(settings.TWILIO_VERIFY_SERVICE_SID),
                priority=0,  # Highest priority for verification
                weight=1.0,
                supports_mms=False,
                supports_unicode=True,
                cost_per_message=0.05,  # Twilio Verify is more expensive
            ),
        }
        
        # TODO: Load custom configurations from database
        # This allows dynamic configuration updates
        
        self.provider_configs = default_configs
    
    def _create_provider(self, provider: SMSProvider, config: ProviderConfig) -> Optional[BaseSMSProvider]:
        """Create provider instance based on type."""
        provider_map = {
            SMSProvider.TWILIO: TwilioProvider,
            SMSProvider.VONAGE: VonageProvider,
            SMSProvider.AWS_SNS: AWSSNSProvider,
            SMSProvider.PLIVO: PlivoProvider,
            SMSProvider.TWILIO_VERIFY: TwilioVerifyProvider,
        }
        
        provider_class = provider_map.get(provider)
        if provider_class:
            return provider_class(config)
        return None
    
    async def _select_provider(self, message: SMSMessage) -> Optional[BaseSMSProvider]:
        """Select the best provider for a message."""
        # If provider is specified, use it
        if message.provider:
            return self.providers.get(message.provider)
        
        # Determine country code from phone number
        country_code = message.country_code or self._extract_country_code(message.to_phone)
        
        # Filter providers by country support
        available_providers = []
        for provider_name, provider in self.providers.items():
            config = self.provider_configs.get(provider_name)
            if config and provider.supports_country(country_code):
                available_providers.append((provider, config))
        
        if not available_providers:
            logger.warning(f"No providers available for country: {country_code}")
            return None
        
        # Sort by priority and weight
        available_providers.sort(key=lambda x: (x[1].priority, -x[1].weight))
        
        # For verification messages, prefer Twilio Verify if available
        if message.sms_type == SMSType.VERIFICATION:
            verify_provider = self.providers.get(SMSProvider.TWILIO_VERIFY)
            if verify_provider and verify_provider.supports_country(country_code):
                return verify_provider
        
        # Return highest priority provider
        return available_providers[0][0]
    
    def _extract_country_code(self, phone_number: str) -> str:
        """Extract country code from phone number."""
        # Simplified extraction - in production use phonenumbers library
        if phone_number.startswith("+1"):
            return "US"
        elif phone_number.startswith("+44"):
            return "GB"
        elif phone_number.startswith("+91"):
            return "IN"
        elif phone_number.startswith("+86"):
            return "CN"
        # Add more as needed
        return "US"  # Default
    
    async def _check_daily_limit(self) -> bool:
        """Check if daily SMS limit is reached."""
        if not settings.SMS_DAILY_LIMIT:
            return True  # No limit
        
        today = datetime.utcnow().strftime("%Y-%m-%d")
        counter_key = f"{self.sms_counter_key}:{today}"
        
        try:
            count = await self.redis.get(counter_key)
            current_count = int(count) if count else 0
            
            if current_count >= settings.SMS_DAILY_LIMIT:
                logger.warning(f"Daily SMS limit reached: {current_count}/{settings.SMS_DAILY_LIMIT}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Failed to check daily limit: {str(e)}")
            return True  # Allow on error
    
    async def _increment_counter(self) -> None:
        """Increment daily SMS counter."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        counter_key = f"{self.sms_counter_key}:{today}"
        
        try:
            await self.redis.incr(counter_key)
            # Set expiry to 2 days to handle timezone differences
            await self.redis.expire(counter_key, 172800)
        except Exception as e:
            logger.error(f"Failed to increment SMS counter: {str(e)}")
    
    async def send_sms(self, message: SMSMessage) -> SMSResponse:
        """
        Send a single SMS with failover support.
        
        Args:
            message: SMS message to send
            
        Returns:
            SMSResponse with result
        """
        if not self.initialized:
            await self.initialize()
        
        # Check daily limit
        if not await self._check_daily_limit():
            return SMSResponse(
                message_id=message.message_id,
                success=False,
                provider=SMSProvider.CUSTOM,
                error="Daily SMS limit reached",
                status=SMSStatus.FAILED
            )
        
        # Select primary provider
        primary_provider = await self._select_provider(message)
        if not primary_provider:
            return SMSResponse(
                message_id=message.message_id,
                success=False,
                provider=SMSProvider.CUSTOM,
                error="No available SMS provider for this country",
                status=SMSStatus.FAILED
            )
        
        # Try primary provider
        response = await primary_provider.send_single(message)
        
        # If primary fails and failover is enabled, try others
        if not response.success and message.provider is None:  # Only failover if provider not explicitly specified
            await self._failover_send(message, response, primary_provider)
        
        # Log and track
        if response.success:
            await self._increment_counter()
            await self._log_sms(message, response)
        
        return response
    
    async def _failover_send(self, message: SMSMessage, failed_response: SMSResponse, failed_provider: BaseSMSProvider) -> None:
        """Try sending with alternative providers."""
        country_code = message.country_code or self._extract_country_code(message.to_phone)
        
        for provider_name, provider in self.providers.items():
            # Skip the failed provider
            if provider == failed_provider:
                continue
            
            # Check if provider supports the country
            config = self.provider_configs.get(provider_name)
            if not config or not provider.supports_country(country_code):
                continue
            
            logger.info(f"Trying failover to {provider_name.value} for message {message.message_id}")
            
            # Try sending with this provider
            retry_response = await provider.send_single(message)
            
            if retry_response.success:
                # Update the original response
                failed_response.success = True
                failed_response.provider = provider_name
                failed_response.provider_message_id = retry_response.provider_message_id
                failed_response.cost = retry_response.cost
                failed_response.segments = retry_response.segments
                failed_response.remaining_balance = retry_response.remaining_balance
                failed_response.status = retry_response.status
                failed_response.error = None
                break
            else:
                logger.warning(f"Failover to {provider_name.value} also failed: {retry_response.error}")
    
    async def send_batch(self, request: BatchSMSRequest) -> List[SMSResponse]:
        """
        Send batch SMS messages.
        
        Args:
            request: Batch SMS request
            
        Returns:
            List of SMS responses
        """
        if not self.initialized:
            await self.initialize()
        
        # Check daily limit for batch
        batch_size = len(request.messages)
        if not await self._check_daily_limit():
            # Create error responses for all messages
            return [
                SMSResponse(
                    message_id=msg.message_id,
                    success=False,
                    provider=SMSProvider.CUSTOM,
                    error="Daily SMS limit reached",
                    status=SMSStatus.FAILED
                )
                for msg in request.messages
            ]
        
        # Group messages by provider for efficiency
        messages_by_provider: Dict[BaseSMSProvider, List[SMSMessage]] = {}
        
        for message in request.messages:
            provider = await self._select_provider(message)
            if provider:
                provider_messages = messages_by_provider.get(provider, [])
                provider_messages.append(message)
                messages_by_provider[provider] = provider_messages
        
        # Send batches per provider
        all_responses = []
        for provider, messages in messages_by_provider.items():
            try:
                responses = await provider.send_batch(messages)
                all_responses.extend(responses)
                
                # Count successful sends
                success_count = sum(1 for r in responses if r.success)
                if success_count > 0:
                    # Increment counter for each successful message
                    for _ in range(success_count):
                        await self._increment_counter()
                
                # Log all responses
                for message, response in zip(messages, responses):
                    await self._log_sms(message, response)
                    
            except Exception as e:
                logger.error(f"Batch send failed for provider {provider.config.provider}: {str(e)}")
                # Create error responses for failed batch
                error_responses = [
                    SMSResponse(
                        message_id=msg.message_id,
                        success=False,
                        provider=provider.config.provider,
                        error=f"Batch send failed: {str(e)}",
                        status=SMSStatus.FAILED
                    )
                    for msg in messages
                ]
                all_responses.extend(error_responses)
        
        return all_responses
    
    async def send_verification_code(self, phone_number: str, channel: str = "sms") -> Dict[str, Any]:
        """
        Send verification code (OTP) to phone number.
        
        Args:
            phone_number: Phone number to verify
            channel: "sms" or "call"
            
        Returns:
            Verification response
        """
        # Use Twilio Verify if available
        verify_provider = self.providers.get(SMSProvider.TWILIO_VERIFY)
        if verify_provider:
            return await verify_provider.send_verification(phone_number, channel)
        
        # Fallback to regular SMS with generated code
        code = self._generate_verification_code()
        
        # Store code in Redis with expiry
        code_key = f"sms_verify:{phone_number}"
        await self.redis.setex(code_key, 600, code)  # 10 minutes expiry
        
        # Send SMS with code
        message = SMSMessage(
            to_phone=phone_number,
            body=f"Your WorldBrief 360 verification code is: {code}",
            sms_type=SMSType.VERIFICATION,
            provider=SMSProvider.TWILIO,  # Prefer Twilio for verification
            metadata={"verification_code": code}
        )
        
        response = await self.send_sms(message)
        
        return {
            "success": response.success,
            "message_id": response.message_id if response.success else None,
            "error": response.error if not response.success else None,
            "method": "sms"
        }
    
    async def verify_code(self, phone_number: str, code: str) -> Dict[str, Any]:
        """
        Verify code entered by user.
        
        Args:
            phone_number: Phone number to verify
            code: Code entered by user
            
        Returns:
            Verification result
        """
        # Try Twilio Verify first
        verify_provider = self.providers.get(SMSProvider.TWILIO_VERIFY)
        if verify_provider:
            return await verify_provider.check_verification(phone_number, code)
        
        # Fallback to Redis verification
        code_key = f"sms_verify:{phone_number}"
        stored_code = await self.redis.get(code_key)
        
        if not stored_code:
            return {
                "success": False,
                "valid": False,
                "error": "Verification code expired or not found"
            }
        
        is_valid = stored_code.decode() == code.strip()
        
        if is_valid:
            # Delete used code
            await self.redis.delete(code_key)
            
            # Mark phone as verified (store in user profile)
            verified_key = f"phone_verified:{phone_number}"
            await self.redis.setex(verified_key, 86400 * 30, "true")  # 30 days
        
        return {
            "success": True,
            "valid": is_valid,
            "error": "Invalid code" if not is_valid else None
        }
    
    def _generate_verification_code(self, length: int = 6) -> str:
        """Generate random verification code."""
        import random
        import string
        
        # Use digits only for SMS verification
        return ''.join(random.choices(string.digits, k=length))
    
    async def _log_sms(self, message: SMSMessage, response: SMSResponse) -> None:
        """Log SMS sending attempt."""
        log_data = {
            "message_id": message.message_id,
            "to_phone": message.to_phone,
            "sms_type": message.sms_type,
            "provider": response.provider.value if response.provider else "unknown",
            "success": response.success,
            "cost": response.cost,
            "segments": response.segments,
            "error": response.error,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if response.success:
            logger.info(f"SMS sent: {log_data}")
        else:
            logger.error(f"SMS failed: {log_data}")
        
        # Store in Redis for analytics (keep for 90 days)
        analytics_key = f"sms_analytics:{datetime.utcnow().strftime('%Y-%m-%d')}"
        await self.redis.lpush(analytics_key, json.dumps(log_data))
        await self.redis.expire(analytics_key, 90 * 24 * 3600)  # 90 days
        
        # Also store in permanent log if needed (would go to database in production)
        if settings.LOG_SMS_TO_DB:
            await self._store_sms_log(log_data)
    
    async def _store_sms_log(self, log_data: Dict[str, Any]) -> None:
        """Store SMS log in database."""
        # This would be implemented with your database model
        # For now, just log
        logger.debug(f"SMS log for database: {log_data}")
    
    async def get_daily_stats(self) -> Dict[str, Any]:
        """Get today's SMS statistics."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        counter_key = f"{self.sms_counter_key}:{today}"
        
        try:
            count = await self.redis.get(counter_key)
            total_count = int(count) if count else 0
            
            # Get analytics for today
            analytics_key = f"sms_analytics:{today}"
            analytics_items = await self.redis.lrange(analytics_key, 0, -1)
            
            successful = 0
            failed = 0
            total_cost = 0.0
            
            for item in analytics_items:
                try:
                    data = json.loads(item)
                    if data.get("success"):
                        successful += 1
                        total_cost += float(data.get("cost", 0))
                    else:
                        failed += 1
                except:
                    continue
            
            return {
                "date": today,
                "total_sent": total_count,
                "successful": successful,
                "failed": failed,
                "total_cost": round(total_cost, 4),
                "daily_limit": settings.SMS_DAILY_LIMIT or "unlimited",
                "remaining": (settings.SMS_DAILY_LIMIT - total_count) if settings.SMS_DAILY_LIMIT else None
            }
        except Exception as e:
            logger.error(f"Failed to get daily stats: {str(e)}")
            return {"error": str(e)}
    
    async def cleanup(self) -> None:
        """Clean up all providers."""
        for provider in self.providers.values():
            await provider.cleanup()
        self.initialized = False
        logger.info("SMS client cleaned up")


# ============ Factory Function ============

_sms_client: Optional[SMSClient] = None

async def get_sms_client() -> SMSClient:
    """
    Get or create an SMS client singleton.
    
    Returns:
        SMSClient instance
    """
    global _sms_client
    
    if _sms_client is None:
        _sms_client = SMSClient()
        await _sms_client.initialize()
    
    return _sms_client


# ============ Utility Functions ============

async def send_incident_alert_sms(
    phone_numbers: List[str],
    incident_title: str,
    incident_location: str,
    severity: str = "medium",
    incident_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Send incident alert SMS to multiple phone numbers.
    
    Args:
        phone_numbers: List of phone numbers to alert
        incident_title: Incident title
        incident_location: Incident location
        severity: "low", "medium", "high", "critical"
        incident_id: Optional incident ID for deep linking
        
    Returns:
        Statistics about the send operation
    """
    client = await get_sms_client()
    
    # Create messages
    messages = []
    for phone in phone_numbers:
        # Create alert message based on severity
        emoji = {
            "low": "🟢",
            "medium": "🟡",
            "high": "🟠",
            "critical": "🔴"
        }.get(severity, "🟡")
        
        body = f"{emoji} ALERT: {incident_title}\n"
        body += f"Location: {incident_location}\n"
        body += f"Severity: {severity.upper()}"
        
        if incident_id:
            body += f"\nDetails: {settings.FRONTEND_URL}/incidents/{incident_id}"
        
        message = SMSMessage(
            to_phone=phone,
            body=body,
            sms_type=SMSType.ALERT,
            metadata={
                "incident_id": incident_id,
                "severity": severity,
                "alert_type": "incident"
            }
        )
        messages.append(message)
    
    # Send batch
    if messages:
        batch_request = BatchSMSRequest(messages=messages, failover_enabled=True)
        responses = await client.send_batch(batch_request)
        
        # Calculate statistics
        successful = sum(1 for r in responses if r.success)
        total_cost = sum(r.cost or 0 for r in responses if r.success)
        
        return {
            "total_recipients": len(phone_numbers),
            "messages_sent": successful,
            "failed": len(responses) - successful,
            "total_cost": round(total_cost, 4),
            "incident_id": incident_id
        }
    
    return {"error": "No valid phone numbers provided"}


async def send_verification_sms(phone_number: str) -> Dict[str, Any]:
    """
    Send verification SMS with OTP.
    
    Args:
        phone_number: Phone number to verify
        
    Returns:
        Verification response
    """
    client = await get_sms_client()
    return await client.send_verification_code(phone_number)


async def send_user_notification_sms(
    user_id: str,
    title: str,
    message: str,
    notification_type: SMSType = SMSType.NOTIFICATION
) -> SMSResponse:
    """
    Send SMS notification to a user.
    
    Args:
        user_id: User ID
        title: Notification title (will be included in message)
        message: Notification message
        notification_type: Type of notification
        
    Returns:
        SMS response
    """
    # In production, you would:
    # 1. Look up user's phone number from database
    # 2. Check user's notification preferences
    # 3. Format message appropriately
    
    # For now, this is a placeholder
    logger.info(f"Would send SMS to user {user_id}: {title} - {message}")
    
    return SMSResponse(
        message_id=str(uuid4()),
        success=False,
        provider=SMSProvider.CUSTOM,
        error="User phone number lookup not implemented",
        status=SMSStatus.FAILED
    )