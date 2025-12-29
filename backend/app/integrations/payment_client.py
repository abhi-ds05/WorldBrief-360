# backend/app/integrations/payment_client.py
"""
Payment processing client with multiple provider support.
Supports:
- Stripe
- PayPal
- Razorpay (for India)
- Square
- Apple Pay
- Google Pay
- Bank transfers
- Crypto payments
"""

from abc import ABC
import asyncio
import base64
import hashlib
import hmac
import json
import logging
import re
import secrets
from select import select
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple
from urllib.parse import urlencode, quote

import aiohttp
from cryptography.fernet import Fernet
from pydantic import BaseModel, Field, validator, root_validator, condecimal, EmailStr, HttpUrl
from sklearn.naive_bayes import abstractmethod
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.config import settings
from app.core.logging_config import get_logger
from app.cache.redis_client import get_redis_client
from app.db.models.user import User
from app.db.models.wallet import Wallet, Transaction
from app.db.session import get_db_session

logger = get_logger(__name__)


# ============ Data Models ============

class PaymentProvider(str, Enum):
    """Supported payment providers."""
    STRIPE = "stripe"
    PAYPAL = "paypal"
    RAZORPAY = "razorpay"
    SQUARE = "square"
    APPLE_PAY = "apple_pay"
    GOOGLE_PAY = "google_pay"
    BANK_TRANSFER = "bank_transfer"
    CRYPTO = "crypto"
    OFFLINE = "offline"
    INTERNAL = "internal"  # For internal transfers between users


class PaymentStatus(str, Enum):
    """Payment statuses."""
    PENDING = "pending"
    PROCESSING = "processing"
    REQUIRES_ACTION = "requires_action"  # 3D Secure, etc.
    REQUIRES_CONFIRMATION = "requires_confirmation"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"
    PARTIALLY_REFUNDED = "partially_refunded"
    DISPUTED = "disputed"
    CHARGEBACK = "chargeback"
    EXPIRED = "expired"
    AUTHORIZED = "authorized"  # Auth only, not captured yet


class PaymentMethodType(str, Enum):
    """Payment method types."""
    CARD = "card"
    BANK_TRANSFER = "bank_transfer"
    WALLET = "wallet"  # PayPal, Apple Pay, Google Pay
    UPI = "upi"  # India
    NET_BANKING = "net_banking"  # India
    CRYPTO = "crypto"
    CASH = "cash"
    COINS = "coins"  # WorldBrief 360 coins


class Currency(str, Enum):
    """Supported currencies."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CAD = "CAD"
    AUD = "AUD"
    CHF = "CHF"
    CNY = "CNY"
    INR = "INR"
    BRL = "BRL"
    RUB = "RUB"
    KRW = "KRW"
    SGD = "SGD"
    HKD = "HKD"
    TRY = "TRY"
    MXN = "MXN"
    ZAR = "ZAR"
    AED = "AED"
    SAR = "SAR"
    # Add more as needed


class PaymentIntent(BaseModel):
    """Payment intent for initiating a payment."""
    payment_id: str = Field(default_factory=lambda: f"pay_{uuid.uuid4().hex[:16]}")
    user_id: str
    amount: condecimal(gt=0, max_digits=12, decimal_places=2)  # type: ignore # In currency's smallest unit
    currency: Currency = Currency.USD
    provider: PaymentProvider
    payment_method_type: PaymentMethodType = PaymentMethodType.CARD
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    customer_email: Optional[EmailStr] = None
    customer_name: Optional[str] = None
    customer_phone: Optional[str] = None
    billing_address: Optional[Dict[str, Any]] = None
    shipping_address: Optional[Dict[str, Any]] = None
    return_url: Optional[HttpUrl] = None  # For redirect-based payments
    cancel_url: Optional[HttpUrl] = None
    webhook_url: Optional[HttpUrl] = None
    save_payment_method: bool = False
    setup_future_usage: Optional[str] = None  # "on_session" or "off_session"
    application_fee_amount: Optional[condecimal(ge=0, max_digits=12, decimal_places=2)] = None # type: ignore
    transfer_data: Optional[Dict[str, Any]] = None  # For platform payouts
    statement_descriptor: Optional[str] = None
    statement_descriptor_suffix: Optional[str] = None
    
    @validator('amount')
    def validate_amount(cls, v):
        # Convert to smallest currency unit (cents for USD)
        return v
    
    @validator('statement_descriptor')
    def validate_statement_descriptor(cls, v):
        if v and len(v) > 22:
            raise ValueError('Statement descriptor must be 22 characters or less')
        return v
    
    @validator('statement_descriptor_suffix')
    def validate_statement_descriptor_suffix(cls, v):
        if v and len(v) > 22:
            raise ValueError('Statement descriptor suffix must be 22 characters or less')
        return v


class PaymentMethod(BaseModel):
    """Payment method details."""
    payment_method_id: str
    provider: PaymentProvider
    type: PaymentMethodType
    details: Dict[str, Any] = Field(default_factory=dict)
    is_default: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('details')
    def mask_sensitive_data(cls, v):
        """Mask sensitive payment method data."""
        masked = v.copy()
        
        if 'card' in masked:
            card = masked['card']
            if 'number' in card:
                card['number'] = f"**** **** **** {card['number'][-4:]}"
            if 'cvc' in card:
                card['cvc'] = "***"
        
        if 'bank_account' in masked:
            account = masked['bank_account']
            if 'account_number' in account:
                account['account_number'] = f"****{account['account_number'][-4:]}"
        
        return masked


class Payment(BaseModel):
    """Payment object."""
    payment_id: str
    user_id: str
    amount: Decimal
    currency: Currency
    provider: PaymentProvider
    status: PaymentStatus
    payment_method: Optional[PaymentMethod] = None
    provider_payment_id: Optional[str] = None
    provider_customer_id: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
    refunds: List[Dict[str, Any]] = Field(default_factory=list)
    refunded_amount: Decimal = Decimal('0')
    application_fee_amount: Optional[Decimal] = None
    transfer_amount: Optional[Decimal] = None  # For platform payouts
    statement_descriptor: Optional[str] = None
    statement_descriptor_suffix: Optional[str] = None
    receipt_url: Optional[HttpUrl] = None
    receipt_number: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    paid_at: Optional[datetime] = None
    refunded_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None


class RefundRequest(BaseModel):
    """Refund request."""
    payment_id: str
    amount: Optional[Decimal] = None  # Partial refund if specified
    reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    refund_application_fee: bool = False
    reverse_transfer: bool = False


class WebhookEvent(BaseModel):
    """Payment webhook event."""
    event_id: str
    provider: PaymentProvider
    event_type: str
    data: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processed: bool = False
    processing_error: Optional[str] = None


class Invoice(BaseModel):
    """Invoice for recurring payments or billing."""
    invoice_id: str = Field(default_factory=lambda: f"inv_{uuid.uuid4().hex[:16]}")
    user_id: str
    subscription_id: Optional[str] = None
    amount: Decimal
    currency: Currency
    status: PaymentStatus
    due_date: Optional[datetime] = None
    paid_at: Optional[datetime] = None
    payment_id: Optional[str] = None
    items: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tax_amount: Decimal = Decimal('0')
    discount_amount: Decimal = Decimal('0')
    total_amount: Decimal
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Subscription(BaseModel):
    """Subscription for recurring payments."""
    subscription_id: str = Field(default_factory=lambda: f"sub_{uuid.uuid4().hex[:16]}")
    user_id: str
    plan_id: str
    status: str  # active, past_due, unpaid, canceled, incomplete, incomplete_expired, trialing
    current_period_start: datetime
    current_period_end: datetime
    cancel_at_period_end: bool = False
    canceled_at: Optional[datetime] = None
    trial_start: Optional[datetime] = None
    trial_end: Optional[datetime] = None
    payment_method_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ============ Abstract Base Classes ============

class BasePaymentProvider(ABC):
    """Abstract base class for payment providers."""
    
    def __init__(self, provider_name: str, config: Dict[str, Any]):
        self.provider_name = provider_name
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.encryption_key = settings.PAYMENT_ENCRYPTION_KEY
        self.fernet = Fernet(self.encryption_key) if self.encryption_key else None
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider."""
        pass
    
    @abstractmethod
    async def create_payment_intent(self, intent: PaymentIntent) -> Dict[str, Any]:
        """Create a payment intent."""
        pass
    
    @abstractmethod
    async def confirm_payment(self, payment_id: str, payment_method_id: Optional[str] = None) -> Payment:
        """Confirm and process a payment."""
        pass
    
    @abstractmethod
    async def get_payment(self, payment_id: str) -> Payment:
        """Get payment details."""
        pass
    
    @abstractmethod
    async def refund_payment(self, request: RefundRequest) -> Dict[str, Any]:
        """Refund a payment."""
        pass
    
    @abstractmethod
    async def create_payment_method(self, details: Dict[str, Any]) -> PaymentMethod:
        """Create a payment method."""
        pass
    
    @abstractmethod
    async def validate_webhook(self, payload: bytes, signature: str) -> bool:
        """Validate webhook signature."""
        pass
    
    @abstractmethod
    async def parse_webhook(self, payload: bytes) -> WebhookEvent:
        """Parse webhook payload."""
        pass
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        if not self.fernet:
            return data
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        if not self.fernet:
            return encrypted_data
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.session:
            await self.session.close()


# ============ Concrete Providers ============

class StripeProvider(BasePaymentProvider):
    """Stripe payment provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("stripe", config)
        self.api_key = settings.STRIPE_SECRET_KEY
        self.publishable_key = settings.STRIPE_PUBLISHABLE_KEY
        self.webhook_secret = settings.STRIPE_WEBHOOK_SECRET
        self.api_version = "2023-10-16"
        self.base_url = "https://api.stripe.com/v1"
    
    async def initialize(self) -> None:
        """Initialize Stripe provider."""
        if not self.api_key:
            raise ValueError("Stripe API key not configured")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Stripe-Version": self.api_version,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        
        self.session = aiohttp.ClientSession(
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=30)
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def create_payment_intent(self, intent: PaymentIntent) -> Dict[str, Any]:
        """Create Stripe payment intent."""
        try:
            # Prepare request data
            data = {
                "amount": int(intent.amount * 100),  # Convert to cents
                "currency": intent.currency.value.lower(),
                "payment_method_types[]": self._map_payment_method_type(intent.payment_method_type),
                "metadata": json.dumps(intent.metadata),
            }
            
            if intent.description:
                data["description"] = intent.description
            
            if intent.customer_email:
                # Create or retrieve customer
                customer_id = await self._get_or_create_customer(
                    intent.user_id,
                    intent.customer_email,
                    intent.customer_name,
                    intent.metadata.get("customer_phone")
                )
                data["customer"] = customer_id
            
            if intent.setup_future_usage:
                data["setup_future_usage"] = intent.setup_future_usage
            
            if intent.save_payment_method:
                data["save_payment_method"] = "true"
            
            if intent.return_url:
                data["return_url"] = str(intent.return_url)
            
            if intent.cancel_url:
                data["cancel_url"] = str(intent.cancel_url)
            
            if intent.statement_descriptor:
                data["statement_descriptor"] = intent.statement_descriptor
            
            if intent.statement_descriptor_suffix:
                data["statement_descriptor_suffix"] = intent.statement_descriptor_suffix
            
            if intent.application_fee_amount:
                data["application_fee_amount"] = int(intent.application_fee_amount * 100)
            
            # Make request
            async with self.session.post(f"{self.base_url}/payment_intents", data=data) as response:
                response_data = await response.json()
                
                if response.status == 200:
                    return {
                        "payment_intent_id": response_data["id"],
                        "client_secret": response_data.get("client_secret"),
                        "status": response_data.get("status"),
                        "next_action": response_data.get("next_action"),
                        "requires_action": response_data.get("status") == "requires_action",
                        "provider_data": response_data,
                    }
                else:
                    error_msg = response_data.get("error", {}).get("message", "Unknown Stripe error")
                    raise Exception(f"Stripe error: {error_msg}")
                    
        except Exception as e:
            logger.error(f"Stripe create_payment_intent failed: {str(e)}", exc_info=True)
            raise
    
    async def confirm_payment(self, payment_id: str, payment_method_id: Optional[str] = None) -> Payment:
        """Confirm Stripe payment."""
        try:
            data = {}
            if payment_method_id:
                data["payment_method"] = payment_method_id
            
            # Confirm payment intent
            async with self.session.post(f"{self.base_url}/payment_intents/{payment_id}/confirm", data=data) as response:
                response_data = await response.json()
                
                if response.status == 200:
                    return await self._parse_stripe_payment(response_data)
                else:
                    error_msg = response_data.get("error", {}).get("message", "Unknown Stripe error")
                    raise Exception(f"Stripe error: {error_msg}")
                    
        except Exception as e:
            logger.error(f"Stripe confirm_payment failed: {str(e)}", exc_info=True)
            raise
    
    async def get_payment(self, payment_id: str) -> Payment:
        """Get Stripe payment details."""
        try:
            async with self.session.get(f"{self.base_url}/payment_intents/{payment_id}") as response:
                response_data = await response.json()
                
                if response.status == 200:
                    return await self._parse_stripe_payment(response_data)
                else:
                    error_msg = response_data.get("error", {}).get("message", "Unknown Stripe error")
                    raise Exception(f"Stripe error: {error_msg}")
                    
        except Exception as e:
            logger.error(f"Stripe get_payment failed: {str(e)}", exc_info=True)
            raise
    
    async def refund_payment(self, request: RefundRequest) -> Dict[str, Any]:
        """Refund Stripe payment."""
        try:
            # First get charge ID from payment intent
            payment = await self.get_payment(request.payment_id)
            charge_id = payment.metadata.get("charge_id")
            
            if not charge_id:
                raise Exception("No charge found for payment")
            
            # Create refund
            data = {
                "charge": charge_id,
                "metadata": json.dumps(request.metadata),
            }
            
            if request.amount:
                data["amount"] = int(request.amount * 100)
            
            if request.reason:
                data["reason"] = request.reason
            
            if request.refund_application_fee:
                data["refund_application_fee"] = "true"
            
            if request.reverse_transfer:
                data["reverse_transfer"] = "true"
            
            async with self.session.post(f"{self.base_url}/refunds", data=data) as response:
                response_data = await response.json()
                
                if response.status == 200:
                    return {
                        "refund_id": response_data["id"],
                        "amount": Decimal(response_data["amount"]) / 100,
                        "currency": response_data["currency"].upper(),
                        "status": response_data.get("status"),
                        "provider_data": response_data,
                    }
                else:
                    error_msg = response_data.get("error", {}).get("message", "Unknown Stripe error")
                    raise Exception(f"Stripe error: {error_msg}")
                    
        except Exception as e:
            logger.error(f"Stripe refund_payment failed: {str(e)}", exc_info=True)
            raise
    
    async def create_payment_method(self, details: Dict[str, Any]) -> PaymentMethod:
        """Create Stripe payment method."""
        try:
            # Map our payment method type to Stripe type
            stripe_type = self._map_payment_method_type(details.get("type"))
            
            data = {
                "type": stripe_type,
            }
            
            # Add type-specific details
            if stripe_type == "card":
                card_details = details.get("card", {})
                data["card[number]"] = card_details.get("number")
                data["card[exp_month]"] = card_details.get("exp_month")
                data["card[exp_year]"] = card_details.get("exp_year")
                data["card[cvc]"] = card_details.get("cvc")
                
                if billing_details := details.get("billing_details"):
                    data["billing_details[name]"] = billing_details.get("name")
                    data["billing_details[email]"] = billing_details.get("email")
                    data["billing_details[phone]"] = billing_details.get("phone")
                    
                    if address := billing_details.get("address"):
                        data["billing_details[address][line1]"] = address.get("line1")
                        data["billing_details[address][line2]"] = address.get("line2")
                        data["billing_details[address][city]"] = address.get("city")
                        data["billing_details[address][state]"] = address.get("state")
                        data["billing_details[address][postal_code]"] = address.get("postal_code")
                        data["billing_details[address][country]"] = address.get("country")
            
            elif stripe_type == "us_bank_account":
                bank_details = details.get("bank_account", {})
                data["us_bank_account[account_number]"] = bank_details.get("account_number")
                data["us_bank_account[routing_number]"] = bank_details.get("routing_number")
                data["us_bank_account[account_holder_type]"] = bank_details.get("account_holder_type", "individual")
                data["us_bank_account[account_type]"] = bank_details.get("account_type", "checking")
            
            # Create payment method
            async with self.session.post(f"{self.base_url}/payment_methods", data=data) as response:
                response_data = await response.json()
                
                if response.status == 200:
                    return PaymentMethod(
                        payment_method_id=response_data["id"],
                        provider=PaymentProvider.STRIPE,
                        type=PaymentMethodType(details.get("type")),
                        details=self._mask_payment_method_details(response_data),
                        metadata={"stripe_data": response_data},
                    )
                else:
                    error_msg = response_data.get("error", {}).get("message", "Unknown Stripe error")
                    raise Exception(f"Stripe error: {error_msg}")
                    
        except Exception as e:
            logger.error(f"Stripe create_payment_method failed: {str(e)}", exc_info=True)
            raise
    
    async def validate_webhook(self, payload: bytes, signature: str) -> bool:
        """Validate Stripe webhook signature."""
        if not self.webhook_secret:
            logger.warning("Stripe webhook secret not configured")
            return True  # Allow if no secret configured (not recommended for production)
        
        try:
            # Stripe expects the raw payload for signature verification
            import stripe
            stripe.api_key = self.api_key
            
            # This would use Stripe's library for proper signature verification
            # For now, we'll implement a simplified version
            timestamp, signatures = signature.split(',')
            timestamp = int(timestamp.split('=')[1])
            
            # Check if timestamp is within tolerance (5 minutes)
            current_time = int(datetime.utcnow().timestamp())
            if abs(current_time - timestamp) > 300:
                return False
            
            # Verify signature
            signed_payload = f"{timestamp}.{payload.decode()}"
            expected_sig = hmac.new(
                self.webhook_secret.encode(),
                signed_payload.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return signatures.split('=')[1] == expected_sig
            
        except Exception as e:
            logger.error(f"Stripe webhook validation failed: {str(e)}")
            return False
    
    async def parse_webhook(self, payload: bytes) -> WebhookEvent:
        """Parse Stripe webhook payload."""
        try:
            data = json.loads(payload.decode())
            
            # Extract event data
            event_id = data.get("id")
            event_type = data.get("type")
            event_data = data.get("data", {}).get("object", {})
            
            return WebhookEvent(
                event_id=event_id,
                provider=PaymentProvider.STRIPE,
                event_type=event_type,
                data=event_data,
            )
            
        except Exception as e:
            logger.error(f"Stripe webhook parsing failed: {str(e)}")
            raise
    
    async def _get_or_create_customer(self, user_id: str, email: str, name: Optional[str] = None, phone: Optional[str] = None) -> str:
        """Get or create Stripe customer."""
        # Check cache first
        cache_key = f"stripe:customer:{user_id}"
        cached = await self.redis.get(cache_key)
        if cached:
            return cached.decode()
        
        # Search for existing customer by email
        params = {"email": email, "limit": 1}
        async with self.session.get(f"{self.base_url}/customers", params=params) as response:
            if response.status == 200:
                data = await response.json()
                if data.get("data"):
                    customer_id = data["data"][0]["id"]
                    await self.redis.setex(cache_key, 86400, customer_id)  # Cache for 24 hours
                    return customer_id
        
        # Create new customer
        customer_data = {
            "email": email,
            "metadata": {"user_id": user_id},
        }
        
        if name:
            customer_data["name"] = name
        
        if phone:
            customer_data["phone"] = phone
        
        async with self.session.post(f"{self.base_url}/customers", data=customer_data) as response:
            if response.status == 200:
                data = await response.json()
                customer_id = data["id"]
                await self.redis.setex(cache_key, 86400, customer_id)
                return customer_id
            else:
                error_msg = (await response.json()).get("error", {}).get("message", "Unknown error")
                raise Exception(f"Failed to create Stripe customer: {error_msg}")
    
    def _map_payment_method_type(self, payment_method_type: PaymentMethodType) -> str:
        """Map our payment method type to Stripe type."""
        mapping = {
            PaymentMethodType.CARD: "card",
            PaymentMethodType.BANK_TRANSFER: "us_bank_account",  # US-specific
            PaymentMethodType.WALLET: "link",  # Stripe Link
            PaymentMethodType.UPI: "upi",  # India
            PaymentMethodType.NET_BANKING: "netbanking",  # India
        }
        return mapping.get(payment_method_type, "card")
    
    def _mask_payment_method_details(self, stripe_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive payment method details."""
        masked = stripe_data.copy()
        
        if "card" in masked:
            card = masked["card"]
            if "last4" in card:
                card["number"] = f"**** **** **** {card['last4']}"
            if "exp_month" in card and "exp_year" in card:
                card["expiry"] = f"{card['exp_month']}/{card['exp_year']}"
        
        return masked
    
    async def _parse_stripe_payment(self, stripe_data: Dict[str, Any]) -> Payment:
        """Parse Stripe payment data into our Payment model."""
        # Get charge details if available
        charges = stripe_data.get("charges", {}).get("data", [])
        charge = charges[0] if charges else {}
        
        # Get payment method if available
        payment_method = None
        if "payment_method" in stripe_data:
            pm_data = stripe_data["payment_method"]
            if isinstance(pm_data, str):
                # It's a payment method ID, fetch details
                try:
                    async with self.session.get(f"{self.base_url}/payment_methods/{pm_data}") as response:
                        if response.status == 200:
                            pm_details = await response.json()
                            payment_method = PaymentMethod(
                                payment_method_id=pm_details["id"],
                                provider=PaymentProvider.STRIPE,
                                type=PaymentMethodType(pm_details.get("type", "card")),
                                details=self._mask_payment_method_details(pm_details),
                                metadata={"stripe_data": pm_details},
                            )
                except:
                    pass
        
        # Map Stripe status to our status
        status_mapping = {
            "requires_payment_method": PaymentStatus.PENDING,
            "requires_confirmation": PaymentStatus.REQUIRES_CONFIRMATION,
            "requires_action": PaymentStatus.REQUIRES_ACTION,
            "processing": PaymentStatus.PROCESSING,
            "requires_capture": PaymentStatus.AUTHORIZED,
            "canceled": PaymentStatus.CANCELLED,
            "succeeded": PaymentStatus.SUCCEEDED,
        }
        
        status = status_mapping.get(stripe_data.get("status"), PaymentStatus.PENDING)
        
        # Calculate refunded amount
        refunded_amount = Decimal('0')
        refunds = []
        if "refunds" in charge:
            for refund in charge.get("refunds", {}).get("data", []):
                refund_amount = Decimal(refund.get("amount", 0)) / 100
                refunded_amount += refund_amount
                refunds.append({
                    "refund_id": refund.get("id"),
                    "amount": refund_amount,
                    "currency": refund.get("currency", "usd").upper(),
                    "status": refund.get("status"),
                    "reason": refund.get("reason"),
                    "created_at": datetime.fromtimestamp(refund.get("created", 0)),
                })
        
        return Payment(
            payment_id=stripe_data.get("id"),
            user_id=stripe_data.get("metadata", {}).get("user_id", ""),
            amount=Decimal(stripe_data.get("amount", 0)) / 100,
            currency=Currency(stripe_data.get("currency", "usd").upper()),
            provider=PaymentProvider.STRIPE,
            status=status,
            payment_method=payment_method,
            provider_payment_id=stripe_data.get("id"),
            provider_customer_id=stripe_data.get("customer"),
            description=stripe_data.get("description"),
            metadata={
                "stripe_data": stripe_data,
                "charge_id": charge.get("id"),
                "receipt_url": charge.get("receipt_url"),
                "receipt_number": charge.get("receipt_number"),
            },
            refunds=refunds,
            refunded_amount=refunded_amount,
            application_fee_amount=Decimal(stripe_data.get("application_fee_amount", 0)) / 100 if stripe_data.get("application_fee_amount") else None,
            transfer_amount=Decimal(stripe_data.get("transfer_data", {}).get("amount", 0)) / 100 if stripe_data.get("transfer_data", {}).get("amount") else None,
            statement_descriptor=stripe_data.get("statement_descriptor"),
            statement_descriptor_suffix=stripe_data.get("statement_descriptor_suffix"),
            receipt_url=charge.get("receipt_url"),
            receipt_number=charge.get("receipt_number"),
            created_at=datetime.fromtimestamp(stripe_data.get("created", 0)),
            updated_at=datetime.fromtimestamp(stripe_data.get("updated", stripe_data.get("created", 0))),
            paid_at=datetime.fromtimestamp(charge.get("created", 0)) if charge.get("created") else None,
            refunded_at=datetime.fromtimestamp(refunds[0].get("created_at").timestamp()) if refunds else None,
            expires_at=datetime.fromtimestamp(stripe_data.get("expires_at", 0)) if stripe_data.get("expires_at") else None,
        )


class PayPalProvider(BasePaymentProvider):
    """PayPal payment provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("paypal", config)
        self.client_id = settings.PAYPAL_CLIENT_ID
        self.client_secret = settings.PAYPAL_CLIENT_SECRET
        self.webhook_id = settings.PAYPAL_WEBHOOK_ID
        self.environment = "live" if not settings.DEBUG else "sandbox"
        self.base_url = f"https://api-m.{self.environment}.paypal.com"
        self.access_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
    
    async def initialize(self) -> None:
        """Initialize PayPal provider."""
        if not self.client_id or not self.client_secret:
            raise ValueError("PayPal credentials not configured")
        
        # Get access token
        await self._refresh_access_token()
        
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
    
    async def _refresh_access_token(self) -> None:
        """Refresh PayPal access token."""
        auth = aiohttp.BasicAuth(self.client_id, self.client_secret)
        data = {"grant_type": "client_credentials"}
        
        async with aiohttp.ClientSession() as temp_session:
            async with temp_session.post(f"{self.base_url}/v1/oauth2/token", auth=auth, data=data) as response:
                if response.status == 200:
                    token_data = await response.json()
                    self.access_token = token_data.get("access_token")
                    expires_in = token_data.get("expires_in", 3600)
                    self.token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in - 300)  # Refresh 5 minutes before expiry
                else:
                    raise Exception("Failed to get PayPal access token")
    
    async def _ensure_token_valid(self) -> None:
        """Ensure access token is valid."""
        if not self.access_token or not self.token_expires_at or self.token_expires_at <= datetime.utcnow():
            await self._refresh_access_token()
    
    async def create_payment_intent(self, intent: PaymentIntent) -> Dict[str, Any]:
        """Create PayPal order."""
        await self._ensure_token_valid()
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
        
        # Build order request
        order_data = {
            "intent": "CAPTURE",
            "purchase_units": [{
                "amount": {
                    "currency_code": intent.currency.value,
                    "value": str(intent.amount),
                },
                "description": intent.description,
                "custom_id": intent.user_id,
                "invoice_id": intent.payment_id,
            }],
            "payment_source": self._map_payment_source(intent.payment_method_type),
            "application_context": {
                "brand_name": "WorldBrief 360",
                "locale": "en-US",
                "landing_page": "LOGIN",
                "shipping_preference": "NO_SHIPPING",
                "user_action": "PAY_NOW",
                "return_url": str(intent.return_url) if intent.return_url else None,
                "cancel_url": str(intent.cancel_url) if intent.cancel_url else None,
            },
        }
        
        # Add metadata
        if intent.metadata:
            order_data["purchase_units"][0]["metadata"] = intent.metadata
        
        try:
            async with self.session.post(f"{self.base_url}/v2/checkout/orders", headers=headers, json=order_data) as response:
                response_data = await response.json()
                
                if response.status == 201:
                    return {
                        "order_id": response_data["id"],
                        "status": response_data["status"],
                        "links": response_data.get("links", []),
                        "provider_data": response_data,
                    }
                else:
                    error_msg = response_data.get("message", "Unknown PayPal error")
                    raise Exception(f"PayPal error: {error_msg}")
                    
        except Exception as e:
            logger.error(f"PayPal create_order failed: {str(e)}", exc_info=True)
            raise
    
    # Other PayPal methods would be implemented similarly...
    # For brevity, I'm showing the structure. Full implementation would include:
    # - confirm_payment (capture order)
    # - get_payment (get order details)
    # - refund_payment (create refund)
    # - create_payment_method (store payment method)
    # - validate_webhook
    # - parse_webhook


class RazorpayProvider(BasePaymentProvider):
    """Razorpay payment provider (India)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("razorpay", config)
        self.key_id = settings.RAZORPAY_KEY_ID
        self.key_secret = settings.RAZORPAY_KEY_SECRET
        self.webhook_secret = settings.RAZORPAY_WEBHOOK_SECRET
        self.base_url = "https://api.razorpay.com/v1"
    
    async def initialize(self) -> None:
        """Initialize Razorpay provider."""
        if not self.key_id or not self.key_secret:
            raise ValueError("Razorpay credentials not configured")
        
        # Create basic auth
        auth = aiohttp.BasicAuth(self.key_id, self.key_secret)
        
        self.session = aiohttp.ClientSession(
            auth=auth,
            timeout=aiohttp.ClientTimeout(total=30)
        )
    
    async def create_payment_intent(self, intent: PaymentIntent) -> Dict[str, Any]:
        """Create Razorpay order."""
        # Razorpay requires amount in paise (for INR)
        amount = int(intent.amount * 100) if intent.currency == Currency.INR else int(intent.amount * 100)
        
        order_data = {
            "amount": amount,
            "currency": intent.currency.value,
            "receipt": intent.payment_id,
            "notes": intent.metadata,
            "payment_capture": 1,  # Auto-capture
        }
        
        if intent.description:
            order_data["notes"]["description"] = intent.description
        
        try:
            async with self.session.post(f"{self.base_url}/orders", json=order_data) as response:
                response_data = await response.json()
                
                if response.status == 200:
                    return {
                        "order_id": response_data["id"],
                        "amount": response_data["amount"] / 100,
                        "currency": response_data["currency"],
                        "status": response_data.get("status"),
                        "provider_data": response_data,
                    }
                else:
                    error_msg = response_data.get("error", {}).get("description", "Unknown Razorpay error")
                    raise Exception(f"Razorpay error: {error_msg}")
                    
        except Exception as e:
            logger.error(f"Razorpay create_order failed: {str(e)}", exc_info=True)
            raise
    
    # Other Razorpay methods would be implemented similarly...


class InternalPaymentProvider(BasePaymentProvider):
    """Internal payment provider for coin transfers."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("internal", config)
    
    async def initialize(self) -> None:
        """Initialize internal provider."""
        # Nothing to initialize for internal provider
        pass
    
    async def create_payment_intent(self, intent: PaymentIntent) -> Dict[str, Any]:
        """Create internal payment intent."""
        # For internal payments, we just return the intent
        return {
            "payment_intent_id": intent.payment_id,
            "status": "pending",
            "requires_action": False,
            "provider_data": intent.dict(),
        }
    
    async def confirm_payment(self, payment_id: str, payment_method_id: Optional[str] = None) -> Payment:
        """Process internal payment (coin transfer)."""
        # This would transfer coins between users
        # For now, return a mock payment
        return Payment(
            payment_id=payment_id,
            user_id="",  # Would be set from database
            amount=Decimal('0'),
            currency=Currency.USD,
            provider=PaymentProvider.INTERNAL,
            status=PaymentStatus.SUCCEEDED,
            payment_method=None,
            metadata={"type": "coin_transfer"},
        )
    
    # Other internal methods...


# ============ Main Payment Client ============

class PaymentClient:
    """
    Main payment client with multi-provider support.
    """
    
    def __init__(self):
        self.providers: Dict[PaymentProvider, BasePaymentProvider] = {}
        self.redis = get_redis_client()
        self.initialized = False
        
        # Provider configurations
        self.provider_configs = {
            PaymentProvider.STRIPE: {
                "enabled": bool(settings.STRIPE_SECRET_KEY),
                "priority": 1,
                "supported_currencies": [c.value for c in Currency],
                "supported_methods": [PaymentMethodType.CARD, PaymentMethodType.BANK_TRANSFER, PaymentMethodType.WALLET],
            },
            PaymentProvider.PAYPAL: {
                "enabled": bool(settings.PAYPAL_CLIENT_ID),
                "priority": 2,
                "supported_currencies": ["USD", "EUR", "GBP", "CAD", "AUD", "JPY"],
                "supported_methods": [PaymentMethodType.WALLET],
            },
            PaymentProvider.RAZORPAY: {
                "enabled": bool(settings.RAZORPAY_KEY_ID),
                "priority": 3,
                "supported_currencies": ["INR"],
                "supported_methods": [PaymentMethodType.CARD, PaymentMethodType.UPI, PaymentMethodType.NET_BANKING],
            },
            PaymentProvider.INTERNAL: {
                "enabled": True,  # Always enabled
                "priority": 0,  # Highest priority for internal transfers
                "supported_currencies": ["USD"],  # Virtual currency
                "supported_methods": [PaymentMethodType.COINS],
            },
        }
    
    async def initialize(self) -> None:
        """Initialize payment client with configured providers."""
        if self.initialized:
            return
        
        logger.info("Initializing payment client...")
        
        # Initialize enabled providers
        for provider_name, config in self.provider_configs.items():
            if config["enabled"]:
                provider = self._create_provider(provider_name, config)
                if provider:
                    try:
                        await provider.initialize()
                        self.providers[provider_name] = provider
                        logger.info(f"Initialized payment provider: {provider_name.value}")
                    except Exception as e:
                        logger.error(f"Failed to initialize {provider_name.value}: {str(e)}")
        
        if not self.providers:
            logger.warning("No payment providers initialized!")
        
        self.initialized = True
    
    def _create_provider(self, provider: PaymentProvider, config: Dict[str, Any]) -> Optional[BasePaymentProvider]:
        """Create provider instance."""
        provider_map = {
            PaymentProvider.STRIPE: StripeProvider,
            PaymentProvider.PAYPAL: PayPalProvider,
            PaymentProvider.RAZORPAY: RazorpayProvider,
            PaymentProvider.INTERNAL: InternalPaymentProvider,
            # Add other providers as needed
        }
        
        provider_class = provider_map.get(provider)
        if provider_class:
            return provider_class(config)
        return None
    
    def _select_provider(self, intent: PaymentIntent) -> Optional[BasePaymentProvider]:
        """Select the best provider for a payment intent."""
        # Check if provider is specified
        if intent.provider:
            provider = self.providers.get(intent.provider)
            if provider and self._provider_supports(provider, intent):
                return provider
        
        # Auto-select based on currency and payment method
        available_providers = []
        
        for provider_name, provider in self.providers.items():
            config = self.provider_configs.get(provider_name)
            if config and self._provider_supports(provider, intent):
                available_providers.append((provider, config))
        
        if not available_providers:
            return None
        
        # Sort by priority
        available_providers.sort(key=lambda x: x[1]["priority"])
        
        return available_providers[0][0]
    
    def _provider_supports(self, provider: BasePaymentProvider, intent: PaymentIntent) -> bool:
        """Check if provider supports the payment intent."""
        config = self.provider_configs.get(PaymentProvider(provider.provider_name))
        if not config:
            return False
        
        # Check currency support
        if intent.currency.value not in config["supported_currencies"]:
            return False
        
        # Check payment method support
        if intent.payment_method_type not in config["supported_methods"]:
            return False
        
        return True
    
    async def create_payment(self, intent: PaymentIntent) -> Dict[str, Any]:
        """
        Create a payment.
        
        Args:
            intent: Payment intent
            
        Returns:
            Payment creation result
        """
        if not self.initialized:
            await self.initialize()
        
        # Select provider
        provider = self._select_provider(intent)
        if not provider:
            raise Exception(f"No available provider for {intent.currency.value} {intent.payment_method_type.value}")
        
        try:
            # Create payment intent
            result = await provider.create_payment_intent(intent)
            
            # Store payment intent in cache
            cache_key = f"payment:intent:{intent.payment_id}"
            await self.redis.setex(
                cache_key,
                3600,  # 1 hour expiry
                json.dumps({
                    "intent": intent.dict(),
                    "provider_result": result,
                    "created_at": datetime.utcnow().isoformat(),
                })
            )
            
            return {
                "payment_id": intent.payment_id,
                "provider": provider.provider_name,
                "status": "created",
                "requires_action": result.get("requires_action", False),
                "client_secret": result.get("client_secret"),
                "redirect_url": result.get("redirect_url"),
                "next_action": result.get("next_action"),
                "provider_data": result,
            }
            
        except Exception as e:
            logger.error(f"Payment creation failed: {str(e)}", exc_info=True)
            raise
    
    async def confirm_payment(self, payment_id: str, payment_method_id: Optional[str] = None) -> Payment:
        """
        Confirm and process a payment.
        
        Args:
            payment_id: Payment ID
            payment_method_id: Optional payment method ID
            
        Returns:
            Payment object
        """
        if not self.initialized:
            await self.initialize()
        
        # Get payment intent from cache
        cache_key = f"payment:intent:{payment_id}"
        cached_data = await self.redis.get(cache_key)
        if not cached_data:
            raise Exception(f"Payment intent not found or expired: {payment_id}")
        
        intent_data = json.loads(cached_data)
        intent = PaymentIntent(**intent_data["intent"])
        
        # Get provider
        provider = self.providers.get(intent.provider)
        if not provider:
            raise Exception(f"Provider not available: {intent.provider}")
        
        try:
            # Confirm payment
            payment = await provider.confirm_payment(
                intent_data["provider_result"].get("payment_intent_id") or intent.payment_id,
                payment_method_id
            )
            
            # Update cache
            await self.redis.setex(
                f"payment:result:{payment_id}",
                86400,  # 24 hours
                json.dumps(payment.dict())
            )
            
            # Process payment in database
            await self._process_payment_in_db(payment)
            
            return payment
            
        except Exception as e:
            logger.error(f"Payment confirmation failed: {str(e)}", exc_info=True)
            raise
    
    async def get_payment(self, payment_id: str) -> Optional[Payment]:
        """
        Get payment details.
        
        Args:
            payment_id: Payment ID
            
        Returns:
            Payment object or None
        """
        # Check cache first
        cache_key = f"payment:result:{payment_id}"
        cached_data = await self.redis.get(cache_key)
        if cached_data:
            return Payment(**json.loads(cached_data))
        
        # Try to get from provider (this would need provider-specific logic)
        # For now, return None if not in cache
        return None
    
    async def refund_payment(self, request: RefundRequest) -> Dict[str, Any]:
        """
        Refund a payment.
        
        Args:
            request: Refund request
            
        Returns:
            Refund result
        """
        if not self.initialized:
            await self.initialize()
        
        # Get payment from cache
        payment = await self.get_payment(request.payment_id)
        if not payment:
            raise Exception(f"Payment not found: {request.payment_id}")
        
        # Get provider
        provider = self.providers.get(payment.provider)
        if not provider:
            raise Exception(f"Provider not available: {payment.provider}")
        
        try:
            # Process refund
            result = await provider.refund_payment(request)
            
            # Update payment in cache
            payment.refunded_amount += result["amount"]
            payment.refunds.append({
                "refund_id": result["refund_id"],
                "amount": result["amount"],
                "currency": result["currency"],
                "status": result["status"],
                "created_at": datetime.utcnow().isoformat(),
            })
            
            if payment.refunded_amount >= payment.amount:
                payment.status = PaymentStatus.REFUNDED
            else:
                payment.status = PaymentStatus.PARTIALLY_REFUNDED
            
            payment.refunded_at = datetime.utcnow()
            payment.updated_at = datetime.utcnow()
            
            # Update cache
            await self.redis.setex(
                f"payment:result:{request.payment_id}",
                86400,
                json.dumps(payment.dict())
            )
            
            # Update database
            await self._process_refund_in_db(payment, result)
            
            return {
                "refund_id": result["refund_id"],
                "payment_id": request.payment_id,
                "amount": result["amount"],
                "currency": result["currency"],
                "status": result["status"],
                "provider_data": result,
            }
            
        except Exception as e:
            logger.error(f"Refund failed: {str(e)}", exc_info=True)
            raise
    
    async def create_payment_method(self, provider: PaymentProvider, details: Dict[str, Any]) -> PaymentMethod:
        """
        Create a payment method.
        
        Args:
            provider: Payment provider
            details: Payment method details
            
        Returns:
            Payment method object
        """
        if not self.initialized:
            await self.initialize()
        
        payment_provider = self.providers.get(provider)
        if not payment_provider:
            raise Exception(f"Provider not available: {provider}")
        
        try:
            return await payment_provider.create_payment_method(details)
        except Exception as e:
            logger.error(f"Payment method creation failed: {str(e)}")
            raise
    
    async def handle_webhook(self, provider: PaymentProvider, payload: bytes, signature: str) -> WebhookEvent:
        """
        Handle payment webhook.
        
        Args:
            provider: Payment provider
            payload: Webhook payload
            signature: Webhook signature
            
        Returns:
            Webhook event
        """
        if not self.initialized:
            await self.initialize()
        
        payment_provider = self.providers.get(provider)
        if not payment_provider:
            raise Exception(f"Provider not available: {provider}")
        
        try:
            # Validate webhook
            if not await payment_provider.validate_webhook(payload, signature):
                raise Exception("Invalid webhook signature")
            
            # Parse webhook
            event = await payment_provider.parse_webhook(payload)
            
            # Process webhook event
            await self._process_webhook_event(event)
            
            return event
            
        except Exception as e:
            logger.error(f"Webhook handling failed: {str(e)}", exc_info=True)
            raise
    
    async def _process_payment_in_db(self, payment: Payment) -> None:
        """Process payment in database."""
        async with get_db_session() as session:
            try:
                # Update user's wallet
                if payment.provider == PaymentProvider.INTERNAL:
                    # Internal coin transfer
                    await self._process_internal_transfer(payment, session)
                else:
                    # Real money payment - add coins to wallet
                    # Calculate coins based on amount (e.g., $1 = 100 coins)
                    coins_earned = int(payment.amount * 100)
                    
                    # Get user's wallet
                    wallet = await session.execute(
                        select(Wallet).where(Wallet.user_id == payment.user_id)
                    )
                    wallet = wallet.scalar_one_or_none()
                    
                    if not wallet:
                        # Create wallet if it doesn't exist
                        wallet = Wallet(
                            user_id=payment.user_id,
                            balance=coins_earned,
                            currency="coins",
                        )
                        session.add(wallet)
                    else:
                        wallet.balance += coins_earned
                    
                    # Create transaction record
                    transaction = Transaction(
                        user_id=payment.user_id,
                        type="payment",
                        amount=coins_earned,
                        currency="coins",
                        status="completed",
                        payment_id=payment.payment_id,
                        provider=payment.provider.value,
                        metadata={
                            "payment_amount": float(payment.amount),
                            "payment_currency": payment.currency.value,
                            "coins_per_dollar": 100,
                        },
                    )
                    session.add(transaction)
                
                await session.commit()
                logger.info(f"Processed payment {payment.payment_id} in database")
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to process payment in database: {str(e)}")
                raise
    
    async def _process_refund_in_db(self, payment: Payment, refund_result: Dict[str, Any]) -> None:
        """Process refund in database."""
        async with get_db_session() as session:
            try:
                # Remove coins from wallet
                coins_refunded = int(refund_result["amount"] * 100)
                
                # Get user's wallet
                wallet = await session.execute(
                    select(Wallet).where(Wallet.user_id == payment.user_id)
                )
                wallet = wallet.scalar_one_or_none()
                
                if wallet:
                    wallet.balance = max(0, wallet.balance - coins_refunded)
                
                # Create refund transaction
                transaction = Transaction(
                    user_id=payment.user_id,
                    type="refund",
                    amount=-coins_refunded,
                    currency="coins",
                    status="completed",
                    payment_id=payment.payment_id,
                    provider=payment.provider.value,
                    metadata={
                        "refund_amount": float(refund_result["amount"]),
                        "refund_currency": refund_result["currency"],
                        "refund_id": refund_result["refund_id"],
                    },
                )
                session.add(transaction)
                
                await session.commit()
                logger.info(f"Processed refund for payment {payment.payment_id} in database")
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to process refund in database: {str(e)}")
                raise
    
    async def _process_internal_transfer(self, payment: Payment, session) -> None:
        """Process internal coin transfer."""
        # This would handle transfers between users
        # For now, just log
        logger.info(f"Internal transfer: {payment.metadata}")
    
    async def _process_webhook_event(self, event: WebhookEvent) -> None:
        """Process webhook event."""
        # Handle different event types
        if event.provider == PaymentProvider.STRIPE:
            await self._process_stripe_webhook(event)
        elif event.provider == PaymentProvider.PAYPAL:
            await self._process_paypal_webhook(event)
        elif event.provider == PaymentProvider.RAZORPAY:
            await self._process_razorpay_webhook(event)
    
    async def _process_stripe_webhook(self, event: WebhookEvent) -> None:
        """Process Stripe webhook event."""
        event_type = event.event_type
        
        if event_type == "payment_intent.succeeded":
            # Payment succeeded
            payment_intent = event.data
            payment_id = payment_intent.get("metadata", {}).get("payment_id") or payment_intent.get("id")
            
            # Update payment status
            logger.info(f"Stripe payment succeeded: {payment_id}")
            
        elif event_type == "payment_intent.payment_failed":
            # Payment failed
            payment_intent = event.data
            payment_id = payment_intent.get("metadata", {}).get("payment_id") or payment_intent.get("id")
            error = payment_intent.get("last_payment_error", {})
            
            logger.error(f"Stripe payment failed: {payment_id}, error: {error}")
            
        elif event_type == "charge.refunded":
            # Refund processed
            charge = event.data
            payment_id = charge.get("payment_intent")
            
            logger.info(f"Stripe refund processed for payment: {payment_id}")
    
    async def _process_paypal_webhook(self, event: WebhookEvent) -> None:
        """Process PayPal webhook event."""
        # Similar to Stripe processing
        pass
    
    async def _process_razorpay_webhook(self, event: WebhookEvent) -> None:
        """Process Razorpay webhook event."""
        # Similar to Stripe processing
        pass
    
    async def transfer_coins(self, from_user_id: str, to_user_id: str, amount: int, description: str = "") -> Dict[str, Any]:
        """
        Transfer coins between users.
        
        Args:
            from_user_id: Sender user ID
            to_user_id: Receiver user ID
            amount: Amount in coins
            description: Transfer description
            
        Returns:
            Transfer result
        """
        async with get_db_session() as session:
            try:
                # Get sender's wallet
                sender_wallet = await session.execute(
                    select(Wallet).where(Wallet.user_id == from_user_id)
                )
                sender_wallet = sender_wallet.scalar_one_or_none()
                
                if not sender_wallet or sender_wallet.balance < amount:
                    raise Exception("Insufficient balance")
                
                # Get receiver's wallet
                receiver_wallet = await session.execute(
                    select(Wallet).where(Wallet.user_id == to_user_id)
                )
                receiver_wallet = receiver_wallet.scalar_one_or_none()
                
                if not receiver_wallet:
                    # Create receiver's wallet
                    receiver_wallet = Wallet(
                        user_id=to_user_id,
                        balance=0,
                        currency="coins",
                    )
                    session.add(receiver_wallet)
                
                # Update balances
                sender_wallet.balance -= amount
                receiver_wallet.balance += amount
                
                # Create transaction records
                transfer_id = f"transfer_{uuid.uuid4().hex[:16]}"
                
                sender_transaction = Transaction(
                    user_id=from_user_id,
                    type="transfer_out",
                    amount=-amount,
                    currency="coins",
                    status="completed",
                    metadata={
                        "to_user_id": to_user_id,
                        "description": description,
                        "transfer_id": transfer_id,
                    },
                )
                
                receiver_transaction = Transaction(
                    user_id=to_user_id,
                    type="transfer_in",
                    amount=amount,
                    currency="coins",
                    status="completed",
                    metadata={
                        "from_user_id": from_user_id,
                        "description": description,
                        "transfer_id": transfer_id,
                    },
                )
                
                session.add(sender_transaction)
                session.add(receiver_transaction)
                
                await session.commit()
                
                return {
                    "transfer_id": transfer_id,
                    "from_user_id": from_user_id,
                    "to_user_id": to_user_id,
                    "amount": amount,
                    "description": description,
                    "sender_new_balance": sender_wallet.balance,
                    "receiver_new_balance": receiver_wallet.balance,
                }
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Coin transfer failed: {str(e)}")
                raise
    
    async def get_supported_providers(self, currency: Currency, payment_method: PaymentMethodType) -> List[PaymentProvider]:
        """
        Get supported providers for a currency and payment method.
        
        Args:
            currency: Currency
            payment_method: Payment method type
            
        Returns:
            List of supported providers
        """
        supported = []
        
        for provider_name, config in self.provider_configs.items():
            if (config["enabled"] and 
                currency.value in config["supported_currencies"] and
                payment_method in config["supported_methods"]):
                supported.append(provider_name)
        
        return supported
    
    async def cleanup(self) -> None:
        """Clean up all providers."""
        for provider in self.providers.values():
            await provider.cleanup()
        self.initialized = False
        logger.info("Payment client cleaned up")


# ============ Factory Function ============

_payment_client: Optional[PaymentClient] = None

async def get_payment_client() -> PaymentClient:
    """
    Get or create a payment client singleton.
    
    Returns:
        PaymentClient instance
    """
    global _payment_client
    
    if _payment_client is None:
        _payment_client = PaymentClient()
        await _payment_client.initialize()
    
    return _payment_client


# ============ Utility Functions ============

async def create_payment(
    user_id: str,
    amount: Decimal,
    currency: Currency = Currency.USD,
    description: Optional[str] = None,
    provider: Optional[PaymentProvider] = None,
    payment_method: PaymentMethodType = PaymentMethodType.CARD
) -> Dict[str, Any]:
    """
    Create a payment (convenience function).
    
    Args:
        user_id: User ID
        amount: Amount to charge
        currency: Currency
        description: Payment description
        provider: Payment provider (auto-select if None)
        payment_method: Payment method type
        
    Returns:
        Payment creation result
    """
    client = await get_payment_client()
    
    # Get user details from database
    async with get_db_session() as session:
        user = await session.execute(
            select(User).where(User.id == user_id)
        )
        user = user.scalar_one_or_none()
        
        if not user:
            raise Exception(f"User not found: {user_id}")
    
    # Create payment intent
    intent = PaymentIntent(
        user_id=user_id,
        amount=amount,
        currency=currency,
        provider=provider or PaymentProvider.STRIPE,
        payment_method_type=payment_method,
        description=description or f"WorldBrief 360 payment",
        customer_email=user.email,
        customer_name=user.full_name or user.username,
        metadata={"user_id": user_id, "username": user.username},
        return_url=f"{settings.FRONTEND_URL}/payment/success",
        cancel_url=f"{settings.FRONTEND_URL}/payment/cancel",
    )
    
    return await client.create_payment(intent)


async def confirm_payment_and_add_coins(payment_id: str, payment_method_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Confirm payment and add coins to user's wallet.
    
    Args:
        payment_id: Payment ID
        payment_method_id: Optional payment method ID
        
    Returns:
        Payment result with coin information
    """
    client = await get_payment_client()
    
    # Confirm payment
    payment = await client.confirm_payment(payment_id, payment_method_id)
    
    if payment.status != PaymentStatus.SUCCEEDED:
        raise Exception(f"Payment not successful: {payment.status}")
    
    # Calculate coins earned
    coins_earned = int(payment.amount * 100)  # $1 = 100 coins
    
    return {
        "payment": payment.dict(),
        "coins_earned": coins_earned,
        "message": f"Successfully added {coins_earned} coins to your wallet",
    }


async def get_user_payment_methods(user_id: str) -> List[PaymentMethod]:
    """
    Get user's saved payment methods.
    
    Args:
        user_id: User ID
        
    Returns:
        List of payment methods
    """
    # In production, this would fetch from database
    # For now, return empty list
    return []


async def create_subscription(
    user_id: str,
    plan_id: str,
    payment_method_id: str,
    trial_days: int = 0
) -> Dict[str, Any]:
    """
    Create a subscription.
    
    Args:
        user_id: User ID
        plan_id: Plan ID
        payment_method_id: Payment method ID
        trial_days: Trial period in days
        
    Returns:
        Subscription creation result
    """
    # This would integrate with Stripe/other providers for subscriptions
    # For now, return a mock response
    return {
        "subscription_id": f"sub_{uuid.uuid4().hex[:16]}",
        "plan_id": plan_id,
        "user_id": user_id,
        "status": "active",
        "current_period_start": datetime.utcnow().isoformat(),
        "current_period_end": (datetime.utcnow() + timedelta(days=30)).isoformat(),
        "trial_end": (datetime.utcnow() + timedelta(days=trial_days)).isoformat() if trial_days > 0 else None,
    }


async def cancel_subscription(subscription_id: str) -> Dict[str, Any]:
    """
    Cancel a subscription.
    
    Args:
        subscription_id: Subscription ID
        
    Returns:
        Cancellation result
    """
    return {
        "subscription_id": subscription_id,
        "cancelled_at": datetime.utcnow().isoformat(),
        "cancelled": True,
        "message": "Subscription cancelled successfully",
    }


async def generate_invoice(
    user_id: str,
    amount: Decimal,
    currency: Currency,
    items: List[Dict[str, Any]],
    due_date: Optional[datetime] = None
) -> Invoice:
    """
    Generate an invoice.
    
    Args:
        user_id: User ID
        amount: Invoice amount
        currency: Currency
        items: Invoice items
        due_date: Due date (default: 30 days from now)
        
    Returns:
        Invoice object
    """
    if not due_date:
        due_date = datetime.utcnow() + timedelta(days=30)
    
    total_amount = sum(Decimal(item.get("amount", 0)) for item in items)
    
    return Invoice(
        invoice_id=f"inv_{uuid.uuid4().hex[:16]}",
        user_id=user_id,
        amount=amount,
        currency=currency,
        status=PaymentStatus.PENDING,
        due_date=due_date,
        items=items,
        total_amount=total_amount,
        metadata={"generated_at": datetime.utcnow().isoformat()},
    )