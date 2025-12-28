"""
subscription.py - Subscription and Billing Model

This module defines models for subscription management, billing, and premium features.
This includes:
- User subscription plans and tiers
- Billing and payment processing
- Subscription lifecycle management
- Premium feature access control
- Usage tracking and limits
- Invoices and receipts
- Coupons and discounts
- Payment methods and billing info

Key Features:
- Multi-tier subscription plans
- Flexible billing cycles (monthly, yearly, etc.)
- Trial periods and freemium model
- Usage-based pricing
- Discounts and promotions
- Payment method management
- Automated billing and dunning
- Subscription analytics
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, TYPE_CHECKING, Union
from enum import Enum
from decimal import Decimal
from sqlalchemy import (
    Column, String, Text, ForeignKey, Integer, DateTime, 
    Boolean, Enum as SQLEnum, JSON, Float, CheckConstraint,
    Index, Table, UniqueConstraint, Numeric, BigInteger
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.sql import func, expression
from sqlalchemy.ext.hybrid import hybrid_property

from db.base import Base
from models.mixins import TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from models.user import User
    from models.organization import Organization


class SubscriptionTier(Enum):
    """Subscription tier levels."""
    FREE = "free"                    # Free tier
    BASIC = "basic"                  # Basic tier
    PROFESSIONAL = "professional"    # Professional tier
    ENTERPRISE = "enterprise"        # Enterprise tier
    CUSTOM = "custom"                # Custom tier


class BillingCycle(Enum):
    """Billing cycle intervals."""
    MONTHLY = "monthly"              # Monthly billing
    QUARTERLY = "quarterly"          # Quarterly billing
    BIANNUALLY = "biannually"        # Every 6 months
    ANNUALLY = "annually"            # Annual billing
    BIENNIALLY = "biennially"        # Every 2 years
    WEEKLY = "weekly"                # Weekly billing
    DAILY = "daily"                  # Daily billing
    ONE_TIME = "one_time"            # One-time payment
    USAGE_BASED = "usage_based"      # Usage-based billing


class SubscriptionStatus(Enum):
    """Subscription status."""
    ACTIVE = "active"                # Active and current
    TRIAL = "trial"                  # In trial period
    PENDING = "pending"              # Pending activation
    PAST_DUE = "past_due"            # Payment past due
    CANCELLED = "cancelled"          # Cancelled
    EXPIRED = "expired"              # Subscription expired
    SUSPENDED = "suspended"          # Suspended
    INCOMPLETE = "incomplete"        # Incomplete setup
    INCOMPLETE_EXPIRED = "incomplete_expired"  # Incomplete expired


class PaymentStatus(Enum):
    """Payment status."""
    PENDING = "pending"              # Payment pending
    PROCESSING = "processing"        # Processing payment
    SUCCEEDED = "succeeded"          # Payment succeeded
    FAILED = "failed"                # Payment failed
    REFUNDED = "refunded"            # Payment refunded
    PARTIALLY_REFUNDED = "partially_refunded"  # Partially refunded
    DISPUTED = "disputed"            # Payment disputed
    CANCELLED = "cancelled"          # Payment cancelled
    REQUIRES_ACTION = "requires_action"  # Requires customer action


class PaymentMethodType(Enum):
    """Payment method types."""
    CREDIT_CARD = "credit_card"      # Credit/debit card
    BANK_TRANSFER = "bank_transfer"  # Bank transfer
    PAYPAL = "paypal"                # PayPal
    STRIPE = "stripe"                # Stripe payment
    APPLE_PAY = "apple_pay"          # Apple Pay
    GOOGLE_PAY = "google_pay"        # Google Pay
    CRYPTO = "crypto"                # Cryptocurrency
    CHECK = "check"                  # Paper check
    CASH = "cash"                    # Cash payment
    OTHER = "other"                  # Other payment method


class InvoiceStatus(Enum):
    """Invoice status."""
    DRAFT = "draft"                  # Draft invoice
    OPEN = "open"                    # Open for payment
    PAID = "paid"                    # Invoice paid
    VOID = "void"                    # Invoice voided
    UNCOLLECTIBLE = "uncollectible"  # Uncollectible
    DELETED = "deleted"              # Invoice deleted


class DiscountType(Enum):
    """Discount types."""
    PERCENTAGE = "percentage"        # Percentage discount
    FIXED_AMOUNT = "fixed_amount"    # Fixed amount discount
    FREE_TRIAL = "free_trial"        # Free trial
    FREE_MONTHS = "free_months"      # Free months
    QUANTITY = "quantity"            # Quantity-based discount
    BUNDLE = "bundle"                # Bundle discount
    PROMOTIONAL = "promotional"      # Promotional discount


class Plan(Base, UUIDMixin, TimestampMixin):
    """
    Subscription plan model.
    
    This model defines subscription plans available for purchase,
    including pricing, features, and limitations.
    
    Attributes:
        id: Primary key UUID
        name: Plan name
        description: Plan description
        tier: Subscription tier
        billing_cycle: Billing cycle
        price_amount: Price amount
        price_currency: Currency code (USD, EUR, etc.)
        is_active: Whether plan is active
        features: Plan features
        limitations: Plan limitations
        max_users: Maximum users allowed
        max_storage_gb: Maximum storage in GB
        max_api_calls: Maximum API calls per month
        support_level: Support level included
        trial_days: Trial period in days
        setup_fee: One-time setup fee
        metadata: Additional metadata
        tags: Categorization tags
    """
    
    __tablename__ = "plans"
    
    # Basic information
    name = Column(String(200), nullable=False, index=True)
    description = Column(Text, nullable=True)
    tier = Column(SQLEnum(SubscriptionTier), nullable=False, index=True)
    billing_cycle = Column(SQLEnum(BillingCycle), nullable=False, index=True)
    
    # Pricing
    price_amount = Column(Numeric(10, 2), nullable=False)
    price_currency = Column(String(3), default="USD", nullable=False)  # ISO 4217
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    
    # Features and limitations
    features = Column(JSONB, default=list, nullable=False)
    limitations = Column(JSONB, default=list, nullable=False)
    
    # Resource limits
    max_users = Column(Integer, nullable=True)
    max_storage_gb = Column(Float, nullable=True)
    max_api_calls = Column(BigInteger, nullable=True)
    max_incidents = Column(Integer, nullable=True)
    max_articles = Column(Integer, nullable=True)
    max_datasets = Column(Integer, nullable=True)
    
    # Support
    support_level = Column(String(50), default="basic", nullable=False)
    support_response_hours = Column(Integer, nullable=True)
    
    # Trial and setup
    trial_days = Column(Integer, default=0, nullable=False)
    setup_fee = Column(Numeric(10, 2), nullable=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    tags = Column(ARRAY(String), default=[], nullable=False, index=True)
    
    # Relationships
    subscriptions = relationship("Subscription", back_populates="plan")
    
    # Check constraints
    __table_args__ = (
        UniqueConstraint('name', 'billing_cycle', name='uq_plan_name_billing'),
        CheckConstraint('price_amount >= 0', name='check_price_non_negative'),
        CheckConstraint('trial_days >= 0', name='check_trial_days_non_negative'),
        CheckConstraint('setup_fee IS NULL OR setup_fee >= 0', name='check_setup_fee_non_negative'),
        CheckConstraint('max_users IS NULL OR max_users >= 0', name='check_max_users_non_negative'),
        CheckConstraint('max_storage_gb IS NULL OR max_storage_gb >= 0', name='check_max_storage_non_negative'),
        CheckConstraint('max_api_calls IS NULL OR max_api_calls >= 0', name='check_max_api_calls_non_negative'),
        Index('ix_plans_tier_active', 'tier', 'is_active'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Plan(id={self.id}, name={self.name}, tier={self.tier.value})>"
    
    @property
    def formatted_price(self) -> str:
        """Get formatted price string."""
        return f"{self.price_currency} {self.price_amount:.2f}"
    
    @property
    def annual_price(self) -> Optional[Decimal]:
        """Get annual equivalent price."""
        if self.billing_cycle == BillingCycle.ANNUALLY:
            return self.price_amount
        elif self.billing_cycle == BillingCycle.MONTHLY:
            return self.price_amount * 12
        elif self.billing_cycle == BillingCycle.QUARTERLY:
            return self.price_amount * 4
        elif self.billing_cycle == BillingCycle.BIANNUALLY:
            return self.price_amount * 2
        elif self.billing_cycle == BillingCycle.WEEKLY:
            return self.price_amount * 52
        elif self.billing_cycle == BillingCycle.DAILY:
            return self.price_amount * 365
        return None
    
    @property
    def monthly_price(self) -> Optional[Decimal]:
        """Get monthly equivalent price."""
        if self.billing_cycle == BillingCycle.MONTHLY:
            return self.price_amount
        elif self.billing_cycle == BillingCycle.ANNUALLY:
            return self.price_amount / 12
        elif self.billing_cycle == BillingCycle.QUARTERLY:
            return self.price_amount / 3
        elif self.billing_cycle == BillingCycle.BIANNUALLY:
            return self.price_amount / 6
        elif self.billing_cycle == BillingCycle.WEEKLY:
            return self.price_amount * 4.33  # Approximate
        elif self.billing_cycle == BillingCycle.DAILY:
            return self.price_amount * 30.44  # Approximate
        return None
    
    @property
    def has_trial(self) -> bool:
        """Check if plan has trial period."""
        return self.trial_days > 0
    
    @property
    def is_free(self) -> bool:
        """Check if plan is free."""
        return self.tier == SubscriptionTier.FREE or self.price_amount == 0
    
    @property
    def feature_list(self) -> List[str]:
        """Get list of feature names."""
        return [f.get('name') for f in self.features if f.get('name')]
    
    def has_feature(self, feature_name: str) -> bool:
        """Check if plan includes specific feature."""
        return any(
            f.get('name') == feature_name and f.get('enabled', True)
            for f in self.features
        )
    
    def get_feature_limit(self, feature_name: str) -> Optional[Any]:
        """Get limit for specific feature."""
        for f in self.limitations:
            if f.get('feature') == feature_name:
                return f.get('limit')
        return None
    
    def add_feature(self, name: str, description: str = "", enabled: bool = True) -> None:
        """Add a feature to the plan."""
        self.features.append({
            "name": name,
            "description": description,
            "enabled": enabled,
            "added_at": datetime.utcnow().isoformat()
        })
    
    def add_limitation(self, feature: str, limit: Any, description: str = "") -> None:
        """Add a limitation to the plan."""
        self.limitations.append({
            "feature": feature,
            "limit": limit,
            "description": description,
            "added_at": datetime.utcnow().isoformat()
        })
    
    def to_dict(self, include_features: bool = True) -> Dict[str, Any]:
        """Convert plan to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "tier": self.tier.value,
            "billing_cycle": self.billing_cycle.value,
            "price_amount": float(self.price_amount),
            "price_currency": self.price_currency,
            "formatted_price": self.formatted_price,
            "annual_price": float(self.annual_price) if self.annual_price else None,
            "monthly_price": float(self.monthly_price) if self.monthly_price else None,
            "is_active": self.is_active,
            "is_free": self.is_free,
            "has_trial": self.has_trial,
            "trial_days": self.trial_days,
            "setup_fee": float(self.setup_fee) if self.setup_fee else None,
            "max_users": self.max_users,
            "max_storage_gb": self.max_storage_gb,
            "max_api_calls": self.max_api_calls,
            "max_incidents": self.max_incidents,
            "max_articles": self.max_articles,
            "max_datasets": self.max_datasets,
            "support_level": self.support_level,
            "support_response_hours": self.support_response_hours,
            "metadata": self.metadata,
            "tags": self.tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class Subscription(Base, UUIDMixin, TimestampMixin):
    """
    User subscription model.
    
    This model tracks user subscriptions to plans, including
    billing cycles, status, and renewal information.
    
    Attributes:
        id: Primary key UUID
        user_id: Subscribed user ID
        organization_id: Subscribed organization ID
        plan_id: Subscription plan ID
        status: Subscription status
        current_period_start: Start of current billing period
        current_period_end: End of current billing period
        trial_start: Trial start date
        trial_end: Trial end date
        canceled_at: When subscription was cancelled
        cancel_at_period_end: Whether to cancel at period end
        quantity: Number of units/subscriptions
        amount: Subscription amount (may differ from plan price)
        currency: Currency code
        discount_id: Applied discount
        metadata: Additional metadata
        payment_method_id: Default payment method
        billing_address: Billing address
        shipping_address: Shipping address
        notes: Admin notes
        tags: Categorization tags
    """
    
    __tablename__ = "subscriptions"
    
    # Subscriber
    user_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=True,
        index=True
    )
    organization_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("organizations.id", ondelete="CASCADE"), 
        nullable=True,
        index=True
    )
    
    # Plan
    plan_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("plans.id", ondelete="RESTRICT"), 
        nullable=False,
        index=True
    )
    
    # Status and dates
    status = Column(SQLEnum(SubscriptionStatus), default=SubscriptionStatus.PENDING, nullable=False, index=True)
    current_period_start = Column(DateTime(timezone=True), nullable=False)
    current_period_end = Column(DateTime(timezone=True), nullable=False)
    
    # Trial
    trial_start = Column(DateTime(timezone=True), nullable=True)
    trial_end = Column(DateTime(timezone=True), nullable=True)
    
    # Cancellation
    canceled_at = Column(DateTime(timezone=True), nullable=True, index=True)
    cancel_at_period_end = Column(Boolean, default=False, nullable=False)
    
    # Pricing
    quantity = Column(Integer, default=1, nullable=False)
    amount = Column(Numeric(10, 2), nullable=False)
    currency = Column(String(3), default="USD", nullable=False)
    
    # Discount
    discount_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("discounts.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Payment and shipping
    payment_method_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("payment_methods.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    billing_address = Column(JSONB, nullable=True)
    shipping_address = Column(JSONB, nullable=True)
    
    # Admin
    notes = Column(Text, nullable=True)
    tags = Column(ARRAY(String), default=[], nullable=False, index=True)
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id])
    organization = relationship("Organization", foreign_keys=[organization_id])
    plan = relationship("Plan", back_populates="subscriptions")
    discount = relationship("Discount", foreign_keys=[discount_id])
    payment_method = relationship("PaymentMethod", foreign_keys=[payment_method_id])
    invoices = relationship("Invoice", back_populates="subscription")
    payments = relationship("Payment", back_populates="subscription")
    usage_records = relationship("UsageRecord", back_populates="subscription")
    
    # Check constraints
    __table_args__ = (
        CheckConstraint(
            'user_id IS NOT NULL OR organization_id IS NOT NULL',
            name='check_subscriber_exists'
        ),
        CheckConstraint(
            'NOT (user_id IS NOT NULL AND organization_id IS NOT NULL)',
            name='check_single_subscriber'
        ),
        CheckConstraint('amount >= 0', name='check_amount_non_negative'),
        CheckConstraint('quantity >= 1', name='check_quantity_minimum'),
        CheckConstraint(
            'current_period_end > current_period_start',
            name='check_period_valid'
        ),
        Index('ix_subscriptions_status_dates', 'status', 'current_period_end'),
        Index('ix_subscriptions_trial_end', 'trial_end'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        subscriber = f"user={self.user_id}" if self.user_id else f"org={self.organization_id}"
        return f"<Subscription(id={self.id}, {subscriber}, plan={self.plan_id}, status={self.status.value})>"
    
    @property
    def subscriber_id(self) -> uuid.UUID:
        """Get subscriber ID (user or organization)."""
        return self.user_id or self.organization_id
    
    @property
    def subscriber_type(self) -> str:
        """Get subscriber type."""
        return "user" if self.user_id else "organization"
    
    @property
    def is_active(self) -> bool:
        """Check if subscription is active."""
        return self.status in [
            SubscriptionStatus.ACTIVE,
            SubscriptionStatus.TRIAL,
            SubscriptionStatus.PENDING
        ]
    
    @property
    def is_trialing(self) -> bool:
        """Check if subscription is in trial period."""
        if not self.trial_end:
            return False
        
        now = datetime.utcnow()
        return self.status == SubscriptionStatus.TRIAL and now <= self.trial_end
    
    @property
    def trial_days_remaining(self) -> Optional[int]:
        """Get days remaining in trial."""
        if not self.is_trialing or not self.trial_end:
            return None
        
        remaining = self.trial_end - datetime.utcnow()
        return max(0, remaining.days)
    
    @property
    def days_until_renewal(self) -> int:
        """Get days until subscription renewal."""
        remaining = self.current_period_end - datetime.utcnow()
        return max(0, remaining.days)
    
    @property
    def period_duration_days(self) -> int:
        """Get billing period duration in days."""
        duration = self.current_period_end - self.current_period_start
        return duration.days
    
    @property
    def is_cancelled(self) -> bool:
        """Check if subscription is cancelled."""
        return self.status == SubscriptionStatus.CANCELLED or self.canceled_at is not None
    
    @property
    def will_cancel_at_period_end(self) -> bool:
        """Check if subscription will cancel at period end."""
        return self.cancel_at_period_end and not self.is_cancelled
    
    @property
    def formatted_amount(self) -> str:
        """Get formatted amount string."""
        return f"{self.currency} {self.amount:.2f}"
    
    @property
    def monthly_amount(self) -> Decimal:
        """Get monthly equivalent amount."""
        if self.plan.billing_cycle == BillingCycle.MONTHLY:
            return self.amount
        elif self.plan.billing_cycle == BillingCycle.ANNUALLY:
            return self.amount / 12
        elif self.plan.billing_cycle == BillingCycle.QUARTERLY:
            return self.amount / 3
        elif self.plan.billing_cycle == BillingCycle.BIANNUALLY:
            return self.amount / 6
        else:
            return self.amount
    
    @property
    def annual_amount(self) -> Decimal:
        """Get annual equivalent amount."""
        if self.plan.billing_cycle == BillingCycle.ANNUALLY:
            return self.amount
        elif self.plan.billing_cycle == BillingCycle.MONTHLY:
            return self.amount * 12
        elif self.plan.billing_cycle == BillingCycle.QUARTERLY:
            return self.amount * 4
        elif self.plan.billing_cycle == BillingCycle.BIANNUALLY:
            return self.amount * 2
        else:
            return self.amount
    
    @property
    def total_paid(self) -> Decimal:
        """Get total amount paid for this subscription."""
        from sqlalchemy import func
        from models import Payment
        
        # This would typically be calculated via query
        # For now, return 0 and handle in service layer
        return Decimal('0')
    
    def has_feature(self, feature_name: str) -> bool:
        """Check if subscription includes specific feature."""
        return self.plan.has_feature(feature_name)
    
    def get_feature_limit(self, feature_name: str) -> Optional[Any]:
        """Get limit for specific feature."""
        return self.plan.get_feature_limit(feature_name)
    
    def start_trial(self, trial_days: int) -> None:
        """Start trial period for subscription."""
        now = datetime.utcnow()
        self.trial_start = now
        self.trial_end = now + timedelta(days=trial_days)
        self.status = SubscriptionStatus.TRIAL
    
    def cancel(self, at_period_end: bool = True) -> None:
        """Cancel subscription."""
        if at_period_end:
            self.cancel_at_period_end = True
        else:
            self.canceled_at = datetime.utcnow()
            self.status = SubscriptionStatus.CANCELLED
            self.current_period_end = datetime.utcnow()
    
    def reactivate(self) -> None:
        """Reactivate cancelled subscription."""
        if self.cancel_at_period_end:
            self.cancel_at_period_end = False
        elif self.is_cancelled:
            self.canceled_at = None
            self.status = SubscriptionStatus.ACTIVE
            # Extend period if needed
            if self.current_period_end < datetime.utcnow():
                self.current_period_start = datetime.utcnow()
                self.current_period_end = datetime.utcnow() + timedelta(days=30)
    
    def renew_period(self) -> None:
        """Renew subscription period."""
        # Calculate next period based on billing cycle
        if self.plan.billing_cycle == BillingCycle.MONTHLY:
            period_days = 30
        elif self.plan.billing_cycle == BillingCycle.ANNUALLY:
            period_days = 365
        elif self.plan.billing_cycle == BillingCycle.QUARTERLY:
            period_days = 90
        elif self.plan.billing_cycle == BillingCycle.BIANNUALLY:
            period_days = 180
        elif self.plan.billing_cycle == BillingCycle.WEEKLY:
            period_days = 7
        elif self.plan.billing_cycle == BillingCycle.DAILY:
            period_days = 1
        else:
            period_days = 30  # Default
        
        self.current_period_start = self.current_period_end
        self.current_period_end = self.current_period_start + timedelta(days=period_days)
    
    def to_dict(self, include_plan: bool = True, include_subscriber: bool = True) -> Dict[str, Any]:
        """Convert subscription to dictionary."""
        result = {
            "id": str(self.id),
            "user_id": str(self.user_id) if self.user_id else None,
            "organization_id": str(self.organization_id) if self.organization_id else None,
            "subscriber_type": self.subscriber_type,
            "subscriber_id": str(self.subscriber_id),
            "plan_id": str(self.plan_id),
            "status": self.status.value,
            "is_active": self.is_active,
            "is_trialing": self.is_trialing,
            "is_cancelled": self.is_cancelled,
            "will_cancel_at_period_end": self.will_cancel_at_period_end,
            "current_period_start": self.current_period_start.isoformat() if self.current_period_start else None,
            "current_period_end": self.current_period_end.isoformat() if self.current_period_end else None,
            "trial_start": self.trial_start.isoformat() if self.trial_start else None,
            "trial_end": self.trial_end.isoformat() if self.trial_end else None,
            "trial_days_remaining": self.trial_days_remaining,
            "days_until_renewal": self.days_until_renewal,
            "period_duration_days": self.period_duration_days,
            "canceled_at": self.canceled_at.isoformat() if self.canceled_at else None,
            "cancel_at_period_end": self.cancel_at_period_end,
            "quantity": self.quantity,
            "amount": float(self.amount),
            "currency": self.currency,
            "formatted_amount": self.formatted_amount,
            "monthly_amount": float(self.monthly_amount),
            "annual_amount": float(self.annual_amount),
            "discount_id": str(self.discount_id) if self.discount_id else None,
            "payment_method_id": str(self.payment_method_id) if self.payment_method_id else None,
            "billing_address": self.billing_address,
            "shipping_address": self.shipping_address,
            "notes": self.notes,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_plan and self.plan:
            result["plan"] = self.plan.to_dict(include_features=False)
        
        if include_subscriber:
            if self.user:
                result["user"] = {
                    "id": str(self.user.id),
                    "username": self.user.username,
                    "email": getattr(self.user, 'email', None)
                }
            elif self.organization:
                result["organization"] = {
                    "id": str(self.organization.id),
                    "name": self.organization.name
                }
        
        return result


class PaymentMethod(Base, UUIDMixin, TimestampMixin):
    """
    Payment method model.
    
    This model stores payment methods for users/organizations,
    such as credit cards, bank accounts, or digital wallets.
    
    Attributes:
        id: Primary key UUID
        user_id: User who owns payment method
        organization_id: Organization that owns payment method
        type: Payment method type
        provider: Payment provider (Stripe, PayPal, etc.)
        provider_id: Provider's ID for this method
        is_default: Whether this is default payment method
        is_active: Whether payment method is active
        card_brand: Credit card brand
        card_last4: Last 4 digits of card
        card_exp_month: Card expiration month
        card_exp_year: Card expiration year
        bank_name: Bank name
        bank_account_last4: Last 4 digits of bank account
        billing_address: Billing address
        metadata: Additional metadata
    """
    
    __tablename__ = "payment_methods"
    
    # Owner
    user_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=True,
        index=True
    )
    organization_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("organizations.id", ondelete="CASCADE"), 
        nullable=True,
        index=True
    )
    
    # Payment method details
    type = Column(SQLEnum(PaymentMethodType), nullable=False, index=True)
    provider = Column(String(50), nullable=False, index=True)
    provider_id = Column(String(200), nullable=True, index=True)
    
    # Status
    is_default = Column(Boolean, default=False, nullable=False, index=True)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    
    # Credit card details (if applicable)
    card_brand = Column(String(50), nullable=True)
    card_last4 = Column(String(4), nullable=True)
    card_exp_month = Column(Integer, nullable=True)
    card_exp_year = Column(Integer, nullable=True)
    
    # Bank account details (if applicable)
    bank_name = Column(String(100), nullable=True)
    bank_account_last4 = Column(String(4), nullable=True)
    bank_routing_number = Column(String(9), nullable=True)
    
    # Digital wallet details
    wallet_type = Column(String(50), nullable=True)
    wallet_id = Column(String(200), nullable=True)
    
    # Billing
    billing_address = Column(JSONB, nullable=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id])
    organization = relationship("Organization", foreign_keys=[organization_id])
    payments = relationship("Payment", back_populates="payment_method")
    subscriptions = relationship("Subscription", back_populates="payment_method")
    
    # Check constraints
    __table_args__ = (
        CheckConstraint(
            'user_id IS NOT NULL OR organization_id IS NOT NULL',
            name='check_payment_method_owner_exists'
        ),
        CheckConstraint(
            'NOT (user_id IS NOT NULL AND organization_id IS NOT NULL)',
            name='check_single_payment_method_owner'
        ),
        CheckConstraint(
            'card_exp_month IS NULL OR (card_exp_month >= 1 AND card_exp_month <= 12)',
            name='check_card_exp_month_range'
        ),
        CheckConstraint(
            'card_exp_year IS NULL OR card_exp_year >= 2020',
            name='check_card_exp_year_minimum'
        ),
        Index('ix_payment_methods_owner_active', 'user_id', 'organization_id', 'is_active'),
        Index('ix_payment_methods_provider', 'provider', 'provider_id'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        owner = f"user={self.user_id}" if self.user_id else f"org={self.organization_id}"
        return f"<PaymentMethod(id={self.id}, {owner}, type={self.type.value})>"
    
    @property
    def owner_id(self) -> uuid.UUID:
        """Get owner ID (user or organization)."""
        return self.user_id or self.organization_id
    
    @property
    def owner_type(self) -> str:
        """Get owner type."""
        return "user" if self.user_id else "organization"
    
    @property
    def masked_number(self) -> Optional[str]:
        """Get masked card/bank number."""
        if self.type == PaymentMethodType.CREDIT_CARD and self.card_last4:
            return f"•••• •••• •••• {self.card_last4}"
        elif self.type == PaymentMethodType.BANK_TRANSFER and self.bank_account_last4:
            return f"•••• {self.bank_account_last4}"
        return None
    
    @property
    def expiration_date(self) -> Optional[str]:
        """Get formatted expiration date."""
        if self.card_exp_month and self.card_exp_year:
            return f"{self.card_exp_month:02d}/{self.card_exp_year}"
        return None
    
    @property
    def is_expired(self) -> bool:
        """Check if payment method is expired."""
        if not self.card_exp_month or not self.card_exp_year:
            return False
        
        now = datetime.utcnow()
        return (self.card_exp_year < now.year) or (
            self.card_exp_year == now.year and self.card_exp_month < now.month
        )
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert payment method to dictionary."""
        result = {
            "id": str(self.id),
            "user_id": str(self.user_id) if self.user_id else None,
            "organization_id": str(self.organization_id) if self.organization_id else None,
            "owner_type": self.owner_type,
            "owner_id": str(self.owner_id),
            "type": self.type.value,
            "provider": self.provider,
            "provider_id": self.provider_id,
            "is_default": self.is_default,
            "is_active": self.is_active,
            "is_expired": self.is_expired,
            "masked_number": self.masked_number,
            "expiration_date": self.expiration_date,
            "card_brand": self.card_brand,
            "card_last4": self.card_last4,
            "bank_name": self.bank_name,
            "bank_account_last4": self.bank_account_last4,
            "wallet_type": self.wallet_type,
            "billing_address": self.billing_address,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_sensitive:
            result["card_exp_month"] = self.card_exp_month
            result["card_exp_year"] = self.card_exp_year
            result["bank_routing_number"] = self.bank_routing_number
            result["wallet_id"] = self.wallet_id
        
        return result


class Invoice(Base, UUIDMixin, TimestampMixin):
    """
    Invoice model.
    
    This model stores invoices for subscriptions, payments, and other charges.
    
    Attributes:
        id: Primary key UUID
        subscription_id: Related subscription
        number: Invoice number
        status: Invoice status
        amount_due: Amount due
        amount_paid: Amount paid
        amount_remaining: Amount remaining
        currency: Currency code
        invoice_date: Invoice date
        due_date: Due date
        paid_at: When invoice was paid
        billing_reason: Reason for invoice
        lines: Invoice line items
        metadata: Additional metadata
        pdf_url: URL to PDF invoice
    """
    
    __tablename__ = "invoices"
    
    # Subscription
    subscription_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("subscriptions.id", ondelete="CASCADE"), 
        nullable=True,
        index=True
    )
    
    # Invoice details
    number = Column(String(100), nullable=False, unique=True, index=True)
    status = Column(SQLEnum(InvoiceStatus), default=InvoiceStatus.DRAFT, nullable=False, index=True)
    
    # Amounts
    amount_due = Column(Numeric(10, 2), nullable=False)
    amount_paid = Column(Numeric(10, 2), default=0, nullable=False)
    amount_remaining = Column(Numeric(10, 2), nullable=False)
    currency = Column(String(3), default="USD", nullable=False)
    
    # Dates
    invoice_date = Column(DateTime(timezone=True), nullable=False, index=True)
    due_date = Column(DateTime(timezone=True), nullable=True, index=True)
    paid_at = Column(DateTime(timezone=True), nullable=True)
    
    # Billing
    billing_reason = Column(String(100), nullable=True)
    lines = Column(JSONB, default=list, nullable=False)
    
    # Metadata and files
    metadata = Column(JSONB, default=dict, nullable=False)
    pdf_url = Column(String(2000), nullable=True)
    hosted_invoice_url = Column(String(2000), nullable=True)
    
    # Relationships
    subscription = relationship("Subscription", back_populates="invoices")
    payments = relationship("Payment", back_populates="invoice")
    
    # Check constraints
    __table_args__ = (
        CheckConstraint('amount_due >= 0', name='check_amount_due_non_negative'),
        CheckConstraint('amount_paid >= 0', name='check_amount_paid_non_negative'),
        CheckConstraint('amount_remaining >= 0', name='check_amount_remaining_non_negative'),
        CheckConstraint('amount_due = amount_paid + amount_remaining', name='check_amount_consistency'),
        Index('ix_invoices_status_dates', 'status', 'invoice_date', 'due_date'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Invoice(id={self.id}, number={self.number}, status={self.status.value}, amount={self.amount_due})>"
    
    @property
    def is_paid(self) -> bool:
        """Check if invoice is paid."""
        return self.status == InvoiceStatus.PAID or self.amount_remaining == 0
    
    @property
    def is_overdue(self) -> bool:
        """Check if invoice is overdue."""
        if not self.due_date or self.is_paid:
            return False
        return datetime.utcnow() > self.due_date
    
    @property
    def days_overdue(self) -> Optional[int]:
        """Get days overdue."""
        if not self.is_overdue:
            return None
        overdue = datetime.utcnow() - self.due_date
        return overdue.days
    
    @property
    def formatted_amount_due(self) -> str:
        """Get formatted amount due."""
        return f"{self.currency} {self.amount_due:.2f}"
    
    @property
    def line_items(self) -> List[Dict[str, Any]]:
        """Get line items."""
        return self.lines or []
    
    @property
    def subtotal(self) -> Decimal:
        """Get subtotal from line items."""
        subtotal = Decimal('0')
        for line in self.line_items:
            amount = Decimal(str(line.get('amount', 0)))
            subtotal += amount
        return subtotal
    
    @property
    def tax_amount(self) -> Decimal:
        """Get tax amount from line items."""
        tax = Decimal('0')
        for line in self.line_items:
            if line.get('type') == 'tax':
                tax += Decimal(str(line.get('amount', 0)))
        return tax
    
    def add_line_item(
        self,
        description: str,
        amount: Decimal,
        quantity: int = 1,
        unit_price: Optional[Decimal] = None,
        tax_rate: Optional[Decimal] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a line item to the invoice."""
        line_item = {
            "description": description,
            "amount": float(amount),
            "quantity": quantity,
            "unit_price": float(unit_price) if unit_price else float(amount / quantity),
            "tax_rate": float(tax_rate) if tax_rate else None,
            "metadata": metadata or {},
            "added_at": datetime.utcnow().isoformat()
        }
        
        self.lines.append(line_item)
        self._recalculate_totals()
    
    def _recalculate_totals(self) -> None:
        """Recalculate invoice totals from line items."""
        subtotal = Decimal('0')
        for line in self.lines:
            amount = Decimal(str(line.get('amount', 0)))
            subtotal += amount
        
        # For now, just set amount_due to subtotal
        # In production, you'd add taxes, discounts, etc.
        self.amount_due = subtotal
        self.amount_remaining = subtotal - self.amount_paid
    
    def mark_as_paid(self, paid_at: Optional[datetime] = None) -> None:
        """Mark invoice as paid."""
        self.status = InvoiceStatus.PAID
        self.amount_paid = self.amount_due
        self.amount_remaining = Decimal('0')
        self.paid_at = paid_at or datetime.utcnow()
    
    def to_dict(self, include_lines: bool = True) -> Dict[str, Any]:
        """Convert invoice to dictionary."""
        return {
            "id": str(self.id),
            "subscription_id": str(self.subscription_id) if self.subscription_id else None,
            "number": self.number,
            "status": self.status.value,
            "is_paid": self.is_paid,
            "is_overdue": self.is_overdue,
            "days_overdue": self.days_overdue,
            "amount_due": float(self.amount_due),
            "amount_paid": float(self.amount_paid),
            "amount_remaining": float(self.amount_remaining),
            "formatted_amount_due": self.formatted_amount_due,
            "currency": self.currency,
            "invoice_date": self.invoice_date.isoformat() if self.invoice_date else None,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "paid_at": self.paid_at.isoformat() if self.paid_at else None,
            "billing_reason": self.billing_reason,
            "subtotal": float(self.subtotal),
            "tax_amount": float(self.tax_amount),
            "pdf_url": self.pdf_url,
            "hosted_invoice_url": self.hosted_invoice_url,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class Payment(Base, UUIDMixin, TimestampMixin):
    """
    Payment model.
    
    This model stores individual payment transactions.
    
    Attributes:
        id: Primary key UUID
        subscription_id: Related subscription
        invoice_id: Related invoice
        payment_method_id: Payment method used
        amount: Payment amount
        currency: Currency code
        status: Payment status
        provider: Payment provider
        provider_id: Provider's transaction ID
        provider_fee: Fee charged by provider
        tax_amount: Tax amount
        description: Payment description
        metadata: Additional metadata
        receipt_url: URL to receipt
        refunded_amount: Amount refunded
        refund_reason: Reason for refund
    """
    
    __tablename__ = "payments"
    
    # References
    subscription_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("subscriptions.id", ondelete="CASCADE"), 
        nullable=True,
        index=True
    )
    invoice_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("invoices.id", ondelete="CASCADE"), 
        nullable=True,
        index=True
    )
    payment_method_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("payment_methods.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    
    # Payment details
    amount = Column(Numeric(10, 2), nullable=False)
    currency = Column(String(3), default="USD", nullable=False)
    status = Column(SQLEnum(PaymentStatus), default=PaymentStatus.PENDING, nullable=False, index=True)
    
    # Provider details
    provider = Column(String(50), nullable=False, index=True)
    provider_id = Column(String(200), nullable=True, index=True)
    provider_fee = Column(Numeric(10, 2), nullable=True)
    
    # Tax
    tax_amount = Column(Numeric(10, 2), nullable=True)
    
    # Description
    description = Column(Text, nullable=True)
    
    # Metadata and receipts
    metadata = Column(JSONB, default=dict, nullable=False)
    receipt_url = Column(String(2000), nullable=True)
    
    # Refunds
    refunded_amount = Column(Numeric(10, 2), default=0, nullable=False)
    refund_reason = Column(Text, nullable=True)
    
    # Relationships
    subscription = relationship("Subscription", back_populates="payments")
    invoice = relationship("Invoice", back_populates="payments")
    payment_method = relationship("PaymentMethod", back_populates="payments")
    
    # Check constraints
    __table_args__ = (
        CheckConstraint('amount > 0', name='check_amount_positive'),
        CheckConstraint('provider_fee IS NULL OR provider_fee >= 0', name='check_provider_fee_non_negative'),
        CheckConstraint('tax_amount IS NULL OR tax_amount >= 0', name='check_tax_amount_non_negative'),
        CheckConstraint('refunded_amount >= 0', name='check_refunded_amount_non_negative'),
        CheckConstraint('refunded_amount <= amount', name='check_refunded_amount_not_exceed'),
        Index('ix_payments_status_dates', 'status', 'created_at'),
        Index('ix_payments_provider', 'provider', 'provider_id'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Payment(id={self.id}, amount={self.amount}, status={self.status.value})>"
    
    @property
    def net_amount(self) -> Decimal:
        """Get net amount after fees."""
        net = self.amount
        if self.provider_fee:
            net -= self.provider_fee
        if self.tax_amount:
            net -= self.tax_amount
        return net
    
    @property
    def is_successful(self) -> bool:
        """Check if payment was successful."""
        return self.status == PaymentStatus.SUCCEEDED
    
    @property
    def is_refunded(self) -> bool:
        """Check if payment is refunded."""
        return self.status == PaymentStatus.REFUNDED or self.refunded_amount > 0
    
    @property
    def is_partially_refunded(self) -> bool:
        """Check if payment is partially refunded."""
        return self.status == PaymentStatus.PARTIALLY_REFUNDED or (
            self.refunded_amount > 0 and self.refunded_amount < self.amount
        )
    
    @property
    def formatted_amount(self) -> str:
        """Get formatted amount."""
        return f"{self.currency} {self.amount:.2f}"
    
    @property
    def refund_percentage(self) -> float:
        """Get percentage refunded."""
        if self.amount == 0:
            return 0.0
        return (self.refunded_amount / self.amount) * 100
    
    def mark_as_successful(self, provider_id: Optional[str] = None) -> None:
        """Mark payment as successful."""
        self.status = PaymentStatus.SUCCEEDED
        if provider_id:
            self.provider_id = provider_id
    
    def refund(self, amount: Optional[Decimal] = None, reason: Optional[str] = None) -> None:
        """Refund payment."""
        refund_amount = amount or self.amount
        
        if refund_amount > self.amount - self.refunded_amount:
            raise ValueError("Refund amount exceeds available amount")
        
        self.refunded_amount += refund_amount
        
        if self.refunded_amount == self.amount:
            self.status = PaymentStatus.REFUNDED
        else:
            self.status = PaymentStatus.PARTIALLY_REFUNDED
        
        if reason:
            self.refund_reason = reason
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert payment to dictionary."""
        return {
            "id": str(self.id),
            "subscription_id": str(self.subscription_id) if self.subscription_id else None,
            "invoice_id": str(self.invoice_id) if self.invoice_id else None,
            "payment_method_id": str(self.payment_method_id) if self.payment_method_id else None,
            "amount": float(self.amount),
            "currency": self.currency,
            "formatted_amount": self.formatted_amount,
            "net_amount": float(self.net_amount),
            "status": self.status.value,
            "is_successful": self.is_successful,
            "is_refunded": self.is_refunded,
            "is_partially_refunded": self.is_partially_refunded,
            "provider": self.provider,
            "provider_id": self.provider_id,
            "provider_fee": float(self.provider_fee) if self.provider_fee else None,
            "tax_amount": float(self.tax_amount) if self.tax_amount else None,
            "description": self.description,
            "refunded_amount": float(self.refunded_amount),
            "refund_percentage": self.refund_percentage,
            "refund_reason": self.refund_reason,
            "receipt_url": self.receipt_url,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class Discount(Base, UUIDMixin, TimestampMixin):
    """
    Discount and coupon model.
    
    This model stores discounts, coupons, and promotional codes.
    
    Attributes:
        id: Primary key UUID
        code: Discount code
        name: Discount name
        description: Discount description
        type: Discount type
        value: Discount value
        currency: Currency (for fixed amount)
        is_active: Whether discount is active
        max_redemptions: Maximum number of redemptions
        redemption_count: Number of times redeemed
        applies_to: What the discount applies to
        min_amount: Minimum amount required
        max_amount: Maximum discount amount
        start_date: When discount becomes active
        end_date: When discount expires
        metadata: Additional metadata
    """
    
    __tablename__ = "discounts"
    
    # Code and info
    code = Column(String(100), nullable=False, unique=True, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    
    # Discount details
    type = Column(SQLEnum(DiscountType), nullable=False, index=True)
    value = Column(Numeric(10, 2), nullable=False)
    currency = Column(String(3), default="USD", nullable=True)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    
    # Redemption limits
    max_redemptions = Column(Integer, nullable=True)
    redemption_count = Column(Integer, default=0, nullable=False)
    
    # Application rules
    applies_to = Column(JSONB, default=list, nullable=False)
    min_amount = Column(Numeric(10, 2), nullable=True)
    max_amount = Column(Numeric(10, 2), nullable=True)
    
    # Validity period
    start_date = Column(DateTime(timezone=True), nullable=True, index=True)
    end_date = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    subscriptions = relationship("Subscription", back_populates="discount")
    
    # Check constraints
    __table_args__ = (
        CheckConstraint('value >= 0', name='check_value_non_negative'),
        CheckConstraint(
            'type != \'percentage\' OR (value >= 0 AND value <= 100)',
            name='check_percentage_range'
        ),
        CheckConstraint('redemption_count >= 0', name='check_redemption_count_non_negative'),
        CheckConstraint('max_redemptions IS NULL OR max_redemptions >= 1', name='check_max_redemptions_valid'),
        CheckConstraint(
            'max_redemptions IS NULL OR redemption_count <= max_redemptions',
            name='check_redemption_count_limit'
        ),
        Index('ix_discounts_active_dates', 'is_active', 'start_date', 'end_date'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Discount(id={self.id}, code={self.code}, type={self.type.value}, value={self.value})>"
    
    @property
    def formatted_value(self) -> str:
        """Get formatted discount value."""
        if self.type == DiscountType.PERCENTAGE:
            return f"{self.value}%"
        elif self.type == DiscountType.FIXED_AMOUNT:
            return f"{self.currency} {self.value:.2f}"
        elif self.type == DiscountType.FREE_TRIAL:
            return f"{int(self.value)} days free trial"
        elif self.type == DiscountType.FREE_MONTHS:
            return f"{int(self.value)} free months"
        else:
            return str(self.value)
    
    @property
    def is_valid(self) -> bool:
        """Check if discount is currently valid."""
        if not self.is_active:
            return False
        
        if self.max_redemptions and self.redemption_count >= self.max_redemptions:
            return False
        
        now = datetime.utcnow()
        if self.start_date and now < self.start_date:
            return False
        
        if self.end_date and now > self.end_date:
            return False
        
        return True
    
    @property
    def days_remaining(self) -> Optional[int]:
        """Get days remaining until expiration."""
        if not self.end_date:
            return None
        
        remaining = self.end_date - datetime.utcnow()
        return max(0, remaining.days)
    
    @property
    def redemption_percentage(self) -> Optional[float]:
        """Get redemption percentage if max_redemptions is set."""
        if not self.max_redemptions:
            return None
        return (self.redemption_count / self.max_redemptions) * 100
    
    def calculate_discount(self, amount: Decimal) -> Decimal:
        """Calculate discount amount for given price."""
        if not self.is_valid:
            return Decimal('0')
        
        if self.type == DiscountType.PERCENTAGE:
            discount = amount * (self.value / 100)
        elif self.type == DiscountType.FIXED_AMOUNT:
            discount = Decimal(str(self.value))
        else:
            discount = Decimal('0')
        
        # Apply max amount limit
        if self.max_amount and discount > self.max_amount:
            discount = self.max_amount
        
        # Ensure discount doesn't exceed amount
        if discount > amount:
            discount = amount
        
        return discount
    
    def redeem(self) -> bool:
        """Redeem the discount code."""
        if not self.is_valid:
            return False
        
        self.redemption_count += 1
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert discount to dictionary."""
        return {
            "id": str(self.id),
            "code": self.code,
            "name": self.name,
            "description": self.description,
            "type": self.type.value,
            "value": float(self.value),
            "formatted_value": self.formatted_value,
            "currency": self.currency,
            "is_active": self.is_active,
            "is_valid": self.is_valid,
            "max_redemptions": self.max_redemptions,
            "redemption_count": self.redemption_count,
            "redemption_percentage": self.redemption_percentage,
            "applies_to": self.applies_to,
            "min_amount": float(self.min_amount) if self.min_amount else None,
            "max_amount": float(self.max_amount) if self.max_amount else None,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "days_remaining": self.days_remaining,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class UsageRecord(Base, UUIDMixin, TimestampMixin):
    """
    Usage record model for usage-based billing.
    
    This model tracks usage of metered features for usage-based pricing.
    
    Attributes:
        id: Primary key UUID
        subscription_id: Related subscription
        feature_name: Name of metered feature
        quantity: Quantity used
        unit: Unit of measurement
        timestamp: When usage occurred
        period_start: Start of billing period
        period_end: End of billing period
        metadata: Additional metadata
    """
    
    __tablename__ = "usage_records"
    
    # Subscription
    subscription_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("subscriptions.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # Usage details
    feature_name = Column(String(200), nullable=False, index=True)
    quantity = Column(Numeric(10, 4), nullable=False)
    unit = Column(String(50), nullable=True)
    
    # Timing
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    period_start = Column(DateTime(timezone=True), nullable=True, index=True)
    period_end = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    subscription = relationship("Subscription", back_populates="usage_records")
    
    # Check constraints
    __table_args__ = (
        CheckConstraint('quantity >= 0', name='check_quantity_non_negative'),
        CheckConstraint(
            'period_end IS NULL OR period_end > period_start',
            name='check_period_valid'
        ),
        Index('ix_usage_subscription_feature', 'subscription_id', 'feature_name'),
        Index('ix_usage_timestamp_period', 'timestamp', 'period_start'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<UsageRecord(id={self.id}, subscription={self.subscription_id}, feature={self.feature_name}, quantity={self.quantity})>"
    
    @property
    def formatted_quantity(self) -> str:
        """Get formatted quantity."""
        if self.unit:
            return f"{self.quantity} {self.unit}"
        return str(self.quantity)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert usage record to dictionary."""
        return {
            "id": str(self.id),
            "subscription_id": str(self.subscription_id),
            "feature_name": self.feature_name,
            "quantity": float(self.quantity),
            "formatted_quantity": self.formatted_quantity,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class BillingAlert(Base, UUIDMixin, TimestampMixin):
    """
    Billing alert model.
    
    This model stores billing alerts and notifications for users.
    
    Attributes:
        id: Primary key UUID
        user_id: User to alert
        organization_id: Organization to alert
        alert_type: Type of alert
        threshold: Threshold value
        currency: Currency for amount thresholds
        is_active: Whether alert is active
        last_triggered_at: When alert was last triggered
        trigger_count: Number of times triggered
        metadata: Additional metadata
    """
    
    __tablename__ = "billing_alerts"
    
    # Recipient
    user_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=True,
        index=True
    )
    organization_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("organizations.id", ondelete="CASCADE"), 
        nullable=True,
        index=True
    )
    
    # Alert details
    alert_type = Column(String(100), nullable=False, index=True)
    threshold = Column(Numeric(10, 2), nullable=True)
    currency = Column(String(3), nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    last_triggered_at = Column(DateTime(timezone=True), nullable=True)
    trigger_count = Column(Integer, default=0, nullable=False)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Check constraints
    __table_args__ = (
        CheckConstraint(
            'user_id IS NOT NULL OR organization_id IS NOT NULL',
            name='check_alert_recipient_exists'
        ),
        CheckConstraint(
            'NOT (user_id IS NOT NULL AND organization_id IS NOT NULL)',
            name='check_single_alert_recipient'
        ),
        CheckConstraint('trigger_count >= 0', name='check_trigger_count_non_negative'),
        Index('ix_billing_alerts_type_active', 'alert_type', 'is_active'),
    )
    
    def trigger(self) -> None:
        """Trigger the alert."""
        self.last_triggered_at = datetime.utcnow()
        self.trigger_count += 1


# Helper functions
def generate_invoice_number() -> str:
    """Generate unique invoice number."""
    from datetime import datetime
    import random
    
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    random_suffix = str(random.randint(1000, 9999))
    return f"INV-{timestamp}-{random_suffix}"


def calculate_prorated_amount(
    plan_amount: Decimal,
    billing_cycle: BillingCycle,
    days_used: int,
    days_in_period: int
) -> Decimal:
    """
    Calculate prorated amount for subscription changes.
    
    Args:
        plan_amount: Plan amount for full period
        billing_cycle: Billing cycle
        days_used: Days already used in current period
        days_in_period: Total days in billing period
        
    Returns:
        Prorated amount
    """
    if days_in_period == 0:
        return Decimal('0')
    
    daily_rate = plan_amount / Decimal(days_in_period)
    return daily_rate * Decimal(days_used)


def validate_payment_card(
    card_number: str,
    exp_month: int,
    exp_year: int,
    cvc: str
) -> Dict[str, Any]:
    """
    Validate payment card details.
    
    Note: This is basic validation. In production, use a payment processor API.
    
    Args:
        card_number: Card number
        exp_month: Expiration month
        exp_year: Expiration year
        cvc: CVC/CVV code
        
    Returns:
        Validation result
    """
    errors = []
    
    # Validate card number (basic Luhn check)
    def luhn_check(card_num: str) -> bool:
        def digits_of(n):
            return [int(d) for d in str(n)]
        digits = digits_of(card_num)
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        checksum = sum(odd_digits)
        for d in even_digits:
            checksum += sum(digits_of(d * 2))
        return checksum % 10 == 0
    
    card_number_clean = card_number.replace(" ", "").replace("-", "")
    if not card_number_clean.isdigit() or len(card_number_clean) < 13:
        errors.append("Invalid card number")
    elif not luhn_check(card_number_clean):
        errors.append("Invalid card number (failed Luhn check)")
    
    # Validate expiration
    now = datetime.utcnow()
    if exp_month < 1 or exp_month > 12:
        errors.append("Invalid expiration month")
    if exp_year < now.year:
        errors.append("Card has expired")
    elif exp_year == now.year and exp_month < now.month:
        errors.append("Card has expired")
    
    # Validate CVC
    if not cvc.isdigit() or len(cvc) not in [3, 4]:
        errors.append("Invalid CVC/CVV")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "card_last4": card_number_clean[-4:] if card_number_clean else None
    }