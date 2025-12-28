"""
Withdrawal model for managing cryptocurrency and fiat withdrawal requests.
Handles withdrawal requests, approvals, processing, and tracking for user funds.
"""

from sqlalchemy import Column, Integer, String, DateTime, Numeric, Enum, ForeignKey, Boolean, Text, JSON, Index,Tuple
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship, validates
from datetime import datetime,timedelta
import enum
import json
import uuid
from decimal import Decimal
from typing import Any, Dict, Optional

from app.db.base import Base
from app.core.exceptions import ValidationError


class WithdrawalStatus(str, enum.Enum):
    """
    Status of a withdrawal request throughout its lifecycle.
    """
    DRAFT = "draft"                    # Draft (user hasn't submitted yet)
    PENDING = "pending"                # Submitted, waiting for processing
    UNDER_REVIEW = "under_review"      # Being reviewed by admin/automated system
    APPROVED = "approved"              # Approved for processing
    REJECTED = "rejected"              # Rejected (not approved)
    PROCESSING = "processing"          # Being processed by payment processor
    QUEUED = "queued"                  # Queued for blockchain transaction
    SENT = "sent"                      # Funds sent to external address
    CONFIRMING = "confirming"          # Waiting for blockchain confirmations
    COMPLETED = "completed"            # Successfully completed
    FAILED = "failed"                  # Failed during processing
    CANCELLED = "cancelled"            # Cancelled by user or system
    REFUNDED = "refunded"              # Funds refunded to user wallet
    EXPIRED = "expired"                # Expired before processing
    SUSPENDED = "suspended"            # Suspended for investigation
    ON_HOLD = "on_hold"                # On hold (requires manual intervention)


class WithdrawalType(str, enum.Enum):
    """
    Types of withdrawals supported.
    """
    CRYPTO = "crypto"                  # Cryptocurrency withdrawal
    FIAT = "fiat"                      # Fiat currency withdrawal (bank transfer)
    INTERNAL = "internal"              # Internal transfer to another user
    CHARITY = "charity"                # Donation/charity withdrawal
    VOUCHER = "voucher"                # Convert to voucher/gift card
    PRODUCT = "product"                # Convert to product purchase
    SERVICE = "service"                # Pay for service


class NetworkType(str, enum.Enum):
    """
    Blockchain networks supported for cryptocurrency withdrawals.
    """
    # Ethereum and EVM-compatible
    ETHEREUM = "ethereum"              # Ethereum Mainnet
    ARBITRUM = "arbitrum"              # Arbitrum
    OPTIMISM = "optimism"              # Optimism
    POLYGON = "polygon"                # Polygon (Matic)
    BINANCE_SMART_CHAIN = "bsc"        # Binance Smart Chain
    AVALANCHE = "avalanche"            # Avalanche C-Chain
    FANTOM = "fantom"                  # Fantom Opera
    
    # Bitcoin and derivatives
    BITCOIN = "bitcoin"                # Bitcoin Mainnet
    BITCOIN_TESTNET = "bitcoin_testnet" # Bitcoin Testnet
    LITECOIN = "litecoin"              # Litecoin
    BITCOIN_CASH = "bitcoin_cash"      # Bitcoin Cash
    
    # Other networks
    SOLANA = "solana"                  # Solana
    CARDANO = "cardano"                # Cardano
    POLKADOT = "polkadot"              # Polkadot
    COSMOS = "cosmos"                  # Cosmos
    TRON = "tron"                      # Tron
    
    # Testnets
    GOERLI = "goerli"                  # Ethereum Goerli Testnet
    SEPOLIA = "sepolia"                # Ethereum Sepolia Testnet
    MUMBAI = "mumbai"                  # Polygon Mumbai Testnet
    
    # Internal
    INTERNAL = "internal"              # Internal network (for internal transfers)


class FiatCurrency(str, enum.Enum):
    """
    Fiat currencies supported for withdrawals.
    """
    USD = "usd"                        # US Dollar
    EUR = "eur"                        # Euro
    GBP = "gbp"                        # British Pound
    JPY = "jpy"                        # Japanese Yen
    CAD = "cad"                        # Canadian Dollar
    AUD = "aud"                        # Australian Dollar
    CHF = "chf"                        # Swiss Franc
    CNY = "cny"                        # Chinese Yuan
    INR = "inr"                        # Indian Rupee
    BRL = "brl"                        # Brazilian Real
    RUB = "rub"                        # Russian Ruble
    KRW = "krw"                        # South Korean Won
    SGD = "sgd"                        # Singapore Dollar
    HKD = "hkd"                        # Hong Kong Dollar


class BankAccountType(str, enum.Enum):
    """
    Types of bank accounts for fiat withdrawals.
    """
    CHECKING = "checking"              # Checking account
    SAVINGS = "savings"                # Savings account
    CURRENT = "current"                # Current account
    SALARY = "salary"                  # Salary account
    BUSINESS = "business"              # Business account


class Withdrawal(Base):
    """
    Main withdrawal model for tracking all withdrawal requests.
    
    Each withdrawal represents a user's request to withdraw funds from their wallet
    to an external address (crypto) or bank account (fiat).
    """
    
    __tablename__ = "withdrawals"
    
    # Primary key and identification
    id = Column(Integer, primary_key=True, index=True)
    withdrawal_id = Column(String(100), unique=True, index=True, nullable=False, 
                          default=lambda: str(uuid.uuid4()), comment="Public UUID for external reference")
    external_reference = Column(String(100), nullable=True, unique=True, index=True, 
                               comment="External reference ID from payment processor")
    
    # User information
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    wallet_id = Column(Integer, ForeignKey("wallets.id", ondelete="SET NULL"), nullable=True, index=True)
    
    # Withdrawal details
    withdrawal_type = Column(Enum(WithdrawalType), nullable=False, index=True)
    status = Column(Enum(WithdrawalStatus), default=WithdrawalStatus.PENDING, nullable=False, index=True)
    
    # Amount information
    requested_amount = Column(Numeric(20, 8), nullable=False, comment="Amount requested by user")
    approved_amount = Column(Numeric(20, 8), nullable=True, comment="Amount approved by admin")
    net_amount = Column(Numeric(20, 8), nullable=True, comment="Amount after fees")
    fees = Column(JSONB, nullable=True, comment="JSON breakdown of fees")
    
    # Cryptocurrency details (for crypto withdrawals)
    crypto_currency = Column(String(10), nullable=True, comment="Cryptocurrency symbol (ETH, BTC, etc.)")
    network = Column(Enum(NetworkType), nullable=True, comment="Blockchain network")
    wallet_address = Column(String(255), nullable=True, comment="Destination wallet address")
    address_label = Column(String(255), nullable=True, comment="Label/name for the address")
    
    # Fiat details (for fiat withdrawals)
    fiat_currency = Column(Enum(FiatCurrency), nullable=True)
    bank_account_details = Column(JSONB, nullable=True, comment="JSON with bank account details")
    payment_method = Column(String(50), nullable=True, comment="Payment method (wire, sepa, swift, etc.)")
    
    # Transaction information
    transaction_hash = Column(String(255), nullable=True, unique=True, index=True, 
                             comment="Blockchain transaction hash")
    transaction_url = Column(String(500), nullable=True, comment="URL to view transaction on explorer")
    confirmation_count = Column(Integer, default=0, comment="Number of blockchain confirmations")
    required_confirmations = Column(Integer, default=3, comment="Required confirmations for completion")
    
    # Processing information
    processor_id = Column(String(100), nullable=True, index=True, comment="Payment processor ID")
    processor_response = Column(JSONB, nullable=True, comment="Raw response from payment processor")
    processor_fee = Column(Numeric(20, 8), nullable=True, comment="Fee charged by processor")
    
    # Timeline
    requested_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    reviewed_at = Column(DateTime, nullable=True, index=True)
    approved_at = Column(DateTime, nullable=True, index=True)
    processed_at = Column(DateTime, nullable=True, index=True)
    sent_at = Column(DateTime, nullable=True, index=True)
    completed_at = Column(DateTime, nullable=True, index=True)
    failed_at = Column(DateTime, nullable=True, index=True)
    cancelled_at = Column(DateTime, nullable=True, index=True)
    expires_at = Column(DateTime, nullable=True, index=True, comment="When withdrawal expires if not processed")
    
    # Review and approval
    reviewed_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    approved_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    processed_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    
    # Rejection information
    rejection_reason = Column(Text, nullable=True, comment="Reason for rejection")
    rejection_code = Column(String(50), nullable=True, comment="Standard rejection code")
    
    # Failure information
    failure_reason = Column(Text, nullable=True, comment="Reason for failure")
    failure_code = Column(String(50), nullable=True, comment="Standard failure code")
    can_retry = Column(Boolean, default=True, comment="Whether withdrawal can be retried")
    retry_count = Column(Integer, default=0, comment="Number of retry attempts")
    max_retries = Column(Integer, default=3, comment="Maximum retry attempts")
    
    # Security and verification
    verification_hash = Column(String(255), nullable=True, unique=True, index=True, 
                              comment="Hash for withdrawal verification")
    two_factor_verified = Column(Boolean, default=False, comment="Whether 2FA was verified")
    ip_address = Column(String(45), nullable=True, comment="IP address of request")
    user_agent = Column(Text, nullable=True, comment="User agent of request")
    
    # Metadata
    description = Column(Text, nullable=True, comment="User-provided description")
    notes = Column(Text, nullable=True, comment="Internal notes")
    metadata = Column(JSONB, nullable=True, comment="Additional metadata")
    tags = Column(JSONB, nullable=True, comment="Tags for categorization")
    
    # Audit trail
    audit_trail = Column(JSONB, nullable=True, comment="JSON audit trail of status changes")
    
    # Compliance
    compliance_check_id = Column(String(100), nullable=True, comment="ID from compliance check")
    compliance_status = Column(String(50), nullable=True, comment="Compliance check status")
    requires_manual_review = Column(Boolean, default=False, comment="Whether requires manual review")
    
    # Risk scoring
    risk_score = Column(Integer, nullable=True, comment="Risk score (0-100)")
    risk_level = Column(String(20), nullable=True, comment="Risk level (low, medium, high)")
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id], backref="withdrawals")
    wallet = relationship("Wallet", foreign_keys=[wallet_id])
    reviewer = relationship("User", foreign_keys=[reviewed_by])
    approver = relationship("User", foreign_keys=[approved_by])
    processor = relationship("User", foreign_keys=[processed_by])
    
    # Indexes
    __table_args__ = (
        Index('ix_withdrawals_user_status', 'user_id', 'status'),
        Index('ix_withdrawals_type_status', 'withdrawal_type', 'status'),
        Index('ix_withdrawals_network_status', 'network', 'status'),
        Index('ix_withdrawals_requested_status', 'requested_at', 'status'),
        Index('ix_withdrawals_amount_status', 'requested_amount', 'status'),
        Index('ix_withdrawals_expires_status', 'expires_at', 'status'),
        Index('ix_withdrawals_compliance', 'compliance_status', 'risk_level'),
        Index('ix_withdrawals_tags', 'tags', postgresql_using='gin'),
    )
    
    @validates('withdrawal_id')
    def validate_withdrawal_id(self, key, withdrawal_id):
        """Validate withdrawal ID is not empty."""
        if not withdrawal_id or len(withdrawal_id.strip()) == 0:
            raise ValidationError("Withdrawal ID cannot be empty")
        return withdrawal_id.strip()
    
    @validates('requested_amount', 'approved_amount', 'net_amount')
    def validate_amounts(self, key, amount):
        """Validate withdrawal amounts are positive."""
        if amount is not None and amount <= 0:
            raise ValidationError(f"{key} must be positive")
        return amount
    
    @validates('wallet_address')
    def validate_wallet_address(self, key, address):
        """Validate wallet address format (basic validation)."""
        if address and self.withdrawal_type == WithdrawalType.CRYPTO:
            if len(address) < 10 or len(address) > 255:
                raise ValidationError("Wallet address must be between 10 and 255 characters")
        return address
    
    @validates('confirmation_count', 'required_confirmations')
    def validate_confirmations(self, key, count):
        """Validate confirmation counts are non-negative."""
        if count is not None and count < 0:
            raise ValidationError(f"{key} cannot be negative")
        return count
    
    @validates('retry_count', 'max_retries')
    def validate_retries(self, key, retries):
        """Validate retry counts are non-negative."""
        if retries < 0:
            raise ValidationError(f"{key} cannot be negative")
        return retries
    
    @validates('risk_score')
    def validate_risk_score(self, key, score):
        """Validate risk score is between 0 and 100."""
        if score is not None and (score < 0 or score > 100):
            raise ValidationError("Risk score must be between 0 and 100")
        return score
    
    def __repr__(self):
        return f"<Withdrawal {self.withdrawal_id} ({self.withdrawal_type} - {self.status})>"
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        Convert withdrawal to dictionary representation.
        
        Args:
            include_sensitive: Whether to include sensitive information
            
        Returns:
            Dictionary representation
        """
        result = {
            "id": self.id,
            "withdrawal_id": self.withdrawal_id,
            "external_reference": self.external_reference,
            "user_id": self.user_id,
            "wallet_id": self.wallet_id,
            "withdrawal_type": self.withdrawal_type.value,
            "status": self.status.value,
            "requested_amount": float(self.requested_amount) if self.requested_amount else 0.0,
            "approved_amount": float(self.approved_amount) if self.approved_amount else None,
            "net_amount": float(self.net_amount) if self.net_amount else None,
            "fees": self.fees,
            "crypto_currency": self.crypto_currency,
            "network": self.network.value if self.network else None,
            "wallet_address": self.wallet_address if include_sensitive else self._mask_address(self.wallet_address),
            "address_label": self.address_label,
            "fiat_currency": self.fiat_currency.value if self.fiat_currency else None,
            "payment_method": self.payment_method,
            "transaction_hash": self.transaction_hash,
            "transaction_url": self.transaction_url,
            "confirmation_count": self.confirmation_count,
            "required_confirmations": self.required_confirmations,
            "processor_id": self.processor_id,
            "processor_fee": float(self.processor_fee) if self.processor_fee else None,
            "requested_at": self.requested_at.isoformat() if self.requested_at else None,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "failed_at": self.failed_at.isoformat() if self.failed_at else None,
            "cancelled_at": self.cancelled_at.isoformat() if self.cancelled_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "reviewed_by": self.reviewed_by,
            "approved_by": self.approved_by,
            "processed_by": self.processed_by,
            "rejection_reason": self.rejection_reason,
            "rejection_code": self.rejection_code,
            "failure_reason": self.failure_reason,
            "failure_code": self.failure_code,
            "can_retry": self.can_retry,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "two_factor_verified": self.two_factor_verified,
            "description": self.description,
            "compliance_check_id": self.compliance_check_id,
            "compliance_status": self.compliance_status,
            "requires_manual_review": self.requires_manual_review,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
        }
        
        if include_sensitive:
            result.update({
                "bank_account_details": self.bank_account_details,
                "processor_response": self.processor_response,
                "verification_hash": self.verification_hash,
                "ip_address": self.ip_address,
                "user_agent": self.user_agent,
                "notes": self.notes,
                "metadata": self.metadata,
                "tags": self.tags,
                "audit_trail": self.audit_trail,
            })
        
        return result
    
    def _mask_address(self, address: Optional[str]) -> Optional[str]:
        """Mask sensitive address information for public view."""
        if not address:
            return None
        
        if len(address) <= 8:
            return address
        
        # Show first 4 and last 4 characters
        return f"{address[:4]}...{address[-4:]}"
    
    def to_public_dict(self) -> Dict[str, Any]:
        """
        Convert to public dictionary (safe for user viewing).
        
        Returns:
            Public-safe dictionary representation
        """
        result = self.to_dict(include_sensitive=False)
        
        # Remove internal fields
        internal_fields = [
            'external_reference', 'wallet_id', 'processor_id', 'processor_fee',
            'reviewed_by', 'approved_by', 'processed_by', 'rejection_code',
            'failure_code', 'can_retry', 'retry_count', 'max_retries',
            'compliance_check_id', 'compliance_status', 'requires_manual_review',
            'risk_score', 'risk_level', 'processor_response'
        ]
        
        for field in internal_fields:
            if field in result:
                del result[field]
        
        return result
    
    @property
    def is_pending(self) -> bool:
        """
        Check if withdrawal is pending.
        
        Returns:
            True if withdrawal is pending
        """
        return self.status == WithdrawalStatus.PENDING
    
    @property
    def is_approved(self) -> bool:
        """
        Check if withdrawal is approved.
        
        Returns:
            True if withdrawal is approved
        """
        return self.status == WithdrawalStatus.APPROVED
    
    @property
    def is_processing(self) -> bool:
        """
        Check if withdrawal is being processed.
        
        Returns:
            True if withdrawal is processing
        """
        return self.status == WithdrawalStatus.PROCESSING
    
    @property
    def is_completed(self) -> bool:
        """
        Check if withdrawal is completed.
        
        Returns:
            True if withdrawal is completed
        """
        return self.status == WithdrawalStatus.COMPLETED
    
    @property
    def is_failed(self) -> bool:
        """
        Check if withdrawal failed.
        
        Returns:
            True if withdrawal failed
        """
        return self.status == WithdrawalStatus.FAILED
    
    @property
    def is_cancelled(self) -> bool:
        """
        Check if withdrawal is cancelled.
        
        Returns:
            True if withdrawal is cancelled
        """
        return self.status == WithdrawalStatus.CANCELLED
    
    @property
    def is_expired(self) -> bool:
        """
        Check if withdrawal has expired.
        
        Returns:
            True if withdrawal has expired
        """
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False
    
    @property
    def can_be_cancelled(self) -> bool:
        """
        Check if withdrawal can be cancelled.
        
        Returns:
            True if withdrawal can be cancelled
        """
        cancellable_statuses = [
            WithdrawalStatus.PENDING,
            WithdrawalStatus.UNDER_REVIEW,
            WithdrawalStatus.APPROVED,
            WithdrawalStatus.QUEUED,
        ]
        return self.status in cancellable_statuses and not self.is_expired
    
    @property
    def can_be_retried(self) -> bool:
        """
        Check if withdrawal can be retried.
        
        Returns:
            True if withdrawal can be retried
        """
        return (
            self.status == WithdrawalStatus.FAILED and 
            self.can_retry and 
            self.retry_count < self.max_retries
        )
    
    @property
    def amount_to_process(self) -> Decimal:
        """
        Get the amount to process (approved amount or requested amount).
        
        Returns:
            Amount to process as Decimal
        """
        if self.approved_amount:
            return Decimal(str(self.approved_amount))
        return Decimal(str(self.requested_amount))
    
    @property
    def calculated_fees(self) -> Dict[str, Decimal]:
        """
        Calculate total fees for the withdrawal.
        
        Returns:
            Dictionary of fee breakdown
        """
        if self.fees:
            return {k: Decimal(str(v)) for k, v in self.fees.items()}
        return {}
    
    @property
    def total_fees(self) -> Decimal:
        """
        Calculate total fees.
        
        Returns:
            Total fees as Decimal
        """
        fees = self.calculated_fees
        return sum(fees.values())
    
    @property
    def net_amount_calculated(self) -> Decimal:
        """
        Calculate net amount after fees.
        
        Returns:
            Net amount as Decimal
        """
        amount = self.amount_to_process
        fees = self.total_fees
        return amount - fees
    
    @property
    def is_crypto_withdrawal(self) -> bool:
        """
        Check if this is a cryptocurrency withdrawal.
        
        Returns:
            True if cryptocurrency withdrawal
        """
        return self.withdrawal_type == WithdrawalType.CRYPTO
    
    @property
    def is_fiat_withdrawal(self) -> bool:
        """
        Check if this is a fiat withdrawal.
        
        Returns:
            True if fiat withdrawal
        """
        return self.withdrawal_type == WithdrawalType.FIAT
    
    @property
    def confirmation_progress(self) -> float:
        """
        Get confirmation progress percentage.
        
        Returns:
            Progress percentage (0-100)
        """
        if self.required_confirmations == 0:
            return 100.0
        return min((self.confirmation_count / self.required_confirmations) * 100, 100.0)
    
    @property
    def days_since_requested(self) -> int:
        """
        Get days since withdrawal was requested.
        
        Returns:
            Number of days
        """
        if not self.requested_at:
            return 0
        delta = datetime.utcnow() - self.requested_at
        return delta.days
    
    def add_audit_entry(self, action: str, details: Dict[str, Any], user_id: Optional[int] = None):
        """
        Add an entry to the audit trail.
        
        Args:
            action: Action performed
            details: Action details
            user_id: ID of user performing action
        """
        if self.audit_trail is None:
            self.audit_trail = []
        
        entry = {
            "action": action,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "details": details,
            "previous_status": self.status.value,
        }
        
        self.audit_trail.append(entry)
    
    def submit(self, ip_address: Optional[str] = None, user_agent: Optional[str] = None):
        """
        Submit the withdrawal request.
        
        Args:
            ip_address: IP address of request
            user_agent: User agent of request
        """
        if self.status != WithdrawalStatus.DRAFT:
            raise ValidationError("Only draft withdrawals can be submitted")
        
        self.status = WithdrawalStatus.PENDING
        self.requested_at = datetime.utcnow()
        
        if ip_address:
            self.ip_address = ip_address
        if user_agent:
            self.user_agent = user_agent
        
        # Set expiration (7 days from now)
        self.expires_at = datetime.utcnow() + timedelta(days=7)
        
        # Add audit entry
        self.add_audit_entry(
            action="submitted",
            details={"ip_address": ip_address, "user_agent": user_agent}
        )
    
    def review(self, reviewed_by: int, notes: Optional[str] = None):
        """
        Mark withdrawal as under review.
        
        Args:
            reviewed_by: ID of user reviewing
            notes: Review notes
        """
        if self.status != WithdrawalStatus.PENDING:
            raise ValidationError("Only pending withdrawals can be reviewed")
        
        self.status = WithdrawalStatus.UNDER_REVIEW
        self.reviewed_by = reviewed_by
        self.reviewed_at = datetime.utcnow()
        
        if notes:
            if not self.notes:
                self.notes = notes
            else:
                self.notes += f"\nReview: {notes}"
        
        # Add audit entry
        self.add_audit_entry(
            action="review_started",
            details={"reviewed_by": reviewed_by, "notes": notes},
            user_id=reviewed_by
        )
    
    def approve(self, approved_by: int, approved_amount: Optional[Decimal] = None, 
                fees: Optional[Dict[str, Decimal]] = None, notes: Optional[str] = None):
        """
        Approve the withdrawal.
        
        Args:
            approved_by: ID of user approving
            approved_amount: Optional approved amount (different from requested)
            fees: Optional fee breakdown
            notes: Approval notes
        """
        if self.status not in [WithdrawalStatus.PENDING, WithdrawalStatus.UNDER_REVIEW]:
            raise ValidationError("Only pending or under review withdrawals can be approved")
        
        self.status = WithdrawalStatus.APPROVED
        self.approved_by = approved_by
        self.approved_at = datetime.utcnow()
        
        if approved_amount:
            self.approved_amount = approved_amount
        
        if fees:
            self.fees = {k: float(v) for k, v in fees.items()}
        
        # Calculate net amount
        self.net_amount = self.net_amount_calculated
        
        if notes:
            if not self.notes:
                self.notes = f"Approved: {notes}"
            else:
                self.notes += f"\nApproved: {notes}"
        
        # Add audit entry
        self.add_audit_entry(
            action="approved",
            details={
                "approved_by": approved_by,
                "approved_amount": float(approved_amount) if approved_amount else None,
                "fees": fees,
                "notes": notes
            },
            user_id=approved_by
        )
    
    def reject(self, rejected_by: int, reason: str, code: Optional[str] = None, 
               notes: Optional[str] = None):
        """
        Reject the withdrawal.
        
        Args:
            rejected_by: ID of user rejecting
            reason: Rejection reason
            code: Rejection code
            notes: Additional notes
        """
        if self.status not in [WithdrawalStatus.PENDING, WithdrawalStatus.UNDER_REVIEW]:
            raise ValidationError("Only pending or under review withdrawals can be rejected")
        
        self.status = WithdrawalStatus.REJECTED
        self.rejection_reason = reason
        self.rejection_code = code
        
        if notes:
            if not self.notes:
                self.notes = f"Rejected: {notes}"
            else:
                self.notes += f"\nRejected: {notes}"
        
        # Add audit entry
        self.add_audit_entry(
            action="rejected",
            details={"rejected_by": rejected_by, "reason": reason, "code": code, "notes": notes},
            user_id=rejected_by
        )
    
    def process(self, processed_by: int, processor_id: Optional[str] = None):
        """
        Mark withdrawal as being processed.
        
        Args:
            processed_by: ID of user processing
            processor_id: Payment processor ID
        """
        if self.status != WithdrawalStatus.APPROVED:
            raise ValidationError("Only approved withdrawals can be processed")
        
        self.status = WithdrawalStatus.PROCESSING
        self.processed_by = processed_by
        self.processed_at = datetime.utcnow()
        
        if processor_id:
            self.processor_id = processor_id
        
        # Add audit entry
        self.add_audit_entry(
            action="processing_started",
            details={"processed_by": processed_by, "processor_id": processor_id},
            user_id=processed_by
        )
    
    def send(self, transaction_hash: str, processor_response: Optional[Dict] = None):
        """
        Mark withdrawal as sent.
        
        Args:
            transaction_hash: Blockchain transaction hash
            processor_response: Processor response data
        """
        if self.status != WithdrawalStatus.PROCESSING:
            raise ValidationError("Only processing withdrawals can be sent")
        
        self.status = WithdrawalStatus.SENT
        self.transaction_hash = transaction_hash
        self.sent_at = datetime.utcnow()
        
        if processor_response:
            self.processor_response = processor_response
        
        # Generate transaction URL if possible
        if self.network and transaction_hash:
            self.transaction_url = self._generate_transaction_url(transaction_hash)
        
        # Add audit entry
        self.add_audit_entry(
            action="sent",
            details={"transaction_hash": transaction_hash}
        )
    
    def _generate_transaction_url(self, transaction_hash: str) -> str:
        """Generate blockchain explorer URL for transaction."""
        explorers = {
            NetworkType.ETHEREUM: f"https://etherscan.io/tx/{transaction_hash}",
            NetworkType.POLYGON: f"https://polygonscan.com/tx/{transaction_hash}",
            NetworkType.BINANCE_SMART_CHAIN: f"https://bscscan.com/tx/{transaction_hash}",
            NetworkType.ARBITRUM: f"https://arbiscan.io/tx/{transaction_hash}",
            NetworkType.OPTIMISM: f"https://optimistic.etherscan.io/tx/{transaction_hash}",
            NetworkType.AVALANCHE: f"https://snowtrace.io/tx/{transaction_hash}",
            NetworkType.FANTOM: f"https://ftmscan.com/tx/{transaction_hash}",
            NetworkType.BITCOIN: f"https://blockchain.com/btc/tx/{transaction_hash}",
            NetworkType.SOLANA: f"https://explorer.solana.com/tx/{transaction_hash}",
            NetworkType.GOERLI: f"https://goerli.etherscan.io/tx/{transaction_hash}",
            NetworkType.SEPOLIA: f"https://sepolia.etherscan.io/tx/{transaction_hash}",
            NetworkType.MUMBAI: f"https://mumbai.polygonscan.com/tx/{transaction_hash}",
        }
        
        return explorers.get(self.network, "")
    
    def update_confirmations(self, count: int):
        """
        Update confirmation count.
        
        Args:
            count: New confirmation count
        """
        self.confirmation_count = count
        
        # If we have enough confirmations, mark as completed
        if count >= self.required_confirmations and self.status == WithdrawalStatus.SENT:
            self.complete()
        
        # Update status to confirming if we have some confirmations but not enough
        elif count > 0 and self.status == WithdrawalStatus.SENT:
            self.status = WithdrawalStatus.CONFIRMING
        
        # Add audit entry
        self.add_audit_entry(
            action="confirmations_updated",
            details={"new_count": count, "required": self.required_confirmations}
        )
    
    def complete(self):
        """Mark withdrawal as completed."""
        if self.status not in [WithdrawalStatus.SENT, WithdrawalStatus.CONFIRMING]:
            raise ValidationError("Only sent or confirming withdrawals can be completed")
        
        self.status = WithdrawalStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        
        # Add audit entry
        self.add_audit_entry(action="completed", details={})
    
    def fail(self, reason: str, code: Optional[str] = None, can_retry: bool = True):
        """
        Mark withdrawal as failed.
        
        Args:
            reason: Failure reason
            code: Failure code
            can_retry: Whether withdrawal can be retried
        """
        self.status = WithdrawalStatus.FAILED
        self.failure_reason = reason
        self.failure_code = code
        self.can_retry = can_retry
        self.failed_at = datetime.utcnow()
        self.retry_count += 1
        
        # Add audit entry
        self.add_audit_entry(
            action="failed",
            details={"reason": reason, "code": code, "can_retry": can_retry, "retry_count": self.retry_count}
        )
    
    def cancel(self, cancelled_by: Optional[int] = None, reason: Optional[str] = None):
        """
        Cancel the withdrawal.
        
        Args:
            cancelled_by: ID of user cancelling
            reason: Cancellation reason
        """
        if not self.can_be_cancelled:
            raise ValidationError("Withdrawal cannot be cancelled in its current state")
        
        self.status = WithdrawalStatus.CANCELLED
        self.cancelled_at = datetime.utcnow()
        
        if reason:
            if not self.notes:
                self.notes = f"Cancelled: {reason}"
            else:
                self.notes += f"\nCancelled: {reason}"
        
        # Add audit entry
        self.add_audit_entry(
            action="cancelled",
            details={"cancelled_by": cancelled_by, "reason": reason},
            user_id=cancelled_by
        )
    
    def retry(self, retried_by: Optional[int] = None):
        """Retry a failed withdrawal."""
        if not self.can_be_retried:
            raise ValidationError("Withdrawal cannot be retried")
        
        self.status = WithdrawalStatus.PENDING
        self.failure_reason = None
        self.failure_code = None
        
        # Add audit entry
        self.add_audit_entry(
            action="retry",
            details={"retried_by": retried_by, "attempt": self.retry_count + 1},
            user_id=retried_by
        )
    
    def refund(self, refunded_by: int, transaction_hash: Optional[str] = None, 
               notes: Optional[str] = None):
        """
        Mark withdrawal as refunded.
        
        Args:
            refunded_by: ID of user processing refund
            transaction_hash: Refund transaction hash
            notes: Refund notes
        """
        if self.status != WithdrawalStatus.FAILED:
            raise ValidationError("Only failed withdrawals can be refunded")
        
        self.status = WithdrawalStatus.REFUNDED
        
        if transaction_hash:
            self.transaction_hash = transaction_hash
        
        if notes:
            if not self.notes:
                self.notes = f"Refunded: {notes}"
            else:
                self.notes += f"\nRefunded: {notes}"
        
        # Add audit entry
        self.add_audit_entry(
            action="refunded",
            details={"refunded_by": refunded_by, "transaction_hash": transaction_hash, "notes": notes},
            user_id=refunded_by
        )
    
    def suspend(self, suspended_by: int, reason: str, notes: Optional[str] = None):
        """
        Suspend withdrawal for investigation.
        
        Args:
            suspended_by: ID of user suspending
            reason: Suspension reason
            notes: Additional notes
        """
        self.status = WithdrawalStatus.SUSPENDED
        self.requires_manual_review = True
        
        if notes:
            if not self.notes:
                self.notes = f"Suspended: {notes}"
            else:
                self.notes += f"\nSuspended: {notes}"
        
        # Add audit entry
        self.add_audit_entry(
            action="suspended",
            details={"suspended_by": suspended_by, "reason": reason, "notes": notes},
            user_id=suspended_by
        )
    
    def put_on_hold(self, reason: str, notes: Optional[str] = None):
        """
        Put withdrawal on hold.
        
        Args:
            reason: Hold reason
            notes: Additional notes
        """
        self.status = WithdrawalStatus.ON_HOLD
        
        if notes:
            if not self.notes:
                self.notes = f"On hold: {notes}"
            else:
                self.notes += f"\nOn hold: {notes}"
        
        # Add audit entry
        self.add_audit_entry(
            action="on_hold",
            details={"reason": reason, "notes": notes}
        )


class WithdrawalLimit(Base):
    """
    Model for defining withdrawal limits for users.
    """
    
    __tablename__ = "withdrawal_limits"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Daily limits
    daily_limit = Column(Numeric(20, 8), nullable=True, comment="Daily withdrawal limit")
    daily_used = Column(Numeric(20, 8), default=0, comment="Amount used today")
    daily_reset_at = Column(DateTime, nullable=True, comment="When daily limit resets")
    
    # Weekly limits
    weekly_limit = Column(Numeric(20, 8), nullable=True, comment="Weekly withdrawal limit")
    weekly_used = Column(Numeric(20, 8), default=0, comment="Amount used this week")
    weekly_reset_at = Column(DateTime, nullable=True, comment="When weekly limit resets")
    
    # Monthly limits
    monthly_limit = Column(Numeric(20, 8), nullable=True, comment="Monthly withdrawal limit")
    monthly_used = Column(Numeric(20, 8), default=0, comment="Amount used this month")
    monthly_reset_at = Column(DateTime, nullable=True, comment="When monthly limit resets")
    
    # Per transaction limits
    min_per_transaction = Column(Numeric(20, 8), nullable=True, comment="Minimum per transaction")
    max_per_transaction = Column(Numeric(20, 8), nullable=True, comment="Maximum per transaction")
    
    # Count limits
    daily_count_limit = Column(Integer, nullable=True, comment="Max withdrawals per day")
    daily_count_used = Column(Integer, default=0, comment="Withdrawals used today")
    
    # Tier-based limits
    tier = Column(String(50), nullable=True, index=True, comment="User tier for limits")
    
    # Status
    is_active = Column(Boolean, default=True, index=True)
    is_verified = Column(Boolean, default=False, comment="Whether user is verified")
    
    # Restrictions
    allowed_networks = Column(JSONB, nullable=True, comment="JSON array of allowed networks")
    allowed_currencies = Column(JSONB, nullable=True, comment="JSON array of allowed currencies")
    restricted_until = Column(DateTime, nullable=True, comment="Restricted until date")
    
    # Metadata
    notes = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", backref="withdrawal_limits")
    
    def check_limit(self, amount: Decimal, withdrawal_type: WithdrawalType) -> Tuple[bool, str]:
        """
        Check if withdrawal is within limits.
        
        Args:
            amount: Withdrawal amount
            withdrawal_type: Type of withdrawal
            
        Returns:
            Tuple of (within_limit, error_message)
        """
        # Check if restricted
        if self.restricted_until and datetime.utcnow() < self.restricted_until:
            return False, "Withdrawals are temporarily restricted"
        
        # Check per transaction limits
        if self.min_per_transaction and amount < Decimal(str(self.min_per_transaction)):
            return False, f"Minimum withdrawal is {self.min_per_transaction}"
        
        if self.max_per_transaction and amount > Decimal(str(self.max_per_transaction)):
            return False, f"Maximum per transaction is {self.max_per_transaction}"
        
        # Check daily limits
        if self.daily_limit:
            # Reset daily used if reset time has passed
            if self.daily_reset_at and datetime.utcnow() > self.daily_reset_at:
                self.daily_used = 0
                self.daily_reset_at = datetime.utcnow() + timedelta(days=1)
            
            if self.daily_used + amount > Decimal(str(self.daily_limit)):
                return False, f"Daily limit exceeded. Remaining: {self.daily_limit - self.daily_used}"
        
        # Check daily count limit
        if self.daily_count_limit and self.daily_count_used >= self.daily_count_limit:
            return False, f"Daily withdrawal count limit ({self.daily_count_limit}) reached"
        
        # Check weekly limits
        if self.weekly_limit:
            if self.weekly_reset_at and datetime.utcnow() > self.weekly_reset_at:
                self.weekly_used = 0
                self.weekly_reset_at = datetime.utcnow() + timedelta(weeks=1)
            
            if self.weekly_used + amount > Decimal(str(self.weekly_limit)):
                return False, f"Weekly limit exceeded. Remaining: {self.weekly_limit - self.weekly_used}"
        
        # Check monthly limits
        if self.monthly_limit:
            if self.monthly_reset_at and datetime.utcnow() > self.monthly_reset_at:
                self.monthly_used = 0
                self.monthly_reset_at = datetime.utcnow() + timedelta(days=30)
            
            if self.monthly_used + amount > Decimal(str(self.monthly_limit)):
                return False, f"Monthly limit exceeded. Remaining: {self.monthly_limit - self.monthly_used}"
        
        return True, ""
    
    def record_withdrawal(self, amount: Decimal):
        """Record a withdrawal against limits."""
        self.daily_used += amount
        self.weekly_used += amount
        self.monthly_used += amount
        self.daily_count_used += 1
        self.updated_at = datetime.utcnow()


class WithdrawalFeeSchedule(Base):
    """
    Model for defining withdrawal fee schedules.
    """
    
    __tablename__ = "withdrawal_fee_schedules"
    
    id = Column(Integer, primary_key=True, index=True)
    schedule_id = Column(String(100), unique=True, index=True, nullable=False)
    
    # Fee schedule details
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # Applicability
    withdrawal_type = Column(Enum(WithdrawalType), nullable=False, index=True)
    network = Column(Enum(NetworkType), nullable=True, index=True)
    currency = Column(String(10), nullable=True, index=True, comment="Currency symbol")
    
    # Fee calculation
    fee_type = Column(String(50), nullable=False, comment="fixed, percentage, tiered, dynamic")
    fee_amount = Column(Numeric(20, 8), nullable=True, comment="Fixed fee amount")
    fee_percentage = Column(Numeric(5, 2), nullable=True, comment="Percentage fee (0-100)")
    minimum_fee = Column(Numeric(20, 8), nullable=True, comment="Minimum fee")
    maximum_fee = Column(Numeric(20, 8), nullable=True, comment="Maximum fee")
    
    # Tiered fees (JSON structure)
    tiered_fees = Column(JSONB, nullable=True, comment="JSON array of tiered fee rates")
    
    # Timing
    valid_from = Column(DateTime, nullable=False, index=True)
    valid_until = Column(DateTime, nullable=True, index=True)
    
    # Priority
    priority = Column(Integer, default=1, nullable=False, index=True)
    is_active = Column(Boolean, default=True, index=True)
    
    # Usage tracking
    usage_count = Column(Integer, default=0)
    last_used_at = Column(DateTime, nullable=True)
    
    # Metadata
    tags = Column(JSONB, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def calculate_fee(self, amount: Decimal) -> Decimal:
        """
        Calculate fee for a withdrawal amount.
        
        Args:
            amount: Withdrawal amount
            
        Returns:
            Calculated fee
        """
        if self.fee_type == "fixed":
            fee = Decimal(str(self.fee_amount)) if self.fee_amount else Decimal('0')
        
        elif self.fee_type == "percentage":
            percentage = Decimal(str(self.fee_percentage)) if self.fee_percentage else Decimal('0')
            fee = amount * (percentage / Decimal('100'))
        
        elif self.fee_type == "tiered" and self.tiered_fees:
            fee = Decimal('0')
            # Implement tiered fee calculation
            # This would parse the tiered_fees JSON and apply appropriate rate
            pass
        
        else:
            fee = Decimal('0')
        
        # Apply minimum and maximum
        if self.minimum_fee and fee < Decimal(str(self.minimum_fee)):
            fee = Decimal(str(self.minimum_fee))
        
        if self.maximum_fee and fee > Decimal(str(self.maximum_fee)):
            fee = Decimal(str(self.maximum_fee))
        
        return fee


class WithdrawalBlacklist(Base):
    """
    Model for blacklisting addresses, users, or countries from withdrawals.
    """
    
    __tablename__ = "withdrawal_blacklist"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Blacklist target
    target_type = Column(String(50), nullable=False, index=True, comment="user, address, country, ip")
    target_value = Column(String(255), nullable=False, index=True, comment="Value to blacklist")
    
    # Details
    reason = Column(Text, nullable=False)
    blacklisted_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Scope
    scope = Column(JSONB, nullable=True, comment="JSON scope (networks, currencies, etc.)")
    
    # Duration
    starts_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ends_at = Column(DateTime, nullable=True, comment="Null means permanent")
    
    # Status
    is_active = Column(Boolean, default=True, index=True)
    
    # Metadata
    notes = Column(Text, nullable=True)
    reference_id = Column(String(100), nullable=True, comment="Reference to related entity")
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    blacklister = relationship("User", foreign_keys=[blacklisted_by])
    
    @property
    def is_expired(self) -> bool:
        """Check if blacklist has expired."""
        if self.ends_at is None:
            return False
        return datetime.utcnow() > self.ends_at
    
    @property
    def is_effective(self) -> bool:
        """Check if blacklist is currently effective."""
        return self.is_active and not self.is_expired
    
    def matches(self, value: str, target_type: str) -> bool:
        """
        Check if a value matches this blacklist entry.
        
        Args:
            value: Value to check
            target_type: Type of value
            
        Returns:
            True if matches
        """
        if not self.is_effective:
            return False
        
        if self.target_type != target_type:
            return False
        
        # Exact match
        if self.target_value == value:
            return True
        
        # Wildcard match (if target_value contains *)
        if '*' in self.target_value:
            import fnmatch
            return fnmatch.fnmatch(value, self.target_value)
        
        return False