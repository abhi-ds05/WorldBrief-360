"""
transaction.py - Transaction and Audit Model

This module defines models for transaction management, audit trails, and data consistency.
This includes:
- Financial transactions and ledgers
- Data audit trails and change logs
- Transaction rollback and recovery
- Multi-step transaction coordination
- Compliance and regulatory reporting
- Data consistency and integrity

Key Features:
- ACID transaction support
- Comprehensive audit trails
- Transaction rollback and recovery
- Financial ledger management
- Data change tracking
- Compliance reporting
- Distributed transaction support
- Transaction state management
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from enum import Enum
from decimal import Decimal
from sqlalchemy import (
    Column, String, Text, ForeignKey, Integer, DateTime, 
    Boolean, Enum as SQLEnum, JSON, Float, CheckConstraint,
    Index, Table, UniqueConstraint, Numeric, LargeBinary
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
    from models.subscription import Subscription, Invoice, Payment
    from models.incident import Incident
    from models.article import Article


class TransactionType(Enum):
    """Types of transactions."""
    # Financial transactions
    PAYMENT = "payment"                      # Payment transaction
    REFUND = "refund"                        # Refund transaction
    CHARGE = "charge"                        # Charge transaction
    TRANSFER = "transfer"                    # Transfer between accounts
    ADJUSTMENT = "adjustment"                # Manual adjustment
    FEE = "fee"                              # Fee transaction
    TAX = "tax"                              # Tax transaction
    DISCOUNT = "discount"                    # Discount application
    CREDIT = "credit"                        # Credit application
    
    # Data transactions
    CREATE = "create"                        # Data creation
    UPDATE = "update"                        # Data update
    DELETE = "delete"                        # Data deletion
    MERGE = "merge"                          # Data merge
    SPLIT = "split"                          # Data split
    IMPORT = "import"                        # Data import
    EXPORT = "export"                        # Data export
    BACKUP = "backup"                        # Data backup
    RESTORE = "restore"                      # Data restore
    
    # System transactions
    SYSTEM_ACTION = "system_action"          # System automated action
    BATCH_PROCESS = "batch_process"          # Batch processing
    SCHEDULED_TASK = "scheduled_task"        # Scheduled task execution
    MAINTENANCE = "maintenance"              # Maintenance operation
    UPGRADE = "upgrade"                      # System upgrade
    CONFIG_CHANGE = "config_change"          # Configuration change
    
    # User transactions
    LOGIN = "login"                          # User login
    LOGOUT = "logout"                        # User logout
    PASSWORD_CHANGE = "password_change"      # Password change
    PROFILE_UPDATE = "profile_update"        # Profile update
    PERMISSION_CHANGE = "permission_change"  # Permission change
    
    # Business transactions
    SUBSCRIPTION_CREATE = "subscription_create"      # Subscription creation
    SUBSCRIPTION_UPDATE = "subscription_update"      # Subscription update
    SUBSCRIPTION_CANCEL = "subscription_cancel"      # Subscription cancellation
    INVOICE_CREATE = "invoice_create"                # Invoice creation
    INVOICE_UPDATE = "invoice_update"                # Invoice update
    INVOICE_PAYMENT = "invoice_payment"              # Invoice payment


class TransactionStatus(Enum):
    """Transaction status."""
    PENDING = "pending"                      # Transaction pending
    PROCESSING = "processing"                # Transaction being processed
    COMPLETED = "completed"                  # Transaction completed successfully
    FAILED = "failed"                        # Transaction failed
    CANCELLED = "cancelled"                  # Transaction cancelled
    REVERSED = "reversed"                    # Transaction reversed
    PARTIALLY_COMPLETED = "partially_completed"  # Partially completed
    ROLLED_BACK = "rolled_back"              # Transaction rolled back
    SUSPENDED = "suspended"                  # Transaction suspended


class LedgerEntryType(Enum):
    """Ledger entry types (debit/credit)."""
    DEBIT = "debit"                          # Debit entry
    CREDIT = "credit"                        # Credit entry
    OPENING_BALANCE = "opening_balance"      # Opening balance
    CLOSING_BALANCE = "closing_balance"      # Closing balance
    ADJUSTMENT = "adjustment"                # Adjustment entry


class AuditAction(Enum):
    """Audit log actions."""
    CREATE = "create"                        # Create operation
    READ = "read"                            # Read operation
    UPDATE = "update"                        # Update operation
    DELETE = "delete"                        # Delete operation
    EXECUTE = "execute"                      # Execute operation
    APPROVE = "approve"                      # Approve operation
    REJECT = "reject"                        # Reject operation
    EXPORT = "export"                        # Export operation
    IMPORT = "import"                        # Import operation
    LOGIN = "login"                          # Login operation
    LOGOUT = "logout"                        # Logout operation


class Transaction(Base, UUIDMixin, TimestampMixin):
    """
    Transaction model for tracking operations and ensuring data consistency.
    
    This model represents a unit of work that follows ACID properties
    (Atomicity, Consistency, Isolation, Durability).
    
    Attributes:
        id: Primary key UUID
        transaction_type: Type of transaction
        status: Transaction status
        reference_id: External reference ID
        correlation_id: Correlation ID for related transactions
        parent_transaction_id: Parent transaction for nested transactions
        amount: Transaction amount
        currency: Currency code
        description: Transaction description
        initiated_by: User who initiated transaction
        initiated_by_ip: IP address of initiator
        initiated_by_user_agent: User agent of initiator
        completed_at: When transaction completed
        error_message: Error message if failed
        error_code: Error code if failed
        retry_count: Number of retry attempts
        max_retries: Maximum retry attempts
        timeout_seconds: Transaction timeout in seconds
        metadata: Additional metadata
        tags: Categorization tags
        organization_id: Related organization
        entity_type: Type of entity affected
        entity_id: ID of entity affected
        rollback_transaction_id: Transaction that rolled back this one
    """
    
    __tablename__ = "transactions"
    
    # Transaction identification
    transaction_type = Column(SQLEnum(TransactionType), nullable=False, index=True)
    status = Column(SQLEnum(TransactionStatus), default=TransactionStatus.PENDING, nullable=False, index=True)
    reference_id = Column(String(200), nullable=True, unique=True, index=True)
    correlation_id = Column(String(200), nullable=True, index=True)
    
    # Transaction hierarchy
    parent_transaction_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("transactions.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    
    # Financial details (if applicable)
    amount = Column(Numeric(15, 2), nullable=True)
    currency = Column(String(3), nullable=True)
    
    # Description
    description = Column(Text, nullable=True)
    
    # Initiation details
    initiated_by = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    initiated_by_ip = Column(String(50), nullable=True)
    initiated_by_user_agent = Column(Text, nullable=True)
    
    # Completion details
    completed_at = Column(DateTime(timezone=True), nullable=True, index=True)
    error_message = Column(Text, nullable=True)
    error_code = Column(String(100), nullable=True, index=True)
    
    # Retry and timeout
    retry_count = Column(Integer, default=0, nullable=False)
    max_retries = Column(Integer, default=3, nullable=False)
    timeout_seconds = Column(Integer, nullable=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    tags = Column(ARRAY(String), default=[], nullable=False, index=True)
    
    # Related entities
    organization_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("organizations.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    entity_type = Column(String(100), nullable=True, index=True)
    entity_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    
    # Rollback tracking
    rollback_transaction_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("transactions.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    
    # Relationships
    initiator = relationship("User", foreign_keys=[initiated_by])
    organization = relationship("Organization")
    parent_transaction = relationship("Transaction", remote_side=[id], backref="child_transactions")
    rollback_transaction = relationship("Transaction", remote_side=[rollback_transaction_id], backref="rolled_back_by")
    ledger_entries = relationship("LedgerEntry", back_populates="transaction", cascade="all, delete-orphan")
    transaction_steps = relationship("TransactionStep", back_populates="transaction", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="transaction")
    
    # Check constraints
    __table_args__ = (
        CheckConstraint('amount IS NULL OR amount >= 0', name='check_amount_non_negative'),
        CheckConstraint('retry_count >= 0', name='check_retry_count_non_negative'),
        CheckConstraint('max_retries >= 0', name='check_max_retries_non_negative'),
        CheckConstraint('timeout_seconds IS NULL OR timeout_seconds > 0', name='check_timeout_positive'),
        Index('ix_transactions_status_type', 'status', 'transaction_type'),
        Index('ix_transactions_correlation', 'correlation_id', 'created_at'),
        Index('ix_transactions_entity', 'entity_type', 'entity_id'),
        Index('ix_transactions_completed', 'status', 'completed_at'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Transaction(id={self.id}, type={self.transaction_type.value}, status={self.status.value})>"
    
    @property
    def is_completed(self) -> bool:
        """Check if transaction is completed."""
        return self.status == TransactionStatus.COMPLETED
    
    @property
    def is_failed(self) -> bool:
        """Check if transaction failed."""
        return self.status == TransactionStatus.FAILED
    
    @property
    def is_pending(self) -> bool:
        """Check if transaction is pending."""
        return self.status == TransactionStatus.PENDING
    
    @property
    def is_processing(self) -> bool:
        """Check if transaction is processing."""
        return self.status == TransactionStatus.PROCESSING
    
    @property
    def is_cancelled(self) -> bool:
        """Check if transaction is cancelled."""
        return self.status == TransactionStatus.CANCELLED
    
    @property
    def is_rolled_back(self) -> bool:
        """Check if transaction was rolled back."""
        return self.status == TransactionStatus.ROLLED_BACK
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get transaction duration in seconds."""
        if self.completed_at and self.created_at:
            return (self.completed_at - self.created_at).total_seconds()
        return None
    
    @property
    def is_timed_out(self) -> bool:
        """Check if transaction has timed out."""
        if not self.timeout_seconds or not self.created_at:
            return False
        
        elapsed = (datetime.utcnow() - self.created_at).total_seconds()
        return elapsed > self.timeout_seconds and self.is_pending
    
    @property
    def can_retry(self) -> bool:
        """Check if transaction can be retried."""
        return (
            self.status in [TransactionStatus.FAILED, TransactionStatus.SUSPENDED] and
            self.retry_count < self.max_retries
        )
    
    @property
    def formatted_amount(self) -> Optional[str]:
        """Get formatted amount string."""
        if self.amount is not None and self.currency:
            return f"{self.currency} {self.amount:.2f}"
        return None
    
    @property
    def total_debit(self) -> Decimal:
        """Get total debit amount from ledger entries."""
        total = Decimal('0')
        for entry in self.ledger_entries:
            if entry.entry_type == LedgerEntryType.DEBIT:
                total += entry.amount
        return total
    
    @property
    def total_credit(self) -> Decimal:
        """Get total credit amount from ledger entries."""
        total = Decimal('0')
        for entry in self.ledger_entries:
            if entry.entry_type == LedgerEntryType.CREDIT:
                total += entry.amount
        return total
    
    @property
    def is_balanced(self) -> bool:
        """Check if transaction ledger is balanced."""
        return self.total_debit == self.total_credit
    
    def start_processing(self) -> None:
        """Start processing the transaction."""
        self.status = TransactionStatus.PROCESSING
    
    def complete(self, completed_at: Optional[datetime] = None) -> None:
        """Mark transaction as completed."""
        self.status = TransactionStatus.COMPLETED
        self.completed_at = completed_at or datetime.utcnow()
        self.retry_count = 0
        self.error_message = None
        self.error_code = None
    
    def fail(self, error_message: str, error_code: Optional[str] = None) -> None:
        """Mark transaction as failed."""
        self.status = TransactionStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
        self.error_code = error_code
        self.retry_count += 1
    
    def cancel(self, reason: Optional[str] = None) -> None:
        """Cancel the transaction."""
        self.status = TransactionStatus.CANCELLED
        self.completed_at = datetime.utcnow()
        if reason:
            self.metadata["cancellation_reason"] = reason
    
    def suspend(self, reason: Optional[str] = None) -> None:
        """Suspend the transaction."""
        self.status = TransactionStatus.SUSPENDED
        if reason:
            self.metadata["suspension_reason"] = reason
    
    def retry(self) -> bool:
        """Retry the transaction."""
        if not self.can_retry:
            return False
        
        self.status = TransactionStatus.PENDING
        self.error_message = None
        self.error_code = None
        return True
    
    def rollback(self, rollback_transaction_id: Optional[uuid.UUID] = None) -> None:
        """Rollback the transaction."""
        self.status = TransactionStatus.ROLLED_BACK
        self.completed_at = datetime.utcnow()
        if rollback_transaction_id:
            self.rollback_transaction_id = rollback_transaction_id
    
    def add_ledger_entry(
        self,
        account_id: uuid.UUID,
        entry_type: LedgerEntryType,
        amount: Decimal,
        currency: str = "USD",
        description: Optional[str] = None,
        reference: Optional[str] = None
    ) -> 'LedgerEntry':
        """Add a ledger entry to the transaction."""
        from models.transaction import LedgerEntry
        
        entry = LedgerEntry(
            transaction_id=self.id,
            account_id=account_id,
            entry_type=entry_type,
            amount=amount,
            currency=currency,
            description=description,
            reference=reference
        )
        self.ledger_entries.append(entry)
        return entry
    
    def add_step(
        self,
        step_name: str,
        step_order: int,
        handler_class: Optional[str] = None,
        handler_method: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> 'TransactionStep':
        """Add a step to the transaction."""
        from models.transaction import TransactionStep
        
        step = TransactionStep(
            transaction_id=self.id,
            step_name=step_name,
            step_order=step_order,
            handler_class=handler_class,
            handler_method=handler_method,
            parameters=parameters or {}
        )
        self.transaction_steps.append(step)
        return step
    
    def to_dict(
        self,
        include_steps: bool = False,
        include_ledger: bool = False,
        include_children: bool = False
    ) -> Dict[str, Any]:
        """Convert transaction to dictionary."""
        result = {
            "id": str(self.id),
            "transaction_type": self.transaction_type.value,
            "status": self.status.value,
            "reference_id": self.reference_id,
            "correlation_id": self.correlation_id,
            "parent_transaction_id": str(self.parent_transaction_id) if self.parent_transaction_id else None,
            "amount": float(self.amount) if self.amount else None,
            "currency": self.currency,
            "formatted_amount": self.formatted_amount,
            "description": self.description,
            "initiated_by": str(self.initiated_by) if self.initiated_by else None,
            "initiated_by_ip": self.initiated_by_ip,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "is_completed": self.is_completed,
            "is_failed": self.is_failed,
            "is_pending": self.is_pending,
            "is_processing": self.is_processing,
            "is_cancelled": self.is_cancelled,
            "is_rolled_back": self.is_rolled_back,
            "is_timed_out": self.is_timed_out,
            "can_retry": self.can_retry,
            "is_balanced": self.is_balanced,
            "duration_seconds": self.duration_seconds,
            "total_debit": float(self.total_debit),
            "total_credit": float(self.total_credit),
            "organization_id": str(self.organization_id) if self.organization_id else None,
            "entity_type": self.entity_type,
            "entity_id": str(self.entity_id) if self.entity_id else None,
            "rollback_transaction_id": str(self.rollback_transaction_id) if self.rollback_transaction_id else None,
            "metadata": self.metadata,
            "tags": self.tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_steps:
            result["steps"] = [
                step.to_dict(include_transaction=False)
                for step in sorted(self.transaction_steps, key=lambda x: x.step_order)
            ]
        
        if include_ledger:
            result["ledger_entries"] = [
                entry.to_dict(include_transaction=False)
                for entry in self.ledger_entries
            ]
        
        if include_children and self.child_transactions:
            result["child_transactions"] = [
                child.to_dict(include_steps=False, include_ledger=False, include_children=False)
                for child in self.child_transactions
            ]
        
        if self.initiator:
            result["initiator"] = {
                "id": str(self.initiator.id),
                "username": self.initiator.username
            }
        
        if self.organization:
            result["organization"] = {
                "id": str(self.organization.id),
                "name": self.organization.name,
                "slug": self.organization.slug
            }
        
        return result
    
    @classmethod
    def create(
        cls,
        transaction_type: TransactionType,
        initiated_by: Optional[uuid.UUID] = None,
        reference_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        parent_transaction_id: Optional[uuid.UUID] = None,
        amount: Optional[Decimal] = None,
        currency: Optional[str] = None,
        description: Optional[str] = None,
        initiated_by_ip: Optional[str] = None,
        initiated_by_user_agent: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        max_retries: int = 3,
        organization_id: Optional[uuid.UUID] = None,
        entity_type: Optional[str] = None,
        entity_id: Optional[uuid.UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> 'Transaction':
        """
        Factory method to create a new transaction.
        
        Args:
            transaction_type: Type of transaction
            initiated_by: User who initiated transaction
            reference_id: External reference ID
            correlation_id: Correlation ID
            parent_transaction_id: Parent transaction ID
            amount: Transaction amount
            currency: Currency code
            description: Transaction description
            initiated_by_ip: IP address of initiator
            initiated_by_user_agent: User agent of initiator
            timeout_seconds: Transaction timeout
            max_retries: Maximum retry attempts
            organization_id: Related organization
            entity_type: Type of entity affected
            entity_id: ID of entity affected
            metadata: Additional metadata
            tags: Categorization tags
            **kwargs: Additional arguments
            
        Returns:
            A new Transaction instance
        """
        # Generate reference ID if not provided
        if not reference_id and transaction_type in [
            TransactionType.PAYMENT,
            TransactionType.REFUND,
            TransactionType.CHARGE
        ]:
            reference_id = cls._generate_reference_id()
        
        transaction = cls(
            transaction_type=transaction_type,
            status=TransactionStatus.PENDING,
            reference_id=reference_id,
            correlation_id=correlation_id,
            parent_transaction_id=parent_transaction_id,
            amount=amount,
            currency=currency,
            description=description,
            initiated_by=initiated_by,
            initiated_by_ip=initiated_by_ip,
            initiated_by_user_agent=initiated_by_user_agent,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_count=0,
            organization_id=organization_id,
            entity_type=entity_type,
            entity_id=entity_id,
            metadata=metadata or {},
            tags=tags or [],
            **kwargs
        )
        
        return transaction
    
    @staticmethod
    def _generate_reference_id() -> str:
        """Generate unique reference ID."""
        import secrets
        import base64
        
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        random_bytes = secrets.token_bytes(8)
        random_part = base64.urlsafe_b64encode(random_bytes).decode('ascii')[:12]
        return f"TX-{timestamp}-{random_part}"
    
    @classmethod
    def create_payment_transaction(
        cls,
        payment_id: uuid.UUID,
        amount: Decimal,
        currency: str,
        initiated_by: uuid.UUID,
        description: Optional[str] = None,
        **kwargs
    ) -> 'Transaction':
        """Create a payment transaction."""
        return cls.create(
            transaction_type=TransactionType.PAYMENT,
            amount=amount,
            currency=currency,
            description=description or f"Payment for invoice {payment_id}",
            initiated_by=initiated_by,
            entity_type="payment",
            entity_id=payment_id,
            tags=["payment", "financial"],
            **kwargs
        )
    
    @classmethod
    def create_subscription_transaction(
        cls,
        subscription_id: uuid.UUID,
        transaction_type: TransactionType,
        initiated_by: uuid.UUID,
        amount: Optional[Decimal] = None,
        currency: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs
    ) -> 'Transaction':
        """Create a subscription-related transaction."""
        type_descriptions = {
            TransactionType.SUBSCRIPTION_CREATE: "Create subscription",
            TransactionType.SUBSCRIPTION_UPDATE: "Update subscription",
            TransactionType.SUBSCRIPTION_CANCEL: "Cancel subscription"
        }
        
        return cls.create(
            transaction_type=transaction_type,
            amount=amount,
            currency=currency,
            description=description or type_descriptions.get(transaction_type, "Subscription transaction"),
            initiated_by=initiated_by,
            entity_type="subscription",
            entity_id=subscription_id,
            tags=["subscription", "billing"],
            **kwargs
        )


class LedgerEntry(Base, UUIDMixin, TimestampMixin):
    """
    Ledger entry model for double-entry bookkeeping.
    
    This model represents individual debit/credit entries in a
    transaction for financial tracking and audit purposes.
    
    Attributes:
        id: Primary key UUID
        transaction_id: Related transaction ID
        account_id: Account ID
        entry_type: Debit or credit entry
        amount: Entry amount
        currency: Currency code
        description: Entry description
        reference: External reference
        reversal_entry_id: Entry that reverses this one
        metadata: Additional metadata
    """
    
    __tablename__ = "ledger_entries"
    
    # Transaction and account
    transaction_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("transactions.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    account_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("ledger_accounts.id", ondelete="RESTRICT"), 
        nullable=False,
        index=True
    )
    
    # Entry details
    entry_type = Column(SQLEnum(LedgerEntryType), nullable=False, index=True)
    amount = Column(Numeric(15, 2), nullable=False)
    currency = Column(String(3), nullable=False)
    description = Column(Text, nullable=True)
    reference = Column(String(200), nullable=True, index=True)
    
    # Reversal tracking
    reversal_entry_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("ledger_entries.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    transaction = relationship("Transaction", back_populates="ledger_entries")
    account = relationship("LedgerAccount", back_populates="ledger_entries")
    reversal_entry = relationship("LedgerEntry", remote_side=[id], backref="reversed_by")
    
    # Check constraints
    __table_args__ = (
        CheckConstraint('amount > 0', name='check_amount_positive'),
        Index('ix_ledger_entries_account_date', 'account_id', 'created_at'),
        Index('ix_ledger_entries_type_amount', 'entry_type', 'amount'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<LedgerEntry(id={self.id}, account={self.account_id}, type={self.entry_type.value}, amount={self.amount})>"
    
    @property
    def is_debit(self) -> bool:
        """Check if entry is a debit."""
        return self.entry_type == LedgerEntryType.DEBIT
    
    @property
    def is_credit(self) -> bool:
        """Check if entry is a credit."""
        return self.entry_type == LedgerEntryType.CREDIT
    
    @property
    def formatted_amount(self) -> str:
        """Get formatted amount string."""
        return f"{self.currency} {self.amount:.2f}"
    
    @property
    def is_reversed(self) -> bool:
        """Check if entry has been reversed."""
        return self.reversal_entry_id is not None or len(self.reversed_by) > 0
    
    def reverse(self, reversal_entry: 'LedgerEntry') -> None:
        """Reverse this ledger entry."""
        self.reversal_entry_id = reversal_entry.id
    
    def to_dict(self, include_transaction: bool = False, include_account: bool = False) -> Dict[str, Any]:
        """Convert ledger entry to dictionary."""
        result = {
            "id": str(self.id),
            "transaction_id": str(self.transaction_id),
            "account_id": str(self.account_id),
            "entry_type": self.entry_type.value,
            "is_debit": self.is_debit,
            "is_credit": self.is_credit,
            "amount": float(self.amount),
            "currency": self.currency,
            "formatted_amount": self.formatted_amount,
            "description": self.description,
            "reference": self.reference,
            "reversal_entry_id": str(self.reversal_entry_id) if self.reversal_entry_id else None,
            "is_reversed": self.is_reversed,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_transaction and self.transaction:
            result["transaction"] = {
                "id": str(self.transaction.id),
                "transaction_type": self.transaction.transaction_type.value,
                "reference_id": self.transaction.reference_id
            }
        
        if include_account and self.account:
            result["account"] = {
                "id": str(self.account.id),
                "name": self.account.name,
                "code": self.account.code
            }
        
        return result


class LedgerAccount(Base, UUIDMixin, TimestampMixin):
    """
    Ledger account model for financial tracking.
    
    This model represents accounts in a double-entry bookkeeping system.
    
    Attributes:
        id: Primary key UUID
        code: Account code
        name: Account name
        description: Account description
        account_type: Type of account
        parent_account_id: Parent account for hierarchy
        is_active: Whether account is active
        normal_balance: Normal balance type (debit/credit)
        currency: Default currency
        opening_balance: Opening balance amount
        current_balance: Current balance
        metadata: Additional metadata
        organization_id: Owning organization
    """
    
    __tablename__ = "ledger_accounts"
    
    # Account identification
    code = Column(String(50), nullable=False, unique=True, index=True)
    name = Column(String(200), nullable=False, index=True)
    description = Column(Text, nullable=True)
    account_type = Column(String(100), nullable=False, index=True)  # asset, liability, equity, revenue, expense
    
    # Hierarchy
    parent_account_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("ledger_accounts.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    
    # Status and balances
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    normal_balance = Column(SQLEnum(LedgerEntryType), default=LedgerEntryType.DEBIT, nullable=False)
    currency = Column(String(3), default="USD", nullable=False)
    opening_balance = Column(Numeric(15, 2), default=0, nullable=False)
    current_balance = Column(Numeric(15, 2), default=0, nullable=False)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Organization
    organization_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("organizations.id", ondelete="CASCADE"), 
        nullable=True,
        index=True
    )
    
    # Relationships
    parent_account = relationship("LedgerAccount", remote_side=[id], backref="child_accounts")
    organization = relationship("Organization")
    ledger_entries = relationship("LedgerEntry", back_populates="account")
    
    # Check constraints
    __table_args__ = (
        CheckConstraint(
            'normal_balance IN (\'debit\', \'credit\')',
            name='check_normal_balance_valid'
        ),
        CheckConstraint(
            'account_type IN (\'asset\', \'liability\', \'equity\', \'revenue\', \'expense\')',
            name='check_account_type_valid'
        ),
        Index('ix_ledger_accounts_type_active', 'account_type', 'is_active'),
        Index('ix_ledger_accounts_org_code', 'organization_id', 'code'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<LedgerAccount(id={self.id}, code={self.code}, name={self.name}, type={self.account_type})>"
    
    @property
    def formatted_balance(self) -> str:
        """Get formatted balance string."""
        return f"{self.currency} {self.current_balance:.2f}"
    
    @property
    def formatted_opening_balance(self) -> str:
        """Get formatted opening balance string."""
        return f"{self.currency} {self.opening_balance:.2f}"
    
    @property
    def balance_change(self) -> Decimal:
        """Get balance change from opening balance."""
        return self.current_balance - self.opening_balance
    
    @property
    def is_debit_normal(self) -> bool:
        """Check if normal balance is debit."""
        return self.normal_balance == LedgerEntryType.DEBIT
    
    @property
    def is_credit_normal(self) -> bool:
        """Check if normal balance is credit."""
        return self.normal_balance == LedgerEntryType.CREDIT
    
    @property
    def child_count(self) -> int:
        """Get number of child accounts."""
        return len(self.child_accounts)
    
    @property
    def full_code_path(self) -> str:
        """Get full account code path in hierarchy."""
        if not self.parent_account:
            return self.code
        return f"{self.parent_account.full_code_path}.{self.code}"
    
    def update_balance(self, entry_type: LedgerEntryType, amount: Decimal) -> None:
        """Update account balance based on ledger entry."""
        if entry_type == self.normal_balance:
            # Increase balance for normal balance entries
            self.current_balance += amount
        else:
            # Decrease balance for opposite entries
            self.current_balance -= amount
    
    def get_balance_as_of(self, as_of_date: datetime) -> Decimal:
        """Get account balance as of specific date."""
        # This would typically query ledger entries
        # For now, return current balance
        return self.current_balance
    
    def to_dict(self, include_children: bool = False, include_parent: bool = False) -> Dict[str, Any]:
        """Convert ledger account to dictionary."""
        result = {
            "id": str(self.id),
            "code": self.code,
            "name": self.name,
            "description": self.description,
            "account_type": self.account_type,
            "parent_account_id": str(self.parent_account_id) if self.parent_account_id else None,
            "full_code_path": self.full_code_path,
            "is_active": self.is_active,
            "normal_balance": self.normal_balance.value,
            "is_debit_normal": self.is_debit_normal,
            "is_credit_normal": self.is_credit_normal,
            "currency": self.currency,
            "opening_balance": float(self.opening_balance),
            "current_balance": float(self.current_balance),
            "formatted_balance": self.formatted_balance,
            "formatted_opening_balance": self.formatted_opening_balance,
            "balance_change": float(self.balance_change),
            "child_count": self.child_count,
            "organization_id": str(self.organization_id) if self.organization_id else None,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_children and self.child_accounts:
            result["children"] = [
                child.to_dict(include_children=False, include_parent=False)
                for child in self.child_accounts
                if child.is_active
            ]
        
        if include_parent and self.parent_account:
            result["parent"] = {
                "id": str(self.parent_account.id),
                "code": self.parent_account.code,
                "name": self.parent_account.name
            }
        
        if self.organization:
            result["organization"] = {
                "id": str(self.organization.id),
                "name": self.organization.name
            }
        
        return result


class TransactionStep(Base, UUIDMixin, TimestampMixin):
    """
    Transaction step model for multi-step transactions.
    
    This model represents individual steps in a transaction workflow,
    enabling complex, multi-step operations with rollback capabilities.
    
    Attributes:
        id: Primary key UUID
        transaction_id: Related transaction ID
        step_name: Step name
        step_order: Step execution order
        status: Step status
        handler_class: Handler class for execution
        handler_method: Handler method for execution
        parameters: Step parameters
        started_at: When step started
        completed_at: When step completed
        error_message: Error message if failed
        rollback_data: Data for rollback
        metadata: Additional metadata
    """
    
    __tablename__ = "transaction_steps"
    
    # Transaction
    transaction_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("transactions.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # Step information
    step_name = Column(String(200), nullable=False, index=True)
    step_order = Column(Integer, nullable=False)
    status = Column(String(50), default="pending", nullable=False, index=True)
    
    # Execution details
    handler_class = Column(String(200), nullable=True)
    handler_method = Column(String(100), nullable=True)
    parameters = Column(JSONB, default=dict, nullable=False)
    
    # Execution timing
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    error_details = Column(JSONB, nullable=True)
    
    # Rollback
    rollback_data = Column(JSONB, nullable=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    transaction = relationship("Transaction", back_populates="transaction_steps")
    
    # Check constraints
    __table_args__ = (
        UniqueConstraint('transaction_id', 'step_order', name='uq_transaction_step_order'),
        UniqueConstraint('transaction_id', 'step_name', name='uq_transaction_step_name'),
        CheckConstraint('step_order >= 0', name='check_step_order_non_negative'),
        Index('ix_transaction_steps_status', 'transaction_id', 'status'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<TransactionStep(id={self.id}, transaction={self.transaction_id}, step={self.step_name}, order={self.step_order})>"
    
    @property
    def is_completed(self) -> bool:
        """Check if step is completed."""
        return self.status == "completed"
    
    @property
    def is_failed(self) -> bool:
        """Check if step failed."""
        return self.status == "failed"
    
    @property
    def is_pending(self) -> bool:
        """Check if step is pending."""
        return self.status == "pending"
    
    @property
    def is_processing(self) -> bool:
        """Check if step is processing."""
        return self.status == "processing"
    
    @property
    def is_rolled_back(self) -> bool:
        """Check if step was rolled back."""
        return self.status == "rolled_back"
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get step duration in seconds."""
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def start(self) -> None:
        """Start the step."""
        self.status = "processing"
        self.started_at = datetime.utcnow()
    
    def complete(self, rollback_data: Optional[Dict[str, Any]] = None) -> None:
        """Complete the step."""
        self.status = "completed"
        self.completed_at = datetime.utcnow()
        if rollback_data:
            self.rollback_data = rollback_data
    
    def fail(self, error_message: str, error_details: Optional[Dict[str, Any]] = None) -> None:
        """Mark step as failed."""
        self.status = "failed"
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
        self.error_details = error_details
    
    def rollback(self) -> None:
        """Mark step as rolled back."""
        self.status = "rolled_back"
    
    def to_dict(self, include_transaction: bool = False) -> Dict[str, Any]:
        """Convert transaction step to dictionary."""
        result = {
            "id": str(self.id),
            "transaction_id": str(self.transaction_id),
            "step_name": self.step_name,
            "step_order": self.step_order,
            "status": self.status,
            "handler_class": self.handler_class,
            "handler_method": self.handler_method,
            "parameters": self.parameters,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "error_details": self.error_details,
            "rollback_data": self.rollback_data,
            "is_completed": self.is_completed,
            "is_failed": self.is_failed,
            "is_pending": self.is_pending,
            "is_processing": self.is_processing,
            "is_rolled_back": self.is_rolled_back,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_transaction and self.transaction:
            result["transaction"] = {
                "id": str(self.transaction.id),
                "transaction_type": self.transaction.transaction_type.value,
                "status": self.transaction.status.value
            }
        
        return result


class AuditLog(Base, UUIDMixin, TimestampMixin):
    """
    Audit log model for tracking system changes and user actions.
    
    This model provides comprehensive audit trails for compliance,
    security, and debugging purposes.
    
    Attributes:
        id: Primary key UUID
        transaction_id: Related transaction ID
        user_id: User who performed action
        organization_id: Organization context
        action: Audit action
        resource_type: Type of resource affected
        resource_id: ID of resource affected
        resource_name: Name of resource
        old_values: Previous values before change
        new_values: New values after change
        changes: Summary of changes
        ip_address: IP address of requester
        user_agent: User agent string
        request_id: Request correlation ID
        metadata: Additional metadata
    """
    
    __tablename__ = "audit_logs"
    
    # Context
    transaction_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("transactions.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    user_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    organization_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("organizations.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    
    # Action details
    action = Column(SQLEnum(AuditAction), nullable=False, index=True)
    resource_type = Column(String(100), nullable=False, index=True)
    resource_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    resource_name = Column(String(500), nullable=True)
    
    # Change tracking
    old_values = Column(JSONB, nullable=True)
    new_values = Column(JSONB, nullable=True)
    changes = Column(JSONB, nullable=True)
    
    # Request details
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(Text, nullable=True)
    request_id = Column(String(200), nullable=True, index=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    transaction = relationship("Transaction", back_populates="audit_logs")
    user = relationship("User")
    organization = relationship("Organization")
    
    # Check constraints
    __table_args__ = (
        Index('ix_audit_logs_resource', 'resource_type', 'resource_id'),
        Index('ix_audit_logs_action_date', 'action', 'created_at'),
        Index('ix_audit_logs_user_date', 'user_id', 'created_at'),
        Index('ix_audit_logs_org_date', 'organization_id', 'created_at'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<AuditLog(id={self.id}, action={self.action.value}, resource={self.resource_type}:{self.resource_id})>"
    
    @property
    def composite_resource_id(self) -> str:
        """Get composite resource identifier."""
        return f"{self.resource_type}:{self.resource_id}"
    
    @property
    def has_changes(self) -> bool:
        """Check if audit log contains changes."""
        return bool(self.old_values) or bool(self.new_values) or bool(self.changes)
    
    @property
    def change_count(self) -> int:
        """Get number of changes."""
        if self.changes and isinstance(self.changes, dict):
            return len(self.changes)
        return 0
    
    @classmethod
    def log(
        cls,
        action: AuditAction,
        resource_type: str,
        resource_id: uuid.UUID,
        user_id: Optional[uuid.UUID] = None,
        organization_id: Optional[uuid.UUID] = None,
        transaction_id: Optional[uuid.UUID] = None,
        old_values: Optional[Dict[str, Any]] = None,
        new_values: Optional[Dict[str, Any]] = None,
        changes: Optional[Dict[str, Any]] = None,
        resource_name: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'AuditLog':
        """Create an audit log entry."""
        # Calculate changes if not provided
        if not changes and old_values and new_values:
            changes = cls._calculate_changes(old_values, new_values)
        
        return cls(
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_name=resource_name,
            user_id=user_id,
            organization_id=organization_id,
            transaction_id=transaction_id,
            old_values=old_values,
            new_values=new_values,
            changes=changes,
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id,
            metadata=metadata or {}
        )
    
    @staticmethod
    def _calculate_changes(
        old_values: Dict[str, Any],
        new_values: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate changes between old and new values."""
        changes = {}
        all_keys = set(old_values.keys()) | set(new_values.keys())
        
        for key in all_keys:
            old_val = old_values.get(key)
            new_val = new_values.get(key)
            
            if old_val != new_val:
                changes[key] = {
                    "old": old_val,
                    "new": new_val,
                    "changed": True
                }
        
        return changes
    
    def to_dict(self, include_user: bool = False, include_transaction: bool = False) -> Dict[str, Any]:
        """Convert audit log to dictionary."""
        result = {
            "id": str(self.id),
            "transaction_id": str(self.transaction_id) if self.transaction_id else None,
            "user_id": str(self.user_id) if self.user_id else None,
            "organization_id": str(self.organization_id) if self.organization_id else None,
            "action": self.action.value,
            "resource_type": self.resource_type,
            "resource_id": str(self.resource_id),
            "composite_resource_id": self.composite_resource_id,
            "resource_name": self.resource_name,
            "old_values": self.old_values,
            "new_values": self.new_values,
            "changes": self.changes,
            "has_changes": self.has_changes,
            "change_count": self.change_count,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "request_id": self.request_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_user and self.user:
            result["user"] = {
                "id": str(self.user.id),
                "username": self.user.username
            }
        
        if include_transaction and self.transaction:
            result["transaction"] = {
                "id": str(self.transaction.id),
                "transaction_type": self.transaction.transaction_type.value,
                "reference_id": self.transaction.reference_id
            }
        
        if self.organization:
            result["organization"] = {
                "id": str(self.organization.id),
                "name": self.organization.name
            }
        
        return result


class DataVersion(Base, UUIDMixin, TimestampMixin):
    """
    Data version model for version control and change tracking.
    
    This model enables versioning of data entities with full
    change history and rollback capabilities.
    
    Attributes:
        id: Primary key UUID
        entity_type: Type of entity
        entity_id: ID of entity
        version_number: Version number
        previous_version_id: Previous version ID
        data: Version data
        changes: Changes from previous version
        created_by: User who created version
        transaction_id: Related transaction
        is_current: Whether this is current version
        metadata: Additional metadata
    """
    
    __tablename__ = "data_versions"
    
    # Entity identification
    entity_type = Column(String(100), nullable=False, index=True)
    entity_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    version_number = Column(Integer, nullable=False)
    
    # Version chain
    previous_version_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("data_versions.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    
    # Version data
    data = Column(JSONB, nullable=False)
    changes = Column(JSONB, nullable=True)
    
    # Creation context
    created_by = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    transaction_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("transactions.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    
    # Status
    is_current = Column(Boolean, default=True, nullable=False, index=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    previous_version = relationship("DataVersion", remote_side=[id], backref="next_versions")
    creator = relationship("User", foreign_keys=[created_by])
    transaction = relationship("Transaction")
    
    # Check constraints
    __table_args__ = (
        UniqueConstraint('entity_type', 'entity_id', 'version_number', name='uq_entity_version'),
        CheckConstraint('version_number >= 1', name='check_version_number_minimum'),
        Index('ix_data_versions_entity_current', 'entity_type', 'entity_id', 'is_current'),
        Index('ix_data_versions_created_by', 'created_by', 'created_at'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<DataVersion(id={self.id}, entity={self.entity_type}:{self.entity_id}, version={self.version_number})>"
    
    @property
    def composite_entity_id(self) -> str:
        """Get composite entity identifier."""
        return f"{self.entity_type}:{self.entity_id}"
    
    @property
    def has_previous_version(self) -> bool:
        """Check if version has previous version."""
        return self.previous_version_id is not None
    
    @property
    def has_changes(self) -> bool:
        """Check if version has changes."""
        return bool(self.changes)
    
    @property
    def change_count(self) -> int:
        """Get number of changes."""
        if self.changes and isinstance(self.changes, dict):
            return len(self.changes)
        return 0
    
    def get_version_diff(self, other_version: 'DataVersion') -> Dict[str, Any]:
        """Get differences between this version and another version."""
        # This is a simplified diff implementation
        # In production, use a proper diff library
        diff = {}
        
        if not isinstance(self.data, dict) or not isinstance(other_version.data, dict):
            return diff
        
        all_keys = set(self.data.keys()) | set(other_version.data.keys())
        
        for key in all_keys:
            val1 = self.data.get(key)
            val2 = other_version.data.get(key)
            
            if val1 != val2:
                diff[key] = {
                    "from": val2,
                    "to": val1,
                    "changed": True
                }
        
        return diff
    
    def restore_as_current(self, restored_by: Optional[uuid.UUID] = None) -> 'DataVersion':
        """Create a new version that restores this version as current."""
        # Mark all current versions for this entity as not current
        # In practice, this would be done via query
        
        # Create new version with this version's data
        new_version = DataVersion(
            entity_type=self.entity_type,
            entity_id=self.entity_id,
            version_number=self.version_number + 1,  # Would need to get next version
            previous_version_id=self.id,
            data=self.data,
            changes={"restored_from_version": self.version_number},
            created_by=restored_by,
            is_current=True,
            metadata={
                "restored_at": datetime.utcnow().isoformat(),
                "restored_from": str(self.id)
            }
        )
        
        return new_version
    
    def to_dict(self, include_creator: bool = False, include_transaction: bool = False) -> Dict[str, Any]:
        """Convert data version to dictionary."""
        result = {
            "id": str(self.id),
            "entity_type": self.entity_type,
            "entity_id": str(self.entity_id),
            "composite_entity_id": self.composite_entity_id,
            "version_number": self.version_number,
            "previous_version_id": str(self.previous_version_id) if self.previous_version_id else None,
            "has_previous_version": self.has_previous_version,
            "data": self.data,
            "changes": self.changes,
            "has_changes": self.has_changes,
            "change_count": self.change_count,
            "created_by": str(self.created_by) if self.created_by else None,
            "transaction_id": str(self.transaction_id) if self.transaction_id else None,
            "is_current": self.is_current,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_creator and self.creator:
            result["creator"] = {
                "id": str(self.creator.id),
                "username": self.creator.username
            }
        
        if include_transaction and self.transaction:
            result["transaction"] = {
                "id": str(self.transaction.id),
                "transaction_type": self.transaction.transaction_type.value
            }
        
        return result


class TransactionLock(Base, UUIDMixin, TimestampMixin):
    """
    Transaction lock model for distributed locking.
    
    This model provides distributed locking to prevent concurrent
    modifications to the same resources.
    
    Attributes:
        id: Primary key UUID
        lock_key: Lock key (resource identifier)
        transaction_id: Transaction holding lock
        acquired_at: When lock was acquired
        expires_at: When lock expires
        lock_owner: Owner of lock (process/node identifier)
        metadata: Additional metadata
    """
    
    __tablename__ = "transaction_locks"
    
    # Lock identification
    lock_key = Column(String(500), nullable=False, unique=True, index=True)
    transaction_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("transactions.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # Lock timing
    acquired_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Lock owner
    lock_owner = Column(String(200), nullable=False)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    transaction = relationship("Transaction")
    
    # Check constraints
    __table_args__ = (
        CheckConstraint('expires_at > acquired_at', name='check_lock_expiry_valid'),
        Index('ix_transaction_locks_expiry', 'expires_at'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<TransactionLock(id={self.id}, lock_key={self.lock_key}, transaction={self.transaction_id})>"
    
    @property
    def is_expired(self) -> bool:
        """Check if lock has expired."""
        return datetime.utcnow() > self.expires_at
    
    @property
    def seconds_until_expiry(self) -> float:
        """Get seconds until lock expiry."""
        remaining = self.expires_at - datetime.utcnow()
        return max(0, remaining.total_seconds())
    
    @property
    def lock_duration_seconds(self) -> float:
        """Get lock duration in seconds."""
        duration = datetime.utcnow() - self.acquired_at
        return duration.total_seconds()
    
    def renew(self, additional_seconds: int = 300) -> None:
        """Renew the lock."""
        self.expires_at = datetime.utcnow() + timedelta(seconds=additional_seconds)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction lock to dictionary."""
        return {
            "id": str(self.id),
            "lock_key": self.lock_key,
            "transaction_id": str(self.transaction_id),
            "acquired_at": self.acquired_at.isoformat() if self.acquired_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "lock_owner": self.lock_owner,
            "is_expired": self.is_expired,
            "seconds_until_expiry": self.seconds_until_expiry,
            "lock_duration_seconds": self.lock_duration_seconds,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


# Helper functions
def begin_transaction(
    transaction_type: TransactionType,
    initiated_by: Optional[uuid.UUID] = None,
    description: Optional[str] = None,
    timeout_seconds: int = 300,
    **kwargs
) -> Transaction:
    """
    Begin a new transaction.
    
    Args:
        transaction_type: Type of transaction
        initiated_by: User who initiated transaction
        description: Transaction description
        timeout_seconds: Transaction timeout
        **kwargs: Additional arguments for Transaction.create
        
    Returns:
        New Transaction instance
    """
    return Transaction.create(
        transaction_type=transaction_type,
        initiated_by=initiated_by,
        description=description,
        timeout_seconds=timeout_seconds,
        **kwargs
    )


def commit_transaction(transaction: Transaction) -> None:
    """Commit a transaction."""
    # Validate transaction can be committed
    if transaction.is_completed or transaction.is_cancelled:
        raise ValueError(f"Transaction {transaction.id} cannot be committed in state {transaction.status}")
    
    # Check if transaction is balanced (for financial transactions)
    if transaction.amount is not None and not transaction.is_balanced:
        raise ValueError(f"Transaction {transaction.id} is not balanced")
    
    # Complete the transaction
    transaction.complete()


def rollback_transaction(
    transaction: Transaction,
    rollback_reason: Optional[str] = None,
    create_rollback_transaction: bool = True
) -> Transaction:
    """
    Rollback a transaction.
    
    Args:
        transaction: Transaction to rollback
        rollback_reason: Reason for rollback
        create_rollback_transaction: Whether to create a rollback transaction
        
    Returns:
        Rollback transaction if created, otherwise original transaction
    """
    if transaction.is_completed:
        # Create a rollback transaction
        if create_rollback_transaction:
            rollback_tx = Transaction.create(
                transaction_type=TransactionType.SYSTEM_ACTION,
                description=f"Rollback of transaction {transaction.reference_id}",
                correlation_id=transaction.correlation_id,
                parent_transaction_id=transaction.id,
                organization_id=transaction.organization_id,
                metadata={
                    "rollback_of": str(transaction.id),
                    "rollback_reason": rollback_reason,
                    "original_transaction_type": transaction.transaction_type.value
                },
                tags=["rollback", "recovery"]
            )
            
            # Mark original transaction as rolled back
            transaction.rollback(rollback_transaction_id=rollback_tx.id)
            
            # Process rollback steps
            for step in reversed(transaction.transaction_steps):
                if step.is_completed and step.rollback_data:
                    # Execute rollback for each step
                    step.rollback()
            
            # Complete rollback transaction
            rollback_tx.complete()
            
            return rollback_tx
        else:
            # Just mark as rolled back
            transaction.rollback()
            return transaction
    else:
        # Transaction not completed, just cancel it
        transaction.cancel(reason=rollback_reason)
        return transaction


def acquire_lock(
    lock_key: str,
    transaction_id: uuid.UUID,
    lock_owner: str,
    lock_timeout_seconds: int = 300
) -> TransactionLock:
    """
    Acquire a distributed lock.
    
    Args:
        lock_key: Lock key
        transaction_id: Transaction ID
        lock_owner: Lock owner identifier
        lock_timeout_seconds: Lock timeout in seconds
        
    Returns:
        TransactionLock instance
    """
    # In production, this would check for existing locks and handle conflicts
    lock = TransactionLock(
        lock_key=lock_key,
        transaction_id=transaction_id,
        lock_owner=lock_owner,
        acquired_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(seconds=lock_timeout_seconds)
    )
    
    return lock


def record_audit_log(
    action: AuditAction,
    resource_type: str,
    resource_id: uuid.UUID,
    user_id: Optional[uuid.UUID] = None,
    old_values: Optional[Dict[str, Any]] = None,
    new_values: Optional[Dict[str, Any]] = None,
    **kwargs
) -> AuditLog:
    """
    Record an audit log entry.
    
    Args:
        action: Audit action
        resource_type: Resource type
        resource_id: Resource ID
        user_id: User ID
        old_values: Old values
        new_values: New values
        **kwargs: Additional arguments for AuditLog.log
        
    Returns:
        AuditLog instance
    """
    return AuditLog.log(
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        user_id=user_id,
        old_values=old_values,
        new_values=new_values,
        **kwargs
    )


def create_data_version(
    entity_type: str,
    entity_id: uuid.UUID,
    data: Dict[str, Any],
    created_by: Optional[uuid.UUID] = None,
    transaction_id: Optional[uuid.UUID] = None,
    changes: Optional[Dict[str, Any]] = None
) -> DataVersion:
    """
    Create a new data version.
    
    Args:
        entity_type: Entity type
        entity_id: Entity ID
        data: Version data
        created_by: User who created version
        transaction_id: Related transaction
        changes: Changes from previous version
        
    Returns:
        DataVersion instance
    """
    # In production, would query for current version and increment
    version_number = 1  # Would need to query for next version number
    
    version = DataVersion(
        entity_type=entity_type,
        entity_id=entity_id,
        version_number=version_number,
        data=data,
        changes=changes,
        created_by=created_by,
        transaction_id=transaction_id,
        is_current=True,
        metadata={"created_via": "api"}
    )
    
    return version