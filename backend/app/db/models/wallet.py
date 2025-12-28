"""
wallet.py - Digital Wallet and Cryptocurrency Model

This module defines models for digital wallets, cryptocurrency transactions,
and payment systems. This includes:
- User wallet management
- Cryptocurrency and token support
- Transaction processing and tracking
- Smart contract integration
- Payment gateway integration
- Wallet security and recovery
- Multi-currency support
- Staking and rewards

Key Features:
- Multi-currency wallet support
- Secure transaction processing
- Smart contract integration
- Wallet backup and recovery
- Transaction history and audit trail
- Gas fee management
- Token staking and rewards
- Payment gateway abstraction
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


class WalletType(Enum):
    """Types of wallets."""
    USER = "user"                    # User personal wallet
    ORGANIZATION = "organization"    # Organization wallet
    ESCROW = "escrow"                # Escrow wallet
    EXCHANGE = "exchange"            # Exchange wallet
    SMART_CONTRACT = "smart_contract"  # Smart contract wallet
    SYSTEM = "system"                # System wallet
    MERCHANT = "merchant"            # Merchant wallet
    CUSTODIAL = "custodial"          # Custodial wallet
    NON_CUSTODIAL = "non_custodial"  # Non-custodial wallet
    HARDWARE = "hardware"            # Hardware wallet
    MULTISIG = "multisig"            # Multi-signature wallet


class CurrencyType(Enum):
    """Types of currencies."""
    FIAT = "fiat"                    # Traditional fiat currency
    CRYPTO = "crypto"                # Cryptocurrency
    TOKEN = "token"                  # Token (ERC-20, BEP-20, etc.)
    STABLE_COIN = "stable_coin"      # Stablecoin
    NFT = "nft"                      # Non-fungible token
    REWARD_POINT = "reward_point"    # Reward points
    GIFT_CARD = "gift_card"          # Gift card
    VIRTUAL_CURRENCY = "virtual_currency"  # Virtual currency
    OTHER = "other"                  # Other currency type


class TransactionType(Enum):
    """Types of transactions."""
    DEPOSIT = "deposit"              # Deposit funds
    WITHDRAWAL = "withdrawal"        # Withdraw funds
    TRANSFER = "transfer"            # Transfer between wallets
    PAYMENT = "payment"              # Payment for goods/services
    REFUND = "refund"                # Refund payment
    EXCHANGE = "exchange"            # Currency exchange
    STAKE = "stake"                  # Stake tokens
    UNSTAKE = "unstake"              # Unstake tokens
    REWARD = "reward"                # Reward distribution
    AIRDROP = "airdrop"              # Airdrop distribution
    BURN = "burn"                    # Token burn
    MINT = "mint"                    # Token mint
    FEE = "fee"                      # Transaction fee
    GAS = "gas"                      # Gas fee
    OTHER = "other"                  # Other transaction type


class TransactionStatus(Enum):
    """Transaction status."""
    PENDING = "pending"              # Transaction pending
    PROCESSING = "processing"        # Transaction processing
    COMPLETED = "completed"          # Transaction completed
    FAILED = "failed"                # Transaction failed
    CANCELLED = "cancelled"          # Transaction cancelled
    CONFIRMING = "confirming"        # Blockchain confirmations
    REVERSED = "reversed"            # Transaction reversed
    EXPIRED = "expired"              # Transaction expired
    REFUNDED = "refunded"            # Transaction refunded


class BlockchainNetwork(Enum):
    """Blockchain networks."""
    ETHEREUM = "ethereum"            # Ethereum Mainnet
    ETHEREUM_GOERLI = "ethereum_goerli"  # Ethereum Goerli Testnet
    ETHEREUM_SEPOLIA = "ethereum_sepolia"  # Ethereum Sepolia Testnet
    BINANCE_SMART_CHAIN = "binance_smart_chain"  # BSC Mainnet
    BSC_TESTNET = "bsc_testnet"      # BSC Testnet
    POLYGON = "polygon"              # Polygon Mainnet
    POLYGON_MUMBAI = "polygon_mumbai"  # Polygon Mumbai Testnet
    ARBITRUM = "arbitrum"            # Arbitrum Mainnet
    OPTIMISM = "optimism"            # Optimism Mainnet
    AVALANCHE = "avalanche"          # Avalanche Mainnet
    SOLANA = "solana"                # Solana Mainnet
    BITCOIN = "bitcoin"              # Bitcoin Mainnet
    BITCOIN_TESTNET = "bitcoin_testnet"  # Bitcoin Testnet
    CUSTOM = "custom"                # Custom blockchain


class WalletSecurityLevel(Enum):
    """Wallet security levels."""
    LOW = "low"                      # Basic security
    MEDIUM = "medium"                # Medium security (2FA)
    HIGH = "high"                    # High security (multi-sig, hardware)
    ENTERPRISE = "enterprise"        # Enterprise security


class Currency(Base, UUIDMixin, TimestampMixin):
    """
    Currency model for supported cryptocurrencies and tokens.
    
    This model defines supported currencies, tokens, and their properties.
    
    Attributes:
        id: Primary key UUID
        symbol: Currency symbol (e.g., BTC, ETH, USDT)
        name: Currency name (e.g., Bitcoin, Ethereum, Tether)
        currency_type: Type of currency
        decimals: Number of decimal places
        is_active: Whether currency is active
        is_stablecoin: Whether currency is a stablecoin
        contract_address: Smart contract address (for tokens)
        blockchain_network: Blockchain network
        icon_url: URL to currency icon
        color: Currency color code
        coingecko_id: CoinGecko API ID
        cmc_id: CoinMarketCap ID
        metadata: Additional metadata
    """
    
    __tablename__ = "currencies"
    
    # Currency information
    symbol = Column(String(20), nullable=False, unique=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    currency_type = Column(SQLEnum(CurrencyType), nullable=False, index=True)
    decimals = Column(Integer, default=18, nullable=False)  # Default for Ethereum tokens
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    is_stablecoin = Column(Boolean, default=False, nullable=False, index=True)
    is_default = Column(Boolean, default=False, nullable=False, index=True)
    
    # Blockchain information
    contract_address = Column(String(255), nullable=True, index=True)
    blockchain_network = Column(SQLEnum(BlockchainNetwork), nullable=True, index=True)
    
    # Visual representation
    icon_url = Column(String(2000), nullable=True)
    color = Column(String(7), nullable=True)  # Hex color
    
    # External API IDs
    coingecko_id = Column(String(100), nullable=True, index=True)
    cmc_id = Column(String(100), nullable=True, index=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    wallets = relationship("WalletBalance", back_populates="currency")
    transactions = relationship("Transaction", back_populates="currency")
    
    # Check constraints
    __table_args__ = (
        CheckConstraint('decimals >= 0 AND decimals <= 18', name='check_decimals_range'),
        CheckConstraint(
            'currency_type != \'token\' OR contract_address IS NOT NULL',
            name='check_token_contract_address'
        ),
        Index('ix_currencies_type_active', 'currency_type', 'is_active'),
        Index('ix_currencies_network', 'blockchain_network', 'is_active'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Currency(id={self.id}, symbol={self.symbol}, name={self.name})>"
    
    @property
    def is_crypto(self) -> bool:
        """Check if currency is cryptocurrency."""
        return self.currency_type in [CurrencyType.CRYPTO, CurrencyType.TOKEN, CurrencyType.STABLE_COIN]
    
    @property
    def is_fiat(self) -> bool:
        """Check if currency is fiat."""
        return self.currency_type == CurrencyType.FIAT
    
    @property
    def is_token(self) -> bool:
        """Check if currency is a token."""
        return self.currency_type == CurrencyType.TOKEN
    
    @property
    def display_decimals(self) -> int:
        """Get display decimals based on currency type."""
        if self.is_fiat:
            return 2
        elif self.is_crypto:
            if self.symbol == "BTC":
                return 8
            elif self.symbol == "ETH":
                return 4
            else:
                return 6
        return self.decimals
    
    def to_unit_amount(self, amount: Decimal) -> Decimal:
        """Convert human-readable amount to smallest unit."""
        return amount * (Decimal(10) ** self.decimals)
    
    def from_unit_amount(self, unit_amount: Decimal) -> Decimal:
        """Convert smallest unit to human-readable amount."""
        return unit_amount / (Decimal(10) ** self.decimals)
    
    def formatted_amount(self, amount: Decimal) -> str:
        """Format amount with proper decimals."""
        return f"{amount:.{self.display_decimals}f} {self.symbol}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert currency to dictionary."""
        return {
            "id": str(self.id),
            "symbol": self.symbol,
            "name": self.name,
            "currency_type": self.currency_type.value,
            "decimals": self.decimals,
            "display_decimals": self.display_decimals,
            "is_active": self.is_active,
            "is_stablecoin": self.is_stablecoin,
            "is_default": self.is_default,
            "is_crypto": self.is_crypto,
            "is_fiat": self.is_fiat,
            "is_token": self.is_token,
            "contract_address": self.contract_address,
            "blockchain_network": self.blockchain_network.value if self.blockchain_network else None,
            "icon_url": self.icon_url,
            "color": self.color,
            "coingecko_id": self.coingecko_id,
            "cmc_id": self.cmc_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class Wallet(Base, UUIDMixin, TimestampMixin):
    """
    Digital wallet model.
    
    This model represents user or organization digital wallets
    for storing cryptocurrencies and tokens.
    
    Attributes:
        id: Primary key UUID
        user_id: User who owns wallet
        organization_id: Organization that owns wallet
        wallet_type: Type of wallet
        security_level: Wallet security level
        address: Blockchain wallet address
        public_key: Wallet public key
        encrypted_private_key: Encrypted private key (for custodial wallets)
        mnemonic_hash: Hashed mnemonic phrase
        is_active: Whether wallet is active
        is_verified: Whether wallet is verified
        is_custodial: Whether wallet is custodial
        label: Wallet label/name
        description: Wallet description
        metadata: Additional metadata
        tags: Categorization tags
    """
    
    __tablename__ = "wallets"
    
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
    
    # Wallet type and security
    wallet_type = Column(SQLEnum(WalletType), nullable=False, index=True)
    security_level = Column(SQLEnum(WalletSecurityLevel), default=WalletSecurityLevel.MEDIUM, nullable=False)
    
    # Blockchain information
    address = Column(String(255), nullable=False, unique=True, index=True)
    public_key = Column(Text, nullable=True)
    encrypted_private_key = Column(Text, nullable=True)  # Only for custodial wallets
    mnemonic_hash = Column(String(255), nullable=True)  # Hashed mnemonic phrase
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    is_verified = Column(Boolean, default=False, nullable=False, index=True)
    is_custodial = Column(Boolean, default=True, nullable=False, index=True)
    
    # Labels
    label = Column(String(200), nullable=True, index=True)
    description = Column(Text, nullable=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    tags = Column(ARRAY(String), default=[], nullable=False, index=True)
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id])
    organization = relationship("Organization", foreign_keys=[organization_id])
    balances = relationship("WalletBalance", back_populates="wallet", cascade="all, delete-orphan")
    sent_transactions = relationship("Transaction", foreign_keys="Transaction.from_wallet_id", back_populates="from_wallet")
    received_transactions = relationship("Transaction", foreign_keys="Transaction.to_wallet_id", back_populates="to_wallet")
    signatures = relationship("WalletSignature", back_populates="wallet", cascade="all, delete-orphan")
    backup_keys = relationship("WalletBackupKey", back_populates="wallet", cascade="all, delete-orphan")
    
    # Check constraints
    __table_args__ = (
        CheckConstraint(
            'user_id IS NOT NULL OR organization_id IS NOT NULL',
            name='check_wallet_owner_exists'
        ),
        CheckConstraint(
            'NOT (user_id IS NOT NULL AND organization_id IS NOT NULL)',
            name='check_single_wallet_owner'
        ),
        Index('ix_wallets_owner_type', 'user_id', 'organization_id', 'wallet_type'),
        Index('ix_wallets_active_verified', 'is_active', 'is_verified'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        owner = f"user={self.user_id}" if self.user_id else f"org={self.organization_id}"
        return f"<Wallet(id={self.id}, {owner}, address={self.address[:10]}..., type={self.wallet_type.value})>"
    
    @property
    def owner_id(self) -> uuid.UUID:
        """Get owner ID (user or organization)."""
        return self.user_id or self.organization_id
    
    @property
    def owner_type(self) -> str:
        """Get owner type."""
        return "user" if self.user_id else "organization"
    
    @property
    def is_non_custodial(self) -> bool:
        """Check if wallet is non-custodial."""
        return not self.is_custodial
    
    @property
    def is_hardware_wallet(self) -> bool:
        """Check if wallet is hardware wallet."""
        return self.wallet_type == WalletType.HARDWARE
    
    @property
    def is_multisig(self) -> bool:
        """Check if wallet is multi-signature."""
        return self.wallet_type == WalletType.MULTISIG
    
    @property
    def short_address(self) -> str:
        """Get shortened wallet address."""
        if len(self.address) <= 16:
            return self.address
        return f"{self.address[:8]}...{self.address[-8:]}"
    
    @property
    def total_balance(self) -> Dict[str, Decimal]:
        """Get total balance across all currencies."""
        total = {}
        for balance in self.balances:
            if balance.is_active:
                total[balance.currency.symbol] = balance.balance
        return total
    
    @property
    def total_value_usd(self) -> Optional[Decimal]:
        """Get total wallet value in USD."""
        # In production, you would fetch current prices from an API
        # This is a simplified version
        total = Decimal('0')
        for balance in self.balances:
            if balance.is_active and balance.currency.symbol in ["BTC", "ETH", "USDT"]:
                # Mock prices for example
                prices = {
                    "BTC": Decimal('50000'),
                    "ETH": Decimal('3000'),
                    "USDT": Decimal('1')
                }
                price = prices.get(balance.currency.symbol, Decimal('0'))
                total += balance.balance * price
        return total if total > 0 else None
    
    def get_balance(self, currency_symbol: str) -> Optional[Decimal]:
        """Get balance for specific currency."""
        for balance in self.balances:
            if balance.currency.symbol == currency_symbol and balance.is_active:
                return balance.balance
        return None
    
    def add_balance(
        self,
        currency_id: uuid.UUID,
        initial_balance: Decimal = Decimal('0'),
        is_active: bool = True
    ) -> 'WalletBalance':
        """Add a currency balance to wallet."""
        from models.wallet import WalletBalance
        
        balance = WalletBalance(
            wallet_id=self.id,
            currency_id=currency_id,
            balance=initial_balance,
            is_active=is_active
        )
        self.balances.append(balance)
        return balance
    
    def update_balance(
        self,
        currency_symbol: str,
        amount: Decimal,
        transaction_type: TransactionType
    ) -> bool:
        """Update wallet balance."""
        for balance in self.balances:
            if balance.currency.symbol == currency_symbol:
                if transaction_type in [TransactionType.DEPOSIT, TransactionType.TRANSFER, TransactionType.REWARD]:
                    balance.balance += amount
                elif transaction_type in [TransactionType.WITHDRAWAL, TransactionType.PAYMENT, TransactionType.FEE]:
                    if balance.balance >= amount:
                        balance.balance -= amount
                    else:
                        return False  # Insufficient balance
                elif transaction_type == TransactionType.EXCHANGE:
                    # Handle exchange logic
                    pass
                
                balance.last_updated = datetime.utcnow()
                return True
        return False
    
    def verify(self) -> None:
        """Verify the wallet."""
        self.is_verified = True
    
    def suspend(self, reason: Optional[str] = None) -> None:
        """Suspend the wallet."""
        self.is_active = False
        if reason:
            self.metadata["suspension_reason"] = reason
            self.metadata["suspended_at"] = datetime.utcnow().isoformat()
    
    def activate(self) -> None:
        """Activate the wallet."""
        self.is_active = True
        if "suspension_reason" in self.metadata:
            del self.metadata["suspension_reason"]
        if "suspended_at" in self.metadata:
            del self.metadata["suspended_at"]
    
    def to_dict(self, include_balances: bool = True, include_owner: bool = True) -> Dict[str, Any]:
        """Convert wallet to dictionary."""
        result = {
            "id": str(self.id),
            "user_id": str(self.user_id) if self.user_id else None,
            "organization_id": str(self.organization_id) if self.organization_id else None,
            "owner_type": self.owner_type,
            "owner_id": str(self.owner_id),
            "wallet_type": self.wallet_type.value,
            "security_level": self.security_level.value,
            "address": self.address,
            "short_address": self.short_address,
            "public_key": self.public_key,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "is_custodial": self.is_custodial,
            "is_non_custodial": self.is_non_custodial,
            "is_hardware_wallet": self.is_hardware_wallet,
            "is_multisig": self.is_multisig,
            "label": self.label,
            "description": self.description,
            "total_balance": {k: float(v) for k, v in self.total_balance.items()},
            "total_value_usd": float(self.total_value_usd) if self.total_value_usd else None,
            "metadata": self.metadata,
            "tags": self.tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_balances and self.balances:
            result["balances"] = [
                balance.to_dict(include_currency=True, include_wallet=False)
                for balance in self.balances
                if balance.is_active
            ]
        
        if include_owner:
            if self.user:
                result["user"] = {
                    "id": str(self.user.id),
                    "username": self.user.username,
                    "email": getattr(self.user, 'email', None)
                }
            elif self.organization:
                result["organization"] = {
                    "id": str(self.organization.id),
                    "name": self.organization.name,
                    "slug": self.organization.slug
                }
        
        return result


class WalletBalance(Base, UUIDMixin, TimestampMixin):
    """
    Wallet balance model for specific currencies.
    
    This model tracks balances for different currencies within a wallet.
    
    Attributes:
        id: Primary key UUID
        wallet_id: Wallet ID
        currency_id: Currency ID
        balance: Current balance
        locked_balance: Balance locked (e.g., for staking, orders)
        available_balance: Available balance (balance - locked)
        is_active: Whether balance tracking is active
        last_updated: When balance was last updated
        metadata: Additional metadata
    """
    
    __tablename__ = "wallet_balances"
    
    wallet_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("wallets.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    currency_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("currencies.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # Balance information
    balance = Column(Numeric(36, 18), default=Decimal('0'), nullable=False)
    locked_balance = Column(Numeric(36, 18), default=Decimal('0'), nullable=False)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    last_updated = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    wallet = relationship("Wallet", back_populates="balances")
    currency = relationship("Currency", back_populates="wallets")
    
    # Check constraints
    __table_args__ = (
        UniqueConstraint('wallet_id', 'currency_id', name='uq_wallet_currency_balance'),
        CheckConstraint('balance >= 0', name='check_balance_non_negative'),
        CheckConstraint('locked_balance >= 0', name='check_locked_balance_non_negative'),
        Index('ix_wallet_balances_updated', 'wallet_id', 'last_updated'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<WalletBalance(id={self.id}, wallet={self.wallet_id}, currency={self.currency_id}, balance={self.balance})>"
    
    @hybrid_property
    def available_balance(self) -> Decimal:
        """Get available balance (balance - locked)."""
        return self.balance - self.locked_balance
    
    @property
    def formatted_balance(self) -> str:
        """Get formatted balance string."""
        return self.currency.formatted_amount(self.balance)
    
    @property
    def formatted_available_balance(self) -> str:
        """Get formatted available balance string."""
        return self.currency.formatted_amount(self.available_balance)
    
    @property
    def formatted_locked_balance(self) -> str:
        """Get formatted locked balance string."""
        return self.currency.formatted_amount(self.locked_balance)
    
    def lock_balance(self, amount: Decimal) -> bool:
        """Lock balance for transactions."""
        if self.available_balance >= amount:
            self.locked_balance += amount
            self.last_updated = datetime.utcnow()
            return True
        return False
    
    def unlock_balance(self, amount: Decimal) -> bool:
        """Unlock previously locked balance."""
        if self.locked_balance >= amount:
            self.locked_balance -= amount
            self.last_updated = datetime.utcnow()
            return True
        return False
    
    def add_balance(self, amount: Decimal) -> None:
        """Add to balance."""
        self.balance += amount
        self.last_updated = datetime.utcnow()
    
    def subtract_balance(self, amount: Decimal) -> bool:
        """Subtract from balance."""
        if self.available_balance >= amount:
            self.balance -= amount
            self.last_updated = datetime.utcnow()
            return True
        return False
    
    def to_dict(self, include_currency: bool = True, include_wallet: bool = False) -> Dict[str, Any]:
        """Convert wallet balance to dictionary."""
        result = {
            "id": str(self.id),
            "wallet_id": str(self.wallet_id),
            "currency_id": str(self.currency_id),
            "balance": float(self.balance),
            "locked_balance": float(self.locked_balance),
            "available_balance": float(self.available_balance),
            "formatted_balance": self.formatted_balance,
            "formatted_available_balance": self.formatted_available_balance,
            "formatted_locked_balance": self.formatted_locked_balance,
            "is_active": self.is_active,
            "last_updated": self.last_updated.isoformat(),
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_currency and self.currency:
            result["currency"] = self.currency.to_dict()
        
        if include_wallet and self.wallet:
            result["wallet"] = {
                "id": str(self.wallet.id),
                "address": self.wallet.address,
                "short_address": self.wallet.short_address
            }
        
        return result


class Transaction(Base, UUIDMixin, TimestampMixin):
    """
    Transaction model for wallet operations.
    
    This model tracks all wallet transactions including deposits,
    withdrawals, transfers, and payments.
    
    Attributes:
        id: Primary key UUID
        transaction_type: Type of transaction
        status: Transaction status
        from_wallet_id: Source wallet ID
        to_wallet_id: Destination wallet ID
        currency_id: Currency ID
        amount: Transaction amount
        fee_amount: Transaction fee amount
        fee_currency_id: Fee currency ID
        net_amount: Net amount (amount - fee)
        exchange_rate: Exchange rate (for currency conversions)
        reference_id: External reference ID
        blockchain_tx_hash: Blockchain transaction hash
        blockchain_network: Blockchain network
        confirmation_count: Number of blockchain confirmations
        required_confirmations: Required confirmations
        signed_by: User who signed transaction
        signed_at: When transaction was signed
        metadata: Additional metadata
        tags: Categorization tags
    """
    
    __tablename__ = "transactions"
    
    # Transaction details
    transaction_type = Column(SQLEnum(TransactionType), nullable=False, index=True)
    status = Column(SQLEnum(TransactionStatus), default=TransactionStatus.PENDING, nullable=False, index=True)
    
    # Wallets
    from_wallet_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("wallets.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    to_wallet_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("wallets.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    
    # Currency and amounts
    currency_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("currencies.id", ondelete="RESTRICT"), 
        nullable=False,
        index=True
    )
    amount = Column(Numeric(36, 18), nullable=False)
    fee_amount = Column(Numeric(36, 18), nullable=True)
    fee_currency_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("currencies.id", ondelete="RESTRICT"), 
        nullable=True,
        index=True
    )
    
    # Exchange and net amounts
    exchange_rate = Column(Numeric(36, 18), nullable=True)
    net_amount = Column(Numeric(36, 18), nullable=True)
    
    # Reference IDs
    reference_id = Column(String(255), nullable=True, index=True)
    blockchain_tx_hash = Column(String(255), nullable=True, unique=True, index=True)
    blockchain_network = Column(SQLEnum(BlockchainNetwork), nullable=True, index=True)
    
    # Blockchain confirmations
    confirmation_count = Column(Integer, default=0, nullable=False)
    required_confirmations = Column(Integer, default=3, nullable=False)
    
    # Signing
    signed_by = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    signed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    tags = Column(ARRAY(String), default=[], nullable=False, index=True)
    
    # Relationships
    from_wallet = relationship("Wallet", foreign_keys=[from_wallet_id], back_populates="sent_transactions")
    to_wallet = relationship("Wallet", foreign_keys=[to_wallet_id], back_populates="received_transactions")
    currency = relationship("Currency", foreign_keys=[currency_id], back_populates="transactions")
    fee_currency = relationship("Currency", foreign_keys=[fee_currency_id])
    signer = relationship("User", foreign_keys=[signed_by])
    
    # Check constraints
    __table_args__ = (
        CheckConstraint('amount > 0', name='check_amount_positive'),
        CheckConstraint('fee_amount IS NULL OR fee_amount >= 0', name='check_fee_non_negative'),
        CheckConstraint('confirmation_count >= 0', name='check_confirmation_count_non_negative'),
        CheckConstraint('required_confirmations >= 0', name='check_required_confirmations_non_negative'),
        CheckConstraint(
            'from_wallet_id IS NOT NULL OR to_wallet_id IS NOT NULL',
            name='check_transaction_wallets'
        ),
        Index('ix_transactions_status_type', 'status', 'transaction_type'),
        Index('ix_transactions_dates', 'created_at', 'status'),
        Index('ix_transactions_wallets', 'from_wallet_id', 'to_wallet_id'),
        Index('ix_transactions_blockchain', 'blockchain_network', 'blockchain_tx_hash'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Transaction(id={self.id}, type={self.transaction_type.value}, status={self.status.value}, amount={self.amount} {self.currency.symbol})>"
    
    @property
    def is_completed(self) -> bool:
        """Check if transaction is completed."""
        return self.status == TransactionStatus.COMPLETED
    
    @property
    def is_pending(self) -> bool:
        """Check if transaction is pending."""
        return self.status == TransactionStatus.PENDING
    
    @property
    def is_failed(self) -> bool:
        """Check if transaction failed."""
        return self.status == TransactionStatus.FAILED
    
    @property
    def is_confirmed(self) -> bool:
        """Check if transaction has enough blockchain confirmations."""
        return self.confirmation_count >= self.required_confirmations
    
    @property
    def is_signed(self) -> bool:
        """Check if transaction is signed."""
        return self.signed_at is not None
    
    @property
    def is_deposit(self) -> bool:
        """Check if transaction is a deposit."""
        return self.transaction_type == TransactionType.DEPOSIT
    
    @property
    def is_withdrawal(self) -> bool:
        """Check if transaction is a withdrawal."""
        return self.transaction_type == TransactionType.WITHDRAWAL
    
    @property
    def is_transfer(self) -> bool:
        """Check if transaction is a transfer."""
        return self.transaction_type == TransactionType.TRANSFER
    
    @property
    def formatted_amount(self) -> str:
        """Get formatted amount string."""
        return self.currency.formatted_amount(self.amount)
    
    @property
    def formatted_fee(self) -> Optional[str]:
        """Get formatted fee string."""
        if self.fee_amount and self.fee_currency:
            return self.fee_currency.formatted_amount(self.fee_amount)
        return None
    
    @property
    def age_minutes(self) -> float:
        """Get transaction age in minutes."""
        delta = datetime.utcnow() - self.created_at
        return delta.total_seconds() / 60
    
    def calculate_net_amount(self) -> Decimal:
        """Calculate net amount (amount - fee)."""
        if self.fee_amount and self.currency_id == self.fee_currency_id:
            return Decimal(str(self.amount)) - Decimal(str(self.fee_amount))
        return Decimal(str(self.amount))
    
    def mark_as_processing(self) -> None:
        """Mark transaction as processing."""
        self.status = TransactionStatus.PROCESSING
    
    def mark_as_completed(self, blockchain_tx_hash: Optional[str] = None) -> None:
        """Mark transaction as completed."""
        self.status = TransactionStatus.COMPLETED
        if blockchain_tx_hash:
            self.blockchain_tx_hash = blockchain_tx_hash
    
    def mark_as_failed(self, error_message: Optional[str] = None) -> None:
        """Mark transaction as failed."""
        self.status = TransactionStatus.FAILED
        if error_message:
            self.metadata["error_message"] = error_message
    
    def add_confirmation(self) -> None:
        """Add a blockchain confirmation."""
        self.confirmation_count += 1
        
        # Auto-complete if enough confirmations
        if self.is_confirmed and self.status == TransactionStatus.CONFIRMING:
            self.status = TransactionStatus.COMPLETED
    
    def sign(self, user_id: uuid.UUID) -> None:
        """Sign the transaction."""
        self.signed_by = user_id
        self.signed_at = datetime.utcnow()
    
    def to_dict(self, include_wallets: bool = True, include_currencies: bool = True) -> Dict[str, Any]:
        """Convert transaction to dictionary."""
        net_amount = self.calculate_net_amount()
        
        result = {
            "id": str(self.id),
            "transaction_type": self.transaction_type.value,
            "status": self.status.value,
            "from_wallet_id": str(self.from_wallet_id) if self.from_wallet_id else None,
            "to_wallet_id": str(self.to_wallet_id) if self.to_wallet_id else None,
            "currency_id": str(self.currency_id),
            "amount": float(self.amount),
            "formatted_amount": self.formatted_amount,
            "fee_amount": float(self.fee_amount) if self.fee_amount else None,
            "fee_currency_id": str(self.fee_currency_id) if self.fee_currency_id else None,
            "formatted_fee": self.formatted_fee,
            "exchange_rate": float(self.exchange_rate) if self.exchange_rate else None,
            "net_amount": float(net_amount),
            "reference_id": self.reference_id,
            "blockchain_tx_hash": self.blockchain_tx_hash,
            "blockchain_network": self.blockchain_network.value if self.blockchain_network else None,
            "confirmation_count": self.confirmation_count,
            "required_confirmations": self.required_confirmations,
            "is_completed": self.is_completed,
            "is_pending": self.is_pending,
            "is_failed": self.is_failed,
            "is_confirmed": self.is_confirmed,
            "is_signed": self.is_signed,
            "is_deposit": self.is_deposit,
            "is_withdrawal": self.is_withdrawal,
            "is_transfer": self.is_transfer,
            "signed_by": str(self.signed_by) if self.signed_by else None,
            "signed_at": self.signed_at.isoformat() if self.signed_at else None,
            "age_minutes": round(self.age_minutes, 2),
            "metadata": self.metadata,
            "tags": self.tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_wallets:
            if self.from_wallet:
                result["from_wallet"] = {
                    "id": str(self.from_wallet.id),
                    "address": self.from_wallet.address,
                    "short_address": self.from_wallet.short_address
                }
            if self.to_wallet:
                result["to_wallet"] = {
                    "id": str(self.to_wallet.id),
                    "address": self.to_wallet.address,
                    "short_address": self.to_wallet.short_address
                }
        
        if include_currencies:
            if self.currency:
                result["currency"] = self.currency.to_dict()
            if self.fee_currency:
                result["fee_currency"] = self.fee_currency.to_dict()
        
        if self.signer:
            result["signer"] = {
                "id": str(self.signer.id),
                "username": self.signer.username
            }
        
        return result
    
    @classmethod
    def create_deposit(
        cls,
        to_wallet_id: uuid.UUID,
        currency_id: uuid.UUID,
        amount: Decimal,
        blockchain_tx_hash: Optional[str] = None,
        blockchain_network: Optional[BlockchainNetwork] = None,
        reference_id: Optional[str] = None,
        fee_amount: Optional[Decimal] = None,
        fee_currency_id: Optional[uuid.UUID] = None,
        **kwargs
    ) -> 'Transaction':
        """Create a deposit transaction."""
        return cls.create(
            transaction_type=TransactionType.DEPOSIT,
            to_wallet_id=to_wallet_id,
            currency_id=currency_id,
            amount=amount,
            blockchain_tx_hash=blockchain_tx_hash,
            blockchain_network=blockchain_network,
            reference_id=reference_id,
            fee_amount=fee_amount,
            fee_currency_id=fee_currency_id,
            **kwargs
        )
    
    @classmethod
    def create_withdrawal(
        cls,
        from_wallet_id: uuid.UUID,
        to_wallet_id: uuid.UUID,
        currency_id: uuid.UUID,
        amount: Decimal,
        fee_amount: Optional[Decimal] = None,
        fee_currency_id: Optional[uuid.UUID] = None,
        **kwargs
    ) -> 'Transaction':
        """Create a withdrawal transaction."""
        return cls.create(
            transaction_type=TransactionType.WITHDRAWAL,
            from_wallet_id=from_wallet_id,
            to_wallet_id=to_wallet_id,
            currency_id=currency_id,
            amount=amount,
            fee_amount=fee_amount,
            fee_currency_id=fee_currency_id,
            **kwargs
        )
    
    @classmethod
    def create_transfer(
        cls,
        from_wallet_id: uuid.UUID,
        to_wallet_id: uuid.UUID,
        currency_id: uuid.UUID,
        amount: Decimal,
        fee_amount: Optional[Decimal] = None,
        fee_currency_id: Optional[uuid.UUID] = None,
        **kwargs
    ) -> 'Transaction':
        """Create a transfer transaction."""
        return cls.create(
            transaction_type=TransactionType.TRANSFER,
            from_wallet_id=from_wallet_id,
            to_wallet_id=to_wallet_id,
            currency_id=currency_id,
            amount=amount,
            fee_amount=fee_amount,
            fee_currency_id=fee_currency_id,
            **kwargs
        )
    
    @classmethod
    def create(
        cls,
        transaction_type: TransactionType,
        currency_id: uuid.UUID,
        amount: Decimal,
        from_wallet_id: Optional[uuid.UUID] = None,
        to_wallet_id: Optional[uuid.UUID] = None,
        status: TransactionStatus = TransactionStatus.PENDING,
        fee_amount: Optional[Decimal] = None,
        fee_currency_id: Optional[uuid.UUID] = None,
        exchange_rate: Optional[Decimal] = None,
        reference_id: Optional[str] = None,
        blockchain_tx_hash: Optional[str] = None,
        blockchain_network: Optional[BlockchainNetwork] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> 'Transaction':
        """
        Factory method to create a new transaction.
        
        Args:
            transaction_type: Type of transaction
            currency_id: Currency ID
            amount: Transaction amount
            from_wallet_id: Source wallet ID
            to_wallet_id: Destination wallet ID
            status: Transaction status
            fee_amount: Transaction fee
            fee_currency_id: Fee currency ID
            exchange_rate: Exchange rate
            reference_id: External reference ID
            blockchain_tx_hash: Blockchain transaction hash
            blockchain_network: Blockchain network
            metadata: Additional metadata
            tags: Categorization tags
            **kwargs: Additional arguments
            
        Returns:
            A new Transaction instance
        """
        transaction = cls(
            transaction_type=transaction_type,
            status=status,
            from_wallet_id=from_wallet_id,
            to_wallet_id=to_wallet_id,
            currency_id=currency_id,
            amount=amount,
            fee_amount=fee_amount,
            fee_currency_id=fee_currency_id,
            exchange_rate=exchange_rate,
            reference_id=reference_id,
            blockchain_tx_hash=blockchain_tx_hash,
            blockchain_network=blockchain_network,
            metadata=metadata or {},
            tags=tags or [],
            **kwargs
        )
        
        # Calculate net amount
        transaction.net_amount = transaction.calculate_net_amount()
        
        return transaction


class WalletSignature(Base, UUIDMixin, TimestampMixin):
    """
    Wallet signature model for multi-signature wallets.
    
    This model tracks signatures for multi-signature transactions.
    
    Attributes:
        id: Primary key UUID
        wallet_id: Wallet ID
        transaction_id: Transaction ID
        user_id: User who signed
        signature: Digital signature
        signature_type: Type of signature
        status: Signature status
        signed_at: When signature was created
        metadata: Additional metadata
    """
    
    __tablename__ = "wallet_signatures"
    
    wallet_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("wallets.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    transaction_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("transactions.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # Signature information
    user_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    signature = Column(Text, nullable=False)
    signature_type = Column(String(50), default="ecdsa", nullable=False)
    
    # Status
    status = Column(String(50), default="pending", nullable=False, index=True)
    signed_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    wallet = relationship("Wallet", back_populates="signatures")
    transaction = relationship("Transaction")
    user = relationship("User")
    
    # Check constraints
    __table_args__ = (
        UniqueConstraint('transaction_id', 'user_id', name='uq_transaction_user_signature'),
        Index('ix_wallet_signatures_wallet_user', 'wallet_id', 'user_id'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<WalletSignature(id={self.id}, wallet={self.wallet_id}, transaction={self.transaction_id}, user={self.user_id})>"


class WalletBackupKey(Base, UUIDMixin, TimestampMixin):
    """
    Wallet backup key model for recovery and backup.
    
    This model stores encrypted backup keys for wallet recovery.
    
    Attributes:
        id: Primary key UUID
        wallet_id: Wallet ID
        backup_type: Type of backup
        encrypted_data: Encrypted backup data
        encryption_method: Encryption method used
        is_active: Whether backup is active
        used_at: When backup was used
        expires_at: When backup expires
        metadata: Additional metadata
    """
    
    __tablename__ = "wallet_backup_keys"
    
    wallet_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("wallets.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # Backup information
    backup_type = Column(String(50), nullable=False, index=True)
    encrypted_data = Column(Text, nullable=False)
    encryption_method = Column(String(50), nullable=False)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    used_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    wallet = relationship("Wallet", back_populates="backup_keys")
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<WalletBackupKey(id={self.id}, wallet={self.wallet_id}, type={self.backup_type})>"


class StakingPool(Base, UUIDMixin, TimestampMixin):
    """
    Staking pool model for token staking.
    
    This model represents staking pools where users can stake tokens
    to earn rewards.
    
    Attributes:
        id: Primary key UUID
        name: Staking pool name
        description: Pool description
        currency_id: Staking currency ID
        reward_currency_id: Reward currency ID
        apr: Annual percentage rate
        lock_period_days: Lock period in days
        min_stake_amount: Minimum stake amount
        max_stake_amount: Maximum stake amount
        total_staked: Total amount staked
        reward_distributed: Total rewards distributed
        is_active: Whether pool is active
        start_date: When staking starts
        end_date: When staking ends
        metadata: Additional metadata
        tags: Categorization tags
    """
    
    __tablename__ = "staking_pools"
    
    # Pool information
    name = Column(String(200), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Currencies
    currency_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("currencies.id", ondelete="RESTRICT"), 
        nullable=False,
        index=True
    )
    reward_currency_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("currencies.id", ondelete="RESTRICT"), 
        nullable=False,
        index=True
    )
    
    # Staking parameters
    apr = Column(Numeric(10, 4), nullable=False)  # Annual percentage rate
    lock_period_days = Column(Integer, nullable=False)
    min_stake_amount = Column(Numeric(36, 18), nullable=False)
    max_stake_amount = Column(Numeric(36, 18), nullable=True)
    
    # Statistics
    total_staked = Column(Numeric(36, 18), default=Decimal('0'), nullable=False)
    reward_distributed = Column(Numeric(36, 18), default=Decimal('0'), nullable=False)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    start_date = Column(DateTime(timezone=True), nullable=False, index=True)
    end_date = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    tags = Column(ARRAY(String), default=[], nullable=False, index=True)
    
    # Relationships
    currency = relationship("Currency", foreign_keys=[currency_id])
    reward_currency = relationship("Currency", foreign_keys=[reward_currency_id])
    stakes = relationship("Staking", back_populates="pool", cascade="all, delete-orphan")
    
    # Check constraints
    __table_args__ = (
        CheckConstraint('apr >= 0', name='check_apr_non_negative'),
        CheckConstraint('lock_period_days >= 0', name='check_lock_period_non_negative'),
        CheckConstraint('min_stake_amount >= 0', name='check_min_stake_non_negative'),
        CheckConstraint('total_staked >= 0', name='check_total_staked_non_negative'),
        CheckConstraint('reward_distributed >= 0', name='check_reward_distributed_non_negative'),
        Index('ix_staking_pools_active_dates', 'is_active', 'start_date', 'end_date'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<StakingPool(id={self.id}, name={self.name}, apr={self.apr}%)>"
    
    @property
    def has_ended(self) -> bool:
        """Check if staking pool has ended."""
        if not self.end_date:
            return False
        return datetime.utcnow() > self.end_date
    
    @property
    def is_available(self) -> bool:
        """Check if staking pool is available for staking."""
        if not self.is_active or self.has_ended:
            return False
        
        now = datetime.utcnow()
        if now < self.start_date:
            return False
        
        if self.max_stake_amount and self.total_staked >= self.max_stake_amount:
            return False
        
        return True
    
    @property
    def available_capacity(self) -> Optional[Decimal]:
        """Get available staking capacity."""
        if not self.max_stake_amount:
            return None
        return max(Decimal('0'), self.max_stake_amount - self.total_staked)
    
    @property
    def daily_reward_rate(self) -> Decimal:
        """Get daily reward rate."""
        return Decimal(str(self.apr)) / Decimal('365') / Decimal('100')
    
    def calculate_reward(self, stake_amount: Decimal, days_staked: int) -> Decimal:
        """Calculate reward for staked amount."""
        daily_rate = self.daily_reward_rate
        return stake_amount * daily_rate * Decimal(str(days_staked))
    
    def add_stake(self, amount: Decimal) -> bool:
        """Add stake to pool."""
        if not self.is_available:
            return False
        
        if amount < self.min_stake_amount:
            return False
        
        if self.max_stake_amount and (self.total_staked + amount) > self.max_stake_amount:
            return False
        
        self.total_staked += amount
        return True
    
    def remove_stake(self, amount: Decimal) -> bool:
        """Remove stake from pool."""
        if amount > self.total_staked:
            return False
        
        self.total_staked -= amount
        return True
    
    def distribute_reward(self, amount: Decimal) -> None:
        """Record reward distribution."""
        self.reward_distributed += amount
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert staking pool to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "currency_id": str(self.currency_id),
            "reward_currency_id": str(self.reward_currency_id),
            "apr": float(self.apr),
            "lock_period_days": self.lock_period_days,
            "min_stake_amount": float(self.min_stake_amount),
            "max_stake_amount": float(self.max_stake_amount) if self.max_stake_amount else None,
            "total_staked": float(self.total_staked),
            "reward_distributed": float(self.reward_distributed),
            "available_capacity": float(self.available_capacity) if self.available_capacity else None,
            "daily_reward_rate": float(self.daily_reward_rate),
            "is_active": self.is_active,
            "is_available": self.is_available,
            "has_ended": self.has_ended,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "metadata": self.metadata,
            "tags": self.tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class Staking(Base, UUIDMixin, TimestampMixin):
    """
    Staking model for user stakes in staking pools.
    
    This model tracks individual user stakes in staking pools.
    
    Attributes:
        id: Primary key UUID
        user_id: User who staked
        wallet_id: Wallet used for staking
        pool_id: Staking pool ID
        amount: Staked amount
        reward_amount: Total rewards earned
        status: Staking status
        staked_at: When tokens were staked
        unlock_at: When tokens can be unstaked
        unstaked_at: When tokens were unstaked
        metadata: Additional metadata
    """
    
    __tablename__ = "stakings"
    
    # Stake information
    user_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    wallet_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("wallets.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    pool_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("staking_pools.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # Amounts
    amount = Column(Numeric(36, 18), nullable=False)
    reward_amount = Column(Numeric(36, 18), default=Decimal('0'), nullable=False)
    
    # Status and dates
    status = Column(String(50), default="active", nullable=False, index=True)
    staked_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    unlock_at = Column(DateTime(timezone=True), nullable=False)
    unstaked_at = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    user = relationship("User")
    wallet = relationship("Wallet")
    pool = relationship("StakingPool", back_populates="stakes")
    
    # Check constraints
    __table_args__ = (
        CheckConstraint('amount > 0', name='check_amount_positive'),
        CheckConstraint('reward_amount >= 0', name='check_reward_non_negative'),
        CheckConstraint('unlock_at > staked_at', name='check_unlock_after_stake'),
        Index('ix_stakings_user_status', 'user_id', 'status'),
        Index('ix_stakings_pool_status', 'pool_id', 'status'),
        Index('ix_stakings_unlock_date', 'unlock_at', 'status'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Staking(id={self.id}, user={self.user_id}, pool={self.pool_id}, amount={self.amount})>"
    
    @property
    def is_active(self) -> bool:
        """Check if stake is active."""
        return self.status == "active"
    
    @property
    def is_unlocked(self) -> bool:
        """Check if stake is unlocked (can be unstaked)."""
        return datetime.utcnow() >= self.unlock_at
    
    @property
    def is_ended(self) -> bool:
        """Check if stake has ended (unstaked)."""
        return self.status == "ended"
    
    @property
    def days_staked(self) -> int:
        """Get number of days staked."""
        end_date = self.unstaked_at or datetime.utcnow()
        duration = end_date - self.staked_at
        return duration.days
    
    @property
    def days_until_unlock(self) -> int:
        """Get days until unlock."""
        if self.is_unlocked:
            return 0
        remaining = self.unlock_at - datetime.utcnow()
        return max(0, remaining.days)
    
    @property
    def estimated_reward(self) -> Decimal:
        """Get estimated reward for current stake."""
        if not self.is_active:
            return self.reward_amount
        
        days = self.days_staked
        return self.pool.calculate_reward(self.amount, days)
    
    def add_reward(self, amount: Decimal) -> None:
        """Add reward to stake."""
        self.reward_amount += amount
    
    def unstake(self) -> bool:
        """Unstake tokens."""
        if not self.is_active or not self.is_unlocked:
            return False
        
        self.status = "ended"
        self.unstaked_at = datetime.utcnow()
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert staking to dictionary."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "wallet_id": str(self.wallet_id),
            "pool_id": str(self.pool_id),
            "amount": float(self.amount),
            "reward_amount": float(self.reward_amount),
            "estimated_reward": float(self.estimated_reward),
            "status": self.status,
            "is_active": self.is_active,
            "is_unlocked": self.is_unlocked,
            "is_ended": self.is_ended,
            "staked_at": self.staked_at.isoformat(),
            "unlock_at": self.unlock_at.isoformat(),
            "unstaked_at": self.unstaked_at.isoformat() if self.unstaked_at else None,
            "days_staked": self.days_staked,
            "days_until_unlock": self.days_until_unlock,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class ExchangeRate(Base, UUIDMixin, TimestampMixin):
    """
    Exchange rate model for currency conversions.
    
    This model stores exchange rates between different currencies.
    
    Attributes:
        id: Primary key UUID
        base_currency_id: Base currency ID
        quote_currency_id: Quote currency ID
        rate: Exchange rate (1 base = X quote)
        source: Rate source (API, manual, etc.)
        is_active: Whether rate is active
        valid_from: When rate is valid from
        valid_to: When rate is valid to
        metadata: Additional metadata
    """
    
    __tablename__ = "exchange_rates"
    
    # Currencies
    base_currency_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("currencies.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    quote_currency_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("currencies.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # Rate information
    rate = Column(Numeric(36, 18), nullable=False)
    source = Column(String(100), nullable=False, index=True)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    valid_from = Column(DateTime(timezone=True), nullable=False, index=True)
    valid_to = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    base_currency = relationship("Currency", foreign_keys=[base_currency_id])
    quote_currency = relationship("Currency", foreign_keys=[quote_currency_id])
    
    # Check constraints
    __table_args__ = (
        UniqueConstraint('base_currency_id', 'quote_currency_id', 'valid_from', name='uq_exchange_rate'),
        CheckConstraint('rate > 0', name='check_rate_positive'),
        CheckConstraint('base_currency_id != quote_currency_id', name='check_different_currencies'),
        CheckConstraint('valid_to IS NULL OR valid_to > valid_from', name='check_valid_dates'),
        Index('ix_exchange_rates_active', 'is_active', 'valid_from', 'valid_to'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<ExchangeRate(id={self.id}, base={self.base_currency_id}, quote={self.quote_currency_id}, rate={self.rate})>"
    
    @property
    def pair(self) -> str:
        """Get currency pair string."""
        return f"{self.base_currency.symbol}/{self.quote_currency.symbol}"
    
    @property
    def inverse_rate(self) -> Decimal:
        """Get inverse exchange rate."""
        return Decimal('1') / Decimal(str(self.rate))
    
    @property
    def is_valid(self) -> bool:
        """Check if exchange rate is currently valid."""
        if not self.is_active:
            return False
        
        now = datetime.utcnow()
        if now < self.valid_from:
            return False
        
        if self.valid_to and now > self.valid_to:
            return False
        
        return True
    
    def convert(self, amount: Decimal, from_base: bool = True) -> Decimal:
        """Convert amount using exchange rate."""
        if from_base:
            return amount * Decimal(str(self.rate))
        else:
            return amount * self.inverse_rate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exchange rate to dictionary."""
        return {
            "id": str(self.id),
            "base_currency_id": str(self.base_currency_id),
            "quote_currency_id": str(self.quote_currency_id),
            "pair": self.pair,
            "rate": float(self.rate),
            "inverse_rate": float(self.inverse_rate),
            "source": self.source,
            "is_active": self.is_active,
            "is_valid": self.is_valid,
            "valid_from": self.valid_from.isoformat(),
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


# Helper functions
def validate_ethereum_address(address: str) -> bool:
    """Validate Ethereum address format."""
    import re
    
    # Basic Ethereum address validation
    pattern = r'^0x[a-fA-F0-9]{40}$'
    if not re.match(pattern, address):
        return False
    
    # Optional: EIP-55 checksum validation
    # In production, you would implement proper checksum validation
    return True


def generate_wallet_address(network: BlockchainNetwork) -> Dict[str, Any]:
    """
    Generate new wallet address.
    
    Note: In production, use proper cryptographic libraries
    like web3.py for Ethereum, etc.
    
    Args:
        network: Blockchain network
        
    Returns:
        Dictionary with address, private key, and mnemonic
    """
    import secrets
    import hashlib
    import binascii
    
    # This is a simplified example. In production, use:
    # - web3.py for Ethereum
    # - bitcoinlib for Bitcoin
    # - solana-py for Solana
    
    # Generate random private key
    private_key = secrets.token_hex(32)
    
    # Generate address from private key (simplified)
    if network in [BlockchainNetwork.ETHEREUM, BlockchainNetwork.ETHEREUM_GOERLI, BlockchainNetwork.ETHEREUM_SEPOLIA]:
        # Simplified Ethereum address generation
        # In production, use proper ECDSA and Keccak256
        address_hash = hashlib.sha256(private_key.encode()).hexdigest()
        address = f"0x{address_hash[-40:]}"
    elif network == BlockchainNetwork.BITCOIN:
        # Simplified Bitcoin address generation
        address_hash = hashlib.sha256(private_key.encode()).hexdigest()
        address = f"1{address_hash[:33]}"
    else:
        # Generic address generation
        address_hash = hashlib.sha256(private_key.encode()).hexdigest()
        address = f"0x{address_hash[:40]}"
    
    # Generate mnemonic (simplified)
    mnemonic_words = []
    for _ in range(12):
        word = secrets.token_hex(4)
        mnemonic_words.append(word)
    mnemonic = " ".join(mnemonic_words)
    
    return {
        "address": address,
        "private_key": private_key,  # In production, encrypt this
        "mnemonic": mnemonic,  # In production, hash/encrypt this
        "network": network.value
    }


def calculate_transaction_fee(
    network: BlockchainNetwork,
    gas_limit: int = 21000,
    gas_price_gwei: Optional[float] = None
) -> Decimal:
    """
    Calculate transaction fee for blockchain networks.
    
    Args:
        network: Blockchain network
        gas_limit: Gas limit for transaction
        gas_price_gwei: Gas price in gwei (optional)
        
    Returns:
        Transaction fee in ETH/BNB/etc.
    """
    # Default gas prices (in gwei)
    default_gas_prices = {
        BlockchainNetwork.ETHEREUM: 30.0,
        BlockchainNetwork.BINANCE_SMART_CHAIN: 5.0,
        BlockchainNetwork.POLYGON: 50.0,
        BlockchainNetwork.ARBITRUM: 0.1,
        BlockchainNetwork.OPTIMISM: 0.001,
    }
    
    gas_price = gas_price_gwei or default_gas_prices.get(network, 10.0)
    
    # Calculate fee in wei
    fee_wei = gas_limit * (gas_price * 1e9)
    
    # Convert to ETH/BNB/etc. (1e18 wei = 1 ETH)
    fee_eth = Decimal(str(fee_wei)) / Decimal('1e18')
    
    return fee_eth


def format_crypto_amount(amount: Decimal, symbol: str) -> str:
    """Format cryptocurrency amount with appropriate decimals."""
    decimals_map = {
        "BTC": 8,
        "ETH": 4,
        "BNB": 4,
        "MATIC": 4,
        "USDT": 2,
        "USDC": 2,
        "DAI": 2,
    }
    
    decimals = decimals_map.get(symbol, 6)
    return f"{amount:.{decimals}f} {symbol}"


def get_wallet_explorer_url(network: BlockchainNetwork, address: str) -> Optional[str]:
    """Get blockchain explorer URL for wallet address."""
    explorer_urls = {
        BlockchainNetwork.ETHEREUM: f"https://etherscan.io/address/{address}",
        BlockchainNetwork.ETHEREUM_GOERLI: f"https://goerli.etherscan.io/address/{address}",
        BlockchainNetwork.ETHEREUM_SEPOLIA: f"https://sepolia.etherscan.io/address/{address}",
        BlockchainNetwork.BINANCE_SMART_CHAIN: f"https://bscscan.com/address/{address}",
        BlockchainNetwork.BSC_TESTNET: f"https://testnet.bscscan.com/address/{address}",
        BlockchainNetwork.POLYGON: f"https://polygonscan.com/address/{address}",
        BlockchainNetwork.POLYGON_MUMBAI: f"https://mumbai.polygonscan.com/address/{address}",
        BlockchainNetwork.ARBITRUM: f"https://arbiscan.io/address/{address}",
        BlockchainNetwork.OPTIMISM: f"https://optimistic.etherscan.io/address/{address}",
        BlockchainNetwork.AVALANCHE: f"https://snowtrace.io/address/{address}",
        BlockchainNetwork.SOLANA: f"https://explorer.solana.com/address/{address}",
        BlockchainNetwork.BITCOIN: f"https://www.blockchain.com/btc/address/{address}",
        BlockchainNetwork.BITCOIN_TESTNET: f"https://www.blockchain.com/btc-testnet/address/{address}",
    }
    
    return explorer_urls.get(network)


def get_transaction_explorer_url(network: BlockchainNetwork, tx_hash: str) -> Optional[str]:
    """Get blockchain explorer URL for transaction."""
    explorer_urls = {
        BlockchainNetwork.ETHEREUM: f"https://etherscan.io/tx/{tx_hash}",
        BlockchainNetwork.ETHEREUM_GOERLI: f"https://goerli.etherscan.io/tx/{tx_hash}",
        BlockchainNetwork.ETHEREUM_SEPOLIA: f"https://sepolia.etherscan.io/tx/{tx_hash}",
        BlockchainNetwork.BINANCE_SMART_CHAIN: f"https://bscscan.com/tx/{tx_hash}",
        BlockchainNetwork.BSC_TESTNET: f"https://testnet.bscscan.com/tx/{tx_hash}",
        BlockchainNetwork.POLYGON: f"https://polygonscan.com/tx/{tx_hash}",
        BlockchainNetwork.POLYGON_MUMBAI: f"https://mumbai.polygonscan.com/tx/{tx_hash}",
        BlockchainNetwork.ARBITRUM: f"https://arbiscan.io/tx/{tx_hash}",
        BlockchainNetwork.OPTIMISM: f"https://optimistic.etherscan.io/tx/{tx_hash}",
        BlockchainNetwork.AVALANCHE: f"https://snowtrace.io/tx/{tx_hash}",
        BlockchainNetwork.SOLANA: f"https://explorer.solana.com/tx/{tx_hash}",
        BlockchainNetwork.BITCOIN: f"https://www.blockchain.com/btc/tx/{tx_hash}",
        BlockchainNetwork.BITCOIN_TESTNET: f"https://www.blockchain.com/btc-testnet/tx/{tx_hash}",
    }
    
    return explorer_urls.get(network)