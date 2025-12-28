"""
Wallet Subscriber for handling cryptocurrency wallet operations, transactions, and rewards.
Listens for financial events, processes transactions, manages rewards, and handles wallet operations.
"""

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple,DefaultDict
import json
import uuid
from enum import Enum
from dataclasses import dataclass, asdict
from decimal import Decimal
import hashlib
import hmac

from sqlalchemy import select, update, and_, or_, desc, func, text , distinct
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload

from app.core.logging_config import logger
from app.db.session import AsyncSessionLocal
from app.db.models.user import User
from app.db.models.wallet import Wallet, WalletStatus, WalletType
from app.db.models.transaction import (
    Transaction, 
    TransactionType, 
    TransactionStatus, 
    TransactionSource
)
from app.db.models.reward import Reward, RewardType, RewardStatus
from app.db.models.withdrawal import Withdrawal, WithdrawalStatus
from app.db.models.audit_log import AuditLog
from app.events.event_bus import EventBus
from app.events.event_types import EventType
from app.services.community.wallet_service import WalletService
from app.services.utils.caching import cache
from app.core.config import settings
from app.core.exceptions import (
    InsufficientFundsError,
    InvalidTransactionError,
    WalletSuspendedError,
    RateLimitExceededError
)


class CoinAction(str, Enum):
    """Actions that can be performed with coins."""
    EARN = "earn"           # Earn coins through activities
    SPEND = "spend"         # Spend coins on purchases
    TRANSFER = "transfer"   # Transfer coins between users
    WITHDRAW = "withdraw"   # Withdraw coins to external wallet
    DEPOSIT = "deposit"     # Deposit coins from external source
    REFUND = "refund"       # Refund coins
    BURN = "burn"           # Burn/destroy coins
    MINT = "mint"           # Mint/create new coins


@dataclass
class CoinTransaction:
    """Data structure for coin transactions."""
    from_user_id: Optional[int]  # None for system transactions
    to_user_id: Optional[int]    # None for system transactions
    amount: Decimal
    action: CoinAction
    source: TransactionSource
    reference_id: Optional[str] = None  # ID of related entity (incident, briefing, etc.)
    metadata: Dict[str, Any] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        """Set default values."""
        if self.metadata is None:
            self.metadata = {}
        if self.amount <= 0:
            raise ValueError("Amount must be positive")
        
        # Generate reference ID if not provided
        if not self.reference_id:
            self.reference_id = str(uuid.uuid4())


@dataclass
class RewardRequest:
    """Data structure for reward requests."""
    user_id: int
    reward_type: RewardType
    amount: Decimal
    source_id: Optional[str] = None  # ID of source entity
    source_type: Optional[str] = None  # Type of source entity
    metadata: Dict[str, Any] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        """Set default values."""
        if self.metadata is None:
            self.metadata = {}
        if self.amount <= 0:
            raise ValueError("Reward amount must be positive")


class WalletSubscriber:
    """
    Subscriber that listens for wallet and financial events.
    Handles coin transactions, rewards, withdrawals, and wallet management.
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.wallet_service = WalletService()
        self._subscriptions = []
        
        # Rate limiting
        self.user_transaction_count = defaultdict(int) # type : DefaultDict[str,int]
        self.user_transaction_reset = defaultdict(datetime)
        
        # Reward multipliers based on user activity
        self.reward_multipliers = {
            'new_user': Decimal('2.0'),      # 2x rewards for first week
            'active_user': Decimal('1.5'),   # 1.5x for active users
            'veteran': Decimal('1.2'),       # 1.2x for veteran users
            'moderator': Decimal('1.3'),     # 1.3x for moderators
            'content_creator': Decimal('1.4') # 1.4x for content creators
        }
        
        # Daily reward limits
        self.daily_limits = {
            'incident_report': Decimal('100'),
            'incident_verification': Decimal('50'),
            'comment': Decimal('10'),
            'briefing_generation': Decimal('25'),
            'chat_interaction': Decimal('5'),
        }
    
    async def initialize(self):
        """Subscribe to wallet and financial events."""
        # User activity reward events
        self._subscriptions.extend([
            await self.event_bus.subscribe(EventType.INCIDENT_REPORTED, self.handle_incident_reported),
            await self.event_bus.subscribe(EventType.INCIDENT_VERIFIED, self.handle_incident_verified),
            await self.event_bus.subscribe(EventType.COMMENT_CREATED, self.handle_comment_created),
            await self.event_bus.subscribe(EventType.COMMENT_REPLY, self.handle_comment_reply),
            await self.event_bus.subscribe(EventType.BRIEFING_GENERATED, self.handle_briefing_generated),
            await self.event_bus.subscribe(EventType.BRIEFING_READY, self.handle_briefing_ready),
            await self.event_bus.subscribe(EventType.CHAT_MESSAGE_SENT, self.handle_chat_message),
        ])
        
        # Content creation and quality events
        self._subscriptions.extend([
            await self.event_bus.subscribe(EventType.CONTENT_CREATED, self.handle_content_created),
            await self.event_bus.subscribe(EventType.CONTENT_APPROVED, self.handle_content_approved),
            await self.event_bus.subscribe(EventType.CONTENT_FEATURED, self.handle_content_featured),
            await self.event_bus.subscribe(EventType.CONTENT_VIRAL, self.handle_content_viral),
        ])
        
        # Moderation and community events
        self._subscriptions.extend([
            await self.event_bus.subscribe(EventType.MODERATION_ACTION_TAKEN, self.handle_moderation_action),
            await self.event_bus.subscribe(EventType.APPEAL_REVIEWED, self.handle_appeal_reviewed),
            await self.event_bus.subscribe(EventType.USER_REPORTED, self.handle_user_reported_valid),
        ])
        
        # Transaction and wallet events
        self._subscriptions.extend([
            await self.event_bus.subscribe(EventType.COIN_TRANSFER_REQUESTED, self.handle_coin_transfer),
            await self.event_bus.subscribe(EventType.WITHDRAWAL_REQUESTED, self.handle_withdrawal_requested),
            await self.event_bus.subscribe(EventType.WITHDRAWAL_APPROVED, self.handle_withdrawal_approved),
            await self.event_bus.subscribe(EventType.WITHDRAWAL_COMPLETED, self.handle_withdrawal_completed),
            await self.event_bus.subscribe(EventType.WITHDRAWAL_FAILED, self.handle_withdrawal_failed),
            await self.event_bus.subscribe(EventType.DEPOSIT_RECEIVED, self.handle_deposit_received),
        ])
        
        # System reward events
        self._subscriptions.extend([
            await self.event_bus.subscribe(EventType.REFERRAL_COMPLETED, self.handle_referral_completed),
            await self.event_bus.subscribe(EventType.ACHIEVEMENT_UNLOCKED, self.handle_achievement_unlocked),
            await self.event_bus.subscribe(EventType.DAILY_LOGIN, self.handle_daily_login),
            await self.event_bus.subscribe(EventType.STREAK_MAINTAINED, self.handle_streak_maintained),
        ])
        
        # Direct wallet operations
        self._subscriptions.extend([
            await self.event_bus.subscribe(EventType.COIN_REWARD_REQUESTED, self.handle_coin_reward_request),
            await self.event_bus.subscribe(EventType.COIN_PURCHASE_REQUESTED, self.handle_coin_purchase),
            await self.event_bus.subscribe(EventType.COIN_BURN_REQUESTED, self.handle_coin_burn),
        ])
        
        # Start background tasks
        asyncio.create_task(self._process_transaction_queue())
        asyncio.create_task(self._cleanup_old_transactions())
        asyncio.create_task(self._update_daily_limits())
        
        logger.info("WalletSubscriber initialized")
    
    async def cleanup(self):
        """Cleanup subscriptions."""
        for subscription in self._subscriptions:
            await self.event_bus.unsubscribe(subscription)
        self._subscriptions.clear()
        logger.info("WalletSubscriber cleaned up")
    
    async def _get_or_create_wallet(self, user_id: int) -> Wallet:
        """
        Get user's wallet, create if it doesn't exist.
        
        Args:
            user_id: User ID
            
        Returns:
            User's wallet
        """
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(Wallet)
                    .where(Wallet.user_id == user_id)
                    .where(Wallet.wallet_type == WalletType.PRIMARY)
                )
                wallet = result.scalar_one_or_none()
                
                if not wallet:
                    # Create new wallet
                    wallet = Wallet(
                        user_id=user_id,
                        wallet_type=WalletType.PRIMARY,
                        balance=Decimal('0'),
                        total_earned=Decimal('0'),
                        total_spent=Decimal('0'),
                        status=WalletStatus.ACTIVE,
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                    session.add(wallet)
                    await session.commit()
                    await session.refresh(wallet)
                    
                    logger.info(f"Created new wallet for user {user_id}")
                
                return wallet
                
        except Exception as e:
            logger.error(f"Error getting/creating wallet for user {user_id}: {e}", exc_info=True)
            raise
    
    async def _check_rate_limit(self, user_id: int, action: str) -> bool:
        """
        Check if user has exceeded rate limit for an action.
        
        Args:
            user_id: User ID
            action: Action type
            
        Returns:
            True if allowed, False if rate limited
        """
        current_time = datetime.utcnow()
        key = f"{user_id}:{action}"
        
        # Reset counter if new day
        if (
            key not in self.user_transaction_reset or
            current_time - self.user_transaction_reset[key] >= timedelta(days=1)
        ):
            self.user_transaction_count[key] = 0
            self.user_transaction_reset[key] = current_time
        
        # Get limit for this action
        limits = {
            'incident_report': 50,
            'incident_verification': 100,
            'comment': 200,
            'briefing_generation': 20,
            'coin_transfer': 100,
            'withdrawal': 5,
        }
        
        limit = limits.get(action, 100)  # Default limit
        
        # Check limit
        if self.user_transaction_count[key] >= limit:
            logger.warning(f"User {user_id} exceeded rate limit for {action}")
            return False
        
        self.user_transaction_count[key] += 1
        return True
    
    async def _check_daily_limit(self, user_id: int, reward_type: str, amount: Decimal) -> bool:
        """
        Check if user has exceeded daily reward limit for a type.
        
        Args:
            user_id: User ID
            reward_type: Type of reward
            amount: Amount to award
            
        Returns:
            True if within limit, False if exceeded
        """
        try:
            async with AsyncSessionLocal() as session:
                # Calculate rewards received today for this type
                today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                
                result = await session.execute(
                    select(func.sum(Reward.amount))
                    .where(Reward.user_id == user_id)
                    .where(Reward.reward_type == reward_type)
                    .where(Reward.created_at >= today_start)
                    .where(Reward.status == RewardStatus.COMPLETED)
                )
                
                daily_total = result.scalar() or Decimal('0')
                
                # Check limit
                limit = self.daily_limits.get(reward_type, Decimal('100'))
                
                if daily_total + amount > limit:
                    logger.warning(
                        f"User {user_id} would exceed daily {reward_type} limit: "
                        f"{daily_total} + {amount} > {limit}"
                    )
                    return False
                
                return True
                
        except Exception as e:
            logger.error(f"Error checking daily limit: {e}")
            return False
    
    async def _get_reward_multiplier(self, user_id: int) -> Decimal:
        """
        Get reward multiplier for a user based on their status and activity.
        
        Args:
            user_id: User ID
            
        Returns:
            Reward multiplier
        """
        try:
            async with AsyncSessionLocal() as session:
                # Get user
                result = await session.execute(
                    select(User).where(User.id == user_id)
                )
                user = result.scalar_one_or_none()
                
                if not user:
                    return Decimal('1.0')
                
                # Check account age (days)
                account_age_days = (datetime.utcnow() - user.created_at).days
                
                # Determine multiplier based on user attributes
                if account_age_days < 7:
                    return self.reward_multipliers['new_user']
                elif user.is_moderator:
                    return self.reward_multipliers['moderator']
                elif account_age_days > 365:
                    return self.reward_multipliers['veteran']
                elif account_age_days > 30:
                    # Check if active user (based on recent activity)
                    # This would need additional user activity tracking
                    return self.reward_multipliers['active_user']
                else:
                    return Decimal('1.0')
                
        except Exception as e:
            logger.error(f"Error getting reward multiplier: {e}")
            return Decimal('1.0')
    
    async def _create_transaction(
        self,
        transaction: CoinTransaction,
        status: TransactionStatus = TransactionStatus.PENDING
    ) -> Transaction:
        """
        Create a transaction record.
        
        Args:
            transaction: Transaction data
            status: Initial transaction status
            
        Returns:
            Created transaction record
        """
        try:
            async with AsyncSessionLocal() as session:
                # Generate transaction ID
                transaction_id = str(uuid.uuid4())
                
                # Determine transaction type based on action
                transaction_type_map = {
                    CoinAction.EARN: TransactionType.CREDIT,
                    CoinAction.SPEND: TransactionType.DEBIT,
                    CoinAction.TRANSFER: TransactionType.TRANSFER,
                    CoinAction.WITHDRAW: TransactionType.WITHDRAWAL,
                    CoinAction.DEPOSIT: TransactionType.DEPOSIT,
                    CoinAction.REFUND: TransactionType.REFUND,
                    CoinAction.BURN: TransactionType.BURN,
                    CoinAction.MINT: TransactionType.MINT,
                }
                
                transaction_type = transaction_type_map.get(transaction.action, TransactionType.CREDIT)
                
                # Create transaction
                db_transaction = Transaction(
                    transaction_id=transaction_id,
                    from_user_id=transaction.from_user_id,
                    to_user_id=transaction.to_user_id,
                    amount=float(transaction.amount),
                    transaction_type=transaction_type,
                    status=status,
                    source=transaction.source,
                    reference_id=transaction.reference_id,
                    metadata=json.dumps(transaction.metadata),
                    description=transaction.description,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                
                session.add(db_transaction)
                await session.commit()
                await session.refresh(db_transaction)
                
                logger.info(f"Created transaction {transaction_id}: {transaction.action} {transaction.amount}")
                return db_transaction
                
        except Exception as e:
            logger.error(f"Failed to create transaction: {e}", exc_info=True)
            raise
    
    async def _create_reward(
        self,
        request: RewardRequest,
        status: RewardStatus = RewardStatus.PENDING
    ) -> Reward:
        """
        Create a reward record.
        
        Args:
            request: Reward request
            status: Initial reward status
            
        Returns:
            Created reward record
        """
        try:
            async with AsyncSessionLocal() as session:
                # Generate reward ID
                reward_id = str(uuid.uuid4())
                
                # Apply multiplier
                multiplier = await self._get_reward_multiplier(request.user_id)
                final_amount = request.amount * multiplier
                
                # Create reward
                reward = Reward(
                    reward_id=reward_id,
                    user_id=request.user_id,
                    reward_type=request.reward_type,
                    amount=float(final_amount),
                    multiplier=float(multiplier),
                    source_id=request.source_id,
                    source_type=request.source_type,
                    status=status,
                    metadata=json.dumps(request.metadata),
                    description=request.description,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                
                session.add(reward)
                await session.commit()
                await session.refresh(reward)
                
                logger.info(f"Created reward {reward_id}: {request.reward_type} {final_amount}")
                return reward
                
        except Exception as e:
            logger.error(f"Failed to create reward: {e}", exc_info=True)
            raise
    
    async def _execute_transaction(self, transaction: CoinTransaction) -> Tuple[bool, str]:
        """
        Execute a coin transaction.
        
        Args:
            transaction: Transaction to execute
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            async with AsyncSessionLocal() as session:
                # Start a transaction
                async with session.begin_nested():
                    # Check if this is a transfer between users
                    if transaction.action == CoinAction.TRANSFER:
                        if not transaction.from_user_id or not transaction.to_user_id:
                            return False, "Transfer requires both from and to users"
                        
                        # Get both wallets
                        from_wallet_result = await session.execute(
                            select(Wallet)
                            .where(Wallet.user_id == transaction.from_user_id)
                            .where(Wallet.wallet_type == WalletType.PRIMARY)
                            .with_for_update()  # Lock for update
                        )
                        from_wallet = from_wallet_result.scalar_one_or_none()
                        
                        to_wallet_result = await session.execute(
                            select(Wallet)
                            .where(Wallet.user_id == transaction.to_user_id)
                            .where(Wallet.wallet_type == WalletType.PRIMARY)
                            .with_for_update()
                        )
                        to_wallet = to_wallet_result.scalar_one_or_none()
                        
                        if not from_wallet or not to_wallet:
                            return False, "One or both wallets not found"
                        
                        if from_wallet.status != WalletStatus.ACTIVE:
                            return False, "Sender wallet is not active"
                        
                        if to_wallet.status != WalletStatus.ACTIVE:
                            return False, "Recipient wallet is not active"
                        
                        # Check balance
                        if from_wallet.balance < transaction.amount:
                            return False, "Insufficient funds"
                        
                        # Perform transfer
                        from_wallet.balance -= transaction.amount
                        from_wallet.total_spent += transaction.amount
                        from_wallet.updated_at = datetime.utcnow()
                        
                        to_wallet.balance += transaction.amount
                        to_wallet.total_earned += transaction.amount
                        to_wallet.updated_at = datetime.utcnow()
                        
                    elif transaction.action == CoinAction.EARN:
                        # Credit coins to user
                        if not transaction.to_user_id:
                            return False, "Earn action requires recipient user"
                        
                        wallet_result = await session.execute(
                            select(Wallet)
                            .where(Wallet.user_id == transaction.to_user_id)
                            .where(Wallet.wallet_type == WalletType.PRIMARY)
                            .with_for_update()
                        )
                        wallet = wallet_result.scalar_one_or_none()
                        
                        if not wallet:
                            return False, "Wallet not found"
                        
                        if wallet.status != WalletStatus.ACTIVE:
                            return False, "Wallet is not active"
                        
                        wallet.balance += transaction.amount
                        wallet.total_earned += transaction.amount
                        wallet.updated_at = datetime.utcnow()
                        
                    elif transaction.action == CoinAction.SPEND:
                        # Debit coins from user
                        if not transaction.from_user_id:
                            return False, "Spend action requires sender user"
                        
                        wallet_result = await session.execute(
                            select(Wallet)
                            .where(Wallet.user_id == transaction.from_user_id)
                            .where(Wallet.wallet_type == WalletType.PRIMARY)
                            .with_for_update()
                        )
                        wallet = wallet_result.scalar_one_or_none()
                        
                        if not wallet:
                            return False, "Wallet not found"
                        
                        if wallet.status != WalletStatus.ACTIVE:
                            return False, "Wallet is not active"
                        
                        if wallet.balance < transaction.amount:
                            return False, "Insufficient funds"
                        
                        wallet.balance -= transaction.amount
                        wallet.total_spent += transaction.amount
                        wallet.updated_at = datetime.utcnow()
                    
                    elif transaction.action == CoinAction.WITHDRAW:
                        # Special handling for withdrawals (requires approval)
                        pass
                    
                    elif transaction.action == CoinAction.DEPOSIT:
                        # Special handling for deposits
                        pass
                    
                    else:
                        return False, f"Unsupported action: {transaction.action}"
                    
                    # Create transaction record
                    await self._create_transaction(transaction, TransactionStatus.COMPLETED)
                    
                    return True, ""
                
        except Exception as e:
            logger.error(f"Error executing transaction: {e}", exc_info=True)
            return False, str(e)
    
    async def _execute_reward(self, reward: Reward) -> Tuple[bool, str]:
        """
        Execute a reward transaction.
        
        Args:
            reward: Reward to execute
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Create coin transaction for the reward
            transaction = CoinTransaction(
                from_user_id=None,  # System gives coins
                to_user_id=reward.user_id,
                amount=Decimal(str(reward.amount)),
                action=CoinAction.EARN,
                source=TransactionSource.REWARD,
                reference_id=reward.reward_id,
                metadata={
                    'reward_type': reward.reward_type.value,
                    'multiplier': reward.multiplier,
                    'source_id': reward.source_id,
                    'source_type': reward.source_type,
                },
                description=reward.description
            )
            
            # Execute the transaction
            success, error = await self._execute_transaction(transaction)
            
            if success:
                # Update reward status
                async with AsyncSessionLocal() as session:
                    db_reward = await session.get(Reward, reward.id)
                    if db_reward:
                        db_reward.status = RewardStatus.COMPLETED
                        db_reward.completed_at = datetime.utcnow()
                        await session.commit()
                
                # Emit reward completed event
                await self.event_bus.emit(EventType.REWARD_COMPLETED, {
                    'reward_id': reward.reward_id,
                    'user_id': reward.user_id,
                    'amount': float(reward.amount),
                    'reward_type': reward.reward_type.value,
                    'source_id': reward.source_id,
                    'timestamp': datetime.utcnow().isoformat()
                })
                
                return True, ""
            else:
                # Mark reward as failed
                async with AsyncSessionLocal() as session:
                    db_reward = await session.get(Reward, reward.id)
                    if db_reward:
                        db_reward.status = RewardStatus.FAILED
                        db_reward.error_message = error
                        await session.commit()
                
                return False, error
                
        except Exception as e:
            logger.error(f"Error executing reward: {e}", exc_info=True)
            return False, str(e)
    
    async def _validate_wallet_operation(self, user_id: int, amount: Decimal) -> Tuple[bool, str]:
        """
        Validate if a wallet operation can be performed.
        
        Args:
            user_id: User ID
            amount: Amount to operate with
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            async with AsyncSessionLocal() as session:
                # Get wallet
                result = await session.execute(
                    select(Wallet)
                    .where(Wallet.user_id == user_id)
                    .where(Wallet.wallet_type == WalletType.PRIMARY)
                )
                wallet = result.scalar_one_or_none()
                
                if not wallet:
                    return False, "Wallet not found"
                
                # Check wallet status
                if wallet.status != WalletStatus.ACTIVE:
                    return False, f"Wallet is {wallet.status.value}"
                
                # Check balance if debiting
                if amount < 0 and wallet.balance < abs(amount):
                    return False, "Insufficient funds"
                
                # Check daily transaction limit
                if not await self._check_rate_limit(user_id, 'wallet_operation'):
                    return False, "Daily transaction limit exceeded"
                
                return True, ""
                
        except Exception as e:
            logger.error(f"Error validating wallet operation: {e}")
            return False, str(e)
    
    async def _log_audit_event(self, user_id: Optional[int], action: str, details: Dict[str, Any]):
        """Log wallet operation to audit trail."""
        try:
            await self.event_bus.emit(EventType.AUDIT_LOG_CREATED, {
                'user_id': user_id,
                'action': action,
                'resource_type': 'WALLET',
                'details': details,
                'timestamp': datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.error(f"Failed to emit audit event: {e}")
    
    # Event Handlers
    
    async def handle_incident_reported(self, event):
        """Handle incident report rewards."""
        incident_id = event.data.get('incident_id')
        user_id = event.user_id
        severity = event.data.get('severity', 'medium')
        
        # Determine reward amount based on severity
        reward_amounts = {
            'critical': Decimal('50'),
            'high': Decimal('25'),
            'medium': Decimal('10'),
            'low': Decimal('5'),
        }
        
        base_amount = reward_amounts.get(severity, Decimal('10'))
        
        # Check rate limit
        if not await self._check_rate_limit(user_id, 'incident_report'):
            logger.warning(f"User {user_id} exceeded incident report rate limit")
            return
        
        # Check daily limit
        if not await self._check_daily_limit(user_id, 'incident_report', base_amount):
            logger.warning(f"User {user_id} would exceed daily incident report limit")
            return
        
        # Create reward request
        reward_request = RewardRequest(
            user_id=user_id,
            reward_type=RewardType.INCIDENT_REPORT,
            amount=base_amount,
            source_id=incident_id,
            source_type='incident',
            metadata={
                'incident_id': incident_id,
                'severity': severity,
                'reported_at': event.timestamp.isoformat() if hasattr(event, 'timestamp') else datetime.utcnow().isoformat()
            },
            description=f"Reward for reporting {severity} severity incident"
        )
        
        # Execute reward
        reward = await self._create_reward(reward_request)
        success, error = await self._execute_reward(reward)
        
        if success:
            logger.info(f"Rewarded {base_amount} coins to user {user_id} for incident report")
            
            # Emit coin rewarded event
            await self.event_bus.emit(EventType.COIN_REWARDED, {
                'user_id': user_id,
                'amount': float(base_amount),
                'reason': 'incident_report',
                'incident_id': incident_id,
                'severity': severity,
                'timestamp': datetime.utcnow().isoformat()
            })
        else:
            logger.error(f"Failed to reward user {user_id} for incident report: {error}")
    
    async def handle_incident_verified(self, event):
        """Handle incident verification rewards."""
        incident_id = event.data.get('incident_id')
        user_id = event.user_id  # Verifier user
        verification_type = event.data.get('verification_type', 'community')
        confidence_score = event.data.get('confidence_score', 0)
        
        # Base reward for verification
        base_amount = Decimal('5')
        
        # Bonus for high confidence
        if confidence_score >= 80:
            base_amount += Decimal('10')
        elif confidence_score >= 60:
            base_amount += Decimal('5')
        
        # Check rate limit
        if not await self._check_rate_limit(user_id, 'incident_verification'):
            logger.warning(f"User {user_id} exceeded verification rate limit")
            return
        
        # Check daily limit
        if not await self._check_daily_limit(user_id, 'incident_verification', base_amount):
            logger.warning(f"User {user_id} would exceed daily verification limit")
            return
        
        # Create reward request
        reward_request = RewardRequest(
            user_id=user_id,
            reward_type=RewardType.INCIDENT_VERIFICATION,
            amount=base_amount,
            source_id=incident_id,
            source_type='incident',
            metadata={
                'incident_id': incident_id,
                'verification_type': verification_type,
                'confidence_score': confidence_score,
                'verified_at': event.timestamp.isoformat() if hasattr(event, 'timestamp') else datetime.utcnow().isoformat()
            },
            description=f"Reward for verifying incident ({verification_type})"
        )
        
        # Execute reward
        reward = await self._create_reward(reward_request)
        success, error = await self._execute_reward(reward)
        
        if success:
            logger.info(f"Rewarded {base_amount} coins to user {user_id} for incident verification")
            
            # Also reward the reporter (if different user)
            reporter_id = event.data.get('reporter_id')
            if reporter_id and reporter_id != user_id:
                reporter_amount = Decimal('2')  # Small bonus for reporter
                reporter_request = RewardRequest(
                    user_id=reporter_id,
                    reward_type=RewardType.INCIDENT_VERIFICATION_BONUS,
                    amount=reporter_amount,
                    source_id=incident_id,
                    source_type='incident',
                    metadata={
                        'incident_id': incident_id,
                        'verifier_id': user_id,
                        'verification_type': verification_type,
                    },
                    description=f"Bonus for incident being verified"
                )
                
                reporter_reward = await self._create_reward(reporter_request)
                await self._execute_reward(reporter_reward)
                
                logger.info(f"Rewarded {reporter_amount} coins to reporter {reporter_id}")
            
            # Emit coin rewarded event
            await self.event_bus.emit(EventType.COIN_REWARDED, {
                'user_id': user_id,
                'amount': float(base_amount),
                'reason': 'incident_verification',
                'incident_id': incident_id,
                'verification_type': verification_type,
                'timestamp': datetime.utcnow().isoformat()
            })
        else:
            logger.error(f"Failed to reward user {user_id} for incident verification: {error}")
    
    async def handle_comment_created(self, event):
        """Handle comment creation rewards."""
        comment_id = event.data.get('comment_id')
        user_id = event.user_id
        content_type = event.data.get('content_type', 'post')
        
        # Small reward for commenting
        base_amount = Decimal('1')
        
        # Bonus for detailed comments
        comment_length = len(event.data.get('content', ''))
        if comment_length > 500:
            base_amount += Decimal('2')
        elif comment_length > 200:
            base_amount += Decimal('1')
        
        # Check rate limit
        if not await self._check_rate_limit(user_id, 'comment'):
            logger.warning(f"User {user_id} exceeded comment rate limit")
            return
        
        # Check daily limit
        if not await self._check_daily_limit(user_id, 'comment', base_amount):
            logger.warning(f"User {user_id} would exceed daily comment limit")
            return
        
        # Create reward request
        reward_request = RewardRequest(
            user_id=user_id,
            reward_type=RewardType.COMMENT,
            amount=base_amount,
            source_id=comment_id,
            source_type='comment',
            metadata={
                'comment_id': comment_id,
                'content_type': content_type,
                'content_id': event.data.get('content_id'),
                'comment_length': comment_length,
                'created_at': event.timestamp.isoformat() if hasattr(event, 'timestamp') else datetime.utcnow().isoformat()
            },
            description=f"Reward for creating comment on {content_type}"
        )
        
        # Execute reward
        reward = await self._create_reward(reward_request)
        success, error = await self._execute_reward(reward)
        
        if success:
            logger.info(f"Rewarded {base_amount} coins to user {user_id} for comment")
            
            # Emit coin rewarded event
            await self.event_bus.emit(EventType.COIN_REWARDED, {
                'user_id': user_id,
                'amount': float(base_amount),
                'reason': 'comment_created',
                'comment_id': comment_id,
                'timestamp': datetime.utcnow().isoformat()
            })
        else:
            logger.error(f"Failed to reward user {user_id} for comment: {error}")
    
    async def handle_comment_reply(self, event):
        """Handle comment reply rewards."""
        # Similar to comment creation, but with different reward type
        await self.handle_comment_created(event)
    
    async def handle_briefing_generated(self, event):
        """Handle briefing generation rewards (to system, not user)."""
        # Briefing generation might cost coins or be free based on user tier
        # This is handled elsewhere in the system
        pass
    
    async def handle_briefing_ready(self, event):
        """Handle briefing completion rewards."""
        briefing_id = event.data.get('briefing_id')
        user_id = event.user_id
        briefing_type = event.data.get('briefing_type', 'standard')
        
        # Reward for receiving briefing
        reward_amounts = {
            'comprehensive': Decimal('10'),
            'detailed': Decimal('5'),
            'standard': Decimal('2'),
            'summary': Decimal('1'),
        }
        
        base_amount = reward_amounts.get(briefing_type, Decimal('2'))
        
        # Check daily limit
        if not await self._check_daily_limit(user_id, 'briefing_generation', base_amount):
            logger.warning(f"User {user_id} would exceed daily briefing limit")
            return
        
        # Create reward request
        reward_request = RewardRequest(
            user_id=user_id,
            reward_type=RewardType.BRIEFING_GENERATION,
            amount=base_amount,
            source_id=briefing_id,
            source_type='briefing',
            metadata={
                'briefing_id': briefing_id,
                'briefing_type': briefing_type,
                'generated_at': event.timestamp.isoformat() if hasattr(event, 'timestamp') else datetime.utcnow().isoformat()
            },
            description=f"Reward for {briefing_type} briefing"
        )
        
        # Execute reward
        reward = await self._create_reward(reward_request)
        success, error = await self._execute_reward(reward)
        
        if success:
            logger.info(f"Rewarded {base_amount} coins to user {user_id} for briefing")
            
            # Emit coin rewarded event
            await self.event_bus.emit(EventType.COIN_REWARDED, {
                'user_id': user_id,
                'amount': float(base_amount),
                'reason': 'briefing_generated',
                'briefing_id': briefing_id,
                'briefing_type': briefing_type,
                'timestamp': datetime.utcnow().isoformat()
            })
        else:
            logger.error(f"Failed to reward user {user_id} for briefing: {error}")
    
    async def handle_chat_message(self, event):
        """Handle chat message rewards."""
        message_id = event.data.get('message_id')
        user_id = event.user_id
        chat_type = event.data.get('chat_type', 'direct')
        
        # Small reward for active chat participation
        base_amount = Decimal('0.5')
        
        # Bonus for helpful answers in Q&A chats
        if chat_type == 'qa' and event.data.get('is_answer', False):
            base_amount = Decimal('2')
        
        # Check rate limit
        if not await self._check_rate_limit(user_id, 'chat_interaction'):
            logger.warning(f"User {user_id} exceeded chat rate limit")
            return
        
        # Check daily limit
        if not await self._check_daily_limit(user_id, 'chat_interaction', base_amount):
            logger.warning(f"User {user_id} would exceed daily chat limit")
            return
        
        # Create reward request
        reward_request = RewardRequest(
            user_id=user_id,
            reward_type=RewardType.CHAT_INTERACTION,
            amount=base_amount,
            source_id=message_id,
            source_type='chat_message',
            metadata={
                'message_id': message_id,
                'chat_type': chat_type,
                'chat_id': event.data.get('chat_id'),
                'is_answer': event.data.get('is_answer', False),
                'sent_at': event.timestamp.isoformat() if hasattr(event, 'timestamp') else datetime.utcnow().isoformat()
            },
            description=f"Reward for chat participation ({chat_type})"
        )
        
        # Execute reward
        reward = await self._create_reward(reward_request)
        success, error = await self._execute_reward(reward)
        
        if success:
            logger.info(f"Rewarded {base_amount} coins to user {user_id} for chat")
            
            # Emit coin rewarded event
            await self.event_bus.emit(EventType.COIN_REWARDED, {
                'user_id': user_id,
                'amount': float(base_amount),
                'reason': 'chat_participation',
                'chat_type': chat_type,
                'timestamp': datetime.utcnow().isoformat()
            })
        else:
            logger.error(f"Failed to reward user {user_id} for chat: {error}")
    
    async def handle_content_created(self, event):
        """Handle content creation rewards."""
        content_id = event.data.get('content_id')
        user_id = event.user_id
        content_type = event.data.get('content_type', 'post')
        
        # Base reward for content creation
        reward_amounts = {
            'article': Decimal('20'),
            'analysis': Decimal('15'),
            'report': Decimal('10'),
            'post': Decimal('5'),
            'image': Decimal('3'),
            'video': Decimal('8'),
        }
        
        base_amount = reward_amounts.get(content_type, Decimal('5'))
        
        # Check daily limit
        daily_limit_type = f'content_{content_type}'
        if not await self._check_daily_limit(user_id, daily_limit_type, base_amount):
            logger.warning(f"User {user_id} would exceed daily {content_type} limit")
            return
        
        # Create reward request
        reward_request = RewardRequest(
            user_id=user_id,
            reward_type=RewardType.CONTENT_CREATION,
            amount=base_amount,
            source_id=content_id,
            source_type=content_type,
            metadata={
                'content_id': content_id,
                'content_type': content_type,
                'created_at': event.timestamp.isoformat() if hasattr(event, 'timestamp') else datetime.utcnow().isoformat()
            },
            description=f"Reward for creating {content_type}"
        )
        
        # Execute reward
        reward = await self._create_reward(reward_request)
        success, error = await self._execute_reward(reward)
        
        if success:
            logger.info(f"Rewarded {base_amount} coins to user {user_id} for {content_type}")
            
            # Emit coin rewarded event
            await self.event_bus.emit(EventType.COIN_REWARDED, {
                'user_id': user_id,
                'amount': float(base_amount),
                'reason': f'content_creation_{content_type}',
                'content_id': content_id,
                'content_type': content_type,
                'timestamp': datetime.utcnow().isoformat()
            })
        else:
            logger.error(f"Failed to reward user {user_id} for {content_type}: {error}")
    
    async def handle_content_approved(self, event):
        """Handle content approval rewards (to approver)."""
        content_id = event.data.get('content_id')
        user_id = event.user_id  # Approver user
        content_type = event.data.get('content_type', 'post')
        
        # Reward for moderation work
        base_amount = Decimal('2')
        
        # Create reward request
        reward_request = RewardRequest(
            user_id=user_id,
            reward_type=RewardType.CONTENT_MODERATION,
            amount=base_amount,
            source_id=content_id,
            source_type=f'{content_type}_moderation',
            metadata={
                'content_id': content_id,
                'content_type': content_type,
                'approved_at': event.timestamp.isoformat() if hasattr(event, 'timestamp') else datetime.utcnow().isoformat()
            },
            description=f"Reward for approving {content_type}"
        )
        
        # Execute reward
        reward = await self._create_reward(reward_request)
        success, error = await self._execute_reward(reward)
        
        if success:
            logger.info(f"Rewarded {base_amount} coins to moderator {user_id} for content approval")
        else:
            logger.error(f"Failed to reward moderator {user_id} for content approval: {error}")
    
    async def handle_content_featured(self, event):
        """Handle featured content bonus rewards."""
        content_id = event.data.get('content_id')
        user_id = event.data.get('creator_id')
        
        # Bonus for featured content
        base_amount = Decimal('25')
        
        # Create reward request
        reward_request = RewardRequest(
            user_id=user_id,
            reward_type=RewardType.CONTENT_FEATURED,
            amount=base_amount,
            source_id=content_id,
            source_type='featured_content',
            metadata={
                'content_id': content_id,
                'featured_at': event.timestamp.isoformat() if hasattr(event, 'timestamp') else datetime.utcnow().isoformat(),
                'feature_type': event.data.get('feature_type', 'editor_pick')
            },
            description="Bonus for featured content"
        )
        
        # Execute reward
        reward = await self._create_reward(reward_request)
        success, error = await self._execute_reward(reward)
        
        if success:
            logger.info(f"Rewarded {base_amount} coins to user {user_id} for featured content")
        else:
            logger.error(f"Failed to reward user {user_id} for featured content: {error}")
    
    async def handle_content_viral(self, event):
        """Handle viral content bonus rewards."""
        content_id = event.data.get('content_id')
        user_id = event.data.get('creator_id')
        view_count = event.data.get('view_count', 0)
        
        # Bonus based on virality
        if view_count > 10000:
            base_amount = Decimal('100')
        elif view_count > 5000:
            base_amount = Decimal('50')
        elif view_count > 1000:
            base_amount = Decimal('25')
        else:
            base_amount = Decimal('10')
        
        # Create reward request
        reward_request = RewardRequest(
            user_id=user_id,
            reward_type=RewardType.CONTENT_VIRAL,
            amount=base_amount,
            source_id=content_id,
            source_type='viral_content',
            metadata={
                'content_id': content_id,
                'view_count': view_count,
                'went_viral_at': event.timestamp.isoformat() if hasattr(event, 'timestamp') else datetime.utcnow().isoformat()
            },
            description=f"Bonus for viral content ({view_count} views)"
        )
        
        # Execute reward
        reward = await self._create_reward(reward_request)
        success, error = await self._execute_reward(reward)
        
        if success:
            logger.info(f"Rewarded {base_amount} coins to user {user_id} for viral content")
        else:
            logger.error(f"Failed to reward user {user_id} for viral content: {error}")
    
    async def handle_moderation_action(self, event):
        """Handle moderation action rewards."""
        action = event.data.get('action')
        moderator_id = event.user_id
        
        # Different rewards for different moderation actions
        reward_amounts = {
            'warn_user': Decimal('1'),
            'remove_content': Decimal('2'),
            'suspend_user': Decimal('3'),
            'ban_user': Decimal('5'),
            'resolve_appeal': Decimal('2'),
        }
        
        base_amount = reward_amounts.get(action, Decimal('1'))
        
        # Create reward request
        reward_request = RewardRequest(
            user_id=moderator_id,
            reward_type=RewardType.MODERATION_ACTION,
            amount=base_amount,
            source_id=event.data.get('reference_id'),
            source_type='moderation',
            metadata={
                'action': action,
                'target_user_id': event.data.get('user_id'),
                'reason': event.data.get('reason'),
                'performed_at': event.timestamp.isoformat() if hasattr(event, 'timestamp') else datetime.utcnow().isoformat()
            },
            description=f"Reward for moderation action: {action}"
        )
        
        # Execute reward
        reward = await self._create_reward(reward_request)
        success, error = await self._execute_reward(reward)
        
        if success:
            logger.info(f"Rewarded {base_amount} coins to moderator {moderator_id} for {action}")
        else:
            logger.error(f"Failed to reward moderator {moderator_id} for {action}: {error}")
    
    async def handle_appeal_reviewed(self, event):
        """Handle appeal review rewards."""
        appeal_id = event.data.get('appeal_id')
        reviewer_id = event.user_id
        approved = event.data.get('approved', False)
        
        # Reward for reviewing appeals
        base_amount = Decimal('3')
        
        # Create reward request
        reward_request = RewardRequest(
            user_id=reviewer_id,
            reward_type=RewardType.APPEAL_REVIEW,
            amount=base_amount,
            source_id=appeal_id,
            source_type='appeal',
            metadata={
                'appeal_id': appeal_id,
                'approved': approved,
                'reviewed_at': event.timestamp.isoformat() if hasattr(event, 'timestamp') else datetime.utcnow().isoformat()
            },
            description=f"Reward for reviewing appeal ({'approved' if approved else 'rejected'})"
        )
        
        # Execute reward
        reward = await self._create_reward(reward_request)
        success, error = await self._execute_reward(reward)
        
        if success:
            logger.info(f"Rewarded {base_amount} coins to reviewer {reviewer_id} for appeal review")
        else:
            logger.error(f"Failed to reward reviewer {reviewer_id} for appeal review: {error}")
    
    async def handle_user_reported_valid(self, event):
        """Handle valid user report rewards."""
        reporter_id = event.user_id
        reported_user_id = event.data.get('reported_user_id')
        report_valid = event.data.get('valid', True)
        
        if not report_valid:
            return  # No reward for invalid reports
        
        # Reward for valid reports
        base_amount = Decimal('2')
        
        # Create reward request
        reward_request = RewardRequest(
            user_id=reporter_id,
            reward_type=RewardType.VALID_REPORT,
            amount=base_amount,
            source_id=str(reported_user_id),
            source_type='user_report',
            metadata={
                'reported_user_id': reported_user_id,
                'report_reason': event.data.get('reason'),
                'validated_at': event.timestamp.isoformat() if hasattr(event, 'timestamp') else datetime.utcnow().isoformat()
            },
            description="Reward for valid user report"
        )
        
        # Execute reward
        reward = await self._create_reward(reward_request)
        success, error = await self._execute_reward(reward)
        
        if success:
            logger.info(f"Rewarded {base_amount} coins to reporter {reporter_id} for valid report")
        else:
            logger.error(f"Failed to reward reporter {reporter_id} for valid report: {error}")
    
    async def handle_coin_transfer(self, event):
        """Handle coin transfer between users."""
        from_user_id = event.data.get('from_user_id')
        to_user_id = event.data.get('to_user_id')
        amount = Decimal(str(event.data.get('amount', 0)))
        description = event.data.get('description', '')
        
        # Validate the transfer
        is_valid, error = await self._validate_wallet_operation(from_user_id, -amount)
        if not is_valid:
            logger.error(f"Invalid coin transfer: {error}")
            
            # Emit transfer failed event
            await self.event_bus.emit(EventType.COIN_TRANSFER_FAILED, {
                'from_user_id': from_user_id,
                'to_user_id': to_user_id,
                'amount': float(amount),
                'reason': error,
                'timestamp': datetime.utcnow().isoformat()
            })
            return
        
        # Create transaction
        transaction = CoinTransaction(
            from_user_id=from_user_id,
            to_user_id=to_user_id,
            amount=amount,
            action=CoinAction.TRANSFER,
            source=TransactionSource.USER_TRANSFER,
            metadata={
                'initiated_by': event.user_id,
                'description': description,
            },
            description=description or f"Transfer to user {to_user_id}"
        )
        
        # Execute transaction
        success, error = await self._execute_transaction(transaction)
        
        if success:
            logger.info(f"Transferred {amount} coins from user {from_user_id} to user {to_user_id}")
            
            # Emit transfer completed event
            await self.event_bus.emit(EventType.COIN_TRANSFER_COMPLETED, {
                'from_user_id': from_user_id,
                'to_user_id': to_user_id,
                'amount': float(amount),
                'transaction_id': transaction.reference_id,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Log audit event
            await self._log_audit_event(from_user_id, 'COIN_TRANSFER', {
                'to_user_id': to_user_id,
                'amount': float(amount),
                'transaction_id': transaction.reference_id
            })
        else:
            logger.error(f"Failed to transfer coins: {error}")
            
            # Emit transfer failed event
            await self.event_bus.emit(EventType.COIN_TRANSFER_FAILED, {
                'from_user_id': from_user_id,
                'to_user_id': to_user_id,
                'amount': float(amount),
                'reason': error,
                'timestamp': datetime.utcnow().isoformat()
            })
    
    async def handle_withdrawal_requested(self, event):
        """Handle coin withdrawal request."""
        user_id = event.user_id
        amount = Decimal(str(event.data.get('amount', 0)))
        wallet_address = event.data.get('wallet_address')
        network = event.data.get('network', 'ETH')
        
        # Validate withdrawal
        is_valid, error = await self._validate_wallet_operation(user_id, -amount)
        if not is_valid:
            logger.error(f"Invalid withdrawal request: {error}")
            
            # Emit withdrawal failed event
            await self.event_bus.emit(EventType.WITHDRAWAL_FAILED, {
                'user_id': user_id,
                'amount': float(amount),
                'reason': error,
                'timestamp': datetime.utcnow().isoformat()
            })
            return
        
        try:
            async with AsyncSessionLocal() as session:
                # Create withdrawal record
                withdrawal = Withdrawal(
                    user_id=user_id,
                    amount=float(amount),
                    wallet_address=wallet_address,
                    network=network,
                    status=WithdrawalStatus.PENDING,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                
                session.add(withdrawal)
                await session.commit()
                await session.refresh(withdrawal)
                
                # Lock coins (create a pending transaction)
                transaction = CoinTransaction(
                    from_user_id=user_id,
                    to_user_id=None,
                    amount=amount,
                    action=CoinAction.WITHDRAW,
                    source=TransactionSource.WITHDRAWAL,
                    reference_id=str(withdrawal.id),
                    metadata={
                        'wallet_address': wallet_address,
                        'network': network,
                        'withdrawal_id': withdrawal.id
                    },
                    description=f"Withdrawal to {network} wallet"
                )
                
                # Create pending transaction (coins are locked but not yet sent)
                await self._create_transaction(transaction, TransactionStatus.PENDING)
                
                logger.info(f"Created withdrawal request {withdrawal.id} for user {user_id}: {amount} coins")
                
                # Emit withdrawal requested event
                await self.event_bus.emit(EventType.WITHDRAWAL_REQUESTED, {
                    'user_id': user_id,
                    'amount': float(amount),
                    'wallet_address': wallet_address,
                    'network': network,
                    'withdrawal_id': withdrawal.id,
                    'timestamp': datetime.utcnow().isoformat()
                })
                
                # Log audit event
                await self._log_audit_event(user_id, 'WITHDRAWAL_REQUESTED', {
                    'amount': float(amount),
                    'wallet_address': wallet_address,
                    'network': network,
                    'withdrawal_id': withdrawal.id
                })
                
        except Exception as e:
            logger.error(f"Failed to create withdrawal request: {e}", exc_info=True)
            
            # Emit withdrawal failed event
            await self.event_bus.emit(EventType.WITHDRAWAL_FAILED, {
                'user_id': user_id,
                'amount': float(amount),
                'reason': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
    
    async def handle_withdrawal_approved(self, event):
        """Handle withdrawal approval by admin."""
        withdrawal_id = event.data.get('withdrawal_id')
        approved_by = event.user_id
        
        try:
            async with AsyncSessionLocal() as session:
                # Get withdrawal
                result = await session.execute(
                    select(Withdrawal).where(Withdrawal.id == withdrawal_id)
                )
                withdrawal = result.scalar_one_or_none()
                
                if not withdrawal:
                    logger.error(f"Withdrawal {withdrawal_id} not found")
                    return
                
                if withdrawal.status != WithdrawalStatus.PENDING:
                    logger.error(f"Withdrawal {withdrawal_id} is not pending")
                    return
                
                # Update withdrawal status
                withdrawal.status = WithdrawalStatus.APPROVED
                withdrawal.approved_by = approved_by
                withdrawal.approved_at = datetime.utcnow()
                withdrawal.updated_at = datetime.utcnow()
                
                # Update transaction status
                result = await session.execute(
                    select(Transaction)
                    .where(Transaction.reference_id == str(withdrawal_id))
                    .where(Transaction.source == TransactionSource.WITHDRAWAL)
                )
                transaction = result.scalar_one_or_none()
                
                if transaction:
                    transaction.status = TransactionStatus.COMPLETED
                    transaction.updated_at = datetime.utcnow()
                
                await session.commit()
                
                logger.info(f"Approved withdrawal {withdrawal_id}")
                
                # Emit withdrawal approved event
                await self.event_bus.emit(EventType.WITHDRAWAL_APPROVED, {
                    'withdrawal_id': withdrawal_id,
                    'user_id': withdrawal.user_id,
                    'amount': withdrawal.amount,
                    'approved_by': approved_by,
                    'timestamp': datetime.utcnow().isoformat()
                })
                
                # Log audit event
                await self._log_audit_event(approved_by, 'WITHDRAWAL_APPROVED', {
                    'withdrawal_id': withdrawal_id,
                    'user_id': withdrawal.user_id,
                    'amount': withdrawal.amount
                })
                
        except Exception as e:
            logger.error(f"Failed to approve withdrawal: {e}", exc_info=True)
    
    async def handle_withdrawal_completed(self, event):
        """Handle withdrawal completion (coins sent to external wallet)."""
        withdrawal_id = event.data.get('withdrawal_id')
        transaction_hash = event.data.get('transaction_hash')
        
        try:
            async with AsyncSessionLocal() as session:
                # Get withdrawal
                result = await session.execute(
                    select(Withdrawal).where(Withdrawal.id == withdrawal_id)
                )
                withdrawal = result.scalar_one_or_none()
                
                if not withdrawal:
                    logger.error(f"Withdrawal {withdrawal_id} not found")
                    return
                
                if withdrawal.status != WithdrawalStatus.APPROVED:
                    logger.error(f"Withdrawal {withdrawal_id} is not approved")
                    return
                
                # Update withdrawal status
                withdrawal.status = WithdrawalStatus.COMPLETED
                withdrawal.transaction_hash = transaction_hash
                withdrawal.completed_at = datetime.utcnow()
                withdrawal.updated_at = datetime.utcnow()
                
                # Deduct coins from user's wallet
                wallet_result = await session.execute(
                    select(Wallet)
                    .where(Wallet.user_id == withdrawal.user_id)
                    .where(Wallet.wallet_type == WalletType.PRIMARY)
                    .with_for_update()
                )
                wallet = wallet_result.scalar_one_or_none()
                
                if wallet:
                    wallet.balance -= Decimal(str(withdrawal.amount))
                    wallet.total_spent += Decimal(str(withdrawal.amount))
                    wallet.updated_at = datetime.utcnow()
                
                await session.commit()
                
                logger.info(f"Completed withdrawal {withdrawal_id}, transaction hash: {transaction_hash}")
                
                # Emit withdrawal completed event
                await self.event_bus.emit(EventType.WITHDRAWAL_COMPLETED, {
                    'withdrawal_id': withdrawal_id,
                    'user_id': withdrawal.user_id,
                    'amount': withdrawal.amount,
                    'transaction_hash': transaction_hash,
                    'timestamp': datetime.utcnow().isoformat()
                })
                
                # Log audit event
                await self._log_audit_event(withdrawal.user_id, 'WITHDRAWAL_COMPLETED', {
                    'withdrawal_id': withdrawal_id,
                    'amount': withdrawal.amount,
                    'transaction_hash': transaction_hash
                })
                
        except Exception as e:
            logger.error(f"Failed to complete withdrawal: {e}", exc_info=True)
    
    async def handle_withdrawal_failed(self, event):
        """Handle withdrawal failure."""
        withdrawal_id = event.data.get('withdrawal_id')
        reason = event.data.get('reason', 'Unknown error')
        
        try:
            async with AsyncSessionLocal() as session:
                # Get withdrawal
                result = await session.execute(
                    select(Withdrawal).where(Withdrawal.id == withdrawal_id)
                )
                withdrawal = result.scalar_one_or_none()
                
                if not withdrawal:
                    logger.error(f"Withdrawal {withdrawal_id} not found")
                    return
                
                # Update withdrawal status
                withdrawal.status = WithdrawalStatus.FAILED
                withdrawal.failure_reason = reason
                withdrawal.updated_at = datetime.utcnow()
                
                # Update transaction status
                result = await session.execute(
                    select(Transaction)
                    .where(Transaction.reference_id == str(withdrawal_id))
                    .where(Transaction.source == TransactionSource.WITHDRAWAL)
                )
                transaction = result.scalar_one_or_none()
                
                if transaction:
                    transaction.status = TransactionStatus.FAILED
                    transaction.updated_at = datetime.utcnow()
                
                await session.commit()
                
                logger.info(f"Failed withdrawal {withdrawal_id}: {reason}")
                
                # Emit withdrawal failed event
                await self.event_bus.emit(EventType.WITHDRAWAL_FAILED, {
                    'withdrawal_id': withdrawal_id,
                    'user_id': withdrawal.user_id,
                    'amount': withdrawal.amount,
                    'reason': reason,
                    'timestamp': datetime.utcnow().isoformat()
                })
                
        except Exception as e:
            logger.error(f"Failed to mark withdrawal as failed: {e}", exc_info=True)
    
    async def handle_deposit_received(self, event):
        """Handle coin deposit from external source."""
        user_id = event.data.get('user_id')
        amount = Decimal(str(event.data.get('amount', 0)))
        transaction_hash = event.data.get('transaction_hash')
        network = event.data.get('network', 'ETH')
        
        try:
            # Create deposit transaction
            transaction = CoinTransaction(
                from_user_id=None,  # External source
                to_user_id=user_id,
                amount=amount,
                action=CoinAction.DEPOSIT,
                source=TransactionSource.DEPOSIT,
                metadata={
                    'transaction_hash': transaction_hash,
                    'network': network,
                    'deposited_at': event.timestamp.isoformat() if hasattr(event, 'timestamp') else datetime.utcnow().isoformat()
                },
                description=f"Deposit from {network} network"
            )
            
            # Execute transaction
            success, error = await self._execute_transaction(transaction)
            
            if success:
                logger.info(f"Deposited {amount} coins to user {user_id}, transaction hash: {transaction_hash}")
                
                # Emit deposit received event
                await self.event_bus.emit(EventType.DEPOSIT_RECEIVED, {
                    'user_id': user_id,
                    'amount': float(amount),
                    'transaction_hash': transaction_hash,
                    'network': network,
                    'timestamp': datetime.utcnow().isoformat()
                })
                
                # Log audit event
                await self._log_audit_event(user_id, 'DEPOSIT_RECEIVED', {
                    'amount': float(amount),
                    'transaction_hash': transaction_hash,
                    'network': network
                })
            else:
                logger.error(f"Failed to process deposit: {error}")
                
        except Exception as e:
            logger.error(f"Failed to handle deposit: {e}", exc_info=True)
    
    async def handle_referral_completed(self, event):
        """Handle referral completion rewards."""
        referrer_id = event.data.get('referrer_id')
        referred_id = event.data.get('referred_id')
        
        # Reward for successful referral
        base_amount = Decimal('50')
        
        # Create reward request
        reward_request = RewardRequest(
            user_id=referrer_id,
            reward_type=RewardType.REFERRAL,
            amount=base_amount,
            source_id=str(referred_id),
            source_type='referral',
            metadata={
                'referred_user_id': referred_id,
                'completed_at': event.timestamp.isoformat() if hasattr(event, 'timestamp') else datetime.utcnow().isoformat()
            },
            description="Reward for successful referral"
        )
        
        # Execute reward
        reward = await self._create_reward(reward_request)
        success, error = await self._execute_reward(reward)
        
        if success:
            logger.info(f"Rewarded {base_amount} coins to referrer {referrer_id} for referral")
            
            # Also reward the referred user (welcome bonus)
            welcome_amount = Decimal('25')
            welcome_request = RewardRequest(
                user_id=referred_id,
                reward_type=RewardType.WELCOME_BONUS,
                amount=welcome_amount,
                source_id=str(referrer_id),
                source_type='referral_welcome',
                metadata={
                    'referrer_id': referrer_id,
                    'received_at': datetime.utcnow().isoformat()
                },
                description="Welcome bonus for joining via referral"
            )
            
            welcome_reward = await self._create_reward(welcome_request)
            await self._execute_reward(welcome_reward)
            
            logger.info(f"Rewarded {welcome_amount} coins to referred user {referred_id}")
        else:
            logger.error(f"Failed to reward referrer {referrer_id} for referral: {error}")
    
    async def handle_achievement_unlocked(self, event):
        """Handle achievement unlocked rewards."""
        user_id = event.data.get('user_id')
        achievement_id = event.data.get('achievement_id')
        achievement_name = event.data.get('achievement_name', '')
        
        # Reward based on achievement difficulty
        reward_amounts = {
            'bronze': Decimal('10'),
            'silver': Decimal('25'),
            'gold': Decimal('50'),
            'platinum': Decimal('100'),
            'diamond': Decimal('250'),
        }
        
        # Determine reward tier from achievement name or ID
        tier = 'bronze'  # Default
        for reward_tier in reward_amounts.keys():
            if reward_tier in achievement_name.lower():
                tier = reward_tier
                break
        
        base_amount = reward_amounts.get(tier, Decimal('10'))
        
        # Create reward request
        reward_request = RewardRequest(
            user_id=user_id,
            reward_type=RewardType.ACHIEVEMENT,
            amount=base_amount,
            source_id=achievement_id,
            source_type='achievement',
            metadata={
                'achievement_id': achievement_id,
                'achievement_name': achievement_name,
                'tier': tier,
                'unlocked_at': event.timestamp.isoformat() if hasattr(event, 'timestamp') else datetime.utcnow().isoformat()
            },
            description=f"Reward for unlocking achievement: {achievement_name}"
        )
        
        # Execute reward
        reward = await self._create_reward(reward_request)
        success, error = await self._execute_reward(reward)
        
        if success:
            logger.info(f"Rewarded {base_amount} coins to user {user_id} for achievement {achievement_name}")
        else:
            logger.error(f"Failed to reward user {user_id} for achievement: {error}")
    
    async def handle_daily_login(self, event):
        """Handle daily login rewards."""
        user_id = event.user_id
        streak_days = event.data.get('streak_days', 1)
        
        # Base reward for daily login
        base_amount = Decimal('1')
        
        # Streak bonus
        if streak_days >= 7:
            base_amount += Decimal('5')  # Weekly bonus
        if streak_days >= 30:
            base_amount += Decimal('25')  # Monthly bonus
        
        # Check daily limit
        if not await self._check_daily_limit(user_id, 'daily_login', base_amount):
            logger.warning(f"User {user_id} would exceed daily login limit")
            return
        
        # Create reward request
        reward_request = RewardRequest(
            user_id=user_id,
            reward_type=RewardType.DAILY_LOGIN,
            amount=base_amount,
            source_id=str(streak_days),
            source_type='daily_login',
            metadata={
                'streak_days': streak_days,
                'login_date': datetime.utcnow().date().isoformat(),
                'logged_in_at': event.timestamp.isoformat() if hasattr(event, 'timestamp') else datetime.utcnow().isoformat()
            },
            description=f"Daily login reward ({streak_days} day streak)"
        )
        
        # Execute reward
        reward = await self._create_reward(reward_request)
        success, error = await self._execute_reward(reward)
        
        if success:
            logger.info(f"Rewarded {base_amount} coins to user {user_id} for daily login (streak: {streak_days} days)")
        else:
            logger.error(f"Failed to reward user {user_id} for daily login: {error}")
    
    async def handle_streak_maintained(self, event):
        """Handle streak maintenance bonus."""
        user_id = event.data.get('user_id')
        streak_type = event.data.get('streak_type', 'daily')
        streak_days = event.data.get('streak_days', 0)
        
        # Bonus for maintaining streaks
        if streak_type == 'daily':
            if streak_days == 7:
                bonus_amount = Decimal('10')
            elif streak_days == 30:
                bonus_amount = Decimal('50')
            elif streak_days == 100:
                bonus_amount = Decimal('100')
            else:
                return  # No bonus for other days
            
            # Create reward request
            reward_request = RewardRequest(
                user_id=user_id,
                reward_type=RewardType.STREAK_BONUS,
                amount=bonus_amount,
                source_id=f"{streak_type}_{streak_days}",
                source_type='streak',
                metadata={
                    'streak_type': streak_type,
                    'streak_days': streak_days,
                    'milestone_reached_at': event.timestamp.isoformat() if hasattr(event, 'timestamp') else datetime.utcnow().isoformat()
                },
                description=f"Streak bonus: {streak_days} days {streak_type} streak"
            )
            
            # Execute reward
            reward = await self._create_reward(reward_request)
            success, error = await self._execute_reward(reward)
            
            if success:
                logger.info(f"Rewarded {bonus_amount} coins to user {user_id} for {streak_days} day streak")
            else:
                logger.error(f"Failed to reward user {user_id} for streak: {error}")
    
    async def handle_coin_reward_request(self, event):
        """Handle direct coin reward request (manual rewards)."""
        user_id = event.data.get('user_id')
        amount = Decimal(str(event.data.get('amount', 0)))
        reason = event.data.get('reason', '')
        
        # Only allow from admins or system
        if not await self._is_admin_or_system(event.user_id):
            logger.warning(f"Non-admin user {event.user_id} attempted manual reward")
            return
        
        # Create reward request
        reward_request = RewardRequest(
            user_id=user_id,
            reward_type=RewardType.MANUAL,
            amount=amount,
            source_id=event.data.get('reference_id'),
            source_type='manual_reward',
            metadata={
                'rewarded_by': event.user_id,
                'reason': reason,
                'rewarded_at': datetime.utcnow().isoformat()
            },
            description=f"Manual reward: {reason}"
        )
        
        # Execute reward
        reward = await self._create_reward(reward_request)
        success, error = await self._execute_reward(reward)
        
        if success:
            logger.info(f"Manually rewarded {amount} coins to user {user_id} by {event.user_id}")
            
            # Log audit event
            await self._log_audit_event(event.user_id, 'MANUAL_REWARD', {
                'target_user_id': user_id,
                'amount': float(amount),
                'reason': reason
            })
        else:
            logger.error(f"Failed to manually reward user {user_id}: {error}")
    
    async def handle_coin_purchase(self, event):
        """Handle coin purchase with fiat currency."""
        user_id = event.user_id
        amount = Decimal(str(event.data.get('amount', 0)))
        fiat_amount = Decimal(str(event.data.get('fiat_amount', 0)))
        currency = event.data.get('currency', 'USD')
        payment_method = event.data.get('payment_method', 'stripe')
        
        # Create purchase transaction
        transaction = CoinTransaction(
            from_user_id=None,  # System provides coins
            to_user_id=user_id,
            amount=amount,
            action=CoinAction.DEPOSIT,
            source=TransactionSource.PURCHASE,
            metadata={
                'fiat_amount': float(fiat_amount),
                'currency': currency,
                'payment_method': payment_method,
                'purchase_id': event.data.get('purchase_id'),
                'purchased_at': datetime.utcnow().isoformat()
            },
            description=f"Coin purchase ({fiat_amount} {currency})"
        )
        
        # Execute transaction
        success, error = await self._execute_transaction(transaction)
        
        if success:
            logger.info(f"Purchased {amount} coins for user {user_id} for {fiat_amount} {currency}")
            
            # Emit purchase completed event
            await self.event_bus.emit(EventType.COIN_PURCHASE_COMPLETED, {
                'user_id': user_id,
                'coin_amount': float(amount),
                'fiat_amount': float(fiat_amount),
                'currency': currency,
                'payment_method': payment_method,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Log audit event
            await self._log_audit_event(user_id, 'COIN_PURCHASE', {
                'coin_amount': float(amount),
                'fiat_amount': float(fiat_amount),
                'currency': currency,
                'payment_method': payment_method
            })
        else:
            logger.error(f"Failed to process coin purchase: {error}")
            
            # Emit purchase failed event
            await self.event_bus.emit(EventType.COIN_PURCHASE_FAILED, {
                'user_id': user_id,
                'coin_amount': float(amount),
                'fiat_amount': float(fiat_amount),
                'currency': currency,
                'reason': error,
                'timestamp': datetime.utcnow().isoformat()
            })
    
    async def handle_coin_burn(self, event):
        """Handle coin burn (destroy coins)."""
        user_id = event.data.get('user_id')
        amount = Decimal(str(event.data.get('amount', 0)))
        reason = event.data.get('reason', '')
        
        # Only allow from admins or system
        if not await self._is_admin_or_system(event.user_id):
            logger.warning(f"Non-admin user {event.user_id} attempted coin burn")
            return
        
        # Validate user has enough coins
        is_valid, error = await self._validate_wallet_operation(user_id, -amount)
        if not is_valid:
            logger.error(f"Invalid coin burn: {error}")
            return
        
        # Create burn transaction
        transaction = CoinTransaction(
            from_user_id=user_id,
            to_user_id=None,
            amount=amount,
            action=CoinAction.BURN,
            source=TransactionSource.SYSTEM,
            metadata={
                'burned_by': event.user_id,
                'reason': reason,
                'burned_at': datetime.utcnow().isoformat()
            },
            description=f"Coin burn: {reason}"
        )
        
        # Execute transaction
        success, error = await self._execute_transaction(transaction)
        
        if success:
            logger.info(f"Burned {amount} coins from user {user_id} by {event.user_id}")
            
            # Log audit event
            await self._log_audit_event(event.user_id, 'COIN_BURN', {
                'target_user_id': user_id,
                'amount': float(amount),
                'reason': reason
            })
        else:
            logger.error(f"Failed to burn coins: {error}")
    
    async def _is_admin_or_system(self, user_id: int) -> bool:
        """Check if user is admin or system."""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(User).where(User.id == user_id)
                )
                user = result.scalar_one_or_none()
                
                if not user:
                    return False
                
                return user.is_admin or user.is_system
                
        except Exception as e:
            logger.error(f"Error checking admin status: {e}")
            return False
    
    # Background tasks
    
    async def _process_transaction_queue(self):
        """Process pending transactions in the queue."""
        while True:
            try:
                async with AsyncSessionLocal() as session:
                    # Get pending transactions
                    result = await session.execute(
                        select(Transaction)
                        .where(Transaction.status == TransactionStatus.PENDING)
                        .where(Transaction.created_at >= datetime.utcnow() - timedelta(hours=24))
                        .order_by(Transaction.created_at.asc())
                        .limit(50)
                    )
                    
                    pending_transactions = result.scalars().all()
                    
                    for transaction in pending_transactions:
                        try:
                            # Convert to CoinTransaction
                            coin_transaction = CoinTransaction(
                                from_user_id=transaction.from_user_id,
                                to_user_id=transaction.to_user_id,
                                amount=Decimal(str(transaction.amount)),
                                action=self._get_action_from_type(transaction.transaction_type),
                                source=transaction.source,
                                reference_id=transaction.reference_id,
                                metadata=json.loads(transaction.metadata) if transaction.metadata else {},
                                description=transaction.description
                            )
                            
                            # Execute transaction
                            success, error = await self._execute_transaction(coin_transaction)
                            
                            if success:
                                transaction.status = TransactionStatus.COMPLETED
                            else:
                                transaction.status = TransactionStatus.FAILED
                                transaction.error_message = error
                            
                            transaction.updated_at = datetime.utcnow()
                            
                        except Exception as e:
                            logger.error(f"Error processing transaction {transaction.transaction_id}: {e}")
                            transaction.status = TransactionStatus.FAILED
                            transaction.error_message = str(e)
                            transaction.updated_at = datetime.utcnow()
                    
                    await session.commit()
                    
                    if pending_transactions:
                        logger.info(f"Processed {len(pending_transactions)} pending transactions")
                
            except Exception as e:
                logger.error(f"Error processing transaction queue: {e}", exc_info=True)
            
            # Run every 5 minutes
            await asyncio.sleep(300)
    
    def _get_action_from_type(self, transaction_type: TransactionType) -> CoinAction:
        """Convert transaction type to coin action."""
        type_map = {
            TransactionType.CREDIT: CoinAction.EARN,
            TransactionType.DEBIT: CoinAction.SPEND,
            TransactionType.TRANSFER: CoinAction.TRANSFER,
            TransactionType.WITHDRAWAL: CoinAction.WITHDRAW,
            TransactionType.DEPOSIT: CoinAction.DEPOSIT,
            TransactionType.REFUND: CoinAction.REFUND,
            TransactionType.BURN: CoinAction.BURN,
            TransactionType.MINT: CoinAction.MINT,
        }
        return type_map.get(transaction_type, CoinAction.EARN)
    
    async def _cleanup_old_transactions(self):
        """Clean up old transaction records."""
        while True:
            try:
                async with AsyncSessionLocal() as session:
                    # Archive transactions older than 90 days
                    archive_cutoff = datetime.utcnow() - timedelta(days=90)
                    
                    # This would move old transactions to an archive table
                    # For now, just log the count
                    result = await session.execute(
                        select(func.count(Transaction.id))
                        .where(Transaction.created_at < archive_cutoff)
                        .where(Transaction.status.in_([TransactionStatus.COMPLETED, TransactionStatus.FAILED]))
                    )
                    
                    old_count = result.scalar() or 0
                    
                    if old_count > 0:
                        logger.info(f"{old_count} transactions eligible for archiving")
                    
                    # Delete failed transactions older than 30 days
                    delete_cutoff = datetime.utcnow() - timedelta(days=30)
                    
                    result = await session.execute(
                        select(Transaction)
                        .where(Transaction.created_at < delete_cutoff)
                        .where(Transaction.status == TransactionStatus.FAILED)
                    )
                    
                    failed_transactions = result.scalars().all()
                    
                    for transaction in failed_transactions:
                        await session.delete(transaction)
                    
                    deleted_count = len(failed_transactions)
                    await session.commit()
                    
                    if deleted_count > 0:
                        logger.info(f"Deleted {deleted_count} old failed transactions")
                
            except Exception as e:
                logger.error(f"Error cleaning up old transactions: {e}")
            
            # Run daily
            await asyncio.sleep(86400)
    
    async def _update_daily_limits(self):
        """Update daily reward limits based on system metrics."""
        while True:
            try:
                # Adjust limits based on system load, coin supply, etc.
                # This is a placeholder for more sophisticated logic
                
                # Example: Increase limits during low-activity periods
                current_hour = datetime.utcnow().hour
                if 2 <= current_hour <= 6:  # 2 AM to 6 AM UTC
                    for key in self.daily_limits:
                        self.daily_limits[key] *= Decimal('1.5')
                        logger.debug(f"Increased {key} limit to {self.daily_limits[key]}")
                
                # Reset user transaction counts at UTC midnight
                current_time = datetime.utcnow()
                if current_time.hour == 0 and current_time.minute < 5:
                    self.user_transaction_count.clear()
                    self.user_transaction_reset.clear()
                    logger.info("Reset daily transaction counts")
                
            except Exception as e:
                logger.error(f"Error updating daily limits: {e}")
            
            # Run hourly
            await asyncio.sleep(3600)
    
    # Public API methods
    
    async def get_wallet_balance(self, user_id: int) -> Dict[str, Any]:
        """Get wallet balance and statistics."""
        try:
            async with AsyncSessionLocal() as session:
                # Get wallet
                result = await session.execute(
                    select(Wallet)
                    .where(Wallet.user_id == user_id)
                    .where(Wallet.wallet_type == WalletType.PRIMARY)
                )
                wallet = result.scalar_one_or_none()
                
                if not wallet:
                    return {'error': 'Wallet not found'}
                
                # Get recent transactions
                transactions_result = await session.execute(
                    select(Transaction)
                    .where(
                        or_(
                            Transaction.from_user_id == user_id,
                            Transaction.to_user_id == user_id
                        )
                    )
                    .order_by(desc(Transaction.created_at))
                    .limit(10)
                )
                recent_transactions = transactions_result.scalars().all()
                
                # Calculate daily earned
                today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                
                daily_earned_result = await session.execute(
                    select(func.sum(Transaction.amount))
                    .where(Transaction.to_user_id == user_id)
                    .where(Transaction.status == TransactionStatus.COMPLETED)
                    .where(Transaction.created_at >= today_start)
                )
                daily_earned = daily_earned_result.scalar() or 0.0
                
                return {
                    'user_id': user_id,
                    'balance': float(wallet.balance),
                    'total_earned': float(wallet.total_earned),
                    'total_spent': float(wallet.total_spent),
                    'wallet_status': wallet.status.value,
                    'daily_earned': daily_earned,
                    'recent_transactions': [
                        {
                            'id': t.transaction_id,
                            'type': t.transaction_type.value,
                            'amount': t.amount,
                            'status': t.status.value,
                            'created_at': t.created_at.isoformat(),
                            'description': t.description
                        }
                        for t in recent_transactions
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error getting wallet balance: {e}")
            return {'error': str(e)}
    
    async def get_transaction_history(
        self,
        user_id: int,
        page: int = 1,
        per_page: int = 20,
        transaction_type: Optional[TransactionType] = None
    ) -> Dict[str, Any]:
        """Get transaction history for a user."""
        try:
            async with AsyncSessionLocal() as session:
                # Build query
                query = select(Transaction).where(
                    or_(
                        Transaction.from_user_id == user_id,
                        Transaction.to_user_id == user_id
                    )
                )
                
                if transaction_type:
                    query = query.where(Transaction.transaction_type == transaction_type)
                
                # Get total count
                count_result = await session.execute(
                    select(func.count(Transaction.id)).select_from(query.subquery())
                )
                total_count = count_result.scalar() or 0
                
                # Get paginated results
                offset = (page - 1) * per_page
                query = query.order_by(desc(Transaction.created_at)).offset(offset).limit(per_page)
                
                result = await session.execute(query)
                transactions = result.scalars().all()
                
                return {
                    'transactions': [
                        {
                            'id': t.transaction_id,
                            'type': t.transaction_type.value,
                            'amount': t.amount,
                            'status': t.status.value,
                            'source': t.source.value,
                            'from_user_id': t.from_user_id,
                            'to_user_id': t.to_user_id,
                            'description': t.description,
                            'created_at': t.created_at.isoformat(),
                            'metadata': json.loads(t.metadata) if t.metadata else {}
                        }
                        for t in transactions
                    ],
                    'pagination': {
                        'page': page,
                        'per_page': per_page,
                        'total_count': total_count,
                        'total_pages': (total_count + per_page - 1) // per_page
                    }
                }
                
        except Exception as e:
            logger.error(f"Error getting transaction history: {e}")
            return {'error': str(e)}
    
    async def get_reward_summary(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """Get reward summary for a user."""
        try:
            async with AsyncSessionLocal() as session:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                # Get rewards by type
                result = await session.execute(
                    select(Reward.reward_type, func.sum(Reward.amount), func.count(Reward.id))
                    .where(Reward.user_id == user_id)
                    .where(Reward.status == RewardStatus.COMPLETED)
                    .where(Reward.created_at >= cutoff_date)
                    .group_by(Reward.reward_type)
                    .order_by(func.sum(Reward.amount).desc())
                )
                
                rewards_by_type = result.all()
                
                # Get total rewards
                total_result = await session.execute(
                    select(func.sum(Reward.amount))
                    .where(Reward.user_id == user_id)
                    .where(Reward.status == RewardStatus.COMPLETED)
                    .where(Reward.created_at >= cutoff_date)
                )
                total_rewards = total_result.scalar() or 0.0
                
                # Get streak info
                streak_result = await session.execute(
                    select(func.count(distinct(func.date(Reward.created_at))))
                    .where(Reward.user_id == user_id)
                    .where(Reward.status == RewardStatus.COMPLETED)
                    .where(Reward.created_at >= cutoff_date)
                    .where(Reward.reward_type == RewardType.DAILY_LOGIN)
                )
                login_streak = streak_result.scalar() or 0
                
                return {
                    'user_id': user_id,
                    'period_days': days,
                    'total_rewards': total_rewards,
                    'login_streak': login_streak,
                    'rewards_by_type': [
                        {
                            'type': reward_type.value,
                            'total_amount': float(total_amount),
                            'count': count
                        }
                        for reward_type, total_amount, count in rewards_by_type
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error getting reward summary: {e}")
            return {'error': str(e)}


# Factory function
async def create_wallet_subscriber(event_bus: EventBus) -> WalletSubscriber:
    """Create and initialize a wallet subscriber."""
    subscriber = WalletSubscriber(event_bus)
    await subscriber.initialize()
    return subscriber