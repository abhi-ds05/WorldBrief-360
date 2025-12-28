"""
Reward model for tracking user rewards, bonuses, and incentives.
This includes earned rewards, pending rewards, reward multipliers, and reward history.
"""

from sqlalchemy import Column, Integer, String, DateTime, Numeric, Enum, ForeignKey, Boolean, Text, JSON, Index,Tuple
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship, validates
from datetime import datetime
import enum
import json
import uuid
from decimal import Decimal
from typing import Any, Dict, Optional

from app.db.base import Base
from app.core.exceptions import ValidationError


class RewardType(str, enum.Enum):
    """
    Types of rewards that users can earn.
    """
    # Incident-related rewards
    INCIDENT_REPORT = "incident_report"                     # Reporting a new incident
    INCIDENT_VERIFICATION = "incident_verification"         # Verifying an incident
    INCIDENT_VERIFICATION_BONUS = "incident_verification_bonus" # Bonus for incident being verified
    INCIDENT_RESOLUTION = "incident_resolution"             # Incident resolved
    INCIDENT_ESCALATION = "incident_escalation"             # Incident escalated to authorities
    
    # Content creation rewards
    COMMENT_CREATION = "comment_creation"                   # Creating a comment
    COMMENT_REPLY = "comment_reply"                         # Replying to a comment
    COMMENT_MENTION = "comment_mention"                     # Being mentioned in a comment
    POST_CREATION = "post_creation"                         # Creating a post
    ARTICLE_CREATION = "article_creation"                   # Writing an article
    ANALYSIS_CREATION = "analysis_creation"                 # Creating analysis
    REPORT_CREATION = "report_creation"                     # Creating a report
    
    # Briefing rewards
    BRIEFING_GENERATION = "briefing_generation"             # Generating a briefing
    BRIEFING_SHARE = "briefing_share"                       # Sharing a briefing
    BRIEFING_FEEDBACK = "briefing_feedback"                 # Providing feedback on briefing
    
    # Chat rewards
    CHAT_MESSAGE = "chat_message"                           # Sending chat message
    CHAT_MENTION = "chat_mention"                           # Being mentioned in chat
    CHAT_HELPFUL = "chat_helpful"                           # Helpful chat response
    CHAT_ANSWER = "chat_answer"                             # Answering a question in chat
    
    # Moderation rewards
    MODERATION_ACTION = "moderation_action"                 # Taking moderation action
    CONTENT_REVIEW = "content_review"                       # Reviewing content
    APPEAL_REVIEW = "appeal_review"                         # Reviewing an appeal
    VALID_REPORT = "valid_report"                           # Submitting valid report
    COMMUNITY_GUIDELINES = "community_guidelines"           # Following community guidelines
    
    # Engagement rewards
    DAILY_LOGIN = "daily_login"                             # Daily login streak
    STREAK_BONUS = "streak_bonus"                           # Streak milestone bonus
    WEEKLY_ACTIVITY = "weekly_activity"                     # Weekly activity completion
    MONTHLY_ACTIVITY = "monthly_activity"                   # Monthly activity completion
    
    # Achievement rewards
    ACHIEVEMENT_BRONZE = "achievement_bronze"               # Bronze achievement
    ACHIEVEMENT_SILVER = "achievement_silver"               # Silver achievement
    ACHIEVEMENT_GOLD = "achievement_gold"                   # Gold achievement
    ACHIEVEMENT_PLATINUM = "achievement_platinum"           # Platinum achievement
    ACHIEVEMENT_DIAMOND = "achievement_diamond"             # Diamond achievement
    
    # Referral rewards
    REFERRAL_SIGNUP = "referral_signup"                     # Referral signup completed
    REFERRAL_ACTIVITY = "referral_activity"                 # Referral user activity
    REFERRAL_PURCHASE = "referral_purchase"                 # Referral made purchase
    
    # Quality rewards
    QUALITY_CONTENT = "quality_content"                     # High-quality content
    FACT_CHECKED = "fact_checked"                           # Content fact-checked
    SOURCE_CITED = "source_cited"                           # Proper source citation
    DETAILED_ANALYSIS = "detailed_analysis"                 # Detailed analysis
    
    # Special rewards
    WELCOME_BONUS = "welcome_bonus"                         # New user welcome bonus
    BIRTHDAY_BONUS = "birthday_bonus"                       # Birthday bonus
    ANNIVERSARY_BONUS = "anniversary_bonus"                 # Account anniversary
    HOLIDAY_BONUS = "holiday_bonus"                         # Holiday special bonus
    EVENT_BONUS = "event_bonus"                             # Special event bonus
    
    # System rewards
    BUG_REPORT = "bug_report"                               # Reporting a bug
    FEATURE_REQUEST = "feature_request"                     # Requesting a feature
    FEEDBACK_SUBMISSION = "feedback_submission"             # Submitting feedback
    SURVEY_COMPLETION = "survey_completion"                 # Completing survey
    
    # Manual rewards
    MANUAL_ADJUSTMENT = "manual_adjustment"                 # Manual adjustment by admin
    COMPENSATION = "compensation"                           # Compensation for issue
    GOODWILL = "goodwill"                                   # Goodwill gesture
    
    # Tier rewards
    TIER_UPGRADE = "tier_upgrade"                           # Upgrading subscription tier
    LOYALTY_BONUS = "loyalty_bonus"                         # Loyalty program bonus
    
    # Testing
    TEST_REWARD = "test_reward"                             # Test reward (development only)


class RewardStatus(str, enum.Enum):
    """
    Status of a reward throughout its lifecycle.
    """
    DRAFT = "draft"                    # Draft reward (not yet processed)
    PENDING = "pending"                # Reward pending processing
    PROCESSING = "processing"          # Reward being processed
    APPROVED = "approved"              # Reward approved for distribution
    DISTRIBUTED = "distributed"        # Reward distributed to user
    COMPLETED = "completed"            # Reward completed (coins received)
    FAILED = "failed"                  # Reward failed
    CANCELLED = "cancelled"            # Reward cancelled
    REVERSED = "reversed"              # Reward reversed
    EXPIRED = "expired"                # Reward expired before claim
    ON_HOLD = "on_hold"                # Reward on hold (requires review)


class RewardTier(str, enum.Enum):
    """
    Tiers for rewards (affects multiplier).
    """
    BASIC = "basic"                    # Basic tier (1x multiplier)
    SILVER = "silver"                  # Silver tier (1.2x multiplier)
    GOLD = "gold"                      # Gold tier (1.5x multiplier)
    PLATINUM = "platinum"              # Platinum tier (2x multiplier)
    DIAMOND = "diamond"                # Diamond tier (3x multiplier)


class RewardSource(str, enum.Enum):
    """
    Source of the reward (who/what granted it).
    """
    SYSTEM = "system"                  # Automated system reward
    MANUAL = "manual"                  # Manual reward by admin
    COMMUNITY = "community"            # Community-granted reward
    ALGORITHMIC = "algorithmic"        # Algorithm-generated reward
    USER = "user"                      # User-to-user reward
    PROMOTIONAL = "promotional"        # Promotional reward
    EVENT = "event"                    # Event-based reward
    PARTNER = "partner"                # Partner-sponsored reward


class Reward(Base):
    """
    Main reward model for tracking all user rewards.
    
    Each reward represents an amount of coins earned by a user for a specific action.
    Rewards can have multipliers, expiration dates, and various statuses.
    """
    
    __tablename__ = "rewards"
    
    # Primary key and identification
    id = Column(Integer, primary_key=True, index=True)
    reward_id = Column(String(100), unique=True, index=True, nullable=False, 
                      default=lambda: str(uuid.uuid4()), comment="Public UUID for external reference")
    
    # User information
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Reward details
    reward_type = Column(Enum(RewardType), nullable=False, index=True)
    reward_tier = Column(Enum(RewardTier), default=RewardTier.BASIC, nullable=False, index=True)
    source = Column(Enum(RewardSource), default=RewardSource.SYSTEM, nullable=False, index=True)
    
    # Amount and multipliers
    base_amount = Column(Numeric(20, 8), nullable=False, comment="Base reward amount before multipliers")
    multiplier = Column(Numeric(5, 2), default=1.00, nullable=False, comment="Multiplier applied to base amount")
    bonus_amount = Column(Numeric(20, 8), default=0, nullable=False, comment="Additional bonus amount")
    total_amount = Column(Numeric(20, 8), nullable=False, comment="Total amount after all calculations")
    
    # Status and lifecycle
    status = Column(Enum(RewardStatus), default=RewardStatus.PENDING, nullable=False, index=True)
    priority = Column(Integer, default=1, nullable=False, index=True, comment="Processing priority (1-10)")
    
    # Source tracking
    source_id = Column(String(100), nullable=True, index=True, comment="ID of the source entity")
    source_type = Column(String(50), nullable=True, index=True, comment="Type of source entity")
    source_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, 
                           comment="User who triggered/approved the reward")
    
    # Context and metadata
    title = Column(String(255), nullable=True, comment="Reward title")
    description = Column(Text, nullable=True, comment="Reward description")
    metadata = Column(JSONB, nullable=True, comment="Additional metadata about the reward")
    tags = Column(JSONB, nullable=True, comment="Tags for categorization")
    
    # Eligibility and constraints
    eligibility_criteria = Column(JSONB, nullable=True, comment="JSON criteria for eligibility")
    constraints = Column(JSONB, nullable=True, comment="Constraints (max per day, etc.)")
    requires_approval = Column(Boolean, default=False, comment="Whether reward requires approval")
    approved_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    approved_at = Column(DateTime, nullable=True, index=True)
    distributed_at = Column(DateTime, nullable=True, index=True, comment="When reward was distributed")
    completed_at = Column(DateTime, nullable=True, index=True)
    expires_at = Column(DateTime, nullable=True, index=True, comment="When reward expires")
    
    # Distribution details
    distribution_method = Column(String(50), nullable=True, comment="Method of distribution")
    transaction_id = Column(String(100), nullable=True, unique=True, index=True, 
                           comment="ID of related transaction")
    wallet_address = Column(String(255), nullable=True, comment="Wallet address for crypto rewards")
    
    # Audit and verification
    verification_hash = Column(String(255), nullable=True, unique=True, index=True, 
                              comment="Hash for verification")
    audit_trail = Column(JSONB, nullable=True, comment="Audit trail of status changes")
    notes = Column(Text, nullable=True, comment="Internal notes")
    
    # Error handling
    error_message = Column(Text, nullable=True, comment="Error message if reward failed")
    retry_count = Column(Integer, default=0, comment="Number of retry attempts")
    max_retries = Column(Integer, default=3, comment="Maximum retry attempts")
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id], backref="rewards")
    source_user = relationship("User", foreign_keys=[source_user_id], backref="given_rewards")
    approver = relationship("User", foreign_keys=[approved_by], backref="approved_rewards")
    
    # Indexes
    __table_args__ = (
        Index('ix_rewards_user_status', 'user_id', 'status'),
        Index('ix_rewards_type_status', 'reward_type', 'status'),
        Index('ix_rewards_created_status', 'created_at', 'status'),
        Index('ix_rewards_expires_status', 'expires_at', 'status'),
        Index('ix_rewards_source_details', 'source_id', 'source_type'),
        Index('ix_rewards_amount_tier', 'total_amount', 'reward_tier'),
        Index('ix_rewards_tags', 'tags', postgresql_using='gin'),
    )
    
    @validates('reward_id')
    def validate_reward_id(self, key, reward_id):
        """Validate reward ID is not empty."""
        if not reward_id or len(reward_id.strip()) == 0:
            raise ValidationError("Reward ID cannot be empty")
        return reward_id.strip()
    
    @validates('base_amount', 'bonus_amount', 'total_amount')
    def validate_amounts(self, key, amount):
        """Validate reward amounts are non-negative."""
        if amount is not None and amount < 0:
            raise ValidationError(f"{key} cannot be negative")
        return amount
    
    @validates('multiplier')
    def validate_multiplier(self, key, multiplier):
        """Validate multiplier is reasonable."""
        if multiplier is not None:
            if multiplier < 0.1:
                raise ValidationError("Multiplier cannot be less than 0.1")
            if multiplier > 100:
                raise ValidationError("Multiplier cannot exceed 100")
        return multiplier
    
    @validates('priority')
    def validate_priority(self, key, priority):
        """Validate priority is between 1 and 10."""
        if priority < 1 or priority > 10:
            raise ValidationError("Priority must be between 1 and 10")
        return priority
    
    @validates('retry_count', 'max_retries')
    def validate_retries(self, key, retries):
        """Validate retry counts are non-negative."""
        if retries < 0:
            raise ValidationError(f"{key} cannot be negative")
        return retries
    
    def __repr__(self):
        return f"<Reward {self.reward_id} ({self.reward_type} - {self.status})>"
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        Convert reward to dictionary representation.
        
        Args:
            include_sensitive: Whether to include sensitive information
            
        Returns:
            Dictionary representation
        """
        result = {
            "id": self.id,
            "reward_id": self.reward_id,
            "user_id": self.user_id,
            "reward_type": self.reward_type.value,
            "reward_tier": self.reward_tier.value,
            "source": self.source.value,
            "base_amount": float(self.base_amount) if self.base_amount else 0.0,
            "multiplier": float(self.multiplier) if self.multiplier else 1.0,
            "bonus_amount": float(self.bonus_amount) if self.bonus_amount else 0.0,
            "total_amount": float(self.total_amount) if self.total_amount else 0.0,
            "status": self.status.value,
            "priority": self.priority,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "source_user_id": self.source_user_id,
            "title": self.title,
            "description": self.description,
            "requires_approval": self.requires_approval,
            "approved_by": self.approved_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "distributed_at": self.distributed_at.isoformat() if self.distributed_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "distribution_method": self.distribution_method,
            "transaction_id": self.transaction_id,
            "error_message": self.error_message if include_sensitive else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "verification_hash": self.verification_hash if include_sensitive else None,
            "notes": self.notes if include_sensitive else None,
        }
        
        if include_sensitive:
            result.update({
                "metadata": self.metadata,
                "tags": self.tags,
                "eligibility_criteria": self.eligibility_criteria,
                "constraints": self.constraints,
                "audit_trail": self.audit_trail,
                "wallet_address": self.wallet_address,
            })
        
        return result
    
    def to_public_dict(self) -> Dict[str, Any]:
        """
        Convert to public dictionary (safe for user viewing).
        
        Returns:
            Public-safe dictionary representation
        """
        result = self.to_dict(include_sensitive=False)
        
        # Remove internal fields
        internal_fields = [
            'priority', 'requires_approval', 'approved_by', 'source_user_id',
            'error_message', 'retry_count', 'max_retries', 'verification_hash',
            'distribution_method', 'transaction_id'
        ]
        
        for field in internal_fields:
            if field in result:
                del result[field]
        
        return result
    
    @property
    def is_pending(self) -> bool:
        """
        Check if reward is pending.
        
        Returns:
            True if reward is pending
        """
        return self.status == RewardStatus.PENDING
    
    @property
    def is_completed(self) -> bool:
        """
        Check if reward is completed.
        
        Returns:
            True if reward is completed
        """
        return self.status == RewardStatus.COMPLETED
    
    @property
    def is_failed(self) -> bool:
        """
        Check if reward failed.
        
        Returns:
            True if reward failed
        """
        return self.status == RewardStatus.FAILED
    
    @property
    def is_expired(self) -> bool:
        """
        Check if reward has expired.
        
        Returns:
            True if reward has expired
        """
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False
    
    @property
    def can_retry(self) -> bool:
        """
        Check if reward can be retried.
        
        Returns:
            True if reward can be retried
        """
        return (
            self.status == RewardStatus.FAILED and 
            self.retry_count < self.max_retries
        )
    
    @property
    def requires_manual_approval(self) -> bool:
        """
        Check if reward requires manual approval.
        
        Returns:
            True if reward requires approval
        """
        return self.requires_approval and self.status == RewardStatus.PENDING
    
    @property
    def days_until_expiry(self) -> Optional[int]:
        """
        Get days until reward expires.
        
        Returns:
            Days until expiry or None
        """
        if not self.expires_at:
            return None
        
        delta = self.expires_at - datetime.utcnow()
        return max(delta.days, 0)
    
    def calculate_total(self):
        """Calculate total amount based on base amount, multiplier, and bonus."""
        if self.base_amount is None:
            raise ValidationError("Base amount is required to calculate total")
        
        base = Decimal(str(self.base_amount))
        multiplier = Decimal(str(self.multiplier))
        bonus = Decimal(str(self.bonus_amount)) if self.bonus_amount else Decimal('0')
        
        # Calculate: (base * multiplier) + bonus
        self.total_amount = (base * multiplier) + bonus
    
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
    
    def approve(self, approved_by: int, notes: Optional[str] = None):
        """
        Approve the reward.
        
        Args:
            approved_by: ID of user approving
            notes: Optional approval notes
        """
        if not self.requires_approval:
            raise ValidationError("Reward does not require approval")
        
        if self.status != RewardStatus.PENDING:
            raise ValidationError("Only pending rewards can be approved")
        
        self.status = RewardStatus.APPROVED
        self.approved_by = approved_by
        self.approved_at = datetime.utcnow()
        
        if notes:
            self.notes = notes
        
        # Add audit entry
        self.add_audit_entry(
            action="approved",
            details={"approved_by": approved_by, "notes": notes},
            user_id=approved_by
        )
    
    def distribute(self, transaction_id: str, method: str = "wallet"):
        """
        Mark reward as distributed.
        
        Args:
            transaction_id: ID of distribution transaction
            method: Distribution method
        """
        if self.status not in [RewardStatus.APPROVED, RewardStatus.PENDING]:
            raise ValidationError("Only approved or pending rewards can be distributed")
        
        self.status = RewardStatus.DISTRIBUTED
        self.distributed_at = datetime.utcnow()
        self.transaction_id = transaction_id
        self.distribution_method = method
        
        # Add audit entry
        self.add_audit_entry(
            action="distributed",
            details={"transaction_id": transaction_id, "method": method}
        )
    
    def complete(self):
        """Mark reward as completed."""
        if self.status != RewardStatus.DISTRIBUTED:
            raise ValidationError("Only distributed rewards can be completed")
        
        self.status = RewardStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        
        # Add audit entry
        self.add_audit_entry(action="completed", details={})
    
    def fail(self, error_message: str):
        """
        Mark reward as failed.
        
        Args:
            error_message: Error message
        """
        self.status = RewardStatus.FAILED
        self.error_message = error_message
        self.retry_count += 1
        
        # Add audit entry
        self.add_audit_entry(
            action="failed",
            details={"error_message": error_message, "retry_count": self.retry_count}
        )
    
    def retry(self):
        """Retry a failed reward."""
        if not self.can_retry:
            raise ValidationError("Reward cannot be retried")
        
        self.status = RewardStatus.PENDING
        self.error_message = None
        
        # Add audit entry
        self.add_audit_entry(action="retry", details={"attempt": self.retry_count})
    
    def cancel(self, reason: str, cancelled_by: Optional[int] = None):
        """
        Cancel the reward.
        
        Args:
            reason: Cancellation reason
            cancelled_by: ID of user cancelling
        """
        if self.status in [RewardStatus.COMPLETED, RewardStatus.CANCELLED, RewardStatus.EXPIRED]:
            raise ValidationError("Cannot cancel a completed, cancelled, or expired reward")
        
        self.status = RewardStatus.CANCELLED
        
        if not self.notes:
            self.notes = f"Cancelled: {reason}"
        else:
            self.notes += f"\nCancelled: {reason}"
        
        # Add audit entry
        self.add_audit_entry(
            action="cancelled",
            details={"reason": reason, "cancelled_by": cancelled_by},
            user_id=cancelled_by
        )
    
    def reverse(self, reason: str, reversed_by: int):
        """
        Reverse a completed reward.
        
        Args:
            reason: Reversal reason
            reversed_by: ID of user reversing
        """
        if self.status != RewardStatus.COMPLETED:
            raise ValidationError("Only completed rewards can be reversed")
        
        self.status = RewardStatus.REVERSED
        
        if not self.notes:
            self.notes = f"Reversed: {reason}"
        else:
            self.notes += f"\nReversed: {reason}"
        
        # Add audit entry
        self.add_audit_entry(
            action="reversed",
            details={"reason": reason},
            user_id=reversed_by
        )
    
    def add_bonus(self, bonus_amount: Decimal, reason: str):
        """
        Add bonus amount to reward.
        
        Args:
            bonus_amount: Bonus amount to add
            reason: Reason for bonus
        """
        if bonus_amount <= 0:
            raise ValidationError("Bonus amount must be positive")
        
        current_bonus = Decimal(str(self.bonus_amount)) if self.bonus_amount else Decimal('0')
        self.bonus_amount = current_bonus + bonus_amount
        
        # Recalculate total
        self.calculate_total()
        
        # Add note
        if not self.notes:
            self.notes = f"Bonus added: {reason} (+{bonus_amount})"
        else:
            self.notes += f"\nBonus added: {reason} (+{bonus_amount})"
        
        # Add audit entry
        self.add_audit_entry(
            action="bonus_added",
            details={"amount": float(bonus_amount), "reason": reason}
        )
    
    def apply_multiplier(self, multiplier: Decimal, reason: str):
        """
        Apply a multiplier to the reward.
        
        Args:
            multiplier: Multiplier to apply
            reason: Reason for multiplier
        """
        if multiplier <= 0:
            raise ValidationError("Multiplier must be positive")
        
        current_multiplier = Decimal(str(self.multiplier)) if self.multiplier else Decimal('1')
        self.multiplier = current_multiplier * multiplier
        
        # Recalculate total
        self.calculate_total()
        
        # Add note
        if not self.notes:
            self.notes = f"Multiplier applied: {reason} (x{multiplier})"
        else:
            self.notes += f"\nMultiplier applied: {reason} (x{multiplier})"
        
        # Add audit entry
        self.add_audit_entry(
            action="multiplier_applied",
            details={"multiplier": float(multiplier), "reason": reason}
        )


class RewardCampaign(Base):
    """
    Model for managing reward campaigns.
    
    Campaigns define rules for rewarding users for specific actions
    during a certain time period.
    """
    
    __tablename__ = "reward_campaigns"
    
    id = Column(Integer, primary_key=True, index=True)
    campaign_id = Column(String(100), unique=True, index=True, nullable=False, 
                        default=lambda: str(uuid.uuid4()))
    
    # Campaign details
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    campaign_type = Column(String(50), nullable=False, index=True, comment="Type of campaign")
    
    # Reward configuration
    reward_type = Column(Enum(RewardType), nullable=False, index=True)
    base_amount = Column(Numeric(20, 8), nullable=False)
    multiplier = Column(Numeric(5, 2), default=1.00)
    bonus_amount = Column(Numeric(20, 8), default=0)
    
    # Campaign period
    starts_at = Column(DateTime, nullable=False, index=True)
    ends_at = Column(DateTime, nullable=False, index=True)
    
    # Eligibility rules
    eligibility_rules = Column(JSONB, nullable=True, comment="JSON rules for eligibility")
    user_segment = Column(JSONB, nullable=True, comment="User segment criteria")
    max_rewards_per_user = Column(Integer, nullable=True, comment="Max rewards per user")
    total_budget = Column(Numeric(20, 8), nullable=True, comment="Total campaign budget")
    
    # Status
    status = Column(String(50), default="draft", nullable=False, index=True, 
                   comment="draft, active, paused, completed, cancelled")
    is_public = Column(Boolean, default=False, index=True, comment="Whether campaign is public")
    
    # Tracking
    rewards_distributed = Column(Integer, default=0, comment="Number of rewards distributed")
    total_amount_distributed = Column(Numeric(20, 8), default=0, comment="Total amount distributed")
    remaining_budget = Column(Numeric(20, 8), nullable=True, comment="Remaining budget")
    
    # Metadata
    tags = Column(JSONB, nullable=True)
    metadata = Column(JSONB, nullable=True)
    
    # Audit
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    creator = relationship("User", foreign_keys=[created_by])
    
    @property
    def is_active(self) -> bool:
        """Check if campaign is currently active."""
        current_time = datetime.utcnow()
        return (
            self.status == "active" and
            self.starts_at <= current_time <= self.ends_at
        )
    
    @property
    def has_budget(self) -> bool:
        """Check if campaign has remaining budget."""
        if self.total_budget is None:
            return True
        if self.remaining_budget is None:
            return True
        return self.remaining_budget > 0
    
    def can_award_reward(self, user_id: int) -> Tuple[bool, str]:
        """
        Check if a reward can be awarded to a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Tuple of (can_award, reason)
        """
        if not self.is_active:
            return False, "Campaign is not active"
        
        if not self.has_budget:
            return False, "Campaign budget exhausted"
        
        # Check max rewards per user
        if self.max_rewards_per_user:
            # Would need to query existing rewards for this user in this campaign
            pass
        
        return True, ""


class RewardMultiplierRule(Base):
    """
    Model for defining reward multiplier rules.
    
    Rules determine when and how reward multipliers should be applied.
    """
    
    __tablename__ = "reward_multiplier_rules"
    
    id = Column(Integer, primary_key=True, index=True)
    rule_id = Column(String(100), unique=True, index=True, nullable=False)
    
    # Rule configuration
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # Conditions
    condition_type = Column(String(50), nullable=False, comment="Type of condition")
    condition_value = Column(String(255), nullable=False, comment="Value to compare against")
    condition_operator = Column(String(10), default="equals", comment="Comparison operator")
    
    # Multiplier
    multiplier = Column(Numeric(5, 2), nullable=False)
    applies_to = Column(JSONB, nullable=True, comment="JSON array of reward types this applies to")
    
    # Priority and ordering
    priority = Column(Integer, default=1, nullable=False, index=True)
    is_active = Column(Boolean, default=True, index=True)
    
    # Time constraints
    valid_from = Column(DateTime, nullable=True)
    valid_until = Column(DateTime, nullable=True)
    
    # Usage tracking
    usage_count = Column(Integer, default=0)
    last_used_at = Column(DateTime, nullable=True)
    
    # Metadata
    tags = Column(JSONB, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """
        Evaluate the rule against a context.
        
        Args:
            context: Context data to evaluate against
            
        Returns:
            True if rule applies
        """
        if not self.is_active:
            return False
        
        # Check time validity
        current_time = datetime.utcnow()
        if self.valid_from and current_time < self.valid_from:
            return False
        if self.valid_until and current_time > self.valid_until:
            return False
        
        # Check condition
        value = context.get(self.condition_type)
        if value is None:
            return False
        
        # Apply operator
        operators = {
            "equals": lambda a, b: a == b,
            "not_equals": lambda a, b: a != b,
            "greater_than": lambda a, b: a > b,
            "less_than": lambda a, b: a < b,
            "greater_than_equal": lambda a, b: a >= b,
            "less_than_equal": lambda a, b: a <= b,
            "contains": lambda a, b: b in a if isinstance(a, (list, str)) else False,
            "starts_with": lambda a, b: str(a).startswith(str(b)),
            "ends_with": lambda a, b: str(a).endswith(str(b)),
        }
        
        operator_func = operators.get(self.condition_operator)
        if not operator_func:
            return False
        
        return operator_func(value, self.condition_value)


class UserRewardStats(Base):
    """
    Model for tracking user reward statistics.
    """
    
    __tablename__ = "user_reward_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False, index=True)
    
    # Lifetime statistics
    total_rewards_received = Column(Integer, default=0)
    total_amount_received = Column(Numeric(20, 8), default=0)
    largest_reward = Column(Numeric(20, 8), default=0)
    average_reward = Column(Numeric(10, 2), default=0)
    
    # By reward type
    rewards_by_type = Column(JSONB, nullable=True, comment="JSON map of reward_type -> count")
    amount_by_type = Column(JSONB, nullable=True, comment="JSON map of reward_type -> amount")
    
    # Streaks and consistency
    current_streak = Column(Integer, default=0, comment="Current daily streak")
    longest_streak = Column(Integer, default=0, comment="Longest daily streak")
    last_reward_date = Column(DateTime, nullable=True)
    
    # Performance metrics
    reward_frequency = Column(Numeric(5, 2), default=0, comment="Rewards per day")
    reward_consistency = Column(Numeric(5, 2), default=0, comment="Consistency score 0-100")
    
    # Tier information
    current_tier = Column(Enum(RewardTier), default=RewardTier.BASIC, nullable=False, index=True)
    tier_progress = Column(Numeric(5, 2), default=0, comment="Progress to next tier (0-100)")
    
    # Special achievements
    achievements_unlocked = Column(JSONB, nullable=True, comment="JSON array of achievement IDs")
    badges_earned = Column(JSONB, nullable=True, comment="JSON array of badge IDs")
    
    # Leaderboard
    global_rank = Column(Integer, nullable=True, comment="Global rank by rewards")
    percentile = Column(Numeric(5, 2), nullable=True, comment="Percentile (0-100)")
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", backref="reward_stats")
    
    def update_stats(self, reward: Reward):
        """Update statistics with a new reward."""
        self.total_rewards_received += 1
        self.total_amount_received += Decimal(str(reward.total_amount))
        
        # Update largest reward
        if Decimal(str(reward.total_amount)) > Decimal(str(self.largest_reward)):
            self.largest_reward = reward.total_amount
        
        # Update by type
        if self.rewards_by_type is None:
            self.rewards_by_type = {}
        if self.amount_by_type is None:
            self.amount_by_type = {}
        
        reward_type = reward.reward_type.value
        self.rewards_by_type[reward_type] = self.rewards_by_type.get(reward_type, 0) + 1
        self.amount_by_type[reward_type] = self.amount_by_type.get(reward_type, 0) + float(reward.total_amount)
        
        # Update streak
        current_date = datetime.utcnow().date()
        last_date = self.last_reward_date.date() if self.last_reward_date else None
        
        if last_date:
            days_diff = (current_date - last_date).days
            if days_diff == 1:
                self.current_streak += 1
            elif days_diff > 1:
                self.current_streak = 1
        else:
            self.current_streak = 1
        
        if self.current_streak > self.longest_streak:
            self.longest_streak = self.current_streak
        
        self.last_reward_date = datetime.utcnow()
        
        # Recalculate averages
        if self.total_rewards_received > 0:
            self.average_reward = self.total_amount_received / self.total_rewards_received
        
        # Update frequency
        account_age_days = max((datetime.utcnow() - self.created_at).days, 1)
        self.reward_frequency = self.total_rewards_received / account_age_days
        
        # Update consistency (percentage of days with at least one reward)
        if account_age_days > 0:
            # This would need more sophisticated tracking
            pass


class RewardBlacklist(Base):
    """
    Model for blacklisting users from specific rewards.
    """
    
    __tablename__ = "reward_blacklist"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Blacklist details
    blacklist_type = Column(String(50), nullable=False, index=True, comment="Type of blacklist")
    reward_types = Column(JSONB, nullable=True, comment="JSON array of reward types affected")
    
    # Reason and details
    reason = Column(Text, nullable=False)
    blacklisted_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Duration
    starts_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ends_at = Column(DateTime, nullable=True, comment="Null means permanent")
    
    # Status
    is_active = Column(Boolean, default=True, index=True)
    
    # Metadata
    notes = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id], backref="reward_blacklists")
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