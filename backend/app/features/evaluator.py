"""
Feature flag evaluation engine.
Handles complex evaluation logic, segmentation, and variant assignment.
"""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Callable, Set, Tuple
from enum import Enum
import re
import logging

from attr import dataclass

from .flags import FeatureFlag, FlagType, VariantType
from .context import FeatureContext, UserContext, SystemContext

logger = logging.getLogger(__name__)


class EvaluationStrategy(str, Enum):
    """Strategies for feature flag evaluation."""
    CONSISTENT_HASH = "consistent_hash"      # Stable assignments
    RANDOM = "random"                        # Random assignment
    STICKY_BUCKET = "sticky_bucket"          # Sticky assignments
    LOAD_BALANCED = "load_balanced"          # Load-based assignment


class SegmentRule(str, Enum):
    """Types of segmentation rules."""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    IN = "in"
    NOT_IN = "not_in"
    MATCHES_REGEX = "matches_regex"


@dataclass
class SegmentationRule:
    """Rule for user segmentation."""
    
    field: str                    # Field to evaluate (e.g., "country", "plan")
    rule_type: SegmentRule        # Rule type
    value: Any                    # Comparison value
    negate: bool = False          # Whether to negate the rule
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate rule against context."""
        if self.field not in context:
            return False
        
        field_value = context[self.field]
        
        try:
            result = self._evaluate_rule(field_value)
            return not result if self.negate else result
        except (TypeError, ValueError):
            return False
    
    def _evaluate_rule(self, field_value: Any) -> bool:
        """Internal rule evaluation."""
        if self.rule_type == SegmentRule.EQUALS:
            return field_value == self.value
        elif self.rule_type == SegmentRule.NOT_EQUALS:
            return field_value != self.value
        elif self.rule_type == SegmentRule.CONTAINS:
            if isinstance(field_value, str) and isinstance(self.value, str):
                return self.value in field_value
            elif isinstance(field_value, list):
                return self.value in field_value
            return False
        elif self.rule_type == SegmentRule.NOT_CONTAINS:
            if isinstance(field_value, str) and isinstance(self.value, str):
                return self.value not in field_value
            elif isinstance(field_value, list):
                return self.value not in field_value
            return True
        elif self.rule_type == SegmentRule.STARTS_WITH:
            return isinstance(field_value, str) and field_value.startswith(self.value)
        elif self.rule_type == SegmentRule.ENDS_WITH:
            return isinstance(field_value, str) and field_value.endswith(self.value)
        elif self.rule_type == SegmentRule.GREATER_THAN:
            return field_value > self.value
        elif self.rule_type == SegmentRule.LESS_THAN:
            return field_value < self.value
        elif self.rule_type == SegmentRule.IN:
            return field_value in self.value if isinstance(self.value, (list, set, tuple)) else False
        elif self.rule_type == SegmentRule.NOT_IN:
            return field_value not in self.value if isinstance(self.value, (list, set, tuple)) else True
        elif self.rule_type == SegmentRule.MATCHES_REGEX:
            if isinstance(field_value, str):
                return bool(re.match(self.value, field_value))
            return False
        else:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary."""
        return {
            "field": self.field,
            "rule_type": self.rule_type.value,
            "value": self.value,
            "negate": self.negate,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SegmentationRule':
        """Create rule from dictionary."""
        return cls(
            field=data["field"],
            rule_type=SegmentRule(data["rule_type"]),
            value=data["value"],
            negate=data.get("negate", False),
        )


@dataclass
class Segment:
    """User segment definition."""
    
    name: str
    description: str
    rules: List[SegmentationRule]
    priority: int = 0
    
    def matches(self, context: Dict[str, Any]) -> bool:
        """Check if user matches all segment rules."""
        if not self.rules:
            return False
        
        for rule in self.rules:
            if not rule.evaluate(context):
                return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert segment to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "rules": [rule.to_dict() for rule in self.rules],
            "priority": self.priority,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Segment':
        """Create segment from dictionary."""
        rules = [SegmentationRule.from_dict(rule) for rule in data.get("rules", [])]
        return cls(
            name=data["name"],
            description=data["description"],
            rules=rules,
            priority=data.get("priority", 0),
        )


class FeatureEvaluator:
    """
    Evaluates feature flags against user context.
    Handles complex logic including segmentation, targeting, and variant assignment.
    """
    
    def __init__(
        self,
        strategy: EvaluationStrategy = EvaluationStrategy.CONSISTENT_HASH,
        segments: Optional[Dict[str, Segment]] = None,
        seed: str = "worldbrief360"
    ):
        """
        Initialize evaluator.
        
        Args:
            strategy: Evaluation strategy for variant assignment
            segments: Predefined user segments
            seed: Seed for consistent hashing
        """
        self.strategy = strategy
        self.segments = segments or self._get_default_segments()
        self.seed = seed
        self._segment_cache: Dict[str, Set[str]] = {}  # Cache segment memberships
    
    def evaluate(
        self,
        flag: FeatureFlag,
        user_id: Optional[str] = None,
        context: Optional[Union[Dict, FeatureContext]] = None,
        environment: str = "production"
    ) -> Dict[str, Any]:
        """
        Evaluate feature flag for a user.
        
        Args:
            flag: Feature flag to evaluate
            user_id: Optional user identifier
            context: Evaluation context
            environment: Current environment
        
        Returns:
            Dictionary with evaluation results
        """
        # Convert context to dictionary if needed
        if isinstance(context, FeatureContext):
            context_dict = context.to_dict()
        else:
            context_dict = context or {}
        
        # Add user_id to context if provided
        if user_id and "user_id" not in context_dict:
            context_dict["user_id"] = user_id
        
        # Get user segments
        user_segments = self._get_user_segments(context_dict)
        context_dict["segments"] = user_segments
        
        # Basic checks
        is_enabled = self._check_basic_conditions(flag, environment)
        if not is_enabled:
            return self._disabled_result(flag)
        
        # Time-based checks
        if not self._check_time_conditions(flag):
            return self._disabled_result(flag)
        
        # Targeted user check
        if self._is_targeted_user(flag, user_id):
            return self._enabled_result(flag, user_id, context_dict, environment)
        
        # Segment-based targeting
        if self._matches_segments(flag, user_segments):
            return self._enabled_result(flag, user_id, context_dict, environment)
        
        # Percentage-based rollout
        if self._check_percentage_rollout(flag, user_id, context_dict):
            return self._enabled_result(flag, user_id, context_dict, environment)
        
        # Fallback to disabled
        return self._disabled_result(flag)
    
    def get_variant(
        self,
        flag: FeatureFlag,
        user_id: Optional[str] = None,
        context: Optional[Union[Dict, FeatureContext]] = None,
        environment: str = "production"
    ) -> Any:
        """
        Get variant value for user.
        
        Args:
            flag: Feature flag to evaluate
            user_id: Optional user identifier
            context: Evaluation context
            environment: Current environment
        
        Returns:
            Variant value or default if disabled
        """
        result = self.evaluate(flag, user_id, context, environment)
        return result.get("variant", flag.default_variant)
    
    def is_enabled(
        self,
        flag: FeatureFlag,
        user_id: Optional[str] = None,
        context: Optional[Union[Dict, FeatureContext]] = None,
        environment: str = "production"
    ) -> bool:
        """
        Check if flag is enabled for user.
        
        Args:
            flag: Feature flag to evaluate
            user_id: Optional user identifier
            context: Evaluation context
            environment: Current environment
        
        Returns:
            True if enabled, False otherwise
        """
        result = self.evaluate(flag, user_id, context, environment)
        return result.get("enabled", False)
    
    def _check_basic_conditions(self, flag: FeatureFlag, environment: str) -> bool:
        """Check basic conditions (enabled, environment)."""
        if not flag.enabled:
            logger.debug(f"Flag {flag.name} is globally disabled")
            return False
        
        if environment not in flag.environments:
            logger.debug(f"Flag {flag.name} not available in environment {environment}")
            return False
        
        return True
    
    def _check_time_conditions(self, flag: FeatureFlag) -> bool:
        """Check time-based conditions."""
        now = datetime.utcnow()
        
        if flag.start_time and now < flag.start_time:
            logger.debug(f"Flag {flag.name} not active yet (starts at {flag.start_time})")
            return False
        
        if flag.end_time and now > flag.end_time:
            logger.debug(f"Flag {flag.name} expired (ended at {flag.end_time})")
            return False
        
        return True
    
    def _is_targeted_user(self, flag: FeatureFlag, user_id: Optional[str]) -> bool:
        """Check if user is in targeted users list."""
        if not user_id or not flag.target_users:
            return False
        
        is_targeted = user_id in flag.target_users
        if is_targeted:
            logger.debug(f"User {user_id} is in targeted users for flag {flag.name}")
        
        return is_targeted
    
    def _matches_segments(self, flag: FeatureFlag, user_segments: List[str]) -> bool:
        """Check if user matches any target segments."""
        if not flag.target_segments:
            return False
        
        matches = any(segment in user_segments for segment in flag.target_segments)
        if matches:
            logger.debug(f"User matches segments {flag.target_segments} for flag {flag.name}")
        
        return matches
    
    def _check_percentage_rollout(
        self, 
        flag: FeatureFlag, 
        user_id: Optional[str], 
        context: Dict[str, Any]
    ) -> bool:
        """Check if user is included in percentage rollout."""
        if flag.rollout_percentage >= 1.0:
            return True
        
        if not user_id:
            # Anonymous users - check based on session or IP
            session_key = context.get("session_id") or context.get("ip_address") or "anonymous"
            user_key = f"anon:{session_key}"
        else:
            user_key = user_id
        
        # Calculate stable hash
        hash_val = self._hash_key(f"{flag.name}:{user_key}")
        percentage = hash_val % 10000 / 10000.0
        
        is_included = percentage < flag.rollout_percentage
        if is_included:
            logger.debug(f"User {user_id} included in {flag.rollout_percentage*100}% rollout for {flag.name}")
        
        return is_included
    
    def _get_user_segments(self, context: Dict[str, Any]) -> List[str]:
        """Determine which segments the user belongs to."""
        user_segments = []
        
        for segment_name, segment in self.segments.items():
            cache_key = f"{segment_name}:{json.dumps(context, sort_keys=True)}"
            
            if cache_key in self._segment_cache:
                if segment_name in self._segment_cache[cache_key]:
                    user_segments.append(segment_name)
            else:
                if segment.matches(context):
                    user_segments.append(segment_name)
                    self._segment_cache[cache_key] = {segment_name}
                else:
                    self._segment_cache[cache_key] = set()
        
        return user_segments
    
    def _select_variant(
        self, 
        flag: FeatureFlag, 
        user_id: Optional[str], 
        context: Dict[str, Any]
    ) -> Any:
        """Select variant based on strategy."""
        if flag.flag_type == FlagType.BOOLEAN:
            return True
        
        if flag.flag_type == FlagType.PERCENTAGE:
            return flag.rollout_percentage
        
        if flag.flag_type == FlagType.MULTIVARIATE and flag.variants:
            variants = list(flag.variants.items())
            
            if self.strategy == EvaluationStrategy.CONSISTENT_HASH:
                # Consistent assignment based on user_id
                if user_id:
                    hash_val = self._hash_key(f"{flag.name}:{user_id}")
                else:
                    session_key = context.get("session_id") or context.get("ip_address") or "anonymous"
                    hash_val = self._hash_key(f"{flag.name}:{session_key}")
                
                variant_idx = hash_val % len(variants)
                variant_name, variant_value = variants[variant_idx]
                logger.debug(f"Assigned variant {variant_name} to user {user_id}")
                return variant_value
            
            elif self.strategy == EvaluationStrategy.RANDOM:
                # Random assignment
                import random
                variant_name, variant_value = random.choice(variants)
                return variant_value
            
            elif self.strategy == EvaluationStrategy.STICKY_BUCKET:
                # Sticky assignment (first variant for new users)
                if user_id and user_id in flag.target_users:
                    # Use targeted variant if specified
                    for variant_name, variant_value in variants:
                        if variant_name == flag.target_users.get(user_id):
                            return variant_value
                
                # Fallback to consistent hash
                hash_val = self._hash_key(f"{flag.name}:{user_id or 'anonymous'}")
                variant_idx = hash_val % len(variants)
                variant_name, variant_value = variants[variant_idx]
                return variant_value
        
        return flag.default_variant
    
    def _hash_key(self, key: str) -> int:
        """Generate consistent hash for key."""
        salted_key = f"{self.seed}:{key}"
        return int(hashlib.md5(salted_key.encode()).hexdigest()[:8], 16)
    
    def _enabled_result(
        self, 
        flag: FeatureFlag, 
        user_id: Optional[str], 
        context: Dict[str, Any],
        environment: str
    ) -> Dict[str, Any]:
        """Create result dictionary for enabled flag."""
        variant = self._select_variant(flag, user_id, context)
        
        return {
            "enabled": True,
            "variant": variant,
            "flag_name": flag.name,
            "flag_type": flag.flag_type.value,
            "reason": self._get_enabled_reason(flag, user_id, context),
            "timestamp": datetime.utcnow().isoformat(),
            "environment": environment,
            "user_id": user_id,
            "segments": context.get("segments", []),
        }
    
    def _disabled_result(self, flag: FeatureFlag) -> Dict[str, Any]:
        """Create result dictionary for disabled flag."""
        return {
            "enabled": False,
            "variant": flag.default_variant,
            "flag_name": flag.name,
            "flag_type": flag.flag_type.value,
            "reason": "flag_disabled",
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": None,
            "segments": [],
        }
    
    def _get_enabled_reason(
        self, 
        flag: FeatureFlag, 
        user_id: Optional[str], 
        context: Dict[str, Any]
    ) -> str:
        """Determine why flag was enabled."""
        if flag.target_users and user_id in flag.target_users:
            return "targeted_user"
        
        user_segments = context.get("segments", [])
        if flag.target_segments and any(seg in user_segments for seg in flag.target_segments):
            return "segment_match"
        
        if flag.rollout_percentage > 0:
            return "percentage_rollout"
        
        return "globally_enabled"
    
    def _get_default_segments(self) -> Dict[str, Segment]:
        """Get default user segments for WorldBrief360."""
        return {
            # User type segments
            "admin": Segment(
                name="admin",
                description="Administrators",
                rules=[SegmentationRule("role", SegmentRule.EQUALS, "admin")],
                priority=100,
            ),
            "moderator": Segment(
                name="moderator",
                description="Content moderators",
                rules=[SegmentationRule("role", SegmentRule.EQUALS, "moderator")],
                priority=90,
            ),
            "power_user": Segment(
                name="power_user",
                description="Highly active users",
                rules=[
                    SegmentationRule("engagement_score", SegmentRule.GREATER_THAN, 80),
                    SegmentationRule("days_active", SegmentRule.GREATER_THAN, 30),
                ],
                priority=80,
            ),
            "new_user": Segment(
                name="new_user",
                description="Recently registered users",
                rules=[SegmentationRule("days_since_registration", SegmentRule.LESS_THAN, 7)],
                priority=10,
            ),
            
            # Geographic segments
            "north_america": Segment(
                name="north_america",
                description="Users in North America",
                rules=[SegmentationRule("country", SegmentRule.IN, ["US", "CA", "MX"])],
                priority=50,
            ),
            "europe": Segment(
                name="europe",
                description="Users in Europe",
                rules=[SegmentationRule("continent", SegmentRule.EQUALS, "Europe")],
                priority=50,
            ),
            "asia_pacific": Segment(
                name="asia_pacific",
                description="Users in Asia Pacific",
                rules=[SegmentationRule("continent", SegmentRule.EQUALS, "Asia")],
                priority=50,
            ),
            
            # Subscription segments
            "premium": Segment(
                name="premium",
                description="Premium subscribers",
                rules=[SegmentationRule("subscription_tier", SegmentRule.EQUALS, "premium")],
                priority=70,
            ),
            "enterprise": Segment(
                name="enterprise",
                description="Enterprise customers",
                rules=[SegmentationRule("subscription_tier", SegmentRule.EQUALS, "enterprise")],
                priority=80,
            ),
            "free": Segment(
                name="free",
                description="Free tier users",
                rules=[SegmentationRule("subscription_tier", SegmentRule.EQUALS, "free")],
                priority=20,
            ),
            
            # Behavioral segments
            "frequent_reporter": Segment(
                name="frequent_reporter",
                description="Users who frequently report incidents",
                rules=[SegmentationRule("incidents_reported", SegmentRule.GREATER_THAN, 10)],
                priority=60,
            ),
            "chat_active": Segment(
                name="chat_active",
                description="Users who actively use chat",
                rules=[SegmentationRule("chat_sessions", SegmentRule.GREATER_THAN, 20)],
                priority=60,
            ),
            "briefing_consumer": Segment(
                name="briefing_consumer",
                description="Users who frequently read briefings",
                rules=[SegmentationRule("briefings_viewed", SegmentRule.GREATER_THAN, 50)],
                priority=60,
            ),
        }
    
    def add_segment(self, segment: Segment):
        """Add a custom segment."""
        self.segments[segment.name] = segment
        self._segment_cache.clear()  # Clear cache when segments change
    
    def remove_segment(self, segment_name: str):
        """Remove a segment."""
        if segment_name in self.segments:
            del self.segments[segment_name]
            self._segment_cache.clear()
    
    def clear_cache(self):
        """Clear segment evaluation cache."""
        self._segment_cache.clear()
    
    def get_segment_membership(self, context: Dict[str, Any]) -> Dict[str, bool]:
        """Get which segments a user belongs to."""
        membership = {}
        for segment_name, segment in self.segments.items():
            membership[segment_name] = segment.matches(context)
        return membership


# Factory function for creating evaluators
def create_evaluator(
    strategy: Union[str, EvaluationStrategy] = "consistent_hash",
    segments: Optional[Dict[str, Segment]] = None,
    seed: str = "worldbrief360"
) -> FeatureEvaluator:
    """
    Create a feature evaluator with specified configuration.
    
    Args:
        strategy: Evaluation strategy
        segments: Custom segments
        seed: Seed for consistent hashing
    
    Returns:
        Configured FeatureEvaluator
    """
    if isinstance(strategy, str):
        strategy = EvaluationStrategy(strategy)
    
    return FeatureEvaluator(
        strategy=strategy,
        segments=segments,
        seed=seed,
    )


# Default evaluator instance
default_evaluator = create_evaluator()