"""
Utility functions for feature flag system.
Provides helper functions for common operations, validations, and transformations.
"""

import asyncio
import functools
import hashlib
import json
import logging
import random
import re
import string
import time
import uuid
from typing import Dict, Any, Optional, List, Union, Set, Tuple, Callable
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
import base64
import zlib
import pickle
from collections import defaultdict
import inspect
import math

logger = logging.getLogger(__name__)


# ==================== HASHING AND ID GENERATION ====================

def generate_feature_id(flag_name: str, prefix: str = "feat_") -> str:
    """
    Generate a deterministic ID for a feature flag.
    
    Args:
        flag_name: Name of the feature flag
        prefix: ID prefix
    
    Returns:
        Unique feature ID
    
    Example:
        >>> generate_feature_id("dark_mode")
        'feat_8a5d7f3b'
    """
    # Create hash from flag name
    hash_bytes = hashlib.md5(flag_name.encode()).digest()
    # Take first 4 bytes for short ID
    short_hash = hash_bytes[:4].hex()
    return f"{prefix}{short_hash}"


def generate_evaluation_id() -> str:
    """
    Generate a unique evaluation ID.
    
    Returns:
        Unique evaluation ID
    
    Example:
        >>> generate_evaluation_id()
        'eval_7f3a1b9c2d5e8f0a'
    """
    return f"eval_{uuid.uuid4().hex[:16]}"


def hash_user_id(user_id: str, salt: str = "", algorithm: str = "md5") -> int:
    """
    Create a stable hash for user ID for consistent assignments.
    
    Args:
        user_id: User identifier
        salt: Optional salt for hash
        algorithm: Hash algorithm (md5, sha1, sha256)
    
    Returns:
        Integer hash value
    
    Example:
        >>> hash_user_id("user123")
        1234567890
    """
    key = f"{salt}:{user_id}" if salt else user_id
    
    if algorithm == "md5":
        hash_bytes = hashlib.md5(key.encode()).digest()
    elif algorithm == "sha1":
        hash_bytes = hashlib.sha1(key.encode()).digest()
    elif algorithm == "sha256":
        hash_bytes = hashlib.sha256(key.encode()).digest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    # Convert to integer
    return int.from_bytes(hash_bytes[:8], byteorder='big')


def bucket_user_id(user_id: str, num_buckets: int = 1000, salt: str = "") -> int:
    """
    Assign user to a consistent bucket.
    
    Args:
        user_id: User identifier
        num_buckets: Number of buckets (default: 1000 for 0.1% granularity)
        salt: Optional salt for bucket assignment
    
    Returns:
        Bucket number (0 to num_buckets-1)
    
    Example:
        >>> bucket_user_id("user123", num_buckets=1000)
        423  # User is in bucket 423
    """
    if num_buckets <= 0:
        raise ValueError("num_buckets must be positive")
    
    hash_val = hash_user_id(user_id, salt)
    return hash_val % num_buckets


def should_include_user(
    user_id: str,
    percentage: float,
    salt: str = "",
    num_buckets: int = 10000,
) -> bool:
    """
    Determine if user should be included in percentage rollout.
    
    Args:
        user_id: User identifier
        percentage: Rollout percentage (0.0 to 1.0)
        salt: Optional salt for consistent assignment
        num_buckets: Granularity of percentage (higher = more precise)
    
    Returns:
        True if user should be included
    
    Example:
        >>> should_include_user("user123", 0.3)  # 30% rollout
        True  # User is in the 30%
    """
    if percentage >= 1.0:
        return True
    if percentage <= 0.0:
        return False
    
    bucket = bucket_user_id(user_id, num_buckets, salt)
    threshold = int(percentage * num_buckets)
    return bucket < threshold


def generate_session_id() -> str:
    """
    Generate a unique session ID.
    
    Returns:
        Session ID
    
    Example:
        >>> generate_session_id()
        'sess_7a3b1c9d2e5f8a0b'
    """
    return f"sess_{uuid.uuid4().hex[:16]}"


def generate_request_id() -> str:
    """
    Generate a unique request ID.
    
    Returns:
        Request ID
    
    Example:
        >>> generate_request_id()
        'req_9d2e5f8a0b7a3b1c'
    """
    timestamp = int(time.time() * 1000)
    random_part = ''.join(random.choices(string.hexdigits.lower(), k=8))
    return f"req_{timestamp}_{random_part}"


# ==================== VALIDATION FUNCTIONS ====================

def validate_flag_name(name: str) -> Tuple[bool, Optional[str]]:
    """
    Validate feature flag name.
    
    Args:
        name: Flag name to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    
    Example:
        >>> validate_flag_name("dark-mode")
        (False, "Flag name can only contain lowercase letters, numbers, and underscores")
    """
    if not name:
        return False, "Flag name cannot be empty"
    
    if len(name) > 100:
        return False, "Flag name cannot exceed 100 characters"
    
    # Allow letters, numbers, underscores, hyphens
    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
        return False, "Flag name can only contain letters, numbers, underscores, and hyphens"
    
    # Reserved names
    reserved_names = {
        'all', 'none', 'true', 'false', 'null', 'undefined',
        'new', 'delete', 'create', 'update', 'list', 'get',
    }
    if name.lower() in reserved_names:
        return False, f"Flag name '{name}' is reserved"
    
    return True, None


def validate_percentage(percentage: float) -> Tuple[bool, Optional[str]]:
    """
    Validate percentage value.
    
    Args:
        percentage: Percentage value (0.0 to 1.0)
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(percentage, (int, float)):
        return False, "Percentage must be a number"
    
    if percentage < 0.0 or percentage > 1.0:
        return False, "Percentage must be between 0.0 and 1.0"
    
    return True, None


def validate_variants(variants: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate variants dictionary.
    
    Args:
        variants: Variants dictionary
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(variants, dict):
        return False, "Variants must be a dictionary"
    
    if not variants:
        return True, None  # Empty dict is valid
    
    for variant_name, variant_value in variants.items():
        # Validate variant name
        if not isinstance(variant_name, str):
            return False, "Variant names must be strings"
        
        if not variant_name:
            return False, "Variant name cannot be empty"
        
        if len(variant_name) > 50:
            return False, "Variant name cannot exceed 50 characters"
        
        # Validate variant value
        if not is_json_serializable(variant_value):
            return False, f"Variant value for '{variant_name}' must be JSON serializable"
    
    return True, None


def is_json_serializable(value: Any) -> bool:
    """
    Check if value is JSON serializable.
    
    Args:
        value: Value to check
    
    Returns:
        True if JSON serializable
    """
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def validate_environment(environment: str) -> Tuple[bool, Optional[str]]:
    """
    Validate environment name.
    
    Args:
        environment: Environment name
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not environment:
        return False, "Environment cannot be empty"
    
    if len(environment) > 50:
        return False, "Environment name cannot exceed 50 characters"
    
    if not re.match(r'^[a-zA-Z0-9_-]+$', environment):
        return False, "Environment can only contain letters, numbers, underscores, and hyphens"
    
    return True, None


def validate_target_users(user_ids: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate list of target user IDs.
    
    Args:
        user_ids: List of user IDs
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(user_ids, list):
        return False, "Target users must be a list"
    
    for user_id in user_ids:
        if not isinstance(user_id, str):
            return False, "User IDs must be strings"
        
        if not user_id:
            return False, "User ID cannot be empty"
        
        if len(user_id) > 200:
            return False, "User ID cannot exceed 200 characters"
    
    return True, None


# ==================== SERIALIZATION AND ENCODING ====================

def serialize_flag(flag: Dict[str, Any], format: str = "json") -> bytes:
    """
    Serialize flag data.
    
    Args:
        flag: Flag data dictionary
        format: Serialization format (json, pickle, msgpack)
    
    Returns:
        Serialized bytes
    
    Example:
        >>> flag_data = {"name": "dark_mode", "enabled": True}
        >>> serialize_flag(flag_data, "json")
        b'{"name": "dark_mode", "enabled": true}'
    """
    if format == "json":
        return json.dumps(flag, default=_json_serializer).encode('utf-8')
    
    elif format == "pickle":
        return pickle.dumps(flag, protocol=pickle.HIGHEST_PROTOCOL)
    
    elif format == "msgpack":
        import msgpack
        return msgpack.packb(flag, default=_msgpack_serializer)
    
    else:
        raise ValueError(f"Unsupported serialization format: {format}")


def deserialize_flag(data: bytes, format: str = "json") -> Dict[str, Any]:
    """
    Deserialize flag data.
    
    Args:
        data: Serialized bytes
        format: Serialization format (json, pickle, msgpack)
    
    Returns:
        Flag data dictionary
    """
    if format == "json":
        return json.loads(data.decode('utf-8'))
    
    elif format == "pickle":
        return pickle.loads(data)
    
    elif format == "msgpack":
        import msgpack
        return msgpack.unpackb(data)
    
    else:
        raise ValueError(f"Unsupported serialization format: {format}")


def _json_serializer(obj):
    """Custom JSON serializer for unsupported types."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, Enum):
        return obj.value
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _msgpack_serializer(obj):
    """Custom msgpack serializer for unsupported types."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, Enum):
        return obj.value
    else:
        return str(obj)


def compress_data(data: bytes) -> bytes:
    """
    Compress data using zlib.
    
    Args:
        data: Data to compress
    
    Returns:
        Compressed data with 'compressed:' prefix
    
    Example:
        >>> compress_data(b"hello world")
        b'compressed:x\x9c\xcbH\xcd\xc9\xc9W(\xcf/\xcaI\x01\x00\x18\xab\x04='
    """
    compressed = zlib.compress(data, level=zlib.Z_BEST_COMPRESSION)
    return b'compressed:' + compressed


def decompress_data(data: bytes) -> bytes:
    """
    Decompress data compressed with compress_data.
    
    Args:
        data: Compressed data with 'compressed:' prefix
    
    Returns:
        Decompressed data
    
    Raises:
        ValueError: If data is not compressed
    """
    if data.startswith(b'compressed:'):
        compressed = data[11:]  # Remove 'compressed:' prefix
        return zlib.decompress(compressed)
    else:
        raise ValueError("Data is not compressed")


def base64_encode(data: bytes) -> str:
    """
    Encode bytes to base64 string.
    
    Args:
        data: Data to encode
    
    Returns:
        Base64 encoded string
    """
    return base64.b64encode(data).decode('utf-8')


def base64_decode(encoded: str) -> bytes:
    """
    Decode base64 string to bytes.
    
    Args:
        encoded: Base64 encoded string
    
    Returns:
        Decoded bytes
    """
    return base64.b64decode(encoded)


# ==================== DATA TRANSFORMATION ====================

def normalize_flag_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize flag data to standard format.
    
    Args:
        data: Raw flag data
    
    Returns:
        Normalized flag data
    """
    normalized = data.copy()
    
    # Ensure consistent field names
    field_mapping = {
        'featureName': 'name',
        'FeatureName': 'name',
        'isEnabled': 'enabled',
        'IsEnabled': 'enabled',
        'is_enabled': 'enabled',
        'description': 'description',
        'Description': 'description',
        'type': 'flag_type',
        'Type': 'flag_type',
        'flagType': 'flag_type',
    }
    
    for old_key, new_key in field_mapping.items():
        if old_key in normalized and new_key not in normalized:
            normalized[new_key] = normalized.pop(old_key)
    
    # Ensure required fields
    if 'name' not in normalized:
        normalized['name'] = ''
    if 'description' not in normalized:
        normalized['description'] = ''
    if 'flag_type' not in normalized:
        normalized['flag_type'] = 'boolean'
    if 'enabled' not in normalized:
        normalized['enabled'] = False
    
    # Normalize boolean fields
    if isinstance(normalized.get('enabled'), str):
        normalized['enabled'] = normalized['enabled'].lower() in ('true', '1', 'yes', 'on')
    
    # Normalize percentage
    if 'rollout_percentage' in normalized:
        try:
            percentage = float(normalized['rollout_percentage'])
            normalized['rollout_percentage'] = max(0.0, min(1.0, percentage))
        except (ValueError, TypeError):
            normalized['rollout_percentage'] = 0.0
    
    # Normalize lists
    list_fields = ['target_users', 'target_segments', 'environments']
    for field in list_fields:
        if field in normalized:
            if isinstance(normalized[field], str):
                # Try to parse comma-separated string
                try:
                    normalized[field] = [item.strip() for item in normalized[field].split(',') if item.strip()]
                except:
                    normalized[field] = []
            elif not isinstance(normalized[field], list):
                normalized[field] = []
    
    return normalized


def convert_to_feature_flag(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert various flag formats to standard FeatureFlag format.
    
    Args:
        data: Flag data in various formats
    
    Returns:
        Standardized FeatureFlag data
    """
    from .flags import FeatureFlag
    
    # Try to detect format
    if 'version' in data and 'features' in data:
        # Unleash format
        return _convert_unleash_to_standard(data)
    elif 'items' in data and isinstance(data['items'], list):
        # LaunchDarkly format
        return _convert_launchdarkly_to_standard(data)
    else:
        # Assume already in or close to standard format
        return normalize_flag_data(data)


def _convert_unleash_to_standard(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Unleash format to standard format."""
    features = data.get('features', [])
    if not features:
        return {}
    
    # Convert first feature (simplified)
    if features:
        unleash_feature = features[0]
        return {
            'name': unleash_feature.get('name', ''),
            'description': unleash_feature.get('description', ''),
            'enabled': unleash_feature.get('enabled', False),
            'strategies': unleash_feature.get('strategies', []),
            'metadata': {
                'source': 'unleash',
                'created_at': unleash_feature.get('createdAt'),
                'stale': unleash_feature.get('stale', False),
            },
        }
    
    return {}


def _convert_launchdarkly_to_standard(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert LaunchDarkly format to standard format."""
    items = data.get('items', [])
    if not items:
        return {}
    
    # Convert first item (simplified)
    if items:
        ld_item = items[0]
        return {
            'name': ld_item.get('key', ''),
            'description': ld_item.get('description', ''),
            'enabled': ld_item.get('on', False),
            'variations': ld_item.get('variations', []),
            'metadata': {
                'source': 'launchdarkly',
                'version': ld_item.get('version'),
                'creationDate': ld_item.get('creationDate'),
            },
        }
    
    return {}


def filter_flags_by_environment(
    flags: List[Dict[str, Any]],
    environment: str,
    include_all: bool = False,
) -> List[Dict[str, Any]]:
    """
    Filter flags by environment.
    
    Args:
        flags: List of flag dictionaries
        environment: Target environment
        include_all: Include flags with no environment restriction
    
    Returns:
        Filtered list of flags
    """
    filtered = []
    
    for flag in flags:
        flag_envs = flag.get('environments', [])
        
        if include_all and not flag_envs:
            # Flag has no environment restriction
            filtered.append(flag)
        elif environment in flag_envs:
            # Flag is available in target environment
            filtered.append(flag)
    
    return filtered


# ==================== DATE AND TIME UTILITIES ====================

def parse_datetime(value: Any) -> Optional[datetime]:
    """
    Parse datetime from various formats.
    
    Args:
        value: Datetime value (string, timestamp, datetime)
    
    Returns:
        datetime object or None
    """
    if value is None:
        return None
    
    if isinstance(value, datetime):
        return value
    
    if isinstance(value, (int, float)):
        # Assume timestamp
        try:
            return datetime.fromtimestamp(value)
        except (ValueError, OSError):
            return None
    
    if isinstance(value, str):
        # Try ISO format
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            pass
        
        # Try common formats
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d',
            '%d/%m/%Y %H:%M:%S',
            '%d/%m/%Y',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
    
    return None


def is_within_time_range(
    start_time: Optional[datetime],
    end_time: Optional[datetime],
    check_time: Optional[datetime] = None,
) -> bool:
    """
    Check if time is within specified range.
    
    Args:
        start_time: Start of time range (None = no start limit)
        end_time: End of time range (None = no end limit)
        check_time: Time to check (None = current time)
    
    Returns:
        True if within range
    """
    if check_time is None:
        check_time = datetime.utcnow()
    
    if start_time and check_time < start_time:
        return False
    
    if end_time and check_time > end_time:
        return False
    
    return True


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Human-readable duration
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def get_time_buckets(
    start_time: datetime,
    end_time: datetime,
    bucket_size: str = "hour"
) -> List[Tuple[datetime, datetime]]:
    """
    Generate time buckets for analytics.
    
    Args:
        start_time: Start time
        end_time: End time
        bucket_size: Bucket size (hour, day, week, month)
    
    Returns:
        List of (bucket_start, bucket_end) tuples
    """
    buckets = []
    current = start_time
    
    if bucket_size == "hour":
        delta = timedelta(hours=1)
        while current < end_time:
            bucket_end = min(current + delta, end_time)
            buckets.append((current, bucket_end))
            current = bucket_end
    
    elif bucket_size == "day":
        # Start at beginning of day
        current = current.replace(hour=0, minute=0, second=0, microsecond=0)
        delta = timedelta(days=1)
        while current < end_time:
            bucket_end = min(current + delta, end_time)
            buckets.append((current, bucket_end))
            current = bucket_end
    
    elif bucket_size == "week":
        # Start at beginning of week (Monday)
        days_to_monday = current.weekday()
        current = current.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_to_monday)
        delta = timedelta(weeks=1)
        while current < end_time:
            bucket_end = min(current + delta, end_time)
            buckets.append((current, bucket_end))
            current = bucket_end
    
    elif bucket_size == "month":
        # Start at beginning of month
        current = current.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        while current < end_time:
            # Calculate end of month
            if current.month == 12:
                next_month = current.replace(year=current.year + 1, month=1)
            else:
                next_month = current.replace(month=current.month + 1)
            bucket_end = min(next_month, end_time)
            buckets.append((current, bucket_end))
            current = bucket_end
    
    else:
        raise ValueError(f"Unsupported bucket size: {bucket_size}")
    
    return buckets


# ==================== STATISTICAL UTILITIES ====================

def calculate_confidence_interval(
    successes: int,
    total: int,
    confidence_level: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Calculate confidence interval for binomial distribution.
    
    Args:
        successes: Number of successes
        total: Total number of trials
        confidence_level: Confidence level (0.95 for 95%)
    
    Returns:
        Tuple of (lower_bound, upper_bound, point_estimate)
    
    Example:
        >>> calculate_confidence_interval(95, 100, 0.95)
        (0.88, 0.99, 0.95)  # 95% CI for 95/100 success rate
    """
    if total == 0:
        return 0.0, 0.0, 0.0
    
    point_estimate = successes / total
    
    # Wilson score interval
    import math
    z = {
        0.90: 1.645,
        0.95: 1.960,
        0.99: 2.576,
    }.get(confidence_level, 1.960)
    
    denominator = 1 + z**2 / total
    centre_adjusted_probability = point_estimate + z**2 / (2 * total)
    adjusted_standard_deviation = math.sqrt(
        (point_estimate * (1 - point_estimate) + z**2 / (4 * total)) / total
    )
    
    lower_bound = (
        centre_adjusted_probability - z * adjusted_standard_deviation
    ) / denominator
    
    upper_bound = (
        centre_adjusted_probability + z * adjusted_standard_deviation
    ) / denominator
    
    return (
        max(0.0, lower_bound),
        min(1.0, upper_bound),
        point_estimate,
    )


def calculate_sample_size(
    confidence_level: float = 0.95,
    margin_of_error: float = 0.05,
    expected_proportion: float = 0.5,
) -> int:
    """
    Calculate required sample size for A/B test.
    
    Args:
        confidence_level: Confidence level (0.95 for 95%)
        margin_of_error: Desired margin of error
        expected_proportion: Expected proportion (0.5 for maximum sample size)
    
    Returns:
        Required sample size
    
    Example:
        >>> calculate_sample_size(confidence_level=0.95, margin_of_error=0.05)
        385  # Need 385 samples for 95% confidence with ±5% margin
    """
    z = {
        0.90: 1.645,
        0.95: 1.960,
        0.99: 2.576,
    }.get(confidence_level, 1.960)
    
    if margin_of_error <= 0:
        raise ValueError("Margin of error must be positive")
    
    sample_size = (z**2 * expected_proportion * (1 - expected_proportion)) / (margin_of_error**2)
    return math.ceil(sample_size)


def calculate_statistical_significance(
    variant_a_successes: int,
    variant_a_total: int,
    variant_b_successes: int,
    variant_b_total: int,
) -> float:
    """
    Calculate p-value for A/B test using chi-squared test.
    
    Args:
        variant_a_successes: Successes in variant A
        variant_a_total: Total trials in variant A
        variant_b_successes: Successes in variant B
        variant_b_total: Total trials in variant B
    
    Returns:
        p-value (probability that difference is due to chance)
    
    Example:
        >>> calculate_statistical_significance(100, 1000, 120, 1000)
        0.045  # p-value < 0.05, difference is statistically significant
    """
    from scipy.stats import chi2_contingency
    import numpy as np
    
    # Create contingency table
    contingency_table = np.array([
        [variant_a_successes, variant_a_total - variant_a_successes],
        [variant_b_successes, variant_b_total - variant_b_successes],
    ])
    
    try:
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        return p_value
    except ImportError:
        # Fallback if scipy not available
        logger.warning("scipy not available, using approximate calculation")
        return _approximate_p_value(
            variant_a_successes, variant_a_total,
            variant_b_successes, variant_b_total,
        )


def _approximate_p_value(
    successes_a: int,
    total_a: int,
    successes_b: int,
    total_b: int,
) -> float:
    """Approximate p-value using normal approximation."""
    if total_a == 0 or total_b == 0:
        return 1.0
    
    p_a = successes_a / total_a
    p_b = successes_b / total_b
    
    # Pooled proportion
    p_pool = (successes_a + successes_b) / (total_a + total_b)
    
    # Standard error
    se = math.sqrt(p_pool * (1 - p_pool) * (1/total_a + 1/total_b))
    
    if se == 0:
        return 1.0
    
    # Z-score
    z = abs(p_a - p_b) / se
    
    # Two-tailed p-value from normal distribution
    from scipy.stats import norm
    try:
        p_value = 2 * (1 - norm.cdf(z))
        return p_value
    except ImportError:
        # Very rough approximation
        if z > 2.576:
            return 0.01
        elif z > 1.960:
            return 0.05
        elif z > 1.645:
            return 0.10
        else:
            return 0.20


# ==================== CACHE UTILITIES ====================

class LRUCache:
    """Least Recently Used cache implementation."""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.cache = {}
        self.order = []
    
    def get(self, key: str) -> Any:
        """Get value from cache."""
        if key in self.cache:
            # Move to end (most recently used)
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set value in cache."""
        if key in self.cache:
            # Update existing key
            self.cache[key] = value
            self.order.remove(key)
            self.order.append(key)
        else:
            # Add new key
            if len(self.cache) >= self.capacity:
                # Remove least recently used
                lru_key = self.order.pop(0)
                del self.cache[lru_key]
            self.cache[key] = value
            self.order.append(key)
    
    def delete(self, key: str):
        """Delete key from cache."""
        if key in self.cache:
            del self.cache[key]
            self.order.remove(key)
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.order.clear()
    
    def __len__(self) -> int:
        return len(self.cache)
    
    def __contains__(self, key: str) -> bool:
        return key in self.cache


class TTLDict:
    """Dictionary with Time-To-Live expiration."""
    
    def __init__(self, default_ttl: int = 300):
        self.default_ttl = default_ttl
        self.data = {}
        self.expiry_times = {}
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value with TTL."""
        ttl = ttl if ttl is not None else self.default_ttl
        self.data[key] = value
        self.expiry_times[key] = time.time() + ttl
    
    def get(self, key: str) -> Any:
        """Get value, checking expiration."""
        if key in self.data:
            expiry = self.expiry_times.get(key)
            if expiry and time.time() > expiry:
                # Expired
                self.delete(key)
                return None
            return self.data[key]
        return None
    
    def delete(self, key: str):
        """Delete key."""
        if key in self.data:
            del self.data[key]
        if key in self.expiry_times:
            del self.expiry_times[key]
    
    def clear_expired(self) -> int:
        """Clear all expired items."""
        current_time = time.time()
        expired_keys = []
        
        for key, expiry in self.expiry_times.items():
            if current_time > expiry:
                expired_keys.append(key)
        
        for key in expired_keys:
            self.delete(key)
        
        return len(expired_keys)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None


# ==================== LOGGING AND DEBUGGING ====================

def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def log_feature_evaluation(
    flag_name: str,
    user_id: Optional[str],
    enabled: bool,
    variant: Any = None,
    context: Optional[Dict[str, Any]] = None,
    level: str = "DEBUG",
):
    """
    Log feature flag evaluation.
    
    Args:
        flag_name: Feature flag name
        user_id: User identifier
        enabled: Whether flag is enabled
        variant: Variant value (if any)
        context: Evaluation context
        level: Logging level
    """
    log_func = getattr(logger, level.lower(), logger.debug)
    
    log_data = {
        "flag": flag_name,
        "user": user_id or "anonymous",
        "enabled": enabled,
        "variant": variant,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    if context:
        log_data["context"] = context
    
    log_func(f"Feature evaluation: {json.dumps(log_data, default=str)}")


def measure_execution_time(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to measure
    
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
            raise
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return wrapper


# ==================== CONFIGURATION UTILITIES ====================

def load_config_file(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        file_path: Path to config file
    
    Returns:
        Configuration dictionary
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    
    suffix = path.suffix.lower()
    
    if suffix == '.json':
        with open(file_path, 'r') as f:
            return json.load(f)
    
    elif suffix == '.yaml' or suffix == '.yml':
        try:
            import yaml
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML required for YAML config files")
    
    elif suffix == '.toml':
        try:
            import tomli
            with open(file_path, 'rb') as f:
                return tomli.load(f)
        except ImportError:
            raise ImportError("tomli required for TOML config files")
    
    else:
        raise ValueError(f"Unsupported config file format: {suffix}")


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two configuration dictionaries.
    
    Args:
        base: Base configuration
        override: Override configuration
    
    Returns:
        Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge dictionaries
            result[key] = merge_configs(result[key], value)
        else:
            # Override or add value
            result[key] = value
    
    return result


def get_env_variable(
    name: str,
    default: Any = None,
    required: bool = False,
    env_type: type = str,
) -> Any:
    """
    Get environment variable with type conversion.
    
    Args:
        name: Environment variable name
        default: Default value if not set
        required: Raise error if not set
        env_type: Type to convert to (str, int, float, bool, list)
    
    Returns:
        Environment variable value
    
    Raises:
        ValueError: If required and not set, or if conversion fails
    """
    import os
    
    value = os.environ.get(name)
    
    if value is None:
        if required:
            raise ValueError(f"Environment variable {name} is required but not set")
        return default
    
    # Convert type
    try:
        if env_type == str:
            return value
        elif env_type == int:
            return int(value)
        elif env_type == float:
            return float(value)
        elif env_type == bool:
            if value.lower() in ('true', '1', 'yes', 'on'):
                return True
            elif value.lower() in ('false', '0', 'no', 'off'):
                return False
            else:
                raise ValueError(f"Cannot convert '{value}' to bool")
        elif env_type == list:
            return [item.strip() for item in value.split(',') if item.strip()]
        else:
            raise ValueError(f"Unsupported type: {env_type}")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot convert environment variable {name} to {env_type.__name__}: {e}")


# ==================== EXPORTS ====================

__all__ = [
    # Hashing and ID generation
    "generate_feature_id",
    "generate_evaluation_id",
    "hash_user_id",
    "bucket_user_id",
    "should_include_user",
    "generate_session_id",
    "generate_request_id",
    
    # Validation
    "validate_flag_name",
    "validate_percentage",
    "validate_variants",
    "is_json_serializable",
    "validate_environment",
    "validate_target_users",
    
    # Serialization
    "serialize_flag",
    "deserialize_flag",
    "compress_data",
    "decompress_data",
    "base64_encode",
    "base64_decode",
    
    # Data transformation
    "normalize_flag_data",
    "convert_to_feature_flag",
    "filter_flags_by_environment",
    
    # Date and time
    "parse_datetime",
    "is_within_time_range",
    "format_duration",
    "get_time_buckets",
    
    # Statistical utilities
    "calculate_confidence_interval",
    "calculate_sample_size",
    "calculate_statistical_significance",
    
    # Cache utilities
    "LRUCache",
    "TTLDict",
    
    # Logging and debugging
    "setup_logging",
    "log_feature_evaluation",
    "measure_execution_time",
    
    # Configuration
    "load_config_file",
    "merge_configs",
    "get_env_variable",
]