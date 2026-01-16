"""
Advanced Rate Limiting System

This module provides sophisticated rate limiting with:
- Multiple algorithms (Token Bucket, Sliding Window, Fixed Window)
- Distributed rate limiting with Redis
- Adaptive rate limiting based on user behavior
- Burst protection and graceful degradation
- API key and IP-based rate limiting
- Rate limit headers (RFC 6585)
- Analytics and monitoring
- Customizable rate limit strategies
"""

import asyncio
import time
import json
import hashlib
import inspect
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from collections import deque, defaultdict
from functools import wraps, lru_cache
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from app.core.config import get_settings
from app.security.audit_logger import AuditLogger, AuditEventType, AuditSeverity

# Get settings
settings = get_settings()


class RateLimitAlgorithm(str, Enum):
    """Rate limiting algorithms."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"
    ADAPTIVE = "adaptive"


class RateLimitScope(str, Enum):
    """Scope for rate limiting."""
    IP = "ip"                # Rate limit by IP address
    USER = "user"            # Rate limit by user ID
    API_KEY = "api_key"      # Rate limit by API key
    ENDPOINT = "endpoint"    # Rate limit by endpoint
    GLOBAL = "global"        # Global rate limit
    SESSION = "session"      # Rate limit by session
    DEVICE = "device"        # Rate limit by device ID


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies."""
    REJECT = "reject"        # Reject requests when limit exceeded
    QUEUE = "queue"          # Queue requests and process when available
    THROTTLE = "throttle"    # Throttle requests (slow down response)
    ADAPTIVE = "adaptive"    # Adaptive rate limiting based on load
    GRADUAL = "gradual"      # Gradually increase delay
    BYPASS = "bypass"        # Bypass for certain users/conditions


class RateLimitTier(str, Enum):
    """Rate limit tiers for different user types."""
    ANONYMOUS = "anonymous"      # Unauthenticated users
    AUTHENTICATED = "authenticated"  # Regular authenticated users
    PREMIUM = "premium"          # Premium users
    PARTNER = "partner"          # API partners
    ADMIN = "admin"              # Administrators
    SYSTEM = "system"            # Internal system


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    # Basic limits
    requests: int = 100                    # Number of requests
    period: int = 3600                     # Period in seconds (default: 1 hour)
    burst: int = 10                        # Burst capacity
    
    # Algorithm
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW
    
    # Strategy
    strategy: RateLimitStrategy = RateLimitStrategy.REJECT
    queue_timeout: int = 30                # Timeout for queued requests (seconds)
    throttle_delay: float = 0.5            # Delay for throttled requests (seconds)
    
    # Scope
    scope: RateLimitScope = RateLimitScope.IP
    tier: RateLimitTier = RateLimitTier.ANONYMOUS
    
    # Advanced settings
    cost_per_request: int = 1              # Cost per request (for weighted limits)
    max_cost: Optional[int] = None         # Maximum cost per period
    decay_factor: float = 0.9              # Decay factor for adaptive limiting
    learning_rate: float = 0.1             # Learning rate for adaptive algorithm
    min_requests: int = 10                 # Minimum requests for adaptive algorithm
    max_requests: int = 10000              # Maximum requests for adaptive algorithm
    
    # Headers
    enable_headers: bool = True            # Enable rate limit headers
    header_prefix: str = "X-RateLimit"     # Header prefix
    
    # Monitoring
    enable_monitoring: bool = True         # Enable monitoring
    sampling_rate: float = 0.1             # Sampling rate for monitoring (0-1)
    
    def __post_init__(self):
        """Validate configuration."""
        if self.requests <= 0:
            raise ValueError("requests must be positive")
        if self.period <= 0:
            raise ValueError("period must be positive")
        if self.burst < 0:
            raise ValueError("burst cannot be negative")
        if self.cost_per_request <= 0:
            raise ValueError("cost_per_request must be positive")


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool                          # Whether request is allowed
    limit: int                             # Total limit
    remaining: int                         # Remaining requests
    reset_time: int                        # Unix timestamp when limit resets
    retry_after: Optional[int] = None      # Retry after seconds (if rejected)
    cost: int = 1                          # Cost of this request
    wait_time: float = 0.0                 # Wait time if queued/throttled
    queue_position: Optional[int] = None   # Position in queue (if queued)
    tier: RateLimitTier = RateLimitTier.ANONYMOUS
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW
    strategy: RateLimitStrategy = RateLimitStrategy.REJECT
    
    # Headers
    headers: Dict[str, str] = field(default_factory=dict)
    
    def to_headers(self) -> Dict[str, str]:
        """Convert result to HTTP headers."""
        if not self.headers:
            headers = {
                f"X-RateLimit-Limit": str(self.limit),
                f"X-RateLimit-Remaining": str(self.remaining),
                f"X-RateLimit-Reset": str(self.reset_time),
            }
            
            if self.retry_after:
                headers["Retry-After"] = str(self.retry_after)
            
            if self.queue_position is not None:
                headers["X-RateLimit-Queue-Position"] = str(self.queue_position)
            
            if self.wait_time > 0:
                headers["X-RateLimit-Wait-Time"] = str(self.wait_time)
            
            self.headers = headers
        
        return self.headers


@dataclass
class RateLimitEvent:
    """Rate limit event for monitoring."""
    timestamp: datetime
    identifier: str
    endpoint: str
    allowed: bool
    limit: int
    remaining: int
    cost: int
    tier: RateLimitTier
    algorithm: RateLimitAlgorithm
    strategy: RateLimitStrategy
    ip_address: Optional[str] = None
    user_id: Optional[str] = None
    api_key: Optional[str] = None
    user_agent: Optional[str] = None
    response_time: Optional[float] = None


class TokenBucket:
    """Token Bucket algorithm implementation."""
    
    def __init__(
        self,
        capacity: int,
        refill_rate: float,  # tokens per second
        initial_tokens: Optional[int] = None
    ):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = initial_tokens or capacity
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> Tuple[bool, float, int]:
        """
        Consume tokens from the bucket.
        
        Returns:
            Tuple of (success, wait_time, remaining_tokens)
        """
        async with self.lock:
            now = time.time()
            time_passed = now - self.last_refill
            
            # Refill tokens
            new_tokens = time_passed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_refill = now
            
            # Check if enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True, 0.0, int(self.tokens)
            else:
                # Calculate wait time
                deficit = tokens - self.tokens
                wait_time = deficit / self.refill_rate
                return False, wait_time, int(self.tokens)


class SlidingWindow:
    """Sliding Window algorithm implementation."""
    
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = deque()
        self.lock = asyncio.Lock()
    
    async def add_request(self, timestamp: float, weight: int = 1) -> Tuple[bool, int, float]:
        """
        Add a request to the sliding window.
        
        Returns:
            Tuple of (allowed, remaining, oldest_timestamp)
        """
        async with self.lock:
            # Remove old requests
            cutoff = timestamp - self.window_size
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()
            
            # Check if allowed
            if len(self.requests) + weight <= self.max_requests:
                # Add new requests
                for _ in range(weight):
                    self.requests.append(timestamp)
                return True, self.max_requests - len(self.requests), cutoff
            
            # Calculate when the oldest request will expire
            oldest = self.requests[0] if self.requests else timestamp
            retry_after = oldest + self.window_size - timestamp
            
            return False, 0, retry_after
    
    def get_remaining(self, timestamp: float) -> int:
        """Get remaining requests."""
        cutoff = timestamp - self.window_size
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()
        return max(0, self.max_requests - len(self.requests))


class FixedWindow:
    """Fixed Window algorithm implementation."""
    
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size
        self.max_requests = max_requests
        self.current_window = None
        self.current_count = 0
        self.lock = asyncio.Lock()
    
    async def add_request(self, timestamp: float, weight: int = 1) -> Tuple[bool, int, float]:
        """
        Add a request to the fixed window.
        
        Returns:
            Tuple of (allowed, remaining, window_reset)
        """
        async with self.lock:
            window_start = int(timestamp // self.window_size) * self.window_size
            
            # Check if we're in a new window
            if self.current_window != window_start:
                self.current_window = window_start
                self.current_count = 0
            
            # Check if allowed
            if self.current_count + weight <= self.max_requests:
                self.current_count += weight
                remaining = self.max_requests - self.current_count
                window_reset = window_start + self.window_size
                return True, remaining, window_reset
            
            # Calculate retry after
            retry_after = window_start + self.window_size - timestamp
            return False, 0, retry_after


class AdaptiveRateLimiter:
    """Adaptive rate limiting based on system load and user behavior."""
    
    def __init__(
        self,
        base_limit: int,
        min_limit: int,
        max_limit: int,
        learning_rate: float = 0.1,
        decay_factor: float = 0.9
    ):
        self.base_limit = base_limit
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.current_limit = base_limit
        self.success_history = deque(maxlen=100)  # Track last 100 requests
        self.load_history = deque(maxlen=10)      # Track last 10 load samples
        self.lock = asyncio.Lock()
    
    async def update_load(self, current_load: float):
        """Update system load measurement."""
        async with self.lock:
            self.load_history.append(current_load)
            avg_load = sum(self.load_history) / len(self.load_history) if self.load_history else 0
            
            # Adjust limit based on load
            if avg_load > 0.8:  # High load
                self.current_limit = max(self.min_limit, int(self.current_limit * 0.9))
            elif avg_load < 0.3:  # Low load
                self.current_limit = min(self.max_limit, int(self.current_limit * 1.1))
    
    async def record_success(self, success: bool):
        """Record request success/failure."""
        async with self.lock:
            self.success_history.append(1 if success else 0)
            
            # Calculate success rate
            if len(self.success_history) >= 10:
                success_rate = sum(self.success_history) / len(self.success_history)
                
                # Adjust limit based on success rate
                if success_rate < 0.95:  # Too many failures
                    self.current_limit = max(self.min_limit, int(self.current_limit * 0.95))
                elif success_rate > 0.99:  # Very successful
                    self.current_limit = min(self.max_limit, int(self.current_limit * 1.05))
    
    async def get_limit(self) -> int:
        """Get current adaptive limit."""
        async with self.lock:
            return self.current_limit


class DistributedRateLimiter:
    """Distributed rate limiting using Redis."""
    
    def __init__(
        self,
        redis_client: aioredis.Redis,
        namespace: str = "ratelimit"
    ):
        self.redis = redis_client
        self.namespace = namespace
        self.scripts = {}
        self._load_scripts()
    
    def _load_scripts(self):
        """Load Lua scripts for atomic operations."""
        # Token Bucket Lua script
        self.scripts['token_bucket'] = self.redis.register_script("""
            local key = KEYS[1]
            local capacity = tonumber(ARGV[1])
            local refill_rate = tonumber(ARGV[2])
            local tokens = tonumber(ARGV[3])
            local now = tonumber(ARGV[4])
            local cost = tonumber(ARGV[5])
            
            -- Get current state
            local data = redis.call('HMGET', key, 'tokens', 'last_refill')
            local current_tokens = data[1]
            local last_refill = data[2]
            
            -- Initialize if not exists
            if not current_tokens then
                current_tokens = capacity
                last_refill = now
            else
                current_tokens = tonumber(current_tokens)
                last_refill = tonumber(last_refill)
            end
            
            -- Refill tokens
            local time_passed = now - last_refill
            local new_tokens = time_passed * refill_rate
            current_tokens = math.min(capacity, current_tokens + new_tokens)
            
            -- Check if enough tokens
            if current_tokens >= cost then
                current_tokens = current_tokens - cost
                local ttl = math.ceil((capacity - current_tokens) / refill_rate)
                
                -- Update state
                redis.call('HMSET', key, 
                    'tokens', current_tokens,
                    'last_refill', now
                )
                redis.call('EXPIRE', key, ttl)
                
                return {1, current_tokens, 0}
            else
                -- Calculate wait time
                local deficit = cost - current_tokens
                local wait_time = deficit / refill_rate
                
                return {0, current_tokens, wait_time}
            end
        """)
        
        # Sliding Window Lua script
        self.scripts['sliding_window'] = self.redis.register_script("""
            local key = KEYS[1]
            local window = tonumber(ARGV[1])
            local limit = tonumber(ARGV[2])
            local now = tonumber(ARGV[3])
            local cost = tonumber(ARGV[4])
            
            -- Remove old requests
            local cutoff = now - window
            redis.call('ZREMRANGEBYSCORE', key, 0, cutoff)
            
            -- Get current count
            local current = redis.call('ZCARD', key)
            
            if current + cost <= limit then
                -- Add new request(s)
                for i = 1, cost do
                    redis.call('ZADD', key, now, now .. ':' .. i)
                end
                redis.call('EXPIRE', key, window)
                
                return {1, limit - current - cost, 0}
            else
                -- Get oldest request time
                local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
                local retry_after = 0
                if oldest and #oldest > 0 then
                    retry_after = oldest[2] + window - now
                end
                
                return {0, 0, retry_after}
            end
        """)
    
    async def token_bucket(
        self,
        key: str,
        capacity: int,
        refill_rate: float,
        cost: int = 1
    ) -> Tuple[bool, int, float]:
        """Distributed token bucket algorithm."""
        now = time.time()
        result = await self.scripts['token_bucket'](
            keys=[f"{self.namespace}:{key}"],
            args=[capacity, refill_rate, capacity, now, cost]
        )
        
        allowed = bool(result[0])
        remaining = int(result[1])
        wait_time = float(result[2])
        
        return allowed, remaining, wait_time
    
    async def sliding_window(
        self,
        key: str,
        window: int,
        limit: int,
        cost: int = 1
    ) -> Tuple[bool, int, float]:
        """Distributed sliding window algorithm."""
        now = time.time()
        result = await self.scripts['sliding_window'](
            keys=[f"{self.namespace}:{key}"],
            args=[window, limit, now, cost]
        )
        
        allowed = bool(result[0])
        remaining = int(result[1])
        retry_after = float(result[2])
        
        return allowed, remaining, retry_after


class RateLimiter:
    """
    Main rate limiting class with multiple algorithms and strategies.
    """
    
    def __init__(
        self,
        redis_client: Optional[aioredis.Redis] = None,
        default_config: Optional[RateLimitConfig] = None,
        enable_monitoring: bool = True,
        enable_adaptive: bool = True
    ):
        self.redis = redis_client
        self.default_config = default_config or RateLimitConfig()
        self.enable_monitoring = enable_monitoring
        self.enable_adaptive = enable_adaptive
        
        # Initialize components
        self.distributed_limiter = None
        if self.redis:
            self.distributed_limiter = DistributedRateLimiter(self.redis)
        
        self.adaptive_limiter = None
        if enable_adaptive:
            self.adaptive_limiter = AdaptiveRateLimiter(
                base_limit=self.default_config.requests,
                min_limit=self.default_config.min_requests,
                max_limit=self.default_config.max_requests,
                learning_rate=self.default_config.learning_rate,
                decay_factor=self.default_config.decay_factor
            )
        
        # Local rate limiters (for single instance)
        self.local_limiters: Dict[str, Any] = {}
        self.limiter_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        
        # Monitoring
        self.audit_logger = AuditLogger()
        self.event_history: deque = deque(maxlen=10000)
        
        # Request queues for queue strategy
        self.queues: Dict[str, asyncio.Queue] = {}
        self.queue_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        
        # Tier configurations
        self.tier_configs = self._init_tier_configs()
        
        print(f"RateLimiter initialized with algorithm: {self.default_config.algorithm}")
    
    def _init_tier_configs(self) -> Dict[RateLimitTier, RateLimitConfig]:
        """Initialize rate limit configurations for different tiers."""
        return {
            RateLimitTier.ANONYMOUS: RateLimitConfig(
                requests=100,
                period=3600,
                burst=10,
                tier=RateLimitTier.ANONYMOUS,
                strategy=RateLimitStrategy.REJECT
            ),
            RateLimitTier.AUTHENTICATED: RateLimitConfig(
                requests=1000,
                period=3600,
                burst=50,
                tier=RateLimitTier.AUTHENTICATED,
                strategy=RateLimitStrategy.REJECT
            ),
            RateLimitTier.PREMIUM: RateLimitConfig(
                requests=10000,
                period=3600,
                burst=100,
                tier=RateLimitTier.PREMIUM,
                strategy=RateLimitStrategy.QUEUE
            ),
            RateLimitTier.PARTNER: RateLimitConfig(
                requests=50000,
                period=3600,
                burst=500,
                tier=RateLimitTier.PARTNER,
                strategy=RateLimitStrategy.THROTTLE
            ),
            RateLimitTier.ADMIN: RateLimitConfig(
                requests=100000,
                period=3600,
                burst=1000,
                tier=RateLimitTier.ADMIN,
                strategy=RateLimitStrategy.BYPASS
            ),
            RateLimitTier.SYSTEM: RateLimitConfig(
                requests=1000000,
                period=3600,
                burst=10000,
                tier=RateLimitTier.SYSTEM,
                strategy=RateLimitStrategy.BYPASS
            ),
        }
    
    def get_tier_config(self, tier: RateLimitTier) -> RateLimitConfig:
        """Get rate limit configuration for a tier."""
        return self.tier_configs.get(tier, self.default_config)
    
    def _get_identifier(
        self,
        request: Optional[Request] = None,
        identifier: Optional[str] = None,
        scope: RateLimitScope = RateLimitScope.IP
    ) -> str:
        """Get rate limit identifier based on scope."""
        if identifier:
            return identifier
        
        if not request:
            return "global"
        
        if scope == RateLimitScope.IP:
            ip = request.client.host if request.client else "0.0.0.0"
            return f"ip:{ip}"
        elif scope == RateLimitScope.USER:
            # Get user ID from request state or headers
            user_id = "anonymous"
            if hasattr(request.state, 'auth_context'):
                user_id = getattr(request.state.auth_context, 'user_id', 'anonymous')
            return f"user:{user_id}"
        elif scope == RateLimitScope.API_KEY:
            api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
            if api_key:
                return f"apikey:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
            else:
                return "apikey:anonymous"
        elif scope == RateLimitScope.ENDPOINT:
            path = request.url.path
            method = request.method
            return f"endpoint:{method}:{path}"
        elif scope == RateLimitScope.SESSION:
            session_id = request.cookies.get("session_id") or request.headers.get("X-Session-ID")
            if session_id:
                return f"session:{session_id}"
            else:
                return self._get_identifier(request, None, RateLimitScope.IP)
        elif scope == RateLimitScope.DEVICE:
            device_id = request.headers.get("X-Device-ID") or request.cookies.get("device_id")
            if device_id:
                return f"device:{device_id}"
            else:
                # Fallback to user agent + IP
                user_agent = request.headers.get("User-Agent", "unknown")
                ip = request.client.host if request.client else "0.0.0.0"
                device_hash = hashlib.sha256(f"{user_agent}:{ip}".encode()).hexdigest()[:16]
                return f"device:{device_hash}"
        else:  # GLOBAL
            return "global"
    
    def _get_tier(
        self,
        request: Optional[Request] = None,
        tier: Optional[RateLimitTier] = None
    ) -> RateLimitTier:
        """Get rate limit tier for request."""
        if tier:
            return tier
        
        if not request:
            return RateLimitTier.ANONYMOUS
        
        # Check for admin users
        if hasattr(request.state, 'auth_context'):
            auth_context = request.state.auth_context
            if hasattr(auth_context, 'role') and auth_context.role == 'admin':
                return RateLimitTier.ADMIN
            elif hasattr(auth_context, 'is_authenticated') and auth_context.is_authenticated:
                # Check for premium users
                # This would typically come from user data
                return RateLimitTier.AUTHENTICATED
        
        # Check for API key tier
        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if api_key:
            # Check if API key has special tier
            # This would typically come from database
            return RateLimitTier.PARTNER
        
        return RateLimitTier.ANONYMOUS
    
    async def is_allowed(
        self,
        request: Optional[Request] = None,
        identifier: Optional[str] = None,
        endpoint: Optional[str] = None,
        config: Optional[RateLimitConfig] = None,
        cost: int = 1
    ) -> RateLimitResult:
        """
        Check if request is allowed based on rate limits.
        
        Args:
            request: FastAPI request
            identifier: Custom identifier
            endpoint: Endpoint path
            config: Rate limit configuration
            cost: Cost of this request
            
        Returns:
            RateLimitResult object
        """
        start_time = time.time()
        
        # Get configuration
        if not config:
            tier = self._get_tier(request)
            config = self.get_tier_config(tier)
        
        # Get identifier
        identifier_key = self._get_identifier(request, identifier, config.scope)
        full_key = f"{config.algorithm}:{identifier_key}"
        if endpoint:
            full_key = f"{full_key}:{endpoint}"
        
        # Apply algorithm
        result = await self._apply_algorithm(full_key, config, cost)
        
        # Apply strategy
        result = await self._apply_strategy(result, config, request)
        
        # Update result with metadata
        result.tier = config.tier
        result.algorithm = config.algorithm
        result.strategy = config.strategy
        result.cost = cost
        
        # Add headers
        if config.enable_headers:
            result.headers = result.to_headers()
        
        # Monitor and log
        if self.enable_monitoring:
            await self._monitor_request(
                request=request,
                identifier=identifier_key,
                endpoint=endpoint or (request.url.path if request else "unknown"),
                result=result,
                response_time=time.time() - start_time
            )
        
        return result
    
    async def _apply_algorithm(
        self,
        key: str,
        config: RateLimitConfig,
        cost: int
    ) -> RateLimitResult:
        """Apply rate limiting algorithm."""
        now = time.time()
        reset_time = int(now + config.period)
        
        # Try distributed limiter first
        if self.distributed_limiter and config.algorithm in [RateLimitAlgorithm.TOKEN_BUCKET, RateLimitAlgorithm.SLIDING_WINDOW]:
            return await self._apply_distributed_algorithm(key, config, cost, now)
        
        # Local algorithm
        async with self.limiter_locks[key]:
            if key not in self.local_limiters:
                self.local_limiters[key] = self._create_local_limiter(config)
            
            limiter = self.local_limiters[key]
            
            if config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                allowed, wait_time, remaining = await limiter.consume(cost)
                remaining = max(0, remaining)
                reset_time = int(now + (config.requests - remaining) / (config.requests / config.period))
                
            elif config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                allowed, remaining, retry_after = await limiter.add_request(now, cost)
                reset_time = int(now + config.period) if allowed else int(now + retry_after)
                wait_time = retry_after if not allowed else 0.0
                
            elif config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
                allowed, remaining, window_reset = await limiter.add_request(now, cost)
                reset_time = int(window_reset)
                wait_time = window_reset - now if not allowed else 0.0
                
            else:
                # Default to sliding window
                allowed, remaining, retry_after = await limiter.add_request(now, cost)
                reset_time = int(now + config.period) if allowed else int(now + retry_after)
                wait_time = retry_after if not allowed else 0.0
        
        return RateLimitResult(
            allowed=allowed,
            limit=config.requests,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=int(wait_time) if not allowed and wait_time > 0 else None,
            wait_time=wait_time
        )
    
    def _create_local_limiter(self, config: RateLimitConfig) -> Any:
        """Create local rate limiter instance."""
        if config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            refill_rate = config.requests / config.period
            return TokenBucket(config.requests + config.burst, refill_rate, config.requests)
        elif config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            return SlidingWindow(config.period, config.requests + config.burst)
        elif config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            return FixedWindow(config.period, config.requests + config.burst)
        else:
            # Default to sliding window
            return SlidingWindow(config.period, config.requests + config.burst)
    
    async def _apply_distributed_algorithm(
        self,
        key: str,
        config: RateLimitConfig,
        cost: int,
        now: float
    ) -> RateLimitResult:
        """Apply distributed rate limiting algorithm."""
        if config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            refill_rate = config.requests / config.period
            allowed, remaining, wait_time = await self.distributed_limiter.token_bucket(
                key=key,
                capacity=config.requests + config.burst,
                refill_rate=refill_rate,
                cost=cost
            )
            reset_time = int(now + (config.requests - remaining) / refill_rate if remaining < config.requests else now + config.period)
            
        elif config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            allowed, remaining, retry_after = await self.distributed_limiter.sliding_window(
                key=key,
                window=config.period,
                limit=config.requests + config.burst,
                cost=cost
            )
            wait_time = retry_after
            reset_time = int(now + config.period) if allowed else int(now + retry_after)
            
        else:
            # Fallback to local algorithm
            return await self._apply_algorithm(key, config, cost)
        
        return RateLimitResult(
            allowed=allowed,
            limit=config.requests,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=int(wait_time) if not allowed and wait_time > 0 else None,
            wait_time=wait_time
        )
    
    async def _apply_strategy(
        self,
        result: RateLimitResult,
        config: RateLimitConfig,
        request: Optional[Request]
    ) -> RateLimitResult:
        """Apply rate limiting strategy."""
        if result.allowed or config.strategy == RateLimitStrategy.REJECT:
            return result
        
        if config.strategy == RateLimitStrategy.QUEUE:
            return await self._queue_request(result, config, request)
        elif config.strategy == RateLimitStrategy.THROTTLE:
            return await self._throttle_request(result, config)
        elif config.strategy == RateLimitStrategy.GRADUAL:
            return await self._gradual_throttle(result, config)
        elif config.strategy == RateLimitStrategy.BYPASS:
            # Bypass rate limit (for admins/system)
            result.allowed = True
            result.remaining = result.limit
            result.retry_after = None
            result.wait_time = 0
            return result
        else:
            # Default to reject
            return result
    
    async def _queue_request(
        self,
        result: RateLimitResult,
        config: RateLimitConfig,
        request: Optional[Request]
    ) -> RateLimitResult:
        """Queue request for later processing."""
        queue_key = f"queue:{hashlib.sha256(str(request.url.path).encode()).hexdigest()[:16]}" if request else "queue:global"
        
        async with self.queue_locks[queue_key]:
            if queue_key not in self.queues:
                self.queues[queue_key] = asyncio.Queue(maxsize=100)
            
            queue = self.queues[queue_key]
            
            try:
                # Try to put request in queue with timeout
                await asyncio.wait_for(
                    queue.put((result, request)),
                    timeout=config.queue_timeout
                )
                
                # Get queue position
                queue_position = queue.qsize()
                
                # Wait for processing
                processed_result = await self._process_queue(queue_key, queue)
                
                if processed_result:
                    processed_result.queue_position = queue_position
                    return processed_result
                else:
                    # Timeout or error
                    result.allowed = False
                    result.retry_after = config.queue_timeout
                    return result
                    
            except asyncio.TimeoutError:
                result.allowed = False
                result.retry_after = config.queue_timeout
                return result
    
    async def _process_queue(self, queue_key: str, queue: asyncio.Queue) -> Optional[RateLimitResult]:
        """Process queued requests."""
        try:
            # Wait for item to be available
            result, request = await asyncio.wait_for(queue.get(), timeout=30)
            
            # Check rate limit again
            if request:
                new_result = await self.is_allowed(request=request, cost=result.cost)
                return new_result
            else:
                return result
                
        except asyncio.TimeoutError:
            return None
    
    async def _throttle_request(self, result: RateLimitResult, config: RateLimitConfig) -> RateLimitResult:
        """Throttle request by adding delay."""
        if result.wait_time > 0:
            await asyncio.sleep(min(result.wait_time, config.throttle_delay))
        
        # Allow after throttle
        result.allowed = True
        result.wait_time = min(result.wait_time, config.throttle_delay)
        return result
    
    async def _gradual_throttle(self, result: RateLimitResult, config: RateLimitConfig) -> RateLimitResult:
        """Apply gradual throttling based on how much limit is exceeded."""
        if result.remaining < 0:
            exceeded_by = abs(result.remaining)
            throttle_factor = min(1.0, exceeded_by / config.requests)
            wait_time = config.throttle_delay * throttle_factor
            
            await asyncio.sleep(wait_time)
            result.wait_time = wait_time
        
        result.allowed = True
        return result
    
    async def _monitor_request(
        self,
        request: Optional[Request],
        identifier: str,
        endpoint: str,
        result: RateLimitResult,
        response_time: float
    ):
        """Monitor rate limit events."""
        event = RateLimitEvent(
            timestamp=datetime.utcnow(),
            identifier=identifier,
            endpoint=endpoint,
            allowed=result.allowed,
            limit=result.limit,
            remaining=result.remaining,
            cost=result.cost,
            tier=result.tier,
            algorithm=result.algorithm,
            strategy=result.strategy,
            ip_address=request.client.host if request and request.client else None,
            user_agent=request.headers.get("User-Agent") if request else None,
            response_time=response_time
        )
        
        # Add to history
        self.event_history.append(event)
        
        # Log to audit system
        if not result.allowed:
            await self._log_rate_limit_event(event, request)
        
        # Update adaptive limiter
        if self.adaptive_limiter:
            await self.adaptive_limiter.record_success(result.allowed)
            
            # Update load based on recent events
            if len(self.event_history) >= 10:
                recent_events = list(self.event_history)[-10:]
                load = sum(1 for e in recent_events if not e.allowed) / len(recent_events)
                await self.adaptive_limiter.update_load(load)
    
    async def _log_rate_limit_event(self, event: RateLimitEvent, request: Optional[Request]):
        """Log rate limit event to audit system."""
        self.audit_logger.log_security_event(
            event_type=AuditEventType.RATE_LIMIT_EXCEEDED,
            description=f"Rate limit exceeded for {event.identifier}",
            details={
                "identifier": event.identifier,
                "endpoint": event.endpoint,
                "limit": event.limit,
                "remaining": event.remaining,
                "tier": event.tier.value,
                "algorithm": event.algorithm.value,
                "strategy": event.strategy.value,
                "ip_address": event.ip_address,
                "user_agent": event.user_agent,
            },
            severity=AuditSeverity.WARNING
        )
    
    async def get_stats(
        self,
        identifier: Optional[str] = None,
        time_range: int = 3600
    ) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=time_range)
        
        # Filter events by time range
        events = [e for e in self.event_history if e.timestamp >= cutoff]
        
        if identifier:
            events = [e for e in events if e.identifier == identifier]
        
        if not events:
            return {
                "total_requests": 0,
                "allowed_requests": 0,
                "blocked_requests": 0,
                "block_rate": 0.0,
                "avg_response_time": 0.0,
                "top_endpoints": [],
                "top_identifiers": [],
            }
        
        # Calculate statistics
        total = len(events)
        allowed = sum(1 for e in events if e.allowed)
        blocked = total - allowed
        block_rate = blocked / total if total > 0 else 0.0
        
        avg_response_time = sum(e.response_time or 0 for e in events) / total
        
        # Top endpoints
        endpoint_counts = defaultdict(int)
        for e in events:
            endpoint_counts[e.endpoint] += 1
        
        top_endpoints = sorted(
            endpoint_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Top identifiers
        identifier_counts = defaultdict(int)
        for e in events:
            identifier_counts[e.identifier] += 1
        
        top_identifiers = sorted(
            identifier_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "total_requests": total,
            "allowed_requests": allowed,
            "blocked_requests": blocked,
            "block_rate": block_rate,
            "avg_response_time": avg_response_time,
            "top_endpoints": top_endpoints,
            "top_identifiers": top_identifiers,
            "time_range": time_range,
            "sample_size": total,
        }
    
    async def reset_limit(self, identifier: str):
        """Reset rate limit for an identifier."""
        # Clear local limiters
        for key in list(self.local_limiters.keys()):
            if identifier in key:
                del self.local_limiters[key]
        
        # Clear Redis keys
        if self.redis:
            pattern = f"*{identifier}*"
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)
        
        # Clear queues
        for key in list(self.queues.keys()):
            if identifier in key:
                del self.queues[key]
    
    async def update_config(
        self,
        tier: RateLimitTier,
        config: RateLimitConfig
    ):
        """Update rate limit configuration for a tier."""
        self.tier_configs[tier] = config
        
        # Update adaptive limiter if enabled
        if self.adaptive_limiter:
            await self.adaptive_limiter.update_load(0)  # Trigger recalculation


# FastAPI Middleware
class RateLimitMiddleware:
    """FastAPI middleware for rate limiting."""
    
    def __init__(
        self,
        app,
        rate_limiter: Optional[RateLimiter] = None,
        default_config: Optional[RateLimitConfig] = None,
        excluded_paths: Optional[List[str]] = None,
        enable_by_default: bool = True
    ):
        self.app = app
        self.rate_limiter = rate_limiter or RateLimiter()
        self.default_config = default_config or RateLimitConfig()
        self.excluded_paths = excluded_paths or [
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/health",
        ]
        self.enable_by_default = enable_by_default
        
        # Endpoint-specific configurations
        self.endpoint_configs: Dict[str, RateLimitConfig] = {}
        
        # Initialize default endpoint configs
        self._init_endpoint_configs()
    
    def _init_endpoint_configs(self):
        """Initialize endpoint-specific rate limit configurations."""
        # Authentication endpoints - stricter limits
        self.endpoint_configs["/api/v1/auth/login"] = RateLimitConfig(
            requests=5,
            period=300,  # 5 minutes
            burst=2,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
            strategy=RateLimitStrategy.REJECT,
            scope=RateLimitScope.IP,
            tier=RateLimitTier.ANONYMOUS
        )
        
        self.endpoint_configs["/api/v1/auth/register"] = RateLimitConfig(
            requests=3,
            period=3600,  # 1 hour
            burst=1,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
            strategy=RateLimitStrategy.REJECT,
            scope=RateLimitScope.IP,
            tier=RateLimitTier.ANONYMOUS
        )
        
        # API endpoints - different tiers
        self.endpoint_configs["/api/v1/chat"] = RateLimitConfig(
            requests=100,
            period=3600,
            burst=20,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            strategy=RateLimitStrategy.THROTTLE,
            scope=RateLimitScope.USER,
            tier=RateLimitTier.AUTHENTICATED
        )
        
        self.endpoint_configs["/api/v1/briefings"] = RateLimitConfig(
            requests=50,
            period=3600,
            burst=10,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
            strategy=RateLimitStrategy.QUEUE,
            scope=RateLimitScope.USER,
            tier=RateLimitTier.AUTHENTICATED
        )
        
        self.endpoint_configs["/api/v1/incidents"] = RateLimitConfig(
            requests=1000,
            period=3600,
            burst=100,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
            strategy=RateLimitStrategy.REJECT,
            scope=RateLimitScope.IP,
            tier=RateLimitTier.ANONYMOUS
        )
        
        # Admin endpoints - higher limits
        self.endpoint_configs["/api/v1/admin"] = RateLimitConfig(
            requests=10000,
            period=3600,
            burst=1000,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
            strategy=RateLimitStrategy.BYPASS,
            scope=RateLimitScope.USER,
            tier=RateLimitTier.ADMIN
        )
    
    async def __call__(self, request: Request, call_next):
        """Process request with rate limiting."""
        
        # Check if path is excluded
        if self._is_excluded(request.url.path):
            return await call_next(request)
        
        # Get endpoint-specific configuration
        config = self._get_endpoint_config(request.url.path)
        
        # Check rate limit
        result = await self.rate_limiter.is_allowed(
            request=request,
            endpoint=request.url.path,
            config=config
        )
        
        if not result.allowed:
            return self._create_rate_limit_response(result)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        if config.enable_headers:
            for header, value in result.headers.items():
                response.headers[header] = value
        
        return response
    
    def _is_excluded(self, path: str) -> bool:
        """Check if path is excluded from rate limiting."""
        for excluded in self.excluded_paths:
            if path.startswith(excluded):
                return True
        return False
    
    def _get_endpoint_config(self, path: str) -> RateLimitConfig:
        """Get rate limit configuration for endpoint."""
        for endpoint_pattern, config in self.endpoint_configs.items():
            if path.startswith(endpoint_pattern):
                return config
        
        # Check for exact match with path components
        path_parts = path.strip("/").split("/")
        for i in range(len(path_parts)):
            check_path = "/" + "/".join(path_parts[:i+1])
            if check_path in self.endpoint_configs:
                return self.endpoint_configs[check_path]
        
        return self.default_config
    
    def _create_rate_limit_response(self, result: RateLimitResult) -> Response:
        """Create rate limit exceeded response."""
        headers = result.headers
        
        if result.retry_after:
            headers["Retry-After"] = str(result.retry_after)
        
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "detail": {
                    "limit": result.limit,
                    "remaining": result.remaining,
                    "reset_time": result.reset_time,
                    "retry_after": result.retry_after,
                    "tier": result.tier.value,
                }
            },
            headers=headers
        )
    
    def add_endpoint_config(self, path: str, config: RateLimitConfig):
        """Add endpoint-specific rate limit configuration."""
        self.endpoint_configs[path] = config


# FastAPI Dependency
def get_rate_limiter() -> RateLimiter:
    """Dependency to get rate limiter instance."""
    # This would typically initialize with Redis from settings
    redis_url = settings.REDIS_URL if hasattr(settings, 'REDIS_URL') else None
    
    if redis_url:
        redis_client = aioredis.from_url(redis_url, decode_responses=True)
        return RateLimiter(redis_client=redis_client)
    else:
        return RateLimiter()


# Decorator for rate limiting
def rate_limit(
    requests: int = 100,
    period: int = 3600,
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW,
    strategy: RateLimitStrategy = RateLimitStrategy.REJECT,
    scope: RateLimitScope = RateLimitScope.IP,
    tier: RateLimitTier = RateLimitTier.ANONYMOUS,
    cost: int = 1,
    limiter: Optional[RateLimiter] = None
):
    """
    Decorator to apply rate limiting to endpoints.
    
    Args:
        requests: Number of requests allowed
        period: Time period in seconds
        algorithm: Rate limiting algorithm
        strategy: Rate limiting strategy
        scope: Rate limiting scope
        tier: Rate limit tier
        cost: Cost of this request
        limiter: Custom rate limiter instance
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Find request in args/kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                for value in kwargs.values():
                    if isinstance(value, Request):
                        request = value
                        break
            
            if not request:
                # No request found, proceed without rate limiting
                return await func(*args, **kwargs)
            
            # Create config
            config = RateLimitConfig(
                requests=requests,
                period=period,
                algorithm=algorithm,
                strategy=strategy,
                scope=scope,
                tier=tier
            )
            
            # Get limiter
            rate_limiter_instance = limiter or get_rate_limiter()
            
            # Check rate limit
            result = await rate_limiter_instance.is_allowed(
                request=request,
                endpoint=request.url.path,
                config=config,
                cost=cost
            )
            
            if not result.allowed:
                # Create rate limit response
                response = JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "detail": {
                            "limit": result.limit,
                            "remaining": result.remaining,
                            "reset_time": result.reset_time,
                            "retry_after": result.retry_after,
                        }
                    },
                    headers=result.headers
                )
                return response
            
            # Add headers to response
            response = await func(*args, **kwargs)
            if isinstance(response, Response):
                for header, value in result.headers.items():
                    response.headers[header] = value
            
            return response
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we need to handle differently
            # This is a simplified version
            return func(*args, **kwargs)
        
        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Context manager for rate limiting
@asynccontextmanager
async def rate_limit_context(
    identifier: str,
    config: RateLimitConfig,
    limiter: Optional[RateLimiter] = None
):
    """
    Context manager for rate limiting.
    
    Args:
        identifier: Rate limit identifier
        config: Rate limit configuration
        limiter: Rate limiter instance
    """
    rate_limiter_instance = limiter or get_rate_limiter()
    
    # Create a mock request for the context
    class MockRequest:
        def __init__(self, identifier):
            self.url = type('Url', (), {'path': identifier})()
            self.headers = {}
            self.client = type('Client', (), {'host': identifier})()
    
    mock_request = MockRequest(identifier)
    
    result = await rate_limiter_instance.is_allowed(
        request=mock_request,
        config=config
    )
    
    if not result.allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded for {identifier}"
        )
    
    try:
        yield result
    finally:
        # Cleanup if needed
        pass


# Utility functions
async def check_rate_limit(
    identifier: str,
    endpoint: str = "global",
    requests: int = 100,
    period: int = 3600
) -> RateLimitResult:
    """Quick rate limit check utility."""
    limiter = RateLimiter()
    config = RateLimitConfig(requests=requests, period=period)
    
    result = await limiter.is_allowed(
        identifier=identifier,
        endpoint=endpoint,
        config=config
    )
    
    return result


def get_rate_limit_headers(result: RateLimitResult) -> Dict[str, str]:
    """Get rate limit headers from result."""
    return result.to_headers()


# Default rate limiter instance
default_limiter = RateLimiter()


# Export main components
__all__ = [
    # Classes
    "RateLimiter",
    "RateLimitMiddleware",
    "TokenBucket",
    "SlidingWindow",
    "FixedWindow",
    "AdaptiveRateLimiter",
    "DistributedRateLimiter",
    
    # Data classes
    "RateLimitConfig",
    "RateLimitResult",
    "RateLimitEvent",
    
    # Enums
    "RateLimitAlgorithm",
    "RateLimitScope",
    "RateLimitStrategy",
    "RateLimitTier",
    
    # Functions
    "rate_limit",
    "get_rate_limiter",
    "check_rate_limit",
    "get_rate_limit_headers",
    "rate_limit_context",
    
    # Default instances
    "default_limiter",
]